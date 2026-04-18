"""
Shared bootstrap + utilities for all Exp_Climb runs.

Imported by every exp_NN_*.py file. Keeps dependency-install, autocast wrapper,
hardware profile, config dataclass, and general helpers in one place.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import random
import subprocess
import sys
import warnings
from dataclasses import dataclass, field
from typing import Dict

# Reduce CUDA memory fragmentation (critical for large-batch / long-seq runs).
# Must be set BEFORE torch imports.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# ---------------------------------------------------------------------------
# Bootstrap – auto-install missing packages (Kaggle-friendly)
# ---------------------------------------------------------------------------

def ensure_deps():
    required = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("sklearn", "scikit-learn"),
        ("accelerate", "accelerate"),
    ]
    optional = [
        ("tree_sitter", "tree-sitter"),
        ("tree_sitter_languages", "tree-sitter-languages"),
    ]
    missing = [pip for imp, pip in required if importlib.util.find_spec(imp) is None]
    if missing:
        print(f"[bootstrap] Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    for imp, pip in optional:
        if importlib.util.find_spec(imp) is None:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip])
            except subprocess.CalledProcessError:
                print(f"[bootstrap] Optional dep unavailable: {pip}")


ensure_deps()

# ---------------------------------------------------------------------------
# Third-party (after bootstrap)
# ---------------------------------------------------------------------------

import numpy as np
import torch

try:
    from torch.amp import autocast as _autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler  # type: ignore
    _NEW_AMP = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp_climb")


# ---------------------------------------------------------------------------
# AMP helpers
# ---------------------------------------------------------------------------

def autocast(device_type: str = "cuda", enabled: bool = True, dtype=None):
    if _NEW_AMP:
        if dtype is None:
            return _autocast(device_type=device_type, enabled=enabled)
        return _autocast(device_type=device_type, enabled=enabled, dtype=dtype)
    return _autocast(enabled=enabled)


def make_grad_scaler(enabled: bool):
    return GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Errors & misc
# ---------------------------------------------------------------------------

class PreflightError(RuntimeError):
    """Raised when preflight checks fail (bad data, missing deps, etc.)."""


def get_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Shared model + training config
# ---------------------------------------------------------------------------

@dataclass
class SpectralConfig:
    """Shared model + training config. Per-method classes (hier loss etc.)
    only add their own hyperparameters and inherit the rest."""

    task: str = "T1"
    benchmark: str = "codet_m4"

    # Encoder
    encoder_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512

    # Latents
    z_style_dim: int = 256
    z_content_dim: int = 256
    gnn_hidden_dim: int = 128
    gnn_layers: int = 2
    num_ast_node_types: int = 256
    ast_embed_dim: int = 64
    ast_seq_len: int = 128

    # Optimization
    epochs: int = 3
    batch_size: int = 32
    grad_accum_steps: int = 2
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Contrastive / geometry defaults (methods may override)
    temperature: float = 0.07
    lambda_hier: float = 0.4
    hier_margin: float = 0.3

    # Sampling (CLIMB: small train, FULL test)
    max_train_samples: int = 100_000
    max_val_samples: int = 20_000
    max_test_samples: int = -1   # -1 => full test (no subsampling)

    # Runtime
    num_workers: int = 2
    prefetch_factor: int = 2
    seed: int = 42
    precision: str = "auto"   # auto | bf16 | fp16 | fp32
    auto_h100_profile: bool = True
    require_tree_sitter: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    non_blocking: bool = True
    save_dir: str = "./codet_m4_checkpoints"
    log_every: int = 100
    eval_every: int = 1000
    # Kaggle has ~20 GB /kaggle/working quota. Saving a "latest" checkpoint
    # every epoch (in addition to "best") can bust the quota across 16 climb
    # runs (16 * 2 * 600MB ≈ 19 GB). Default False = save only "best".
    save_latest_ckpt: bool = False


def apply_hardware_profile(config: SpectralConfig) -> SpectralConfig:
    """Tune batch size / workers / precision automatically for H100."""
    if config.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if not (config.auto_h100_profile and config.device == "cuda"):
        return config

    gpu_name = get_gpu_name().upper()
    if "H100" not in gpu_name:
        return config

    # H100 80GB profile: target ~40 GB utilization. Empirically safe under
    # ModernBERT-base + bidirectional attention + RoPE buffers:
    #   - batch 128 (2x vs old 64): ~40 GB forward + backward fits
    #   - seq 512 (same as paper baseline): keeps direct comparability
    #   - seq 1024 OOMs at batch 128 -- attention + RoPE blow 80 GB
    #   - LR scaled sqrt(2) = 1.4x to match 2x batch
    # If user wants seq 1024, they must drop batch to 64 (set explicitly).
    config.precision = "bf16" if config.precision == "auto" else config.precision
    config.batch_size = max(config.batch_size, 128)
    # Note: max_length is NOT auto-bumped. Keeping 512 as safe default under
    # batch 128. Override manually in your Config if you want seq 1024+batch 64.
    config.grad_accum_steps = 1
    config.num_workers = max(config.num_workers, 8)
    config.prefetch_factor = max(config.prefetch_factor, 4)
    config.log_every = max(config.log_every, 200)
    config.eval_every = max(config.eval_every, 2000)
    # Scale learning rate for 2x batch (sqrt rule): 2e-5 -> ~2.8e-5
    config.lr_encoder = max(config.lr_encoder, 2.8e-5)
    config.lr_heads = max(config.lr_heads, 1.4e-4)
    logger.info(
        "Applied H100 80GB profile | precision=%s | batch=%d | seq=%d | "
        "lr_enc=%.2e | lr_heads=%.2e | workers=%d",
        config.precision, config.batch_size, config.max_length,
        config.lr_encoder, config.lr_heads, config.num_workers,
    )
    return config
