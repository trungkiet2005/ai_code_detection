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
    """Auto-tune batch size / workers / precision / LR based on detected GPU.

    Presets (triggered by total VRAM on GPU 0):
      >= 70 GB  -- H100 80GB / A100 80GB: batch 128 bf16 workers 8 LR*sqrt(2)
      30-70 GB  -- A100 40GB / V100 32GB: batch 64 bf16/fp16 workers 4
      10-30 GB  -- T4 15GB / P100 16GB / L4 24GB (Kaggle/Colab free): batch 32
                   fp16 workers 4 grad_accum 2 (preserves effective batch 64)
      < 10 GB   -- consumer GPU (3060 / 4070): batch 16 fp16 workers 2 grad_accum 4
      CPU       -- unchanged (caller config wins)

    For tiers < H100 we INCREASE grad_accum_steps so the effective batch
    matches the H100 128-sample target. This keeps the optimization
    trajectory (and therefore hyperparams like LR) comparable across GPUs.

    Honours `auto_h100_profile` flag name for back-compat (set to False
    to skip auto-tuning entirely). Device-agnostic TF32 / cuDNN benchmark
    flags are always applied on CUDA.
    """
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

    try:
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        gpu_name = torch.cuda.get_device_name(0)
    except Exception:
        return config

    name_upper = gpu_name.upper()
    supports_bf16 = ("H100" in name_upper) or ("A100" in name_upper) or ("L4" in name_upper) or ("L40" in name_upper)

    if total_gb >= 70:
        # H100 80GB / A100 80GB. Empirically safe under ModernBERT-base +
        # bidirectional attention + RoPE at batch 128 seq 512 bf16.
        config.precision = "bf16" if config.precision == "auto" else config.precision
        config.batch_size = max(config.batch_size, 128)
        config.grad_accum_steps = 1
        config.num_workers = max(config.num_workers, 8)
        config.prefetch_factor = max(config.prefetch_factor, 4)
        config.log_every = max(config.log_every, 200)
        config.eval_every = max(config.eval_every, 2000)
        # Scale LR sqrt(2) for 2x batch vs the 64-baseline from earlier climb runs.
        config.lr_encoder = max(config.lr_encoder, 2.8e-5)
        config.lr_heads = max(config.lr_heads, 1.4e-4)
        tier = "H100/A100-80GB"
    elif total_gb >= 30:
        # A100 40GB / V100 32GB. Halve batch vs H100, keep bf16 where the
        # GPU supports it (A100); V100 falls back to fp16.
        config.precision = ("bf16" if supports_bf16 else "fp16") if config.precision == "auto" else config.precision
        config.batch_size = max(config.batch_size, 64)
        # Effective batch 128 via grad_accum = 2.
        config.grad_accum_steps = max(config.grad_accum_steps, 2)
        config.num_workers = max(config.num_workers, 4)
        config.prefetch_factor = max(config.prefetch_factor, 2)
        config.log_every = max(config.log_every, 100)
        config.eval_every = max(config.eval_every, 1000)
        # Same effective batch -> same LR scaling as H100.
        config.lr_encoder = max(config.lr_encoder, 2.8e-5)
        config.lr_heads = max(config.lr_heads, 1.4e-4)
        tier = f"{name_upper.split()[-1]}-30-70GB"
    elif total_gb >= 10:
        # Kaggle free-tier T4 15GB / P100 16GB / Colab L4 24GB.
        # T4 lacks bf16 -> fp16 fallback. Batch 32 fits ModernBERT-base
        # with seq 512 fp16 at ~6-7 GB activation; grad_accum=4 maintains
        # effective batch 128 so the optim trajectory matches H100 runs.
        config.precision = ("bf16" if supports_bf16 else "fp16") if config.precision == "auto" else config.precision
        config.batch_size = max(config.batch_size, 32)
        config.grad_accum_steps = max(config.grad_accum_steps, 4)
        config.num_workers = max(config.num_workers, 4)
        config.prefetch_factor = max(config.prefetch_factor, 2)
        config.log_every = max(config.log_every, 100)
        config.eval_every = max(config.eval_every, 1000)
        # Same effective batch 128 -> same LR.
        config.lr_encoder = max(config.lr_encoder, 2.8e-5)
        config.lr_heads = max(config.lr_heads, 1.4e-4)
        tier = "Kaggle-T4/P100-tier"
    else:
        # Consumer GPU (3060 12GB, 4070 12GB). Tiny batch + more accum.
        config.precision = "fp16" if config.precision == "auto" else config.precision
        config.batch_size = max(config.batch_size, 16)
        config.grad_accum_steps = max(config.grad_accum_steps, 8)
        config.num_workers = max(config.num_workers, 2)
        config.prefetch_factor = max(config.prefetch_factor, 2)
        config.lr_encoder = max(config.lr_encoder, 2.8e-5)
        config.lr_heads = max(config.lr_heads, 1.4e-4)
        tier = "consumer-<10GB"

    effective = config.batch_size * config.grad_accum_steps
    logger.info(
        "Applied %s profile | gpu=%s (%.0f GB) | precision=%s | batch=%d x grad_accum=%d "
        "(eff=%d) | seq=%d | lr_enc=%.2e | lr_heads=%.2e | workers=%d",
        tier, gpu_name, total_gb, config.precision,
        config.batch_size, config.grad_accum_steps, effective,
        config.max_length, config.lr_encoder, config.lr_heads, config.num_workers,
    )
    return config
