"""
Shared bootstrap + utilities for Exp_FewShot runs.

Lean version of Exp_Climb/_common.py, designed for Kaggle T4 (16 GB VRAM, fp16).
Standalone per suite (no imports from Exp_Climb / Exp_DM / Exp_CodeDet).

Default profile: bs=16, fp16, seq=384, 1 epoch — fits T4 with ModernBERT-base.
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
from typing import Dict, List, Optional, Tuple

# Reduce CUDA memory fragmentation (helps T4 with long sequences).
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


# ---------------------------------------------------------------------------
# Bootstrap (Kaggle-friendly auto-install)
# ---------------------------------------------------------------------------

def ensure_deps():
    required = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("sklearn", "scikit-learn"),
    ]
    missing = [pip for imp, pip in required if importlib.util.find_spec(imp) is None]
    if missing:
        print(f"[fs-bootstrap] Installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


ensure_deps()

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
logger = logging.getLogger("exp_fewshot")


def autocast(device_type: str = "cuda", enabled: bool = True, dtype=None):
    if _NEW_AMP:
        if dtype is None:
            return _autocast(device_type=device_type, enabled=enabled)
        return _autocast(device_type=device_type, enabled=enabled, dtype=dtype)
    return _autocast(enabled=enabled)


def make_grad_scaler(enabled: bool = True):
    if _NEW_AMP:
        try:
            return GradScaler(device="cuda", enabled=enabled)
        except TypeError:
            return GradScaler(enabled=enabled)
    return GradScaler(enabled=enabled)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------------------
# Few-shot config
# ---------------------------------------------------------------------------

@dataclass
class FSConfig:
    """Few-shot training config — T4-first defaults.

    Per-method exp files inherit and only override method-specific hyperparams.
    """

    # Benchmark + task
    benchmark: str = "codet_m4"        # codet_m4 | droid
    task: str = "author"                # author (6-cls) | binary | t3 (Droid 3-cls)

    # Few-shot regime
    k_shot: int = 32                    # K examples per class in TRAIN
    n_classes: int = 6                  # 6 for CoDET author, 2 for binary, 3 for Droid T3
    fs_seed: int = 42                   # sampling seed (reported in tracker)

    # Encoder
    encoder_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 384               # T4-friendly (was 512 on H100)

    # Optimization (1 epoch, no scheduler)
    epochs: int = 1
    batch_size: int = 16                # T4 16GB fp16
    grad_accum_steps: int = 1
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0           # no warmup with 1 epoch
    early_stop_patience: int = 5        # eval steps without val improvement

    # Few-shot specifics
    use_class_weights: bool = True
    val_size_per_class: int = 64        # mini-val held out from full val pool
    test_max_samples: int = -1          # -1 = full test (paper-comparable)

    # NTK alignment (Exp_FS_01 specific, ignored by other methods)
    lambda_ntk: float = 0.4
    ntk_proj_dim: int = 128

    # Runtime
    num_workers: int = 2
    prefetch_factor: int = 2
    seed: int = 42
    precision: str = "fp16"             # fp16 default for T4
    auto_t4_profile: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    save_dir: str = "./fewshot_checkpoints"
    log_every: int = 10
    eval_every: int = 50

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.n_classes = 6 if self.task == "author" else 2
        elif self.benchmark == "droid":
            self.n_classes = 3 if self.task == "t3" else 2


# ---------------------------------------------------------------------------
# Hardware profile (T4-first, but auto-adapts to whatever GPU we get)
# ---------------------------------------------------------------------------

def get_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    return torch.cuda.get_device_name(0)


def apply_hardware_profile(cfg: FSConfig) -> FSConfig:
    """Auto-tune for detected GPU. Default = T4 profile (bs=16, fp16, seq=384).

    H100/A100 (>=70 GB): keep bs=16 (few-shot is tiny anyway), bf16, seq=512
    T4/P100/L4 (10-30 GB): bs=16 fp16 seq=384 (DEFAULT)
    Consumer (<10 GB): bs=8 fp16 seq=256
    CPU: bs=4 fp32 seq=256 (smoke test only)
    """
    if not cfg.auto_t4_profile:
        return cfg

    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            total_mem_gb = 16.0
        gpu_name = get_gpu_name()
        if total_mem_gb >= 70:
            cfg.batch_size = 16
            cfg.precision = "bf16"
            cfg.max_length = 512
            cfg.num_workers = 4
        elif total_mem_gb >= 30:
            cfg.batch_size = 16
            cfg.precision = "fp16"
            cfg.max_length = 512
            cfg.num_workers = 4
        elif total_mem_gb >= 10:
            cfg.batch_size = 16
            cfg.precision = "fp16"
            cfg.max_length = 384
            cfg.num_workers = 2
        else:
            cfg.batch_size = 8
            cfg.precision = "fp16"
            cfg.max_length = 256
            cfg.num_workers = 2
        logger.info(
            f"[hw-profile] GPU={gpu_name} VRAM={total_mem_gb:.1f}GB -> "
            f"bs={cfg.batch_size} prec={cfg.precision} seq={cfg.max_length}"
        )
    else:
        cfg.batch_size = 4
        cfg.precision = "fp32"
        cfg.max_length = 256
        cfg.num_workers = 0
        logger.info("[hw-profile] CPU mode -> smoke test only")

    return cfg


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def emit_paper_table(method_name: str, exp_id: str, cfg: FSConfig, results: Dict):
    """Emit a BEGIN_FS_TABLE/END_FS_TABLE block parsed by tracker.md updater."""
    lines = [
        f"BEGIN_FS_TABLE method={method_name} exp_id={exp_id}",
        f"  benchmark: {cfg.benchmark}",
        f"  task:      {cfg.task}",
        f"  k_shot:    {cfg.k_shot}",
        f"  n_classes: {cfg.n_classes}",
        f"  fs_seed:   {cfg.fs_seed}",
        f"  encoder:   {cfg.encoder_name}",
        f"  precision: {cfg.precision}",
        f"  bs:        {cfg.batch_size}",
        f"  seq:       {cfg.max_length}",
    ]
    for key, value in results.items():
        if isinstance(value, float):
            lines.append(f"  {key:<10} {value:.4f}")
        else:
            lines.append(f"  {key:<10} {value}")
    lines.append("END_FS_TABLE")
    print("\n".join(lines), flush=True)
