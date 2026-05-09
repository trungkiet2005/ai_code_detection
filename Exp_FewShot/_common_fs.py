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

    # Sampling regime — choose ONE:
    #   k_shot > 0  AND  train_fraction <= 0   -> K-shot per-class (few-shot mode)
    #   k_shot <= 0 AND  train_fraction > 0    -> percent of full train (phase-transition mode)
    # k_shot wins if both are set.
    k_shot: int = 32                    # K examples per class in TRAIN (few-shot mode)
    train_fraction: float = 0.0         # fraction of train to use (phase-transition mode)
    n_classes: int = 6                  # 6 for CoDET author, 2 for binary, 3 for Droid T3
    fs_seed: int = 42                   # sampling seed (reported in tracker)

    # Encoder
    encoder_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 384               # T4-friendly (was 512 on H100)

    # Optimization
    # Few-shot K=32 needs only 1 epoch; %-fraction with thousands of samples
    # benefits from 3 epochs + LR warmup. Auto-tuned in apply_hardware_profile.
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

    # Regime-dependent epoch tuning. K-shot has so few train samples that
    # 1 epoch is enough; %-fraction needs more for the LR schedule to mean
    # anything. We never set epochs > 5 -- the original Exp_Climb climb runs
    # used 3 epochs on 100K samples, so 3 here on >5K samples is comparable.
    fewshot_mode = cfg.k_shot > 0 and cfg.train_fraction <= 0
    if fewshot_mode:
        cfg.epochs = 1
        cfg.warmup_ratio = 0.0
    else:
        cfg.epochs = 3
        cfg.warmup_ratio = 0.1
    logger.info(
        f"[hw-profile] regime={'K-shot' if fewshot_mode else 'fraction'} "
        f"epochs={cfg.epochs} warmup={cfg.warmup_ratio}"
    )
    return cfg


# ---------------------------------------------------------------------------
# Output formatting
# ---------------------------------------------------------------------------

def emit_paper_table(method_name: str, exp_id: str, cfg: FSConfig, results: Dict):
    """Emit BEGIN_FS_TABLE block AND save the same payload as JSON.

    JSON path:
      - /kaggle/working/results/<exp_id>_<label>.json   (Kaggle, downloadable)
      - ./results/<exp_id>_<label>.json                  (local fallback)

    The label encodes the regime so multiple sweeps don't overwrite each other:
      K-shot:    <exp_id>_K<n>_seed<s>.json
      fraction:  <exp_id>_frac<n>_seed<s>.json
    """
    import json
    from datetime import datetime

    fewshot_mode = cfg.k_shot > 0 and cfg.train_fraction <= 0
    regime_label = f"K={cfg.k_shot}" if fewshot_mode else f"frac={cfg.train_fraction:.4f}"
    file_label = (
        f"K{cfg.k_shot}" if fewshot_mode
        else f"frac{cfg.train_fraction:.4f}".rstrip("0").rstrip(".")
    )
    lines = [
        f"BEGIN_FS_TABLE method={method_name} exp_id={exp_id}",
        f"  benchmark:    {cfg.benchmark}",
        f"  task:         {cfg.task}",
        f"  regime:       {regime_label}",
        f"  k_shot:       {cfg.k_shot}",
        f"  train_fraction: {cfg.train_fraction:.4f}",
        f"  n_classes:    {cfg.n_classes}",
        f"  fs_seed:      {cfg.fs_seed}",
        f"  epochs:       {cfg.epochs}",
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

    # ----- Save same payload as JSON (downloadable from Kaggle) -----
    payload = {
        "method": method_name,
        "exp_id": exp_id,
        "regime": regime_label,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {
            "benchmark": cfg.benchmark,
            "task": cfg.task,
            "k_shot": cfg.k_shot,
            "train_fraction": cfg.train_fraction,
            "n_classes": cfg.n_classes,
            "fs_seed": cfg.fs_seed,
            "encoder": cfg.encoder_name,
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "max_length": cfg.max_length,
            "precision": cfg.precision,
            "lr_encoder": cfg.lr_encoder,
            "lr_heads": cfg.lr_heads,
            "lambda_ntk": cfg.lambda_ntk,
        },
        "results": _jsonify(results),
    }

    # Pick the right destination: Kaggle (POSIX + /kaggle/working real) else local.
    # On Windows local, "/kaggle/working" resolves to D:\kaggle\working and might
    # exist accidentally, so we also require POSIX.
    candidate_dirs = []
    if os.name == "posix" and os.path.isdir("/kaggle/working"):
        candidate_dirs.append("/kaggle/working/results")
    candidate_dirs.append("./results")

    saved = False
    for d in candidate_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{exp_id}_{file_label}_seed{cfg.fs_seed}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(f"[json] saved -> {path}")
            saved = True
            break
        except (OSError, PermissionError) as e:
            logger.warning(f"[json] could not write to {d}: {e}")
            continue
    if not saved:
        logger.warning("[json] all save paths failed; result available only in stdout")


def _jsonify(obj):
    """Recursively convert numpy / torch scalars + dict-keys to JSON-safe types."""
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if hasattr(obj, "item"):  # numpy / torch scalar
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return str(obj)


# ---------------------------------------------------------------------------
# Sweep helpers (each exp file calls these from main())
# ---------------------------------------------------------------------------

def parse_sweep_configs():
    """Read FS_SWEEP_KS and FS_SWEEP_FRACS from env, return list of configs.

    Default sweep is conservative for a single Kaggle T4 session:
      K-shot:    K = 32, 128                      (~2 quick runs)
      Fraction:  f = 0.01, 0.05                   (~2 medium runs)
    Override:
      FS_SWEEP_KS="8,16,32,64,128"
      FS_SWEEP_FRACS="0.01,0.05,0.1,0.2"
      FS_SWEEP_KS=""                  -> skip K-shot regime
      FS_SWEEP_FRACS=""               -> skip fraction regime

    Returns:
      list of (kind, value) where kind in {'kshot', 'fraction'}.
    """
    raw_ks = os.environ.get("FS_SWEEP_KS", "32,128").strip()
    raw_fracs = os.environ.get("FS_SWEEP_FRACS", "0.01,0.05").strip()
    out = []
    if raw_ks:
        for x in raw_ks.split(","):
            if x.strip():
                out.append(("kshot", int(x.strip())))
    if raw_fracs:
        for x in raw_fracs.split(","):
            if x.strip():
                out.append(("fraction", float(x.strip())))
    return out


def cleanup_after_run():
    """Release GPU memory between sweep runs so a fresh model can fit."""
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_sweep_summary(summary, exp_id: str, method_name: str):
    """Pretty per-method leaderboard at the end of a sweep.

    summary: list of (kind, value, test_f1, val_f1, wall_s) tuples.
    """
    print(f"\n{'=' * 70}")
    print(f"[{exp_id}] SWEEP SUMMARY -- {method_name}")
    print(f"{'=' * 70}")
    header = f"{'config':<16}{'test_F1':>10}{'val_F1':>10}{'gap':>10}{'wall':>10}"
    print(header)
    print("-" * len(header))
    for kind, value, test_f1, val_f1, wall_s in summary:
        label = f"K={value}" if kind == "kshot" else f"frac={value:.4f}"
        gap = val_f1 - test_f1
        print(f"{label:<16}{test_f1:>10.4f}{val_f1:>10.4f}{gap:>+10.4f}{wall_s:>10.0f}")
    print("-" * len(header))
    print(f"{len(summary)} configs total. JSON files saved to /kaggle/working/results/")
