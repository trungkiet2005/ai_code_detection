"""Shared bootstrap + config for Exp_ZeroShot.

Zero-shot protocol: NO training. Every exp_zs_NN_*.py computes a per-sample
score s(x), calibrates a threshold τ on the Droid dev split at target
coverage α = 0.95 (human-recall pin per paper Table 5), and reports binary
test F1 + human/adversarial recall on the Droid test split.

Budget on H100 BF16: ~3-10 min per exp (one or two forward passes, no
training). Separate folder from Exp_Climb/ to avoid mixing the training
story with the zero-shot story in the main paper.
"""
from __future__ import annotations

import logging
import os
import random
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("exp_zs")
if not logger.handlers:
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(h)
    # Don't propagate to the root logger -- otherwise messages are printed
    # twice when the root logger also has a handler (e.g. basicConfig is
    # called elsewhere, or under pytest/uvicorn).
    logger.propagate = False
logger.setLevel(logging.INFO)


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class ZSConfig:
    """Config for a zero-shot detector.

    All zero-shot runs SHARE this config. Method-specific hyperparams live
    on the exp_zs_NN_*.py file.
    """
    # Target benchmark: "droid_T3" (3-class) or "droid_T1" (2-class binary)
    # or "codet_binary". Zero-shot focuses on Droid because the ZS baselines
    # (Fast-DetectGPT, GPTZero, ...) are directly comparable to the Droid
    # paper Table 3/4 zero-shot block.
    benchmark: str = "droid_T3"

    # Evaluation budget
    max_dev_samples: int = 5_000     # for threshold calibration
    max_test_samples: int = -1       # -1 = FULL test

    # Threshold calibration target: pin human recall at >= this value
    # (paper Table 5 shows DroidDetectCLS-L holds 0.98 -- our target 0.95
    # with zero training is a strong claim).
    human_recall_target: float = 0.95

    # Scoring LM (for methods that need a proxy LM: Fast-DetectGPT,
    # Binoculars). Default = a small code LM that fits alongside ModernBERT
    # on H100 BF16. Override per exp if needed.
    scorer_lm: str = "microsoft/codebert-base-mlm"
    scorer_max_length: int = 512

    # Backbone for spectral/AST feature extraction (matches Exp_Climb).
    backbone: str = "answerdotai/ModernBERT-base"

    # Hardware (defaults are CPU-safe; apply_hardware_profile auto-bumps
    # to H100 80GB preset at runtime if the GPU is detected).
    batch_size: int = 32
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = False
    device: str = "cuda"   # auto-downgraded to cpu if unavailable
    precision: str = "bf16"

    # Fast-DetectGPT curvature estimator: Monte-Carlo mask samples per code.
    # Paper uses 100; we default to 3 on CPU and auto-bump to 10 on H100.
    fast_detect_n_samples: int = 3

    # Binoculars cosine-entropy block size (bump on GPU to limit O(N^2)
    # memory; 256 is fine on CPU, 1024 on H100).
    binoculars_block_size: int = 256

    seed: int = 42
    save_root: str = "./zs_outputs"


def resolve_device(requested: str = "cuda") -> str:
    try:
        import torch
        if requested == "cuda" and torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def calibrate_threshold_at_human_recall(
    scores: np.ndarray,
    labels: np.ndarray,
    target_human_recall: float,
    human_label: int = 0,
) -> Tuple[float, Dict[str, float]]:
    """Select a decision threshold on a score array so that human recall on
    the dev split equals or exceeds `target_human_recall`. Score semantics:
    **higher score = more likely AI**. Threshold τ: predict AI iff s >= τ.

    Returns (tau, dev_metrics dict with keys human_recall, overall_acc).
    """
    is_human = labels == human_label
    human_scores = scores[is_human]
    if len(human_scores) == 0:
        logger.warning("No human samples in calibration split; using median")
        return float(np.median(scores)), {"human_recall": 0.0}

    # We want P(predict human | is human) >= target_human_recall.
    # Predict human iff s < τ. So we want the τ such that
    # (1 - target_human_recall) fraction of human scores exceed τ.
    # Equivalently, τ = quantile(human_scores, target_human_recall).
    tau = float(np.quantile(human_scores, target_human_recall))

    preds = (scores >= tau).astype(int)
    preds_human_mask = (preds == 0) & is_human
    dev_human_recall = float(preds_human_mask.sum() / max(is_human.sum(), 1))
    # Binarise labels for overall acc on the dev split
    y_bin = (labels != human_label).astype(int)
    acc = float((preds == y_bin).mean())
    return tau, {
        "human_recall": dev_human_recall,
        "overall_acc": acc,
        "n_human": int(is_human.sum()),
        "n_ai": int((~is_human).sum()),
    }


def apply_hardware_profile(cfg: ZSConfig) -> ZSConfig:
    """Auto-tune ZSConfig for the detected GPU.

    Presets:
      H100 80GB / A100 80GB:
        batch 128, bf16, 8 workers, prefetch 4, pin_memory,
        Fast-DetectGPT MC samples bumped 3 -> 10,
        Binoculars block 256 -> 1024 (limits O(N^2) cosine matrix footprint).
      A100 40GB / V100 / T4:
        batch 64, fp16 fallback if bf16 unsupported.
      CPU:
        batch 8, fp32, single worker.
    """
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            # Detect ~80GB (H100, A100 80GB) via memory
            total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            if total_gb >= 70:
                # H100 80GB / A100 80GB preset.
                # ModernBERT-base @ batch 128 seq 512 bf16 ≈ 6-7 GB activations
                # + CodeBERT-base for Fast-DetectGPT adds another ~3 GB. Both
                # fit comfortably in 80 GB with plenty of headroom.
                cfg.batch_size = 128
                cfg.precision = "bf16"
                cfg.num_workers = 8
                cfg.prefetch_factor = 4
                cfg.pin_memory = True
                cfg.fast_detect_n_samples = 10
                cfg.binoculars_block_size = 1024
                logger.info(
                    f"[ZS hw-profile] {name} ({total_gb:.0f} GB) detected -> "
                    f"batch={cfg.batch_size} bf16 workers={cfg.num_workers} "
                    f"prefetch={cfg.prefetch_factor} pin_memory={cfg.pin_memory} "
                    f"FDG_samples={cfg.fast_detect_n_samples} "
                    f"bino_block={cfg.binoculars_block_size}"
                )
            elif total_gb >= 30:
                # A100 40GB / V100 32GB preset
                cfg.batch_size = 64
                cfg.precision = "bf16" if ("A100" in name or "H100" in name) else "fp16"
                cfg.num_workers = 4
                cfg.prefetch_factor = 2
                cfg.pin_memory = True
                cfg.fast_detect_n_samples = 5
                cfg.binoculars_block_size = 512
                logger.info(f"[ZS hw-profile] {name} ({total_gb:.0f} GB) -> batch=64 {cfg.precision}")
            else:
                # Smaller consumer / T4 / P100
                cfg.batch_size = 32
                cfg.precision = "fp16"
                cfg.num_workers = 2
                logger.info(f"[ZS hw-profile] {name} ({total_gb:.0f} GB) -> batch=32 fp16")
        else:
            # CPU fallback: shrink everything so laptop tests run
            cfg.batch_size = 8
            cfg.precision = "fp32"
            cfg.num_workers = 0
            cfg.prefetch_factor = 2
            cfg.pin_memory = False
            logger.info("[ZS hw-profile] No CUDA -> CPU preset: batch=8 fp32")
    except ImportError:
        pass
    cfg.device = resolve_device(cfg.device)
    return cfg
