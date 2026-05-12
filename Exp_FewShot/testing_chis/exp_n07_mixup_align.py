"""
exp_n07_mixup_align.py — Mixup with Gradient Alignment for Source Invariance

NAME : Mixup Gradient Alignment (MGA)
ONE-LINE CLAIM : Mixup interpolates between examples, but source (CF/LC/GH)
leaks into features. MGA adds a gradient penalty that decorrelates
representation from source to achieve do(S)-invariance.
EQUATION : L_mga = CE(λx₁+(1-λ)x₂, λy₁+(1-λ)y₂) + λ_gp · ||∂L/∂z - E[∂L/∂z]||²
THEORY HOOK : Arjovsky et al. 2019 IRM; source invariance is the key to
OOD generalization in the causal attribution model.
WHY NOT BEFORE : Standard mixup does not address source confounding;
IRM adds complexity. MGA is a lightweight gradient penalty for invariance.
FALSIFIER : If source-probe accuracy does not drop under MGA, the
gradient penalty is not enforcing invariance.

Target: EMNLP Oral — Theory contribution (novel gradient penalty object).

Self-contained. Runs: 2 encoders × 4 benchmarks × 3 fractions = 24 experiments.

Config:
  - Encoders: ModernBERT-base, unixcoder-base
  - Benchmarks: codet_m4 (headline), aicd_t2 (stress), droid_t3, droid_t4
  - Fractions: 0.01, 0.05, 0.20
  - Batch: 256, seq=512

Usage:
  python exp_n07_mixup_align.py
"""
from __future__ import annotations


# =============================================================================
# Theory-Track exp — Mixup Gradient Alignment (MGA):
# mixup with gradient penalty for source invariance in few-shot attribution.
#
# NAME : Mixup Gradient Alignment (MGA).
# ONE-LINE CLAIM : Standard mixup interpolates examples; MGA adds a gradient
# penalty that decorrelates representations from source (CF/LC/GH).
# EQUATION : L_mga = CE(mix(x₁,x₂), mix(y₁,y₂)) + λ_gp · ||∇_z L - E[∇_z L]||²
# where mix(x₁,x₂) = λx₁ + (1-λ)x₂.
# PROPERTY : The gradient penalty forces the representation to ignore
# source-specific features, achieving approximate do(S)-invariance.
# In few-shot attribution, this means the model learns generator signal
# that is invariant to the coding platform (CF vs LC vs GH).
# WHY NOT BEFORE : IRM (Exp_02) uses a complex penalty. MGA is simpler:
# mixup provides data augmentation, gradient penalty enforces invariance.
# FALSIFIER : Train a linear probe to predict source S from embeddings.
# If source-probe accuracy does not drop under MGA, invariance is not achieved.
# =============================================================================

# === KAGGLE PATHS ===
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_DROID = "/kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/data"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from collections import defaultdict
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler

def _autocast_ctx(dev):
    return autocast(enabled=(dev.type == "cuda"))
