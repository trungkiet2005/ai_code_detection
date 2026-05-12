"""
================================================================================
Theory-Track exp -- Spectral Contrastive Learning (SCL):
Spectral methods for contrastive representation learning.

ARXIV_ID      : ICML 2021 Spectral (2010.07875); NeurIPS 2021 SCAN (2103.15595)
NAME          : Spectral Contrastive Learning (SCL)
ONE-LINE CLAIM: Using spectral methods to align contrastive representations improves
                clustering structure in embedding space.
EQUATION      : L_scl = -log(exp(sim(z_i, z_j)/τ)) / Σ_{k≠i} exp(sim(z_i, z_k)/τ)
                where z_i are spectrally-normalized embeddings.
PROPERTY      : Spectral normalization ensures Lipshitz continuity, providing
                better contrastive learning dynamics.
WHY NOT BEFORE: Standard contrastive learning doesn't enforce Lipschitz continuity.
                Spectral normalization provides this guarantee.
FALSIFIER     : If SCL does NOT improve cluster separation (higher silhouette score)
                without hurting accuracy, the spectral component is not helping.
================================================================================

exp28_spectral.py — Spectral contrastive learning for few-shot AI-code attribution.
Protocol: FIXED_TOTAL_TRAIN = 72 samples across all benchmarks.
"""
from __future__ import annotations


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
