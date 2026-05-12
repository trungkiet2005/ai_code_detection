"""
================================================================================
Theory-Track exp -- Rényi Attribution Divergence (RAD):
α-parameterized loss interpolating max-likelihood to minimax attribution.

ARXIV_ID      : Rényi 1961 information measures; Van Erven 2014 (1206.2459);
                Li 2021 Rényi robust learning (2103.12028)
NAME          : Rényi Attribution Divergence (RAD)
ONE-LINE CLAIM: Replacing cross-entropy (α=1 limit) with Rényi-α divergence
                provides a single knob that interpolates between maximum-likelihood
                (α→1) and minimax/worst-case (α→∞) — the optimal α depends on
                the few-shot regime (samples per class).
EQUATION      : L_α(p, y) = 1/(α-1) · log(Σ_k p_k^α · y_k)
                α=1: recovers CE. α=2: down-weights confident samples.
                α=0.5: emphasises uncertain predictions (calibration).
PROPERTY      : In few-shot, CE over-fits to frequent patterns. Higher α (e.g. 2)
                provides implicit regularization by flattening the loss landscape.
                The optimal α is a function of n/K — this α-regime connection is
                our novel theoretical claim.
WHY NOT BEFORE: Rényi divergence is foundational in information theory but has
                never been used as a classification loss for code attribution.
                The claim that α* ∝ log(n/K) links information theory to the
                few-shot phase transition.
FALSIFIER     : If optimal α does not shift across 1%→5%→20% regimes,
                the Rényi-regime connection is wrong.
================================================================================

exp33_renyi.py — Rényi Attribution Divergence for few-shot AI-code attribution.
Protocol: fraction-based (1% / 5% / 20%), unixcoder-base only.
Sweeps α ∈ {0.5, 1.0, 2.0, 5.0} at each fraction.
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
