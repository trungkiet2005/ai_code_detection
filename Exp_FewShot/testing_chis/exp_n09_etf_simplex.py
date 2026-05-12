"""
exp_n09_etf_simplex.py — Equiangular Tight Frame Simplex Classifier

NAME : Equiangular Tight Frame Simplex (ETF-Simplex)
ONE-LINE CLAIM : Neural collapse predicts classifiers align to an ETF simplex.
This method explicitly parameterizes the classifier as an ETF, giving the
optimal K-class configuration a priori.
EQUATION : W_etf[i] = sqrt(K/(K-1)) · (e_i - 1/K · 1) where {e_i} is standard basis
PROPERTY : The ETF simplex has equidistant class means on the sphere,
maximizing inter-class separation and minimizing intra-class variance.
WHY NOT BEFORE : Standard classifiers learn arbitrary directions.
ETF gives the optimal K-class geometry from first principles.
FALSIFIER : If ETF parameterization does not improve over learnable
classifiers at frac=0.05, then the neural collapse prior is wrong.

Target: EMNLP Oral — Theory contribution (novel classifier object).

Self-contained. Runs: 2 encoders × 4 benchmarks × 3 fractions = 24 experiments.

Config:
  - Encoders: ModernBERT-base, unixcoder-base
  - Benchmarks: codet_m4 (headline), aicd_t2 (stress), droid_t3, droid_t4
  - Fractions: 0.01, 0.05, 0.20
  - Batch: 256, seq=512

Usage:
  python exp_n09_etf_simplex.py
"""
from __future__ import annotations


# =============================================================================
# Theory-Track exp — Equiangular Tight Frame Simplex (ETF-Simplex):
# optimal K-class geometry from neural collapse theory.
#
# NAME : Equiangular Tight Frame Simplex (ETF-Simplex).
# ONE-LINE CLAIM : The ETF simplex is the optimal K-class configuration,
# with equidistant class means maximizing inter-class separation.
# EQUATION : W_etf[i] = sqrt(K/(K-1)) · (e_i - 1/K · 1)
# where {e_i} is the standard basis in R^K.
# PROPERTY : The ETF classifier has K equidistant directions on the
# sphere. Each class mean aligns to its ETF direction, achieving
# maximum classification margin by construction.
# WHY NOT BEFORE : Standard learnable classifiers discover this geometry
# only with prolonged training (Neural Collapse). ETF-Simplex encodes
# this prior explicitly, critical in few-shot regimes.
# FALSIFIER : Compare ETF vs learnable classifier at frac=0.05.
# If ETF does not improve, neural collapse geometry is not optimal.
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
