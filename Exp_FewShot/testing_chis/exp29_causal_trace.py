"""
================================================================================
Theory-Track exp -- Causal Feature Tracing (CFT):
Tracing causal features in code representations for attribution.

ARXIV_ID      : ICLR 2021 RACE (2010.14497); ICML 2021 DARE (2105.06487)
NAME          : Causal Feature Tracing (CFT)
ONE-LINE CLAIM: Identifying and emphasizing causally-relevant features improves
                attribution by removing spurious correlations.
EQUATION      : RACE Score: S(x) = ||∂L/∂x|| · |x - μ|
                Features with high RACE scores are causally relevant.
PROPERTY      : Gradient-based feature importance identifies features that
                actually affect the prediction, not just correlate.
WHY NOT BEFORE: Standard attention identifies what the model looks at, not
                what causes the prediction.
FALSIFIER     : If CFT does NOT improve attribution by emphasizing causal
                features, the gradient-based approach is not correct.
================================================================================

exp29_causal_trace.py — Causal feature tracing for few-shot AI-code attribution.
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
