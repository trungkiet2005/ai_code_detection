"""
exp16_ce_baseline.py — Cross-Entropy baseline with standardized few-shot protocol

Self-contained. Runs: 2 encoders × 2 benchmarks = 4 experiments.

Key Protocol Change (v16):
  - FIXED_TOTAL_TRAIN = 72: All benchmarks get same 72 total training samples
  - This ensures fair comparison across different dataset sizes
  - CoDET-M4: 72 = 12 samples × 6 classes
  - AICD-T2: 72 = 6 samples × 12 classes

Config:
  - Encoders: ModernBERT-base, unixcoder-base
  - Benchmarks: codet_m4 (6 classes), aicd_t2 (12 classes)
  - Total training: 72 samples (fixed)
  - Batch: 256, seq=512

Usage:
  python exp16_ce_baseline.py
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
