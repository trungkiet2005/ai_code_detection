"""
================================================================================
Theory-Track exp -- Contrastive Mixup Attribution (CMA):
Mixup augmentation with contrastive positive pairs.

ARXIV_ID      : arXiv 2017 Mixup (1710.09412); ICCV 2019 CutMix (1905.04849)
NAME          : Contrastive Mixup Attribution (CMA)
ONE-LINE CLAIM: Mixup between same-family samples creates interpolated features
                that improve boundary smoothness while preserving family identity.
EQUATION      : x_mix = λx_i + (1-λ)x_j, y_mix = λy_i + (1-λ)y_j
                where same-family pairs (y_i == y_j) are positive.
PROPERTY      : Mixup regularizes decision boundaries without introducing
                cross-family confusion.
WHY NOT BEFORE: Standard mixup mixes any classes. CMA uses genealogical
                structure to only mix within families.
FALSIFIER     : If CMA does NOT improve boundary stability (lower logit entropy)
                without hurting accuracy, the mixup strategy is suboptimal.
================================================================================

exp27_cmixup.py — Contrastive mixup for few-shot AI-code attribution.
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
