"""
================================================================================
Theory-Track exp -- Kernel Alignment Curriculum (KAC):
Phase-transition curriculum from our own Theorem 2.

ARXIV_ID      : OUR Theorem 2 (§2.4 CLAUDE.md); Bengio 2009 curriculum (1206.6416)
NAME          : Kernel Alignment Curriculum (KAC)
ONE-LINE CLAIM: Our Theorem 2 predicts a phase transition at n* = Θ(K·h/λ_min²).
                KAC dynamically weights the genealogy kernel loss based on estimated
                distance from n*: below n*, trust the encoder; above n*, trust the tree.
EQUATION      : λ_tree(n) = σ(β · (n - n*) / n*)
                L = CE + λ_tree(n) · L_htka
                n << n*: λ_tree ≈ 0 (encoder regime)
                n >> n*: λ_tree ≈ 1 (genealogy regime)
PROPERTY      : This is the ONLY experiment that directly tests our own theorem.
                If the curriculum schedule matches the empirical phase transition,
                Theorem 2 is validated. If not, the theorem is falsified.
WHY NOT BEFORE: Curriculum learning uses sample difficulty (Bengio 2009) or
                loss values (Jiang 2018). KAC uses THEORETICAL phase transition
                as the schedule. This is curriculum from first principles.
FALSIFIER     : If the optimal λ_tree does not exhibit a sigmoid transition near n*,
                Theorem 2's n* formula is wrong.
================================================================================

exp35_kac.py — Kernel Alignment Curriculum for few-shot AI-code attribution.
Protocol: fraction-based (1% / 5% / 20%), unixcoder-base only.
"""
from __future__ import annotations


# === KAGGLE PATHS ===
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_DROID = "/kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/data"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
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
