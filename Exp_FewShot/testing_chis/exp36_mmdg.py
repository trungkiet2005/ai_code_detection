"""
================================================================================
Theory-Track exp -- MMD with Genealogical Kernel (MMDG):
Maximum Mean Discrepancy with tree-structured kernel for distributional attribution.

ARXIV_ID      : Gretton 2012 MMD (JMLR 13:723-773); Muandet 2017 kernel mean
                embedding (1605.09522); Li 2015 generative moment matching (1502.02761)
NAME          : Maximum Mean Discrepancy with Genealogical Kernel (MMDG)
ONE-LINE CLAIM: Using the genealogy tree to define a characteristic kernel for MMD
                creates a distributional test that measures whether same-family
                code embeddings are drawn from the same distribution — a stronger
                signal than point-wise classification.
EQUATION      : k_gene(y_i, y_j) = exp(-d_tree(y_i, y_j)² / (2σ²))
                MMD²(F_k, F_l) = E[k(z_i,z_j)] - 2E[k(z_i,z_l)] + E[k(z_l,z_m)]
                L = CE + λ · Σ_{(k,l) related} MMD²(F_k, F_l)
PROPERTY      : MMD with genealogical kernel tests whether sibling generators
                produce distinguishable code distributions. Maximising inter-family
                MMD while minimising intra-family variance respects genealogical distance.
WHY NOT BEFORE: MMD has been used for distribution matching (GAN, domain adaptation)
                but never with a kernel defined by label genealogy. Our kernel
                k_gene makes the two-sample test genealogy-aware.
FALSIFIER     : If MMDG does not improve sibling separation (sibling-pair F1)
                over CE baseline, the genealogical kernel is not characteristic.
================================================================================

exp36_mmdg.py — MMD with Genealogical Kernel for few-shot AI-code attribution.
Protocol: fraction-based (1% / 5% / 20%), unixcoder-base only.
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
