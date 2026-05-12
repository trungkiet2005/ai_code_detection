"""
================================================================================
Theory-Track exp -- Hyperbolic Prototype Attribution (HPA):
Class prototypes in Poincaré ball where genealogy trees embed with zero distortion.

ARXIV_ID      : Nickel 2017 Poincaré embeddings (1705.08039); Khrulkov 2020
                hyperbolic image embeddings (1904.02239); Fang 2021 hyperbolic
                few-shot (2005.00966)
NAME          : Hyperbolic Prototype Attribution (HPA)
ONE-LINE CLAIM: Computing class prototypes in the Poincaré ball exploits the fact
                that trees embed with exponentially less distortion in hyperbolic
                vs Euclidean geometry — the generator genealogy IS a tree, so
                hyperbolic prototypes are the natural representation space.
EQUATION      : d_P(u,v) = arcosh(1 + 2||u-v||²/((1-||u||²)(1-||v||²)))
                c_k = ⊕_{i:y_i=k} z_i / n_k  (Einstein midpoint in Poincaré ball)
                L = -log(exp(-d_P(z, c_y)²) / Σ_k exp(-d_P(z, c_k)²))
PROPERTY      : In Euclidean space, embedding a binary tree of depth D requires
                O(2^D) dimensions. In hyperbolic space, O(D) suffices.
                CoDET-M4's genealogy has depth 3 → hyperbolic needs only 3
                effective dimensions vs ~8 Euclidean for equivalent distortion.
WHY NOT BEFORE: Hyperbolic embeddings exist for hierarchical NLP and few-shot
                image classification, but never for code attribution where the
                label tree is the MODEL GENEALOGY. Our contribution: prototypes
                live WHERE the tree lives.
FALSIFIER     : If hyperbolic prototypes do not improve family-group accuracy
                over Euclidean prototypes (exp20_proto), curvature prior is wrong.
================================================================================

exp34_hyper_proto.py — Hyperbolic Prototype Attribution for few-shot AI-code attribution.
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
