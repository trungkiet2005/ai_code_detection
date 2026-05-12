"""
================================================================================
Theory-Track exp -- Spectral Genealogy Embedding (SGE):
Laplacian eigenvectors of the genealogy graph as target coordinate system.

ARXIV_ID      : Belkin 2003 Laplacian Eigenmaps; Von Luxburg 2007 spectral
                clustering tutorial; Spielman 2012 spectral graph theory.
NAME          : Spectral Genealogy Embedding (SGE)
ONE-LINE CLAIM: The eigenvectors of the genealogy graph Laplacian provide a smooth,
                low-dimensional target coordinate system where sibling generators
                are naturally close — aligning embeddings to this basis gives a
                continuous relaxation of the discrete tree prior.
EQUATION      : L_gene = D - A  (graph Laplacian of genealogy tree)
                {v_1, ..., v_K} = eigenvectors of L_gene sorted by eigenvalue
                L_sge = ||Z·Z^T - V·V^T||_F  (Frobenius alignment)
PROPERTY      : The Fiedler vector (2nd eigenvector) of L_gene naturally cuts
                the tree into families. Higher eigenvectors refine within families.
                This spectral basis is smoother than one-hot and richer than HIER_FAM.
WHY NOT BEFORE: Graph Laplacian spectral methods have been applied to DATA graphs
                (GNN, spectral clustering) but never to the LABEL structure graph.
                Our label graph IS the genealogy tree — its spectrum IS the attribution basis.
FALSIFIER     : If spectral embedding alignment does not improve family-group
                accuracy over flat one-hot, the spectral structure is not informative.
================================================================================

exp32_spectral_gene.py — Spectral Genealogy Embedding for few-shot AI-code attribution.
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
