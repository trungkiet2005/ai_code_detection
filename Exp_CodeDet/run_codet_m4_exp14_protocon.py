"""
[CoDET-M4] ProtoCon Runner – Full Benchmark Evaluation  (Kaggle standalone)
      Exp14: ProtoCon (Prototype Contrastive Memory Bank)

Key novelty: Prototype memory bank with EMA updates + hyperspherical uniformity
regularization maximally separates generator classes on the unit sphere.
Directly addresses Nxcode/Qwen1.5 authorship confusion.

Inspired by: DINO (ICCV 2021), ProtoNet, NeurIPS 2024 memory bank contrastive,
             hyperspherical uniformity (NeurIPS 2018 Wang et al.)

Target: NeurIPS/ICLR 2026 (A* venue)

Architecture:
  ModernBERT-base (answerdotai/ModernBERT-base, SDPA attention)
  + ProjectionHead: 768 → 512 → 256, L2-normalised
  + PrototypicalMemoryBank: EMA-updated class prototypes (256-dim), momentum=0.995
  + HypersphericalUniformityLoss: repels prototypes on unit sphere
  + SupConLoss: supervised contrastive, temperature=0.07, hard-negative mining
  + ClassifierHead: linear 768 → num_classes

Loss: L_focal + 0.3*L_proto_attract + 0.1*L_unif + 0.2*L_supcon

Baseline (Exp11 SpectralCode):
  Binary IID:  99.06 Macro-F1
  Author IID:  69.82 Macro-F1  (Nxcode/Qwen confusion is key weakness)
"""

import importlib.util
import json
import math
import os
import random
import subprocess
import sys
import logging
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap – auto-install missing packages (Kaggle-friendly)
# ---------------------------------------------------------------------------

def _ensure_deps():
    required = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("sklearn", "scikit-learn"),
        ("accelerate", "accelerate"),
    ]
    missing = [pip for imp, pip in required if importlib.util.find_spec(imp) is None]
    if missing:
        print(f"[bootstrap] Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

_ensure_deps()

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader

try:
    from torch.amp import autocast as _autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler
    _NEW_AMP = False

from datasets import Dataset, load_dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

def autocast(device_type="cuda", enabled=True, dtype=None):
    if _NEW_AMP:
        if dtype is None:
            return _autocast(device_type=device_type, enabled=enabled)
        return _autocast(device_type=device_type, enabled=enabled, dtype=dtype)
    return _autocast(enabled=enabled)


class PreflightError(RuntimeError):
    pass


def _get_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class ProtoConConfig:
    # Model
    encoder_name: str = "answerdotai/ModernBERT-base"
    encoder_hidden_size: int = 768
    proj_hidden_dim: int = 512
    proj_out_dim: int = 256          # L2-normalised embedding dim
    max_length: int = 512

    # Prototype memory bank
    proto_momentum: float = 0.995
    proto_init_batches: int = 1      # number of batches used for prototype init

    # Loss weights
    focal_gamma: float = 2.0
    w_proto_attract: float = 0.3
    w_unif: float = 0.1
    w_supcon: float = 0.2
    supcon_temperature: float = 0.07

    # Training
    epochs: int = 3
    batch_size: int = 32
    grad_accum_steps: int = 2
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Data caps
    max_train_samples: int = 100_000
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000

    # Hardware / runtime
    num_workers: int = 2
    prefetch_factor: int = 2
    seed: int = 42
    precision: str = "auto"          # "auto" | "bf16" | "fp16" | "fp32"
    auto_h100_profile: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    non_blocking: bool = True
    save_dir: str = "./codet_m4_protocon_ckpts"
    log_every: int = 100
    eval_every: int = 1000

    # Internal (set at runtime)
    task: str = "binary"
    benchmark: str = "codet_m4"


def apply_hardware_profile(cfg: ProtoConConfig) -> ProtoConConfig:
    """Tune settings for detected GPU (H100 gets bf16 + larger batch)."""
    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if not (cfg.auto_h100_profile and cfg.device == "cuda"):
        return cfg

    gpu_name = _get_gpu_name().upper()
    if "H100" not in gpu_name:
        return cfg

    cfg.precision = "bf16" if cfg.precision == "auto" else cfg.precision
    cfg.batch_size = max(cfg.batch_size, 64)
    cfg.grad_accum_steps = 1
    cfg.num_workers = max(cfg.num_workers, 8)
    cfg.prefetch_factor = max(cfg.prefetch_factor, 4)
    cfg.log_every = max(cfg.log_every, 200)
    cfg.eval_every = max(cfg.eval_every, 2000)
    logger.info(
        "Applied H100 profile | precision=%s | batch=%d | accum=%d | workers=%d",
        cfg.precision, cfg.batch_size, cfg.grad_accum_steps, cfg.num_workers,
    )
    return cfg


# ============================================================================
# CoDET-M4 Dataset Config & Utilities
# ============================================================================

@dataclass
class CoDETM4Config:
    dataset_id: str = "DaniilOr/CoDET-M4"
    code_field_priority: Tuple[str, ...] = ("cleaned_code", "code")
    split_field: str = "split"

    task: str = "binary"   # "binary" | "author"

    max_train_samples: int = 100_000
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000

    eval_breakdown: bool = True
    seed: int = 42
    save_root: str = "./codet_m4_protocon_ckpts"


# ---------------------------------------------------------------------------
# Author label constants (fixed across all runs)
# ---------------------------------------------------------------------------
AUTHOR_CLASSES = ["human", "codellama", "gpt", "llama3.1", "nxcode", "qwen1.5"]
AUTHOR_VOCAB: Dict[str, int] = {name: idx for idx, name in enumerate(AUTHOR_CLASSES)}


def _sample_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), max_samples)
    return dataset.select(indices)


def _extract_code(row: Dict[str, object], code_fields: Tuple[str, ...]) -> str:
    for fname in code_fields:
        val = row.get(fname, "")
        if isinstance(val, str) and val.strip():
            return val
    return ""


def _normalize_target(value: object) -> str:
    return str(value or "").strip().lower()


def _is_human_target(target: str) -> bool:
    return target in {"human", "human_written", "human-generated", "human_generated"}


def _build_author_vocab_from_split(train_split: Dataset) -> Dict[str, int]:
    """Build author vocab from training split; falls back to AUTHOR_VOCAB."""
    # First try the 'generator' column (CoDET-M4 native)
    if "generator" in train_split.column_names:
        names = set(str(x).strip().lower() for x in train_split["generator"])
        names.discard("")
    else:
        names = set()
        for row in train_split:
            target = _normalize_target(row.get("target", ""))
            model_name = str(row.get("model", "") or "").strip().lower()
            if not _is_human_target(target) and model_name:
                names.add(model_name)

    # Always include human as class 0; align with AUTHOR_CLASSES when possible
    aligned = {name: idx for idx, name in enumerate(AUTHOR_CLASSES) if name in names or name == "human"}
    if len(aligned) < 2:
        # Fallback: build dynamically
        aligned = {"human": 0}
        for i, name in enumerate(sorted(names - {"human"}), start=1):
            aligned[name] = i
    return aligned


def _map_binary_label(row: Dict[str, object]) -> int:
    # CoDET-M4 uses 'generator' column: 'human' → 0, else → 1
    gen = str(row.get("generator", "") or "").strip().lower()
    if gen == "human":
        return 0
    # Fallback to legacy 'target' / 'label' fields
    target = _normalize_target(row.get("target", ""))
    if _is_human_target(target):
        return 0
    label_val = row.get("label", None)
    if label_val is not None:
        try:
            lv = int(label_val)
            return 0 if lv == 0 else 1
        except (ValueError, TypeError):
            pass
    return 1


def _map_author_label(row: Dict[str, object], author_vocab: Dict[str, int]) -> int:
    gen = str(row.get("generator", "") or "").strip().lower()
    if gen:
        return author_vocab.get(gen, -1)
    target = _normalize_target(row.get("target", ""))
    if _is_human_target(target):
        return author_vocab.get("human", 0)
    model_name = str(row.get("model", "") or "").strip().lower()
    return author_vocab.get(model_name, -1)


def _convert_split(
    split_ds: Dataset,
    cfg: CoDETM4Config,
    author_vocab: Dict[str, int],
) -> Dataset:
    task = cfg.task

    def _convert_row(row):
        code = _extract_code(row, cfg.code_field_priority)
        if not code.strip():
            # Try plain 'code' key as last resort
            code = str(row.get("code", "") or "").strip()
        if task == "binary":
            label = _map_binary_label(row)
        elif task == "author":
            label = _map_author_label(row, author_vocab)
        else:
            raise ValueError(f"Unsupported task: {task}")

        gen_raw = str(row.get("generator", "") or "").strip().lower()
        return {
            "code": code,
            "label": label,
            "language": str(row.get("language", "") or "").strip().lower(),
            "source": str(row.get("source", "") or "").strip().lower(),
            "generator": gen_raw if gen_raw else "unknown",
        }

    converted = split_ds.map(_convert_row, remove_columns=split_ds.column_names)
    converted = converted.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)
    return converted


def _quick_code_stats(dataset: Dataset, sample_size: int = 512) -> Dict[str, float]:
    n = min(len(dataset), sample_size)
    if n == 0:
        return {"samples": 0, "avg_chars": 0.0, "avg_lines": 0.0, "empty_ratio": 1.0}
    sample = dataset.select(range(n))
    codes = sample["code"]
    lengths = [len(c) for c in codes]
    lines = [c.count("\n") + 1 for c in codes]
    empty = sum(1 for c in codes if len(c.strip()) == 0)
    return {
        "samples": n,
        "avg_chars": float(np.mean(lengths)),
        "avg_lines": float(np.mean(lines)),
        "empty_ratio": float(empty / n),
    }


# ============================================================================
# Data Loading: IID
# ============================================================================

def _load_raw_splits(cfg: CoDETM4Config):
    logger.info("Loading dataset: %s", cfg.dataset_id)
    ds = load_dataset(cfg.dataset_id, split="train")

    has_split = cfg.split_field in ds.column_names
    if has_split:
        logger.info("Using built-in split column from CoDET-M4")
        train_raw = ds.filter(lambda x: str(x.get(cfg.split_field, "")).lower() == "train")
        val_raw = ds.filter(lambda x: str(x.get(cfg.split_field, "")).lower() in {"val", "validation", "dev"})
        test_raw = ds.filter(lambda x: str(x.get(cfg.split_field, "")).lower() == "test")
    else:
        logger.warning("No split column; creating 80/10/10 with seed=%d", cfg.seed)
        s1 = ds.train_test_split(test_size=0.1, seed=cfg.seed)
        test_raw = s1["test"]
        s2 = s1["train"].train_test_split(test_size=1 / 9, seed=cfg.seed)
        train_raw, val_raw = s2["train"], s2["test"]

    if len(train_raw) == 0 or len(val_raw) == 0 or len(test_raw) == 0:
        raise RuntimeError("One or more CoDET-M4 splits are empty.")
    return train_raw, val_raw, test_raw


def load_codet_m4_data(cfg: CoDETM4Config):
    train_raw, val_raw, test_raw = _load_raw_splits(cfg)

    author_vocab = _build_author_vocab_from_split(train_raw) if cfg.task == "author" else {}
    if cfg.task == "author":
        logger.info("Author vocab (%d classes): %s", len(author_vocab), sorted(author_vocab.items(), key=lambda x: x[1]))

    train_data = _convert_split(train_raw, cfg, author_vocab)
    val_data = _convert_split(val_raw, cfg, author_vocab)
    test_data = _convert_split(test_raw, cfg, author_vocab)

    train_data = _sample_dataset(train_data, cfg.max_train_samples, cfg.seed)
    val_data = _sample_dataset(val_data, cfg.max_val_samples, cfg.seed + 1)
    test_data = _sample_dataset(test_data, cfg.max_test_samples, cfg.seed + 2)

    num_classes = len(set(train_data["label"]))
    logger.info("IID | train=%d val=%d test=%d | classes=%d", len(train_data), len(val_data), len(test_data), num_classes)
    return train_data, val_data, test_data, num_classes


# ============================================================================
# Data Loading: OOD Leave-One-Out
# ============================================================================

def load_codet_m4_loo(
    cfg: CoDETM4Config,
    hold_out_field: str,
    hold_out_value: str,
) -> Tuple[Dataset, Dataset, Dataset, int]:
    train_raw, val_raw, test_raw = _load_raw_splits(cfg)

    # CoDET-M4 uses native column names directly
    raw_field = hold_out_field  # "generator", "language", "source"

    def _matches(row):
        val = str(row.get(raw_field, "") or "").strip().lower()
        return val == hold_out_value.lower()

    train_in = train_raw.filter(lambda x: not _matches(x))
    val_in = val_raw.filter(lambda x: not _matches(x))
    test_ood = test_raw.filter(_matches)

    if len(train_in) == 0 or len(val_in) == 0:
        raise RuntimeError(
            f"LOO train/val empty: hold_out={hold_out_field}={hold_out_value} "
            f"| train_in={len(train_in)} val_in={len(val_in)}"
        )
    if len(test_ood) == 0:
        raise RuntimeError(
            f"LOO test OOD empty: hold_out={hold_out_field}={hold_out_value} "
            f"| test_ood={len(test_ood)}"
        )

    author_vocab = _build_author_vocab_from_split(train_in) if cfg.task == "author" else {}

    train_data = _convert_split(train_in, cfg, author_vocab)
    val_data = _convert_split(val_in, cfg, author_vocab)
    test_data = _convert_split(test_ood, cfg, author_vocab)

    train_data = _sample_dataset(train_data, cfg.max_train_samples, cfg.seed)
    val_data = _sample_dataset(val_data, cfg.max_val_samples, cfg.seed + 1)
    test_data = _sample_dataset(test_data, cfg.max_test_samples, cfg.seed + 2)

    num_classes = len(set(train_data["label"]))
    logger.info(
        "LOO[%s != %s] | train=%d val=%d test_ood=%d | classes=%d",
        hold_out_field, hold_out_value, len(train_data), len(val_data), len(test_data), num_classes,
    )
    return train_data, val_data, test_data, num_classes


# ============================================================================
# PyTorch Dataset
# ============================================================================

class CoDETDataset(TorchDataset):
    """Minimal tokenizing dataset — no AST/spectral overhead for ProtoCon."""

    def __init__(self, data: Dataset, tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        code: str = item["code"]
        label: int = int(item["label"])

        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


def _make_loaders(
    train_data: Dataset,
    val_data: Dataset,
    test_data: Dataset,
    tokenizer,
    cfg: ProtoConConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = CoDETDataset(train_data, tokenizer, cfg.max_length)
    val_ds = CoDETDataset(val_data, tokenizer, cfg.max_length)
    test_ds = CoDETDataset(test_data, tokenizer, cfg.max_length)

    pin = cfg.pin_memory and cfg.device == "cuda"
    kw: Dict[str, Any] = {
        "num_workers": cfg.num_workers,
        "pin_memory": pin,
        "persistent_workers": cfg.num_workers > 0,
    }
    if cfg.num_workers > 0:
        kw["prefetch_factor"] = cfg.prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size * 2, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size * 2, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ============================================================================
# Model Components
# ============================================================================

class ProjectionHead(nn.Module):
    """2-layer MLP: encoder_dim → proj_hidden → proj_out, L2-normalised output."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class PrototypeMemoryBank(nn.Module):
    """
    Stores one L2-normalised prototype per class.
    EMA update: proto[c] = m * proto[c] + (1-m) * mean(batch_z[c])
    Initialised lazily on first call via `init_from_batch`.
    """

    def __init__(self, num_classes: int, embed_dim: int, momentum: float = 0.995):
        super().__init__()
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.momentum = momentum
        # Buffer: not a parameter, not updated by optimiser
        self.register_buffer("prototypes", torch.zeros(num_classes, embed_dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def init_from_batch(self, z: torch.Tensor, labels: torch.Tensor):
        """Hard initialisation from a single batch; called once before training loop."""
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() > 0:
                proto_c = F.normalize(z[mask].mean(dim=0, keepdim=False), dim=-1)
                self.prototypes[c] = proto_c
                self.initialized[c] = True

    @torch.no_grad()
    def ema_update(self, z: torch.Tensor, labels: torch.Tensor):
        """EMA update from a mini-batch."""
        for c in range(self.num_classes):
            mask = labels == c
            if mask.sum() == 0:
                continue
            mean_z = F.normalize(z[mask].mean(dim=0), dim=-1)
            if self.initialized[c]:
                updated = self.momentum * self.prototypes[c] + (1.0 - self.momentum) * mean_z
                self.prototypes[c] = F.normalize(updated, dim=-1)
            else:
                self.prototypes[c] = mean_z
                self.initialized[c] = True

    def get_prototypes(self) -> torch.Tensor:
        """Return all prototypes; uninitialised ones are zero-vectors (ignored in loss)."""
        return self.prototypes  # (num_classes, embed_dim)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss (Khosla et al. NeurIPS 2020).
    Uses hard-negative mining: negatives with highest cosine similarity to anchor
    are emphasised via the standard SupCon formulation.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (N, D) L2-normalised embeddings
            labels: (N,) integer class labels
        Returns:
            scalar loss
        """
        device = z.device
        n = z.size(0)
        if n < 2:
            return z.new_tensor(0.0)

        # Cosine similarity matrix (already L2-normed)
        sim = torch.matmul(z, z.T) / self.temperature  # (N, N)

        # Mask: positive pairs share the same label (exclude self)
        labels_row = labels.unsqueeze(1)  # (N, 1)
        labels_col = labels.unsqueeze(0)  # (1, N)
        pos_mask = (labels_row == labels_col).float()  # (N, N)
        # Remove self-pairs
        eye = torch.eye(n, device=device)
        pos_mask = pos_mask * (1.0 - eye)

        # Numerical stability: subtract row-max
        sim_max, _ = sim.max(dim=1, keepdim=True)
        sim = sim - sim_max.detach()

        exp_sim = torch.exp(sim) * (1.0 - eye)  # (N, N), exclude self

        # Log-sum-exp over all negatives
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)  # (N, N)

        # Mean over positive pairs per anchor
        num_pos = pos_mask.sum(dim=1)  # (N,)
        valid = num_pos > 0
        if valid.sum() == 0:
            return z.new_tensor(0.0)

        loss_per_anchor = -(pos_mask * log_prob).sum(dim=1) / (num_pos + 1e-8)
        return loss_per_anchor[valid].mean()


def hyperspherical_uniformity_loss(prototypes: torch.Tensor, initialized: torch.Tensor) -> torch.Tensor:
    """
    Hyperspherical Uniformity Loss (Wang & Isola, NeurIPS 2018).
    L_unif = log( mean( exp(-2 * ||p_i - p_j||^2) ) )  for all i < j
    Repels all class prototypes apart on the unit sphere.

    Args:
        prototypes: (C, D) L2-normalised prototype vectors
        initialized: (C,) bool mask – only use initialised prototypes
    Returns:
        scalar uniformity loss (minimise to spread prototypes)
    """
    active = prototypes[initialized]  # (K, D)
    k = active.size(0)
    if k < 2:
        return prototypes.new_tensor(0.0)

    # Pairwise squared Euclidean distance via ||p_i - p_j||^2 = 2 - 2*<p_i, p_j>
    # (since vectors are on unit sphere)
    sim = torch.matmul(active, active.T)  # (K, K)
    sq_dist = 2.0 - 2.0 * sim  # (K, K)

    # Extract upper triangle (i < j)
    idx = torch.triu_indices(k, k, offset=1, device=active.device)
    pairwise_sq = sq_dist[idx[0], idx[1]]  # (K*(K-1)/2,)

    uniformity = torch.log(torch.exp(-2.0 * pairwise_sq).mean() + 1e-8)
    return uniformity


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce)
        return ((1.0 - pt) ** self.gamma * ce).mean()


# ============================================================================
# Full Model: ProtoConModel
# ============================================================================

class ProtoConModel(nn.Module):
    """
    ProtoCon = ModernBERT-base + ProjectionHead + PrototypeMemoryBank + ClassifierHead.

    Forward returns:
        logits          (N, C)  – for CE / Focal loss
        z               (N, D)  – L2-normalised projection embeddings
    """

    def __init__(self, cfg: ProtoConConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes

        # Backbone
        self.backbone = AutoModel.from_pretrained(
            cfg.encoder_name,
            attn_implementation="sdpa",
        )
        hidden_size = self.backbone.config.hidden_size  # 768 for ModernBERT-base

        # Projection head (contrastive branch)
        self.projector = ProjectionHead(
            input_dim=hidden_size,
            hidden_dim=cfg.proj_hidden_dim,
            output_dim=cfg.proj_out_dim,
        )

        # Prototype memory bank
        self.memory_bank = PrototypeMemoryBank(
            num_classes=num_classes,
            embed_dim=cfg.proj_out_dim,
            momentum=cfg.proto_momentum,
        )

        # Classifier head (CE / focal branch; operates on raw [CLS])
        self.classifier = nn.Linear(hidden_size, num_classes)

    def encode(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return (cls_repr, z_proj)."""
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        cls = out.last_hidden_state[:, 0, :]   # (N, hidden_size)
        z = self.projector(cls)                # (N, proj_out_dim), L2-normed
        return cls, z

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        cls, z = self.encode(input_ids, attention_mask)
        logits = self.classifier(cls)
        return {"logits": logits, "z": z}


# ============================================================================
# Training utilities
# ============================================================================

def _resolve_precision(cfg: ProtoConConfig) -> Tuple[bool, torch.dtype]:
    precision = cfg.precision.lower()
    if precision == "auto":
        precision = "bf16" if cfg.device == "cuda" else "fp32"
    use_amp = cfg.device == "cuda" and precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return use_amp, amp_dtype


def _build_optimizer(model: ProtoConModel, cfg: ProtoConConfig) -> torch.optim.Optimizer:
    encoder_params = list(model.backbone.parameters())
    head_params = [p for n, p in model.named_parameters() if "backbone" not in n]
    return torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.lr_encoder, "weight_decay": cfg.weight_decay},
            {"params": head_params, "lr": cfg.lr_heads, "weight_decay": cfg.weight_decay},
        ]
    )


# ============================================================================
# Train Epoch
# ============================================================================

def train_epoch(
    model: ProtoConModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    cfg: ProtoConConfig,
    focal_loss_fn: FocalLoss,
    supcon_loss_fn: SupConLoss,
    epoch_idx: int,
    use_amp: bool,
    amp_dtype: torch.dtype,
) -> Dict[str, float]:
    model.train()
    total_losses: Dict[str, float] = defaultdict(float)
    num_batches = 0
    global_step = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(cfg.device, non_blocking=cfg.non_blocking)
        attention_mask = batch["attention_mask"].to(cfg.device, non_blocking=cfg.non_blocking)
        labels = batch["label"].to(cfg.device, non_blocking=cfg.non_blocking)

        with autocast(device_type=cfg.device, enabled=use_amp, dtype=amp_dtype):
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]
            z = outputs["z"]

            # 1) Focal loss (CE branch)
            l_focal = focal_loss_fn(logits, labels)

            # 2) Prototype attraction: mean(1 - cos_sim(z_i, proto[c_i]))
            protos = model.memory_bank.get_prototypes()  # (C, D)
            target_protos = protos[labels]               # (N, D)
            cos_sim_to_proto = (z * target_protos).sum(dim=-1)   # (N,)
            l_proto = (1.0 - cos_sim_to_proto).mean()

            # 3) Hyperspherical uniformity (repulsion between prototypes)
            l_unif = hyperspherical_uniformity_loss(protos, model.memory_bank.initialized)

            # 4) Supervised contrastive loss
            l_supcon = supcon_loss_fn(z, labels)

            loss = (
                l_focal
                + cfg.w_proto_attract * l_proto
                + cfg.w_unif * l_unif
                + cfg.w_supcon * l_supcon
            ) / cfg.grad_accum_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % cfg.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1

        # EMA update prototypes (detached; no grad needed)
        with torch.no_grad():
            model.memory_bank.ema_update(z.detach(), labels)

        total_losses["focal"] += l_focal.item()
        total_losses["proto_attract"] += l_proto.item()
        total_losses["unif"] += l_unif.item()
        total_losses["supcon"] += l_supcon.item()
        total_losses["total"] += (
            l_focal.item()
            + cfg.w_proto_attract * l_proto.item()
            + cfg.w_unif * l_unif.item()
            + cfg.w_supcon * l_supcon.item()
        )
        num_batches += 1

        if (batch_idx + 1) % cfg.log_every == 0:
            avg = total_losses["total"] / num_batches
            try:
                lr = scheduler.get_last_lr()[0]
            except Exception:
                lr = cfg.lr_encoder
            logger.info(
                "Epoch %d | Step %d/%d | Loss: %.4f | focal=%.4f proto=%.4f unif=%.4f supcon=%.4f | LR: %.2e",
                epoch_idx + 1, batch_idx + 1, len(loader),
                avg,
                total_losses["focal"] / num_batches,
                total_losses["proto_attract"] / num_batches,
                total_losses["unif"] / num_batches,
                total_losses["supcon"] / num_batches,
                lr,
            )

    return {k: v / max(num_batches, 1) for k, v in total_losses.items()}


# ============================================================================
# Evaluate
# ============================================================================

@torch.no_grad()
def evaluate(
    model: ProtoConModel,
    loader: DataLoader,
    cfg: ProtoConConfig,
    use_amp: bool,
    amp_dtype: torch.dtype,
    split_name: str = "Val",
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """Returns (macro_f1, weighted_f1, all_preds, all_labels)."""
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    total_loss = 0.0
    num_batches = 0

    for batch in loader:
        input_ids = batch["input_ids"].to(cfg.device, non_blocking=cfg.non_blocking)
        attention_mask = batch["attention_mask"].to(cfg.device, non_blocking=cfg.non_blocking)
        labels = batch["label"].to(cfg.device, non_blocking=cfg.non_blocking)

        with autocast(device_type=cfg.device, enabled=use_amp, dtype=amp_dtype):
            outputs = model(input_ids, attention_mask)

        preds = outputs["logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        total_loss += F.cross_entropy(outputs["logits"], labels).item()
        num_batches += 1

    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0))
    weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0))
    avg_loss = total_loss / max(num_batches, 1)

    logger.info(
        "%s | Loss: %.4f | Macro-F1: %.4f | Weighted-F1: %.4f",
        split_name, avg_loss, macro_f1, weighted_f1,
    )

    if split_name in ("Test", "OOD-Test"):
        report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
        logger.info("%s Classification Report:\n%s", split_name, report)

    return macro_f1, weighted_f1, np.array(all_preds), np.array(all_labels)


# ============================================================================
# Breakdown Evaluation (per-language, per-source, per-generator)
# ============================================================================

def run_breakdown_eval(
    preds: np.ndarray,
    labels: np.ndarray,
    test_data: Dataset,
    task: str,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    n = min(len(preds), len(test_data))
    preds, labels = preds[:n], labels[:n]

    overall_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    overall_weighted = float(f1_score(labels, preds, average="weighted", zero_division=0))
    results["overall"] = {"macro_f1": overall_macro, "weighted_f1": overall_weighted}
    logger.info("  Overall: macro=%.4f  weighted=%.4f", overall_macro, overall_weighted)

    for dim_name in ["language", "source", "generator"]:
        if dim_name not in test_data.column_names:
            continue
        dim_vals = test_data[dim_name][:n]
        unique = sorted(set(dim_vals))
        dim_results: Dict[str, Any] = {}
        logger.info("  Breakdown by %s:", dim_name)
        for val in unique:
            mask = np.array([v == val for v in dim_vals])
            if mask.sum() < 10:
                continue
            mf1 = float(f1_score(labels[mask], preds[mask], average="macro", zero_division=0))
            wf1 = float(f1_score(labels[mask], preds[mask], average="weighted", zero_division=0))
            dim_results[val] = {"n": int(mask.sum()), "macro_f1": mf1, "weighted_f1": wf1}
            logger.info("    %12s: n=%6d  macro=%.4f  weighted=%.4f", val, int(mask.sum()), mf1, wf1)
        results[dim_name] = dim_results

    if task == "author":
        cm = confusion_matrix(labels, preds)
        results["confusion_matrix"] = cm.tolist()
        logger.info("  Confusion matrix (rows=true, cols=pred):\n  %s", cm)

    return results


# ============================================================================
# Preflight Check
# ============================================================================

def preflight_check(codet_cfg: CoDETM4Config, proto_cfg: ProtoConConfig) -> Dict[str, object]:
    logger.info("=" * 70)
    logger.info("PREFLIGHT | dataset=%s | task=%s", codet_cfg.dataset_id, codet_cfg.task)
    logger.info("=" * 70)

    train_data, val_data, test_data, num_classes = load_codet_m4_data(codet_cfg)
    if num_classes < 2:
        raise PreflightError(f"Invalid class count: {num_classes}")

    label_counts = Counter(train_data["label"])
    code_stats = _quick_code_stats(train_data)
    if code_stats["empty_ratio"] > 0.05:
        raise PreflightError(f"Empty code ratio too high: {code_stats['empty_ratio']:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(proto_cfg.encoder_name)
    probe_code = train_data[0]["code"]
    enc = tokenizer(probe_code, max_length=proto_cfg.max_length, padding="max_length", truncation=True, return_tensors="pt")
    if enc["input_ids"].shape[-1] != proto_cfg.max_length:
        raise PreflightError("Tokenizer output length mismatch.")

    report = {
        "task": codet_cfg.task,
        "num_classes": int(num_classes),
        "sizes": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "code_stats": code_stats,
        "device": proto_cfg.device,
        "gpu": _get_gpu_name(),
        "encoder": proto_cfg.encoder_name,
        "precision": proto_cfg.precision,
    }
    logger.info("PREFLIGHT OK | classes=%d | sizes=%s", num_classes, report["sizes"])
    return report


# ============================================================================
# Core Training Loop (shared by IID + OOD runners)
# ============================================================================

def _run_training(
    train_data: Dataset,
    val_data: Dataset,
    test_data: Dataset,
    num_classes: int,
    codet_cfg: CoDETM4Config,
    proto_cfg: ProtoConConfig,
    task_tag: str,
) -> Dict[str, Any]:
    set_seed(proto_cfg.seed)

    tokenizer = AutoTokenizer.from_pretrained(proto_cfg.encoder_name)
    class_weights = compute_class_weights(train_data["label"], num_classes).to(proto_cfg.device)
    logger.info("Classes=%d | weights=%s", num_classes, class_weights.cpu().numpy())

    train_loader, val_loader, test_loader = _make_loaders(train_data, val_data, test_data, tokenizer, proto_cfg)

    model = ProtoConModel(proto_cfg, num_classes).to(proto_cfg.device)
    tp = sum(p.numel() for p in model.parameters())
    logger.info("ProtoCon parameters: %s total", f"{tp:,}")

    # Loss functions
    focal_loss_fn = FocalLoss(gamma=proto_cfg.focal_gamma, weight=class_weights)
    supcon_loss_fn = SupConLoss(temperature=proto_cfg.supcon_temperature)

    # Optimiser + scheduler
    optimizer = _build_optimizer(model, proto_cfg)
    total_steps = len(train_loader) * proto_cfg.epochs // proto_cfg.grad_accum_steps
    warmup_steps = int(total_steps * proto_cfg.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    use_amp, amp_dtype = _resolve_precision(proto_cfg)
    scaler = GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # ── Prototype initialisation from first train batch ─────────────────
    logger.info("Initialising prototype memory bank from first batch …")
    first_batch = next(iter(train_loader))
    init_ids = first_batch["input_ids"].to(proto_cfg.device)
    init_mask = first_batch["attention_mask"].to(proto_cfg.device)
    init_labels = first_batch["label"].to(proto_cfg.device)
    with torch.no_grad():
        with autocast(device_type=proto_cfg.device, enabled=use_amp, dtype=amp_dtype):
            _, init_z = model.encode(init_ids, init_mask)
        model.memory_bank.init_from_batch(init_z, init_labels)
    logger.info(
        "Prototypes initialised for classes: %s",
        model.memory_bank.initialized.nonzero(as_tuple=False).flatten().tolist(),
    )

    # ── Training loop ────────────────────────────────────────────────────
    best_val_f1 = 0.0
    best_ckpt_path = os.path.join(proto_cfg.save_dir, f"protocon_{task_tag}_best.pt")
    os.makedirs(proto_cfg.save_dir, exist_ok=True)

    for epoch in range(proto_cfg.epochs):
        logger.info("\n%s Epoch %d/%d %s", "=" * 30, epoch + 1, proto_cfg.epochs, "=" * 30)
        train_losses = train_epoch(
            model, train_loader, optimizer, scheduler, scaler,
            proto_cfg, focal_loss_fn, supcon_loss_fn,
            epoch_idx=epoch, use_amp=use_amp, amp_dtype=amp_dtype,
        )
        logger.info(
            "Epoch %d Train: %s",
            epoch + 1,
            " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items()),
        )

        val_f1, val_wf1, _, _ = evaluate(model, val_loader, proto_cfg, use_amp, amp_dtype, split_name="Val")
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "memory_bank": model.memory_bank.state_dict(),
                    "best_val_f1": best_val_f1,
                    "epoch": epoch,
                },
                best_ckpt_path,
            )
            logger.info("*** New best Val Macro-F1: %.4f *** (saved)", best_val_f1)

    # ── Load best checkpoint for final test ─────────────────────────────
    if os.path.exists(best_ckpt_path):
        ckpt = torch.load(best_ckpt_path, map_location=proto_cfg.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        model.memory_bank.load_state_dict(ckpt["memory_bank"])
        logger.info("Loaded best checkpoint (val_f1=%.4f)", ckpt.get("best_val_f1", 0.0))

    logger.info("\n%s FINAL TEST EVALUATION %s", "=" * 20, "=" * 20)
    test_f1, test_wf1, test_preds, test_labels = evaluate(
        model, test_loader, proto_cfg, use_amp, amp_dtype, split_name="Test"
    )
    logger.info("*** Final Test Macro-F1: %.4f | Weighted-F1: %.4f ***", test_f1, test_wf1)

    results: Dict[str, Any] = {
        "test_f1": float(test_f1),
        "test_weighted_f1": float(test_wf1),
        "best_val_f1": float(best_val_f1),
    }

    if codet_cfg.eval_breakdown:
        logger.info("-" * 50)
        logger.info("[BREAKDOWN] task=%s", codet_cfg.task)
        results["breakdown"] = run_breakdown_eval(test_preds, test_labels, test_data, codet_cfg.task)

    # Clean up GPU memory between runs
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return results


# ============================================================================
# IID Suite
# ============================================================================

def run_iid_suite(
    codet_cfg: CoDETM4Config,
    proto_cfg: ProtoConConfig,
    task: str,
    run_preflight: bool = True,
) -> Dict[str, Any]:
    codet_cfg_t = CoDETM4Config(
        dataset_id=codet_cfg.dataset_id,
        code_field_priority=codet_cfg.code_field_priority,
        split_field=codet_cfg.split_field,
        task=task,
        max_train_samples=codet_cfg.max_train_samples,
        max_val_samples=codet_cfg.max_val_samples,
        max_test_samples=codet_cfg.max_test_samples,
        eval_breakdown=codet_cfg.eval_breakdown,
        seed=codet_cfg.seed,
        save_root=codet_cfg.save_root,
    )

    logger.info("=" * 70)
    logger.info("[Exp14][IID] ProtoCon | task=%s | GPU=%s | precision=%s | batch=%dx%d | epochs=%d",
                task, _get_gpu_name(), proto_cfg.precision,
                proto_cfg.batch_size, proto_cfg.grad_accum_steps, proto_cfg.epochs)
    logger.info("=" * 70)

    if run_preflight:
        preflight_check(codet_cfg_t, proto_cfg)

    train_data, val_data, test_data, num_classes = load_codet_m4_data(codet_cfg_t)
    task_tag = f"iid_{task}"
    proto_cfg.task = task

    return _run_training(train_data, val_data, test_data, num_classes, codet_cfg_t, proto_cfg, task_tag)


# ============================================================================
# OOD Suite
# ============================================================================

def _run_single_loo(
    hold_out_field: str,
    hold_out_value: str,
    codet_cfg: CoDETM4Config,
    proto_cfg: ProtoConConfig,
) -> Dict[str, Any]:
    loo_cfg = CoDETM4Config(
        dataset_id=codet_cfg.dataset_id,
        code_field_priority=codet_cfg.code_field_priority,
        split_field=codet_cfg.split_field,
        task="binary",   # OOD always evaluated in binary mode
        max_train_samples=codet_cfg.max_train_samples,
        max_val_samples=codet_cfg.max_val_samples,
        max_test_samples=codet_cfg.max_test_samples,
        eval_breakdown=codet_cfg.eval_breakdown,
        seed=codet_cfg.seed,
        save_root=codet_cfg.save_root,
    )
    proto_cfg.task = "binary"

    logger.info("=" * 70)
    logger.info("[OOD] LOO %s=%s", hold_out_field, hold_out_value)
    logger.info("=" * 70)

    train_data, val_data, test_data, num_classes = load_codet_m4_loo(loo_cfg, hold_out_field, hold_out_value)
    task_tag = f"ood_{hold_out_field}_{hold_out_value}"

    res = _run_training(train_data, val_data, test_data, num_classes, loo_cfg, proto_cfg, task_tag)
    res["hold_out_field"] = hold_out_field
    res["hold_out_value"] = hold_out_value
    return res


def run_ood_suite(
    mode: str,   # "generator" | "language" | "source"
    codet_cfg: CoDETM4Config,
    proto_cfg: ProtoConConfig,
) -> Dict[str, Any]:
    mode_map = {
        "generator": ["codellama", "gpt", "llama3.1", "nxcode", "qwen1.5"],
        "language": ["cpp", "java", "python"],
        "source": ["cf", "gh", "lc"],
    }
    hold_out_values = mode_map.get(mode)
    if hold_out_values is None:
        raise ValueError(f"Unknown OOD mode: {mode}. Choose from: {list(mode_map.keys())}")

    results: Dict[str, Any] = {}
    for val in hold_out_values:
        logger.info("\n%s\n[OOD-%s] Holding out %s=%s\n%s",
                    "=" * 70, mode.upper(), mode, val, "=" * 70)
        try:
            results[val] = _run_single_loo(mode, val, codet_cfg, proto_cfg)
        except RuntimeError as exc:
            logger.error("OOD %s=%s failed: %s", mode, val, exc)
            results[val] = {"error": str(exc)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _log_ood_summary(f"OOD-{mode.upper()}", results)
    return results


def _log_ood_summary(title: str, results: Dict[str, Any]):
    logger.info("\n%s", "=" * 70)
    logger.info("%s SUMMARY", title)
    logger.info("=" * 70)
    logger.info("| held-out | test_macro_f1 | test_weighted_f1 | best_val_f1 |")
    logger.info("|---|---:|---:|---:|")
    for key, stats in results.items():
        if "error" in stats:
            logger.info("| %s | ERROR | - | - |", key)
        else:
            logger.info(
                "| %s | %.4f | %.4f | %.4f |",
                key,
                stats.get("test_f1", 0.0),
                stats.get("test_weighted_f1", 0.0),
                stats.get("best_val_f1", 0.0),
            )


# ============================================================================
# Main – full benchmark orchestration
# ============================================================================

FULL_RUN_PLAN: List[Tuple[str, str]] = [
    ("iid", "binary"),       # Table 2 equiv
    ("iid", "author"),       # Table 7 equiv
    ("ood", "generator"),    # proxy Table 8
    ("ood", "language"),     # proxy Table 10
    ("ood", "source"),       # proxy Table 9
]


def _log_final_summary(all_results: Dict[str, Any], run_plan: List[Tuple[str, str]]):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("\n%s", "=" * 70)
    logger.info("CoDET-M4 BENCHMARK COMPLETE | %s", ts)
    logger.info("=" * 70)

    logger.info("\n=== IID RESULTS ===")
    for mode, task in run_plan:
        if mode != "iid":
            continue
        key = f"iid_{task}"
        r = all_results.get(key, {})
        logger.info(
            "  %8s: macro_f1=%.4f  weighted_f1=%.4f  val_f1=%.4f",
            task,
            r.get("test_f1", 0.0),
            r.get("test_weighted_f1", 0.0),
            r.get("best_val_f1", 0.0),
        )

    for mode, task in run_plan:
        if mode != "ood":
            continue
        key = f"ood_{task}"
        ood_block = all_results.get(key, {})
        if not isinstance(ood_block, dict):
            continue
        logger.info("\n=== OOD %s RESULTS ===", task.upper())
        for held, stats in ood_block.items():
            if isinstance(stats, dict) and "test_f1" in stats:
                logger.info(
                    "  held_out=%12s: macro_f1=%.4f  weighted_f1=%.4f",
                    held,
                    stats["test_f1"],
                    stats.get("test_weighted_f1", 0.0),
                )

    # Build JSON-safe copy (strip numpy arrays)
    def _make_safe(obj):
        if isinstance(obj, dict):
            return {k: _make_safe(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_safe(v) for v in obj]
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist() if hasattr(obj, "tolist") else str(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return obj

    safe = _make_safe(all_results)
    try:
        json_str = json.dumps(
            {"timestamp": ts, "method": "ProtoCon_CoDET_M4_Exp14", "results": safe},
            ensure_ascii=True,
            default=str,
        )
        print(f"\nSUITE_RESULTS_JSON={json_str}")
        logger.info("SUITE_RESULTS_JSON=%s", json_str)
    except (TypeError, ValueError) as exc:
        logger.warning("JSON serialisation of full results skipped: %s", exc)


def main():
    # ── Run configuration ────────────────────────────────────────────────
    # "full"      → IID binary, IID author, OOD generator/language/source
    # "iid_only"  → IID binary + author only
    # "ood_only"  → OOD suites only
    RUN_MODE = "full"
    RUN_PREFLIGHT = True

    codet_cfg = CoDETM4Config(
        max_train_samples=100_000,
        max_val_samples=20_000,
        max_test_samples=50_000,
        eval_breakdown=True,
        save_root="./codet_m4_protocon_ckpts",
    )

    # Build base ProtoConConfig and apply hardware profile
    proto_cfg = ProtoConConfig(
        epochs=3,
        batch_size=32,
        grad_accum_steps=2,
        precision="auto",
        auto_h100_profile=True,
        num_workers=4,
        prefetch_factor=2,
        save_dir=codet_cfg.save_root,
    )
    proto_cfg = apply_hardware_profile(proto_cfg)

    logger.info("\n%s", "=" * 70)
    logger.info("CoDET-M4 FULL BENCHMARK: ProtoCon (Exp14)")
    logger.info("GPU: %s | precision: %s | batch: %dx%d | epochs: %d",
                _get_gpu_name(), proto_cfg.precision,
                proto_cfg.batch_size, proto_cfg.grad_accum_steps, proto_cfg.epochs)
    logger.info("=" * 70)

    if RUN_MODE == "full":
        run_plan = FULL_RUN_PLAN
    elif RUN_MODE == "iid_only":
        run_plan = [e for e in FULL_RUN_PLAN if e[0] == "iid"]
    elif RUN_MODE == "ood_only":
        run_plan = [e for e in FULL_RUN_PLAN if e[0] == "ood"]
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")

    logger.info("Run plan (%d entries):", len(run_plan))
    for i, (mode, task) in enumerate(run_plan):
        logger.info("  [%d] %s/%s", i + 1, mode, task)

    all_results: Dict[str, Any] = {}

    try:
        for mode, task in run_plan:
            key = f"{mode}_{task}"
            logger.info("\n%s\nSUITE: %s\n%s", "#" * 70, key, "#" * 70)

            if mode == "iid":
                all_results[key] = run_iid_suite(codet_cfg, proto_cfg, task, run_preflight=RUN_PREFLIGHT)
                RUN_PREFLIGHT = False  # only run preflight once

            elif mode == "ood":
                all_results[key] = run_ood_suite(task, codet_cfg, proto_cfg)
            else:
                logger.warning("Unknown suite entry: %s/%s — skipping", mode, task)

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    except PreflightError as exc:
        logger.error("PRE-FLIGHT FAILED: %s", exc)
        raise SystemExit(1)

    _log_final_summary(all_results, run_plan)


if __name__ == "__main__":
    main()
