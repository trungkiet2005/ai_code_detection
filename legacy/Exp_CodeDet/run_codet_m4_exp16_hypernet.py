"""
[CoDET-M4] HyperNetCode Runner – Full Benchmark Evaluation  (Kaggle standalone)
      Exp16: HyperNetCode (Style-Content Disentanglement via HyperNetwork)

Key novelty: Factorizes code representations into content (what code does) and
style (how the generator writes). A HyperNetwork generates generator-specific
classifier weights from the style embedding. MMD loss enforces content invariance
while SupCon on style_z maximizes generator separability.

Inspired by: HyperStyle (CVPR 2022), ICLR 2025 disentanglement papers,
             MMD-AAE (ICML 2018), Style-Content separation in NLP

Target: NeurIPS/ICLR 2026 (A* venue)
"""

import importlib.util
import json
import math
import os
import random
import re
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
class HyperNetConfig:
    # Model
    encoder_name: str = "answerdotai/ModernBERT-base"
    encoder_hidden: int = 768
    content_dim: int = 256      # content_z dimension
    style_dim: int = 128        # style_z dimension
    max_length: int = 512

    # Training
    epochs: int = 3
    batch_size: int = 32
    grad_accum_steps: int = 2
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Loss weights
    focal_gamma: float = 2.0
    w_style_aux: float = 0.2
    w_mmd: float = 0.3
    w_style_con: float = 0.2
    mmd_sigma: float = 1.0
    supcon_temperature: float = 0.07

    # Data
    max_train_samples: int = 100_000
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000

    # Hardware / runtime
    num_workers: int = 2
    prefetch_factor: int = 2
    pin_memory: bool = True
    non_blocking: bool = True
    precision: str = "auto"
    auto_h100_profile: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    save_dir: str = "./hypernet_checkpoints"
    log_every: int = 200
    eval_every: int = 2000


def apply_hardware_profile(config: HyperNetConfig) -> HyperNetConfig:
    """Enable TF32, and upgrade to H100 batch/workers when on H100."""
    if config.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if not (config.auto_h100_profile and config.device == "cuda"):
        return config

    gpu_name = _get_gpu_name().upper()
    if "H100" not in gpu_name:
        return config

    config.precision = "bf16" if config.precision == "auto" else config.precision
    config.batch_size = max(config.batch_size, 64)
    config.grad_accum_steps = 1
    config.num_workers = max(config.num_workers, 8)
    config.prefetch_factor = max(config.prefetch_factor, 4)
    config.log_every = 200
    config.eval_every = 2000
    logger.info(
        "Applied H100 profile | precision=%s | batch=%d | accum=%d | workers=%d",
        config.precision, config.batch_size, config.grad_accum_steps, config.num_workers,
    )
    return config


def preflight_check(cfg: HyperNetConfig, num_classes: int, train_data: Dataset) -> Dict[str, Any]:
    """Validate configuration and data before training."""
    logger.info("=" * 70)
    logger.info("PREFLIGHT | HyperNetCode | Exp16")
    logger.info("=" * 70)

    if num_classes < 2:
        raise PreflightError(f"Invalid class count: {num_classes}")

    n = min(len(train_data), 256)
    sample = train_data.select(range(n))
    codes = sample["code"]
    empty = sum(1 for c in codes if len(c.strip()) == 0)
    empty_ratio = empty / max(n, 1)
    if empty_ratio > 0.05:
        raise PreflightError(f"Empty code ratio too high: {empty_ratio:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
    probe = codes[0]
    enc = tokenizer(probe, max_length=cfg.max_length, padding="max_length",
                    truncation=True, return_tensors="pt")
    if enc["input_ids"].shape[-1] != cfg.max_length:
        raise PreflightError("Tokenizer output length mismatch.")

    label_counts = Counter(train_data["label"])
    report = {
        "num_classes": num_classes,
        "train_size": len(train_data),
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "empty_ratio": float(empty_ratio),
        "device": cfg.device,
        "gpu": _get_gpu_name(),
        "encoder": cfg.encoder_name,
        "precision": cfg.precision,
    }
    logger.info("PREFLIGHT OK | classes=%d | train=%d | gpu=%s", num_classes, len(train_data), cfg.device)
    return report


# ============================================================================
# Dataset
# ============================================================================

class CoDETDataset(TorchDataset):
    """Minimal tokenization dataset for HyperNetCode (no AST needed)."""

    def __init__(self, data: Dataset, tokenizer, max_length: int = 512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        code = item["code"]
        label = item["label"]

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


# ============================================================================
# Model Components
# ============================================================================

class ContentEncoder(nn.Module):
    """
    2-layer MLP: 768 → 512 → 256, with L2-normalization.
    Learns semantic content that should be invariant across generators.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 512, output_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        z = self.net(cls_embedding)
        return F.normalize(z, p=2, dim=-1)


class StyleEncoder(nn.Module):
    """
    2-layer MLP: 768 → 256 → 128, with L2-normalization.
    Captures generator-specific coding style fingerprint.
    """

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
        )
        self.output_dim = output_dim

    def forward(self, cls_embedding: torch.Tensor) -> torch.Tensor:
        z = self.net(cls_embedding)
        return F.normalize(z, p=2, dim=-1)


class HyperNetwork(nn.Module):
    """
    Maps style_z (128d) → classifier weight matrix (num_classes × concat_dim).
    The generated weights W are used as: logits = einsum('nc,bc->bn', W, concat_z)

    This allows the classification boundary to be dynamically shaped by the
    inferred generator style, rather than using a fixed shared classifier.
    """

    def __init__(self, style_dim: int = 128, num_classes: int = 6, concat_dim: int = 384):
        super().__init__()
        self.num_classes = num_classes
        self.concat_dim = concat_dim

        # HyperNet: style_z → flattened weight matrix
        self.hyper = nn.Sequential(
            nn.Linear(style_dim, style_dim * 2),
            nn.LayerNorm(style_dim * 2),
            nn.GELU(),
            nn.Linear(style_dim * 2, num_classes * concat_dim),
        )
        # Bias generator
        self.hyper_bias = nn.Linear(style_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # Small init to prevent degenerate solutions early in training
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, style_z: torch.Tensor, concat_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            style_z:  (B, style_dim)
            concat_z: (B, concat_dim)  = concat(content_z, style_z)
        Returns:
            logits: (B, num_classes)
            W_norms: (num_classes,) — per-class weight norms for degenerate detection
        """
        B = style_z.size(0)
        # Generate per-sample weight matrices
        W_flat = self.hyper(style_z)  # (B, num_classes * concat_dim)
        W = W_flat.view(B, self.num_classes, self.concat_dim)  # (B, num_classes, concat_dim)
        b = self.hyper_bias(style_z)  # (B, num_classes)

        # logits[b,c] = W[b,c,:] · concat_z[b,:]
        logits = torch.einsum("bnc,bc->bn", W, concat_z) + b  # (B, num_classes)

        # Log norms per class (averaged over batch) for degenerate weight detection
        W_norms = W.norm(dim=-1).mean(dim=0)  # (num_classes,)

        return logits, W_norms


class AuxClassifierHead(nn.Module):
    """
    Direct CE classifier on style_z for auxiliary style discrimination.
    Encourages style_z to be discriminative even without the HyperNet.
    """

    def __init__(self, style_dim: int = 128, num_classes: int = 6):
        super().__init__()
        self.fc = nn.Linear(style_dim, num_classes)

    def forward(self, style_z: torch.Tensor) -> torch.Tensor:
        return self.fc(style_z)


# ============================================================================
# Loss Functions
# ============================================================================

def mmd_rbf_loss(
    x: torch.Tensor,
    y: torch.Tensor,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Maximum Mean Discrepancy with RBF kernel.
    Minimizing this pushes the distributions of x and y together.

    Used to make content_z class-agnostic (invariant to generator identity).
    x: content_z from human samples
    y: content_z from AI samples
    """
    def rbf_kernel(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        # ||a_i - b_j||^2 via expansion
        aa = (a * a).sum(dim=1, keepdim=True)  # (n, 1)
        bb = (b * b).sum(dim=1, keepdim=True)  # (m, 1)
        ab = torch.mm(a, b.t())                 # (n, m)
        dist_sq = aa + bb.t() - 2.0 * ab        # (n, m)
        dist_sq = dist_sq.clamp(min=0.0)
        return torch.exp(-dist_sq / (2.0 * sigma ** 2))

    nx, ny = x.size(0), y.size(0)
    if nx < 2 or ny < 2:
        return x.new_zeros(1).squeeze()

    Kxx = rbf_kernel(x, x)
    Kyy = rbf_kernel(y, y)
    Kxy = rbf_kernel(x, y)

    mmd = (
        (Kxx.sum() - Kxx.trace()) / (nx * (nx - 1))
        + (Kyy.sum() - Kyy.trace()) / (ny * (ny - 1))
        - 2.0 * Kxy.mean()
    )
    return mmd.clamp(min=0.0)


class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss on style_z vectors.
    Pushes same-generator embeddings together, different generators apart.

    Reference: Khosla et al. (NeurIPS 2020) — adapted for style embeddings.
    Temperature = 0.07 to sharpen the distribution.
    """

    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: (B, D) — L2-normalized style_z
            labels:   (B,)   — class labels (generator IDs)
        Returns:
            scalar loss
        """
        B = features.size(0)
        if B < 2:
            return features.new_zeros(1).squeeze()

        # Similarity matrix
        sim = torch.mm(features, features.t()) / self.temperature  # (B, B)

        # Mask diagonal (self-similarity)
        diag_mask = torch.eye(B, dtype=torch.bool, device=features.device)
        sim = sim.masked_fill(diag_mask, float("-inf"))

        # Positive mask: same label, different index
        labels_row = labels.unsqueeze(1)  # (B, 1)
        labels_col = labels.unsqueeze(0)  # (1, B)
        pos_mask = (labels_row == labels_col) & ~diag_mask  # (B, B)

        # Skip samples with no positives in batch (common in small batches)
        has_positive = pos_mask.any(dim=1)
        if not has_positive.any():
            return features.new_zeros(1).squeeze()

        sim = sim[has_positive]
        pos_mask = pos_mask[has_positive]

        # Log-softmax over all non-self pairs
        log_probs = F.log_softmax(sim, dim=1)

        # Mean over positive pairs
        # NOTE: use masked_fill instead of direct multiply to avoid 0 * -inf = NaN
        pos_log_probs = (log_probs.masked_fill(~pos_mask, 0.0)).sum(dim=1) / pos_mask.float().sum(dim=1).clamp(min=1.0)
        loss = -pos_log_probs.mean()
        return loss


class FocalLoss(nn.Module):
    """
    Focal loss with class-weighted CE.
    gamma=2.0 focuses on hard examples; class weights handle imbalance.
    """

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else torch.ones(1))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        weight = self.weight.to(logits.device)
        ce = F.cross_entropy(logits, labels, weight=weight, reduction="none")
        pt = torch.exp(-ce)
        focal = ((1.0 - pt) ** self.gamma) * ce
        return focal.mean()


# ============================================================================
# Full Model
# ============================================================================

class HyperNetCodeModel(nn.Module):
    """
    HyperNetCode: Style-Content Disentanglement via HyperNetwork.

    Flow:
        ModernBERT-base [CLS] (768-dim)
             |
      ┌──────┴──────┐
      ContentEncoder  StyleEncoder
      (768→512→256)   (768→256→128)
      L2-norm         L2-norm
             |               |
        content_z (256)  style_z (128)
             |               |
        MMD align       HyperNetwork
      (class-agnostic)  (style_z → W)
             └──── concat (384) ────┘
                         |
               W @ concat(c_z, s_z) → logits
    """

    def __init__(self, cfg: HyperNetConfig, num_classes: int):
        super().__init__()
        self.cfg = cfg
        self.num_classes = num_classes
        self.concat_dim = cfg.content_dim + cfg.style_dim  # 384

        # Backbone
        self.encoder = AutoModel.from_pretrained(cfg.encoder_name)

        # Disentanglement heads
        self.content_encoder = ContentEncoder(
            input_dim=cfg.encoder_hidden,
            hidden_dim=512,
            output_dim=cfg.content_dim,
        )
        self.style_encoder = StyleEncoder(
            input_dim=cfg.encoder_hidden,
            hidden_dim=256,
            output_dim=cfg.style_dim,
        )

        # HyperNetwork for dynamic classifier
        self.hyper_net = HyperNetwork(
            style_dim=cfg.style_dim,
            num_classes=num_classes,
            concat_dim=self.concat_dim,
        )

        # Auxiliary style discriminator (direct style_z → class)
        self.aux_head = AuxClassifierHead(
            style_dim=cfg.style_dim,
            num_classes=num_classes,
        )

    def get_cls_embedding(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token (index 0)
        return out.last_hidden_state[:, 0, :]  # (B, 768)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        cls = self.get_cls_embedding(input_ids, attention_mask)

        content_z = self.content_encoder(cls)   # (B, 256), L2-normalized
        style_z = self.style_encoder(cls)        # (B, 128), L2-normalized

        concat_z = torch.cat([content_z, style_z], dim=-1)  # (B, 384)

        logits, w_norms = self.hyper_net(style_z, concat_z)  # (B, num_classes)
        aux_logits = self.aux_head(style_z)                   # (B, num_classes)

        return {
            "logits": logits,
            "aux_logits": aux_logits,
            "content_z": content_z,
            "style_z": style_z,
            "w_norms": w_norms,
        }


# ============================================================================
# Data utilities (CoDET-M4 specific)
# ============================================================================

def _sample_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), max_samples)
    return dataset.select(indices)


def _normalize_target(value: object) -> str:
    return str(value or "").strip().lower()


def _is_human_target(target: str) -> bool:
    return target in {"human", "human_written", "human-generated", "human_generated"}


def _build_generator_vocab(train_split: Dataset) -> Dict[str, int]:
    """Build a mapping from generator name → integer label.
    human = 0, AI generators sorted alphabetically = 1, 2, ...
    Falls back to 'model' field (CoDET-M4 uses 'model', not 'generator').
    """
    generators = set()
    for row in train_split:
        target = _normalize_target(row.get("target", ""))
        if _is_human_target(target):
            generators.add("human")
        else:
            # CoDET-M4 stores generator name in 'model' field
            gen = str(row.get("model", "") or row.get("generator", "") or "").strip().lower()
            if gen:
                generators.add(gen)
    ai_gens = sorted(g for g in generators if g != "human")
    ordered = (["human"] if "human" in generators else []) + ai_gens
    return {g: i for i, g in enumerate(ordered)}


def _load_raw_splits(dataset_id: str, split_field: str, seed: int):
    logger.info("Loading dataset: %s", dataset_id)
    ds = load_dataset(dataset_id, split="train")

    if split_field in ds.column_names:
        logger.info("Using built-in split column from CoDET-M4")
        train_raw = ds.filter(lambda x: str(x.get(split_field, "")).lower() == "train")
        val_raw = ds.filter(lambda x: str(x.get(split_field, "")).lower() in {"val", "validation", "dev"})
        test_raw = ds.filter(lambda x: str(x.get(split_field, "")).lower() == "test")
    else:
        logger.warning("No split column; creating 80/10/10 with seed=%d", seed)
        s1 = ds.train_test_split(test_size=0.1, seed=seed)
        test_raw = s1["test"]
        s2 = s1["train"].train_test_split(test_size=1 / 9, seed=seed)
        train_raw, val_raw = s2["train"], s2["test"]

    if len(train_raw) == 0 or len(val_raw) == 0 or len(test_raw) == 0:
        raise RuntimeError("One or more CoDET-M4 splits are empty.")
    return train_raw, val_raw, test_raw


def _convert_to_codet_format(
    split_ds: Dataset,
    task: str,
    generator_vocab: Dict[str, int],
) -> Dataset:
    """
    Normalize a raw CoDET-M4 split into a clean dataset with fields:
    code, label, language, source, generator
    """
    def _convert_row(row):
        # Try multiple field names for code content
        code = ""
        for field_name in ("cleaned_code", "code"):
            val = row.get(field_name, "")
            if isinstance(val, str) and val.strip():
                code = val
                break

        target = _normalize_target(row.get("target", ""))
        is_human = _is_human_target(target)
        generator = "human" if is_human else str(row.get("generator", row.get("model", "")) or "").strip().lower()

        if task == "binary":
            label = 0 if is_human else 1
        elif task == "author":
            label = generator_vocab.get(generator, -1)
        else:
            raise ValueError(f"Unknown task: {task}")

        return {
            "code": code,
            "label": label,
            "language": str(row.get("language", "") or "").strip().lower(),
            "source": str(row.get("source", "") or "").strip().lower(),
            "generator": generator,
        }

    converted = split_ds.map(_convert_row, remove_columns=split_ds.column_names)
    converted = converted.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)
    return converted


def load_codet_m4_iid(
    cfg: HyperNetConfig,
    task: str,
    dataset_id: str = "DaniilOr/CoDET-M4",
) -> Tuple[Dataset, Dataset, Dataset, int, Dict[str, int]]:
    train_raw, val_raw, test_raw = _load_raw_splits(dataset_id, "split", cfg.seed)

    # Build vocab from training split only
    if task == "author":
        gen_vocab = _build_generator_vocab(train_raw)
    else:
        # binary: human=0, ai=1
        gen_vocab = {"human": 0, "ai": 1}

    train_data = _convert_to_codet_format(train_raw, task, gen_vocab)
    val_data = _convert_to_codet_format(val_raw, task, gen_vocab)
    test_data = _convert_to_codet_format(test_raw, task, gen_vocab)

    train_data = _sample_dataset(train_data, cfg.max_train_samples, cfg.seed)
    val_data = _sample_dataset(val_data, cfg.max_val_samples, cfg.seed + 1)
    test_data = _sample_dataset(test_data, cfg.max_test_samples, cfg.seed + 2)

    num_classes = len(set(train_data["label"]))
    logger.info(
        "IID[%s] | train=%d val=%d test=%d | classes=%d | vocab=%s",
        task, len(train_data), len(val_data), len(test_data), num_classes, list(gen_vocab.keys())[:8],
    )
    return train_data, val_data, test_data, num_classes, gen_vocab


def load_codet_m4_loo(
    cfg: HyperNetConfig,
    hold_out_field: str,
    hold_out_value: str,
    dataset_id: str = "DaniilOr/CoDET-M4",
) -> Tuple[Dataset, Dataset, Dataset, int, Dict[str, int]]:
    """Leave-one-out split for OOD evaluation."""
    train_raw, val_raw, test_raw = _load_raw_splits(dataset_id, "split", cfg.seed)

    # Map hold_out_field to the raw dataset field name
    raw_field_map = {"generator": "generator", "language": "language", "source": "source"}
    raw_field = raw_field_map.get(hold_out_field, hold_out_field)

    def _matches(row):
        val = str(row.get(raw_field, "") or "").strip().lower()
        return val == hold_out_value.lower()

    train_in = train_raw.filter(lambda x: not _matches(x))
    val_in = val_raw.filter(lambda x: not _matches(x))
    test_ood = test_raw.filter(_matches)

    if len(train_in) == 0 or len(val_in) == 0 or len(test_ood) == 0:
        raise RuntimeError(
            f"LOO split empty: {hold_out_field}={hold_out_value} | "
            f"train_in={len(train_in)} val_in={len(val_in)} test_ood={len(test_ood)}"
        )

    gen_vocab = _build_generator_vocab(train_in)

    train_data = _convert_to_codet_format(train_in, "binary", gen_vocab)
    val_data = _convert_to_codet_format(val_in, "binary", gen_vocab)
    test_data = _convert_to_codet_format(test_ood, "binary", gen_vocab)

    train_data = _sample_dataset(train_data, cfg.max_train_samples, cfg.seed)
    val_data = _sample_dataset(val_data, cfg.max_val_samples, cfg.seed + 1)

    num_classes = len(set(train_data["label"]))
    logger.info(
        "LOO[%s!=%s] | train=%d val=%d test_ood=%d | classes=%d",
        hold_out_field, hold_out_value, len(train_data), len(val_data), len(test_data), num_classes,
    )
    return train_data, val_data, test_data, num_classes, gen_vocab


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Inverse-frequency class weights, clipped to [0.1, 10]."""
    counts = Counter(labels)
    total = len(labels)
    weights = []
    for c in range(num_classes):
        n_c = counts.get(c, 1)
        weights.append(total / (num_classes * n_c))
    w = torch.tensor(weights, dtype=torch.float32)
    w = w.clamp(0.1, 10.0)
    return w / w.sum() * num_classes


def _make_loaders(
    train_data: Dataset,
    val_data: Dataset,
    test_data: Dataset,
    tokenizer,
    cfg: HyperNetConfig,
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
# Training
# ============================================================================

def train_epoch(
    model: HyperNetCodeModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    focal_loss: FocalLoss,
    supcon_loss: SupConLoss,
    cfg: HyperNetConfig,
    epoch: int,
    global_step_start: int,
    scaler: Optional[GradScaler],
    use_amp: bool,
    amp_dtype,
    is_binary: bool,
) -> Tuple[Dict[str, float], int]:
    """
    Single training epoch for HyperNetCode.

    Returns:
        dict of averaged losses, new global_step
    """
    model.train()
    device = cfg.device

    total_losses: Dict[str, float] = defaultdict(float)
    num_batches = 0
    global_step = global_step_start

    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device, non_blocking=cfg.non_blocking)
        attention_mask = batch["attention_mask"].to(device, non_blocking=cfg.non_blocking)
        labels = batch["label"].to(device, non_blocking=cfg.non_blocking)

        with autocast(device_type=device, enabled=use_amp, dtype=amp_dtype):
            outputs = model(input_ids, attention_mask, labels)
            logits = outputs["logits"]
            aux_logits = outputs["aux_logits"]
            content_z = outputs["content_z"]
            style_z = outputs["style_z"]

            # --- Main focal loss ---
            l_focal = focal_loss(logits, labels)

            # --- MMD loss on content_z ---
            # Minimize MMD between human (label=0) and AI (label!=0) content representations
            # to enforce content invariance across generators.
            human_mask = (labels == 0)
            ai_mask = ~human_mask
            if human_mask.sum() >= 2 and ai_mask.sum() >= 2:
                l_mmd = mmd_rbf_loss(
                    content_z[human_mask],
                    content_z[ai_mask],
                    sigma=cfg.mmd_sigma,
                )
            else:
                # Degenerate batch (single class): skip MMD this step
                l_mmd = content_z.new_zeros(1).squeeze()

            # --- Style contrastive loss (SupCon on style_z) ---
            # Binary task: style contrastive is binary (human vs AI)
            # Author task: multi-class SupCon on generator IDs
            if is_binary:
                # Use binary labels for style contrast
                l_style_con = supcon_loss(style_z, labels)
                # Disable aux style loss for binary (would be redundant with 2 classes)
                l_style_aux = aux_logits.new_zeros(1).squeeze()
            else:
                l_style_con = supcon_loss(style_z, labels)
                l_style_aux = F.cross_entropy(aux_logits, labels)

            # --- Total loss ---
            l_total = (
                l_focal
                + cfg.w_mmd * l_mmd
                + cfg.w_style_con * l_style_con
                + cfg.w_style_aux * l_style_aux
            )

        # Gradient accumulation
        accum = cfg.grad_accum_steps
        scaled_loss = l_total / accum

        if use_amp and scaler is not None:
            scaler.scale(scaled_loss).backward()
        else:
            scaled_loss.backward()

        total_losses["total"] += l_total.item()
        total_losses["focal"] += l_focal.item()
        total_losses["mmd"] += l_mmd.item() if not isinstance(l_mmd, float) else l_mmd
        total_losses["style_con"] += l_style_con.item() if not isinstance(l_style_con, float) else l_style_con
        total_losses["style_aux"] += l_style_aux.item() if not isinstance(l_style_aux, float) else l_style_aux
        num_batches += 1

        if num_batches % accum == 0:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            global_step += 1

            # Periodic logging with disentanglement quality metrics
            if global_step % cfg.log_every == 0:
                w_norms = outputs["w_norms"].detach()
                w_norm_str = " ".join(f"{v:.3f}" for v in w_norms.cpu().numpy())
                lr = scheduler.get_last_lr()[0] if hasattr(scheduler, "get_last_lr") else cfg.lr_encoder
                step_in_epoch = batch_idx + 1
                logger.info(
                    "Epoch %d | Step %d/%d (global %d) | "
                    "Loss: %.4f | CE(focal): %.4f | MMD: %.4f | "
                    "StyleCon: %.4f | StyleAux: %.4f | LR: %.2e",
                    epoch + 1, step_in_epoch, len(loader), global_step,
                    total_losses["total"] / num_batches,
                    total_losses["focal"] / num_batches,
                    total_losses["mmd"] / num_batches,
                    total_losses["style_con"] / num_batches,
                    total_losses["style_aux"] / num_batches,
                    lr,
                )
                logger.info("  HyperNet W_norms/class: [%s]", w_norm_str)

    # Final partial accumulation
    if num_batches % accum != 0:
        if use_amp and scaler is not None:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        global_step += 1

    avg_losses = {k: v / max(num_batches, 1) for k, v in total_losses.items()}
    return avg_losses, global_step


@torch.no_grad()
def evaluate(
    model: HyperNetCodeModel,
    loader: DataLoader,
    cfg: HyperNetConfig,
    split_name: str = "Val",
    use_amp: bool = False,
    amp_dtype=None,
) -> Tuple[float, Dict[str, float]]:
    """Evaluate model, return (macro_f1, metrics_dict)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    num_batches = 0
    device = cfg.device

    for batch in loader:
        input_ids = batch["input_ids"].to(device, non_blocking=cfg.non_blocking)
        attention_mask = batch["attention_mask"].to(device, non_blocking=cfg.non_blocking)
        labels = batch["label"].to(device, non_blocking=cfg.non_blocking)

        with autocast(device_type=device, enabled=use_amp, dtype=amp_dtype):
            outputs = model(input_ids, attention_mask)
            logits = outputs["logits"]

        preds = logits.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        total_loss += F.cross_entropy(logits, labels).item()
        num_batches += 1

    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    avg_loss = total_loss / max(num_batches, 1)

    logger.info(
        "%s | Loss: %.4f | Macro-F1: %.4f | Weighted-F1: %.4f",
        split_name, avg_loss, macro_f1, weighted_f1,
    )

    if split_name == "Test":
        report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
        logger.info("\n%s Classification Report:\n%s", split_name, report)

    metrics = {
        "loss": float(avg_loss),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
    }
    return macro_f1, metrics


@torch.no_grad()
def collect_predictions(
    model: HyperNetCodeModel,
    loader: DataLoader,
    cfg: HyperNetConfig,
    use_amp: bool = False,
    amp_dtype=None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Collect all predictions for breakdown analysis."""
    model.eval()
    all_preds, all_labels = [], []
    device = cfg.device

    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        with autocast(device_type=device, enabled=use_amp, dtype=amp_dtype):
            outputs = model(input_ids, attention_mask)

        preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy())

    return np.array(all_preds), np.array(all_labels)


def _build_optimizer_and_scheduler(
    model: HyperNetCodeModel,
    cfg: HyperNetConfig,
    num_training_steps: int,
) -> Tuple[torch.optim.Optimizer, Any]:
    """Dual-LR AdamW: lower LR for encoder, higher LR for task heads."""
    encoder_params = list(model.encoder.parameters())
    encoder_ids = {id(p) for p in encoder_params}
    head_params = [p for p in model.parameters() if id(p) not in encoder_ids]

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": cfg.lr_encoder},
            {"params": head_params, "lr": cfg.lr_heads},
        ],
        weight_decay=cfg.weight_decay,
    )

    warmup_steps = int(cfg.warmup_ratio * num_training_steps)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps,
    )
    return optimizer, scheduler


def _resolve_amp(cfg: HyperNetConfig) -> Tuple[bool, Any]:
    """Resolve AMP precision settings."""
    if cfg.device != "cuda":
        return False, None

    precision = cfg.precision
    if precision == "auto":
        # Default to fp16 if bf16 not available
        if torch.cuda.is_bf16_supported():
            precision = "bf16"
        else:
            precision = "fp16"

    if precision == "bf16":
        return True, torch.bfloat16
    elif precision in ("fp16", "float16"):
        return True, torch.float16
    return False, None


def _save_checkpoint(model: HyperNetCodeModel, tag: str, save_dir: str, task_tag: str, best_f1: float):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f"hypernetcode_{task_tag}_{tag}.pt")
    torch.save({"model_state_dict": model.state_dict(), "best_f1": best_f1}, path)
    logger.info("Saved checkpoint: %s", path)
    return path


def _load_checkpoint(model: HyperNetCodeModel, tag: str, save_dir: str, task_tag: str) -> bool:
    path = os.path.join(save_dir, f"hypernetcode_{task_tag}_{tag}.pt")
    if os.path.exists(path):
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        logger.info("Loaded checkpoint: %s", path)
        return True
    return False


# ============================================================================
# IID Suite
# ============================================================================

def run_iid_suite(
    cfg: HyperNetConfig,
    task: str,
    dataset_id: str = "DaniilOr/CoDET-M4",
) -> Dict[str, Any]:
    """
    Run IID evaluation for a single task (binary or author).
    Handles full train/val/test pipeline including breakdown analysis.
    """
    set_seed(cfg.seed)
    task_tag = f"iid_{task}"
    is_binary = task == "binary"

    logger.info("=" * 70)
    logger.info("[Exp16][IID] HyperNetCode | task=%s | Exp16", task)
    logger.info("GPU=%s | precision=%s | batch=%dx%d | epochs=%d",
                _get_gpu_name(), cfg.precision, cfg.batch_size, cfg.grad_accum_steps, cfg.epochs)
    logger.info("=" * 70)

    # Load data
    train_data, val_data, test_data, num_classes, gen_vocab = load_codet_m4_iid(cfg, task, dataset_id)

    # Preflight
    preflight_check(cfg, num_classes, train_data)

    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
    class_weights = compute_class_weights(train_data["label"], num_classes)
    logger.info("Classes=%d | weights=%s", num_classes, class_weights.numpy().round(3).tolist())

    train_loader, val_loader, test_loader = _make_loaders(train_data, val_data, test_data, tokenizer, cfg)

    # Build model
    model = HyperNetCodeModel(cfg, num_classes).to(cfg.device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("Parameters: %s total, %s trainable", f"{total_params:,}", f"{trainable_params:,}")

    # Loss functions
    use_amp, amp_dtype = _resolve_amp(cfg)
    focal = FocalLoss(gamma=cfg.focal_gamma, weight=class_weights.to(cfg.device))
    supcon = SupConLoss(temperature=cfg.supcon_temperature)
    scaler = GradScaler() if (use_amp and amp_dtype == torch.float16) else None

    # Optimizer and scheduler
    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    total_steps = steps_per_epoch * cfg.epochs
    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg, total_steps)

    logger.info("AMP=%s | dtype=%s | warmup_steps=%d | total_steps=%d",
                use_amp, amp_dtype, int(cfg.warmup_ratio * total_steps), total_steps)

    # Training loop
    best_f1 = 0.0
    global_step = 0

    for epoch in range(cfg.epochs):
        logger.info("\n%s Epoch %d/%d %s", "=" * 40, epoch + 1, cfg.epochs, "=" * 40)
        avg_losses, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            focal_loss=focal,
            supcon_loss=supcon,
            cfg=cfg,
            epoch=epoch,
            global_step_start=global_step,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            is_binary=is_binary,
        )
        logger.info(
            "Epoch %d Train | %s",
            epoch + 1,
            " | ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items()),
        )

        val_f1, val_metrics = evaluate(model, val_loader, cfg, "Val", use_amp, amp_dtype)
        if val_f1 > best_f1:
            best_f1 = val_f1
            _save_checkpoint(model, "best", cfg.save_dir, task_tag, best_f1)
            logger.info("*** New Best Val Macro-F1: %.4f ***", val_f1)
        _save_checkpoint(model, "latest", cfg.save_dir, task_tag, best_f1)

    # Final test evaluation
    logger.info("\n%s\nFINAL TEST EVALUATION [%s]\n%s", "=" * 60, task_tag, "=" * 60)
    if not _load_checkpoint(model, "best", cfg.save_dir, task_tag):
        logger.warning("Best checkpoint not found, using current weights.")

    test_f1, test_metrics = evaluate(model, test_loader, cfg, "Test", use_amp, amp_dtype)
    logger.info("*** Final Test Macro-F1: %.4f | Weighted-F1: %.4f ***",
                test_f1, test_metrics["weighted_f1"])

    # Breakdown analysis
    preds, labels = collect_predictions(model, test_loader, cfg, use_amp, amp_dtype)
    breakdown = _run_breakdown_eval(preds, labels, test_data, task)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "test_f1": float(test_f1),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "best_val_f1": float(best_f1),
        "breakdown": breakdown,
        "task": task,
        "num_classes": int(num_classes),
        "test_per_class": classification_report(
            labels, preds, digits=4, output_dict=True, zero_division=0
        ),
    }


def _run_breakdown_eval(
    preds: np.ndarray,
    labels: np.ndarray,
    test_data: Dataset,
    task: str,
) -> Dict[str, Any]:
    """Compute per-language, per-source, per-generator F1 breakdowns."""
    results: Dict[str, Any] = {}
    n = min(len(preds), len(test_data))
    preds, labels = preds[:n], labels[:n]

    overall_macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    overall_weighted = float(f1_score(labels, preds, average="weighted", zero_division=0))
    results["overall"] = {"macro_f1": overall_macro, "weighted_f1": overall_weighted}
    logger.info("  Breakdown Overall: macro=%.4f  weighted=%.4f", overall_macro, overall_weighted)

    for dim_name in ["language", "source", "generator"]:
        if dim_name not in test_data.column_names:
            continue
        dim_vals = test_data[dim_name][:n]
        unique = sorted(set(dim_vals))
        dim_results = {}
        logger.info("  Breakdown by %s:", dim_name)
        for val in unique:
            mask = np.array([v == val for v in dim_vals])
            if mask.sum() < 10:
                continue
            mf1 = float(f1_score(labels[mask], preds[mask], average="macro", zero_division=0))
            wf1 = float(f1_score(labels[mask], preds[mask], average="weighted", zero_division=0))
            dim_results[val] = {"n": int(mask.sum()), "macro_f1": mf1, "weighted_f1": wf1}
            logger.info("    %12s: n=%6d  macro=%.4f  weighted=%.4f", val, mask.sum(), mf1, wf1)
        results[dim_name] = dim_results

    if task == "author":
        cm = confusion_matrix(labels, preds)
        results["confusion_matrix"] = cm.tolist()
        logger.info("  Confusion matrix (rows=true, cols=pred):\n  %s", cm)

    return results


# ============================================================================
# OOD Suite
# ============================================================================

def _run_single_ood(
    cfg: HyperNetConfig,
    hold_out_field: str,
    hold_out_value: str,
    dataset_id: str = "DaniilOr/CoDET-M4",
) -> Dict[str, Any]:
    """Single leave-one-out OOD evaluation run."""
    set_seed(cfg.seed)
    task_tag = f"ood_{hold_out_field}_{hold_out_value}"

    logger.info("=" * 70)
    logger.info("[OOD] HyperNetCode | %s=%s", hold_out_field, hold_out_value)
    logger.info("=" * 70)

    train_data, val_data, test_data, num_classes, gen_vocab = load_codet_m4_loo(
        cfg, hold_out_field, hold_out_value, dataset_id
    )

    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)
    class_weights = compute_class_weights(train_data["label"], num_classes)

    train_loader, val_loader, test_loader = _make_loaders(train_data, val_data, test_data, tokenizer, cfg)

    model = HyperNetCodeModel(cfg, num_classes).to(cfg.device)
    use_amp, amp_dtype = _resolve_amp(cfg)
    focal = FocalLoss(gamma=cfg.focal_gamma, weight=class_weights.to(cfg.device))
    supcon = SupConLoss(temperature=cfg.supcon_temperature)
    scaler = GradScaler() if (use_amp and amp_dtype == torch.float16) else None

    steps_per_epoch = len(train_loader) // cfg.grad_accum_steps
    total_steps = steps_per_epoch * cfg.epochs
    optimizer, scheduler = _build_optimizer_and_scheduler(model, cfg, total_steps)

    best_f1 = 0.0
    global_step = 0

    for epoch in range(cfg.epochs):
        logger.info("\n%s OOD Epoch %d/%d [%s=%s] %s",
                    "=" * 30, epoch + 1, cfg.epochs, hold_out_field, hold_out_value, "=" * 30)
        avg_losses, global_step = train_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            focal_loss=focal,
            supcon_loss=supcon,
            cfg=cfg,
            epoch=epoch,
            global_step_start=global_step,
            scaler=scaler,
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            is_binary=True,  # OOD is always binary
        )
        logger.info("OOD Epoch %d | %s", epoch + 1,
                    " | ".join(f"{k}: {v:.4f}" for k, v in avg_losses.items()))

        val_f1, _ = evaluate(model, val_loader, cfg, "Val", use_amp, amp_dtype)
        if val_f1 > best_f1:
            best_f1 = val_f1
            _save_checkpoint(model, "best", cfg.save_dir, task_tag, best_f1)

    if not _load_checkpoint(model, "best", cfg.save_dir, task_tag):
        logger.warning("Best checkpoint not found for OOD run, using current weights.")

    test_f1, test_metrics = evaluate(model, test_loader, cfg, "Test", use_amp, amp_dtype)
    preds, labels = collect_predictions(model, test_loader, cfg, use_amp, amp_dtype)
    breakdown = _run_breakdown_eval(preds, labels, test_data, "binary")

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "test_f1": float(test_f1),
        "test_weighted_f1": float(test_metrics["weighted_f1"]),
        "best_val_f1": float(best_f1),
        "hold_out_field": hold_out_field,
        "hold_out_value": hold_out_value,
        "breakdown": breakdown,
    }


def run_ood_suite(
    cfg: HyperNetConfig,
    suite_type: str,
    dataset_id: str = "DaniilOr/CoDET-M4",
) -> Dict[str, Any]:
    """
    Run full OOD leave-one-out suite for one dimension.
    suite_type: "generator" | "language" | "source"
    """
    hold_out_map = {
        "generator": ["codellama", "gpt", "llama3.1", "nxcode", "qwen1.5"],
        "language": ["cpp", "java", "python"],
        "source": ["cf", "gh", "lc"],
    }
    hold_out_values = hold_out_map.get(suite_type, [])
    if not hold_out_values:
        raise ValueError(f"Unknown OOD suite_type: {suite_type}")

    results = {}
    for val in hold_out_values:
        logger.info("\n%s\n[OOD-%s] Holding out %s=%s\n%s",
                    "=" * 70, suite_type.upper(), suite_type, val, "=" * 70)
        try:
            results[val] = _run_single_ood(cfg, suite_type, val, dataset_id)
        except RuntimeError as e:
            logger.error("OOD %s=%s failed: %s", suite_type, val, e)
            results[val] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Summary table
    logger.info("\n%s\nOOD-%s SUMMARY\n%s", "=" * 70, suite_type.upper(), "=" * 70)
    logger.info("| held-out | test_macro_f1 | test_weighted_f1 | best_val_f1 |")
    logger.info("|---|---:|---:|---:|")
    for key, stats in results.items():
        if "error" in stats:
            logger.info("| %s | ERROR | - | - |", key)
        else:
            logger.info("| %s | %.4f | %.4f | %.4f |",
                        key, stats.get("test_f1", 0.0),
                        stats.get("test_weighted_f1", 0.0),
                        stats.get("best_val_f1", 0.0))
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Run mode:
    #   "full"     → all 5 evaluation modes (IID binary + author + 3x OOD)
    #   "iid_only" → IID binary + author only
    #   "ood_only" → 3x OOD only
    #   "single"   → single IID task (set SINGLE_TASK below)
    # ------------------------------------------------------------------
    RUN_MODE = "full"
    SINGLE_TASK = "binary"      # "binary" | "author" (used only if RUN_MODE == "single")
    DATASET_ID = "DaniilOr/CoDET-M4"

    cfg = HyperNetConfig(
        max_train_samples=100_000,
        max_val_samples=20_000,
        max_test_samples=50_000,
        epochs=3,
        batch_size=32,
        grad_accum_steps=2,
        precision="auto",
        auto_h100_profile=True,
        num_workers=2,
        prefetch_factor=2,
        seed=42,
        save_dir="./hypernet_checkpoints",
        log_every=200,
        eval_every=2000,
    )
    cfg = apply_hardware_profile(cfg)
    set_seed(cfg.seed)

    logger.info("\n%s", "=" * 70)
    logger.info("[Exp16] HyperNetCode – Style-Content Disentanglement via HyperNetwork")
    logger.info("Run mode: %s | GPU: %s | Precision: %s", RUN_MODE, _get_gpu_name(), cfg.precision)
    logger.info("Batch: %dx%d | Epochs: %d | Content_z: %d | Style_z: %d",
                cfg.batch_size, cfg.grad_accum_steps, cfg.epochs, cfg.content_dim, cfg.style_dim)
    logger.info("Loss: focal + %.1f*style_aux + %.1f*mmd + %.1f*style_con",
                cfg.w_style_aux, cfg.w_mmd, cfg.w_style_con)
    logger.info("%s\n", "=" * 70)

    all_results: Dict[str, Any] = {}

    FULL_PLAN = [
        ("iid", "binary"),
        ("iid", "author"),
        ("ood", "generator"),
        ("ood", "language"),
        ("ood", "source"),
    ]

    if RUN_MODE == "full":
        plan = FULL_PLAN
    elif RUN_MODE == "iid_only":
        plan = [e for e in FULL_PLAN if e[0] == "iid"]
    elif RUN_MODE == "ood_only":
        plan = [e for e in FULL_PLAN if e[0] == "ood"]
    elif RUN_MODE == "single":
        plan = [("iid", SINGLE_TASK)]
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

    logger.info("Evaluation plan (%d entries):", len(plan))
    for i, (mode, task) in enumerate(plan):
        logger.info("  [%d] %s/%s", i + 1, mode, task)
    logger.info("")

    for mode, task in plan:
        key = f"{mode}_{task}"
        logger.info("\n%s\nSUITE: %s\n%s", "#" * 70, key, "#" * 70)

        try:
            if mode == "iid":
                all_results[key] = run_iid_suite(cfg, task, DATASET_ID)
            elif mode == "ood":
                all_results[key] = run_ood_suite(cfg, task, DATASET_ID)
            else:
                logger.warning("Unknown mode: %s/%s", mode, task)
        except Exception as e:
            logger.error("Suite entry %s failed: %s", key, e, exc_info=True)
            all_results[key] = {"error": str(e)}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("\n%s", "=" * 70)
    logger.info("CoDET-M4 BENCHMARK COMPLETE | %s | HyperNetCode Exp16", ts)
    logger.info("=" * 70)

    logger.info("\n=== IID RESULTS ===")
    for mode, task in plan:
        if mode != "iid":
            continue
        r = all_results.get(f"iid_{task}", {})
        logger.info("  %8s: macro_f1=%.4f  weighted_f1=%.4f  val_f1=%.4f",
                    task,
                    r.get("test_f1", 0.0),
                    r.get("test_weighted_f1", 0.0),
                    r.get("best_val_f1", 0.0))

    for mode, task in plan:
        if mode != "ood":
            continue
        ood_block = all_results.get(f"ood_{task}", {})
        if not isinstance(ood_block, dict):
            continue
        logger.info("\n=== OOD-%s RESULTS ===", task.upper())
        for held, stats in ood_block.items():
            if isinstance(stats, dict) and "test_f1" in stats:
                logger.info("  held_out=%12s: macro_f1=%.4f  weighted_f1=%.4f",
                            held, stats["test_f1"], stats.get("test_weighted_f1", 0.0))

    # Serialize results to JSON (strip non-serializable objects)
    def _make_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        return obj

    safe_results = _make_serializable(all_results)
    suite_json = json.dumps(
        {
            "timestamp": ts,
            "method": "HyperNetCode",
            "experiment": "Exp16",
            "encoder": cfg.encoder_name,
            "content_dim": cfg.content_dim,
            "style_dim": cfg.style_dim,
            "loss_weights": {
                "w_focal": 1.0,
                "w_style_aux": cfg.w_style_aux,
                "w_mmd": cfg.w_mmd,
                "w_style_con": cfg.w_style_con,
            },
            "results": safe_results,
        },
        ensure_ascii=True,
        default=str,
    )
    print(f"\nSUITE_RESULTS_JSON={suite_json}")

    # Paper-ready copy-paste block (headline + per-class + tracker rows)
    try:
        from _paper_table import emit_paper_table
        paper_run_plan = []
        paper_results = {}
        for mode, task in plan:
            key = f"{mode}_{task}"
            r = all_results.get(key, {})
            if mode == "iid" and isinstance(r, dict) and "test_f1" in r:
                paper_run_plan.append(("codet_m4", task))
                paper_results[key] = r
        if paper_run_plan:
            emit_paper_table(
                method_name="HyperNetCode",
                exp_id="exp16",
                run_plan=paper_run_plan,
                results=paper_results,
                timestamp=ts,
                logger=logger,
            )
    except ImportError:
        logger.warning("[_paper_table] helper not found; skipping paper-ready table emission")

    return all_results


if __name__ == "__main__":
    main()
