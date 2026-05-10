"""
run_hier_ntk_portfolio.py -- Hier-NTK × 3 encoders × 3 fractions portfolio runner.

KAGGLE NO-INTERNET VERSION:
    - Models: /kaggle/input/datasets/chiboiz/ai-detection-encoders/models/
    - CoDET-M4: /kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet
    - DroidCollection: /kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/

Usage:
    FS_ENCODER=ModernBERT-base,codebert-base,unixcoder-base
    FS_BENCHMARK=codet_m4  (or: droid_t3, droid_t4)
    FS_SWEEP_FRACS=0.01,0.05,0.20
    FS_SEED=42
"""

from __future__ import annotations
import os

# =============================================================================
# KAGGLE PATHS (no-internet)
# =============================================================================
KAGGLE_MODELS_BASE = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET_PATH = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_DROID_BASE = "/kaggle/input/datasets/chiboiz/droid-collection/DroidCollection"

# =============================================================================
# 1. Bootstrap (pip-install missing deps)
# =============================================================================
import importlib.util
import json
import logging
import math
import os
import random
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Optional, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _ensure_deps():
    required = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("sklearn", "scikit-learn"),
    ]
    missing = [pip for imp, pip in required if importlib.util.find_spec(imp) is None]
    if missing:
        print(f"[bootstrap] installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


_ensure_deps()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoModel, AutoTokenizer

try:
    from torch.amp import autocast as _autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler  # type: ignore
    _NEW_AMP = False

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger('hier_ntk_portfolio')


def autocast(device_type="cuda", enabled=True, dtype=None):
    if _NEW_AMP:
        if dtype is None:
            return _autocast(device_type=device_type, enabled=enabled)
        return _autocast(device_type=device_type, enabled=enabled, dtype=dtype)
    return _autocast(enabled=enabled)


def make_grad_scaler(enabled=True):
    if _NEW_AMP:
        try:
            return GradScaler(device="cuda", enabled=enabled)
        except TypeError:
            return GradScaler(enabled=enabled)
    return GradScaler(enabled=enabled)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# 2. Config + hardware profile
# =============================================================================

@dataclass
class FSConfig:
    benchmark: str = "codet_m4"
    task: str = "author"
    k_shot: int = 32
    train_fraction: float = 0.0
    n_classes: int = 6
    fs_seed: int = 42
    encoder_name: str = "ModernBERT-base"
    max_length: int = 384
    epochs: int = 1
    batch_size: int = 16
    grad_accum_steps: int = 1
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-4
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    warmup_ratio: float = 0.0
    early_stop_patience: int = 5
    use_class_weights: bool = True
    val_size_per_class: int = 64
    test_max_samples: int = -1
    lambda_ntk: float = 0.4
    lambda_hier: float = 0.4
    ntk_proj_dim: int = 128
    num_workers: int = 2
    seed: int = 42
    precision: str = "fp16"
    auto_t4_profile: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True

    def get_encoder_path(self):
        """Return full Kaggle path for encoder."""
        return os.path.join(KAGGLE_MODELS_BASE, self.encoder_name)

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.n_classes = 6 if self.task == "author" else 2
        elif self.benchmark in ("droid", "droid_t3"):
            self.n_classes = 3
            self.task = "t3"
        elif self.benchmark == "droid_t4":
            self.n_classes = 2
            self.task = "t4"


def apply_hardware_profile(cfg: FSConfig) -> FSConfig:
    if cfg.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            mem_gb = 16.0
        gpu = torch.cuda.get_device_name(0)
        # RTX Pro 6000 / Ada Lovelace = 48GB VRAM
        if "RTX Pro 6000" in gpu or "RTX 6000 Ada" in gpu or mem_gb >= 90:
            cfg.batch_size, cfg.precision, cfg.max_length, cfg.num_workers = 128, "bf16", 512, 8
            logger.info(f"[hw] RTX 96GB+ detected -> bs={cfg.batch_size} prec={cfg.precision} seq={cfg.max_length}")
        elif mem_gb >= 40:
            cfg.batch_size, cfg.precision, cfg.max_length, cfg.num_workers = 64, "bf16", 512, 8
            logger.info(f"[hw] RTX Pro 6000 / Ada Lovelace detected -> bs={cfg.batch_size} prec={cfg.precision} seq={cfg.max_length}")
        elif mem_gb >= 70:
            cfg.batch_size, cfg.precision, cfg.max_length, cfg.num_workers = 32, "bf16", 512, 4
        elif mem_gb >= 30:
            cfg.batch_size, cfg.precision, cfg.max_length, cfg.num_workers = 16, "fp16", 512, 4
        elif mem_gb >= 10:
            cfg.batch_size, cfg.precision, cfg.max_length, cfg.num_workers = 16, "fp16", 384, 2
        else:
            cfg.batch_size, cfg.precision, cfg.max_length, cfg.num_workers = 8, "fp16", 256, 2
        logger.info(f"[hw] GPU={gpu} VRAM={mem_gb:.1f}GB -> bs={cfg.batch_size} "
                    f"prec={cfg.precision} seq={cfg.max_length}")
    else:
        cfg.batch_size, cfg.precision, cfg.max_length, cfg.num_workers = 4, "fp32", 256, 0
        logger.info("[hw] CPU mode -> smoke test only")

    fewshot_mode = cfg.k_shot > 0 and cfg.train_fraction <= 0
    cfg.epochs = 1 if fewshot_mode else 3
    cfg.warmup_ratio = 0.0 if fewshot_mode else 0.1
    logger.info(f"[hw] regime={'K-shot' if fewshot_mode else 'fraction'} "
                f"epochs={cfg.epochs} warmup={cfg.warmup_ratio}")
    return cfg


# =============================================================================
# 3. Samplers
# =============================================================================

def kshot_stratified_indices(labels, k_shot, n_classes, seed=42):
    by_class = defaultdict(list)
    for idx, lab in enumerate(labels):
        if lab >= 0:
            by_class[int(lab)].append(idx)
    rng = random.Random(seed)
    chosen, counts = [], {}
    for cls in range(n_classes):
        pool = by_class.get(cls, [])
        n_take = min(k_shot, len(pool))
        if n_take > 0:
            chosen.extend(rng.sample(pool, n_take))
        counts[cls] = n_take
    rng.shuffle(chosen)
    return chosen, counts


def fraction_stratified_indices(labels, fraction, n_classes, seed=42):
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0,1], got {fraction}")
    by_class = defaultdict(list)
    for idx, lab in enumerate(labels):
        if lab >= 0:
            by_class[int(lab)].append(idx)
    rng = random.Random(seed)
    chosen, counts = [], {}
    for cls in range(n_classes):
        pool = by_class.get(cls, [])
        n_take = max(1, int(round(len(pool) * fraction))) if pool else 0
        if n_take > 0:
            chosen.extend(rng.sample(pool, n_take))
        counts[cls] = n_take
    rng.shuffle(chosen)
    return chosen, counts


def build_minival_indices(labels, n_per_class, n_classes, seed=1234):
    by_class = defaultdict(list)
    for idx, lab in enumerate(labels):
        if lab >= 0:
            by_class[int(lab)].append(idx)
    rng = random.Random(seed)
    chosen = []
    for cls in range(n_classes):
        pool = by_class.get(cls, [])
        n_take = min(n_per_class, len(pool))
        if n_take > 0:
            chosen.extend(rng.sample(pool, n_take))
    rng.shuffle(chosen)
    return chosen


# =============================================================================
# 4. Data loader (Kaggle parquet files)
# =============================================================================

def _is_human(t):
    return str(t or "").strip().lower() in {"human", "human_written", "human-generated", "human_generated"}


def _build_author_vocab(train_split):
    names = set()
    for row in train_split:
        if not _is_human(row.get("target", "")):
            m = str(row.get("model", "") or "").strip()
            if m:
                names.add(m)
    return {n: i + 1 for i, n in enumerate(sorted(names))}


def _convert_codet(split, task, vocab):
    def _row(r):
        code = ""
        for f in ("cleaned_code", "code"):
            v = r.get(f, "")
            if isinstance(v, str) and v.strip():
                code = v
                break
        if task == "binary":
            label = 0 if _is_human(r.get("target", "")) else 1
        elif task == "author":
            if _is_human(r.get("target", "")):
                label = 0
            else:
                label = vocab.get(str(r.get("model", "") or "").strip(), -1)
        else:
            raise ValueError(task)
        return {"code": code, "label": label,
                "language": str(r.get("language", "")).strip().lower(),
                "source": str(r.get("source", "")).strip().lower()}

    out = split.map(_row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _convert_droid(split, task):
    def _row(r):
        code = str(r.get("code", "")).strip()
        label = int(r.get("label", -1))
        if task == "t3":
            label = label % 3
        elif task == "t4":
            label = label % 2
        return {"code": code, "label": label,
                "language": str(r.get("language", "")).strip().lower(),
                "source": str(r.get("source", "")).strip().lower()}

    out = split.map(_row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _load_codet_splits(seed):
    logger.info(f"Loading CoDET-M4 from parquet: {KAGGLE_CODET_PATH}")
    ds = load_dataset("parquet", data_files=KAGGLE_CODET_PATH, split="train")
    if "split" in ds.column_names:
        train = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        val = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        test = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
    else:
        s1 = ds.train_test_split(test_size=0.1, seed=seed); test = s1["test"]
        s2 = s1["train"].train_test_split(test_size=1 / 9, seed=seed)
        train, val = s2["train"], s2["test"]
    return train, val, test


def _load_droid_splits(task, seed):
    logger.info(f"Loading DroidCollection from: {KAGGLE_DROID_BASE}")
    # Load all parquet files from the DroidCollection folder
    import glob
    parquet_files = sorted(glob.glob(os.path.join(KAGGLE_DROID_BASE, "data", "*.parquet")))
    logger.info(f"Found {len(parquet_files)} parquet files: {parquet_files}")

    if parquet_files:
        ds = load_dataset("parquet", data_files=parquet_files, split="train")
    else:
        ds = load_dataset(KAGGLE_DROID_BASE, split="train")

    if "split" in ds.column_names:
        train = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        val = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        test = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
    else:
        s1 = ds.train_test_split(test_size=0.1, seed=seed); test = s1["test"]
        s2 = s1["train"].train_test_split(test_size=1 / 9, seed=seed)
        train, val = s2["train"], s2["test"]
    return train, val, test


class _FSDataset(TorchDataset):
    def __init__(self, hf_ds, tokenizer, max_length):
        self.ds = hf_ds; self.tok = tokenizer; self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        enc = self.tok(row["code"], max_length=self.max_length, truncation=True,
                       padding="max_length", return_tensors="pt")
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "label": int(row["label"]),
                "language": row.get("language", ""),
                "source": row.get("source", "")}


def _collate(batch):
    return {"input_ids": torch.stack([b["input_ids"] for b in batch]),
            "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
            "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
            "languages": [b["language"] for b in batch],
            "sources": [b["source"] for b in batch]}


def build_loaders(cfg: FSConfig):
    set_seed(cfg.seed)

    if cfg.benchmark.startswith("droid"):
        train_raw, val_raw, test_raw = _load_droid_splits(cfg.task, cfg.seed)
        vocab = {}
    else:
        train_raw, val_raw, test_raw = _load_codet_splits(cfg.seed)
        vocab = _build_author_vocab(train_raw) if cfg.task == "author" else {}

    if cfg.task == "author":
        logger.info(f"Author vocab ({len(vocab)}): {sorted(vocab.keys())}")

    if cfg.benchmark.startswith("droid"):
        train_ds = _convert_droid(train_raw, cfg.task)
        val_ds = _convert_droid(val_raw, cfg.task)
        test_ds = _convert_droid(test_raw, cfg.task)
    else:
        train_ds = _convert_codet(train_raw, cfg.task, vocab)
        val_ds = _convert_codet(val_raw, cfg.task, vocab)
        test_ds = _convert_codet(test_raw, cfg.task, vocab)

    fewshot_mode = cfg.k_shot > 0 and cfg.train_fraction <= 0
    if fewshot_mode:
        idxs, counts = kshot_stratified_indices(list(train_ds["label"]), cfg.k_shot,
                                                 cfg.n_classes, cfg.fs_seed)
    else:
        idxs, counts = fraction_stratified_indices(list(train_ds["label"]),
                                                    cfg.train_fraction, cfg.n_classes, cfg.fs_seed)
    train_ds = train_ds.select(idxs) if idxs else train_ds.select([])
    logger.info(f"[train] {('K=' + str(cfg.k_shot)) if fewshot_mode else ('frac=' + str(cfg.train_fraction))}"
                f" -> per-class={counts} total={sum(counts.values())}")

    val_idxs = build_minival_indices(list(val_ds["label"]), cfg.val_size_per_class,
                                      cfg.n_classes, seed=cfg.fs_seed + 1000)
    val_ds = val_ds.select(val_idxs) if val_idxs else val_ds
    logger.info(f"[val] size={len(val_ds)}  [test] size={len(test_ds)} (full)")

    # Load tokenizer from LOCAL Kaggle path
    encoder_path = cfg.get_encoder_path()
    logger.info(f"Loading tokenizer from: {encoder_path}")
    tokenizer = AutoTokenizer.from_pretrained(encoder_path, local_files_only=True)

    def _ld(ds, shuffle):
        return DataLoader(_FSDataset(ds, tokenizer, cfg.max_length),
                          batch_size=cfg.batch_size, shuffle=shuffle,
                          num_workers=cfg.num_workers, collate_fn=_collate,
                          pin_memory=cfg.pin_memory)

    return _ld(train_ds, True), _ld(val_ds, False), _ld(test_ds, False), counts, vocab


# =============================================================================
# 5. Model + Loss
# =============================================================================

def _mean_pool(hidden, mask):
    m = mask.unsqueeze(-1).float()
    return (hidden * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)


class FSClassifier(nn.Module):
    def __init__(self, cfg: FSConfig):
        super().__init__()
        self.cfg = cfg
        encoder_path = cfg.get_encoder_path()
        logger.info(f"Loading model from: {encoder_path}")
        self.encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(h, cfg.n_classes)
        self.ntk_proj = nn.Sequential(
            nn.Linear(h, cfg.ntk_proj_dim), nn.GELU(),
            nn.Linear(cfg.ntk_proj_dim, cfg.ntk_proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = _mean_pool(out.last_hidden_state, attention_mask)
        return {"logits": self.classifier(self.dropout(emb)),
                "embedding": emb,
                "ntk_proj": F.normalize(self.ntk_proj(emb), dim=-1)}

    def param_groups(self):
        enc = list(self.encoder.parameters())
        head = list(self.classifier.parameters()) + list(self.ntk_proj.parameters())
        return [{"params": enc, "lr": self.cfg.lr_encoder, "weight_decay": self.cfg.weight_decay},
                {"params": head, "lr": self.cfg.lr_heads, "weight_decay": self.cfg.weight_decay}]


HIER_FAMILY = {0: 0, 1: 1, 2: 2, 3: 3, 4: 1, 5: 4}
HIER_NUM_FAMILIES = 5


def hier_ntk_loss(outputs, labels, lambda_hier=0.4, lambda_ntk=0.4,
                  margin=0.3, class_weights=None):
    """CE + HierTree (family pull/push) + NTK target-kernel alignment."""
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    z = outputs["ntk_proj"]
    B = z.size(0)

    # HierTree family pull/push
    fam = torch.tensor([HIER_FAMILY.get(int(y), int(y)) for y in labels.cpu().tolist()],
                        device=z.device)
    same = (fam.unsqueeze(0) == fam.unsqueeze(1)).float()
    eye = torch.eye(B, device=z.device)
    same = same - eye
    diff = 1.0 - same - eye
    dist = torch.cdist(z, z, p=2)
    pull = (same * dist.pow(2)).sum() / same.sum().clamp(min=1.0)
    push = (diff * F.relu(margin - dist).pow(2)).sum() / diff.sum().clamp(min=1.0)
    hier_l = pull + push

    # NTK target-kernel alignment
    K = z @ z.t()
    Y = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    H = torch.eye(B, device=z.device) - torch.full((B, B), 1.0 / B, device=z.device)
    align = ((H @ K @ H - H @ Y @ H) ** 2).mean()

    total = ce + lambda_hier * hier_l + lambda_ntk * align
    return {"total": total, "ce": ce, "hier": hier_l, "ntk_align": align}


# =============================================================================
# 6. Trainer
# =============================================================================

@dataclass
class FSResults:
    test_macro_f1: float
    test_weighted_f1: float
    test_accuracy: float
    val_macro_f1: float
    per_class_f1: Dict
    per_lang_f1: Dict
    per_source_f1: Dict
    train_steps: int
    wall_time_s: float


def _class_weights(loader, n_classes):
    counts = np.zeros(n_classes, dtype=np.float64)
    for batch in loader:
        for lab in batch["labels"].tolist():
            counts[lab] += 1
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    w = w / w.sum() * n_classes
    return torch.tensor(w, dtype=torch.float32)


@torch.no_grad()
def _evaluate(model, loader, cfg, device):
    model.eval()
    all_p, all_l, all_lang, all_src = [], [], [], []
    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"]
        with autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                      enabled=device.type == "cuda", dtype=amp_dtype):
            out = model(ids, mask)
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_p.extend(preds.tolist()); all_l.extend(labels.tolist())
        all_lang.extend(batch.get("languages", [""] * len(preds)))
        all_src.extend(batch.get("sources", [""] * len(preds)))
    return {"macro_f1": float(f1_score(all_l, all_p, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(all_l, all_p, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(all_l, all_p)),
            "preds": all_p, "labels": all_l, "languages": all_lang, "sources": all_src}


def _per_subgroup(preds, labels, groups):
    out = {}
    for g in sorted(set(groups)):
        if not g:
            continue
        idx = [i for i, x in enumerate(groups) if x == g]
        if len(idx) < 5:
            continue
        out[g] = float(f1_score([labels[i] for i in idx], [preds[i] for i in idx],
                                 average="macro", zero_division=0))
    return out


def _per_class(preds, labels, n):
    rep = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {f"class_{i}": float(rep.get(str(i), {}).get("f1-score", 0.0)) for i in range(n)}


def train_fewshot(cfg, model, train_loader, val_loader, test_loader, method_name="FS"):
    device = torch.device(cfg.device); model = model.to(device)
    class_weights = _class_weights(train_loader, cfg.n_classes).to(device) if cfg.use_class_weights else None
    if class_weights is not None:
        logger.info(f"[trainer] class_weights={class_weights.tolist()}")

    optimizer = torch.optim.AdamW(model.param_groups())
    scaler = make_grad_scaler(enabled=(cfg.precision == "fp16" and device.type == "cuda"))
    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16

    total_steps = max(1, len(train_loader) * cfg.epochs)
    eval_every = max(5, total_steps // 8)
    scheduler = None
    if cfg.warmup_ratio > 0 and total_steps > 20:
        try:
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, int(total_steps * cfg.warmup_ratio), total_steps)
            logger.info(f"[trainer] cosine LR, warmup {int(total_steps * cfg.warmup_ratio)}")
        except ImportError:
            scheduler = None

    logger.info(f"[trainer] {method_name} steps={total_steps} eval_every={eval_every} epochs={cfg.epochs}")

    best_val = -1.0; best_state = None; plateau = 0; step = 0
    t0 = time.time()
    for _ in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            step += 1
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                          enabled=device.type == "cuda", dtype=amp_dtype):
                out = model(ids, mask)
                losses = hier_ntk_loss(out, labels,
                                       lambda_hier=cfg.lambda_hier,
                                       lambda_ntk=cfg.lambda_ntk,
                                       margin=0.3,
                                       class_weights=class_weights)
                loss = losses["total"]
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer); scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if step % 10 == 0:
                logger.info(f"[step {step}/{total_steps}] " +
                            " ".join(f"{k}={v.item():.4f}" for k, v in losses.items()))
            if step % eval_every == 0 or step == total_steps:
                v = _evaluate(model, val_loader, cfg, device)
                logger.info(f"[val @ step {step}] macro_f1={v['macro_f1']:.4f} acc={v['accuracy']:.4f}")
                if v["macro_f1"] > best_val + 1e-4:
                    best_val = v["macro_f1"]
                    best_state = {k: vv.detach().cpu().clone() for k, vv in model.state_dict().items()}
                    plateau = 0
                else:
                    plateau += 1
                    if plateau >= cfg.early_stop_patience:
                        logger.info(f"[early-stop] plateau={plateau}; stop")
                        break
                model.train()
        else:
            continue
        break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"[trainer] restored best val={best_val:.4f}")

    logger.info("[trainer] FULL test eval...")
    t = _evaluate(model, test_loader, cfg, device)
    gap = best_val - t["macro_f1"]
    logger.info(f"[trainer] FINAL test_macro_f1={t['macro_f1']:.4f} weighted={t['weighted_f1']:.4f} "
                f"acc={t['accuracy']:.4f} val={best_val:.4f} gap={gap:+.4f}")
    return FSResults(t["macro_f1"], t["weighted_f1"], t["accuracy"], best_val,
                     _per_class(t["preds"], t["labels"], cfg.n_classes),
                     _per_subgroup(t["preds"], t["labels"], t["languages"]),
                     _per_subgroup(t["preds"], t["labels"], t["sources"]),
                     step, time.time() - t0)


# =============================================================================
# 7. Output + sweep
# =============================================================================

def _jsonify(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(x) for x in obj]
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:
            pass
    if isinstance(obj, (int, float, bool, str)) or obj is None:
        return obj
    return str(obj)


def emit_json(method_name, exp_id, cfg: FSConfig, results: Dict):
    fewshot_mode = cfg.k_shot > 0 and cfg.train_fraction <= 0
    regime_label = f"K={cfg.k_shot}" if fewshot_mode else f"frac={cfg.train_fraction:.4f}"
    file_label = f"K{cfg.k_shot}" if fewshot_mode else f"frac{cfg.train_fraction:.4f}".rstrip("0").rstrip(".")

    payload = {"method": method_name, "exp_id": exp_id, "regime": regime_label,
               "timestamp": datetime.utcnow().isoformat() + "Z",
               "config": {"benchmark": cfg.benchmark, "task": cfg.task, "k_shot": cfg.k_shot,
                          "train_fraction": cfg.train_fraction, "n_classes": cfg.n_classes,
                          "fs_seed": cfg.fs_seed, "encoder": cfg.encoder_name,
                          "encoder_path": cfg.get_encoder_path(),
                          "epochs": cfg.epochs, "batch_size": cfg.batch_size,
                          "max_length": cfg.max_length, "precision": cfg.precision,
                          "lr_encoder": cfg.lr_encoder, "lr_heads": cfg.lr_heads,
                          "lambda_ntk": cfg.lambda_ntk, "lambda_hier": cfg.lambda_hier},
               "results": _jsonify(results)}
    candidate_dirs = []
    if os.name == "posix" and os.path.isdir("/kaggle/working"):
        candidate_dirs.append("/kaggle/working/results")
    candidate_dirs.append("./results")
    for d in candidate_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{exp_id}_{cfg.encoder_name}_{file_label}_seed{cfg.fs_seed}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            logger.info(f"[json] -> {path}")
            return path
        except (OSError, PermissionError) as e:
            logger.warning(f"[json] {d}: {e}")
    return None


def cleanup():
    import gc; gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def print_summary_table(all_results):
    print(f"\n{'=' * 90}")
    print(f"{'Encoder':<30} {'Frac':>6} {'Test F1':>10} {'Val F1':>10} {'Gap':>10} {'Wall(s)':>10}")
    print("-" * 90)
    for enc, frac, test_f1, val_f1, gap, wall in all_results:
        print(f"{enc:<30} {frac:>6.2%} {test_f1:>10.4f} {val_f1:>10.4f} {gap:>+10.4f} {wall:>10.0f}")
    print("-" * 90)


# =============================================================================
# 8. Main portfolio runner
# =============================================================================

def parse_encoders():
    raw = os.environ.get("FS_ENCODER", "ModernBERT-base,codebert-base,unixcoder-base")
    return [e.strip() for e in raw.split(",") if e.strip()]


def parse_fractions():
    raw = os.environ.get("FS_SWEEP_FRACS", "0.01,0.05,0.20")
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main():
    encoders = parse_encoders()
    fractions = parse_fractions()
    benchmark = os.environ.get("FS_BENCHMARK", "codet_m4")
    base_seed = int(os.environ.get("FS_SEED", "42"))
    lambda_ntk = float(os.environ.get("FS_LAMBDA_NTK", "0.4"))
    lambda_hier = float(os.environ.get("FS_LAMBDA_HIER", "0.4"))

    print(f"\n{'=' * 70}")
    print(f"Hier-NTK Portfolio Runner (Kaggle no-internet)")
    print(f"  Encoders: {encoders}")
    print(f"  Fractions: {fractions}")
    print(f"  Benchmark: {benchmark}")
    print(f"  Seed: {base_seed}")
    print(f"  lambda_ntk={lambda_ntk}, lambda_hier={lambda_hier}")
    print(f"  Models base: {KAGGLE_MODELS_BASE}")
    print(f"  CoDET path: {KAGGLE_CODET_PATH}")
    print(f"  Droid path: {KAGGLE_DROID_BASE}")
    print(f"{'=' * 70}\n")

    all_results = []
    total_experiments = len(encoders) * len(fractions)

    for enc_idx, encoder in enumerate(encoders):
        for frac_idx, frac in enumerate(fractions):
            exp_num = enc_idx * len(fractions) + frac_idx + 1
            print(f"\n{'=' * 60}")
            print(f"[{exp_num}/{total_experiments}] {encoder} @ frac={frac:.2%}")
            print(f"{'=' * 60}")

            cfg = FSConfig(
                benchmark=benchmark,
                task="author",
                k_shot=0,
                train_fraction=frac,
                fs_seed=base_seed,
                encoder_name=encoder,
                lambda_ntk=lambda_ntk,
                lambda_hier=lambda_hier,
                lr_encoder=2e-5,
                lr_heads=1e-4,
            )
            cfg = apply_hardware_profile(cfg)

            train_l, val_l, test_l, counts, vocab = build_loaders(cfg)
            model = FSClassifier(cfg)

            results = train_fewshot(cfg, model, train_l, val_l, test_l,
                                   method_name=f"FS-Hier-NTK")

            gap = results.val_macro_f1 - results.test_macro_f1
            payload = {
                "test_macro_f1": results.test_macro_f1,
                "test_weighted_f1": results.test_weighted_f1,
                "test_accuracy": results.test_accuracy,
                "val_macro_f1": results.val_macro_f1,
                "val_test_gap": gap,
                "train_steps": results.train_steps,
                "wall_time_s": f"{results.wall_time_s:.1f}",
                "per_class_f1": results.per_class_f1,
                "per_lang_f1": results.per_lang_f1,
                "per_source_f1": results.per_source_f1,
            }

            emit_json("FS-Hier-NTK", f"hier_ntk_p{exp_num}", cfg, payload)
            all_results.append((encoder, frac, results.test_macro_f1,
                               results.val_macro_f1, gap, results.wall_time_s))

            cleanup()

    print_summary_table(all_results)

    # Save aggregate results
    agg_path = os.path.join("./results", f"hier_ntk_portfolio_seed{base_seed}.json")
    try:
        os.makedirs("./results", exist_ok=True)
        with open(agg_path, "w") as f:
            json.dump({
                "method": "FS-Hier-NTK",
                "encoders": encoders,
                "fractions": fractions,
                "benchmark": benchmark,
                "seed": base_seed,
                "results": [{"encoder": r[0], "frac": r[1], "test_f1": r[2],
                            "val_f1": r[3], "gap": r[4], "wall_s": r[5]} for r in all_results]
            }, f, indent=2)
        print(f"\nAggregate saved: {agg_path}")
    except Exception as e:
        print(f"Failed to save aggregate: {e}")


if __name__ == "__main__":
    main()
