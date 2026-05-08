"""
CoDET-M4 few-shot data loader.

Loads `DaniilOr/CoDET-M4` from HuggingFace, builds the 6-class author label
vocabulary from the FULL train pool (so K-shot sampling sees the same vocab
as full-data baselines), then samples K examples per class deterministically.

Test split is the FULL paper-comparable split. Mini-val is a held-out
n-per-class subset of the original val pool (for early stopping).
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader, Dataset as TorchDataset
from transformers import AutoTokenizer

from _common_fs import FSConfig, logger, set_seed
from _fewshot_sampler import (
    build_minival_indices,
    kshot_stratified_subset,
    report_kshot_distribution,
)


# ---------------------------------------------------------------------------
# Label utilities — mirror Exp_Climb/_data_codet.py without importing it
# ---------------------------------------------------------------------------

def _normalize_target(value: object) -> str:
    return str(value or "").strip().lower()


def _is_human_target(target: str) -> bool:
    return target in {"human", "human_written", "human-generated", "human_generated"}


def _build_author_vocab(train_split: Dataset) -> Dict[str, int]:
    """{model_name -> 1..K}, label 0 reserved for human."""
    model_names = set()
    for row in train_split:
        target = _normalize_target(row.get("target", ""))
        model_name = str(row.get("model", "") or "").strip()
        if not _is_human_target(target) and model_name:
            model_names.add(model_name)
    return {name: idx + 1 for idx, name in enumerate(sorted(model_names))}


def _map_binary_label(row: Dict[str, object]) -> int:
    return 0 if _is_human_target(_normalize_target(row.get("target", ""))) else 1


def _map_author_label(row: Dict[str, object], vocab: Dict[str, int]) -> int:
    if _is_human_target(_normalize_target(row.get("target", ""))):
        return 0
    return vocab.get(str(row.get("model", "") or "").strip(), -1)


def _extract_code(row: Dict[str, object]) -> str:
    for f in ("cleaned_code", "code"):
        v = row.get(f, "")
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _convert(split: Dataset, task: str, vocab: Dict[str, int]) -> Dataset:
    def _row(r):
        code = _extract_code(r)
        if task == "binary":
            label = _map_binary_label(r)
        elif task == "author":
            label = _map_author_label(r, vocab)
        else:
            raise ValueError(task)
        return {
            "code": code,
            "label": label,
            "language": str(r.get("language", "")).strip().lower(),
            "source": str(r.get("source", "")).strip().lower(),
        }

    out = split.map(_row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


# ---------------------------------------------------------------------------
# Splits loader
# ---------------------------------------------------------------------------

def _load_raw_splits(seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    logger.info("Loading dataset: DaniilOr/CoDET-M4")
    ds = load_dataset("DaniilOr/CoDET-M4", split="train")
    if "split" in ds.column_names:
        train = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        val = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        test = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
    else:
        s1 = ds.train_test_split(test_size=0.1, seed=seed)
        test = s1["test"]
        s2 = s1["train"].train_test_split(test_size=1 / 9, seed=seed)
        train, val = s2["train"], s2["test"]
    if min(len(train), len(val), len(test)) == 0:
        raise RuntimeError("CoDET-M4 split is empty")
    return train, val, test


# ---------------------------------------------------------------------------
# Tokenization Dataset
# ---------------------------------------------------------------------------

class CoDETFSDataset(TorchDataset):
    """Tokenizes code on-the-fly. Few-shot sets are tiny so we don't pre-tokenize."""

    def __init__(self, hf_ds: Dataset, tokenizer, max_length: int):
        self.ds = hf_ds
        self.tok = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        row = self.ds[idx]
        enc = self.tok(
            row["code"],
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "label": int(row["label"]),
            "language": row.get("language", ""),
            "source": row.get("source", ""),
        }


def collate_fn(batch):
    return {
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "labels": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "languages": [b["language"] for b in batch],
        "sources": [b["source"] for b in batch],
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class FSDataBundle:
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader
    n_classes: int
    train_per_class: Dict[int, int]
    val_per_class: Dict[int, int]
    test_size: int
    author_vocab: Dict[str, int]


def build_codet_fs_loaders(cfg: FSConfig) -> FSDataBundle:
    """End-to-end pipeline: load -> label -> K-shot sample -> tokenize -> DataLoader."""
    set_seed(cfg.seed)

    train_raw, val_raw, test_raw = _load_raw_splits(cfg.seed)
    vocab = _build_author_vocab(train_raw) if cfg.task == "author" else {}
    if cfg.task == "author":
        logger.info(f"Author vocab ({len(vocab)} generators): {sorted(vocab.keys())}")

    train_ds = _convert(train_raw, cfg.task, vocab)
    val_ds = _convert(val_raw, cfg.task, vocab)
    test_ds = _convert(test_raw, cfg.task, vocab)

    # K-shot sample TRAIN
    train_ds, train_counts = kshot_stratified_subset(
        train_ds, k_shot=cfg.k_shot, n_classes=cfg.n_classes, seed=cfg.fs_seed
    )
    logger.info(f"[K-shot train] {report_kshot_distribution(train_counts, cfg.k_shot)}")

    # Mini-val for early stopping (separate seed)
    val_indices = build_minival_indices(
        list(val_ds["label"]),
        n_per_class=cfg.val_size_per_class,
        n_classes=cfg.n_classes,
        seed=cfg.fs_seed + 1000,
    )
    val_ds = val_ds.select(val_indices) if val_indices else val_ds
    val_counts = dict(Counter(val_ds["label"]))
    logger.info(f"[mini-val] size={len(val_ds)} per-class={val_counts}")

    # Test = FULL paper-comparable
    if cfg.test_max_samples > 0:
        test_ds = test_ds.select(range(min(cfg.test_max_samples, len(test_ds))))
    logger.info(f"[test] size={len(test_ds)} (full)")

    # Tokenizer + loaders
    tokenizer = AutoTokenizer.from_pretrained(cfg.encoder_name)

    def _make_loader(ds, shuffle):
        return DataLoader(
            CoDETFSDataset(ds, tokenizer, cfg.max_length),
            batch_size=cfg.batch_size,
            shuffle=shuffle,
            num_workers=cfg.num_workers,
            collate_fn=collate_fn,
            pin_memory=cfg.pin_memory,
        )

    return FSDataBundle(
        train_loader=_make_loader(train_ds, shuffle=True),
        val_loader=_make_loader(val_ds, shuffle=False),
        test_loader=_make_loader(test_ds, shuffle=False),
        n_classes=cfg.n_classes,
        train_per_class=train_counts,
        val_per_class=val_counts,
        test_size=len(test_ds),
        author_vocab=vocab,
    )
