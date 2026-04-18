"""
CoDET-M4 data loading + IID/OOD evaluation suite.

Covers all paper tables for CoDET-M4 (Orel, Azizov & Nakov, ACL Findings 2025):
  - Table 2: Binary IID Macro-F1
  - Table 3: Binary per-language breakdown (cpp / java / python)
  - Table 4: Binary per-source breakdown (cf / gh / lc)
  - Table 7: Author 6-class IID
  - Table 8 (proxy): OOD Generator Leave-One-Out (5 generators)
  - Table 9 (proxy): OOD Source Leave-One-Out (3 sources)
  - Table 10/12 (proxy): OOD Language Leave-One-Out (3 languages)

CLIMB protocol: `max_train_samples=100_000` (~20% of ~500K train) +
`max_test_samples=-1` (FULL test split, paper-comparable).
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import Dataset, load_dataset
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from _common import PreflightError, SpectralConfig, apply_hardware_profile, get_gpu_name, logger, set_seed
from _features import (
    STRUCTURAL_FEATURE_DIM,
    ast_parser_available,
    extract_ast_sequence,
    extract_structural_features,
)
from _model import AICDDataset, SpectralCode
from _trainer import Trainer, compute_class_weights

# Expose loss-fn type alias for exp files
from _trainer import LossFn  # noqa: F401


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class CoDETM4Config:
    dataset_id: str = "DaniilOr/CoDET-M4"
    code_field_priority: Tuple[str, ...] = ("cleaned_code", "code")
    split_field: str = "split"

    task: str = "binary"  # binary | author

    # CLIMB: small train, FULL test
    max_train_samples: int = 100_000
    max_val_samples: int = 20_000
    max_test_samples: int = -1

    eval_breakdown: bool = True
    seed: int = 42
    save_root: str = "./codet_m4_checkpoints"


# ===========================================================================
# Data utilities
# ===========================================================================

def _sample_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), max_samples)
    return dataset.select(indices)


def _extract_code(row: Dict[str, object], code_fields: Tuple[str, ...]) -> str:
    for field_name in code_fields:
        value = row.get(field_name, "")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _normalize_target(value: object) -> str:
    return str(value or "").strip().lower()


def _is_human_target(target: str) -> bool:
    return target in {"human", "human_written", "human-generated", "human_generated"}


def _build_author_vocab(train_split: Dataset) -> Dict[str, int]:
    model_names = set()
    for row in train_split:
        target = _normalize_target(row.get("target", ""))
        model_name = str(row.get("model", "") or "").strip()
        if not _is_human_target(target) and model_name:
            model_names.add(model_name)
    return {name: idx + 1 for idx, name in enumerate(sorted(model_names))}


def _map_binary_label(row: Dict[str, object]) -> int:
    return 0 if _is_human_target(_normalize_target(row.get("target", ""))) else 1


def _map_author_label(row: Dict[str, object], author_vocab: Dict[str, int]) -> int:
    if _is_human_target(_normalize_target(row.get("target", ""))):
        return 0
    return author_vocab.get(str(row.get("model", "") or "").strip(), -1)


def _convert_split(split_ds: Dataset, cfg: CoDETM4Config, author_vocab: Dict[str, int]) -> Dataset:
    def _convert_row(row):
        code = _extract_code(row, cfg.code_field_priority)
        if cfg.task == "binary":
            label = _map_binary_label(row)
        elif cfg.task == "author":
            label = _map_author_label(row, author_vocab)
        else:
            raise ValueError(f"Unsupported CoDET task: {cfg.task}")
        target = _normalize_target(row.get("target", ""))
        return {
            "code": code,
            "label": label,
            "language": str(row.get("language", "")).strip().lower(),
            "source": str(row.get("source", "")).strip().lower(),
            "generator": "human" if _is_human_target(target) else str(row.get("model", "") or "").strip().lower(),
        }

    converted = split_ds.map(_convert_row, remove_columns=split_ds.column_names)
    return converted.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _quick_code_stats(dataset: Dataset, sample_size: int = 512) -> Dict[str, float]:
    n = min(len(dataset), sample_size)
    if n == 0:
        return {"samples": 0, "avg_chars": 0.0, "avg_lines": 0.0, "empty_ratio": 1.0}
    sample = dataset.select(range(n))
    codes = sample["code"]
    return {
        "samples": n,
        "avg_chars": float(np.mean([len(c) for c in codes])),
        "avg_lines": float(np.mean([c.count("\n") + 1 for c in codes])),
        "empty_ratio": float(sum(1 for c in codes if len(c.strip()) == 0) / n),
    }


# ===========================================================================
# Raw split loading
# ===========================================================================

def _load_raw_splits(cfg: CoDETM4Config):
    logger.info(f"Loading dataset: {cfg.dataset_id}")
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
    author_vocab = _build_author_vocab(train_raw) if cfg.task == "author" else {}
    if cfg.task == "author":
        logger.info("Author vocab (%d generators): %s", len(author_vocab), sorted(author_vocab.keys()))
    train_data = _convert_split(train_raw, cfg, author_vocab)
    val_data = _convert_split(val_raw, cfg, author_vocab)
    test_data = _convert_split(test_raw, cfg, author_vocab)
    train_data = _sample_dataset(train_data, cfg.max_train_samples, cfg.seed)
    val_data = _sample_dataset(val_data, cfg.max_val_samples, cfg.seed + 1)
    test_data = _sample_dataset(test_data, cfg.max_test_samples, cfg.seed + 2)
    num_classes = len(set(train_data["label"]))
    logger.info(
        "IID sizes | train=%d val=%d test=%d | classes=%d",
        len(train_data), len(val_data), len(test_data), num_classes,
    )
    return train_data, val_data, test_data, num_classes


def load_codet_m4_loo(cfg: CoDETM4Config, hold_out_field: str, hold_out_value: str):
    """Leave-One-Out split: drop `hold_out_value` from train+val, keep only it in test."""
    train_raw, val_raw, test_raw = _load_raw_splits(cfg)
    field_map = {"generator": "model", "language": "language", "source": "source"}
    raw_field = field_map.get(hold_out_field, hold_out_field)

    def _matches(row):
        return str(row.get(raw_field, "") or "").strip().lower() == hold_out_value.lower()

    train_in = train_raw.filter(lambda x: not _matches(x))
    val_in = val_raw.filter(lambda x: not _matches(x))
    test_ood = test_raw.filter(_matches)
    if len(train_in) == 0 or len(val_in) == 0 or len(test_ood) == 0:
        raise RuntimeError(
            f"LOO split empty: hold_out={hold_out_field}={hold_out_value} | "
            f"train_in={len(train_in)} val_in={len(val_in)} test_ood={len(test_ood)}"
        )
    author_vocab = _build_author_vocab(train_in) if cfg.task == "author" else {}
    train_data = _convert_split(train_in, cfg, author_vocab)
    val_data = _convert_split(val_in, cfg, author_vocab)
    test_data = _convert_split(test_ood, cfg, author_vocab)
    train_data = _sample_dataset(train_data, cfg.max_train_samples, cfg.seed)
    val_data = _sample_dataset(val_data, cfg.max_val_samples, cfg.seed + 1)
    # Test NOT subsampled (LOO already small relative to full test)
    num_classes = len(set(train_data["label"]))
    logger.info(
        "LOO[%s!=%s] | train=%d val=%d test_ood=%d | classes=%d",
        hold_out_field, hold_out_value, len(train_data), len(val_data), len(test_data), num_classes,
    )
    return train_data, val_data, test_data, num_classes


# ===========================================================================
# Breakdown evaluation
# ===========================================================================

@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: str):
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ast_seq = batch["ast_seq"].to(device)
        struct_feat = batch["struct_feat"].to(device)
        out = model(input_ids, attention_mask, ast_seq, struct_feat)
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy())
    return np.array(all_preds), np.array(all_labels)


def run_breakdown_eval(preds, labels, test_data, task: str) -> Dict[str, Any]:
    results: Dict[str, Any] = {}
    n = min(len(preds), len(test_data))
    preds, labels = preds[:n], labels[:n]
    overall_macro = float(f1_score(labels, preds, average="macro"))
    overall_weighted = float(f1_score(labels, preds, average="weighted"))
    results["overall"] = {"macro_f1": overall_macro, "weighted_f1": overall_weighted}
    logger.info(f"  Overall: macro={overall_macro:.4f}  weighted={overall_weighted:.4f}")

    for dim_name in ("language", "source", "generator"):
        dim_vals = test_data[dim_name][:n]
        unique = sorted(set(dim_vals))
        dim_results: Dict[str, Dict[str, float]] = {}
        logger.info(f"  Breakdown by {dim_name}:")
        for val in unique:
            mask = np.array([v == val for v in dim_vals])
            if mask.sum() < 10:
                continue
            mf1 = float(f1_score(labels[mask], preds[mask], average="macro"))
            wf1 = float(f1_score(labels[mask], preds[mask], average="weighted"))
            dim_results[val] = {"n": int(mask.sum()), "macro_f1": mf1, "weighted_f1": wf1}
            logger.info(f"    {val:>12s}: n={int(mask.sum()):>6d}  macro={mf1:.4f}  weighted={wf1:.4f}")
        results[dim_name] = dim_results

    if task == "author":
        cm = confusion_matrix(labels, preds)
        results["confusion_matrix"] = cm.tolist()
        logger.info("  Confusion matrix (rows=true, cols=pred):")
        logger.info(f"  {cm}")
    return results


# ===========================================================================
# Preflight
# ===========================================================================

def preflight(cfg: CoDETM4Config, exp_cfg: SpectralConfig) -> Dict[str, object]:
    logger.info("\n" + "=" * 70)
    logger.info(f"PREFLIGHT | dataset={cfg.dataset_id} | task={cfg.task}")
    logger.info("=" * 70)
    if exp_cfg.require_tree_sitter and not ast_parser_available():
        raise PreflightError(
            "tree-sitter-languages unavailable. Install it or set require_tree_sitter=False."
        )
    train_data, val_data, test_data, num_classes = load_codet_m4_data(cfg)
    if num_classes < 2:
        raise PreflightError(f"Invalid class count: {num_classes}")
    label_counts = Counter(train_data["label"])
    code_stats = _quick_code_stats(train_data)
    if code_stats["empty_ratio"] > 0.05:
        raise PreflightError(f"Empty ratio too high: {code_stats['empty_ratio']:.3f}")
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg.encoder_name)
    probe_code = train_data[0]["code"]
    enc = tokenizer(probe_code, max_length=exp_cfg.max_length, padding="max_length", truncation=True, return_tensors="pt")
    if enc["input_ids"].shape[-1] != exp_cfg.max_length:
        raise PreflightError("Tokenizer output length mismatch.")
    ast_seq = extract_ast_sequence(probe_code, exp_cfg.ast_seq_len)
    struct_feat = extract_structural_features(probe_code)
    if len(ast_seq) != exp_cfg.ast_seq_len:
        raise PreflightError("AST sequence length mismatch.")
    if len(struct_feat) != STRUCTURAL_FEATURE_DIM:
        raise PreflightError("Structural feature dim mismatch.")
    report = {
        "task": cfg.task, "num_classes": int(num_classes),
        "sizes": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "code_stats": code_stats, "device": exp_cfg.device, "gpu": get_gpu_name(),
        "encoder": exp_cfg.encoder_name, "precision": exp_cfg.precision,
    }
    logger.info(f"PREFLIGHT OK | classes={num_classes} | sizes={report['sizes']}")
    return report


# ===========================================================================
# Config builder + loader factory
# ===========================================================================

def build_codet_config(task_tag: str, save_root: str) -> SpectralConfig:
    strict_ts = ast_parser_available()
    cfg = SpectralConfig(
        task=task_tag,
        benchmark="codet_m4",
        save_dir=f"{save_root}/codet_{task_tag}",
        require_tree_sitter=strict_ts,
        precision="auto",
        auto_h100_profile=True,
        num_workers=4,
        prefetch_factor=2,
    )
    return apply_hardware_profile(cfg)


def make_loaders(train_data, val_data, test_data, tokenizer, exp_cfg: SpectralConfig):
    train_ds = AICDDataset(train_data, tokenizer, exp_cfg.max_length, exp_cfg.ast_seq_len)
    val_ds = AICDDataset(val_data, tokenizer, exp_cfg.max_length, exp_cfg.ast_seq_len)
    test_ds = AICDDataset(test_data, tokenizer, exp_cfg.max_length, exp_cfg.ast_seq_len)
    loader_kwargs = dict(
        batch_size=exp_cfg.batch_size,
        num_workers=exp_cfg.num_workers,
        pin_memory=exp_cfg.pin_memory,
        persistent_workers=exp_cfg.num_workers > 0,
        prefetch_factor=exp_cfg.prefetch_factor if exp_cfg.num_workers > 0 else None,
    )
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader


# ===========================================================================
# IID runner
# ===========================================================================

def run_iid(
    task: str,
    codet_cfg: Optional[CoDETM4Config] = None,
    loss_fn: Optional[Any] = None,
    run_preflight: bool = True,
    checkpoint_tag_prefix: str = "model",
) -> Dict[str, Any]:
    if codet_cfg is None:
        codet_cfg = CoDETM4Config()
    codet_cfg.task = task

    exp_cfg = build_codet_config(f"iid_{task}", codet_cfg.save_root)
    exp_cfg.task = task
    set_seed(codet_cfg.seed)

    if run_preflight:
        preflight(codet_cfg, exp_cfg)

    train_data, val_data, test_data, num_classes = load_codet_m4_data(codet_cfg)
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg.encoder_name)
    train_loader, val_loader, test_loader = make_loaders(train_data, val_data, test_data, tokenizer, exp_cfg)

    class_weights = compute_class_weights(train_data["label"], num_classes)
    model = SpectralCode(exp_cfg, num_classes=num_classes)
    trainer = Trainer(
        exp_cfg, model, train_loader, val_loader, test_loader,
        class_weights=class_weights, loss_fn=loss_fn,
        checkpoint_tag_prefix=checkpoint_tag_prefix,
    )
    result = trainer.train()
    # Breakdown on the full IID test
    if codet_cfg.eval_breakdown:
        preds, labels = collect_predictions(model, test_loader, exp_cfg.device)
        breakdown = run_breakdown_eval(preds, labels, test_data, task)
        result["breakdown"] = breakdown

    # Cleanup
    del model, trainer, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


# ===========================================================================
# OOD LOO runners
# ===========================================================================

def _run_single_loo(
    hold_out_field: str,
    hold_out_value: str,
    codet_cfg: CoDETM4Config,
    loss_fn: Optional[Any] = None,
    checkpoint_tag_prefix: str = "model",
) -> Dict[str, Any]:
    train_data, val_data, test_data, num_classes = load_codet_m4_loo(codet_cfg, hold_out_field, hold_out_value)
    exp_cfg = build_codet_config(f"ood_{hold_out_field}_{hold_out_value}", codet_cfg.save_root)
    exp_cfg.task = codet_cfg.task
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg.encoder_name)
    train_loader, val_loader, test_loader = make_loaders(train_data, val_data, test_data, tokenizer, exp_cfg)
    class_weights = compute_class_weights(train_data["label"], num_classes)
    model = SpectralCode(exp_cfg, num_classes=num_classes)
    trainer = Trainer(
        exp_cfg, model, train_loader, val_loader, test_loader,
        class_weights=class_weights, loss_fn=loss_fn,
        checkpoint_tag_prefix=checkpoint_tag_prefix,
    )
    result = trainer.train()
    preds, labels = collect_predictions(model, test_loader, exp_cfg.device)
    result["breakdown"] = run_breakdown_eval(preds, labels, test_data, codet_cfg.task)
    result["hold_out_field"] = hold_out_field
    result["hold_out_value"] = hold_out_value

    del model, trainer, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_ood_generator(codet_cfg: Optional[CoDETM4Config] = None, loss_fn: Optional[Any] = None) -> Dict[str, Dict]:
    """Proxy Table 8. CoDET-M4 test exposes 5 generator LOOs: codellama/gpt/llama3.1/nxcode/qwen1.5."""
    if codet_cfg is None:
        codet_cfg = CoDETM4Config()
    gens = ["codellama", "gpt", "llama3.1", "nxcode", "qwen1.5"]
    results: Dict[str, Dict] = {}
    for gen in gens:
        logger.info(f"\n{'='*70}\n[OOD-GEN] Holding out generator={gen}\n{'='*70}")
        try:
            results[gen] = _run_single_loo("generator", gen, codet_cfg, loss_fn=loss_fn)
        except RuntimeError as e:
            logger.error(f"OOD generator={gen} failed: {e}")
            results[gen] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _log_ood_summary("OOD-GENERATOR (proxy Table 8)", results)
    return results


def run_ood_language(codet_cfg: Optional[CoDETM4Config] = None, loss_fn: Optional[Any] = None) -> Dict[str, Dict]:
    """Proxy Table 10/12. CoDET-M4 covers cpp / java / python."""
    if codet_cfg is None:
        codet_cfg = CoDETM4Config()
    langs = ["cpp", "java", "python"]
    results: Dict[str, Dict] = {}
    for lang in langs:
        logger.info(f"\n{'='*70}\n[OOD-LANG] Holding out language={lang}\n{'='*70}")
        try:
            results[lang] = _run_single_loo("language", lang, codet_cfg, loss_fn=loss_fn)
        except RuntimeError as e:
            logger.error(f"OOD language={lang} failed: {e}")
            results[lang] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _log_ood_summary("OOD-LANGUAGE (proxy Table 10/12)", results)
    return results


def run_ood_source(codet_cfg: Optional[CoDETM4Config] = None, loss_fn: Optional[Any] = None) -> Dict[str, Dict]:
    """Proxy Table 9. Sources: CodeForces (cf), GitHub (gh), LeetCode (lc)."""
    if codet_cfg is None:
        codet_cfg = CoDETM4Config()
    sources = ["cf", "gh", "lc"]
    results: Dict[str, Dict] = {}
    for src in sources:
        logger.info(f"\n{'='*70}\n[OOD-SRC] Holding out source={src}\n{'='*70}")
        try:
            results[src] = _run_single_loo("source", src, codet_cfg, loss_fn=loss_fn)
        except RuntimeError as e:
            logger.error(f"OOD source={src} failed: {e}")
            results[src] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _log_ood_summary("OOD-SOURCE (proxy Table 9)", results)
    return results


def _log_ood_summary(title: str, results: Dict[str, Dict]):
    logger.info(f"\n{'='*70}\n{title} SUMMARY\n{'='*70}")
    logger.info("| held-out | test_macro_f1 | test_weighted_f1 | best_val_f1 |")
    logger.info("|---|---:|---:|---:|")
    for key, stats in results.items():
        if "error" in stats:
            logger.info(f"| {key} | ERROR | - | - |")
            continue
        logger.info(
            f"| {key} | {stats.get('test_f1', 0):.4f} | "
            f"{stats.get('test_weighted_f1', 0):.4f} | "
            f"{stats.get('best_val_f1', 0):.4f} |"
        )


# ===========================================================================
# Full suite
# ===========================================================================

FULL_RUN_PLAN: List[Tuple[str, str]] = [
    ("iid", "binary"),       # Table 2 + per-lang(3) + per-source(4)
    ("iid", "author"),       # Table 7 + confusion matrix
    ("ood", "generator"),    # proxy Table 8
    ("ood", "language"),     # proxy Table 10/12
    ("ood", "source"),       # proxy Table 9
]


def run_codet_suite(
    run_plan: Optional[List[Tuple[str, str]]] = None,
    base_cfg: Optional[CoDETM4Config] = None,
    loss_fn: Optional[Any] = None,
    run_preflight: bool = True,
    checkpoint_tag_prefix: str = "model",
) -> Dict[str, Any]:
    """Run selected CoDET-M4 evaluation modes. Returns dict keyed by f'{mode}_{task}'."""
    if run_plan is None:
        run_plan = FULL_RUN_PLAN
    if base_cfg is None:
        base_cfg = CoDETM4Config()
    all_results: Dict[str, Any] = {}

    logger.info(f"\n{'='*70}\nCoDET-M4 SUITE: {len(run_plan)} evaluation modes\n{'='*70}")
    for i, (mode, task) in enumerate(run_plan):
        logger.info(f"  [{i+1}] {mode}/{task}")
    logger.info("=" * 70)

    for mode, task in run_plan:
        key = f"{mode}_{task}"
        logger.info(f"\n{'#'*70}\nSUITE: {key}\n{'#'*70}")
        if mode == "iid":
            all_results[key] = run_iid(
                task, base_cfg, loss_fn=loss_fn, run_preflight=run_preflight,
                checkpoint_tag_prefix=checkpoint_tag_prefix,
            )
        elif mode == "ood" and task == "generator":
            all_results[key] = run_ood_generator(base_cfg, loss_fn=loss_fn)
        elif mode == "ood" and task == "language":
            all_results[key] = run_ood_language(base_cfg, loss_fn=loss_fn)
        elif mode == "ood" and task == "source":
            all_results[key] = run_ood_source(base_cfg, loss_fn=loss_fn)
        else:
            logger.warning(f"Unknown suite entry: {mode}/{task}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return all_results
