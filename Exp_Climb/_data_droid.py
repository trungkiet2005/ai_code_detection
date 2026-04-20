"""
DroidCollection data loading + T1/T3/T4 suite.

Covers DroidCollection evaluation protocols (Orel et al. EMNLP 2025):
  - T1 (binary, 2-class): human vs AI (all non-human merged)
  - T3 (multi-class, 3-class): human / machine_generated / machine_refined
        (paper's main table -- Table 3/4 weighted-F1)
  - T4 (4-class, adversarial-aware): human / generated / refined / adversarial

CLIMB protocol: 100K train subsample, FULL test (paper-comparable).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

from _common import PreflightError, SpectralConfig, apply_hardware_profile, logger, set_seed
from _model import SpectralCode
from _trainer import Trainer, compute_class_weights


# ===========================================================================
# Config
# ===========================================================================

@dataclass
class DroidConfig:
    dataset_id: str = "project-droid/DroidCollection"
    config_name: str = "default"
    split_map: Dict[str, str] = field(default_factory=lambda: {
        "train": "train",
        "validation": "dev",
        "test": "test",
    })
    # CLIMB: small train, FULL test
    max_train_samples: int = 100_000
    max_val_samples: int = 20_000
    max_test_samples: int = -1

    seed: int = 42
    save_root: str = "./droid_checkpoints"


# ===========================================================================
# Label mapping
# ===========================================================================

def _map_droid_label_to_task(row: Dict[str, object], task: str) -> int:
    """Droid Labels: HUMAN_GENERATED, MACHINE_GENERATED, MACHINE_REFINED,
    MACHINE_GENERATED_ADVERSARIAL."""
    normalized = str(row.get("Label", "")).upper()

    if task == "T1":
        return 0 if normalized == "HUMAN_GENERATED" else 1

    if task == "T3":
        if normalized == "HUMAN_GENERATED":
            return 0
        if normalized in ("MACHINE_GENERATED", "MACHINE_GENERATED_ADVERSARIAL"):
            return 1
        if normalized == "MACHINE_REFINED":
            return 2
        return -1

    if task == "T4":
        mapping = {
            "HUMAN_GENERATED": 0,
            "MACHINE_GENERATED": 1,
            "MACHINE_REFINED": 2,
            "MACHINE_GENERATED_ADVERSARIAL": 3,
        }
        return mapping.get(normalized, -1)

    return -1


def _sample_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), max_samples)
    return dataset.select(indices)


def _droid_label_raw(row) -> str:
    return str(row.get("Label", "")).upper()


# -----------------------------------------------------------------------------
# Source -> Paper Table 3 domain bucket (verified against HF schema 2026-04-20).
#
# DroidCollection HF columns (confirmed):
#   Code, Label, Language, Generator, Generation_Mode, Source,
#   Sampling_Params, Rewriting_Params, Model_Family
#
# Paper section 3.1 splits the Source values into three domains. The mapping
# is not a single field on HF -- we reconstruct it per the paper's §3.1 text.
# -----------------------------------------------------------------------------
DROID_SOURCE_TO_DOMAIN = {
    # General Use Code (paper §3.1): web, firmware, game-engine etc. from
    # GitHub-hosted code datasets.
    "STARCODER_DATA": "general",
    "THEVAULT_FUNCTION": "general",
    "DROID_PERSONAHUB": "general",  # PersonaHub-prompted open-domain code
    # Algorithmic Problems: competitive-programming platforms.
    "TACO": "algorithmic",
    "CODENET": "algorithmic",
    "ATCODER": "algorithmic",
    "AIZU": "algorithmic",
    "LEETCODE": "algorithmic",
    "CODEFORCES": "algorithmic",
    # Research / Data-Science: ObscuraCoder + Lu 2025 research-paper code.
    # Column values for this family are less documented; we treat anything
    # with "OBSCURA" or "RESEARCH" prefix as Research/DS.  `""` below is a
    # fallthrough that run_breakdown_eval_droid ignores.
    "OBSCURACODER": "research_ds",
    "OBSCURA_CODER": "research_ds",
    "RESEARCH": "research_ds",
    "DATASCIENCE": "research_ds",
}


def _source_to_domain(source_raw: str) -> str:
    """Paper Table 3 domain mapping. Normalises case + separators then looks
    up the fixed table. Returns '' for unknown sources (breakdown skips).
    Also provides a substring-based fallback so variants like "leet-code"
    and "Leet Code" still resolve.
    """
    if not source_raw:
        return ""
    # Strict key: uppercase, underscore as canonical separator.
    key = source_raw.strip().upper().replace("-", "_").replace(" ", "_")
    if key in DROID_SOURCE_TO_DOMAIN:
        return DROID_SOURCE_TO_DOMAIN[key]
    # Substring / no-separator fallback covers "LEET_CODE" vs "LEETCODE",
    # "OBSCURA_CODER" vs "OBSCURACODER", future schema drift, etc.
    nosep = key.replace("_", "")
    if "OBSCURA" in nosep or "RESEARCH" in nosep or "DATASCIENCE" in nosep:
        return "research_ds"
    if any(k in nosep for k in ("LEETCODE", "CODEFORCES", "ATCODER", "AIZU", "TACO", "CODENET")):
        return "algorithmic"
    if any(k in nosep for k in ("STARCODER", "VAULT", "PERSONA", "GITHUB")):
        return "general"
    return ""


def load_droid_data(cfg: DroidConfig, task: str):
    """Load Droid splits and convert to {code, label, language, source,
    domain, generator, model_family, generation_mode, ...}.

    Schema verified against the DroidCollection HF viewer (2026-04-20):
    columns = Code / Label / Language / Generator / Generation_Mode / Source /
              Sampling_Params / Rewriting_Params / Model_Family.

    `domain` is derived from Source via DROID_SOURCE_TO_DOMAIN (paper §3.1);
    any dim that ends up all-empty is silently skipped by the breakdown, so
    future Source values that we haven't catalogued degrade gracefully.
    """
    def _convert_split(split_name: str, max_samples: int) -> Dataset:
        source_split = cfg.split_map[split_name]
        logger.info(f"Loading Droid split '{source_split}' for {split_name}...")
        ds = load_dataset(cfg.dataset_id, name=cfg.config_name, split=source_split)
        ds = _sample_dataset(ds, max_samples, cfg.seed)

        def _convert_row(row):
            raw_label = _droid_label_raw(row)
            source_raw = str(row.get("Source", "") or "").strip()
            return {
                "code": row.get("Code", "") or "",
                "label": _map_droid_label_to_task(row, task),
                # The shared AICDDataset.source_id field maps {cf,gh,lc}->0..2
                # and uses -1 for everything else. Droid sources are not CoDET's
                # cf/gh/lc so we emit "" (sentinel -1) here. Source-conditioned
                # losses (exp_02/08/14/18) short-circuit cleanly.
                "source": "",
                # Paper Table 3/4/5 breakdown dims.
                "language": str(row.get("Language", "") or "").strip().lower(),
                "source_raw": source_raw,                     # e.g. "LEETCODE"
                "domain": _source_to_domain(source_raw),      # "algorithmic"
                "generator": str(row.get("Generator", "") or "").strip().lower(),
                "model_family": str(row.get("Model_Family", "") or "").strip().lower(),
                "generation_mode": str(row.get("Generation_Mode", "") or "").strip().upper(),
                "is_adversarial": int(raw_label == "MACHINE_GENERATED_ADVERSARIAL"),
                "is_human": int(raw_label == "HUMAN_GENERATED"),
            }

        converted = ds.map(_convert_row, remove_columns=ds.column_names)
        return converted.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

    train_data = _convert_split("train", cfg.max_train_samples)
    val_data = _convert_split("validation", cfg.max_val_samples)
    test_data = _convert_split("test", cfg.max_test_samples)
    num_classes = len(set(train_data["label"]))
    logger.info(
        f"Droid sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
    )
    logger.info(f"Droid {task} classes: {num_classes}, Labels: {sorted(set(train_data['label']))}")
    return train_data, val_data, test_data, num_classes


# ===========================================================================
# Loader factory (thin wrapper to avoid importing from _data_codet)
# ===========================================================================

def _make_droid_loaders(train_data, val_data, test_data, tokenizer, exp_cfg: SpectralConfig):
    from torch.utils.data import DataLoader
    from _model import AICDDataset

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
    return (
        DataLoader(train_ds, shuffle=True, **loader_kwargs),
        DataLoader(val_ds, shuffle=False, **loader_kwargs),
        DataLoader(test_ds, shuffle=False, **loader_kwargs),
    )


# ===========================================================================
# Preflight
# ===========================================================================

def preflight_droid(droid_cfg: DroidConfig, tasks: Optional[List[str]] = None) -> Dict[str, Any]:
    """Quick dataset access + label sanity check for Droid.

    Loads a **tiny** sample (1000 rows) from each split for each requested task,
    verifying: HF dataset reachable, non-zero labels per task, sensible class count.
    Raises PreflightError on failure.
    """
    if tasks is None:
        tasks = ["T1", "T3", "T4"]

    logger.info("\n" + "=" * 70)
    logger.info(f"PREFLIGHT DROID | dataset={droid_cfg.dataset_id} | tasks={tasks}")
    logger.info("=" * 70)

    report: Dict[str, Any] = {"dataset_id": droid_cfg.dataset_id, "tasks": {}}
    for task in tasks:
        try:
            logger.info(f"[preflight] Probing Droid {task}...")
            probe_train = load_dataset(
                droid_cfg.dataset_id,
                name=droid_cfg.config_name,
                split=droid_cfg.split_map["train"],
            ).select(range(min(1000, 100_000)))

            mapped_labels = [_map_droid_label_to_task(r, task) for r in probe_train]
            valid = [l for l in mapped_labels if l >= 0]
            if len(valid) < 100:
                raise PreflightError(
                    f"Droid {task}: only {len(valid)}/1000 probe rows mapped to valid labels"
                )
            class_count = len(set(valid))
            expected = {"T1": 2, "T3": 3, "T4": 4}[task]
            if class_count < 2:
                raise PreflightError(
                    f"Droid {task}: found {class_count} classes in probe (need >= 2). "
                    f"Expected {expected}. Check label mapping."
                )
            if class_count > expected:
                logger.warning(
                    f"Droid {task}: probe found {class_count} classes, expected {expected} "
                    f"(may converge on full data)"
                )
            report["tasks"][task] = {
                "probe_size": len(probe_train),
                "valid_labels": len(valid),
                "class_count": class_count,
                "expected_classes": expected,
            }
            logger.info(
                f"  Droid {task} OK | valid={len(valid)}/1000 | classes={class_count} (expected {expected})"
            )
        except PreflightError:
            raise
        except Exception as e:
            raise PreflightError(f"Droid {task} probe failed: {e}") from e

    logger.info("PREFLIGHT DROID OK")
    return report


# ===========================================================================
# Paper-protocol breakdown helper (Droid Tables 3/4/5)
# ===========================================================================

def collect_droid_predictions(model: torch.nn.Module, loader, device: str):
    """Return (preds, labels) arrays for a trained Droid model."""
    import numpy as np
    model.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ast_seq = batch["ast_seq"].to(device)
        struct_feat = batch["struct_feat"].to(device)
        with torch.no_grad():
            out = model(input_ids, attention_mask, ast_seq, struct_feat)
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy())
    return np.array(all_preds), np.array(all_labels)


def run_breakdown_eval_droid(preds, labels, test_data, task: str) -> Dict[str, Any]:
    """Paper-protocol breakdown emitting Table 3 (per-domain), Table 4
    (per-language), and Table 5 (human / adversarial recall) cells from a
    single trained model.  Any dim whose column is all-empty in the HF
    schema is silently skipped so the function is robust to schema drift.
    """
    import numpy as np
    from sklearn.metrics import f1_score, recall_score

    results: Dict[str, Any] = {}
    # `len(test_data)` is number-of-columns for a dict-backed test set and
    # number-of-rows for a HF Dataset, so pick the row count explicitly.
    if hasattr(test_data, "num_rows"):
        n_rows = int(test_data.num_rows)
    else:
        # dict-of-columns: take the first column's length.
        try:
            first_col = next(iter(test_data.values()))
            n_rows = len(first_col)
        except (StopIteration, TypeError):
            n_rows = len(preds)
    n = min(len(preds), n_rows)
    preds, labels = preds[:n], labels[:n]

    overall_macro = float(f1_score(labels, preds, average="macro"))
    overall_weighted = float(f1_score(labels, preds, average="weighted"))
    results["overall"] = {"macro_f1": overall_macro, "weighted_f1": overall_weighted}
    logger.info(f"  Overall: macro={overall_macro:.4f}  weighted={overall_weighted:.4f}")

    # Table 3 analog (per-domain: general/algorithmic/research_ds, derived from Source)
    # Table 4 analog (per-language: C-C++, C#, Go, Java, Python, JS)
    # Extra: per-raw-source (TACO, LEETCODE, STARCODER_DATA, ...)
    #         per-generator (GPT-4o-mini, Qwen2.5-72B, ...)
    #         per-model_family (paper Table 3 bottom: within-model-family stability)
    for dim_name in ("domain", "language", "source_raw", "generator", "model_family"):
        try:
            dim_vals = test_data[dim_name][:n]
        except KeyError:
            continue
        # Skip dim if every value is empty (schema drift)
        if not any(v for v in dim_vals):
            continue
        unique = sorted(set(v for v in dim_vals if v))
        dim_results: Dict[str, Dict[str, float]] = {}
        dim_paper_tag = {
            "domain":       "paper Table 3",
            "language":     "paper Table 4",
            "source_raw":   "raw Source column (pre-domain bucketing)",
            "generator":    "per-generator diagnostic",
            "model_family": "paper §A model-family breakdown",
        }.get(dim_name, dim_name)
        logger.info(f"  Breakdown by {dim_name} ({dim_paper_tag}):")
        for val in unique:
            mask = np.array([v == val for v in dim_vals])
            if mask.sum() < 10:
                continue
            mf1 = float(f1_score(labels[mask], preds[mask], average="macro"))
            wf1 = float(f1_score(labels[mask], preds[mask], average="weighted"))
            dim_results[val] = {"n": int(mask.sum()), "macro_f1": mf1, "weighted_f1": wf1}
            logger.info(f"    {val:>14s}: n={int(mask.sum()):>6d}  macro={mf1:.4f}  weighted={wf1:.4f}")
        results[dim_name] = dim_results

    # Table 5 analog: human-class recall + adversarial-class recall.
    # Only meaningful when we have both flags exposed.
    if task in ("T3", "T4"):
        try:
            is_human = np.asarray(test_data["is_human"][:n])
            is_adv = np.asarray(test_data["is_adversarial"][:n])
            if is_human.any():
                # Label 0 = human in both T3 and T4.
                hum_correct = (preds[is_human == 1] == 0).mean() if is_human.sum() else 0.0
                results["human_recall"] = float(hum_correct)
                logger.info(f"  Human recall (Table 5): {hum_correct:.4f}  (n={int(is_human.sum())})")
            if is_adv.any():
                # Adversarial samples: in T3, labelled 1 (merged with MACHINE_GENERATED);
                # in T4, labelled 3 (own class).
                adv_target = 1 if task == "T3" else 3
                adv_correct = (preds[is_adv == 1] == adv_target).mean() if is_adv.sum() else 0.0
                results["adversarial_recall"] = float(adv_correct)
                logger.info(f"  Adversarial recall (Table 5): {adv_correct:.4f}  (n={int(is_adv.sum())})")
        except (KeyError, IndexError):
            pass

    return results


# ===========================================================================
# Config builder
# ===========================================================================

def build_droid_config(task_tag: str, save_root: str) -> SpectralConfig:
    cfg = SpectralConfig(
        task=task_tag,
        benchmark="droid",
        save_dir=f"{save_root}/droid_{task_tag}",
        precision="auto",
        auto_h100_profile=True,
        num_workers=4,
        prefetch_factor=2,
    )
    return apply_hardware_profile(cfg)


# ===========================================================================
# Runners
# ===========================================================================

def run_droid_iid(
    task: str,
    droid_cfg: Optional[DroidConfig] = None,
    loss_fn: Optional[Any] = None,
    checkpoint_tag_prefix: str = "model",
    eval_breakdown: bool = False,
) -> Dict[str, Any]:
    """Train + eval on one Droid task (T1/T3/T4) with FULL test set.

    eval_breakdown=True -> after training, run the paper-protocol breakdown
    (Table 3 per-domain / Table 4 per-language / Table 5 human + adversarial
    recall).  Adds ~30 s of post-eval inference time; requires the Droid HF
    schema to expose Language / Domain columns (silently skipped if not).
    """
    if droid_cfg is None:
        droid_cfg = DroidConfig()

    logger.info("\n" + "=" * 70)
    logger.info(f"[Droid] Running IID task={task}")
    logger.info("=" * 70)

    set_seed(droid_cfg.seed)
    train_data, val_data, test_data, num_classes = load_droid_data(droid_cfg, task)
    exp_cfg = build_droid_config(task, droid_cfg.save_root)
    exp_cfg.task = task
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg.encoder_name)
    train_loader, val_loader, test_loader = _make_droid_loaders(train_data, val_data, test_data, tokenizer, exp_cfg)

    class_weights = compute_class_weights(train_data["label"], num_classes)
    model = SpectralCode(exp_cfg, num_classes=num_classes)
    trainer = Trainer(
        exp_cfg, model, train_loader, val_loader, test_loader,
        class_weights=class_weights, loss_fn=loss_fn,
        checkpoint_tag_prefix=checkpoint_tag_prefix,
    )
    result = trainer.train()
    result["task"] = task
    result["num_classes"] = int(num_classes)
    result["paper_primary_metric"] = "weighted_f1"  # Droid primary = Weighted-F1

    if eval_breakdown:
        preds, labels = collect_droid_predictions(model, test_loader, exp_cfg.device)
        result["breakdown"] = run_breakdown_eval_droid(preds, labels, test_data, task)

    del model, trainer, train_loader, val_loader, test_loader
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def run_droid_suite(
    droid_cfg: Optional[DroidConfig] = None,
    tasks: Optional[List[str]] = None,
    loss_fn: Optional[Any] = None,
    checkpoint_tag_prefix: str = "model",
) -> Dict[str, Any]:
    """Run T1 + T3 + T4 sequentially. Returns dict keyed by 'droid_<task>'."""
    if droid_cfg is None:
        droid_cfg = DroidConfig()
    if tasks is None:
        tasks = ["T1", "T3", "T4"]

    results: Dict[str, Any] = {}
    for task in tasks:
        key = f"droid_{task}"
        logger.info(f"\n{'#'*70}\nSUITE: {key}\n{'#'*70}")
        try:
            results[key] = run_droid_iid(task, droid_cfg, loss_fn=loss_fn, checkpoint_tag_prefix=checkpoint_tag_prefix)
        except Exception as e:
            logger.error(f"Droid {task} failed: {e}")
            results[key] = {"error": str(e)}

    logger.info(f"\n{'='*70}\nDROID SUITE SUMMARY\n{'='*70}")
    logger.info("| Task | Macro-F1 | Weighted-F1 | Best Val |")
    logger.info("|---|---:|---:|---:|")
    for key, stats in results.items():
        if "error" in stats:
            logger.info(f"| {key} | ERROR | - | - |")
            continue
        logger.info(
            f"| {key} | {stats.get('test_macro_f1', stats.get('test_f1', 0)):.4f} | "
            f"{stats.get('test_weighted_f1', 0):.4f} | {stats.get('best_val_f1', 0):.4f} |"
        )
    return results
