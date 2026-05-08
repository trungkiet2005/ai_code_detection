"""Zero-shot dataset loaders (no training, only dev + test splits).

Contract: each loader returns (dev_rows, test_rows) where each row is
    {"code": str, "label": int, "language": str, "source_raw": str,
     "domain": str, "is_human": int, "is_adversarial": int}

`label` semantics differ per benchmark:
  - droid_T1 (2-class): 0 = human, 1 = machine-generated
  - droid_T3 (3-class): 0 = human, 1 = machine-generated, 2 = machine-refined
  - codet_binary: 0 = human, 1 = machine-generated

For zero-shot we do NOT train on these labels — they're only used for
threshold calibration (on dev) and evaluation (on test). The label set
is kept consistent with Exp_Climb's _data_droid.py / _data_codet.py so
breakdowns are directly comparable.
"""
from __future__ import annotations

import random
from typing import Dict, List, Tuple

from datasets import Dataset, load_dataset

from _common import logger


DROID_DATASET_ID = "project-droid/DroidCollection"
CODET_DATASET_ID = "DaniilOr/CoDET-M4"

# Source -> paper Table 3 domain bucketing. Mirrors _data_droid.py exactly
# so the ZS breakdown is directly comparable to the Climb one.
DROID_SOURCE_TO_DOMAIN = {
    "STARCODER_DATA": "general",
    "THEVAULT_FUNCTION": "general",
    "DROID_PERSONAHUB": "general",
    "TACO": "algorithmic",
    "CODENET": "algorithmic",
    "ATCODER": "algorithmic",
    "AIZU": "algorithmic",
    "LEETCODE": "algorithmic",
    "CODEFORCES": "algorithmic",
    "OBSCURACODER": "research_ds",
    "OBSCURA_CODER": "research_ds",
    "RESEARCH": "research_ds",
    "DATASCIENCE": "research_ds",
}


def _source_to_domain(source_raw: str) -> str:
    if not source_raw:
        return ""
    key = source_raw.strip().upper().replace("-", "_").replace(" ", "_")
    if key in DROID_SOURCE_TO_DOMAIN:
        return DROID_SOURCE_TO_DOMAIN[key]
    nosep = key.replace("_", "")
    if "OBSCURA" in nosep or "RESEARCH" in nosep or "DATASCIENCE" in nosep:
        return "research_ds"
    if any(k in nosep for k in ("LEETCODE", "CODEFORCES", "ATCODER", "AIZU", "TACO", "CODENET")):
        return "algorithmic"
    if any(k in nosep for k in ("STARCODER", "VAULT", "PERSONA", "GITHUB")):
        return "general"
    return ""


def _sample_dataset(dataset, max_samples: int, seed: int):
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), max_samples)
    return dataset.select(indices)


def _map_droid_label(row, task: str) -> int:
    lab = str(row.get("Label", "")).upper()
    if task == "droid_T1":
        return 0 if lab == "HUMAN_GENERATED" else 1
    if task == "droid_T3":
        if lab == "HUMAN_GENERATED":
            return 0
        if lab in ("MACHINE_GENERATED", "MACHINE_GENERATED_ADVERSARIAL"):
            return 1
        if lab == "MACHINE_REFINED":
            return 2
    return -1


def load_droid_zs(
    task: str = "droid_T3",
    max_dev_samples: int = 5_000,
    max_test_samples: int = -1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Load Droid dev + test splits for zero-shot evaluation.

    `task` in {"droid_T1", "droid_T3"}. No `train` split is loaded.
    """
    def _load_split(hf_split: str, max_n: int) -> Dataset:
        logger.info(f"[ZS] Loading Droid '{hf_split}' split ...")
        ds = load_dataset(DROID_DATASET_ID, split=hf_split)
        ds = _sample_dataset(ds, max_n, seed)

        def _convert(row):
            raw_label = str(row.get("Label", "")).upper()
            src_raw = str(row.get("Source", "") or "").strip()
            return {
                "code": row.get("Code", "") or "",
                "label": _map_droid_label(row, task),
                "language": str(row.get("Language", "") or "").strip().lower(),
                "source_raw": src_raw,
                "domain": _source_to_domain(src_raw),
                "generator": str(row.get("Generator", "") or "").strip().lower(),
                "model_family": str(row.get("Model_Family", "") or "").strip().lower(),
                "is_human": int(raw_label == "HUMAN_GENERATED"),
                "is_adversarial": int(raw_label == "MACHINE_GENERATED_ADVERSARIAL"),
            }

        converted = ds.map(_convert, remove_columns=ds.column_names)
        return converted.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

    dev = _load_split("dev", max_dev_samples)
    test = _load_split("test", max_test_samples)
    logger.info(f"[ZS] Droid {task}: dev={len(dev)}, test={len(test)}")
    return dev, test


def load_codet_zs(
    max_dev_samples: int = 5_000,
    max_test_samples: int = -1,
    seed: int = 42,
) -> Tuple[Dataset, Dataset]:
    """Load CoDET-M4 binary classification splits for zero-shot eval.
    The CoDET HF dataset has a built-in `split` column (train/val/test).
    """
    logger.info("[ZS] Loading CoDET-M4 full ...")
    ds = load_dataset(CODET_DATASET_ID, split="train")   # full table; has split col
    rows = {"val": [], "test": []}
    for row in ds:
        split = str(row.get("split", "")).lower()
        if split not in rows:
            continue
        target = str(row.get("target", "")).lower()
        # target == "human" -> 0, else 1. Matches _data_codet binary mapping.
        label = 0 if "human" in target else 1
        code = row.get("cleaned_code") or row.get("code") or ""
        if not code.strip():
            continue
        rows[split].append({
            "code": code,
            "label": label,
            "language": str(row.get("language", "")).strip().lower(),
            "source_raw": str(row.get("source", "")).strip().upper(),
            "domain": str(row.get("source", "")).strip().lower(),   # cf/gh/lc
            "generator": str(row.get("model", "") or "").strip().lower() or "human",
            "model_family": "",
            "is_human": int(label == 0),
            "is_adversarial": 0,
        })

    rng = random.Random(seed)
    if max_dev_samples > 0 and len(rows["val"]) > max_dev_samples:
        rows["val"] = rng.sample(rows["val"], max_dev_samples)
    if max_test_samples > 0 and len(rows["test"]) > max_test_samples:
        rows["test"] = rng.sample(rows["test"], max_test_samples)

    from datasets import Dataset as HFDataset
    dev = HFDataset.from_list(rows["val"])
    test = HFDataset.from_list(rows["test"])
    logger.info(f"[ZS] CoDET binary: dev={len(dev)}, test={len(test)}")
    return dev, test


def load_zs(benchmark: str, cfg) -> Tuple[Dataset, Dataset]:
    """Router — returns (dev, test) for the benchmark requested."""
    if benchmark.startswith("droid"):
        return load_droid_zs(
            task=benchmark,
            max_dev_samples=cfg.max_dev_samples,
            max_test_samples=cfg.max_test_samples,
            seed=cfg.seed,
        )
    if benchmark == "codet_binary":
        return load_codet_zs(
            max_dev_samples=cfg.max_dev_samples,
            max_test_samples=cfg.max_test_samples,
            seed=cfg.seed,
        )
    raise ValueError(f"Unknown ZS benchmark: {benchmark}")
