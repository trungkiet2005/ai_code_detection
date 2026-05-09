"""
exp_fs_baseline_catboost.py -- ONE-FILE self-contained baseline reimplementation.

Method: FS-Baseline-CatBoost
Backbone: handcrafted token-statistics features (NO transformer, NO GPU)
Reimplements the CoDET-M4 paper's CatBoost baseline (paper Table 7: 45.42
Author IID Macro-F1 with full data).

We extract per-sample features in the spirit of Ghostbuster / token-stat
detectors:
  - char counts, line counts, avg/max line length
  - token entropy, burstiness, type-token ratio (TTR), Yule-K
  - punctuation density, identifier-length stats, comment density
  - whitespace patterns
Then train a CatBoost multi-class classifier.

Paste into a Kaggle cell (or run locally on CPU). No `git clone` needed.

Default sweep: K=128 + fraction in {0.01, 0.05} = 3 configs (~10 min CPU).
Override via FS_SWEEP_KS / FS_SWEEP_FRACS env vars.

Output: /kaggle/working/results/exp_fs_baseline_catboost_<label>_seed<S>.json
"""
from __future__ import annotations

import importlib.util
import json
import logging
import math
import os
import random
import re
import subprocess
import sys
import time
import warnings
from collections import Counter, defaultdict
from datetime import datetime
from typing import Dict, List, Tuple

# =============================================================================
# Bootstrap (pip install only -- no clone, no other files)
# =============================================================================

def _ensure_deps():
    required = [
        ("numpy", "numpy"),
        ("datasets", "datasets"),
        ("sklearn", "scikit-learn"),
        ("catboost", "catboost"),
    ]
    missing = [pip for imp, pip in required if importlib.util.find_spec(imp) is None]
    if missing:
        print(f"[catboost-bootstrap] installing: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])


_ensure_deps()

import numpy as np
from catboost import CatBoostClassifier
from datasets import load_dataset
from sklearn.metrics import accuracy_score, classification_report, f1_score

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("exp_fs_baseline_catboost")


METHOD_NAME = "FS-Baseline-CatBoost"
EXP_ID = "exp_fs_baseline_catboost"


# =============================================================================
# Feature extraction (handcrafted token statistics)
# =============================================================================

_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_TOKEN_RE = re.compile(r"\w+|\S")
_COMMENT_PATTERNS = (re.compile(r"//[^\n]*"), re.compile(r"#[^\n]*"),
                     re.compile(r"/\*.*?\*/", re.DOTALL),
                     re.compile(r'""".*?"""', re.DOTALL))


def _entropy(counts):
    total = sum(counts)
    if total == 0:
        return 0.0
    probs = [c / total for c in counts if c > 0]
    return -sum(p * math.log2(p) for p in probs)


def _yule_k(token_counts):
    if not token_counts:
        return 0.0
    n = sum(token_counts)
    if n == 0:
        return 0.0
    m1 = n
    m2 = sum(c * c for c in token_counts)
    if m1 == 0:
        return 0.0
    return 1e4 * (m2 - m1) / (m1 * m1)


def _burstiness(tokens):
    """std/mean of inter-token-position spacing for the most common token."""
    if len(tokens) < 4:
        return 0.0
    counter = Counter(tokens)
    top, _ = counter.most_common(1)[0]
    positions = [i for i, t in enumerate(tokens) if t == top]
    if len(positions) < 2:
        return 0.0
    diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    if not diffs:
        return 0.0
    mean = sum(diffs) / len(diffs)
    if mean == 0:
        return 0.0
    var = sum((d - mean) ** 2 for d in diffs) / len(diffs)
    std = math.sqrt(var)
    return (std - mean) / (std + mean)


def extract_features(code: str) -> List[float]:
    """Return a fixed-length feature vector per sample."""
    if not code:
        code = " "
    n_chars = len(code)
    n_lines = code.count("\n") + 1
    line_lens = [len(line) for line in code.split("\n")]
    avg_line = float(np.mean(line_lens)) if line_lens else 0.0
    max_line = float(np.max(line_lens)) if line_lens else 0.0
    std_line = float(np.std(line_lens)) if line_lens else 0.0

    tokens = _TOKEN_RE.findall(code)
    n_tok = len(tokens)
    if n_tok == 0:
        tokens = [""]
        n_tok = 1
    counts = Counter(tokens)
    n_unique = len(counts)
    ttr = n_unique / n_tok
    h_tok = _entropy(list(counts.values()))
    burst = _burstiness(tokens)
    yule = _yule_k(list(counts.values()))

    idents = _IDENT_RE.findall(code)
    n_idents = len(idents)
    avg_ident_len = float(np.mean([len(i) for i in idents])) if idents else 0.0
    max_ident_len = float(np.max([len(i) for i in idents])) if idents else 0.0

    n_punct = sum(1 for c in code if not c.isalnum() and not c.isspace())
    punct_density = n_punct / n_chars if n_chars else 0.0
    n_ws = sum(1 for c in code if c.isspace())
    ws_density = n_ws / n_chars if n_chars else 0.0
    indent_chars = sum(1 for line in code.split("\n") for c in line[:len(line) - len(line.lstrip())])
    indent_density = indent_chars / n_chars if n_chars else 0.0

    n_comment_chars = sum(len(m) for pat in _COMMENT_PATTERNS for m in pat.findall(code))
    comment_density = n_comment_chars / n_chars if n_chars else 0.0

    char_counts = Counter(code)
    h_char = _entropy(list(char_counts.values()))

    return [
        n_chars, n_lines, avg_line, max_line, std_line,
        n_tok, n_unique, ttr, h_tok, burst, yule,
        n_idents, avg_ident_len, max_ident_len,
        punct_density, ws_density, indent_density, comment_density,
        h_char,
    ]


FEATURE_NAMES = [
    "n_chars", "n_lines", "avg_line", "max_line", "std_line",
    "n_tok", "n_unique", "ttr", "h_tok", "burst", "yule",
    "n_idents", "avg_ident_len", "max_ident_len",
    "punct_density", "ws_density", "indent_density", "comment_density",
    "h_char",
]


# =============================================================================
# CoDET-M4 loader (label vocab + K-shot/fraction subset)
# =============================================================================

def _is_human(t):
    return str(t or "").strip().lower() in {"human", "human_written", "human-generated", "human_generated"}


def _build_vocab(train):
    names = set()
    for row in train:
        if not _is_human(row.get("target", "")):
            m = str(row.get("model", "") or "").strip()
            if m:
                names.add(m)
    return {n: i + 1 for i, n in enumerate(sorted(names))}


def _label(row, vocab):
    if _is_human(row.get("target", "")):
        return 0
    return vocab.get(str(row.get("model", "") or "").strip(), -1)


def _code(row):
    for f in ("cleaned_code", "code"):
        v = row.get(f, "")
        if isinstance(v, str) and v.strip():
            return v
    return ""


def _kshot_indices(labels, k, n_classes, seed):
    by = defaultdict(list)
    for i, l in enumerate(labels):
        if l >= 0:
            by[int(l)].append(i)
    rng = random.Random(seed)
    out, counts = [], {}
    for cls in range(n_classes):
        pool = by.get(cls, [])
        n_take = min(k, len(pool))
        if n_take > 0:
            out.extend(rng.sample(pool, n_take))
        counts[cls] = n_take
    rng.shuffle(out)
    return out, counts


def _frac_indices(labels, frac, n_classes, seed):
    if not (0 < frac <= 1):
        raise ValueError(frac)
    by = defaultdict(list)
    for i, l in enumerate(labels):
        if l >= 0:
            by[int(l)].append(i)
    rng = random.Random(seed)
    out, counts = [], {}
    for cls in range(n_classes):
        pool = by.get(cls, [])
        n_take = max(1, int(round(len(pool) * frac))) if pool else 0
        if n_take > 0:
            out.extend(rng.sample(pool, n_take))
        counts[cls] = n_take
    rng.shuffle(out)
    return out, counts


def _minival_indices(labels, n_per_class, n_classes, seed):
    by = defaultdict(list)
    for i, l in enumerate(labels):
        if l >= 0:
            by[int(l)].append(i)
    rng = random.Random(seed)
    out = []
    for cls in range(n_classes):
        pool = by.get(cls, [])
        n_take = min(n_per_class, len(pool))
        if n_take > 0:
            out.extend(rng.sample(pool, n_take))
    rng.shuffle(out)
    return out


def _materialise(ds, idxs):
    """Extract features for a subset (CPU-bound but vectorisable)."""
    sub = ds.select(idxs) if idxs else ds.select([])
    X = []
    y = []
    langs = []
    sources = []
    for row in sub:
        X.append(extract_features(row["code"]))
        y.append(int(row["label"]))
        langs.append(row.get("language", ""))
        sources.append(row.get("source", ""))
    return np.asarray(X, dtype=np.float64), np.asarray(y, dtype=np.int64), langs, sources


def _load_codet(seed):
    logger.info("Loading dataset: DaniilOr/CoDET-M4")
    ds = load_dataset("DaniilOr/CoDET-M4", split="train")
    if "split" in ds.column_names:
        train = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        val = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        test = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
    else:
        s1 = ds.train_test_split(test_size=0.1, seed=seed); test = s1["test"]
        s2 = s1["train"].train_test_split(test_size=1 / 9, seed=seed)
        train, val = s2["train"], s2["test"]
    return train, val, test


def _convert(split, vocab):
    return split.map(
        lambda r: {"code": _code(r), "label": _label(r, vocab),
                   "language": str(r.get("language", "")).strip().lower(),
                   "source": str(r.get("source", "")).strip().lower()},
        remove_columns=split.column_names,
    ).filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


# =============================================================================
# Training + evaluation
# =============================================================================

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


def run_one(kind, value, base_seed, val_per_class=64):
    logger.info(f"\n{'=' * 60}\n[{EXP_ID}] {kind}={value} seed={base_seed}\n{'=' * 60}")
    train_raw, val_raw, test_raw = _load_codet(base_seed)
    vocab = _build_vocab(train_raw)
    n_classes = 6
    logger.info(f"Author vocab ({len(vocab)}): {sorted(vocab.keys())}")

    train_ds = _convert(train_raw, vocab)
    val_ds = _convert(val_raw, vocab)
    test_ds = _convert(test_raw, vocab)

    if kind == "kshot":
        idxs, counts = _kshot_indices(list(train_ds["label"]), value, n_classes, base_seed)
    else:
        idxs, counts = _frac_indices(list(train_ds["label"]), value, n_classes, base_seed)
    logger.info(f"[train] {kind}={value} -> per-class={counts} total={sum(counts.values())}")

    val_idxs = _minival_indices(list(val_ds["label"]), val_per_class, n_classes, base_seed + 1000)
    logger.info(f"[val] size={len(val_idxs)}  [test] size={len(test_ds)} (full)")

    t0 = time.time()
    logger.info("Extracting features (train)...")
    Xtr, ytr, _, _ = _materialise(train_ds, idxs)
    logger.info("Extracting features (val)...")
    Xv, yv, _, _ = _materialise(val_ds, val_idxs)
    logger.info("Extracting features (test, full)...")
    Xte, yte, te_langs, te_srcs = _materialise(test_ds, list(range(len(test_ds))))
    feat_t = time.time() - t0
    logger.info(f"[features] train={Xtr.shape} val={Xv.shape} test={Xte.shape} took {feat_t:.1f}s")

    # CatBoost training. Use multiclass loss + class weights inversely proportional to counts.
    class_w = np.bincount(ytr, minlength=n_classes).astype(np.float64)
    class_w = np.maximum(class_w, 1.0)
    class_w = (1.0 / class_w) * (class_w.sum() / n_classes)
    iters = 200 if kind == "kshot" else 500
    clf = CatBoostClassifier(
        iterations=iters,
        learning_rate=0.05,
        depth=6,
        loss_function="MultiClass",
        class_weights=class_w.tolist(),
        random_seed=base_seed,
        verbose=False,
        eval_metric="TotalF1",
        feature_names=FEATURE_NAMES,
    )
    clf.fit(Xtr, ytr, eval_set=(Xv, yv), use_best_model=True, early_stopping_rounds=20, verbose=False)

    val_pred = clf.predict(Xv).reshape(-1).astype(int)
    val_macro = float(f1_score(yv, val_pred, average="macro", zero_division=0))

    test_pred = clf.predict(Xte).reshape(-1).astype(int)
    test_macro = float(f1_score(yte, test_pred, average="macro", zero_division=0))
    test_weighted = float(f1_score(yte, test_pred, average="weighted", zero_division=0))
    test_acc = float(accuracy_score(yte, test_pred))
    gap = val_macro - test_macro
    wall = time.time() - t0
    logger.info(f"[FINAL] test_macro={test_macro:.4f} weighted={test_weighted:.4f} "
                f"acc={test_acc:.4f} val={val_macro:.4f} gap={gap:+.4f} wall={wall:.1f}s")

    return {
        "kind": kind, "value": value, "test_macro_f1": test_macro,
        "test_weighted_f1": test_weighted, "test_accuracy": test_acc,
        "val_macro_f1": val_macro, "val_test_gap": gap,
        "wall_time_s": wall, "n_train": int(sum(counts.values())),
        "per_class_f1": _per_class(test_pred.tolist(), yte.tolist(), n_classes),
        "per_lang_f1": _per_subgroup(test_pred.tolist(), yte.tolist(), te_langs),
        "per_source_f1": _per_subgroup(test_pred.tolist(), yte.tolist(), te_srcs),
        "train_per_class": counts,
    }


def _save_json(result, base_seed):
    kind, value = result["kind"], result["value"]
    label = f"K{value}" if kind == "kshot" else f"frac{value:.4f}".rstrip("0").rstrip(".")
    payload = {
        "method": METHOD_NAME, "exp_id": EXP_ID,
        "regime": f"K={value}" if kind == "kshot" else f"frac={value:.4f}",
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "config": {"benchmark": "codet_m4", "task": "author",
                   "k_shot": value if kind == "kshot" else 0,
                   "train_fraction": value if kind == "fraction" else 0.0,
                   "n_classes": 6, "fs_seed": base_seed,
                   "encoder": "catboost-handcrafted-features",
                   "n_features": len(FEATURE_NAMES),
                   "features": FEATURE_NAMES},
        "results": result,
    }
    candidate_dirs = []
    if os.name == "posix" and os.path.isdir("/kaggle/working"):
        candidate_dirs.append("/kaggle/working/results")
    candidate_dirs.append("./results")
    for d in candidate_dirs:
        try:
            os.makedirs(d, exist_ok=True)
            path = os.path.join(d, f"{EXP_ID}_{label}_seed{base_seed}.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, ensure_ascii=False, default=str)
            logger.info(f"[json] -> {path}")
            return
        except (OSError, PermissionError) as e:
            logger.warning(f"[json] {d}: {e}")


def parse_sweep_configs():
    raw_ks = os.environ.get("FS_SWEEP_KS", "32,128").strip()
    raw_fr = os.environ.get("FS_SWEEP_FRACS", "0.01,0.05").strip()
    out = []
    if raw_ks:
        for x in raw_ks.split(","):
            if x.strip():
                out.append(("kshot", int(x.strip())))
    if raw_fr:
        for x in raw_fr.split(","):
            if x.strip():
                out.append(("fraction", float(x.strip())))
    return out


def main():
    base_seed = int(os.environ.get("FS_SEED", "42"))
    configs = parse_sweep_configs()
    logger.info(f"[{EXP_ID}] CatBoost sweep={configs} seed={base_seed}")
    summary = []
    for kind, value in configs:
        r = run_one(kind, value, base_seed)
        _save_json(r, base_seed)
        summary.append((kind, value, r["test_macro_f1"], r["val_macro_f1"], r["wall_time_s"]))

    print(f"\n{'=' * 70}\n[{EXP_ID}] SWEEP SUMMARY -- {METHOD_NAME}\n{'=' * 70}")
    print(f"{'config':<16}{'test_F1':>10}{'val_F1':>10}{'gap':>10}{'wall':>10}")
    print("-" * 56)
    for kind, value, t, v, w in summary:
        label = f"K={value}" if kind == "kshot" else f"frac={value:.4f}"
        print(f"{label:<16}{t:>10.4f}{v:>10.4f}{(v - t):>+10.4f}{w:>10.0f}")
    print("-" * 56)
    print(f"{len(summary)} configs. Paper full-data CatBoost = 0.4542 Macro-F1.")


if __name__ == "__main__":
    main()
