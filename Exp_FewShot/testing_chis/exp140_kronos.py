# exp140 — KRONOS
# =============================================================================
# NAME       : KRONOS (Multi-scale information-theoretic complexity profile)
# REFERENCE  : Shannon 1948 ("A Mathematical Theory of Communication"),
#              Bandt & Pompe 2002 ("Permutation entropy: a natural
#              complexity measure for time series", PRL 88:174102),
#              Richman & Moorman 2000 ("Sample entropy"). Never applied to
#              code authorship.
# CLAIM      : Code is a 1D symbolic time series. Each author has a unique
#              COMPLEXITY PROFILE — a vector of multi-scale entropies that
#              measures different aspects of regularity vs randomness:
#              (a) Shannon entropy of character distribution
#              (b) Conditional entropy at lag 1, 2, 3
#              (c) Permutation entropy at order 3, 4, 5 (Bandt-Pompe)
#              (d) Sample entropy at scale 1 (Richman-Moorman)
#              Attribute by argmin Euclidean distance to per-class mean
#              profile. Pure information theory, zero learning.
# EQUATION   : profile(x) = [
#                  H(x), H(x_t|x_{t-1}), H(x_t|x_{t-2}), H(x_t|x_{t-3}),
#                  H_perm(x, order=3), H_perm(x, order=4), H_perm(x, order=5),
#                  H_sample(x, m=2, r=0.2*std)
#              ]  in R^8
#              y_hat = argmin_a Euclidean(profile(x), mean_profile_a)
# WHY NEW    : Permutation entropy (Bandt-Pompe 2002) was designed for
#              ECG/EEG signals; never applied to code. Sample entropy is
#              used in physiological complexity analysis; never applied
#              to AI code attribution. We bring CHAOS THEORY tools to
#              code attribution. Pure complexity-theoretic features.
# WOW HOOK   : "We attribute LLM-generated code with tools from CHAOS
#              THEORY. Permutation entropy was invented for EEG analysis
#              in 2002. Sample entropy for heart-rate variability in
#              2000. The same complexity measures that distinguish a
#              healthy heart from a sick one distinguish GPT-4 from Llama."
# FALSIFIER  : (F1) Per-feature ANOVA: each of the 8 features must have
#              cross-class F-statistic > 5 (statistically discriminative).
#              (F2) KRONOS composite > simple Shannon-only baseline by
#              >= 0.01 (multi-scale matters).
#              (F3) Permutation + Sample entropy together contribute >= 0.005
#              (chaos-theory features ARE doing work, not just Shannon).
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
import multiprocessing as mp
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("scipy"); _ensure("datasets"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
from scipy import stats
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp140_kronos")

PAPER_BASELINE = 0.6633
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD  = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}


def _gene_distance(u, v, adj):
    if u == v: return 0.0
    queue = [(u, 0)]; visited = {u}
    while queue:
        curr, d = queue.pop(0)
        for nb in adj.get(curr, []):
            if nb == v: return d + 1.0
            if nb not in visited:
                visited.add(nb); queue.append((nb, d + 1))
    return float("inf")


def build_distance_matrix(n_cls, adj, default_dist=4.0):
    D = np.full((n_cls, n_cls), default_dist)
    for i in range(n_cls):
        for j in range(n_cls):
            d = _gene_distance(i, j, adj)
            if d < float("inf"): D[i, j] = d
            elif (i == 0) != (j == 0): D[i, j] = 3.0
    return D


def build_sibling_mask(n_cls, adj):
    M = np.zeros((n_cls, n_cls))
    for i in range(n_cls):
        for j in adj.get(i, []): M[i, j] = 1.0
    return M


# =============================================================================
# Data loading
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD  = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


def _is_human(t):
    return str(t or "").strip().lower() in {"human", "human_written", "human-generated"}


def _vocab(train):
    names = {str(r.get("model", "") or "").strip() for r in train
             if not _is_human(r.get("target", "")) and r.get("model", "")}
    return {n: i + 1 for i, n in enumerate(sorted(names))}


def _conv_codet(split, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code", "code"):
            v = r.get(f, "")
            if isinstance(v, str) and v.strip(): code = v; break
        label = 0 if _is_human(r.get("target", "")) else vocab.get(
            str(r.get("model", "") or "").strip(), -1)
        return {"code": code, "label": label,
                "language": str(r.get("language", "")).strip().lower(),
                "source":   str(r.get("source",   "")).strip().lower()}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _conv_aicd(split):
    def row(r):
        return {"code":     str(r.get("code",     "")).strip(),
                "label":    int(r.get("label",    -1)),
                "language": str(r.get("language", "")).strip().lower(),
                "source":   ""}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _load_codet():
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        return tr, vl, ts
    s  = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


def _load_aicd(task):
    task_name = {"t1": "T1", "t2": "T2", "t3": "T3"}.get(task.lower())
    if task_name is None: raise ValueError(f"[aicd] Unknown task '{task}'")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path): raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found")
    pf = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: No parquet files")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0: return tr, vl, ts
    s  = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


def stratified_subsample(data, frac_or_n_per_class, seed=42):
    rng = random.Random(seed)
    labels = list(range(max(data["label"]) + 1))
    keep = []
    for lbl in labels:
        idx = [i for i, x in enumerate(data["label"]) if x == lbl]
        if frac_or_n_per_class < 1.0:
            n = min(max(1, int(len(idx) * frac_or_n_per_class)), len(idx))
        else:
            n = min(int(frac_or_n_per_class), len(idx))
        keep.extend(rng.sample(idx, n))
    return data.select(keep)


# =============================================================================
# Information-theoretic features
# =============================================================================

MAX_CHARS_SHANNON = 5000
MAX_CHARS_PERM = 5000
MAX_CHARS_SAMPEN = 1500
FEATURE_NAMES = [
    "shannon", "cond_lag1", "cond_lag2", "cond_lag3",
    "perm3", "perm4", "perm5", "sample_entropy",
]
N_FEATURES = len(FEATURE_NAMES)


def shannon_entropy(text):
    if not text: return 0.0
    counts = Counter(text)
    total = sum(counts.values())
    if total == 0: return 0.0
    p = np.array([c / total for c in counts.values()], dtype=np.float64)
    return float(-np.sum(p * np.log2(p + 1e-12)))


def conditional_entropy_lag(text, lag=1):
    """H(X_t | X_{t-lag}) for the character sequence."""
    if len(text) <= lag: return 0.0
    bigrams = Counter()
    unigrams = Counter()
    for t in range(lag, len(text)):
        bigrams[(text[t - lag], text[t])] += 1
        unigrams[text[t - lag]] += 1
    total = sum(unigrams.values())
    if total == 0: return 0.0
    # Group bigrams by their first symbol
    grouped: Dict[str, Dict[str, int]] = {}
    for (x, y), c in bigrams.items():
        grouped.setdefault(x, {})[y] = c
    h = 0.0
    for x, count_x in unigrams.items():
        p_x = count_x / total
        y_given_x = grouped.get(x, {})
        sub_total = sum(y_given_x.values())
        if sub_total == 0: continue
        for y, count_xy in y_given_x.items():
            p_y_given_x = count_xy / sub_total
            if p_y_given_x > 0:
                h -= p_x * p_y_given_x * math.log2(p_y_given_x + 1e-12)
    return float(h)


def permutation_entropy(text, order=3):
    """Bandt-Pompe permutation entropy on ordinal patterns of char-ASCII series."""
    text = text[:MAX_CHARS_PERM]
    if len(text) < order + 1: return 0.0
    arr = np.array([ord(c) for c in text], dtype=np.int32)
    n_patterns_max = math.factorial(order)
    pattern_counts = Counter()
    for t in range(len(arr) - order + 1):
        window = arr[t:t + order]
        ranks = tuple(np.argsort(window))
        pattern_counts[ranks] += 1
    total = sum(pattern_counts.values())
    if total == 0: return 0.0
    p = np.array([c / total for c in pattern_counts.values()], dtype=np.float64)
    h = float(-np.sum(p * np.log2(p + 1e-12)))
    norm = math.log2(n_patterns_max) if n_patterns_max > 1 else 1.0
    return h / norm  # normalised to [0, 1]


def sample_entropy(text, m=2, r_factor=0.2):
    """Richman-Moorman sample entropy on capped char-ASCII series."""
    text = text[:MAX_CHARS_SAMPEN]
    if len(text) < m + 2: return 0.0
    arr = np.array([ord(c) for c in text], dtype=np.float32)
    N = len(arr)
    if N < m + 2: return 0.0
    sd = float(arr.std())
    if sd < 1e-6: return 0.0
    r = r_factor * sd

    def _count_matches(mm):
        # Build templates of length mm
        n_tpl = N - mm + 1
        if n_tpl < 2: return 0
        # Vectorised Chebyshev: for each i, compare against j > i in a window
        templates = np.lib.stride_tricks.sliding_window_view(arr, mm)
        count = 0
        # Pairwise loop, vectorised over j for each i
        for i in range(n_tpl - 1):
            diffs = np.abs(templates[i + 1:] - templates[i])
            cheb = diffs.max(axis=1) if mm > 1 else diffs.ravel()
            count += int(np.sum(cheb <= r))
        return count

    B = _count_matches(m)
    A = _count_matches(m + 1)
    if B <= 0 or A <= 0: return 0.0
    return float(-math.log((A + 1e-12) / (B + 1e-12)))


def kronos_profile(code):
    """Compute the 8-dim complexity profile of one code sample."""
    t = code[:MAX_CHARS_SHANNON]
    h_shannon = shannon_entropy(t)
    h_lag1 = conditional_entropy_lag(t, lag=1)
    h_lag2 = conditional_entropy_lag(t, lag=2)
    h_lag3 = conditional_entropy_lag(t, lag=3)
    h_perm3 = permutation_entropy(code, order=3)
    h_perm4 = permutation_entropy(code, order=4)
    h_perm5 = permutation_entropy(code, order=5)
    h_samp = sample_entropy(code, m=2)
    return np.array(
        [h_shannon, h_lag1, h_lag2, h_lag3, h_perm3, h_perm4, h_perm5, h_samp],
        dtype=np.float32,
    )


def _encode_worker(code):
    try:
        return kronos_profile(code)
    except Exception:
        return np.zeros(N_FEATURES, dtype=np.float32)


def encode_all(codes, n_workers=None, desc="encode"):
    if n_workers is None or n_workers <= 1:
        return np.stack([_encode_worker(c) for c in tqdm(codes, desc=desc)])
    try:
        with mp.Pool(n_workers) as pool:
            out = list(tqdm(pool.imap(_encode_worker, codes, chunksize=8),
                             total=len(codes), desc=desc))
        return np.stack(out)
    except Exception as e:
        logger.warning(f"[kronos] mp failed ({e}); serial fallback")
        return np.stack([_encode_worker(c) for c in tqdm(codes, desc=desc)])


# =============================================================================
# Nearest-centroid classifier with feature standardisation
# =============================================================================

def standardise_fit(X):
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd < 1e-9] = 1.0
    return mu, sd


def standardise_apply(X, mu, sd):
    return (X - mu) / sd


def per_class_centroids(X, y, n_cls):
    d = X.shape[1]
    cents = np.zeros((n_cls, d), dtype=np.float32)
    counts = np.zeros(n_cls, dtype=np.int64)
    for x, lbl in zip(X, y):
        cents[lbl] += x
        counts[lbl] += 1
    for c in range(n_cls):
        if counts[c] > 0:
            cents[c] /= counts[c]
    return cents, counts


def predict_centroid(X, centroids, valid_classes=None):
    n_q = X.shape[0]; n_c = centroids.shape[0]
    preds = np.zeros(n_q, dtype=np.int64)
    for i in range(n_q):
        diffs = centroids - X[i:i+1]
        dists = np.linalg.norm(diffs, axis=1)
        if valid_classes is not None:
            mask = np.full(n_c, np.inf)
            mask[list(valid_classes)] = 0.0
            dists = dists + mask
        preds[i] = int(np.argmin(dists))
    return preds


def per_feature_anova(X, y, n_cls):
    """Per-feature one-way ANOVA F-statistic. Returns list of F-stats."""
    fstats = []
    for k in range(X.shape[1]):
        groups = [X[y == c, k] for c in range(n_cls) if (y == c).sum() >= 2]
        if len(groups) < 2:
            fstats.append(0.0); continue
        try:
            f, _ = stats.f_oneway(*groups)
            fstats.append(float(f if np.isfinite(f) else 0.0))
        except Exception:
            fstats.append(0.0)
    return fstats


# =============================================================================
# Cfg
# =============================================================================

@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    frac: float = 0.20
    n_cls: int = 6
    seed: int = 42
    train_cap_per_class: int = 200
    test_cap_per_class: int = 200
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = 5000


def set_seed(s):
    random.seed(s); np.random.seed(s)


# =============================================================================
# Eval pack
# =============================================================================

def eval_pack(preds, labels, langs, sources, n_cls, sib_mask, dist_mat):
    preds = np.asarray(preds); labels = np.asarray(labels)
    langs = np.asarray(langs); sources = np.asarray(sources)
    overall = {
        "accuracy":        float(accuracy_score(labels, preds)),
        "macro_f1":        float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "weighted_f1":     float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "micro_f1":        float(f1_score(labels, preds, average="micro",    zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(labels, preds, average="macro",    zero_division=0)),
    }
    per_class = {
        "f1":        f1_score(labels, preds, average=None, zero_division=0,
                              labels=list(range(n_cls))).tolist(),
        "precision": precision_score(labels, preds, average=None, zero_division=0,
                                     labels=list(range(n_cls))).tolist(),
        "recall":    recall_score(labels, preds, average=None, zero_division=0,
                                  labels=list(range(n_cls))).tolist(),
    }
    cm       = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    off_diag = int(cm.sum() - cm.trace())
    sib_conf = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls)
                       if i != j and sib_mask[i, j] > 0))
    sib_rate = sib_conf / max(off_diag, 1)
    cross = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls)
                    if i != j and dist_mat[i, j] >= 3.0))
    cross_rate = cross / max(off_diag, 1)
    per_lang, per_src = {}, {}
    if langs.size > 0 and any(l for l in langs.tolist()):
        for L in sorted(set(langs.tolist())):
            if not L: continue
            sel = (langs == L)
            if sel.sum() < 2: continue
            per_lang[L] = {
                "n":           int(sel.sum()),
                "macro_f1":    float(f1_score(labels[sel], preds[sel], average="macro",    zero_division=0)),
                "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                "accuracy":    float(accuracy_score(labels[sel], preds[sel])),
            }
    if sources.size > 0 and any(s for s in sources.tolist()):
        for S in sorted(set(sources.tolist())):
            if not S: continue
            sel = (sources == S)
            if sel.sum() < 2: continue
            per_src[S] = {
                "n":           int(sel.sum()),
                "macro_f1":    float(f1_score(labels[sel], preds[sel], average="macro",    zero_division=0)),
                "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                "accuracy":    float(accuracy_score(labels[sel], preds[sel])),
            }
    return {
        "overall":                     overall,
        "per_class":                   per_class,
        "per_language":                per_lang,
        "per_source":                  per_src,
        "confusion_matrix":            cm.tolist(),
        "sibling_confusion_rate":      float(sib_rate),
        "cross_family_confusion_rate": float(cross_rate),
        "off_diag_total":              off_diag,
        "n_samples":                   int(len(labels)),
    }


# =============================================================================
# run_exp
# =============================================================================

def _fit_predict(X_tr, y_tr, X_te, n_cls, valid_cls):
    """Standardise on train stats, build centroids, predict on test."""
    mu, sd = standardise_fit(X_tr)
    Xtr_s = standardise_apply(X_tr, mu, sd)
    Xte_s = standardise_apply(X_te, mu, sd)
    cents, _ = per_class_centroids(Xtr_s, y_tr, n_cls)
    preds = predict_centroid(Xte_s, cents, valid_classes=valid_cls)
    return preds


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    sib_mask = build_sibling_mask(cfg.n_cls, cfg.gene_adj)

    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, vocab)
        vl_data = _conv_codet(vl_raw, vocab)
        ts_data = _conv_codet(ts_raw, vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)

    cfg.n_cls = max(tr_data["label"]) + 1
    dist_mat = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    sib_mask = build_sibling_mask(cfg.n_cls, cfg.gene_adj)

    tr_data_frac = stratified_subsample(tr_data, cfg.frac, seed=cfg.seed)
    tr_data_capped = stratified_subsample(tr_data_frac, cfg.train_cap_per_class, seed=cfg.seed + 1)
    ts_data_capped = stratified_subsample(ts_data, cfg.test_cap_per_class, seed=cfg.seed + 2)
    vl_data_capped = stratified_subsample(vl_data, cfg.test_cap_per_class // 2, seed=cfg.seed + 3)

    train_codes = [r["code"][:cfg.max_chars] for r in tr_data_capped]
    train_labels = list(tr_data_capped["label"])
    val_codes = [r["code"][:cfg.max_chars] for r in vl_data_capped]
    val_labels = list(vl_data_capped["label"])
    val_langs = [r.get("language", "") or "" for r in vl_data_capped]
    val_sources = [r.get("source", "") or "" for r in vl_data_capped]
    test_codes = [r["code"][:cfg.max_chars] for r in ts_data_capped]
    test_labels = list(ts_data_capped["label"])
    test_langs = [r.get("language", "") or "" for r in ts_data_capped]
    test_sources = [r.get("source", "") or "" for r in ts_data_capped]

    n_workers = cfg.n_workers if cfg.n_workers > 0 else max(1, mp.cpu_count() - 1)
    logger.info(f"[setup] frac={cfg.frac} train={len(train_codes)} val={len(val_codes)} "
                f"test={len(test_codes)} n_cls={cfg.n_cls} workers={n_workers}")

    t_enc0 = time.time()
    X_train = encode_all(train_codes, n_workers=n_workers, desc="encode-train")
    X_val   = encode_all(val_codes,   n_workers=n_workers, desc="encode-val")
    X_test  = encode_all(test_codes,  n_workers=n_workers, desc="encode-test")
    enc_time = time.time() - t_enc0
    logger.info(f"[encode] done in {enc_time:.0f}s shape_train={X_train.shape}")

    y_train = np.array(train_labels, dtype=np.int64)
    y_val   = np.array(val_labels,   dtype=np.int64)
    y_test  = np.array(test_labels,  dtype=np.int64)
    valid_cls = [c for c in range(cfg.n_cls) if (y_train == c).sum() > 0]

    # --- Full KRONOS (8 features) ---
    t0 = time.time()
    val_preds = _fit_predict(X_train, y_train, X_val, cfg.n_cls, valid_cls)
    val_time = time.time() - t0
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                         cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]

    t0 = time.time()
    test_preds = _fit_predict(X_train, y_train, X_test, cfg.n_cls, valid_cls)
    test_time = time.time() - t0
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro

    # --- Falsifier F1: per-feature ANOVA on training features ---
    fstats = per_feature_anova(X_train, y_train, cfg.n_cls)
    fstat_dict = dict(zip(FEATURE_NAMES, fstats))
    all_above_5 = all(f > 5.0 for f in fstats)

    # --- Falsifier F2: Shannon-only baseline ---
    X_tr_sh = X_train[:, 0:1]; X_te_sh = X_test[:, 0:1]
    pred_sh = _fit_predict(X_tr_sh, y_train, X_te_sh, cfg.n_cls, valid_cls)
    shannon_only_macro = float(f1_score(y_test, pred_sh, average="macro", zero_division=0))
    f2_delta = test_macro - shannon_only_macro

    # --- Falsifier F3: Permutation + Sample entropy ONLY (chaos features alone) ---
    # Indices: perm3=4, perm4=5, perm5=6, sample=7
    chaos_idx = [4, 5, 6, 7]
    X_tr_ch = X_train[:, chaos_idx]; X_te_ch = X_test[:, chaos_idx]
    pred_ch = _fit_predict(X_tr_ch, y_train, X_te_ch, cfg.n_cls, valid_cls)
    chaos_only_macro = float(f1_score(y_test, pred_ch, average="macro", zero_division=0))
    # F3 phrased as "Permutation + Sample contribute >= 0.005" -- measured as
    # (full - shannon+cond) where chaos features are ablated OUT
    shannon_cond_idx = [0, 1, 2, 3]
    X_tr_sc = X_train[:, shannon_cond_idx]; X_te_sc = X_test[:, shannon_cond_idx]
    pred_sc = _fit_predict(X_tr_sc, y_train, X_te_sc, cfg.n_cls, valid_cls)
    shannon_cond_only_macro = float(f1_score(y_test, pred_sc, average="macro", zero_division=0))
    f3_chaos_contribution = test_macro - shannon_cond_only_macro

    logger.info(f"[val] macro={val_macro:.4f} time={val_time:.0f}s")
    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} time={test_time:.0f}s")
    logger.info(f"[F1] per-feature F-stats: {fstat_dict} all>5? {all_above_5}")
    logger.info(f"[F2] Shannon-only={shannon_only_macro:.4f} full={test_macro:.4f} "
                f"delta={f2_delta:+.4f}")
    logger.info(f"[F3] chaos-only={chaos_only_macro:.4f} shannon+cond-only={shannon_cond_only_macro:.4f} "
                f"chaos_contribution={f3_chaos_contribution:+.4f}")

    ts_met["falsifier_F1_per_feature_f_stat"] = fstat_dict
    ts_met["falsifier_F1_all_features_above_5"] = bool(all_above_5)
    ts_met["falsifier_F2_shannon_only_macro"] = shannon_only_macro
    ts_met["falsifier_F2_delta"] = f2_delta
    ts_met["falsifier_F3_chaos_only_macro"] = chaos_only_macro
    ts_met["falsifier_F3_shannon_cond_only_macro"] = shannon_cond_only_macro
    ts_met["falsifier_F3_chaos_contribution"] = f3_chaos_contribution

    return {
        "tag": tag, "method": "KRONOS",
        "note": ("Multi-scale information-theoretic complexity profile "
                 "(Shannon + cond. lag1-3 + Bandt-Pompe perm3-5 + sample entropy). "
                 "Standardised features + nearest-centroid. CPU-only, zero learning."),
        "enc": "kronos-profile-8d", "bench": cfg.benchmark,
        "frac": cfg.frac,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "per_feature_f_stat": fstat_dict,
        "all_features_above_5": bool(all_above_5),
        "shannon_only_macro": shannon_only_macro,
        "f2_delta": f2_delta,
        "chaos_only_macro": chaos_only_macro,
        "shannon_cond_only_macro": shannon_cond_only_macro,
        "chaos_contribution": f3_chaos_contribution,
        "encode_time_sec": enc_time,
        "val_time_sec": val_time, "test_time_sec": test_time,
        "test_metrics": ts_met,
        "val_history": [val_macro],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in benchmarks:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp140_kronos_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"chaos_contrib={res['chaos_contribution']:+.4f}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp140_kronos_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Train':>8} {'Test':>8} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'Shannon':>9} {'Chaos':>9} {'+Chaos':>9} {'Wall':>8}")
    print("-"*140)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['train_size_after_cap']:>8d} {r['test_size_after_cap']:>8d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['shannon_only_macro']:>9.4f} "
              f"{r['chaos_only_macro']:>9.4f} {r['chaos_contribution']:>+9.4f} "
              f"{r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
