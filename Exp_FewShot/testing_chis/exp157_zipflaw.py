# exp157 - ZIPFLAW
# =============================================================================
# NAME       : ZIPFLAW (Statistical-laws stylometry combining Zipf, Heaps,
#              Yule, Brunet, Simpson, Sichel per-sample into a 6/7-d author
#              signature; nearest-centroid classification).
# REFERENCE  : Zipf (1949), Heaps (1978), Yule (1944), Brunet (1978),
#              Simpson (1949), Sichel (1975). Burrows (2002) "Delta" method
#              and Mosteller & Wallace (1964) Federalist Papers attribution
#              are the prior art on classical forensic stylometry.
# CLAIM      : Six classical "laws of language" yield a low-dimensional
#              signature per code sample. LLMs follow each law DIFFERENTLY
#              enough to be distinguished by simple nearest-centroid in
#              this 7-d space.
# EQUATION   : tokens     = re.findall(r"\w+|[^\w\s]", code)
#              f_zipf     = - slope(log freq vs log rank)
#              f_heaps    = (K, beta) such that V(N) = K * N^beta
#              f_yule     = 10^4 * (sum_i i^2 V_i - N) / N^2
#              f_brunet   = N^{V^{-0.172}}
#              f_simpson  = sum_i V_i * i * (i-1) / (N * (N-1))
#              f_sichel   = V_2 / V_1
#              phi(x)     = z-score(zipf, K_h, beta_h, yule, brunet,
#                                   simpson, sichel)
#              y_hat      = argmin_c || phi(query) - mu_c ||_2
# WHY NEW    : Burrows-Mosteller-Wallace stylometry has been applied to
#              human-written documents for decades, but the assumption that
#              LLMs follow Zipf / Heaps / Yule differently from each other
#              has never been empirically tested on AI-generated code.
# WOW HOOK   : "Zipf, Heaps, Yule, Brunet, Simpson, Sichel: six statistical
#              laws of language from a century of forensic linguistics. We
#              pack them into a 7-d author signature and ask: do LLMs
#              follow these laws DIFFERENTLY enough to be told apart?"
# FALSIFIER  : (F1) Per-class mean Zipf slopes must have spread >= 0.05
#              across classes (else Zipf alone is uninformative).
#              (F2) The full 7-d combo beats best-single-feature by
#              >= 0.005.
#              (F3) Headline macro exceeds random (1/n_cls) by >= 0.05.
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, re, math
import multiprocessing as mp
from dataclasses import dataclass, field
from collections import Counter
from typing import List, Dict, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


_ensure("numpy"); _ensure("scipy"); _ensure("scikit-learn")
_ensure("datasets"); _ensure("tqdm"); _ensure("pandas")

import numpy as np
from scipy.stats import linregress
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp157_zipflaw")

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
# Data loading (identical plumbing to exp145)
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
# Statistical-laws feature extraction (7-d)
# =============================================================================

FEATURE_NAMES = ["zipf_slope", "heaps_K", "heaps_beta",
                 "yule_K", "brunet_W", "simpson_D", "sichel_S"]
N_FEATURES = len(FEATURE_NAMES)


def _tokenize(code):
    return re.findall(r"\w+|[^\w\s]", code)


def _zipf_slope(freqs_sorted_desc):
    if len(freqs_sorted_desc) < 3:
        return 0.0
    ranks = np.arange(1, len(freqs_sorted_desc) + 1, dtype=np.float64)
    fr = np.asarray(freqs_sorted_desc, dtype=np.float64)
    lr = np.log(ranks)
    lf = np.log(np.maximum(fr, 1.0))
    if lr.std() < 1e-12:
        return 0.0
    try:
        res = linregress(lr, lf)
        return float(-res.slope)
    except Exception:
        return 0.0


def _heaps_fit(tokens, n_points=20):
    """Sweep token counts, count vocab, fit log V = log K + beta log N."""
    N_total = len(tokens)
    if N_total < 10:
        return 0.0, 0.0
    points = np.unique(np.geomspace(2, N_total, num=min(n_points, N_total - 1)).astype(int))
    points = points[points >= 2]
    if len(points) < 2:
        return 0.0, 0.0
    Ns, Vs = [], []
    seen = set()
    last = 0
    vocab_count = 0
    for n in points:
        while last < n and last < N_total:
            t = tokens[last]
            if t not in seen:
                seen.add(t); vocab_count += 1
            last += 1
        Ns.append(n); Vs.append(vocab_count)
    Ns = np.asarray(Ns, dtype=np.float64)
    Vs = np.asarray(Vs, dtype=np.float64)
    Vs = np.maximum(Vs, 1.0)
    lN, lV = np.log(Ns), np.log(Vs)
    if lN.std() < 1e-12:
        return 0.0, 0.0
    try:
        res = linregress(lN, lV)
        beta = float(res.slope)
        K = float(math.exp(res.intercept))
        return K, beta
    except Exception:
        return 0.0, 0.0


def _yule_K(freq_counts):
    """Yule's K = 10^4 * (sum_i i^2 V_i - N) / N^2 where V_i = #tokens with freq i."""
    if not freq_counts:
        return 0.0
    N = sum(i * v for i, v in freq_counts.items())
    if N <= 1:
        return 0.0
    S = sum((i * i) * v for i, v in freq_counts.items())
    return float(10000.0 * (S - N) / (N * N))


def _brunet_W(N, V, a=0.172):
    if V <= 1 or N <= 1:
        return 0.0
    try:
        return float(N ** (V ** (-a)))
    except Exception:
        return 0.0


def _simpson_D(freq_counts, N):
    if N <= 1:
        return 0.0
    s = sum(v * i * (i - 1) for i, v in freq_counts.items())
    return float(s / (N * (N - 1)))


def _sichel_S(freq_counts):
    V1 = freq_counts.get(1, 0)
    V2 = freq_counts.get(2, 0)
    if V1 == 0:
        return 0.0
    return float(V2 / V1)


def extract_features(code):
    """Return 7-d (zipf_slope, heaps_K, heaps_beta, yule_K, brunet_W, simpson_D, sichel_S)."""
    code = code[:8000]
    tokens = _tokenize(code)
    N = len(tokens)
    if N < 5:
        return np.zeros(N_FEATURES, dtype=np.float32)
    cnt = Counter(tokens)
    V = len(cnt)
    freqs_sorted_desc = sorted(cnt.values(), reverse=True)
    freq_counts = Counter(cnt.values())  # i -> V_i

    f_zipf = _zipf_slope(freqs_sorted_desc)
    K_h, beta_h = _heaps_fit(tokens)
    f_yule = _yule_K(freq_counts)
    f_brunet = _brunet_W(N, V)
    f_simpson = _simpson_D(freq_counts, N)
    f_sichel = _sichel_S(freq_counts)
    feats = [f_zipf, K_h, beta_h, f_yule, f_brunet, f_simpson, f_sichel]
    return np.array(feats, dtype=np.float32)


def _extract_one(code):
    try:
        return extract_features(code)
    except Exception:
        return np.zeros(N_FEATURES, dtype=np.float32)


def extract_features_parallel(codes, n_workers=None, desc="feat"):
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if n_workers == 1 or len(codes) < 64:
        feats = [_extract_one(c) for c in tqdm(codes, desc=desc)]
        return np.stack(feats, axis=0).astype(np.float32)
    try:
        with mp.Pool(n_workers) as pool:
            feats = list(tqdm(pool.imap(_extract_one, codes, chunksize=32),
                              total=len(codes), desc=desc))
        return np.stack(feats, axis=0).astype(np.float32)
    except Exception as e:
        logger.warning(f"[feat] multiprocessing failed ({e}); falling back to serial")
        feats = [_extract_one(c) for c in tqdm(codes, desc=desc)]
        return np.stack(feats, axis=0).astype(np.float32)


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
    train_cap_per_class: int = 300
    test_cap_per_class: int = 400
    val_cap_per_class: int = 200
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = 4000


def set_seed(s):
    random.seed(s); np.random.seed(s)


# =============================================================================
# Eval pack (identical to exp145)
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
# Nearest-centroid in z-scored feature space
# =============================================================================

def _nc_predict(X_query, mus):
    K = mus.shape[0]
    Xq = X_query.astype(np.float64)
    d2 = np.zeros((Xq.shape[0], K), dtype=np.float64)
    for c in range(K):
        diff = Xq - mus[c]
        d2[:, c] = (diff * diff).sum(axis=1)
    return d2.argmin(axis=1)


def _centroids(X, y, n_cls):
    mus = np.zeros((n_cls, X.shape[1]), dtype=np.float64)
    for c in range(n_cls):
        sel = (y == c)
        if sel.sum() > 0:
            mus[c] = X[sel].mean(axis=0)
    return mus


# =============================================================================
# run_exp
# =============================================================================

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
    vl_data_capped = stratified_subsample(vl_data, cfg.val_cap_per_class, seed=cfg.seed + 3)

    train_codes  = [r["code"][:cfg.max_chars] for r in tr_data_capped]
    train_labels = np.array(tr_data_capped["label"], dtype=np.int64)
    val_codes    = [r["code"][:cfg.max_chars] for r in vl_data_capped]
    val_labels   = np.array(vl_data_capped["label"], dtype=np.int64)
    val_langs    = [r.get("language", "") or "" for r in vl_data_capped]
    val_sources  = [r.get("source",   "") or "" for r in vl_data_capped]
    test_codes   = [r["code"][:cfg.max_chars] for r in ts_data_capped]
    test_labels  = np.array(ts_data_capped["label"], dtype=np.int64)
    test_langs   = [r.get("language", "") or "" for r in ts_data_capped]
    test_sources = [r.get("source",   "") or "" for r in ts_data_capped]

    n_workers = cfg.n_workers if cfg.n_workers > 0 else max(1, mp.cpu_count() - 1)
    logger.info(f"[setup] frac={cfg.frac} train={len(train_codes)} val={len(val_codes)} "
                f"test={len(test_codes)} workers={n_workers} d_feat={N_FEATURES}")

    t0 = time.time()
    X_train_raw = extract_features_parallel(train_codes, n_workers=n_workers, desc="phi(train)")
    X_val_raw   = extract_features_parallel(val_codes,   n_workers=n_workers, desc="phi(val)")
    X_test_raw  = extract_features_parallel(test_codes,  n_workers=n_workers, desc="phi(test)")
    feat_time = time.time() - t0
    for X in (X_train_raw, X_val_raw, X_test_raw):
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"[feat] {X_train_raw.shape[0]+X_val_raw.shape[0]+X_test_raw.shape[0]} "
                f"vectors d={N_FEATURES} in {feat_time:.0f}s")

    # Standardise per feature on train
    mu_tr = X_train_raw.mean(axis=0)
    sd_tr = X_train_raw.std(axis=0) + 1e-8
    Xtr = (X_train_raw - mu_tr) / sd_tr
    Xvl = (X_val_raw   - mu_tr) / sd_tr
    Xts = (X_test_raw  - mu_tr) / sd_tr

    # Full 7-d nearest-centroid (headline)
    mus = _centroids(Xtr, train_labels, cfg.n_cls)
    val_preds = _nc_predict(Xvl, mus)
    val_macro = float(f1_score(val_labels, val_preds, average="macro", zero_division=0))
    test_preds = _nc_predict(Xts, mus)
    test_macro = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))
    gap = val_macro - test_macro
    logger.info(f"[head] val={val_macro:.4f} test={test_macro:.4f} gap={gap:+.4f}")

    # F1: per-class zipf-slope spread (use raw zipf, not z-scored)
    class_zipf = []
    for c in range(cfg.n_cls):
        sel = (train_labels == c)
        if sel.sum() > 0:
            class_zipf.append(float(X_train_raw[sel, 0].mean()))
        else:
            class_zipf.append(0.0)
    zipf_spread = float(max(class_zipf) - min(class_zipf))
    f1_ok = zipf_spread >= 0.05
    logger.info(f"[F1] zipf-slope per-class means={['%.3f' % z for z in class_zipf]} "
                f"spread={zipf_spread:.4f}")

    # F2: single-feature ablation (use one feature only)
    single_test = {}
    for i, name in enumerate(FEATURE_NAMES):
        mus_i = mus[:, i:i+1]
        preds_i = _nc_predict(Xts[:, i:i+1], mus_i)
        single_test[name] = float(f1_score(test_labels, preds_i, average="macro", zero_division=0))
    best_single_name = max(single_test, key=single_test.get)
    best_single_val = single_test[best_single_name]
    combo_minus_best_single = test_macro - best_single_val
    f2_ok = combo_minus_best_single >= 0.005
    logger.info(f"[F2] per-feature test macros = {single_test}")
    logger.info(f"[F2] best single = {best_single_name}={best_single_val:.4f}  "
                f"combo - best = {combo_minus_best_single:+.4f}")

    # F3: must beat random by >= 0.05
    random_baseline = 1.0 / max(cfg.n_cls, 1)
    f3_delta = test_macro - random_baseline
    f3_ok = f3_delta >= 0.05
    logger.info(f"[F3] random baseline ~ {random_baseline:.4f}  test - random = {f3_delta:+.4f}")

    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    ts_met["falsifier_F1_zipf_slopes_per_class"] = class_zipf
    ts_met["falsifier_F1_zipf_spread"] = zipf_spread
    ts_met["falsifier_F1_ok"] = bool(f1_ok)
    ts_met["falsifier_F2_individual_feature_macros"] = single_test
    ts_met["falsifier_F2_best_single_feature"] = best_single_name
    ts_met["falsifier_F2_combo_minus_best_single"] = float(combo_minus_best_single)
    ts_met["falsifier_F2_ok"] = bool(f2_ok)
    ts_met["falsifier_F3_dpaper_minus_random"] = float(f3_delta)
    ts_met["falsifier_F3_ok"] = bool(f3_ok)

    return {
        "tag": tag, "method": "ZIPFLAW",
        "note": ("Statistical-laws stylometry: Zipf + Heaps + Yule + Brunet "
                 "+ Simpson + Sichel packed into a 7-d author signature; "
                 "nearest-centroid in z-scored space. CPU-only."),
        "enc": f"laws-d{N_FEATURES}", "bench": cfg.benchmark,
        "frac": cfg.frac,
        "n_features": int(N_FEATURES),
        "feature_names": list(FEATURE_NAMES),
        "train_size_after_cap": int(len(train_codes)),
        "val_size_after_cap":   int(len(val_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "zipf_slope_per_class": class_zipf,
        "zipf_spread": zipf_spread,
        "single_feature_macros": single_test,
        "best_single_feature": best_single_name,
        "combo_minus_best_single": float(combo_minus_best_single),
        "random_baseline": float(random_baseline),
        "falsifier_F1_ok": bool(f1_ok),
        "falsifier_F2_ok": bool(f2_ok),
        "falsifier_F3_ok": bool(f3_ok),
        "feat_time_sec": feat_time,
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
            tag = f"exp157_zipflaw_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} "
                            f"combo-best_single={res['combo_minus_best_single']:+.4f} "
                            f"zipf_spread={res['zipf_spread']:.3f} "
                            f"time={res['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp157_zipflaw_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Train':>7} {'Test':>7} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'ComboGain':>10} {'ZipfSprd':>9} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['train_size_after_cap']:>7d} {r['test_size_after_cap']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['combo_minus_best_single']:>+10.4f} {r['zipf_spread']:>9.4f} "
              f"{r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
