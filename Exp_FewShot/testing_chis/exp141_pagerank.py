# exp141 — PAGERANK
# =============================================================================
# NAME       : PAGERANK (Per-author token transition PageRank signature; attribution by JS divergence)
# REFERENCE  : Page et al. 1999 ("The PageRank Citation Ranking: Bringing
#              Order to the Web"), Brin & Page 1998. Lin 1991 ("Divergence
#              measures based on Shannon entropy"). PageRank for AUTHORSHIP
#              ATTRIBUTION is novel.
# CLAIM      : Each author's character/token TRANSITION graph has a
#              distinct stationary distribution (PageRank vector). Authors
#              with similar coding habits have similar transition graphs
#              → similar PageRank → similar signature. We attribute by
#              minimum Jensen-Shannon divergence between the query's
#              PageRank and each class's mean PageRank.
# EQUATION   : Build transition matrix T(x) where T[i,j] = P(char j | char i)
#              empirically from a sample (or all training samples of class c).
#              PageRank: pi = (1-d)/N + d * T^T * pi, solve for fixed point.
#              Per-class signature: mean PageRank over all class-c training
#              samples.
#              y_hat = argmin_c JS-divergence(pi(query), pi_c_mean)
# WHY NEW    : PageRank has been applied to web ranking, fraud detection,
#              biology. Never used for code authorship. Combining PageRank
#              vector (stationary dist) with JS divergence (info-theoretic)
#              gives a principled, parameter-free attribution method.
# WOW HOOK   : "The same PageRank algorithm that Google uses to rank web
#              pages — applied to character transition graphs — distinguishes
#              LLM authors. Each author has a 'page' in their code; we
#              measure them by the rank vector of their characters."
# FALSIFIER  : (F1) Mean intra-class JS-divergence < mean inter-class JS by
#              ratio >= 1.5 (signatures cluster by class).
#              (F2) PageRank composite > raw bigram-distribution baseline
#              (without rank propagation) by >= 0.005 — propagation HELPS.
#              (F3) Damping factor d in [0.7, 0.95] gives stable performance
#              (not single point).
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("datasets"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp141_pagerank")

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
# PageRank core
# =============================================================================

# Universal vocab: printable ASCII subset (128 chars). All transition matrices
# have the same dim → PageRank vectors are directly comparable across samples.
VOCAB_SIZE = 128
UNIVERSAL_VOCAB = {chr(i): i for i in range(VOCAB_SIZE)}


def build_transition_matrix(text, vocab=UNIVERSAL_VOCAB, max_chars=4000):
    """Return (n_chars, n_chars) row-stochastic transition matrix.
    T[i, j] = P(char_j | char_i) estimated empirically from `text`.
    """
    text = text[:max_chars]
    n = len(vocab)
    T = np.zeros((n, n), dtype=np.float32)
    if not text:
        return T, vocab
    for i in range(len(text) - 1):
        a = vocab.get(text[i]); b = vocab.get(text[i + 1])
        if a is not None and b is not None:
            T[a, b] += 1.0
    row_sums = T.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    T = T / row_sums
    return T, vocab


def pagerank_vector(T, d=0.85, max_iter=100, tol=1e-6):
    """Power iteration for the PageRank fixed point.
    pi = (1 - d) / N + d * T^T * pi
    """
    n = T.shape[0]
    pi = np.ones(n, dtype=np.float32) / n
    Tt = T.T
    for _ in range(max_iter):
        pi_new = (1.0 - d) / n + d * (Tt @ pi)
        s = pi_new.sum()
        if s > 0:
            pi_new = pi_new / s
        if np.abs(pi_new - pi).sum() < tol:
            return pi_new
        pi = pi_new
    return pi


def js_divergence(p, q):
    """Jensen-Shannon divergence between two probability distributions (base 2)."""
    p = np.asarray(p, dtype=np.float64) + 1e-12
    q = np.asarray(q, dtype=np.float64) + 1e-12
    p = p / p.sum(); q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b): return np.sum(a * np.log2(a / b))
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def bigram_distribution(text, vocab=UNIVERSAL_VOCAB, max_chars=4000):
    """Raw bigram-marginal distribution (baseline for F2).
    Equivalent to PageRank with d=0 (uniform restart, no propagation), but
    computed as the marginal frequency of each character from observed bigrams.
    """
    text = text[:max_chars]
    n = len(vocab)
    counts = np.zeros(n, dtype=np.float32)
    if not text:
        return np.ones(n, dtype=np.float32) / n
    for ch in text:
        idx = vocab.get(ch)
        if idx is not None:
            counts[idx] += 1.0
    s = counts.sum()
    if s == 0:
        return np.ones(n, dtype=np.float32) / n
    return counts / s


# =============================================================================
# Per-sample worker
# =============================================================================

_WORKER_DAMPING = 0.85
_WORKER_MAX_CHARS = 4000


def _worker_init(damping, max_chars):
    global _WORKER_DAMPING, _WORKER_MAX_CHARS
    _WORKER_DAMPING = damping
    _WORKER_MAX_CHARS = max_chars


def _encode_pagerank(args):
    idx, code = args
    T, _ = build_transition_matrix(code, vocab=UNIVERSAL_VOCAB,
                                   max_chars=_WORKER_MAX_CHARS)
    pi = pagerank_vector(T, d=_WORKER_DAMPING)
    return idx, pi.astype(np.float32)


def _encode_bigram(args):
    idx, code = args
    pi = bigram_distribution(code, vocab=UNIVERSAL_VOCAB,
                             max_chars=_WORKER_MAX_CHARS)
    return idx, pi.astype(np.float32)


def encode_all(codes, damping=0.85, max_chars=4000, n_workers=None,
               mode="pagerank", desc="encode"):
    """Compute per-sample signatures with multiprocessing.
    mode: 'pagerank' or 'bigram'.
    """
    N = len(codes)
    if n_workers is None or n_workers <= 0:
        n_workers = max(1, mp.cpu_count() - 1)
    args_list = [(i, codes[i]) for i in range(N)]
    sigs = np.zeros((N, VOCAB_SIZE), dtype=np.float32)
    worker_fn = _encode_pagerank if mode == "pagerank" else _encode_bigram
    if n_workers == 1:
        _worker_init(damping, max_chars)
        for a in tqdm(args_list, desc=desc):
            idx, pi = worker_fn(a); sigs[idx] = pi
        return sigs
    try:
        with mp.Pool(n_workers, initializer=_worker_init,
                     initargs=(damping, max_chars)) as pool:
            for idx, pi in tqdm(pool.imap_unordered(worker_fn, args_list, chunksize=8),
                                total=N, desc=desc):
                sigs[idx] = pi
        return sigs
    except Exception as e:
        logger.warning(f"[pagerank] mp failed ({e}); serial fallback")
        return encode_all(codes, damping, max_chars, n_workers=1,
                          mode=mode, desc=desc)


# =============================================================================
# Classification: nearest class signature by JS divergence
# =============================================================================

def per_class_mean(sigs, labels, n_cls):
    """Return (n_cls, VOCAB_SIZE) array of mean signatures per class."""
    means = np.zeros((n_cls, sigs.shape[1]), dtype=np.float32)
    counts = np.zeros(n_cls, dtype=np.int64)
    for s, y in zip(sigs, labels):
        means[y] += s; counts[y] += 1
    for c in range(n_cls):
        if counts[c] > 0:
            means[c] /= counts[c]
        else:
            means[c] = 1.0 / sigs.shape[1]
    # re-normalise (in case of float drift)
    row_sums = means.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    means = means / row_sums
    return means


def predict_min_js(query_sigs, class_means):
    """argmin_c JS(query_sig, class_mean_c) for each query."""
    N = query_sigs.shape[0]; C = class_means.shape[0]
    preds = np.zeros(N, dtype=np.int64)
    for i in range(N):
        q = query_sigs[i]
        best_c, best_d = 0, float("inf")
        for c in range(C):
            d = js_divergence(q, class_means[c])
            if d < best_d:
                best_d = d; best_c = c
        preds[i] = best_c
    return preds


def intra_inter_js(sigs, labels, n_cls, max_pairs_per_query=20, seed=42):
    """Compute mean intra-class vs mean inter-class JS divergence on a sample
    of (sig_i, class_mean_c) pairings (F1 falsifier)."""
    rng = random.Random(seed)
    class_means = per_class_mean(sigs, labels, n_cls)
    intra, inter = [], []
    for i in range(len(sigs)):
        y = labels[i]
        intra.append(js_divergence(sigs[i], class_means[y]))
        other_classes = [c for c in range(n_cls) if c != y]
        if other_classes:
            sampled = rng.sample(other_classes, min(len(other_classes),
                                                     max_pairs_per_query))
            for c in sampled:
                inter.append(js_divergence(sigs[i], class_means[c]))
    return (float(np.mean(intra)) if intra else None,
            float(np.mean(inter)) if inter else None)


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
    damping: float = 0.85
    damping_sweep: tuple = (0.70, 0.85, 0.95)
    train_cap_per_class: int = 400
    test_cap_per_class: int = 500
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = 4000


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
                f"test={len(test_codes)} d={cfg.damping} workers={n_workers}")

    # 1) Encode all samples (PageRank, default damping)
    t0 = time.time()
    train_sigs = encode_all(train_codes, damping=cfg.damping,
                            max_chars=cfg.max_chars, n_workers=n_workers,
                            mode="pagerank", desc="train-PR")
    val_sigs   = encode_all(val_codes,   damping=cfg.damping,
                            max_chars=cfg.max_chars, n_workers=n_workers,
                            mode="pagerank", desc="val-PR")
    test_sigs  = encode_all(test_codes,  damping=cfg.damping,
                            max_chars=cfg.max_chars, n_workers=n_workers,
                            mode="pagerank", desc="test-PR")
    encode_time = time.time() - t0

    class_means = per_class_mean(train_sigs, train_labels, cfg.n_cls)

    # Val pass
    t0 = time.time()
    val_preds = predict_min_js(val_sigs, class_means)
    val_time = time.time() - t0
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                        cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]
    logger.info(f"[val] macro={val_macro:.4f} time={val_time:.0f}s")

    # Test pass
    t0 = time.time()
    test_preds = predict_min_js(test_sigs, class_means)
    test_time = time.time() - t0
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro
    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} time={test_time:.0f}s")

    # ---- Falsifier F1: intra-class vs inter-class JS divergence on TEST sigs
    intra_js, inter_js = intra_inter_js(test_sigs, test_labels, cfg.n_cls,
                                         max_pairs_per_query=min(cfg.n_cls - 1, 8),
                                         seed=cfg.seed)
    f1_ratio = (inter_js / intra_js) if (intra_js and intra_js > 0) else None
    logger.info(f"[F1] mean_intra_js={intra_js} mean_inter_js={inter_js} "
                f"ratio={f1_ratio}")

    # ---- Falsifier F2: raw bigram-distribution baseline (no rank propagation)
    bg_train = encode_all(train_codes, damping=cfg.damping,
                          max_chars=cfg.max_chars, n_workers=n_workers,
                          mode="bigram", desc="train-BG")
    bg_test = encode_all(test_codes, damping=cfg.damping,
                         max_chars=cfg.max_chars, n_workers=n_workers,
                         mode="bigram", desc="test-BG")
    bg_class_means = per_class_mean(bg_train, train_labels, cfg.n_cls)
    bg_preds = predict_min_js(bg_test, bg_class_means)
    bg_macro = float(f1_score(test_labels, bg_preds, average="macro",
                              zero_division=0))
    f2_delta = test_macro - bg_macro
    logger.info(f"[F2] pagerank_macro={test_macro:.4f} bigram_macro={bg_macro:.4f} "
                f"delta={f2_delta:+.4f}")

    # ---- Falsifier F3: damping sweep on TEST (encode once at each d, predict)
    damping_macros = {}
    for d_alt in cfg.damping_sweep:
        if abs(d_alt - cfg.damping) < 1e-6:
            damping_macros[f"{d_alt:.2f}"] = test_macro
            continue
        tr_alt = encode_all(train_codes, damping=d_alt, max_chars=cfg.max_chars,
                            n_workers=n_workers, mode="pagerank",
                            desc=f"train-PR-d{d_alt}")
        ts_alt = encode_all(test_codes, damping=d_alt, max_chars=cfg.max_chars,
                            n_workers=n_workers, mode="pagerank",
                            desc=f"test-PR-d{d_alt}")
        cm_alt = per_class_mean(tr_alt, train_labels, cfg.n_cls)
        pr_alt = predict_min_js(ts_alt, cm_alt)
        m_alt = float(f1_score(test_labels, pr_alt, average="macro",
                                zero_division=0))
        damping_macros[f"{d_alt:.2f}"] = m_alt
    logger.info(f"[F3] damping_sweep_macros={damping_macros}")
    d_vals = list(damping_macros.values())
    f3_spread = max(d_vals) - min(d_vals) if d_vals else 0.0

    ts_met["falsifier_F1_mean_intra_class_js"] = intra_js
    ts_met["falsifier_F1_mean_inter_class_js"] = inter_js
    ts_met["falsifier_F1_inter_intra_ratio"]   = f1_ratio
    ts_met["falsifier_F2_bigram_baseline_macro"] = bg_macro
    ts_met["falsifier_F2_pagerank_vs_bigram_delta"] = f2_delta
    ts_met["falsifier_F3_damping_sweep_macros"] = damping_macros
    ts_met["falsifier_F3_damping_spread"] = f3_spread

    return {
        "tag": tag, "method": "PAGERANK",
        "note": ("Per-author character-transition PageRank signature; "
                 "attribution by min Jensen-Shannon divergence to per-class "
                 "mean signature. CPU-only; multiprocessing for per-sample PR."),
        "enc": f"pagerank-d{cfg.damping}", "bench": cfg.benchmark,
        "frac": cfg.frac, "damping": cfg.damping,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "mean_intra_class_js": intra_js,
        "mean_inter_class_js": inter_js,
        "inter_intra_js_ratio": f1_ratio,
        "bigram_baseline_macro": bg_macro,
        "pagerank_vs_bigram_delta": f2_delta,
        "damping_sweep_macros": damping_macros,
        "damping_spread": f3_spread,
        "encode_time_sec": encode_time,
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
            tag = f"exp141_pagerank_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"ratio={res['inter_intra_js_ratio']}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp141_pagerank_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'d':>5} {'Train':>8} {'Test':>8} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'BG-F1':>8} {'dBG':>8} {'Ratio':>7} {'Wall':>8}")
    print("-"*140)
    for r in results:
        ratio = r.get("inter_intra_js_ratio") or 0.0
        bg = r.get("bigram_baseline_macro") or 0.0
        dbg = r.get("pagerank_vs_bigram_delta") or 0.0
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['damping']:>5.2f} "
              f"{r['train_size_after_cap']:>8d} {r['test_size_after_cap']:>8d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {bg:>8.4f} {dbg:>+8.4f} {ratio:>7.3f} "
              f"{r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
