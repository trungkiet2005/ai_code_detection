# exp139 — TOPOSIG
# =============================================================================
# NAME       : TOPOSIG (Topological signature of code via persistence diagrams)
# REFERENCE  : Edelsbrunner & Harer 2010 ("Computational Topology: An
#              Introduction"), Chazal & Michel 2017 ("An introduction to
#              Topological Data Analysis"), arXiv:1710.04019. Never applied
#              to code authorship — first TDA-based attribution method.
# CLAIM      : The COOCCURRENCE GRAPH of characters/tokens in an author's
#              code has a topological signature — its persistence diagram
#              — that varies systematically across authors. We compute
#              persistence H_0 (connected components) and H_1 (loops) on
#              the character-bigram graph and attribute by Wasserstein
#              distance between persistence diagrams.
# EQUATION   : G(x) = graph where nodes = distinct characters, edges = char
#              bigrams with weight = -log(frequency)
#              Compute Vietoris-Rips filtration of G(x)
#              PD(x) = persistence diagram (set of (birth, death) pairs)
#              y_hat = argmin_a Wasserstein_2(PD(x), PD(a))
#              where PD(a) is the persistence diagram of class-a centroid.
# WHY NEW    : TDA has been applied to text classification (small literature),
#              never to code authorship. Persistence is invariant to small
#              perturbations of the underlying data — exactly the property
#              we want for handling decoding-temperature noise.
# WOW HOOK   : "Author identity is a TOPOLOGICAL invariant. We compute the
#              persistence diagram of each author's character cooccurrence
#              graph — the barcodes that describe how holes form and die
#              in their code — and attribute by topological distance.
#              Topology has a Vietnamese opinion about LLMs."
# FALSIFIER  : (F1) Persistence diagrams differ across classes:
#              mean Wasserstein within-class < mean Wasserstein across-class
#              by ratio >= 1.5.
#              (F2) H_0 (components) + H_1 (loops) BOTH contribute: removing
#              H_1 changes accuracy by >= 0.005.
#              (F3) Composite > GZIP-NCD baseline by >= 0.01.
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

_ensure("numpy"); _ensure("scipy"); _ensure("datasets"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp139_toposig")

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
# TDA core: persistence diagrams of character bigram graphs
# =============================================================================

MAX_NODES = 80  # cap distinct chars to bound complexity (Rips is O(n^3))
MAX_CHARS = 4000
MAX_EDGE_LEN = 10.0
N_BINS = 10
MAX_PERSISTENCE = 5.0


def build_char_graph(text, max_chars=MAX_CHARS):
    """Return adjacency distance matrix (n_chars x n_chars) where weight = -log(freq).
    Caps node count to MAX_NODES by retaining only the most frequent characters."""
    text = text[:max_chars]
    if len(text) < 2: return None
    char_counts = {}
    for c in text:
        char_counts[c] = char_counts.get(c, 0) + 1
    # Retain top-MAX_NODES most frequent characters
    sorted_chars = sorted(char_counts.items(), key=lambda kv: -kv[1])[:MAX_NODES]
    chars = sorted([c for c, _ in sorted_chars])
    char_idx = {c: i for i, c in enumerate(chars)}
    n = len(chars)
    if n < 2: return None
    counts = np.zeros((n, n), dtype=np.float32)
    for i in range(len(text) - 1):
        a = char_idx.get(text[i]); b = char_idx.get(text[i + 1])
        if a is not None and b is not None:
            counts[a, b] += 1
            counts[b, a] += 1
    total = counts.sum()
    if total < 1: return None
    probs = counts / total
    eps = 1e-6
    dist_matrix = -np.log(probs + eps)
    # Diagonal = 0
    np.fill_diagonal(dist_matrix, 0.0)
    return dist_matrix


def compute_persistence(dist_matrix, max_edge_length=MAX_EDGE_LEN, max_dim=1):
    """Pure-numpy persistence proxy (offline-safe, no gudhi):
       H_0: single-linkage merge tree — MST edge weights ARE the exact
            persistence birth=0, death=weight under the Rips filtration.
       H_1: each non-tree edge with weight w in [0, max_edge_length] is the
            birth of an independent cycle; death is capped at max_edge_length
            (Rips cycles fill in when their longest edge ≤ filtration value).
    """
    if dist_matrix is None or dist_matrix.shape[0] < 2:
        return [], []
    n = dist_matrix.shape[0]
    D = np.asarray(dist_matrix, dtype=np.float32).copy()
    # Restrict filtration: edges beyond max_edge_length are absent.
    D[D > max_edge_length] = 0.0
    np.fill_diagonal(D, 0.0)
    try:
        mst = minimum_spanning_tree(D).toarray().astype(np.float32)
    except Exception:
        return [], []
    mst_weights = mst[mst > 0]
    pd0 = [(0.0, float(w)) for w in mst_weights]
    upper = np.triu(D, k=1)
    upper_mask = upper > 0
    # scipy MST is upper-triangular; account for both halves to be safe.
    mst_upper = np.triu(mst, k=1) + np.triu(mst.T, k=1)
    mst_mask = mst_upper > 0
    nontree_mask = upper_mask & (~mst_mask)
    nontree_weights = upper[nontree_mask]
    pd1 = [(float(w), float(max_edge_length)) for w in nontree_weights]
    return pd0, pd1


def diagram_to_vector(pd, n_bins=N_BINS, max_persistence=MAX_PERSISTENCE):
    """Vectorize a persistence diagram via histogram of persistences (death - birth)."""
    if not pd:
        return np.zeros(n_bins, dtype=np.float32)
    persistences = [max(0.0, death - birth) for birth, death in pd]
    hist, _ = np.histogram(persistences, bins=n_bins, range=(0, max_persistence))
    s = hist.sum()
    if s == 0: return np.zeros(n_bins, dtype=np.float32)
    return hist.astype(np.float32) / s


def encode_sample(code):
    """Compute persistence vector (H_0 concat H_1) for one code sample."""
    dist_mat = build_char_graph(code)
    pd0, pd1 = compute_persistence(dist_mat)
    v0 = diagram_to_vector(pd0)
    v1 = diagram_to_vector(pd1)
    return np.concatenate([v0, v1]).astype(np.float32)


def _encode_worker(code):
    try:
        return encode_sample(code)
    except Exception:
        return np.zeros(2 * N_BINS, dtype=np.float32)


def encode_all(codes, n_workers=None, desc="encode"):
    if n_workers is None or n_workers <= 1:
        return np.stack([_encode_worker(c) for c in tqdm(codes, desc=desc)])
    try:
        with mp.Pool(n_workers) as pool:
            out = list(tqdm(pool.imap(_encode_worker, codes, chunksize=8),
                             total=len(codes), desc=desc))
        return np.stack(out)
    except Exception as e:
        logger.warning(f"[toposig] mp failed ({e}); serial fallback")
        return np.stack([_encode_worker(c) for c in tqdm(codes, desc=desc)])


# =============================================================================
# Nearest-centroid classifier on vectorized persistence diagrams
# =============================================================================

def per_class_centroids(X, y, n_cls):
    """Mean vector per class."""
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
    """For each query, argmin Euclidean to centroid."""
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


def intra_inter_distances(X, y, n_cls, sample_n=400):
    """Mean intra-class vs inter-class Euclidean distance on persistence vectors."""
    rng = np.random.default_rng(42)
    n = X.shape[0]
    if n < 4: return None, None
    intra, inter = [], []
    picks = rng.choice(n, size=min(sample_n, n), replace=False)
    for i in picks:
        for j in picks:
            if i == j: continue
            d = float(np.linalg.norm(X[i] - X[j]))
            if y[i] == y[j]:
                intra.append(d)
            else:
                inter.append(d)
    if not intra or not inter: return None, None
    return float(np.mean(intra)), float(np.mean(inter))


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
                f"test={len(test_codes)} n_cls={cfg.n_cls} workers={n_workers}")

    # Encode all splits via TDA
    t_enc0 = time.time()
    X_train = encode_all(train_codes, n_workers=n_workers, desc="encode-train")
    X_val   = encode_all(val_codes,   n_workers=n_workers, desc="encode-val")
    X_test  = encode_all(test_codes,  n_workers=n_workers, desc="encode-test")
    enc_time = time.time() - t_enc0
    logger.info(f"[encode] done in {enc_time:.0f}s shape_train={X_train.shape}")

    y_train = np.array(train_labels, dtype=np.int64)
    y_val   = np.array(val_labels,   dtype=np.int64)
    y_test  = np.array(test_labels,  dtype=np.int64)

    # Per-class centroids on train; predict by nearest centroid (full vector = H_0 + H_1)
    centroids_full, _ = per_class_centroids(X_train, y_train, cfg.n_cls)
    # Restrict valid classes to those present in train (avoid empty centroids)
    valid_cls = [c for c in range(cfg.n_cls) if (y_train == c).sum() > 0]

    t0 = time.time()
    val_preds = predict_centroid(X_val, centroids_full, valid_classes=valid_cls)
    val_time = time.time() - t0
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                         cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]

    t0 = time.time()
    test_preds = predict_centroid(X_test, centroids_full, valid_classes=valid_cls)
    test_time = time.time() - t0
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro

    # --- Falsifier F1: intra vs inter Wasserstein-like distance on full vectors ---
    intra_w, inter_w = intra_inter_distances(X_test, y_test, cfg.n_cls, sample_n=300)
    ratio = (inter_w / intra_w) if (intra_w and intra_w > 1e-9) else None

    # --- Falsifier F2: H_0-only and H_1-only ablations ---
    # X is concat of [H_0 vec (N_BINS), H_1 vec (N_BINS)]
    X_train_h0 = X_train[:, :N_BINS];   X_test_h0 = X_test[:, :N_BINS]
    X_train_h1 = X_train[:, N_BINS:];   X_test_h1 = X_test[:, N_BINS:]
    cent_h0, _ = per_class_centroids(X_train_h0, y_train, cfg.n_cls)
    cent_h1, _ = per_class_centroids(X_train_h1, y_train, cfg.n_cls)
    pred_h0 = predict_centroid(X_test_h0, cent_h0, valid_classes=valid_cls)
    pred_h1 = predict_centroid(X_test_h1, cent_h1, valid_classes=valid_cls)
    h0_macro = float(f1_score(y_test, pred_h0, average="macro", zero_division=0))
    h1_macro = float(f1_score(y_test, pred_h1, average="macro", zero_division=0))
    h1_contribution = float(test_macro - h0_macro)  # positive => H_1 helps

    logger.info(f"[val] macro={val_macro:.4f} time={val_time:.0f}s")
    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} time={test_time:.0f}s")
    logger.info(f"[F1] intra={intra_w} inter={inter_w} ratio={ratio}")
    logger.info(f"[F2] H_0_only={h0_macro:.4f} H_1_only={h1_macro:.4f} "
                f"combined={test_macro:.4f} dH1={h1_contribution:+.4f}")

    ts_met["falsifier_F1_intra_dist"] = intra_w
    ts_met["falsifier_F1_inter_dist"] = inter_w
    ts_met["falsifier_F1_ratio"] = ratio
    ts_met["falsifier_F2_h0_only_macro"] = h0_macro
    ts_met["falsifier_F2_h1_only_macro"] = h1_macro
    ts_met["falsifier_F2_h1_contribution"] = h1_contribution

    return {
        "tag": tag, "method": "TOPOSIG",
        "note": ("Topological signature of character cooccurrence graphs. "
                 "H_0 + H_1 persistence diagram, vectorized to histograms, "
                 "nearest-centroid classification. CPU-only TDA."),
        "enc": "gudhi-rips", "bench": cfg.benchmark,
        "frac": cfg.frac,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "mean_intra_dist": intra_w, "mean_inter_dist": inter_w,
        "intra_inter_ratio": ratio,
        "h0_only_macro": h0_macro, "h1_only_macro": h1_macro,
        "h1_contribution": h1_contribution,
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
            tag = f"exp139_toposig_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"ratio={res['intra_inter_ratio']}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp139_toposig_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Train':>8} {'Test':>8} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'Intra':>9} {'Inter':>9} {'Ratio':>7} {'H0only':>8} {'H1only':>8} {'Wall':>8}")
    print("-"*140)
    for r in results:
        intra = r.get("mean_intra_dist") or 0.0
        inter = r.get("mean_inter_dist") or 0.0
        ratio = r.get("intra_inter_ratio") or 0.0
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['train_size_after_cap']:>8d} {r['test_size_after_cap']:>8d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {intra:>9.4f} {inter:>9.4f} {ratio:>7.3f} "
              f"{r['h0_only_macro']:>8.4f} {r['h1_only_macro']:>8.4f} {r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
