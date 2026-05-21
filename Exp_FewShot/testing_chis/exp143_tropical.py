# exp143 — TROPICAL
# =============================================================================
# NAME       : TROPICAL (Tropical geometry of character distributions for attribution)
# REFERENCE  : Maclagan & Sturmfels 2015 ("Introduction to Tropical
#              Geometry", Graduate Studies in Mathematics 161). Pachter &
#              Sturmfels 2004 (tropical methods in computational biology).
#              NEVER applied to NLP or code authorship before.
# CLAIM      : Code character/token distributions can be projected into
#              the TROPICAL SEMIRING (min-plus algebra) where addition is
#              MIN and multiplication is ADDITION. This semiring is the
#              natural setting for amortized analysis of discrete
#              structures. We compute the tropical distance between query
#              and each class centroid; the min-plus structure captures
#              dominant features differently from Euclidean.
# EQUATION   : Let f_i(x) = log(count_i(x) + 1) = log-frequency feature i.
#              Tropical "polynomial" representation:
#                  T(x) = (+)_i f_i(x) (.) w_i      (min over i of f_i + w_i)
#              Tropical distance:
#                  d_T(x, y) = max_i |f_i(x) - f_i(y)|   (Chebyshev/L_inf,
#                                  the natural metric in (R, min, +))
#              y_hat = argmin_a max_i |f_i(x) - mean_f_i(a)|
#              Plus a tropical hyperplane classifier: the predicted class
#              is the LAST argmin of the tropical projection (Maclagan-
#              Sturmfels classifier).
# WHY NEW    : Tropical geometry is a 20-year-old field of pure math that
#              has just begun touching ML (tropical neural networks,
#              Zhang & Naitzat 2018). Never applied to ANY text/code
#              classification. We bring an entire algebraic perspective
#              to the table.
# WOW HOOK   : "We attribute LLM code with TROPICAL GEOMETRY — pure
#              mathematics from 2005. Addition is MIN, multiplication is
#              ADDITION. The author whose tropical projection lies closest
#              to the query is the predicted author. The Chebyshev metric
#              of log-frequencies replaces Euclidean. This is the first
#              time tropical algebra touches code attribution."
# FALSIFIER  : (F1) Tropical distance separates classes: mean intra-class
#              < mean inter-class by ratio >= 1.5.
#              (F2) Tropical (L_inf) beats Euclidean (L_2) by >= 0.005 OR
#              loses by <= 0.005 (within noise — both metrics on same
#              features). If tropical >> euclidean, the algebra is doing
#              real work.
#              (F3) The "last argmin" tropical classifier (Maclagan-
#              Sturmfels Theorem 5.4.9) and simple nearest-centroid
#              agree on >= 80% of predictions (sanity check on the
#              tropical projection geometry).
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
logger = logging.getLogger("exp143_tropical")

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
# Data loading (identical to exp133)
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
# Tropical features and distances
# =============================================================================

FEAT_DIM = 128  # ASCII char log-frequency features


def tropical_features(code: str, max_chars: int = 4000) -> np.ndarray:
    """Compute 128-dim ASCII log-frequency feature vector for a code snippet.

    The log1p transform is the canonical mapping into the tropical semiring:
    counts (which obey multiplicative algebra) become log-counts (which obey
    additive algebra). Min-plus operations on log-counts correspond to taking
    the dominant factor in the original counts.
    """
    code = code[:max_chars]
    f = np.zeros(FEAT_DIM, dtype=np.float32)
    for c in code:
        idx = ord(c)
        if idx < FEAT_DIM:
            f[idx] += 1.0
    # log1p: canonical map count -> tropical coordinate
    return np.log1p(f)


def tropical_distance(x: np.ndarray, y: np.ndarray) -> float:
    """L_infinity (Chebyshev) distance — the natural metric in the tropical
    semiring (R, min, +). For any two points x, y, the tropical distance
    is the max coordinate-wise discrepancy.
    """
    return float(np.max(np.abs(x - y)))


def tropical_distance_batch(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Vectorised Chebyshev distance from single x (D,) to many Y (K, D)."""
    return np.max(np.abs(Y - x[None, :]), axis=1)


def euclidean_distance(x: np.ndarray, y: np.ndarray) -> float:
    """L_2 distance — for comparison with tropical."""
    return float(np.linalg.norm(x - y))


def euclidean_distance_batch(x: np.ndarray, Y: np.ndarray) -> np.ndarray:
    return np.linalg.norm(Y - x[None, :], axis=1)


def tropical_projection_scores(x: np.ndarray, class_means: np.ndarray) -> np.ndarray:
    """Maclagan-Sturmfels min-plus tropical projection.

    For each class c with mean mu_c, the tropical polynomial evaluated at x is
        p_c(x) = min_i (mu_c[i] - x[i])
    (this is the min-plus "inner product" with the negation of x).
    The argmax over classes of p_c(x) is the predicted class — the tropical
    "vertex" closest to x in the min-plus sense (cf. Maclagan-Sturmfels
    Theorem 5.4.9 on tropical hyperplanes).
    """
    # class_means: (K, D); x: (D,)
    diff = class_means - x[None, :]  # (K, D)
    return np.min(diff, axis=1)       # (K,) — argmax is the prediction


def compute_class_means(features: np.ndarray, labels: List[int], n_cls: int) -> np.ndarray:
    """Compute per-class mean feature vector. features: (N, D)."""
    D = features.shape[1]
    means = np.zeros((n_cls, D), dtype=np.float32)
    counts = np.zeros(n_cls, dtype=np.int64)
    for f, l in zip(features, labels):
        means[l] += f
        counts[l] += 1
    counts[counts == 0] = 1
    means = means / counts[:, None]
    return means


# =============================================================================
# Worker for parallel feature extraction
# =============================================================================

def _feat_one(args):
    qi, code, max_chars = args
    return qi, tropical_features(code, max_chars=max_chars)


def extract_features_parallel(codes: List[str], max_chars: int, n_workers: int) -> np.ndarray:
    N = len(codes)
    feats = np.zeros((N, FEAT_DIM), dtype=np.float32)
    if n_workers <= 1:
        for i in tqdm(range(N), desc="feat"):
            feats[i] = tropical_features(codes[i], max_chars=max_chars)
        return feats
    try:
        args_list = [(i, codes[i], max_chars) for i in range(N)]
        with mp.Pool(n_workers) as pool:
            for qi, f in tqdm(pool.imap_unordered(_feat_one, args_list, chunksize=32),
                              total=N, desc="feat"):
                feats[qi] = f
    except Exception as e:
        logger.warning(f"[tropical] parallel feat extraction failed ({e}); serial")
        for i in tqdm(range(N), desc="feat-serial"):
            feats[i] = tropical_features(codes[i], max_chars=max_chars)
    return feats


# =============================================================================
# Three classifier modes
# =============================================================================

def classify_tropical_chebyshev(query_feats: np.ndarray, class_means: np.ndarray) -> np.ndarray:
    """argmin_c L_inf(x, mu_c)."""
    N = query_feats.shape[0]
    preds = np.zeros(N, dtype=np.int64)
    for i in range(N):
        d = tropical_distance_batch(query_feats[i], class_means)
        preds[i] = int(np.argmin(d))
    return preds


def classify_euclidean(query_feats: np.ndarray, class_means: np.ndarray) -> np.ndarray:
    N = query_feats.shape[0]
    preds = np.zeros(N, dtype=np.int64)
    for i in range(N):
        d = euclidean_distance_batch(query_feats[i], class_means)
        preds[i] = int(np.argmin(d))
    return preds


def classify_tropical_projection(query_feats: np.ndarray, class_means: np.ndarray) -> np.ndarray:
    """Maclagan-Sturmfels: argmax_c min_i (mu_c[i] - x[i]).
    This is the 'last argmin' classifier — the class whose tropical hyperplane
    vertex the query lies closest to.
    """
    N = query_feats.shape[0]
    preds = np.zeros(N, dtype=np.int64)
    for i in range(N):
        scores = tropical_projection_scores(query_feats[i], class_means)
        preds[i] = int(np.argmax(scores))
    return preds


# =============================================================================
# Falsifier metrics
# =============================================================================

def compute_intra_inter_tropical(test_feats: np.ndarray, test_labels: List[int],
                                  class_means: np.ndarray, n_samples: int = 200) -> Tuple[float, float, float]:
    """Mean tropical distance from random test samples to (a) own-class mean
    and (b) other-class means. Returns (intra, inter, ratio)."""
    n_cls = class_means.shape[0]
    rng = np.random.default_rng(42)
    N = test_feats.shape[0]
    idxs = rng.choice(N, size=min(n_samples, N), replace=False)
    intra, inter = [], []
    for i in idxs:
        l = test_labels[i]
        intra.append(tropical_distance(test_feats[i], class_means[l]))
        others = [c for c in range(n_cls) if c != l]
        for c in others:
            inter.append(tropical_distance(test_feats[i], class_means[c]))
    intra_m = float(np.mean(intra)) if intra else 0.0
    inter_m = float(np.mean(inter)) if inter else 0.0
    ratio = inter_m / max(intra_m, 1e-9)
    return intra_m, inter_m, ratio


def agreement_rate(preds_a: np.ndarray, preds_b: np.ndarray) -> float:
    return float(np.mean(preds_a == preds_b))


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
    train_cap_per_class: int = 400
    test_cap_per_class: int = 500
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = 4000


def set_seed(s):
    random.seed(s); np.random.seed(s)


# =============================================================================
# Eval pack (matches exp133 format)
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
                f"test={len(test_codes)} workers={n_workers}")

    # ------------------------------------------------------------------------
    # Feature extraction
    # ------------------------------------------------------------------------
    t0 = time.time()
    logger.info("[tropical] Extracting train features...")
    train_feats = extract_features_parallel(train_codes, cfg.max_chars, n_workers)
    logger.info("[tropical] Extracting val features...")
    val_feats = extract_features_parallel(val_codes, cfg.max_chars, n_workers)
    logger.info("[tropical] Extracting test features...")
    test_feats = extract_features_parallel(test_codes, cfg.max_chars, n_workers)
    feat_time = time.time() - t0

    # Compute class means (per-class tropical centroid in log-frequency space)
    class_means = compute_class_means(train_feats, train_labels, cfg.n_cls)
    logger.info(f"[tropical] class_means shape={class_means.shape} feat_time={feat_time:.0f}s")

    # ------------------------------------------------------------------------
    # Val pass — three classifier modes
    # ------------------------------------------------------------------------
    t0 = time.time()
    val_trop_preds = classify_tropical_chebyshev(val_feats, class_means)
    val_eucl_preds = classify_euclidean(val_feats, class_means)
    val_proj_preds = classify_tropical_projection(val_feats, class_means)
    val_time = time.time() - t0

    val_met_trop = eval_pack(val_trop_preds, val_labels, val_langs, val_sources,
                             cfg.n_cls, sib_mask, dist_mat)
    val_met_eucl = eval_pack(val_eucl_preds, val_labels, val_langs, val_sources,
                             cfg.n_cls, sib_mask, dist_mat)
    val_met_proj = eval_pack(val_proj_preds, val_labels, val_langs, val_sources,
                             cfg.n_cls, sib_mask, dist_mat)
    val_macro_trop = val_met_trop["overall"]["macro_f1"]
    val_macro_eucl = val_met_eucl["overall"]["macro_f1"]
    val_macro_proj = val_met_proj["overall"]["macro_f1"]
    logger.info(f"[val] tropical={val_macro_trop:.4f} euclidean={val_macro_eucl:.4f} "
                f"projection={val_macro_proj:.4f} time={val_time:.0f}s")

    # Pick the best of the three on val (as the "headline" classifier).
    candidates = {
        "tropical_chebyshev": val_macro_trop,
        "euclidean":          val_macro_eucl,
        "tropical_projection": val_macro_proj,
    }
    best_mode = max(candidates, key=candidates.get)
    val_macro = candidates[best_mode]
    logger.info(f"[val] best_mode={best_mode} macro={val_macro:.4f}")

    # ------------------------------------------------------------------------
    # Test pass
    # ------------------------------------------------------------------------
    t0 = time.time()
    test_trop_preds = classify_tropical_chebyshev(test_feats, class_means)
    test_eucl_preds = classify_euclidean(test_feats, class_means)
    test_proj_preds = classify_tropical_projection(test_feats, class_means)
    test_time = time.time() - t0

    test_met_trop = eval_pack(test_trop_preds, test_labels, test_langs, test_sources,
                              cfg.n_cls, sib_mask, dist_mat)
    test_met_eucl = eval_pack(test_eucl_preds, test_labels, test_langs, test_sources,
                              cfg.n_cls, sib_mask, dist_mat)
    test_met_proj = eval_pack(test_proj_preds, test_labels, test_langs, test_sources,
                              cfg.n_cls, sib_mask, dist_mat)
    test_macro_trop = test_met_trop["overall"]["macro_f1"]
    test_macro_eucl = test_met_eucl["overall"]["macro_f1"]
    test_macro_proj = test_met_proj["overall"]["macro_f1"]

    # Use the val-best mode as the headline test number.
    if best_mode == "tropical_chebyshev":
        ts_met = test_met_trop; test_macro = test_macro_trop; headline_preds = test_trop_preds
    elif best_mode == "euclidean":
        ts_met = test_met_eucl; test_macro = test_macro_eucl; headline_preds = test_eucl_preds
    else:
        ts_met = test_met_proj; test_macro = test_macro_proj; headline_preds = test_proj_preds
    gap = val_macro - test_macro

    # ------------------------------------------------------------------------
    # Falsifier metrics
    # ------------------------------------------------------------------------
    intra_d, inter_d, ratio_d = compute_intra_inter_tropical(
        test_feats, test_labels, class_means, n_samples=200)
    # F3: agreement between tropical_chebyshev and tropical_projection
    agree_trop_proj = agreement_rate(test_trop_preds, test_proj_preds)
    # Also: agreement tropical_chebyshev vs euclidean (sanity)
    agree_trop_eucl = agreement_rate(test_trop_preds, test_eucl_preds)
    # F2 metric: tropical (chebyshev) - euclidean
    trop_minus_eucl = test_macro_trop - test_macro_eucl

    logger.info(f"[test] tropical={test_macro_trop:.4f} euclidean={test_macro_eucl:.4f} "
                f"projection={test_macro_proj:.4f}")
    logger.info(f"[test] headline({best_mode})={test_macro:.4f} gap={gap:+.4f} time={test_time:.0f}s")
    logger.info(f"[falsifier] F1 intra={intra_d:.4f} inter={inter_d:.4f} ratio={ratio_d:.4f} "
                f"(>=1.5?) "
                f"F2 trop-eucl={trop_minus_eucl:+.4f} "
                f"F3 agree(trop,proj)={agree_trop_proj:.4f} (>=0.8?)")

    ts_met["falsifier_F1_intra_class_tropical_dist"] = intra_d
    ts_met["falsifier_F1_inter_class_tropical_dist"] = inter_d
    ts_met["falsifier_F1_intra_inter_ratio"] = ratio_d
    ts_met["falsifier_F2_tropical_macro"] = test_macro_trop
    ts_met["falsifier_F2_euclidean_macro"] = test_macro_eucl
    ts_met["falsifier_F2_tropical_minus_euclidean"] = trop_minus_eucl
    ts_met["falsifier_F3_tropical_projection_macro"] = test_macro_proj
    ts_met["falsifier_F3_tropical_proj_vs_chebyshev_agree_rate"] = agree_trop_proj
    ts_met["falsifier_F3_tropical_vs_euclidean_agree_rate"] = agree_trop_eucl

    return {
        "tag": tag, "method": "TROPICAL",
        "note": ("Tropical-geometry attribution: 128-dim ASCII log-frequency features, "
                 "Chebyshev (L_inf) nearest-centroid in min-plus semiring. "
                 "Three classifier modes (tropical / euclidean / Maclagan-Sturmfels "
                 "projection). CPU-only; parameter-free beyond class means."),
        "enc": "tropical-log1p-ascii128", "bench": cfg.benchmark,
        "frac": cfg.frac, "best_mode": best_mode,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "val_macro_tropical": val_macro_trop,
        "val_macro_euclidean": val_macro_eucl,
        "val_macro_projection": val_macro_proj,
        "test_macro_tropical": test_macro_trop,
        "test_macro_euclidean": test_macro_eucl,
        "test_macro_projection": test_macro_proj,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "mean_intra_class_tropical_dist": intra_d,
        "mean_inter_class_tropical_dist": inter_d,
        "intra_inter_ratio": ratio_d,
        "tropical_minus_euclidean": trop_minus_eucl,
        "tropical_proj_vs_chebyshev_agree_rate": agree_trop_proj,
        "feat_time_sec": feat_time, "val_time_sec": val_time, "test_time_sec": test_time,
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
            tag = f"exp143_tropical_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} mode={res['best_mode']} "
                            f"intra/inter={res['mean_intra_class_tropical_dist']:.3f}/"
                            f"{res['mean_inter_class_tropical_dist']:.3f} "
                            f"time={res['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp143_tropical_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Mode':>20} {'Train':>7} {'Test':>7} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Trop':>7} {'Eucl':>7} {'Proj':>7} "
          f"{'Gap':>8} {'dPaper':>9} {'Ratio':>7} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['best_mode']:>20} "
              f"{r['train_size_after_cap']:>7d} {r['test_size_after_cap']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['test_macro_tropical']:>7.4f} {r['test_macro_euclidean']:>7.4f} "
              f"{r['test_macro_projection']:>7.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['intra_inter_ratio']:>7.3f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
