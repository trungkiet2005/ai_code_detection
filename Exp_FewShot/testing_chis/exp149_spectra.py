# exp149 — SPECTRA
# =============================================================================
# NAME       : SPECTRA (Laplacian-Spectrum Signature of Per-Author Character-
#              Transition Graphs)
# REFERENCE  : Chung (1997) "Spectral Graph Theory" (CBMS-92); Belkin & Niyogi
#              (2003) "Laplacian Eigenmaps for Dimensionality Reduction and
#              Data Representation", NeurIPS. Spectral graph theory is a
#              classical area of mathematics applied widely (community
#              detection, image segmentation, molecular biology) but NEVER
#              to AI-code authorship attribution.
# CLAIM      : An author's identity is the SPECTRUM (sorted top-k eigenvalues)
#              of the normalized Laplacian of their character-transition
#              graph. Two authors with similar spectra write structurally
#              similar code, regardless of which specific characters they
#              prefer.
# EQUATION   : A_c[i,j] = sum_{x in train_c} count_bigram(x, (i,j));
#              L_c = I - D_c^{-1/2} A_c D_c^{-1/2};
#              spectrum_c = top-k eigenvalues of L_c;
#              y_hat = argmin_c ||spectrum_q - spectrum_c||_2.
# WHY NEW    : Graph Laplacian spectra are a fundamental algebraic invariant
#              studied for 80+ years (Chung 1997). Applied to community
#              detection, image segmentation, molecular biology — NEVER to
#              code authorship. The spectrum is INVARIANT to specific
#              character identities (a permutation of the vocabulary leaves
#              the spectrum unchanged), so this captures pure STRUCTURE,
#              not character-frequency leakage.
# WOW HOOK   : "Author identity is an algebraic invariant. The Laplacian
#              spectrum of their character-transition graph is a fingerprint
#              that survives vocabulary permutation — capturing pure
#              structural style. Two authors with the same spectrum write
#              the same way, even with different alphabets."
# FALSIFIER  : (F1) Per-class spectra are DIFFERENT: mean L2 distance between
#              class-spectra exceeds within-class sample-to-class-spectrum
#              distance by ratio >= 1.5. Report falsifier_F1_intra_inter_ratio.
#              (F2) Spectrum-based attribution beats raw-adjacency-flatten
#              L2 baseline by >= 0.005. Report
#              falsifier_F2_spectrum_minus_raw_macro.
#              (F3) Spectrum stability: random 50% sub-sampling of training
#              for spectrum computation keeps Spearman of class-distance
#              ranking >= 0.7. Report
#              falsifier_F3_spearman_50pct_subsample.
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
from scipy.linalg import eigvalsh
from scipy.stats import spearmanr
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp149_spectra")

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
# Spectral core
# =============================================================================

VOCAB_SIZE = 128   # printable ASCII range (well, all 0..127)
K_EIGS = 32        # top-k eigenvalues
MAX_CHARS = 4000


def _bigram_adj(code: str, V: int = VOCAB_SIZE, max_chars: int = MAX_CHARS) -> np.ndarray:
    """Count-matrix of character bigrams. (V, V) numpy array."""
    code = code[:max_chars]
    if len(code) < 2:
        return np.zeros((V, V), dtype=np.float64)
    arr = np.frombuffer(code.encode("ascii", errors="replace"), dtype=np.uint8).astype(np.int64)
    # Clip to vocab range
    arr = np.clip(arr, 0, V - 1)
    src = arr[:-1]
    dst = arr[1:]
    A = np.zeros((V, V), dtype=np.float64)
    # Use bincount for vectorised count
    idx = src * V + dst
    counts = np.bincount(idx, minlength=V * V)
    A = counts.astype(np.float64).reshape(V, V)
    return A


def _bigram_adj_one(args):
    code, V, max_chars = args
    return _bigram_adj(code, V, max_chars)


def _build_adjs_parallel(codes: List[str], V: int = VOCAB_SIZE,
                         max_chars: int = MAX_CHARS,
                         n_workers: int = None,
                         desc: str = "adj") -> List[np.ndarray]:
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    args_list = [(c, V, max_chars) for c in codes]
    if n_workers == 1 or len(codes) < 32:
        return [_bigram_adj_one(a) for a in tqdm(args_list, desc=desc)]
    try:
        with mp.Pool(n_workers) as pool:
            return list(tqdm(pool.imap(_bigram_adj_one, args_list, chunksize=8),
                             total=len(codes), desc=desc))
    except Exception as e:
        logger.warning(f"[adj] multiprocessing failed ({e}); falling back to serial")
        return [_bigram_adj_one(a) for a in tqdm(args_list, desc=desc)]


def _normalised_laplacian_spectrum(A: np.ndarray, k: int = K_EIGS,
                                    eps: float = 1e-6) -> np.ndarray:
    """Compute top-k eigenvalues of L = I - D^{-1/2} A_sym D^{-1/2}.
    Returns ascending-sorted top-k eigenvalues (descending order kept then we sort
    for L2-comparison stability)."""
    # Symmetrise
    A_sym = 0.5 * (A + A.T)
    # Degree
    d = A_sym.sum(axis=1)
    d_safe = np.where(d > eps, d, 1.0)
    d_inv_sqrt = 1.0 / np.sqrt(d_safe)
    # Mask out isolated vertices to keep L well-defined
    iso = (d <= eps)
    # L = I - D^{-1/2} A D^{-1/2}
    L = np.eye(A_sym.shape[0]) - (d_inv_sqrt[:, None] * A_sym * d_inv_sqrt[None, :])
    # For isolated nodes, set L row/col to identity-like (self only)
    if iso.any():
        L[iso, :] = 0.0
        L[:, iso] = 0.0
        L[iso, iso] = 1.0
    # eigvalsh returns ascending; we take the LARGEST k (top of spectrum)
    eigs = eigvalsh(L)  # length V
    # Top-k by magnitude (largest values)
    top = eigs[-k:][::-1]  # descending
    # Pad if k > V (shouldn't happen)
    if len(top) < k:
        pad = np.zeros(k - len(top), dtype=np.float64)
        top = np.concatenate([top, pad])
    return top.astype(np.float64)


def _spectrum_one(args):
    A, k = args
    try:
        return _normalised_laplacian_spectrum(A, k=k)
    except Exception:
        return np.zeros(k, dtype=np.float64)


def _spectra_parallel(adjs: List[np.ndarray], k: int = K_EIGS,
                       n_workers: int = None, desc: str = "spec") -> np.ndarray:
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    args_list = [(A, k) for A in adjs]
    if n_workers == 1 or len(adjs) < 16:
        out = [_spectrum_one(a) for a in tqdm(args_list, desc=desc)]
    else:
        try:
            with mp.Pool(n_workers) as pool:
                out = list(tqdm(pool.imap(_spectrum_one, args_list, chunksize=4),
                                total=len(adjs), desc=desc))
        except Exception as e:
            logger.warning(f"[spec] multiprocessing failed ({e}); falling back to serial")
            out = [_spectrum_one(a) for a in tqdm(args_list, desc=desc)]
    return np.stack(out, axis=0)


def _class_aggregated_adjs(adjs: List[np.ndarray], labels: List[int],
                            n_cls: int) -> List[np.ndarray]:
    """Sum per-class adjacency matrices."""
    V = adjs[0].shape[0]
    A_cls = [np.zeros((V, V), dtype=np.float64) for _ in range(n_cls)]
    for A, lbl in zip(adjs, labels):
        if 0 <= lbl < n_cls:
            A_cls[lbl] += A
    return A_cls


def _spec_classify(query_specs: np.ndarray, class_specs: np.ndarray) -> np.ndarray:
    """Predict y_hat = argmin_c ||spectrum_q - spectrum_c||_2.
    query_specs: (N, k); class_specs: (n_cls, k)."""
    # Sort each row ascending for permutation-invariant comparison
    q_sorted = np.sort(query_specs, axis=1)
    c_sorted = np.sort(class_specs, axis=1)
    diff = q_sorted[:, None, :] - c_sorted[None, :, :]
    dists = np.linalg.norm(diff, axis=2)  # (N, n_cls)
    return np.argmin(dists, axis=1), dists


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
    test_cap_per_class:  int = 400
    val_cap_per_class:   int = 200
    k_eigs: int = K_EIGS
    vocab_size: int = VOCAB_SIZE
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = MAX_CHARS


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

    tr_data_frac   = stratified_subsample(tr_data, cfg.frac,                     seed=cfg.seed)
    tr_data_capped = stratified_subsample(tr_data_frac, cfg.train_cap_per_class, seed=cfg.seed + 1)
    ts_data_capped = stratified_subsample(ts_data, cfg.test_cap_per_class,       seed=cfg.seed + 2)
    vl_data_capped = stratified_subsample(vl_data, cfg.val_cap_per_class,        seed=cfg.seed + 3)

    train_codes  = [r["code"][:cfg.max_chars] for r in tr_data_capped]
    train_labels = list(tr_data_capped["label"])
    val_codes    = [r["code"][:cfg.max_chars] for r in vl_data_capped]
    val_labels   = list(vl_data_capped["label"])
    val_langs    = [r.get("language", "") or "" for r in vl_data_capped]
    val_sources  = [r.get("source", "") or ""   for r in vl_data_capped]
    test_codes   = [r["code"][:cfg.max_chars] for r in ts_data_capped]
    test_labels  = list(ts_data_capped["label"])
    test_langs   = [r.get("language", "") or "" for r in ts_data_capped]
    test_sources = [r.get("source", "") or ""   for r in ts_data_capped]

    n_workers = cfg.n_workers if cfg.n_workers > 0 else max(1, mp.cpu_count() - 1)
    logger.info(f"[setup] frac={cfg.frac} train={len(train_codes)} "
                f"val={len(val_codes)} test={len(test_codes)} n_cls={cfg.n_cls} "
                f"V={cfg.vocab_size} k={cfg.k_eigs}")

    # --- Train: build per-sample adjacencies, aggregate per class ---
    t0 = time.time()
    train_adjs = _build_adjs_parallel(train_codes, V=cfg.vocab_size,
                                       max_chars=cfg.max_chars,
                                       n_workers=n_workers, desc="adj(train)")
    A_cls = _class_aggregated_adjs(train_adjs, train_labels, cfg.n_cls)
    # Spectrum per class
    class_specs = np.stack(
        [_normalised_laplacian_spectrum(A_cls[c], k=cfg.k_eigs)
         for c in range(cfg.n_cls)], axis=0
    )
    logger.info(f"[spec] class_specs shape={class_specs.shape} adj_time={time.time()-t0:.0f}s")

    # --- Val ---
    val_adjs  = _build_adjs_parallel(val_codes,  V=cfg.vocab_size,
                                      max_chars=cfg.max_chars,
                                      n_workers=n_workers, desc="adj(val)")
    val_specs = _spectra_parallel(val_adjs, k=cfg.k_eigs,
                                   n_workers=n_workers, desc="spec(val)")
    val_preds, _ = _spec_classify(val_specs, class_specs)
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                        cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]
    logger.info(f"[val] macro={val_macro:.4f}")

    # --- Test ---
    t1 = time.time()
    test_adjs  = _build_adjs_parallel(test_codes, V=cfg.vocab_size,
                                       max_chars=cfg.max_chars,
                                       n_workers=n_workers, desc="adj(test)")
    test_specs = _spectra_parallel(test_adjs, k=cfg.k_eigs,
                                    n_workers=n_workers, desc="spec(test)")
    test_preds, test_dists = _spec_classify(test_specs, class_specs)
    test_time = time.time() - t1
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro
    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} time={test_time:.0f}s")

    # ---- F1: intra/inter spectrum distance ratio ----
    # Intra: mean ||spectrum_sample - spectrum_class(true)||
    # Inter: mean ||spectrum_class_i - spectrum_class_j|| (i != j)
    intra_list = []
    test_specs_sorted = np.sort(test_specs, axis=1)
    class_specs_sorted = np.sort(class_specs, axis=1)
    for i, lbl in enumerate(test_labels):
        if 0 <= lbl < cfg.n_cls:
            intra_list.append(float(np.linalg.norm(
                test_specs_sorted[i] - class_specs_sorted[lbl])))
    intra_mean = float(np.mean(intra_list)) if intra_list else float("nan")
    inter_list = []
    for i in range(cfg.n_cls):
        for j in range(cfg.n_cls):
            if i != j:
                inter_list.append(float(np.linalg.norm(
                    class_specs_sorted[i] - class_specs_sorted[j])))
    inter_mean = float(np.mean(inter_list)) if inter_list else float("nan")
    F1_ratio = (inter_mean / intra_mean) if intra_mean and intra_mean > 0 else float("nan")
    F1_pass = bool(F1_ratio >= 1.5)

    # ---- F2: raw-adjacency flatten L2 baseline ----
    # Flatten per-class adjacency to vector, L2-normalise, compare to test sample's flattened adj.
    # This is HUGE (V*V=16384). To avoid blowing memory we compute per-sample distances on the fly.
    def _flat_norm(A):
        v = A.flatten()
        n = np.linalg.norm(v) + 1e-12
        return (v / n).astype(np.float32)

    cls_flat = np.stack([_flat_norm(A_cls[c]) for c in range(cfg.n_cls)], axis=0)
    raw_preds = np.zeros(len(test_codes), dtype=np.int64)
    for i, A in enumerate(test_adjs):
        v = _flat_norm(A)
        d = np.linalg.norm(cls_flat - v[None, :], axis=1)
        raw_preds[i] = int(np.argmin(d))
    raw_macro = float(f1_score(test_labels, raw_preds, average="macro", zero_division=0))
    F2_delta = test_macro - raw_macro
    F2_pass = bool(F2_delta >= 0.005)
    logger.info(f"[F2] spectrum={test_macro:.4f} raw={raw_macro:.4f} delta={F2_delta:+.4f}")

    # ---- F3: spectrum stability under 50% subsampling ----
    # Recompute class spectra using 50% random subset of training samples.
    rng = random.Random(cfg.seed + 7)
    keep_idx = [i for i in range(len(train_codes)) if rng.random() < 0.5]
    if len(keep_idx) >= cfg.n_cls:
        train_adjs_sub = [train_adjs[i] for i in keep_idx]
        train_labels_sub = [train_labels[i] for i in keep_idx]
        A_cls_sub = _class_aggregated_adjs(train_adjs_sub, train_labels_sub, cfg.n_cls)
        class_specs_sub = np.stack(
            [_normalised_laplacian_spectrum(A_cls_sub[c], k=cfg.k_eigs)
             for c in range(cfg.n_cls)], axis=0
        )
        # For each test sample, compute distance-to-class-spectrum vector under both
        # the FULL and SUBSAMPLE spectra, then Spearman across the (N * n_cls) flattened rankings.
        _, dists_full = _spec_classify(test_specs, class_specs)
        _, dists_sub  = _spec_classify(test_specs, class_specs_sub)
        full_flat = dists_full.flatten()
        sub_flat  = dists_sub.flatten()
        try:
            rho, _ = spearmanr(full_flat, sub_flat)
            rho = float(rho) if not np.isnan(rho) else 0.0
        except Exception:
            rho = 0.0
    else:
        rho = float("nan")
    F3_pass = bool(rho >= 0.7)

    ts_met["falsifier_F1_intra_mean"]              = intra_mean
    ts_met["falsifier_F1_inter_mean"]              = inter_mean
    ts_met["falsifier_F1_intra_inter_ratio"]       = F1_ratio
    ts_met["falsifier_F1_pass"]                    = F1_pass
    ts_met["falsifier_F2_raw_macro"]               = raw_macro
    ts_met["falsifier_F2_spectrum_minus_raw_macro"]= F2_delta
    ts_met["falsifier_F2_pass"]                    = F2_pass
    ts_met["falsifier_F3_spearman_50pct_subsample"]= rho
    ts_met["falsifier_F3_pass"]                    = F3_pass

    return {
        "tag": tag, "method": "SPECTRA",
        "note": ("Laplacian spectrum of per-author character-transition graphs. "
                 "Attribution = argmin L2 distance between sorted top-k eigenvalue "
                 "vectors. Permutation-invariant. CPU-only."),
        "enc": f"char-bigram(V={VOCAB_SIZE})/Lspec(k={K_EIGS})",
        "bench": cfg.benchmark, "frac": cfg.frac,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "falsifier_F1_intra_inter_ratio":         F1_ratio,
        "falsifier_F2_spectrum_minus_raw_macro":  F2_delta,
        "falsifier_F2_raw_macro":                 raw_macro,
        "falsifier_F3_spearman_50pct_subsample":  rho,
        "test_time_sec": test_time,
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
            tag = f"exp149_spectra_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"F1ratio={res['falsifier_F1_intra_inter_ratio']:.3f} "
                            f"F2={res['falsifier_F2_spectrum_minus_raw_macro']:+.4f} "
                            f"F3rho={res['falsifier_F3_spearman_50pct_subsample']:.3f}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()

    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp149_spectra_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Train':>7} {'Test':>7} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'F1ratio':>9} {'F2dlt':>9} {'F3rho':>8} {'Wall':>8}")
    print("-" * 140)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['train_size_after_cap']:>7d} {r['test_size_after_cap']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} "
              f"{r['falsifier_F1_intra_inter_ratio']:>9.3f} "
              f"{r['falsifier_F2_spectrum_minus_raw_macro']:>+9.4f} "
              f"{r['falsifier_F3_spearman_50pct_subsample']:>8.3f} "
              f"{r['wall']:>8.0f}s")
    print("=" * 140)


if __name__ == "__main__":
    main()
