# exp135 — POLYGLOT
# =============================================================================
# NAME       : POLYGLOT (Multi-compressor Kolmogorov spectrum for code attribution)
# REFERENCE  : new theory contribution; extends Cilibrasi-Vitanyi 2005 NCD
#              (gzip only) and Jiang 2023 to ENSEMBLE compression.
# CLAIM      : Each compressor (gzip, bzip2, lzma, zstd, brotli) approximates
#              a DIFFERENT aspect of Kolmogorov complexity. An author's
#              fingerprint is the K-DIMENSIONAL VECTOR of normalized
#              compression rates across compressors — their "Kolmogorov
#              spectrum". Authors with similar code style have similar
#              spectra; the DISAGREEMENT between compressors IS the signal.
# EQUATION   : For each query x and each compressor i in {gzip, bzip2, lzma,
#              zstd, brotli}:
#                r_i(x) = |compress_i(x)| / |x|     # normalized rate
#              spectrum(x) = (r_1, r_2, ..., r_K)
#              y_hat = argmin_a Euclidean(spectrum(x), mean_a_spectrum)
# WHY NEW    : Jiang 2023 + Cilibrasi 2005 both use ONE compressor (gzip).
#              POLYGLOT is the first to use ENSEMBLE compression as a
#              feature vector for any classification task.
# WOW HOOK   : "Every author has a Kolmogorov spectrum — the disagreement
#              between gzip, bzip2, lzma, zstd, brotli on their code. We
#              attribute by matching spectra. Five compressors are stronger
#              than one because each measures a different aspect of
#              algorithmic complexity."
# FALSIFIER  : (F1) Ensemble (5-dim spectrum) > best single-compressor by
#              >= 0.02 — multi-compressor diversity helps. (F2) Per-
#              compressor importance via leave-one-out: each compressor
#              must contribute >= 0.003. (F3) Spectrum variance per class
#              < spectrum variance across class (within-class compactness).
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
import gzip, bz2, lzma
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("datasets"); _ensure("scikit-learn"); _ensure("tqdm")
_ensure("zstandard"); _ensure("brotli")

import numpy as np
import zstandard as zstd
import brotli
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp135_polyglot")

PAPER_BASELINE = 0.6633
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD  = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}

COMPRESSORS = ["gzip", "bzip2", "lzma", "zstd", "brotli"]
N_COMPRESSORS = len(COMPRESSORS)


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
    """If frac_or_n_per_class < 1.0: treat as fraction per class.
    If >= 1.0: treat as max count per class."""
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
# POLYGLOT core: 5-compressor spectrum
# =============================================================================

# A module-level zstd compressor instance is NOT picklable cleanly across workers,
# so we instantiate inside compute_spectrum (cheap).

def compute_spectrum(text: str) -> np.ndarray:
    """Compute 5-dim Kolmogorov spectrum (normalized compression rates).

    Returns: np.float32 array of length 5 in order
             [gzip, bzip2, lzma, zstd, brotli].
    """
    b = text.encode("utf-8", errors="replace")
    n = max(len(b), 1)
    try:
        r_gzip = len(gzip.compress(b, compresslevel=9)) / n
    except Exception:
        r_gzip = 1.0
    try:
        r_bz2 = len(bz2.compress(b, compresslevel=9)) / n
    except Exception:
        r_bz2 = 1.0
    try:
        r_lzma = len(lzma.compress(b, preset=6)) / n
    except Exception:
        r_lzma = 1.0
    try:
        cctx = zstd.ZstdCompressor(level=22)
        r_zstd = len(cctx.compress(b)) / n
    except Exception:
        r_zstd = 1.0
    try:
        r_brotli = len(brotli.compress(b, quality=11)) / n
    except Exception:
        r_brotli = 1.0
    return np.array([r_gzip, r_bz2, r_lzma, r_zstd, r_brotli], dtype=np.float32)


def _spectrum_worker(args):
    """Worker for multiprocessing pool.
    args = (idx, text)
    Returns: (idx, spectrum_5dim).
    """
    idx, text = args
    return idx, compute_spectrum(text)


def compute_spectra_parallel(texts: List[str], n_workers: int, desc: str = "spectra"):
    """Compute spectra for a list of texts in parallel. Returns (N, 5) float32."""
    N = len(texts)
    out = np.zeros((N, N_COMPRESSORS), dtype=np.float32)
    args_list = [(i, texts[i]) for i in range(N)]
    if n_workers <= 1:
        for args in tqdm(args_list, desc=desc):
            idx, sp = _spectrum_worker(args)
            out[idx] = sp
        return out
    try:
        with mp.Pool(n_workers) as pool:
            for idx, sp in tqdm(pool.imap_unordered(_spectrum_worker, args_list, chunksize=8),
                                 total=N, desc=desc):
                out[idx] = sp
    except Exception as e:
        logger.warning(f"[polyglot] mp failed ({e}); serial fallback")
        for args in tqdm(args_list, desc=desc):
            idx, sp = _spectrum_worker(args)
            out[idx] = sp
    return out


def build_centroids(train_spectra: np.ndarray, train_labels: List[int], n_cls: int):
    """Compute mean spectrum per class. Returns (n_cls, 5) float32."""
    cent = np.zeros((n_cls, N_COMPRESSORS), dtype=np.float32)
    counts = np.zeros(n_cls, dtype=np.int64)
    tl = np.asarray(train_labels)
    for c in range(n_cls):
        sel = (tl == c)
        if sel.sum() > 0:
            cent[c] = train_spectra[sel].mean(axis=0)
            counts[c] = int(sel.sum())
        else:
            cent[c] = np.full(N_COMPRESSORS, np.inf, dtype=np.float32)
    return cent, counts


def predict_by_centroid(test_spectra: np.ndarray, centroids: np.ndarray,
                         compressor_mask: np.ndarray = None) -> np.ndarray:
    """Argmin Euclidean distance to centroids.

    compressor_mask: optional bool array of length 5 to select features
                     (used for leave-one-out falsifier).
    """
    if compressor_mask is None:
        compressor_mask = np.ones(N_COMPRESSORS, dtype=bool)
    ts = test_spectra[:, compressor_mask]
    cs = centroids[:, compressor_mask]
    # (N, 1, D) - (1, K, D) -> (N, K, D) -> (N, K)
    dists = np.linalg.norm(ts[:, None, :] - cs[None, :, :], axis=-1)
    return np.argmin(dists, axis=1)


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
    n_workers: int = -1  # -1 -> cpu_count() - 1
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
# Falsifier: leave-one-out compressor importance + spectrum compactness
# =============================================================================

def per_compressor_loo(test_spectra, centroids, test_labels, n_cls):
    """For each compressor, drop it from the spectrum, recompute macro-F1.
    Return list of (full_macro - loo_macro) per compressor.
    """
    full_preds = predict_by_centroid(test_spectra, centroids)
    full_macro = f1_score(test_labels, full_preds, average="macro", zero_division=0,
                          labels=list(range(n_cls)))
    drops = []
    per_compressor_macros = []
    for i in range(N_COMPRESSORS):
        mask = np.ones(N_COMPRESSORS, dtype=bool); mask[i] = False
        preds_loo = predict_by_centroid(test_spectra, centroids, compressor_mask=mask)
        macro_loo = f1_score(test_labels, preds_loo, average="macro", zero_division=0,
                              labels=list(range(n_cls)))
        drops.append(float(full_macro - macro_loo))
        per_compressor_macros.append(float(macro_loo))
    # Per-single-compressor scores (just that one feature on its own)
    per_single = []
    for i in range(N_COMPRESSORS):
        mask = np.zeros(N_COMPRESSORS, dtype=bool); mask[i] = True
        preds_s = predict_by_centroid(test_spectra, centroids, compressor_mask=mask)
        macro_s = f1_score(test_labels, preds_s, average="macro", zero_division=0,
                            labels=list(range(n_cls)))
        per_single.append(float(macro_s))
    return {
        "full_macro":               float(full_macro),
        "per_compressor_loo_drop":  drops,
        "per_compressor_loo_macro": per_compressor_macros,
        "per_single_compressor_macro": per_single,
        "best_single_compressor":   COMPRESSORS[int(np.argmax(per_single))],
        "best_single_macro":        float(max(per_single)),
        "ensemble_gain_over_best_single": float(full_macro - max(per_single)),
    }


def spectrum_compactness(train_spectra, train_labels, centroids, n_cls):
    """Mean within-class L2 distance to own centroid; mean between-centroid L2.
    Within < Between => compact, F3 satisfied.
    """
    ts = np.asarray(train_spectra)
    tl = np.asarray(train_labels)
    intra_dists = []
    for c in range(n_cls):
        sel = (tl == c)
        if sel.sum() < 2: continue
        d = np.linalg.norm(ts[sel] - centroids[c][None, :], axis=1)
        intra_dists.append(float(d.mean()))
    inter_dists = []
    for i in range(n_cls):
        for j in range(n_cls):
            if i >= j: continue
            if not np.isfinite(centroids[i]).all(): continue
            if not np.isfinite(centroids[j]).all(): continue
            inter_dists.append(float(np.linalg.norm(centroids[i] - centroids[j])))
    return {
        "mean_intra_class_spectrum_dist": float(np.mean(intra_dists)) if intra_dists else None,
        "mean_inter_centroid_spectrum_dist": float(np.mean(inter_dists)) if inter_dists else None,
        "compactness_ratio": float(np.mean(intra_dists) / max(np.mean(inter_dists), 1e-9))
                              if (intra_dists and inter_dists) else None,
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

    # Compute spectra
    t0 = time.time()
    train_spectra = compute_spectra_parallel(train_codes, n_workers, desc="train-spectra")
    val_spectra   = compute_spectra_parallel(val_codes, n_workers, desc="val-spectra")
    test_spectra  = compute_spectra_parallel(test_codes, n_workers, desc="test-spectra")
    spec_time = time.time() - t0

    # Build per-class mean spectrum
    centroids, counts = build_centroids(train_spectra, train_labels, cfg.n_cls)
    logger.info(f"[centroids] per-class counts={counts.tolist()}")
    logger.info(f"[centroids] spectra_time={spec_time:.0f}s")

    # Val pass
    t0 = time.time()
    val_preds = predict_by_centroid(val_spectra, centroids)
    val_time = time.time() - t0
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                         cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]
    logger.info(f"[val] macro={val_macro:.4f} pred_time={val_time:.2f}s")

    # Test pass
    t0 = time.time()
    test_preds = predict_by_centroid(test_spectra, centroids)
    test_time = time.time() - t0
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro

    # Falsifier metrics
    loo_metrics = per_compressor_loo(test_spectra, centroids, test_labels, cfg.n_cls)
    compact_metrics = spectrum_compactness(train_spectra, train_labels, centroids, cfg.n_cls)

    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} pred_time={test_time:.2f}s")
    logger.info(f"[F1] best_single={loo_metrics['best_single_compressor']} "
                f"({loo_metrics['best_single_macro']:.4f}); "
                f"ensemble_gain={loo_metrics['ensemble_gain_over_best_single']:+.4f}")
    logger.info(f"[F2] per_compressor_loo_drop={[round(d,4) for d in loo_metrics['per_compressor_loo_drop']]}")
    logger.info(f"[F3] intra={compact_metrics['mean_intra_class_spectrum_dist']} "
                f"inter={compact_metrics['mean_inter_centroid_spectrum_dist']} "
                f"ratio={compact_metrics['compactness_ratio']}")

    ts_met["falsifier_F1_ensemble_gain_over_best_single"] = loo_metrics["ensemble_gain_over_best_single"]
    ts_met["falsifier_F1_best_single_compressor"] = loo_metrics["best_single_compressor"]
    ts_met["falsifier_F1_per_single_compressor_macro"] = dict(zip(COMPRESSORS, loo_metrics["per_single_compressor_macro"]))
    ts_met["falsifier_F2_per_compressor_loo_drop"] = dict(zip(COMPRESSORS, loo_metrics["per_compressor_loo_drop"]))
    ts_met["falsifier_F3_mean_intra_class_spectrum_dist"] = compact_metrics["mean_intra_class_spectrum_dist"]
    ts_met["falsifier_F3_mean_inter_centroid_spectrum_dist"] = compact_metrics["mean_inter_centroid_spectrum_dist"]
    ts_met["falsifier_F3_compactness_ratio"] = compact_metrics["compactness_ratio"]
    ts_met["centroids"] = centroids.tolist()

    return {
        "tag": tag, "method": "POLYGLOT",
        "note": ("Multi-compressor Kolmogorov spectrum (gzip+bzip2+lzma+zstd+brotli). "
                 "Per-author mean spectrum; argmin-Euclidean attribution. CPU-only, zero learning."),
        "enc": "ensemble-compressors", "bench": cfg.benchmark,
        "frac": cfg.frac,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "spec_time_sec": spec_time,
        "val_time_sec": val_time, "test_time_sec": test_time,
        "ensemble_gain_over_best_single": loo_metrics["ensemble_gain_over_best_single"],
        "best_single_compressor": loo_metrics["best_single_compressor"],
        "best_single_macro": loo_metrics["best_single_macro"],
        "per_compressor_loo_drop": dict(zip(COMPRESSORS, loo_metrics["per_compressor_loo_drop"])),
        "per_single_compressor_macro": dict(zip(COMPRESSORS, loo_metrics["per_single_compressor_macro"])),
        "mean_intra_class_spectrum_dist": compact_metrics["mean_intra_class_spectrum_dist"],
        "mean_inter_centroid_spectrum_dist": compact_metrics["mean_inter_centroid_spectrum_dist"],
        "compactness_ratio": compact_metrics["compactness_ratio"],
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
            tag = f"exp135_polyglot_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"ens_gain={res['ensemble_gain_over_best_single']:+.4f}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp135_polyglot_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Train':>8} {'Test':>8} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'BestSingle':>12} {'BestF1':>8} {'EnsGain':>9} {'Wall':>8}")
    print("-"*140)
    for r in results:
        bs = r.get("best_single_compressor") or "-"
        bsm = r.get("best_single_macro") or 0.0
        eg = r.get("ensemble_gain_over_best_single") or 0.0
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['train_size_after_cap']:>8d} {r['test_size_after_cap']:>8d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {bs:>12} {bsm:>8.4f} {eg:>+9.4f} {r['wall']:>8.0f}s")
    print("="*140)
    # Print per-compressor LOO breakdown
    print("\nLeave-One-Out compressor drop (F2: each should be >= 0.003):")
    print(f"{'Tag':<40} " + " ".join(f"{c:>10}" for c in COMPRESSORS))
    for r in results:
        drops = r.get("per_compressor_loo_drop") or {}
        cells = " ".join(f"{drops.get(c, 0.0):>+10.4f}" for c in COMPRESSORS)
        print(f"{r['tag']:<40} {cells}")
    print("="*140)


if __name__ == "__main__":
    main()
