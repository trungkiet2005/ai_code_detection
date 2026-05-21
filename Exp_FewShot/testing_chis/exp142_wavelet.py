# exp142 — WAVELET
# =============================================================================
# NAME       : WAVELET (Discrete wavelet transform energy signature for code authorship)
# REFERENCE  : Mallat 1989 ("A theory for multiresolution signal
#              decomposition"), Daubechies 1992 ("Ten Lectures on Wavelets"),
#              sklearn-wavelets / PyWavelets package. Never applied to AI
#              code attribution.
# CLAIM      : Code, viewed as a 1D signal of UTF-8 byte values, has a
#              multi-resolution energy distribution that varies by author.
#              We apply Discrete Wavelet Transform (Daubechies-4) and
#              extract per-level energy statistics. Each author has a
#              unique "energy signature" in wavelet space — coarse-scale
#              energy captures structural patterns (indent depth, line
#              length), fine-scale captures lexical patterns (variable
#              naming, operator usage).
# EQUATION   : signal(x) = [ord(c) for c in x]  in N^L
#              At each DWT level l in {1, ..., 5}: (cA_l, cD_l) = wavelet
#              decompose with db4 mother wavelet.
#              energy_l = log(1 + sum(cD_l^2) / |cD_l|)
#              signature(x) = [energy_1, ..., energy_5, energy_approx] in R^6
#              y_hat = argmin_a Euclidean(signature(x), mean_signature_a)
# WHY NEW    : Wavelets have been applied to speech, image, ECG, financial
#              time-series. Never to code authorship at the byte level.
#              We bring multi-resolution analysis from signal processing
#              to authorship attribution; the multi-scale view captures
#              regularities that flat token statistics miss.
# WOW HOOK   : "Every author has a wavelet energy SIGNATURE — a multi-
#              resolution profile of their code rhythm. Daubechies'
#              orthonormal wavelets from 1992 separate LLM authors by
#              the energy each frequency band carries. Wavelets are the
#              musical notation of code."
# FALSIFIER  : (F1) Per-level energy variance: cross-class F-stat > 5 at
#              each of the 5 wavelet levels.
#              (F2) WAVELET composite > raw mean/std byte baseline by >= 0.01.
#              (F3) Different wavelet families (db4 vs sym4 vs haar) give
#              consistent rankings (Spearman correlation >= 0.7).
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _ensure(pkg, imp_name=None):
    name = imp_name or pkg.split(".")[0]
    if importlib.util.find_spec(name) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("scipy"); _ensure("datasets")
_ensure("scikit-learn", "sklearn"); _ensure("tqdm")
_ensure("PyWavelets", "pywt")

import numpy as np
import pywt
from scipy import stats as scipy_stats
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp142_wavelet")

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
# Wavelet signature core
# =============================================================================

def wavelet_signature(code, max_len=4096, wavelet='db4', levels=5):
    """Compute the multi-resolution wavelet energy signature of a code string.

    Steps:
      1. Convert code to a 1D signal of byte values.
      2. Pad or truncate to a power-of-2 length (>= 2**levels).
      3. Multi-level DWT with the chosen mother wavelet.
      4. Per-level log-energy of detail coefficients + log-energy of the
         final approximation coefficient.

    Returns a numpy array of shape (levels + 1,).
    """
    arr = np.frombuffer(code[:max_len].encode("utf-8", errors="ignore"),
                        dtype=np.uint8).astype(np.float32)
    if len(arr) < 2 ** levels:
        mean_v = float(arr.mean()) if len(arr) > 0 else 0.0
        pad = 2 ** levels - len(arr)
        arr = np.concatenate([arr, np.full(pad, mean_v, dtype=np.float32)])
    # Truncate to nearest power of 2 (clean multi-level decomposition).
    target_len = 2 ** int(np.log2(len(arr)))
    arr = arr[:target_len]
    coeffs = pywt.wavedec(arr, wavelet, level=levels)
    # coeffs = [cA_levels, cD_levels, cD_{levels-1}, ..., cD_1]
    energies = []
    for c in coeffs:
        if len(c) == 0:
            energies.append(0.0)
        else:
            e = float(np.log1p(np.sum(np.asarray(c, dtype=np.float64) ** 2)
                                / len(c)))
            energies.append(e)
    return np.array(energies, dtype=np.float32)


def meanstd_signature(code, max_len=4096):
    """Raw mean/std byte signature (F2 baseline) — no multi-resolution."""
    arr = np.frombuffer(code[:max_len].encode("utf-8", errors="ignore"),
                        dtype=np.uint8).astype(np.float32)
    if len(arr) == 0:
        return np.zeros(6, dtype=np.float32)
    # Use 6 simple summary stats (matches wavelet sig dim for fair compare).
    feats = np.array([
        float(arr.mean()),
        float(arr.std()),
        float(np.median(arr)),
        float(arr.min()),
        float(arr.max()),
        float(np.percentile(arr, 90) - np.percentile(arr, 10)),
    ], dtype=np.float32)
    return feats


# =============================================================================
# Per-sample worker
# =============================================================================

_WORKER_WAVELET = 'db4'
_WORKER_LEVELS = 5
_WORKER_MAX_LEN = 4096


def _worker_init(wavelet, levels, max_len):
    global _WORKER_WAVELET, _WORKER_LEVELS, _WORKER_MAX_LEN
    _WORKER_WAVELET = wavelet
    _WORKER_LEVELS = levels
    _WORKER_MAX_LEN = max_len


def _encode_wavelet(args):
    idx, code = args
    sig = wavelet_signature(code, max_len=_WORKER_MAX_LEN,
                            wavelet=_WORKER_WAVELET, levels=_WORKER_LEVELS)
    return idx, sig


def _encode_meanstd(args):
    idx, code = args
    sig = meanstd_signature(code, max_len=_WORKER_MAX_LEN)
    return idx, sig


def encode_all(codes, wavelet='db4', levels=5, max_len=4096, n_workers=None,
               mode="wavelet", desc="encode"):
    N = len(codes)
    if n_workers is None or n_workers <= 0:
        n_workers = max(1, mp.cpu_count() - 1)
    out_dim = (levels + 1) if mode == "wavelet" else 6
    sigs = np.zeros((N, out_dim), dtype=np.float32)
    args_list = [(i, codes[i]) for i in range(N)]
    worker_fn = _encode_wavelet if mode == "wavelet" else _encode_meanstd
    if n_workers == 1:
        _worker_init(wavelet, levels, max_len)
        for a in tqdm(args_list, desc=desc):
            idx, sig = worker_fn(a); sigs[idx] = sig
        return sigs
    try:
        with mp.Pool(n_workers, initializer=_worker_init,
                     initargs=(wavelet, levels, max_len)) as pool:
            for idx, sig in tqdm(pool.imap_unordered(worker_fn, args_list, chunksize=16),
                                  total=N, desc=desc):
                sigs[idx] = sig
        return sigs
    except Exception as e:
        logger.warning(f"[wavelet] mp failed ({e}); serial fallback")
        return encode_all(codes, wavelet, levels, max_len, n_workers=1,
                          mode=mode, desc=desc)


# =============================================================================
# Classification: nearest class mean (Euclidean) on standardised signatures
# =============================================================================

def standardise(train_sigs, *other_sigs):
    """Z-score standardise based on train statistics."""
    mu = train_sigs.mean(axis=0, keepdims=True)
    sd = train_sigs.std(axis=0, keepdims=True) + 1e-6
    out = [(train_sigs - mu) / sd]
    for s in other_sigs:
        out.append((s - mu) / sd)
    return out, (mu, sd)


def per_class_mean(sigs, labels, n_cls):
    means = np.zeros((n_cls, sigs.shape[1]), dtype=np.float32)
    counts = np.zeros(n_cls, dtype=np.int64)
    for s, y in zip(sigs, labels):
        means[y] += s; counts[y] += 1
    for c in range(n_cls):
        if counts[c] > 0:
            means[c] /= counts[c]
    return means


def predict_min_euclid(query_sigs, class_means):
    # ||q - m_c||^2 = ||q||^2 - 2 q.m_c + ||m_c||^2
    # argmin over c equivalent to argmin of (||m_c||^2 - 2 q.m_c)
    dots = query_sigs @ class_means.T
    mnorm = (class_means ** 2).sum(axis=1)[None, :]
    dists = mnorm - 2.0 * dots
    return np.argmin(dists, axis=1)


def per_level_f_statistic(sigs, labels, n_cls):
    """One-way ANOVA F-statistic at each feature dimension (F1 falsifier)."""
    fstats = []
    n_feat = sigs.shape[1]
    for d in range(n_feat):
        groups = []
        for c in range(n_cls):
            g = sigs[np.asarray(labels) == c, d]
            if len(g) >= 2:
                groups.append(g)
        if len(groups) < 2:
            fstats.append(0.0); continue
        try:
            f_val, _ = scipy_stats.f_oneway(*groups)
            fstats.append(float(f_val) if np.isfinite(f_val) else 0.0)
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
    wavelet: str = "db4"
    levels: int = 5
    wavelet_family_sweep: tuple = ("db4", "sym4", "haar")
    train_cap_per_class: int = 400
    test_cap_per_class: int = 500
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = 4096


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
                f"test={len(test_codes)} wavelet={cfg.wavelet} levels={cfg.levels} "
                f"workers={n_workers}")

    # 1) Wavelet encode all samples
    t0 = time.time()
    train_sigs = encode_all(train_codes, wavelet=cfg.wavelet, levels=cfg.levels,
                            max_len=cfg.max_chars, n_workers=n_workers,
                            mode="wavelet", desc="train-WV")
    val_sigs   = encode_all(val_codes,   wavelet=cfg.wavelet, levels=cfg.levels,
                            max_len=cfg.max_chars, n_workers=n_workers,
                            mode="wavelet", desc="val-WV")
    test_sigs  = encode_all(test_codes,  wavelet=cfg.wavelet, levels=cfg.levels,
                            max_len=cfg.max_chars, n_workers=n_workers,
                            mode="wavelet", desc="test-WV")
    encode_time = time.time() - t0

    # Standardise based on training pool
    (train_z, val_z, test_z), _ = standardise(train_sigs, val_sigs, test_sigs)
    class_means = per_class_mean(train_z, train_labels, cfg.n_cls)

    # Val pass
    t0 = time.time()
    val_preds = predict_min_euclid(val_z, class_means)
    val_time = time.time() - t0
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                        cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]
    logger.info(f"[val] macro={val_macro:.4f} time={val_time:.0f}s")

    # Test pass
    t0 = time.time()
    test_preds = predict_min_euclid(test_z, class_means)
    test_time = time.time() - t0
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro
    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} time={test_time:.0f}s")

    # ---- Falsifier F1: per-level ANOVA F-statistic on standardised test sigs
    f_stats = per_level_f_statistic(test_z, test_labels, cfg.n_cls)
    f1_min_fstat = float(min(f_stats)) if f_stats else 0.0
    logger.info(f"[F1] per_level_f_stats={f_stats} min={f1_min_fstat:.3f}")

    # ---- Falsifier F2: raw mean/std byte baseline
    ms_train = encode_all(train_codes, wavelet=cfg.wavelet, levels=cfg.levels,
                          max_len=cfg.max_chars, n_workers=n_workers,
                          mode="meanstd", desc="train-MS")
    ms_test = encode_all(test_codes, wavelet=cfg.wavelet, levels=cfg.levels,
                         max_len=cfg.max_chars, n_workers=n_workers,
                         mode="meanstd", desc="test-MS")
    (ms_train_z, ms_test_z), _ = standardise(ms_train, ms_test)
    ms_class_means = per_class_mean(ms_train_z, train_labels, cfg.n_cls)
    ms_preds = predict_min_euclid(ms_test_z, ms_class_means)
    ms_macro = float(f1_score(test_labels, ms_preds, average="macro",
                              zero_division=0))
    f2_delta = test_macro - ms_macro
    logger.info(f"[F2] wavelet_macro={test_macro:.4f} meanstd_macro={ms_macro:.4f} "
                f"delta={f2_delta:+.4f}")

    # ---- Falsifier F3: wavelet family sweep — Spearman correlation of
    # per-class F1 vectors across (db4, sym4, haar).
    family_macros = {cfg.wavelet: test_macro}
    family_per_class = {cfg.wavelet:
                        np.array(ts_met["per_class"]["f1"], dtype=np.float32)}
    for wname in cfg.wavelet_family_sweep:
        if wname == cfg.wavelet:
            continue
        try:
            tr_alt = encode_all(train_codes, wavelet=wname, levels=cfg.levels,
                                max_len=cfg.max_chars, n_workers=n_workers,
                                mode="wavelet", desc=f"train-{wname}")
            ts_alt = encode_all(test_codes, wavelet=wname, levels=cfg.levels,
                                max_len=cfg.max_chars, n_workers=n_workers,
                                mode="wavelet", desc=f"test-{wname}")
            (tr_alt_z, ts_alt_z), _ = standardise(tr_alt, ts_alt)
            cm_alt = per_class_mean(tr_alt_z, train_labels, cfg.n_cls)
            pr_alt = predict_min_euclid(ts_alt_z, cm_alt)
            m_alt = float(f1_score(test_labels, pr_alt, average="macro",
                                    zero_division=0))
            pc_alt = f1_score(test_labels, pr_alt, average=None,
                              zero_division=0,
                              labels=list(range(cfg.n_cls)))
            family_macros[wname] = m_alt
            family_per_class[wname] = np.asarray(pc_alt, dtype=np.float32)
        except Exception as e:
            logger.warning(f"[F3] wavelet={wname} failed: {e}")
            family_macros[wname] = None
    spearman_corrs = {}
    base_vec = family_per_class[cfg.wavelet]
    for wname, vec in family_per_class.items():
        if wname == cfg.wavelet: continue
        try:
            r, _ = scipy_stats.spearmanr(base_vec, vec)
            spearman_corrs[f"{cfg.wavelet}_vs_{wname}"] = float(r) if np.isfinite(r) else 0.0
        except Exception:
            spearman_corrs[f"{cfg.wavelet}_vs_{wname}"] = 0.0
    logger.info(f"[F3] family_macros={family_macros} spearman={spearman_corrs}")

    ts_met["falsifier_F1_per_level_f_statistic"] = f_stats
    ts_met["falsifier_F1_min_f_statistic"] = f1_min_fstat
    ts_met["falsifier_F2_meanstd_baseline_macro"] = ms_macro
    ts_met["falsifier_F2_wavelet_vs_meanstd_delta"] = f2_delta
    ts_met["falsifier_F3_family_macros"] = family_macros
    ts_met["falsifier_F3_spearman_corrs"] = spearman_corrs

    return {
        "tag": tag, "method": "WAVELET",
        "note": ("Multi-resolution wavelet-energy signature for code "
                 "authorship; nearest-class-mean classifier on z-scored "
                 "per-level log-energies. CPU-only; PyWavelets db4."),
        "enc": f"wavelet-{cfg.wavelet}-L{cfg.levels}", "bench": cfg.benchmark,
        "frac": cfg.frac, "wavelet": cfg.wavelet, "levels": cfg.levels,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "per_level_f_statistic": f_stats,
        "min_f_statistic": f1_min_fstat,
        "meanstd_baseline_macro": ms_macro,
        "wavelet_vs_meanstd_delta": f2_delta,
        "family_macros": family_macros,
        "spearman_corrs": spearman_corrs,
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
            tag = f"exp142_wavelet_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"minF={res['min_f_statistic']:.2f} "
                            f"dMS={res['wavelet_vs_meanstd_delta']:+.4f}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp142_wavelet_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Wav':>6} {'Train':>8} {'Test':>8} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'MS-F1':>8} {'dMS':>8} {'minF':>8} {'Wall':>8}")
    print("-"*140)
    for r in results:
        ms = r.get("meanstd_baseline_macro") or 0.0
        dms = r.get("wavelet_vs_meanstd_delta") or 0.0
        minf = r.get("min_f_statistic") or 0.0
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['wavelet']:>6} "
              f"{r['train_size_after_cap']:>8d} {r['test_size_after_cap']:>8d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {ms:>8.4f} {dms:>+8.4f} {minf:>8.2f} "
              f"{r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
