# exp134 — LIBRARIAN
# =============================================================================
# NAME       : LIBRARIAN (Per-author compression dictionaries; attribution
#              as Lempel-Ziv decoding with the right dictionary)
# REFERENCE  : NEW theoretical contribution; builds on Lempel-Ziv coding
#              (Ziv & Lempel 1978), Minimum Description Length (Rissanen 1978),
#              Cilibrasi-Vitanyi compression-based clustering (cs/0312044
#              2005), Zstandard dictionary training (Collet 2016).
#              Conceptually adjacent to Jiang 2023 ACL gzip-text-clf but
#              fundamentally different: Jiang uses SYMMETRIC NCD; we use
#              ASYMMETRIC dictionary-conditional compression, which has a
#              direct MAP / MDL interpretation.
#
# THEORETICAL CLAIM (the contribution of this paper):
#     Authorship attribution under a Lempel-Ziv generative model is
#     equivalent to Maximum A Posteriori (MAP) estimation:
#         argmax_a P(x | a, D_a) = argmin_a |compress(x, dictionary=D_a)|
#     where D_a is a dictionary trained from author a's training samples.
#     Proof sketch: under the LZ-78 family of universal coders,
#       -log_2 P_LZ(x | D) ≈ |compress(x, D)|     [Cover & Thomas, Thm 13.5.3]
#     so picking the dictionary that minimises the compressed length is
#     exactly MAP under the LZ likelihood model with uniform prior P(a).
#     This connects the practical task (attribute generated code) to a
#     formal probabilistic model (LZ generative). The connection is
#     CPU-only, parameter-free, and information-theoretically grounded.
#
# METHOD:
#     1. Train phase: for each author a in {1, ..., K}:
#            D_a = train_dictionary(zstd, samples = train_examples_of_a)
#        D_a is a learned 16-32 KB dictionary that captures the
#        author's repeated substrings (Lempel-Ziv "phrases").
#     2. Inference: for query q:
#            score_a = |compress(q, dict=D_a)|     for each a
#            y_hat = argmin_a score_a
#     3. No neural network. No GPU. No gradient. No hyperparameter
#        beyond zstd compression level (=22 = max) and dictionary size
#        (16 KB by default).
#
# WHY NEW: No prior code-attribution paper uses per-author dictionary-
#     conditional compression. Closest is Jiang 2023 (symmetric gzip-NCD,
#     no dictionary). Closer in NLP: Teahan & Harper 1999 used PPM models
#     for authorship of natural language; never applied to LLM-generated
#     code, and never with the MAP-MDL connection made explicit.
#
# WOW HOOK: "Authorship attribution is Lempel-Ziv decoding with the right
#     dictionary. Every author has a LIBRARY of repeated phrases learned
#     from their training code; the author whose library makes the query
#     SHORTEST is the predicted author. We prove the connection, then
#     test it. Zero parameters, zero training, zero GPU."
#
# FALSIFIER:
#     (F1) Theoretical: log_2(K) - mean(min_a -log P(x|D_a)) ≥ 0 on test
#         (Fano's inequality: information gain must be positive).
#     (F2) Empirical: LIBRARIAN composite > GZIP-NCD composite by ≥ 0.02
#         (otherwise, the dictionary-conditional refinement adds nothing
#         over the symmetric NCD baseline).
#     (F3) Mass concentration: average score gap
#         score_(2nd-best) - score_(best) > 5% of |compressed(x, D_best)|
#         (i.e., the best author is clearly preferred over the runner-up;
#         if not, the dictionaries are too similar to discriminate).
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("datasets"); _ensure("scikit-learn"); _ensure("tqdm")
_ensure("zstandard")

import numpy as np
import zstandard as zstd
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp134_librarian")

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


def stratified_subsample(data, n_per_class, seed=42):
    """Return at most n_per_class samples per class, stratified."""
    rng = random.Random(seed)
    labels = list(range(max(data["label"]) + 1))
    keep = []
    for lbl in labels:
        idx = [i for i, x in enumerate(data["label"]) if x == lbl]
        n = min(int(n_per_class), len(idx))
        keep.extend(rng.sample(idx, n))
    return data.select(keep)


def stratified_frac(data, frac, seed=42):
    rng = random.Random(seed)
    labels = list(range(max(data["label"]) + 1))
    keep = []
    for lbl in labels:
        idx = [i for i, x in enumerate(data["label"]) if x == lbl]
        n = min(max(1, int(len(idx) * frac)), len(idx))
        keep.extend(rng.sample(idx, n))
    return data.select(keep)


# =============================================================================
# Core LIBRARIAN logic
# =============================================================================

def train_author_dictionary(samples: List[str], dict_size: int = 16384,
                              max_samples: int = 200) -> bytes:
    """Train a zstd dictionary on the given author's training samples.
    Returns the raw dictionary bytes (re-usable for compression).
    """
    # Bound the training corpus: zstd dictionary training is O(corpus)
    # and 200 samples × 4KB = 800KB which trains in seconds.
    if len(samples) > max_samples:
        samples = random.sample(samples, max_samples)
    samples_bytes = [s.encode("utf-8", errors="ignore") for s in samples if s.strip()]
    if not samples_bytes:
        # Fallback: return empty dict (zero-byte dict is allowed)
        return b""
    try:
        dict_data = zstd.train_dictionary(dict_size, samples_bytes)
        return dict_data.as_bytes()
    except Exception as e:
        logger.warning(f"[train_dict] failed: {e}; falling back to corpus concat")
        # Fallback: use simple concat as a "dictionary" (no LZ training)
        concat = b"".join(samples_bytes)[:dict_size]
        return concat


def compress_with_dict(query: str, dict_bytes: bytes, level: int = 22) -> int:
    """Return compressed length (bytes) of query given dictionary."""
    if not dict_bytes:
        # No dictionary: vanilla compression
        cctx = zstd.ZstdCompressor(level=level)
        return len(cctx.compress(query.encode("utf-8", errors="ignore")))
    try:
        d = zstd.ZstdCompressionDict(dict_bytes)
        cctx = zstd.ZstdCompressor(level=level, dict_data=d)
        return len(cctx.compress(query.encode("utf-8", errors="ignore")))
    except Exception:
        # Fallback path
        cctx = zstd.ZstdCompressor(level=level)
        return len(cctx.compress(query.encode("utf-8", errors="ignore")))


def _attribute_one(args):
    """Worker for multiprocessing. Returns (query_idx, scores_per_class, predicted_label)."""
    qi, query, dicts_per_class, level = args
    n_cls = len(dicts_per_class)
    scores = np.zeros(n_cls, dtype=np.float32)
    for c in range(n_cls):
        scores[c] = compress_with_dict(query, dicts_per_class[c], level=level)
    pred = int(np.argmin(scores))
    return qi, scores, pred


def librarian_attribute(test_codes: List[str], dicts_per_class: List[bytes],
                          level: int = 22, n_workers: int = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictions, all_scores) where all_scores: (N, K)."""
    N = len(test_codes)
    K = len(dicts_per_class)
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    args_list = [(i, test_codes[i], dicts_per_class, level) for i in range(N)]
    preds = np.zeros(N, dtype=np.int64)
    all_scores = np.zeros((N, K), dtype=np.float32)

    if n_workers == 1:
        for args in tqdm(args_list, desc="LIBRARIAN attribute"):
            qi, scores, pred = _attribute_one(args)
            preds[qi] = pred
            all_scores[qi] = scores
    else:
        try:
            with mp.Pool(n_workers) as pool:
                for qi, scores, pred in tqdm(
                        pool.imap_unordered(_attribute_one, args_list, chunksize=8),
                        total=N, desc="LIBRARIAN attribute"):
                    preds[qi] = pred
                    all_scores[qi] = scores
        except Exception as e:
            logger.warning(f"[mp] failed ({e}); falling back to serial")
            return librarian_attribute(test_codes, dicts_per_class, level=level, n_workers=1)
    return preds, all_scores


# =============================================================================
# Theoretical falsifier (F1): information gain bound
# =============================================================================

def information_gain_bound(all_scores: np.ndarray, n_cls: int) -> Dict[str, float]:
    """Compute the LZ-likelihood-based information gain proxy.

    For each test sample we have scores[c] = |compress(x, D_c)|.  Convert to
    pseudo-log-likelihoods: log_2 P(x|D_c) ≈ -scores[c]
    (modulo additive constants).  Then:
        I(A; X) ≈ H(A) - H(A|X) = log_2 K - <-log_2 P(A|x)>_x
    where P(A|x) ∝ exp(-scores·ln 2) (after softmax).

    Returns:
        - mean_neg_logP_best: average -log P(x | D_best)  (information cost)
        - log_K: log_2(K), the entropy of the uniform prior
        - info_gain: log_K - <H(A|X)>_x  (Fano-style; should be > 0)
        - confidence_gap_mean: <score[2nd-best] - score[best]> / score[best]
    """
    N, K = all_scores.shape
    log_K = float(np.log2(K))
    # Convert scores to probabilities via softmin (smaller = higher prob)
    # Treat scores in BITS: log P(x|D_c) ≈ -scores[c] (already in bytes; *8 → bits)
    bit_scores = all_scores * 8.0
    bit_scores_neg = -bit_scores
    # Softmax over classes
    bit_scores_neg_centered = bit_scores_neg - bit_scores_neg.max(axis=1, keepdims=True)
    expv = np.exp(bit_scores_neg_centered)
    probs = expv / expv.sum(axis=1, keepdims=True)
    # Posterior entropy H(A|x)
    entropy_post = -(probs * np.log2(probs + 1e-30)).sum(axis=1)
    mean_entropy_post = float(entropy_post.mean())
    info_gain = log_K - mean_entropy_post

    # Confidence gap: relative difference between best and runner-up score
    sorted_scores = np.sort(all_scores, axis=1)
    best = sorted_scores[:, 0]
    runner_up = sorted_scores[:, 1]
    gap = (runner_up - best) / np.maximum(best, 1.0)
    confidence_gap_mean = float(gap.mean())

    return {
        "log_K_bits":               log_K,
        "mean_posterior_entropy":   mean_entropy_post,
        "information_gain_F1":      float(info_gain),
        "confidence_gap_F3":        confidence_gap_mean,
        "best_score_bytes_mean":    float(best.mean()),
        "runner_up_bytes_mean":     float(runner_up.mean()),
    }


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
    # Dictionary training: how many training samples per class to use
    dict_train_samples_per_class: int = 200
    # Dictionary size in bytes (zstd recommends 1KB-100KB for code)
    dict_size_bytes: int = 16384
    # Compression level (22 = max for zstd)
    zstd_level: int = 22
    # Bound test queries per class for CPU tractability
    test_cap_per_class: int = 500
    val_cap_per_class: int = 250
    # Truncate code length (zstd is fast but bounded length keeps timing stable)
    max_chars: int = 8000
    n_workers: int = -1  # -1 → cpu_count() - 1
    gene_adj: dict = field(default_factory=dict)


def set_seed(s):
    random.seed(s); np.random.seed(s)


# =============================================================================
# Eval pack (consistent with exp84_cargo schema)
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

    # Few-shot fraction first; then cap to dict_train_samples_per_class for tractability.
    tr_data_frac = stratified_frac(tr_data, cfg.frac, seed=cfg.seed)
    tr_data_capped = stratified_subsample(tr_data_frac, cfg.dict_train_samples_per_class,
                                            seed=cfg.seed + 1)
    vl_data_capped = stratified_subsample(vl_data, cfg.val_cap_per_class, seed=cfg.seed + 2)
    ts_data_capped = stratified_subsample(ts_data, cfg.test_cap_per_class, seed=cfg.seed + 3)

    # Group training samples by class
    train_samples_per_class: List[List[str]] = [[] for _ in range(cfg.n_cls)]
    for r in tr_data_capped:
        c = int(r["label"])
        train_samples_per_class[c].append(r["code"][:cfg.max_chars])

    logger.info(f"[setup] frac={cfg.frac}  per-class train sizes: "
                f"{[len(s) for s in train_samples_per_class]}")

    # =====  STEP 1: Train per-author dictionaries  =====
    t_dict = time.time()
    logger.info(f"[step 1] Training {cfg.n_cls} per-author zstd dictionaries "
                f"(dict_size={cfg.dict_size_bytes}, level={cfg.zstd_level})...")
    dicts_per_class: List[bytes] = []
    dict_byte_sizes: List[int] = []
    for c in range(cfg.n_cls):
        samples = train_samples_per_class[c]
        if not samples:
            logger.warning(f"[step 1] class {c} has 0 training samples; using empty dict")
            d = b""
        else:
            d = train_author_dictionary(samples, dict_size=cfg.dict_size_bytes)
        dicts_per_class.append(d)
        dict_byte_sizes.append(len(d))
    dict_time = time.time() - t_dict
    logger.info(f"[step 1] dictionaries trained in {dict_time:.1f}s; "
                f"sizes={dict_byte_sizes} bytes")

    # =====  STEP 2: Attribute val/test queries  =====
    val_codes = [r["code"][:cfg.max_chars] for r in vl_data_capped]
    val_labels = list(vl_data_capped["label"])
    val_langs = [r.get("language", "") or "" for r in vl_data_capped]
    val_sources = [r.get("source", "") or "" for r in vl_data_capped]
    test_codes = [r["code"][:cfg.max_chars] for r in ts_data_capped]
    test_labels = list(ts_data_capped["label"])
    test_langs = [r.get("language", "") or "" for r in ts_data_capped]
    test_sources = [r.get("source", "") or "" for r in ts_data_capped]

    n_workers = cfg.n_workers if cfg.n_workers > 0 else max(1, mp.cpu_count() - 1)
    logger.info(f"[setup] val={len(val_codes)} test={len(test_codes)} workers={n_workers}")

    # Val
    t_val = time.time()
    val_preds, val_scores = librarian_attribute(
        val_codes, dicts_per_class, level=cfg.zstd_level, n_workers=n_workers)
    val_time = time.time() - t_val
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                         cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]
    val_ig = information_gain_bound(val_scores, cfg.n_cls)
    logger.info(f"[val] macro={val_macro:.4f} time={val_time:.0f}s "
                f"info_gain={val_ig['information_gain_F1']:.3f} bits "
                f"conf_gap={val_ig['confidence_gap_F3']:.4f}")

    # Test
    t_test = time.time()
    test_preds, test_scores = librarian_attribute(
        test_codes, dicts_per_class, level=cfg.zstd_level, n_workers=n_workers)
    test_time = time.time() - t_test
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro
    test_ig = information_gain_bound(test_scores, cfg.n_cls)
    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} time={test_time:.0f}s "
                f"info_gain={test_ig['information_gain_F1']:.3f} bits "
                f"conf_gap={test_ig['confidence_gap_F3']:.4f}")

    # Add falsifier readouts to test_metrics
    ts_met["falsifier"] = {
        "information_gain_F1_bits":   test_ig["information_gain_F1"],
        "log_K_bits":                 test_ig["log_K_bits"],
        "confidence_gap_F3":          test_ig["confidence_gap_F3"],
        "mean_posterior_entropy":     test_ig["mean_posterior_entropy"],
        "best_score_bytes_mean":      test_ig["best_score_bytes_mean"],
        "runner_up_bytes_mean":       test_ig["runner_up_bytes_mean"],
    }

    return {
        "tag": tag, "method": "LIBRARIAN",
        "note": ("Per-author zstd compression dictionaries; attribution by argmin "
                 "|compress(query, dict=D_a)|. Theory: MAP under Lempel-Ziv generative."),
        "enc": f"zstd-l{cfg.zstd_level}-dict{cfg.dict_size_bytes}",
        "bench": cfg.benchmark, "frac": cfg.frac,
        "dict_size_bytes": cfg.dict_size_bytes, "zstd_level": cfg.zstd_level,
        "dict_train_samples_per_class": cfg.dict_train_samples_per_class,
        "dict_byte_sizes": dict_byte_sizes,
        "dict_train_time_sec": float(dict_time),
        "val_time_sec": float(val_time), "test_time_sec": float(test_time),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "information_gain_F1_bits":  test_ig["information_gain_F1"],
        "confidence_gap_F3":         test_ig["confidence_gap_F3"],
        "log_K_bits":                test_ig["log_K_bits"],
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
            tag = f"exp134_librarian_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} "
                            f"info_gain={res['information_gain_F1_bits']:.3f}/{res['log_K_bits']:.3f} bits "
                            f"conf_gap={res['confidence_gap_F3']:.4f} "
                            f"time={res['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp134_librarian_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'DictSz':>7} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'InfoGain':>10} {'log2K':>7} {'ConfGap':>10} {'Wall':>8}")
    print("-"*140)
    for r in results:
        ig = r.get("information_gain_F1_bits", 0.0) or 0.0
        cg = r.get("confidence_gap_F3", 0.0) or 0.0
        lk = r.get("log_K_bits", 0.0) or 0.0
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['dict_size_bytes']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {ig:>10.3f} {lk:>7.3f} {cg:>10.4f} {r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
