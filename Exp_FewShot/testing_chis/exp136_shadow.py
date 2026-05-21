# exp136 — SHADOW
# =============================================================================
# NAME       : SHADOW (Per-author character n-gram LM; argmin perplexity attribution)
# REFERENCE  : Teahan & Harper 1999 ("Using compression-based language
#              models for text categorization"). Never applied to AI-CODE
#              authorship. Foundational stylometry method.
# CLAIM      : Train a character-level n-gram language model (n=4) per
#              author. At inference, compute perplexity of query under
#              each author's LM. Argmin perplexity = predicted author.
#              Theory: this is MAP estimation under Markov-n character
#              generative model. ZERO neural parameters.
# EQUATION   : For each author a, train P_a(c_t | c_{t-n+1..t-1}) via
#              Laplace-smoothed frequency counts on training samples.
#              PPL_a(x) = exp(-1/|x| * sum_t log P_a(x_t | x_{t-n+1..t-1}))
#              y_hat = argmin_a PPL_a(x)
# WHY NEW    : Teahan 1999 used PPM (prediction by partial matching) on
#              English. We use simple n-gram with explicit Laplace
#              smoothing; first time applied to MULTI-CLASS AI CODE
#              authorship. Closes the gap: if a 1980s n-gram model beats
#              UniXcoder, the field has overclaimed.
# WOW HOOK   : "The 1980s called. A character 4-gram language model with
#              Laplace smoothing — older than the transformer architecture
#              — produces an honest baseline. We train it in 2 seconds
#              per author. Any neural method must beat THIS."
# FALSIFIER  : (F1) PPL_correct_class < PPL_other_class on test (basic
#              sanity). (F2) SHADOW composite > random class baseline
#              (1/K) by >= 0.05 (n-gram captures REAL signal).
#              (F3) Best-n via val sweep (n in {2,3,4,5}): optimal n
#              should fall in [3, 5] (not collapse to unigram or 7-gram).
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, math, subprocess, importlib.util, warnings, glob
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
logger = logging.getLogger("exp136_shadow")

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
# Character n-gram language model with Laplace smoothing
# =============================================================================

# Default vocabulary size: printable ASCII range (32..126) + newline + tab = ~96.
# We use 128 to cover the full 7-bit ASCII as a safe upper bound; characters
# outside this range are replaced by a sentinel before training. For UTF-8 code
# this is conservative; identifier names with unicode characters get mapped to
# the OOV sentinel.
VOCAB_SIZE = 128
OOV_CHAR = "\x00"


def _normalize_char(c: str) -> str:
    """Restrict to printable ASCII; otherwise OOV sentinel."""
    o = ord(c)
    if o < 128:
        return c
    return OOV_CHAR


class CharNGramLM:
    """Character n-gram LM with Laplace smoothing. Pure-Python dict based."""

    __slots__ = ("n", "vocab_size", "context_counts", "ngram_counts")

    def __init__(self, n: int = 4, vocab_size: int = VOCAB_SIZE):
        self.n = n
        self.vocab_size = vocab_size
        self.context_counts: Dict[tuple, int] = {}
        self.ngram_counts: Dict[tuple, int] = {}

    def train(self, texts: List[str], max_chars: int = 5000):
        """Accumulate counts from a list of training strings."""
        n = self.n
        for text in texts:
            # Map to ASCII-only.
            chars = [_normalize_char(c) for c in text[:max_chars]]
            L = len(chars)
            if L < n: continue
            for t in range(n - 1, L):
                context = tuple(chars[t - n + 1:t])
                next_c = chars[t]
                key = (context, next_c)
                self.ngram_counts[key] = self.ngram_counts.get(key, 0) + 1
                self.context_counts[context] = self.context_counts.get(context, 0) + 1

    def log_prob(self, text: str, max_chars: int = 5000) -> float:
        """Average log probability under Laplace-smoothed n-gram. Returns 0.0
        if the text is too short (T=0)."""
        n = self.n
        chars = [_normalize_char(c) for c in text[:max_chars]]
        L = len(chars)
        if L < n: return 0.0
        log_p_sum = 0.0
        T = 0
        V = self.vocab_size
        gng = self.ngram_counts.get
        gctx = self.context_counts.get
        for t in range(n - 1, L):
            context = tuple(chars[t - n + 1:t])
            next_c = chars[t]
            count_ctx = gctx(context, 0)
            count_ng = gng((context, next_c), 0)
            # Laplace smoothing: P = (count + 1) / (count_ctx + V)
            p = (count_ng + 1.0) / (count_ctx + V)
            log_p_sum += math.log(p)
            T += 1
        return log_p_sum / max(T, 1)

    def perplexity(self, text: str, max_chars: int = 5000) -> float:
        lp = self.log_prob(text, max_chars=max_chars)
        try:
            return math.exp(-lp)
        except OverflowError:
            return float("inf")

    def to_serial(self) -> dict:
        """Compact dict-of-dicts serialization for multiprocessing transport."""
        return {
            "n": self.n,
            "vocab_size": self.vocab_size,
            "context_counts": dict(self.context_counts),
            "ngram_counts": dict(self.ngram_counts),
        }

    @classmethod
    def from_serial(cls, d: dict) -> "CharNGramLM":
        lm = cls(n=d["n"], vocab_size=d["vocab_size"])
        lm.context_counts = d["context_counts"]
        lm.ngram_counts = d["ngram_counts"]
        return lm


def train_per_author(train_codes: List[str], train_labels: List[int],
                      n_cls: int, n_gram: int = 4,
                      max_chars: int = 5000) -> List[CharNGramLM]:
    """Train one LM per class. Returns list of LMs indexed by class."""
    lms = []
    for c in range(n_cls):
        texts_c = [train_codes[i] for i in range(len(train_labels)) if train_labels[i] == c]
        lm = CharNGramLM(n=n_gram)
        if texts_c:
            lm.train(texts_c, max_chars=max_chars)
        lms.append(lm)
    return lms


# --- Multiprocessing inference --------------------------------------------------

# Module-level globals populated by pool initializer (avoids re-pickling LMs
# per task).
_WORKER_LMS: List[CharNGramLM] = []
_WORKER_MAX_CHARS = 5000


def _worker_init(serial_lms: List[dict], max_chars: int):
    global _WORKER_LMS, _WORKER_MAX_CHARS
    _WORKER_LMS = [CharNGramLM.from_serial(s) for s in serial_lms]
    _WORKER_MAX_CHARS = max_chars


def _worker_predict(args):
    """args = (idx, text) -> (idx, pred_label, ppls_list)"""
    idx, text = args
    ppls = [lm.perplexity(text, max_chars=_WORKER_MAX_CHARS) for lm in _WORKER_LMS]
    pred = int(np.argmin(ppls))
    return idx, pred, ppls


def predict_perplexity(test_codes: List[str], lms: List[CharNGramLM],
                        n_workers: int, max_chars: int = 5000,
                        desc: str = "ppl") -> Tuple[np.ndarray, np.ndarray]:
    """Return (preds, ppl_matrix) where ppl_matrix is (N, K)."""
    N = len(test_codes); K = len(lms)
    preds = np.zeros(N, dtype=np.int64)
    ppl_matrix = np.zeros((N, K), dtype=np.float64)
    args_list = [(i, test_codes[i]) for i in range(N)]
    if n_workers <= 1:
        # Serial: no need for worker globals
        for i, txt in tqdm(args_list, desc=desc):
            ppls = [lm.perplexity(txt, max_chars=max_chars) for lm in lms]
            preds[i] = int(np.argmin(ppls))
            ppl_matrix[i] = ppls
        return preds, ppl_matrix
    serial = [lm.to_serial() for lm in lms]
    try:
        with mp.Pool(n_workers, initializer=_worker_init,
                      initargs=(serial, max_chars)) as pool:
            for idx, pred, ppls in tqdm(pool.imap_unordered(_worker_predict, args_list, chunksize=8),
                                         total=N, desc=desc):
                preds[idx] = pred
                ppl_matrix[idx] = ppls
    except Exception as e:
        logger.warning(f"[shadow] mp failed ({e}); serial fallback")
        for i, txt in tqdm(args_list, desc=desc):
            ppls = [lm.perplexity(txt, max_chars=max_chars) for lm in lms]
            preds[i] = int(np.argmin(ppls))
            ppl_matrix[i] = ppls
    return preds, ppl_matrix


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
    n_gram: int = 4
    sweep_ns: tuple = (2, 3, 4, 5)
    train_cap_per_class: int = 400
    test_cap_per_class: int = 500
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
# Falsifier helpers
# =============================================================================

def correct_vs_wrong_ppl(ppl_matrix: np.ndarray, labels: List[int]) -> dict:
    """F1: mean PPL of the correct-class LM vs mean PPL of all other LMs."""
    labels = np.asarray(labels)
    N, K = ppl_matrix.shape
    # Mask infinities for stable means
    finite = np.isfinite(ppl_matrix)
    safe_ppl = np.where(finite, ppl_matrix, np.nan)
    correct_vals = []
    wrong_vals = []
    for i in range(N):
        lab = labels[i]
        if 0 <= lab < K:
            cv = safe_ppl[i, lab]
            if np.isfinite(cv): correct_vals.append(cv)
            wv = [safe_ppl[i, j] for j in range(K) if j != lab and np.isfinite(safe_ppl[i, j])]
            if wv: wrong_vals.append(float(np.mean(wv)))
    return {
        "mean_correct_class_ppl": float(np.mean(correct_vals)) if correct_vals else None,
        "mean_wrong_class_ppl":   float(np.mean(wrong_vals))   if wrong_vals   else None,
        "ppl_separation_gap":     float(np.mean(wrong_vals) - np.mean(correct_vals))
                                    if (correct_vals and wrong_vals) else None,
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
                f"test={len(test_codes)} workers={n_workers} n_grams_sweep={cfg.sweep_ns}")

    # ---------- F3: n-sweep on val ----------
    sweep_results = {}
    best_n = cfg.n_gram
    best_val_macro = -1.0
    best_lms = None
    val_preds_best = None
    val_history = []
    for n_g in cfg.sweep_ns:
        t0 = time.time()
        lms = train_per_author(train_codes, train_labels, cfg.n_cls,
                                n_gram=n_g, max_chars=cfg.max_chars)
        train_time = time.time() - t0
        t0 = time.time()
        v_preds, v_ppl = predict_perplexity(val_codes, lms, n_workers=n_workers,
                                              max_chars=cfg.max_chars,
                                              desc=f"val n={n_g}")
        v_time = time.time() - t0
        v_macro = float(f1_score(val_labels, v_preds, average="macro", zero_division=0,
                                  labels=list(range(cfg.n_cls))))
        sweep_results[n_g] = {
            "val_macro": v_macro,
            "train_time_sec": train_time,
            "val_time_sec": v_time,
        }
        val_history.append(v_macro)
        logger.info(f"[sweep] n={n_g} val_macro={v_macro:.4f} "
                    f"train={train_time:.1f}s val={v_time:.1f}s")
        if v_macro > best_val_macro:
            best_val_macro = v_macro
            best_n = n_g
            best_lms = lms
            val_preds_best = v_preds
    logger.info(f"[sweep] best_n={best_n} best_val_macro={best_val_macro:.4f}")

    # ---------- Eval val with best n ----------
    val_met = eval_pack(val_preds_best, val_labels, val_langs, val_sources,
                         cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]

    # ---------- Test pass with best n ----------
    t0 = time.time()
    test_preds, test_ppl = predict_perplexity(test_codes, best_lms, n_workers=n_workers,
                                                max_chars=cfg.max_chars, desc="test")
    test_time = time.time() - t0
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro

    # ---------- Falsifier F1: PPL separation ----------
    ppl_sep = correct_vs_wrong_ppl(test_ppl, test_labels)

    # F2: composite vs random baseline (1/K)
    random_baseline = 1.0 / cfg.n_cls
    f2_margin = float(test_macro - random_baseline)

    # F3: best_n in [3, 5]?
    f3_pass = bool(3 <= best_n <= 5)

    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} time={test_time:.1f}s")
    logger.info(f"[F1] mean_correct_ppl={ppl_sep['mean_correct_class_ppl']} "
                f"mean_wrong_ppl={ppl_sep['mean_wrong_class_ppl']} "
                f"sep_gap={ppl_sep['ppl_separation_gap']}")
    logger.info(f"[F2] macro_vs_random_margin={f2_margin:+.4f} (random={random_baseline:.4f})")
    logger.info(f"[F3] best_n={best_n} (in [3,5]={f3_pass})")

    ts_met["falsifier_F1_mean_correct_class_ppl"] = ppl_sep["mean_correct_class_ppl"]
    ts_met["falsifier_F1_mean_wrong_class_ppl"] = ppl_sep["mean_wrong_class_ppl"]
    ts_met["falsifier_F1_ppl_separation_gap"] = ppl_sep["ppl_separation_gap"]
    ts_met["falsifier_F2_random_baseline"] = random_baseline
    ts_met["falsifier_F2_macro_vs_random_margin"] = f2_margin
    ts_met["falsifier_F3_best_n"] = best_n
    ts_met["falsifier_F3_in_expected_range"] = f3_pass
    ts_met["falsifier_F3_sweep"] = {str(k): v for k, v in sweep_results.items()}

    return {
        "tag": tag, "method": "SHADOW",
        "note": ("Per-author character n-gram language model with Laplace smoothing. "
                 "Argmin-perplexity attribution. CPU-only, zero neural parameters."),
        "enc": f"char-ngram-n{best_n}", "bench": cfg.benchmark,
        "frac": cfg.frac, "n_gram": best_n,
        "n_sweep_val_macros": dict(zip([str(n) for n in cfg.sweep_ns], val_history)),
        "best_n_via_val_sweep": best_n,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "test_time_sec": test_time,
        "mean_correct_class_ppl": ppl_sep["mean_correct_class_ppl"],
        "mean_wrong_class_ppl":   ppl_sep["mean_wrong_class_ppl"],
        "ppl_separation_gap":     ppl_sep["ppl_separation_gap"],
        "macro_vs_random_margin": f2_margin,
        "test_metrics": ts_met,
        "val_history": val_history,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in benchmarks:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp136_shadow_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"best_n={res['best_n_via_val_sweep']} "
                            f"ppl_sep={res['ppl_separation_gap']}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp136_shadow_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'BestN':>6} {'Train':>8} {'Test':>8} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'PPLsep':>10} {'vsRandom':>10} {'Wall':>8}")
    print("-"*140)
    for r in results:
        sep = r.get("ppl_separation_gap") or 0.0
        vr = r.get("macro_vs_random_margin") or 0.0
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['best_n_via_val_sweep']:>6d} "
              f"{r['train_size_after_cap']:>8d} {r['test_size_after_cap']:>8d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {sep:>10.2f} {vr:>+10.4f} {r['wall']:>8.0f}s")
    print("="*140)
    # Print n-sweep details
    print("\nVal Macro-F1 by n-gram size (F3: best should be in [3,5]):")
    print(f"{'Tag':<40} " + " ".join(f"{'n='+str(n):>10}" for n in (2, 3, 4, 5)))
    for r in results:
        sw = r.get("n_sweep_val_macros") or {}
        cells = " ".join(f"{sw.get(str(n), 0.0):>10.4f}" for n in (2, 3, 4, 5))
        print(f"{r['tag']:<40} {cells}")
    print("="*140)


if __name__ == "__main__":
    main()
