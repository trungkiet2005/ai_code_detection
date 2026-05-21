# exp138 — BAYESCRAFT
# =============================================================================
# NAME       : BAYESCRAFT (Bayesian Information Criterion to select Markov
#              order per author; argmin posterior code attribution)
# REFERENCE  : Schwarz 1978 ("Estimating the dimension of a model", Annals
#              of Statistics) — BIC. MacKay 2003 ("Information Theory,
#              Inference, and Learning Algorithms") — model selection.
#              Never applied to AI-code attribution.
# CLAIM      : Different authors have different RIGHT Markov order for
#              their character sequences (some authors are more repetitive
#              → lower order suffices; some are more varied → higher order
#              needed). We use BIC to SELECT the right order PER AUTHOR,
#              then attribute by posterior-MAP under each author's optimal
#              Markov model. Theoretical grounding: BIC ≈ log model
#              evidence under Laplace approximation.
# EQUATION   : For author a and Markov order p ∈ {1,2,3,4,5}:
#              BIC_p(a) = -2 log L(X_a | M_p) + k_p log N_a
#              where L is likelihood under p-order Markov, k_p = #parameters,
#              N_a = training tokens of author a.
#              Select p_a* = argmin_p BIC_p(a)  (one per author)
#              At inference: y_hat = argmin_a -log P(x | M_{p_a*})
# WHY NEW    : Standard Markov-order methods use ONE order globally
#              (typical n=3 or 4). BAYESCRAFT lets each author pick its
#              own optimal order via principled model selection. First
#              code-attribution paper to use BIC for per-author model
#              selection.
# WOW HOOK   : "Different authors need different Markov orders. We use BIC
#              — the same criterion that selects polynomial degrees in
#              1978 — to learn the right STATISTICAL DEPTH for each LLM.
#              GPT-4 is a deeper Markov chain than Llama. We prove it."
# FALSIFIER  : (F1) Selected orders p_a* should VARY across authors (not
#              all collapse to the same p). If std(p_a*) = 0, BIC selection
#              is trivial.
#              (F2) BIC-selected per-author models beat fixed-order
#              global model (n=4) by ≥ 0.005.
#              (F3) BIC values monotone-decrease then increase with p
#              (classic U-shape; if monotone in one direction, signal is
#              insufficient to balance fit vs complexity).
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

_ensure("numpy"); _ensure("scipy"); _ensure("datasets"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp138_bayescraft")

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
# Data loading (same plumbing as exp133)
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
# Markov LM core
# =============================================================================

# Restrict the alphabet to the printable-ASCII range that dominates source code.
# Vocab size matters for Laplace smoothing's effective denominator.
VOCAB_SIZE = 128
MAX_SAMPLE_CHARS = 5000  # truncate per sample to bound training time


class MarkovLM:
    """p-order character-level Markov LM with add-one (Laplace) smoothing.

    Stores only OBSERVED (context, next_char) pairs; OOV contexts at inference
    fall back to a uniform-Laplace distribution over the vocab.

    Important: `num_parameters()` returns the EFFECTIVE number of parameters
    actually estimated from data (i.e. number of distinct observed contexts ×
    average outcomes per context — proxied as len(ngram_counts)), NOT the
    worst-case V^(p+1). This is the right BIC penalty under sparse models.
    """

    __slots__ = ("order", "vocab_size", "context_counts", "ngram_counts",
                 "n_train_tokens")

    def __init__(self, order, vocab_size=VOCAB_SIZE):
        self.order = int(order)
        self.vocab_size = int(vocab_size)
        self.context_counts = {}
        self.ngram_counts = {}
        self.n_train_tokens = 0

    def train(self, texts):
        for text in texts:
            chars = list(text)[:MAX_SAMPLE_CHARS]
            if len(chars) <= self.order: continue
            for t in range(self.order, len(chars)):
                ctx = tuple(chars[t - self.order:t])
                nxt = chars[t]
                self.ngram_counts[(ctx, nxt)] = self.ngram_counts.get((ctx, nxt), 0) + 1
                self.context_counts[ctx] = self.context_counts.get(ctx, 0) + 1
                self.n_train_tokens += 1

    def log_likelihood(self, text):
        chars = list(text)[:MAX_SAMPLE_CHARS]
        if len(chars) <= self.order: return 0.0
        ll = 0.0
        V = self.vocab_size
        for t in range(self.order, len(chars)):
            ctx = tuple(chars[t - self.order:t])
            nxt = chars[t]
            count_ctx = self.context_counts.get(ctx, 0)
            count_ng = self.ngram_counts.get((ctx, nxt), 0)
            # Laplace add-one smoothing over the vocab
            p = (count_ng + 1.0) / (count_ctx + V)
            ll += math.log(p)
        return ll

    def num_parameters(self):
        """Effective k_p — count of observed (context,next) cells. For
        worst-case-dense models this approaches V^(p+1); for sparse natural
        text it is much smaller, which is the whole point of using BIC for
        practical model selection."""
        return len(self.ngram_counts)

    def bic(self, texts):
        ll = sum(self.log_likelihood(t) for t in texts)
        k = self.num_parameters()
        N = max(self.n_train_tokens, 1)
        return -2.0 * ll + k * math.log(N)


def select_order_via_bic(samples, order_range=(1, 2, 3, 4, 5)):
    """For one author's training samples, fit a MarkovLM at each order in
    `order_range` and return the order with minimum BIC, plus the full BIC
    curve and the trained LMs (so callers can reuse them)."""
    bics = {}
    lms = {}
    for p in order_range:
        lm = MarkovLM(order=p)
        lm.train(samples)
        bics[p] = lm.bic(samples)
        lms[p] = lm
    best_p = min(bics, key=bics.get)
    return best_p, lms[best_p], bics, lms


def train_authors_bic(train_samples_per_class, n_cls, order_range=(1, 2, 3, 4, 5)):
    """Fit BIC-selected LMs for each class. Also returns ALL per-order LMs for
    the fixed-order comparison baseline (falsifier F2)."""
    lms_best = []
    per_class_best_order = []
    per_class_bic_curve = []
    lms_per_order = {p: [] for p in order_range}
    for c in range(n_cls):
        samples_c = train_samples_per_class[c]
        if len(samples_c) == 0:
            # Empty class — fallback uniform LM
            empty = MarkovLM(order=1)
            lms_best.append(empty)
            per_class_best_order.append(1)
            per_class_bic_curve.append([0.0] * len(order_range))
            for p in order_range:
                lms_per_order[p].append(MarkovLM(order=p))
            continue
        best_p, lm, bic_curve, all_lms = select_order_via_bic(samples_c, order_range)
        lms_best.append(lm)
        per_class_best_order.append(best_p)
        per_class_bic_curve.append([float(bic_curve[p]) for p in order_range])
        for p in order_range:
            lms_per_order[p].append(all_lms[p])
    return lms_best, per_class_best_order, per_class_bic_curve, lms_per_order


def attribute(query, lms):
    """y_hat = argmin_c -log P(query | LM_c) (i.e., argmax log-likelihood).
    Returns (pred, list_of_per_class_neg_loglik)."""
    lls = [-lm.log_likelihood(query) for lm in lms]
    return int(np.argmin(lls)), lls


def predict_batch(test_codes, lms):
    preds = []
    for q in tqdm(test_codes, desc="attribute"):
        p, _ = attribute(q, lms)
        preds.append(p)
    return np.array(preds)


def detect_u_shape(curve):
    """A BIC curve is U-shaped if the argmin is in the INTERIOR (not at an
    endpoint), i.e. BIC strictly decreases then strictly increases at least
    once. Returns True if so. (Endpoint argmin → monotone direction.)"""
    if len(curve) < 3: return False
    am = int(np.argmin(curve))
    if am == 0 or am == len(curve) - 1: return False
    return curve[am - 1] > curve[am] and curve[am + 1] > curve[am]


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
    order_range: tuple = (1, 2, 3, 4, 5)
    fixed_order_for_control: int = 4  # falsifier F2 comparison
    train_cap_per_class: int = 400
    test_cap_per_class: int = 500
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

    # Group training codes by class
    train_samples_per_class = [[] for _ in range(cfg.n_cls)]
    for r in tr_data_capped:
        lbl = r["label"]
        if 0 <= lbl < cfg.n_cls:
            train_samples_per_class[lbl].append(r["code"][:cfg.max_chars])

    val_codes = [r["code"][:cfg.max_chars] for r in vl_data_capped]
    val_labels = list(vl_data_capped["label"])
    val_langs = [r.get("language", "") or "" for r in vl_data_capped]
    val_sources = [r.get("source", "") or "" for r in vl_data_capped]
    test_codes = [r["code"][:cfg.max_chars] for r in ts_data_capped]
    test_labels = list(ts_data_capped["label"])
    test_langs = [r.get("language", "") or "" for r in ts_data_capped]
    test_sources = [r.get("source", "") or "" for r in ts_data_capped]

    logger.info(f"[setup] frac={cfg.frac} train_per_class={[len(s) for s in train_samples_per_class]} "
                f"val={len(val_codes)} test={len(test_codes)} order_range={cfg.order_range}")

    # --- BIC-driven per-class order selection ---
    t0 = time.time()
    lms_best, per_class_best_order, per_class_bic_curve, lms_per_order = train_authors_bic(
        train_samples_per_class, cfg.n_cls, cfg.order_range)
    train_time = time.time() - t0

    order_std = float(np.std(per_class_best_order))
    order_unique = len(set(per_class_best_order))
    n_u_shape = sum(detect_u_shape(curve) for curve in per_class_bic_curve)
    u_shape_rate = n_u_shape / max(cfg.n_cls, 1)
    logger.info(f"[train] per_class_orders={per_class_best_order}  std={order_std:.2f}  "
                f"n_unique={order_unique}  u_shape_rate={u_shape_rate:.2f}  time={train_time:.0f}s")

    # --- Val pass (both BIC-selected and fixed-order-4 modes) ---
    t0 = time.time()
    val_preds_bic = predict_batch(val_codes, lms_best)
    val_met_bic = eval_pack(val_preds_bic, val_labels, val_langs, val_sources,
                             cfg.n_cls, sib_mask, dist_mat)
    val_macro_bic = val_met_bic["overall"]["macro_f1"]

    fixed_order = cfg.fixed_order_for_control
    fixed_lms = lms_per_order.get(fixed_order, lms_best)
    val_preds_fix = predict_batch(val_codes, fixed_lms)
    val_met_fix = eval_pack(val_preds_fix, val_labels, val_langs, val_sources,
                             cfg.n_cls, sib_mask, dist_mat)
    val_macro_fix = val_met_fix["overall"]["macro_f1"]
    val_time = time.time() - t0
    logger.info(f"[val] macro_bic={val_macro_bic:.4f}  macro_fixed{fixed_order}={val_macro_fix:.4f}  "
                f"time={val_time:.0f}s")

    # --- Test pass (both modes) ---
    t0 = time.time()
    test_preds_bic = predict_batch(test_codes, lms_best)
    ts_met_bic = eval_pack(test_preds_bic, test_labels, test_langs, test_sources,
                            cfg.n_cls, sib_mask, dist_mat)
    test_macro_bic = ts_met_bic["overall"]["macro_f1"]

    test_preds_fix = predict_batch(test_codes, fixed_lms)
    ts_met_fix = eval_pack(test_preds_fix, test_labels, test_langs, test_sources,
                            cfg.n_cls, sib_mask, dist_mat)
    test_macro_fix = ts_met_fix["overall"]["macro_f1"]
    test_time = time.time() - t0

    gap_bic = val_macro_bic - test_macro_bic
    macro_diff = test_macro_bic - test_macro_fix
    logger.info(f"[test] macro_bic={test_macro_bic:.4f}  macro_fixed{fixed_order}={test_macro_fix:.4f}  "
                f"diff={macro_diff:+.4f}  gap_bic={gap_bic:+.4f}  time={test_time:.0f}s")

    # Stash falsifier signals into BIC-mode eval pack
    ts_met_bic["falsifier_F1_per_class_best_order"] = list(map(int, per_class_best_order))
    ts_met_bic["falsifier_F1_order_std"] = float(order_std)
    ts_met_bic["falsifier_F1_n_unique_orders"] = int(order_unique)
    ts_met_bic["falsifier_F2_macro_diff_vs_fixed4"] = float(macro_diff)
    ts_met_bic["falsifier_F2_test_macro_fixed_order"] = float(test_macro_fix)
    ts_met_bic["falsifier_F2_fixed_order_used"] = int(fixed_order)
    ts_met_bic["falsifier_F3_per_class_bic_curves"] = per_class_bic_curve
    ts_met_bic["falsifier_F3_u_shape_rate"] = float(u_shape_rate)

    return {
        "tag": tag, "method": "BAYESCRAFT",
        "note": ("Per-author Markov order selected by BIC (Schwarz 1978). "
                 "Attribution by max log-likelihood under each author's LM. CPU-only."),
        "enc": "char-markov-bic",
        "bench": cfg.benchmark, "frac": cfg.frac,
        "order_range": list(cfg.order_range),
        "fixed_order_control": int(fixed_order),
        "per_class_best_order_F1": list(map(int, per_class_best_order)),
        "order_std": float(order_std),
        "n_unique_orders": int(order_unique),
        "u_shape_rate_F3": float(u_shape_rate),
        "per_class_bic_curves_F3": per_class_bic_curve,
        "train_size_after_cap": int(sum(len(s) for s in train_samples_per_class)),
        "test_size_after_cap": int(len(test_codes)),
        # Primary (BIC-selected) numbers
        "val_macro": val_macro_bic, "macro": test_macro_bic,
        "weighted": ts_met_bic["overall"]["weighted_f1"],
        "acc": ts_met_bic["overall"]["accuracy"],
        "val_test_gap": gap_bic, "dpaper": test_macro_bic - PAPER_BASELINE,
        # Control (fixed order) numbers
        "val_macro_fixed_order": float(val_macro_fix),
        "test_macro_fixed_order": float(test_macro_fix),
        "comparison_vs_fixed_order_4_F2": float(macro_diff),
        "train_time_sec": train_time,
        "val_time_sec": val_time, "test_time_sec": test_time,
        "test_metrics": ts_met_bic,
        "test_metrics_fixed_order": ts_met_fix,
        "val_history": [val_macro_bic],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in benchmarks:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp138_bayescraft_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test_bic={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"test_fix4={res['test_macro_fixed_order']:.4f} "
                            f"diff={res['comparison_vs_fixed_order_4_F2']:+.4f} "
                            f"orders={res['per_class_best_order_F1']} "
                            f"time={res['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp138_bayescraft_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Train':>8} {'Test':>8} "
          f"{'ValBIC':>8} {'TestBIC':>8} {'TestFix4':>9} {'Diff':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'OrderStd':>9} {'nUniqOrd':>9} {'UshapeR':>8} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['train_size_after_cap']:>8d} {r['test_size_after_cap']:>8d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['test_macro_fixed_order']:>9.4f} "
              f"{r['comparison_vs_fixed_order_4_F2']:>+8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['order_std']:>9.3f} {r['n_unique_orders']:>9d} "
              f"{r['u_shape_rate_F3']:>8.2f} {r['wall']:>8.0f}s")
    print("="*150)
    print("\nPer-class BIC-selected Markov orders (falsifier F1 — variance should be > 0):")
    for r in results:
        print(f"  {r['bench']:<12} frac={r['frac']:>.0%}  orders={r['per_class_best_order_F1']}  "
              f"std={r['order_std']:.2f}  n_unique={r['n_unique_orders']}")
    print("="*150)


if __name__ == "__main__":
    main()
