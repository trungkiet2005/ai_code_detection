# exp156 - MAHALANOBIS
# =============================================================================
# NAME       : MAHALANOBIS (Covariance-aware nearest-class attribution using
#              the full regularised pooled covariance over 57-d stylometric
#              features; equivalent to Fisher's LDA decision rule).
# REFERENCE  : Fisher (1936) "The use of multiple measurements in taxonomic
#              problems"; Mahalanobis (1936) "On the generalised distance in
#              statistics"; Caliskan et al. 2015 (stylometric features for
#              human-code attribution).
# CLAIM      : Replace L2 nearest-centroid with Mahalanobis nearest-centroid
#              using a single regularised pooled within-class covariance
#              shared across classes. Pick lambda (ridge on covariance) on
#              val. Compare to per-class covariance (QDA) and diagonal-only
#              (naive Gaussian) variants.
# EQUATION   : mu_c       = (1/N_c) sum_{x in train_c} x
#              Sigma_pool = (1/N) sum_c sum_{x in train_c} (x-mu_c)(x-mu_c)^T
#              P          = inv(Sigma_pool + lambda * (trace/d) * I)
#              d_c(x)     = (x - mu_c)^T P (x - mu_c)
#              y_hat      = argmin_c d_c(x)
# WHY NEW    : Every nearest-centroid baseline in code-authorship uses raw
#              L2. No published code-attribution work uses raw Mahalanobis
#              with regularised pooled covariance, even though it is the
#              optimal linear decision rule under shared-Gaussian
#              assumptions and 1936 statistics.
# WOW HOOK   : "L2 distance treats every feature dimension as equally
#              informative. Mahalanobis asks: how unusual is this
#              displacement given the FULL covariance of how authors vary
#              together? We use Fisher's 1936 decision rule on stylometric
#              features for the first time on AI-code authorship."
# FALSIFIER  : (F1) Mahalanobis (pooled) macro >= L2 nearest-centroid + 0.005
#              (else covariance buys nothing).
#              (F2) lambda=1e-3 must NOT be worst across the sweep
#              {1e-4, 1e-3, 1e-2, 1e-1}.
#              (F3) Pooled vs per-class vs diagonal: pooled should be best
#              or within +/- 0.003 of best; diagonal must NOT be the best
#              (else covariance structure is decorative).
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, re
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


_ensure("numpy"); _ensure("scipy"); _ensure("scikit-learn")
_ensure("datasets"); _ensure("tqdm"); _ensure("pandas")

import numpy as np
from scipy.linalg import pinvh
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp156_mahalanobis")

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
# Data loading (identical plumbing to exp145_rforest)
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
# Stylometric feature extraction (57-d, identical to exp145_rforest)
# =============================================================================

PYTHON_KEYWORDS = ["if", "else", "elif", "for", "while", "def", "return", "class",
                   "import", "from", "as", "try", "except", "finally", "with",
                   "yield", "lambda", "True", "False", "None", "and", "or", "not",
                   "in", "is", "pass", "break", "continue"]
C_KEYWORDS = ["if", "else", "for", "while", "return", "int", "char", "void",
              "float", "double", "struct", "typedef", "static", "const"]
OPS = ["==", "!=", "<=", ">=", "&&", "||", "++", "--", "+=", "-=", "*=", "/=",
       "+", "-", "*", "/", "%", "=", "<", ">", "!", "&", "|", "^", "~"]
KEYWORDS_15 = (PYTHON_KEYWORDS + C_KEYWORDS)[:15]


def _build_feature_names():
    names = []
    names += [f"line_len_{s}" for s in ["mean", "std", "max", "min", "q25", "q75"]]
    names += ["indent_mean", "indent_max", "indent_change_rate"]
    names += ["id_len_mean", "id_len_std", "snake_ratio", "camel_ratio"]
    names += [f"op_{op}" for op in OPS]
    names += [f"kw_{k}" for k in KEYWORDS_15]
    names += ["comment_py_per_line", "comment_c_per_line",
              "comment_char_density", "comment_present"]
    names += ["space_density", "tab_density", "punc_density", "alpha_ratio"]
    return names


FEATURE_NAMES = _build_feature_names()
N_FEATURES = len(FEATURE_NAMES)


def extract_features(code):
    code = code[:8000]
    lines = code.split("\n")
    n_lines = max(len(lines), 1)
    n_chars = max(len(code), 1)
    line_lens = [len(ln) for ln in lines]
    if line_lens:
        f_line = [
            float(np.mean(line_lens)), float(np.std(line_lens)),
            float(max(line_lens)), float(min(line_lens)),
            float(np.percentile(line_lens, 25)),
            float(np.percentile(line_lens, 75)),
        ]
    else:
        f_line = [0.0] * 6
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    if indents:
        f_indent = [
            float(np.mean(indents)), float(max(indents)),
            float(sum(1 for i in range(1, len(indents)) if indents[i] != indents[i - 1])
                  / max(len(indents), 1)),
        ]
    else:
        f_indent = [0.0, 0.0, 0.0]
    identifiers = re.findall(r"\b[a-zA-Z_]\w+\b", code)
    id_lens = [len(i) for i in identifiers] if identifiers else [0]
    snake = sum(1 for i in identifiers if "_" in i)
    camel = sum(1 for i in identifiers if any(c.isupper() for c in i[1:]))
    f_id = [
        float(np.mean(id_lens)), float(np.std(id_lens)),
        float(snake / max(len(identifiers), 1)),
        float(camel / max(len(identifiers), 1)),
    ]
    f_ops = [float(code.count(op) / n_chars) for op in OPS]
    f_kw = [float(len(re.findall(rf"\b{k}\b", code)) / max(len(identifiers), 1))
            for k in KEYWORDS_15]
    comments_py = re.findall(r"#[^\n]*", code)
    comments_c = re.findall(r"//[^\n]*|/\*[\s\S]*?\*/", code)
    f_comment = [
        float(len(comments_py) / n_lines),
        float(len(comments_c) / n_lines),
        float(sum(len(c) for c in comments_py + comments_c) / n_chars),
        1.0 if comments_py or comments_c else 0.0,
    ]
    n_space = code.count(" ")
    n_tab = code.count("\t")
    n_alpha = sum(1 for c in code if c.isalpha())
    n_punc = sum(1 for c in code if not c.isalnum() and not c.isspace())
    f_ws = [
        float(n_space / n_chars), float(n_tab / n_chars),
        float(n_punc / n_chars),
        float(n_alpha / max(n_alpha + n_punc, 1)),
    ]
    features = f_line + f_indent + f_id + f_ops + f_kw + f_comment + f_ws
    return np.array(features, dtype=np.float32)


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
# Mahalanobis attribution
# =============================================================================

def _class_means(X, y, n_cls):
    mus = np.zeros((n_cls, X.shape[1]), dtype=np.float64)
    counts = np.zeros(n_cls, dtype=np.int64)
    for c in range(n_cls):
        sel = (y == c)
        counts[c] = int(sel.sum())
        if counts[c] > 0:
            mus[c] = X[sel].mean(axis=0)
    return mus, counts


def _pooled_covariance(X, y, mus, n_cls):
    """Pooled within-class covariance: (1/N) * sum_c sum_{x in c} (x-mu_c)(x-mu_c)^T."""
    d = X.shape[1]
    Sigma = np.zeros((d, d), dtype=np.float64)
    for c in range(n_cls):
        sel = (y == c)
        if sel.sum() == 0: continue
        diffs = X[sel].astype(np.float64) - mus[c]
        Sigma += diffs.T @ diffs
    Sigma /= max(X.shape[0], 1)
    return Sigma


def _per_class_covariance(X, y, mus, n_cls, eps=1e-6):
    """Per-class covariance for QDA-style attribution."""
    d = X.shape[1]
    Sigmas = np.zeros((n_cls, d, d), dtype=np.float64)
    for c in range(n_cls):
        sel = (y == c)
        if sel.sum() < 2:
            Sigmas[c] = np.eye(d) * eps
            continue
        diffs = X[sel].astype(np.float64) - mus[c]
        Sigmas[c] = (diffs.T @ diffs) / max(sel.sum(), 1)
    return Sigmas


def _regularised_inv(Sigma, lam_mult):
    d = Sigma.shape[0]
    trace = np.trace(Sigma)
    lam = lam_mult * (trace / max(d, 1))
    return pinvh(Sigma + lam * np.eye(d), atol=1e-12)


def _mahalanobis_predict_pooled(X_query, mus, P):
    """Return argmin_c (x - mu_c)^T P (x - mu_c) for each query."""
    Xq = X_query.astype(np.float64)
    K = mus.shape[0]
    dists = np.zeros((Xq.shape[0], K), dtype=np.float64)
    for c in range(K):
        diff = Xq - mus[c]
        dists[:, c] = np.einsum("ni,ij,nj->n", diff, P, diff)
    return dists.argmin(axis=1)


def _mahalanobis_predict_per_class(X_query, mus, Ps, log_dets):
    """QDA: y_hat = argmin_c [ d_c(x) + log|Sigma_c| ]."""
    Xq = X_query.astype(np.float64)
    K = mus.shape[0]
    scores = np.zeros((Xq.shape[0], K), dtype=np.float64)
    for c in range(K):
        diff = Xq - mus[c]
        scores[:, c] = np.einsum("ni,ij,nj->n", diff, Ps[c], diff) + log_dets[c]
    return scores.argmin(axis=1)


def _l2_nc_predict(X_query, mus):
    Xq = X_query.astype(np.float64)
    K = mus.shape[0]
    d2 = np.zeros((Xq.shape[0], K), dtype=np.float64)
    for c in range(K):
        diff = Xq - mus[c]
        d2[:, c] = (diff * diff).sum(axis=1)
    return d2.argmin(axis=1)


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
    lambda_sweep: tuple = (1e-4, 1e-3, 1e-2, 1e-1)


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
    X_train = extract_features_parallel(train_codes, n_workers=n_workers, desc="phi(train)")
    X_val   = extract_features_parallel(val_codes,   n_workers=n_workers, desc="phi(val)")
    X_test  = extract_features_parallel(test_codes,  n_workers=n_workers, desc="phi(test)")
    feat_time = time.time() - t0
    for X in (X_train, X_val, X_test):
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    logger.info(f"[feat] {X_train.shape[0]+X_val.shape[0]+X_test.shape[0]} vectors "
                f"d={X_train.shape[1]} in {feat_time:.0f}s")

    # Standardise on train
    mu_tr = X_train.mean(axis=0)
    sd_tr = X_train.std(axis=0) + 1e-8
    Xtr = (X_train - mu_tr) / sd_tr
    Xvl = (X_val   - mu_tr) / sd_tr
    Xts = (X_test  - mu_tr) / sd_tr

    # Class means + covariances on TRAIN only
    mus, counts = _class_means(Xtr, train_labels, cfg.n_cls)
    Sigma_pool = _pooled_covariance(Xtr, train_labels, mus, cfg.n_cls)
    Sigmas_pc  = _per_class_covariance(Xtr, train_labels, mus, cfg.n_cls)

    # (a) Pooled with lambda sweep on val
    lambda_val_macros = {}
    best_lam, best_val = None, -1.0
    for lam in cfg.lambda_sweep:
        P = _regularised_inv(Sigma_pool, lam)
        vpred = _mahalanobis_predict_pooled(Xvl, mus, P)
        vmac = float(f1_score(val_labels, vpred, average="macro", zero_division=0))
        lambda_val_macros[f"{lam:g}"] = vmac
        if vmac > best_val:
            best_val = vmac; best_lam = lam
    logger.info(f"[F2] lambda sweep (val macros): {lambda_val_macros}  best lam={best_lam:g}")

    P_best = _regularised_inv(Sigma_pool, best_lam)
    val_preds_pooled = _mahalanobis_predict_pooled(Xvl, mus, P_best)
    val_macro_pooled = float(f1_score(val_labels, val_preds_pooled, average="macro", zero_division=0))
    test_preds_pooled = _mahalanobis_predict_pooled(Xts, mus, P_best)
    test_macro_pooled = float(f1_score(test_labels, test_preds_pooled, average="macro", zero_division=0))

    # (b) Per-class QDA (regularised same lambda fashion)
    Ps_pc = []
    log_dets = np.zeros(cfg.n_cls)
    for c in range(cfg.n_cls):
        d = Sigmas_pc[c].shape[0]
        lam = best_lam * (np.trace(Sigmas_pc[c]) / max(d, 1))
        S_reg = Sigmas_pc[c] + lam * np.eye(d)
        Ps_pc.append(pinvh(S_reg, atol=1e-12))
        sign, logdet = np.linalg.slogdet(S_reg)
        log_dets[c] = logdet if sign > 0 else 0.0
    Ps_pc = np.stack(Ps_pc, axis=0)
    val_preds_pc = _mahalanobis_predict_per_class(Xvl, mus, Ps_pc, log_dets)
    val_macro_pc = float(f1_score(val_labels, val_preds_pc, average="macro", zero_division=0))
    test_preds_pc = _mahalanobis_predict_per_class(Xts, mus, Ps_pc, log_dets)
    test_macro_pc = float(f1_score(test_labels, test_preds_pc, average="macro", zero_division=0))

    # (c) Diagonal-only naive Gaussian
    diag = np.diag(np.diag(Sigma_pool))
    P_diag = _regularised_inv(diag, best_lam)
    val_preds_diag = _mahalanobis_predict_pooled(Xvl, mus, P_diag)
    val_macro_diag = float(f1_score(val_labels, val_preds_diag, average="macro", zero_division=0))
    test_preds_diag = _mahalanobis_predict_pooled(Xts, mus, P_diag)
    test_macro_diag = float(f1_score(test_labels, test_preds_diag, average="macro", zero_division=0))

    # L2 nearest-centroid baseline for F1
    val_preds_l2 = _l2_nc_predict(Xvl, mus)
    val_macro_l2 = float(f1_score(val_labels, val_preds_l2, average="macro", zero_division=0))
    test_preds_l2 = _l2_nc_predict(Xts, mus)
    test_macro_l2 = float(f1_score(test_labels, test_preds_l2, average="macro", zero_division=0))

    # Headline = best of (a)/(b)/(c) on val
    variant_val = {"pooled": val_macro_pooled, "per_class": val_macro_pc, "diagonal": val_macro_diag}
    variant_test = {"pooled": test_macro_pooled, "per_class": test_macro_pc, "diagonal": test_macro_diag}
    headline_variant = max(variant_val, key=variant_val.get)
    val_macro = variant_val[headline_variant]
    test_macro = variant_test[headline_variant]
    head_preds = {"pooled": test_preds_pooled,
                  "per_class": test_preds_pc,
                  "diagonal": test_preds_diag}[headline_variant]
    gap = val_macro - test_macro
    logger.info(f"[head] variant={headline_variant} val={val_macro:.4f} "
                f"test={test_macro:.4f} gap={gap:+.4f}")
    logger.info(f"[F1] L2-NC test={test_macro_l2:.4f}  pooled-Mahal test={test_macro_pooled:.4f}  "
                f"delta={test_macro_pooled - test_macro_l2:+.4f}")

    ts_met = eval_pack(head_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)

    # Falsifiers
    lam_vals = list(lambda_val_macros.values())
    lam_keys = list(lambda_val_macros.keys())
    worst_lam_key = lam_keys[int(np.argmin(lam_vals))]
    f2_ok = worst_lam_key != f"{1e-3:g}"

    variant_macros_test = {"pooled": test_macro_pooled,
                           "per_class": test_macro_pc,
                           "diagonal": test_macro_diag}
    best_variant_test = max(variant_macros_test, key=variant_macros_test.get)
    diag_is_best = (best_variant_test == "diagonal")
    pooled_within_3 = (variant_macros_test["pooled"]
                       >= variant_macros_test[best_variant_test] - 0.003)

    f1_pooled_minus_l2 = test_macro_pooled - test_macro_l2
    f1_ok = f1_pooled_minus_l2 >= 0.005

    ts_met["falsifier_F1_pooled_minus_l2_nc"] = float(f1_pooled_minus_l2)
    ts_met["falsifier_F1_ok"] = bool(f1_ok)
    ts_met["falsifier_F2_lambda_sweep_val_macros"] = lambda_val_macros
    ts_met["falsifier_F2_worst_lambda"] = worst_lam_key
    ts_met["falsifier_F2_ok"] = bool(f2_ok)
    ts_met["falsifier_F3_pooled_vs_per_class_vs_diag"] = variant_macros_test
    ts_met["falsifier_F3_diag_is_best"] = bool(diag_is_best)
    ts_met["falsifier_F3_pooled_within_0p003_of_best"] = bool(pooled_within_3)
    ts_met["falsifier_F3_ok"] = bool((not diag_is_best) and pooled_within_3)

    return {
        "tag": tag, "method": "MAHALANOBIS",
        "note": ("Covariance-aware nearest-class attribution on 57-d "
                 "stylometric features (Fisher 1936). Lambda picked on val. "
                 "Headline = best of pooled / per-class (QDA) / diagonal. "
                 "CPU-only, no neural encoder."),
        "enc": f"stylometric-d{N_FEATURES}", "bench": cfg.benchmark,
        "frac": cfg.frac,
        "headline_variant": headline_variant,
        "best_lambda_mult": float(best_lam),
        "n_features": int(N_FEATURES),
        "train_size_after_cap": int(len(train_codes)),
        "val_size_after_cap":   int(len(val_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "variant_val_macros":  variant_val,
        "variant_test_macros": variant_macros_test,
        "lambda_val_macros": lambda_val_macros,
        "test_macro_pooled":    test_macro_pooled,
        "test_macro_per_class": test_macro_pc,
        "test_macro_diagonal":  test_macro_diag,
        "test_macro_l2_nc":     test_macro_l2,
        "falsifier_F1_pooled_minus_l2": float(f1_pooled_minus_l2),
        "falsifier_F1_ok": bool(f1_ok),
        "falsifier_F2_ok": bool(f2_ok),
        "falsifier_F3_ok": bool((not diag_is_best) and pooled_within_3),
        "feat_time_sec": feat_time,
        "test_metrics": ts_met,
        "val_history": [val_macro_pooled, val_macro_pc, val_macro_diag],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in benchmarks:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp156_mahalanobis_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] var={res['headline_variant']} "
                            f"test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} "
                            f"F1(pooled-L2)={res['falsifier_F1_pooled_minus_l2']:+.4f} "
                            f"time={res['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp156_mahalanobis_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Variant':>10} {'Lambda':>8} "
          f"{'Train':>7} {'Test':>7} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'F1(P-L2)':>10} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['headline_variant']:>10} "
              f"{r['best_lambda_mult']:>8.0e} "
              f"{r['train_size_after_cap']:>7d} {r['test_size_after_cap']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['falsifier_F1_pooled_minus_l2']:>+10.4f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
