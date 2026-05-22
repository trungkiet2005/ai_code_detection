# exp158 - ELM
# =============================================================================
# NAME       : ELM (Extreme Learning Machine on 57-d stylometric features;
#              random frozen input-to-hidden weights + closed-form ridge
#              solution for the output layer).
# REFERENCE  : Huang, Zhu, Siew (2006) "Extreme learning machine: theory
#              and applications", Neurocomputing 70.
# CLAIM      : Random hidden expansion + closed-form linear output gives a
#              CPU-trainable depth-1 non-linear classifier that may close
#              part of the gap to a Random Forest while costing
#              milliseconds. Sweep H, activation, ridge lambda on val.
# EQUATION   : W ~ N(0, 1)^{d x H},   b ~ U[-1, 1]^H   (frozen random)
#              phi(x) = activation(W^T x + b)   in R^H
#              beta   = (Phi^T Phi + lambda I)^{-1} Phi^T Y    (ridge)
#              y_hat  = argmax_c (phi(x) beta)_c
# WHY NEW    : ELM is a 20-year-old ML baseline studied widely outside SE
#              but never applied to AI-code authorship attribution. It
#              tests whether non-linear random feature expansion adds
#              discriminative signal on top of raw stylometry without GPUs.
# WOW HOOK   : "We freeze the input-to-hidden weights at RANDOM and only
#              solve the linear output layer in closed form. A 'neural'
#              network trained in milliseconds on CPU - the cheapest
#              depth-1 non-linearity. Does it close the gap to RandomForest?"
# FALSIFIER  : (F1) ELM (best-on-val) macro >= Linear (Phi=X) by >= 0.005
#              (else random expansion is decorative).
#              (F2) H=128 must NOT be best on the H-sweep (else hidden
#              expansion is useless).
#              (F3) At least 2 activations within +/- 0.003 of best
#              activation (else ELM is fragile).
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
logger = logging.getLogger("exp158_elm")

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
# Data loading (identical plumbing to exp145)
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
# Stylometric feature extraction (57-d, identical to exp145)
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
# ELM core
# =============================================================================

def _activation(name, Z):
    if name == "tanh":
        return np.tanh(Z)
    if name == "sigmoid":
        return 1.0 / (1.0 + np.exp(-np.clip(Z, -30, 30)))
    if name == "relu":
        return np.maximum(Z, 0.0)
    raise ValueError(f"unknown activation: {name}")


def _onehot(y, n_cls):
    Y = np.zeros((len(y), n_cls), dtype=np.float64)
    Y[np.arange(len(y)), y] = 1.0
    return Y


def _elm_fit(X, y, n_cls, H, activation, lam, seed=42, w_scale=1.0):
    """Return (W, b, beta) ready for prediction."""
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W = rng.standard_normal(size=(d, H)).astype(np.float64) * w_scale
    b = rng.uniform(-1.0, 1.0, size=(H,)).astype(np.float64)
    Phi = _activation(activation, X.astype(np.float64) @ W + b)
    Y = _onehot(y, n_cls)
    if lam <= 0.0:
        beta, *_ = np.linalg.lstsq(Phi, Y, rcond=None)
    else:
        # Ridge: (Phi^T Phi + lam I)^-1 Phi^T Y
        PtP = Phi.T @ Phi
        PtY = Phi.T @ Y
        d_h = PtP.shape[0]
        beta = pinvh(PtP + lam * np.eye(d_h), atol=1e-12) @ PtY
    return W, b, beta


def _elm_predict(X, W, b, beta, activation):
    Phi = _activation(activation, X.astype(np.float64) @ W + b)
    scores = Phi @ beta
    return scores.argmax(axis=1)


def _linear_fit(X, y, n_cls, lam):
    Y = _onehot(y, n_cls)
    XtX = X.T @ X
    XtY = X.T @ Y
    d = XtX.shape[0]
    if lam <= 0.0:
        beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    else:
        beta = pinvh(XtX + lam * np.eye(d), atol=1e-12) @ XtY
    return beta


def _linear_predict(X, beta):
    return (X @ beta).argmax(axis=1)


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
    H_sweep: tuple = (128, 256, 512, 1024)
    activation_sweep: tuple = ("tanh", "sigmoid", "relu")
    lambda_sweep: tuple = (0.0, 1e-3, 1e-1)
    w_scale: float = 1.0


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
    Xtr = ((X_train - mu_tr) / sd_tr).astype(np.float64)
    Xvl = ((X_val   - mu_tr) / sd_tr).astype(np.float64)
    Xts = ((X_test  - mu_tr) / sd_tr).astype(np.float64)

    # Sweep H x activation x lambda on val
    sweep_val = {}
    best_key, best_val_macro = None, -1.0
    train_time_total = 0.0
    for H in cfg.H_sweep:
        for act in cfg.activation_sweep:
            for lam in cfg.lambda_sweep:
                t1 = time.time()
                W, b, beta = _elm_fit(Xtr, train_labels, cfg.n_cls,
                                      H=H, activation=act, lam=lam,
                                      seed=cfg.seed, w_scale=cfg.w_scale)
                train_time_total += time.time() - t1
                vpred = _elm_predict(Xvl, W, b, beta, act)
                vmac = float(f1_score(val_labels, vpred, average="macro", zero_division=0))
                key = f"H={H}|act={act}|lam={lam:g}"
                sweep_val[key] = vmac
                if vmac > best_val_macro:
                    best_val_macro = vmac
                    best_key = key
                    best_state = (H, act, lam, W, b, beta)
    logger.info(f"[sweep] {len(sweep_val)} cells; best={best_key}={best_val_macro:.4f}")

    H_best, act_best, lam_best, W_best, b_best, beta_best = best_state
    val_macro = best_val_macro
    test_preds = _elm_predict(Xts, W_best, b_best, beta_best, act_best)
    test_macro = float(f1_score(test_labels, test_preds, average="macro", zero_division=0))
    gap = val_macro - test_macro
    logger.info(f"[head] H={H_best} act={act_best} lam={lam_best:g}  "
                f"val={val_macro:.4f} test={test_macro:.4f} gap={gap:+.4f}")

    # F1: linear baseline (Phi = X identity, no hidden expansion). Sweep lambda on val.
    lin_val_macros = {}
    best_lin_val, best_lin_lam, best_lin_beta = -1.0, None, None
    for lam in cfg.lambda_sweep:
        beta_lin = _linear_fit(Xtr, train_labels, cfg.n_cls, lam=lam)
        vpred = _linear_predict(Xvl, beta_lin)
        vmac = float(f1_score(val_labels, vpred, average="macro", zero_division=0))
        lin_val_macros[f"lam={lam:g}"] = vmac
        if vmac > best_lin_val:
            best_lin_val = vmac; best_lin_lam = lam; best_lin_beta = beta_lin
    lin_test_preds = _linear_predict(Xts, best_lin_beta)
    lin_test_macro = float(f1_score(test_labels, lin_test_preds, average="macro", zero_division=0))
    f1_delta = test_macro - lin_test_macro
    f1_ok = f1_delta >= 0.005
    logger.info(f"[F1] linear (best-lam={best_lin_lam:g}) test={lin_test_macro:.4f}  "
                f"ELM-linear={f1_delta:+.4f}")

    # F2: H-sweep test macros (using best act, best lam per H on val)
    h_test_macros = {}
    for H in cfg.H_sweep:
        # Pick best (act, lam) on val for this H
        best_h_val, best_h_state = -1.0, None
        for act in cfg.activation_sweep:
            for lam in cfg.lambda_sweep:
                key = f"H={H}|act={act}|lam={lam:g}"
                if key in sweep_val and sweep_val[key] > best_h_val:
                    best_h_val = sweep_val[key]
                    best_h_state = (act, lam)
        act_h, lam_h = best_h_state
        W, b, beta = _elm_fit(Xtr, train_labels, cfg.n_cls,
                              H=H, activation=act_h, lam=lam_h,
                              seed=cfg.seed, w_scale=cfg.w_scale)
        tpred = _elm_predict(Xts, W, b, beta, act_h)
        h_test_macros[f"H={H}"] = float(f1_score(test_labels, tpred,
                                                 average="macro", zero_division=0))
    best_H_key = max(h_test_macros, key=h_test_macros.get)
    f2_ok = best_H_key != "H=128"
    logger.info(f"[F2] H-sweep test macros = {h_test_macros}; best={best_H_key}")

    # F3: activation-sweep test macros (using best H, best lam per act on val)
    act_test_macros = {}
    for act in cfg.activation_sweep:
        best_a_val, best_a_state = -1.0, None
        for H in cfg.H_sweep:
            for lam in cfg.lambda_sweep:
                key = f"H={H}|act={act}|lam={lam:g}"
                if key in sweep_val and sweep_val[key] > best_a_val:
                    best_a_val = sweep_val[key]
                    best_a_state = (H, lam)
        H_a, lam_a = best_a_state
        W, b, beta = _elm_fit(Xtr, train_labels, cfg.n_cls,
                              H=H_a, activation=act, lam=lam_a,
                              seed=cfg.seed, w_scale=cfg.w_scale)
        tpred = _elm_predict(Xts, W, b, beta, act)
        act_test_macros[act] = float(f1_score(test_labels, tpred,
                                              average="macro", zero_division=0))
    best_act_score = max(act_test_macros.values())
    near_best = sum(1 for v in act_test_macros.values()
                    if v >= best_act_score - 0.003)
    f3_ok = near_best >= 2
    logger.info(f"[F3] activation-sweep test macros = {act_test_macros}  near-best={near_best}")

    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    ts_met["falsifier_F1_elm_minus_linear_macro"] = float(f1_delta)
    ts_met["falsifier_F1_linear_test_macro"] = float(lin_test_macro)
    ts_met["falsifier_F1_ok"] = bool(f1_ok)
    ts_met["falsifier_F2_h_sweep_val_macros"] = h_test_macros
    ts_met["falsifier_F2_best_H"] = best_H_key
    ts_met["falsifier_F2_ok"] = bool(f2_ok)
    ts_met["falsifier_F3_activation_sweep_val_macros"] = act_test_macros
    ts_met["falsifier_F3_near_best_count"] = int(near_best)
    ts_met["falsifier_F3_ok"] = bool(f3_ok)

    return {
        "tag": tag, "method": "ELM",
        "note": ("Extreme Learning Machine (Huang 2006) on 57-d stylometric "
                 "features: random frozen W,b + closed-form ridge output. "
                 "CPU-only; sweep H x activation x lambda on val."),
        "enc": f"stylometric-d{N_FEATURES}", "bench": cfg.benchmark,
        "frac": cfg.frac,
        "n_features": int(N_FEATURES),
        "best_H": int(H_best), "best_activation": str(act_best),
        "best_lambda": float(lam_best),
        "best_linear_lambda": float(best_lin_lam),
        "train_size_after_cap": int(len(train_codes)),
        "val_size_after_cap":   int(len(val_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "linear_test_macro": float(lin_test_macro),
        "elm_minus_linear": float(f1_delta),
        "sweep_val_macros": sweep_val,
        "h_test_macros": h_test_macros,
        "activation_test_macros": act_test_macros,
        "linear_val_macros": lin_val_macros,
        "falsifier_F1_ok": bool(f1_ok),
        "falsifier_F2_ok": bool(f2_ok),
        "falsifier_F3_ok": bool(f3_ok),
        "feat_time_sec": feat_time,
        "train_time_sec_total_sweep": train_time_total,
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
            tag = f"exp158_elm_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] H={res['best_H']} act={res['best_activation']} "
                            f"lam={res['best_lambda']:g}  "
                            f"test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} "
                            f"ELM-linear={res['elm_minus_linear']:+.4f} "
                            f"time={res['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp158_elm_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*160)
    print(f"{'Benchmark':<12} {'Frac':>6} {'H':>5} {'Act':>8} {'Lam':>8} "
          f"{'Train':>7} {'Test':>7} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'ELM-Lin':>9} {'Wall':>8}")
    print("-"*160)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['best_H']:>5d} "
              f"{r['best_activation']:>8} {r['best_lambda']:>8.0e} "
              f"{r['train_size_after_cap']:>7d} {r['test_size_after_cap']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['elm_minus_linear']:>+9.4f} {r['wall']:>8.0f}s")
    print("="*160)


if __name__ == "__main__":
    main()
