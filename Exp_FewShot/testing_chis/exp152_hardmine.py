# exp152 — HARDMINE
# =============================================================================
# NAME       : HARDMINE (Margin-based exponential sample re-weighting on
#              Random Forest stylometric attribution).
# REFERENCE  : Lin et al. 2017 ("Focal Loss for Dense Object Detection",
#              ICCV) for the inverse-margin re-weighting idea. Shrivastava
#              et al. 2016 ("Training Region-based Object Detectors with
#              Online Hard Example Mining", CVPR) for the hard-example
#              philosophy. Breiman 2001 (Random Forests) for the base
#              classifier. The combination "RF + margin-aware sample
#              weights" has not been used in code authorship attribution.
# CLAIM      : The saturation plateau is not a true ceiling — the easy half
#              of the training data has effectively zero gradient but still
#              counts in tree-split impurity. Margin-based exponential
#              re-weighting (large weight for low-margin / boundary samples)
#              re-focuses a second-pass Random Forest on the hard half and
#              breaks the plateau, or proves the plateau is real.
# EQUATION   : margin_i      = p_clf1(y_i | x_i) - max_{c != y_i} p_clf1(c | x_i)
#              w_i           = exp(-beta * margin_i) / Z
#              clf_2         = RF.fit(X, y, sample_weight=w)
#              y_hat         = argmax_c p_clf2(c | x_query)
#              beta in {0.0, 1.0, 2.0, 4.0}, pick best on VAL.
# WHY NEW    : Focal loss (Lin 2017) and OHEM (Shrivastava 2016) are well-
#              known for deep classifiers, but the analogous *margin-based
#              exponential re-weighting* for tree ensembles has not been
#              used in code authorship attribution. RF + margin-aware sample
#              weights is novel for this benchmark and is a clean CPU-only
#              stand-in for focal-loss-style hard-example mining.
# WOW HOOK   : "At the saturation plateau the easy half of the data is
#              already correctly classified — its gradient is zero, but its
#              WEIGHT still counts. We exponentially re-weight by inverse
#              margin and re-train: the plateau either breaks or proves
#              it's a true ceiling."
# FALSIFIER  : (F1) beta-sweep val-macros: beta=0 must NOT be best on val
#              (else hardmine is decorative).
#              (F2) best_beta_macro - beta0_macro >= 0.003 means hardmine
#              helps.
#              (F3) hardmine_minus_baseline at frac=0.20 specifically: if
#              >= 0.003, the saturation plateau is breakable by re-weighting.
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

_ensure("numpy"); _ensure("scipy"); _ensure("scikit-learn"); _ensure("datasets"); _ensure("tqdm"); _ensure("pandas")

import numpy as np
from datasets import load_dataset
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp152_hardmine")

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
# Stylometric features (identical to exp145)
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
        f_line = [float(np.mean(line_lens)), float(np.std(line_lens)),
                  float(max(line_lens)), float(min(line_lens)),
                  float(np.percentile(line_lens, 25)),
                  float(np.percentile(line_lens, 75))]
    else:
        f_line = [0.0] * 6
    indents = [len(ln) - len(ln.lstrip()) for ln in lines if ln.strip()]
    if indents:
        f_indent = [float(np.mean(indents)), float(max(indents)),
                    float(sum(1 for i in range(1, len(indents)) if indents[i] != indents[i - 1])
                          / max(len(indents), 1))]
    else:
        f_indent = [0.0, 0.0, 0.0]
    identifiers = re.findall(r"\b[a-zA-Z_]\w+\b", code)
    id_lens = [len(i) for i in identifiers] if identifiers else [0]
    snake = sum(1 for i in identifiers if "_" in i)
    camel = sum(1 for i in identifiers if any(c.isupper() for c in i[1:]))
    f_id = [float(np.mean(id_lens)), float(np.std(id_lens)),
            float(snake / max(len(identifiers), 1)),
            float(camel / max(len(identifiers), 1))]
    f_ops = [float(code.count(op) / n_chars) for op in OPS]
    f_kw = [float(len(re.findall(rf"\b{k}\b", code)) / max(len(identifiers), 1))
            for k in KEYWORDS_15]
    comments_py = re.findall(r"#[^\n]*", code)
    comments_c = re.findall(r"//[^\n]*|/\*[\s\S]*?\*/", code)
    f_comment = [float(len(comments_py) / n_lines),
                 float(len(comments_c) / n_lines),
                 float(sum(len(c) for c in comments_py + comments_c) / n_chars),
                 1.0 if comments_py or comments_c else 0.0]
    n_space = code.count(" "); n_tab = code.count("\t")
    n_alpha = sum(1 for c in code if c.isalpha())
    n_punc = sum(1 for c in code if not c.isalnum() and not c.isspace())
    f_ws = [float(n_space / n_chars), float(n_tab / n_chars),
            float(n_punc / n_chars),
            float(n_alpha / max(n_alpha + n_punc, 1))]
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
# Random Forest training (with optional sample_weight)
# =============================================================================

def train_rf(X_train, y_train, n_estimators=200, max_depth=None, n_jobs=-1, seed=42,
             sample_weight=None):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        n_jobs=n_jobs, random_state=seed, class_weight="balanced")
    if sample_weight is not None:
        rf.fit(X_train, y_train, sample_weight=sample_weight)
    else:
        rf.fit(X_train, y_train)
    return rf


def compute_margins(proba, y):
    """proba: [N, n_cls]; y: [N]. Returns margin_i = p_correct - max_other in [-1, 1]."""
    N, K = proba.shape
    p_correct = proba[np.arange(N), y]
    # mask out correct class then take max of the rest
    mask = np.ones_like(proba, dtype=bool)
    mask[np.arange(N), y] = False
    other = np.where(mask, proba, -np.inf)
    p_max_other = other.max(axis=1)
    # Guard if K==1
    p_max_other = np.where(np.isneginf(p_max_other), 0.0, p_max_other)
    return p_correct - p_max_other


def margin_to_weights(margins, beta):
    """w_i = exp(-beta * margin_i), normalised to mean 1.0."""
    if beta <= 0.0:
        return np.ones_like(margins, dtype=np.float64)
    w = np.exp(-beta * margins.astype(np.float64))
    # Normalise so mean(w) == 1 (keeps tree growth scale stable)
    w = w * (len(w) / (w.sum() + 1e-12))
    return w


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
    n_estimators: int = 200
    max_depth: int = None
    train_cap_per_class: int = 300
    test_cap_per_class: int = 400
    val_cap_per_class: int = 200
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = 4000
    beta_grid: tuple = (0.0, 1.0, 2.0, 4.0)


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

    tr_data_frac   = stratified_subsample(tr_data, cfg.frac, seed=cfg.seed)
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
                f"test={len(test_codes)} n_est={cfg.n_estimators} workers={n_workers} "
                f"d_feat={N_FEATURES}")

    # Extract features
    t0 = time.time()
    X_train = extract_features_parallel(train_codes, n_workers=n_workers, desc="phi(train)")
    X_val   = extract_features_parallel(val_codes,   n_workers=n_workers, desc="phi(val)")
    X_test  = extract_features_parallel(test_codes,  n_workers=n_workers, desc="phi(test)")
    feat_time = time.time() - t0

    for X in (X_train, X_val, X_test):
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # PASS 1: baseline RF (uniform weights) — used for margins AND as beta=0 baseline.
    t0 = time.time()
    rf1 = train_rf(X_train, train_labels,
                   n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
                   n_jobs=n_workers, seed=cfg.seed, sample_weight=None)
    pass1_time = time.time() - t0
    train_proba_p1 = rf1.predict_proba(X_train)
    # rf1.classes_ may not be [0..n_cls-1] if a class is missing in train — guard.
    classes_ = rf1.classes_
    if list(classes_) != list(range(cfg.n_cls)):
        # Re-map predict_proba columns onto the full class space.
        full_proba = np.zeros((train_proba_p1.shape[0], cfg.n_cls), dtype=train_proba_p1.dtype)
        for j, c in enumerate(classes_):
            if 0 <= c < cfg.n_cls:
                full_proba[:, c] = train_proba_p1[:, j]
        train_proba_p1 = full_proba
    margins = compute_margins(train_proba_p1, train_labels)
    logger.info(f"[pass1] trained baseline RF in {pass1_time:.0f}s; "
                f"margin mean={float(margins.mean()):.3f} "
                f"min={float(margins.min()):.3f} max={float(margins.max()):.3f}")

    # Beta-sweep: train pass-2 RFs for each beta, evaluate on VAL, pick best.
    beta_val_macros: Dict[float, float] = {}
    beta_val_packs:  Dict[float, dict]  = {}
    beta_test_packs: Dict[float, dict]  = {}
    beta_test_macros: Dict[float, float] = {}
    pass2_times: Dict[float, float] = {}

    for beta in cfg.beta_grid:
        beta = float(beta)
        if beta == 0.0:
            # Re-use pass-1 model (uniform weights == pass 1).
            val_preds  = rf1.predict(X_val)
            test_preds = rf1.predict(X_test)
            pass2_times[beta] = 0.0
        else:
            w = margin_to_weights(margins, beta)
            t0 = time.time()
            rf2 = train_rf(X_train, train_labels,
                           n_estimators=cfg.n_estimators, max_depth=cfg.max_depth,
                           n_jobs=n_workers, seed=cfg.seed, sample_weight=w)
            pass2_times[beta] = time.time() - t0
            val_preds  = rf2.predict(X_val)
            test_preds = rf2.predict(X_test)
        val_pack  = eval_pack(val_preds,  val_labels,  val_langs,  val_sources,
                              cfg.n_cls, sib_mask, dist_mat)
        test_pack = eval_pack(test_preds, test_labels, test_langs, test_sources,
                              cfg.n_cls, sib_mask, dist_mat)
        beta_val_macros[beta]  = val_pack["overall"]["macro_f1"]
        beta_val_packs[beta]   = val_pack
        beta_test_macros[beta] = test_pack["overall"]["macro_f1"]
        beta_test_packs[beta]  = test_pack
        logger.info(f"[beta={beta:.1f}] val={beta_val_macros[beta]:.4f} "
                    f"test={beta_test_macros[beta]:.4f} "
                    f"t={pass2_times[beta]:.0f}s")

    # Choose beta by VAL macro (ties broken by smaller beta).
    best_beta = max(beta_val_macros.keys(),
                    key=lambda b: (beta_val_macros[b], -b))
    val_macro  = beta_val_macros[best_beta]
    test_macro = beta_test_macros[best_beta]
    ts_met     = beta_test_packs[best_beta]
    gap = val_macro - test_macro
    logger.info(f"[headline] best_beta={best_beta:.1f} "
                f"val={val_macro:.4f} test={test_macro:.4f} gap={gap:+.4f}")

    # Falsifiers
    beta0_test_macro = beta_test_macros[0.0]
    f2_delta = test_macro - beta0_test_macro

    ts_met["falsifier_F1_beta_sweep_val_macros"] = {
        f"{b:.1f}": float(beta_val_macros[b]) for b in cfg.beta_grid
    }
    ts_met["falsifier_F1_beta_sweep_test_macros"] = {
        f"{b:.1f}": float(beta_test_macros[b]) for b in cfg.beta_grid
    }
    ts_met["falsifier_F2_hardmine_minus_baseline"] = float(f2_delta)
    ts_met["best_beta"]            = float(best_beta)
    ts_met["beta0_test_macro"]     = float(beta0_test_macro)
    ts_met["beta0_val_macro"]      = float(beta_val_macros[0.0])
    ts_met["best_beta_test_macro"] = float(test_macro)
    ts_met["best_beta_val_macro"]  = float(val_macro)
    ts_met["margin_mean"] = float(margins.mean())
    ts_met["margin_min"]  = float(margins.min())
    ts_met["margin_max"]  = float(margins.max())

    return {
        "tag": tag, "method": "HARDMINE",
        "note": ("Margin-based exponential sample re-weighting on Random Forest. "
                 "Pass-1 RF computes margins -> pass-2 RF re-trained with "
                 "w_i = exp(-beta * margin_i). beta picked by val. CPU-only."),
        "enc": f"stylometric-d{N_FEATURES}+hardmine", "bench": cfg.benchmark,
        "frac": cfg.frac, "n_estimators": cfg.n_estimators,
        "n_features": int(N_FEATURES),
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "best_beta": float(best_beta),
        "beta0_test_macro": float(beta0_test_macro),
        "beta0_val_macro":  float(beta_val_macros[0.0]),
        "falsifier_F1_beta_sweep_val_macros": {
            f"{b:.1f}": float(beta_val_macros[b]) for b in cfg.beta_grid
        },
        "falsifier_F1_beta_sweep_test_macros": {
            f"{b:.1f}": float(beta_test_macros[b]) for b in cfg.beta_grid
        },
        "falsifier_F2_hardmine_minus_baseline": float(f2_delta),
        "feat_time_sec": feat_time, "pass1_time_sec": pass1_time,
        "pass2_times": {f"{b:.1f}": float(pass2_times[b]) for b in cfg.beta_grid},
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
            tag = f"exp152_hardmine_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} "
                            f"best_beta={res['best_beta']:.1f} "
                            f"F2_delta={res['falsifier_F2_hardmine_minus_baseline']:+.4f} "
                            f"time={res['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()

    # F3: hardmine_minus_baseline specifically at frac=0.20 (saturation plateau).
    f3_per_bench = {}
    for bench, _, _ in benchmarks:
        r20 = next((r for r in results if r["bench"] == bench and r["frac"] == 0.20), None)
        if r20 is not None:
            f3_per_bench[bench] = {
                "frac":               0.20,
                "best_beta":          float(r20["best_beta"]),
                "beta0_test_macro":   float(r20["beta0_test_macro"]),
                "best_beta_test_macro": float(r20["macro"]),
                "f3_delta":           float(r20["falsifier_F2_hardmine_minus_baseline"]),
            }
    for r in results:
        r["falsifier_F3_hardmine_minus_baseline_at_20pct"] = f3_per_bench.get(r["bench"], {})

    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp152_hardmine_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*160)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Train':>8} {'Test':>8} {'Val-F1':>8} "
          f"{'Test-F1':>8} {'Gap':>8} {'dPaper':>9} {'Beta*':>6} "
          f"{'Beta0':>8} {'F2delta':>9} {'Wall':>8}")
    print("-"*160)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['train_size_after_cap']:>8d} {r['test_size_after_cap']:>8d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['best_beta']:>6.1f} "
              f"{r['beta0_test_macro']:>8.4f} "
              f"{r['falsifier_F2_hardmine_minus_baseline']:>+9.4f} "
              f"{r['wall']:>8.0f}s")
    print("="*160)
    print(f"F3 (hardmine - baseline @ frac=0.20): {json.dumps(f3_per_bench, indent=2)}")


if __name__ == "__main__":
    main()
