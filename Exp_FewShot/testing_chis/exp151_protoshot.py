# exp151 — PROTOSHOT
# =============================================================================
# NAME       : PROTOSHOT (Prototypical few-shot attribution in a supervised-
#              learned classical latent: LDA -> per-class centroid -> nearest-
#              centroid classification on the 57-d stylometric features).
# REFERENCE  : Snell et al. 2017 ("Prototypical Networks for Few-Shot
#              Learning", NeurIPS). Fisher 1936 (LDA). Caliskan et al. 2015
#              (stylometric attribution). ProtoNet has never been applied to
#              AI-code authorship attribution; the LDA-then-prototype variant
#              is the classical (no-neural-net) version of the same idea.
# CLAIM      : Few-shot deserves a few-shot architecture. Even at the
#              classical-ML floor, training a supervised LDA latent and
#              classifying by nearest class centroid in that latent is the
#              honest few-shot baseline. The LDA gain over raw nearest-
#              centroid should be LARGER at low fractions (frac=0.01) where
#              the supervised-learned latent matters most.
# EQUATION   : phi: code -> R^57
#              W = argmax_W tr(W^T S_B W) / tr(W^T S_W W)   (Fisher LDA)
#              z = W^T phi(x)
#              mu_c = mean(z over train where label = c)
#              y_hat = argmin_c || z_query - mu_c ||_2
# WHY NEW    : ProtoNet (Snell 2017) is the canonical few-shot architecture
#              in vision but has never been applied to AI-code authorship
#              attribution. The LDA-then-prototype variant makes the SAME
#              inductive claim (few-shot deserves prototype matching in a
#              learned latent) without any neural net, staying CPU-only.
# WOW HOOK   : "Few-shot deserves a few-shot architecture. Even at the
#              classical-ML floor, we train an LDA latent that maps code into
#              a space where each author IS their prototype, and the query is
#              just the nearest prototype."
# FALSIFIER  : (F1) LDA-prototype must beat raw-feature nearest-centroid by
#              >= 0.005 (else LDA adds nothing).
#              (F2) cosine vs euclidean on val picks the right metric.
#              (F3) the LDA gain must be larger at frac=0.01 than at
#              frac=0.20 (few-shot is where the latent matters).
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp151_protoshot")

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
# Prototype classifiers
# =============================================================================

def _standardise(X_train, X_others):
    mu = X_train.mean(axis=0, keepdims=True)
    sd = X_train.std(axis=0, keepdims=True) + 1e-8
    return (X_train - mu) / sd, [(X - mu) / sd for X in X_others]


def fit_prototypes(Z, y, n_cls):
    """Per-class mean prototype in latent space Z."""
    protos = np.zeros((n_cls, Z.shape[1]), dtype=np.float32)
    for c in range(n_cls):
        sel = (y == c)
        if sel.sum() == 0:
            protos[c] = 0.0
        else:
            protos[c] = Z[sel].mean(axis=0)
    return protos


def _l2_normalise(X, eps=1e-8):
    n = np.linalg.norm(X, axis=1, keepdims=True) + eps
    return X / n


def predict_nearest_centroid(Z_query, protos, metric="euclidean"):
    if metric == "cosine":
        Zn = _l2_normalise(Z_query); Pn = _l2_normalise(protos)
        # cosine sim -> we want argmax sim = argmin (1 - sim) i.e. argmin -sim
        sims = Zn @ Pn.T   # [N, n_cls]
        return np.argmax(sims, axis=1)
    # Euclidean
    # ||z - p||^2 = ||z||^2 - 2 z.p + ||p||^2
    p_norm = (protos ** 2).sum(axis=1)  # [n_cls]
    z_norm = (Z_query ** 2).sum(axis=1, keepdims=True)  # [N,1]
    cross  = Z_query @ protos.T  # [N, n_cls]
    d2 = z_norm - 2.0 * cross + p_norm[None, :]
    return np.argmin(d2, axis=1)


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
    lda_max_dim: int = 16


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
                f"test={len(test_codes)} workers={n_workers} "
                f"d_feat={N_FEATURES} n_cls={cfg.n_cls}")

    # Extract features
    t0 = time.time()
    X_train = extract_features_parallel(train_codes, n_workers=n_workers, desc="phi(train)")
    X_val   = extract_features_parallel(val_codes,   n_workers=n_workers, desc="phi(val)")
    X_test  = extract_features_parallel(test_codes,  n_workers=n_workers, desc="phi(test)")
    feat_time = time.time() - t0

    for X in (X_train, X_val, X_test):
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    # Standardise (helps both raw NC and LDA numerically)
    X_train_s, (X_val_s, X_test_s) = _standardise(X_train, [X_val, X_test])

    # (a) Raw nearest-centroid (control)
    raw_protos = fit_prototypes(X_train_s, train_labels, cfg.n_cls)
    raw_val_preds  = predict_nearest_centroid(X_val_s,  raw_protos, metric="euclidean")
    raw_test_preds = predict_nearest_centroid(X_test_s, raw_protos, metric="euclidean")
    raw_val_pack   = eval_pack(raw_val_preds, val_labels, val_langs, val_sources,
                               cfg.n_cls, sib_mask, dist_mat)
    raw_test_pack  = eval_pack(raw_test_preds, test_labels, test_langs, test_sources,
                               cfg.n_cls, sib_mask, dist_mat)
    raw_val_macro  = raw_val_pack["overall"]["macro_f1"]
    raw_test_macro = raw_test_pack["overall"]["macro_f1"]
    logger.info(f"[raw_nc] val={raw_val_macro:.4f} test={raw_test_macro:.4f}")

    # (b) LDA -> nearest centroid (euclidean)
    t0 = time.time()
    n_components = min(cfg.n_cls - 1, cfg.lda_max_dim, X_train_s.shape[1])
    lda = LinearDiscriminantAnalysis(n_components=n_components, solver="eigen", shrinkage="auto")
    lda.fit(X_train_s, train_labels)
    Z_train = lda.transform(X_train_s)
    Z_val   = lda.transform(X_val_s)
    Z_test  = lda.transform(X_test_s)
    lda_time = time.time() - t0

    lda_protos = fit_prototypes(Z_train, train_labels, cfg.n_cls)
    lda_eucl_val_preds  = predict_nearest_centroid(Z_val,  lda_protos, metric="euclidean")
    lda_eucl_test_preds = predict_nearest_centroid(Z_test, lda_protos, metric="euclidean")
    lda_eucl_val_pack   = eval_pack(lda_eucl_val_preds, val_labels, val_langs, val_sources,
                                    cfg.n_cls, sib_mask, dist_mat)
    lda_eucl_test_pack  = eval_pack(lda_eucl_test_preds, test_labels, test_langs, test_sources,
                                    cfg.n_cls, sib_mask, dist_mat)
    lda_eucl_val_macro  = lda_eucl_val_pack["overall"]["macro_f1"]
    lda_eucl_test_macro = lda_eucl_test_pack["overall"]["macro_f1"]
    logger.info(f"[lda_eucl] val={lda_eucl_val_macro:.4f} test={lda_eucl_test_macro:.4f}")

    # (c) LDA -> nearest centroid (cosine), prototypes computed on L2-norm latents
    Z_train_n = _l2_normalise(Z_train)
    cos_protos = fit_prototypes(Z_train_n, train_labels, cfg.n_cls)
    lda_cos_val_preds  = predict_nearest_centroid(Z_val,  cos_protos, metric="cosine")
    lda_cos_test_preds = predict_nearest_centroid(Z_test, cos_protos, metric="cosine")
    lda_cos_val_pack   = eval_pack(lda_cos_val_preds, val_labels, val_langs, val_sources,
                                   cfg.n_cls, sib_mask, dist_mat)
    lda_cos_test_pack  = eval_pack(lda_cos_test_preds, test_labels, test_langs, test_sources,
                                   cfg.n_cls, sib_mask, dist_mat)
    lda_cos_val_macro  = lda_cos_val_pack["overall"]["macro_f1"]
    lda_cos_test_macro = lda_cos_test_pack["overall"]["macro_f1"]
    logger.info(f"[lda_cos]  val={lda_cos_val_macro:.4f} test={lda_cos_test_macro:.4f}")

    # Pick best of (b) and (c) on VAL — that becomes the headline.
    if lda_cos_val_macro > lda_eucl_val_macro:
        chosen = "lda_cos"
        val_macro  = lda_cos_val_macro
        test_macro = lda_cos_test_macro
        ts_met     = lda_cos_test_pack
    else:
        chosen = "lda_eucl"
        val_macro  = lda_eucl_val_macro
        test_macro = lda_eucl_test_macro
        ts_met     = lda_eucl_test_pack

    gap = val_macro - test_macro
    logger.info(f"[headline={chosen}] val={val_macro:.4f} test={test_macro:.4f} gap={gap:+.4f}")

    # Falsifiers
    f1_lda_minus_raw = test_macro - raw_test_macro
    f2_cos_minus_eucl_val = lda_cos_val_macro - lda_eucl_val_macro  # val-set
    # F3 is computed across slots at the main() level; we expose lda_minus_raw delta
    # so main() can compare frac=0.01 vs frac=0.20.

    ts_met["falsifier_F1_lda_minus_raw_nc"]       = float(f1_lda_minus_raw)
    ts_met["falsifier_F2_cosine_minus_euclidean"] = float(f2_cos_minus_eucl_val)
    ts_met["raw_nc_val_macro"]   = float(raw_val_macro)
    ts_met["raw_nc_test_macro"]  = float(raw_test_macro)
    ts_met["lda_eucl_val_macro"]  = float(lda_eucl_val_macro)
    ts_met["lda_eucl_test_macro"] = float(lda_eucl_test_macro)
    ts_met["lda_cos_val_macro"]   = float(lda_cos_val_macro)
    ts_met["lda_cos_test_macro"]  = float(lda_cos_test_macro)
    ts_met["chosen_classifier"]   = chosen

    return {
        "tag": tag, "method": "PROTOSHOT",
        "note": ("Prototypical attribution in a supervised-learned LDA latent. "
                 "Few-shot architecture for code authorship, classical-ML floor, CPU-only."),
        "enc": f"stylometric-d{N_FEATURES}+LDA{n_components}", "bench": cfg.benchmark,
        "frac": cfg.frac, "n_features": int(N_FEATURES),
        "lda_n_components": int(n_components),
        "chosen_classifier": chosen,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "raw_nc_val_macro":   float(raw_val_macro),
        "raw_nc_test_macro":  float(raw_test_macro),
        "lda_eucl_val_macro":  float(lda_eucl_val_macro),
        "lda_eucl_test_macro": float(lda_eucl_test_macro),
        "lda_cos_val_macro":   float(lda_cos_val_macro),
        "lda_cos_test_macro":  float(lda_cos_test_macro),
        "falsifier_F1_lda_minus_raw_nc":       float(f1_lda_minus_raw),
        "falsifier_F2_cosine_minus_euclidean": float(f2_cos_minus_eucl_val),
        "feat_time_sec": feat_time, "lda_time_sec": lda_time,
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
            tag = f"exp151_protoshot_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} "
                            f"F1_lda-raw={res['falsifier_F1_lda_minus_raw_nc']:+.4f} "
                            f"time={res['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()

    # F3: compute LDA gain at frac=0.01 vs frac=0.20 per benchmark
    f3_per_bench = {}
    for bench, _, _ in benchmarks:
        g01 = next((r["falsifier_F1_lda_minus_raw_nc"] for r in results
                    if r["bench"] == bench and r["frac"] == 0.01), None)
        g20 = next((r["falsifier_F1_lda_minus_raw_nc"] for r in results
                    if r["bench"] == bench and r["frac"] == 0.20), None)
        if g01 is not None and g20 is not None:
            f3_per_bench[bench] = {
                "lda_gain_at_frac_0.01": float(g01),
                "lda_gain_at_frac_0.20": float(g20),
                "f3_low_minus_high":     float(g01 - g20),
            }
    # Stamp F3 onto every result for convenience
    for r in results:
        r["falsifier_F3_low_frac_gain"] = f3_per_bench.get(r["bench"], {})

    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp151_protoshot_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*160)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Train':>8} {'Test':>8} {'Val-F1':>8} "
          f"{'Test-F1':>8} {'Gap':>8} {'dPaper':>9} {'RawNC':>8} "
          f"{'LDAeu':>8} {'LDAco':>8} {'F1lda-raw':>11} {'Choice':>10} {'Wall':>8}")
    print("-"*160)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['train_size_after_cap']:>8d} {r['test_size_after_cap']:>8d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['raw_nc_test_macro']:>8.4f} "
              f"{r['lda_eucl_test_macro']:>8.4f} {r['lda_cos_test_macro']:>8.4f} "
              f"{r['falsifier_F1_lda_minus_raw_nc']:>+11.4f} {r['chosen_classifier']:>10} "
              f"{r['wall']:>8.0f}s")
    print("="*160)
    print(f"F3 (low_frac_gain): {json.dumps(f3_per_bench, indent=2)}")


if __name__ == "__main__":
    main()
