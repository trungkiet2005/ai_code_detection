# exp146 — SCATTERPLOT
# =============================================================================
# NAME       : SCATTERPLOT (Sliced-Wasserstein distance attribution between
#              per-sample feature distributions)
# REFERENCE  : Bonneel et al. 2015 ("Sliced and Radon Wasserstein
#              Barycenters of Measures", JMIV 51:22-45). Kolouri et al.
#              2019 ("Generalized Sliced Wasserstein Distances", NeurIPS).
#              First time Sliced-Wasserstein is used for AI code authorship.
# CLAIM      : An author's code corpus is a DISTRIBUTION over feature
#              vectors. Standard nearest-centroid attribution compares
#              MEANS; we compare DISTRIBUTIONS via Sliced-Wasserstein-2
#              distance (1D OT along random projections, averaged).
#              SW captures higher-order moments (variance, skew, multi-
#              modality) that the mean misses.
# EQUATION   : Let phi: code -> R^d be stylometric features (reuse from
#              exp145). Per-class empirical distribution: P_c = {phi(x_i) : y_i = c}.
#              For random unit direction theta in R^d:
#                proj_theta(P_c) = {<phi(x_i), theta> : y_i = c}    (1D distribution)
#                W_1d(proj_theta(P_c), proj_theta(P_test)) = 1D-OT via sorting
#              Sliced-Wasserstein:
#                SW(P_c, P_test) = (1/L) sum_{l=1}^L W_1d(proj_theta_l(P_c), proj_theta_l(P_test))
#              y_hat = argmin_c SW(P_c, {phi(query)})
#              (Singleton query distribution; compare to per-class
#              distribution of training samples.)
# WHY NEW    : Sliced-Wasserstein has been applied to generative model
#              comparison, image classification, point cloud matching.
#              Never to authorship attribution. The DISTRIBUTIONAL view
#              is what classical stylometry misses; SW is the principled
#              way to capture it without intractable d-dim OT.
# WOW HOOK   : "Authorship is distribution matching. Each author has a
#              distribution OVER stylometric features; we compare query to
#              each distribution via Sliced-Wasserstein — OT projected onto
#              random 1D slices. The author whose distribution best
#              transports to the query is the predicted author. Distribution
#              comparison replaces nearest-centroid."
# FALSIFIER  : (F1) SCATTERPLOT composite > nearest-centroid (RFOREST mean)
#              by >= 0.005 — distribution comparison adds info over mean.
#              (F2) Per-projection variance: 100 random projections have
#              non-trivial spread in 1D-W values (otherwise projections
#              collapse -> SW degenerates).
#              (F3) Number of projections L = 100 vs L = 25 give similar
#              ranking (Spearman >= 0.9 — Monte Carlo SW estimate stable).
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

_ensure("numpy"); _ensure("scipy"); _ensure("datasets"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
from scipy.stats import spearmanr
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp146_scatterplot")

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
# Data loading (identical plumbing to exp133_gzip)
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
# Stylometric feature extraction (copied from exp145 for self-containment)
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
            float(np.mean(indents)),
            float(max(indents)),
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
# Sliced-Wasserstein core
# =============================================================================

def sliced_wasserstein_1d(p_samples, q_samples):
    """1D Wasserstein-1 between two sets of scalars (quantile-matched)."""
    n_p = len(p_samples); n_q = len(q_samples)
    if n_p == 0 or n_q == 0:
        return float("inf")
    p_sorted = np.sort(p_samples)
    q_sorted = np.sort(q_samples)
    n_grid = max(n_p, n_q, 50)
    qs = np.linspace(0.0, 1.0, n_grid + 2)[1:-1]
    p_qq = np.quantile(p_sorted, qs)
    q_qq = np.quantile(q_sorted, qs)
    return float(np.mean(np.abs(p_qq - q_qq)))


def _random_projection_directions(d, L, seed=42):
    rng = np.random.default_rng(seed)
    thetas = rng.standard_normal((L, d)).astype(np.float32)
    norms = np.linalg.norm(thetas, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return thetas / norms


def sliced_wasserstein(P, Q, thetas, return_per_proj=False):
    """SW distance between two sets of d-dim feature vectors using fixed thetas.

    P: (n_P, d) — training samples of class c
    Q: (n_Q, d) — query samples (typically n_Q=1)
    thetas: (L, d) — pre-computed unit projection directions
    """
    P_proj = P @ thetas.T  # (n_P, L)
    Q_proj = Q @ thetas.T  # (n_Q, L)
    L = thetas.shape[0]
    per_proj = np.zeros(L, dtype=np.float32)
    for l in range(L):
        per_proj[l] = sliced_wasserstein_1d(P_proj[:, l], Q_proj[:, l])
    if return_per_proj:
        return float(per_proj.mean()), per_proj
    return float(per_proj.mean())


# Worker globals (set per-process by _sw_worker_init)
_W = {"class_proj": None, "L": None}


def _sw_worker_init(class_proj_list, L):
    """Class projections: list of (n_c, L) arrays, one per class."""
    _W["class_proj"] = class_proj_list
    _W["L"] = L


def _sw_one_query(args):
    """args = (qi, q_proj_row) where q_proj_row is (L,) — query projected onto all L thetas."""
    qi, q_proj_row = args
    class_proj = _W["class_proj"]
    L = _W["L"]
    K = len(class_proj)
    sw_dists = np.zeros(K, dtype=np.float32)
    # For each class, compute mean over L of |sorted(class_proj[:, l]) - q_proj_row[l]|.
    # Singleton query: 1D-W between a set and a single point is mean_i |p_i - q|.
    for c in range(K):
        cp = class_proj[c]  # (n_c, L)
        # Mean over training samples per slice then mean over slices.
        # |x - q|: shape (n_c, L)
        diffs = np.abs(cp - q_proj_row[None, :])
        # 1D-W between empirical distribution {cp[:, l]} and singleton {q_proj_row[l]}
        # equals mean_i |cp[i, l] - q_proj_row[l]|.
        sw_dists[c] = float(diffs.mean())
    return qi, sw_dists


def attribute_sw_batch(query_features, class_train_features, L=100, seed=42,
                       n_workers=None, return_per_proj_var=False):
    """For each query, compute SW to each class distribution; argmin = prediction.

    query_features: (N, d)
    class_train_features: list of K arrays, each (n_c, d)
    Returns: predictions (N,), and (optionally) per-projection variance for F2.
    """
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    N, d = query_features.shape
    K = len(class_train_features)
    thetas = _random_projection_directions(d, L, seed=seed)

    # Pre-project everything once.
    class_proj = [P @ thetas.T for P in class_train_features]  # list of (n_c, L)
    Q_proj = query_features @ thetas.T  # (N, L)

    # F2: per-projection variance across classes (average over classes of the variance of
    # projected values within a slice). If projections collapse to constants, variance is 0.
    proj_variances = np.array([float(np.var(cp, axis=0).mean()) for cp in class_proj],
                              dtype=np.float32)
    mean_proj_var = float(proj_variances.mean())

    # Per-query attribution. We avoid multiprocessing overhead for small N.
    preds = np.zeros(N, dtype=np.int64)
    args_list = [(i, Q_proj[i]) for i in range(N)]
    if n_workers == 1 or N < 64:
        _sw_worker_init(class_proj, L)
        for args in tqdm(args_list, desc="SW-attribute"):
            qi, sw_dists = _sw_one_query(args)
            preds[qi] = int(np.argmin(sw_dists))
    else:
        try:
            with mp.Pool(n_workers, initializer=_sw_worker_init,
                         initargs=(class_proj, L)) as pool:
                for qi, sw_dists in tqdm(pool.imap_unordered(_sw_one_query, args_list,
                                                            chunksize=8),
                                          total=N, desc="SW-attribute"):
                    preds[qi] = int(np.argmin(sw_dists))
        except Exception as e:
            logger.warning(f"[sw] multiprocessing failed ({e}); falling back to serial")
            _sw_worker_init(class_proj, L)
            for args in tqdm(args_list, desc="SW-attribute"):
                qi, sw_dists = _sw_one_query(args)
                preds[qi] = int(np.argmin(sw_dists))

    info = {"mean_proj_variance": mean_proj_var,
            "per_class_proj_variance": proj_variances.tolist()}
    return preds, info


# =============================================================================
# Nearest-centroid (control)
# =============================================================================

def attribute_nearest_centroid(query_features, class_train_features):
    """Baseline: predict argmin_c ||q - mean(P_c)||_2."""
    N, d = query_features.shape
    K = len(class_train_features)
    centroids = np.stack([P.mean(axis=0) for P in class_train_features], axis=0)  # (K, d)
    # Standardise to avoid feature-scale dominating: use raw L2 on raw features (consistent
    # with SW which is also scale-sensitive — fair comparison).
    diffs = query_features[:, None, :] - centroids[None, :, :]  # (N, K, d)
    d2 = np.sum(diffs * diffs, axis=-1)  # (N, K)
    return np.argmin(d2, axis=1).astype(np.int64)


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
    L_projections: int = 100
    L_projections_small: int = 25  # for F3 stability check
    train_cap_per_class: int = 400
    test_cap_per_class: int = 500
    n_workers: int = -1
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

def _per_query_sw_scores(query_features, class_train_features, L, seed):
    """Return (N, K) matrix of SW distances (for ranking stability tests)."""
    N, d = query_features.shape
    K = len(class_train_features)
    thetas = _random_projection_directions(d, L, seed=seed)
    class_proj = [P @ thetas.T for P in class_train_features]
    Q_proj = query_features @ thetas.T  # (N, L)
    out = np.zeros((N, K), dtype=np.float32)
    for c in range(K):
        cp = class_proj[c]  # (n_c, L)
        # mean over training samples then mean over slices for singleton query
        # for each query: mean_l mean_i |cp[i, l] - Q_proj[qi, l]|
        # = mean_l (mean_i |cp[i, l]| centred at Q_proj[qi, l])
        # Compute per query in a vectorised loop over l for memory.
        acc = np.zeros(N, dtype=np.float32)
        for l in range(L):
            # |cp[:, l] - Q_proj[:, l, None]| -> (N, n_c)
            acc += np.mean(np.abs(cp[:, l][None, :] - Q_proj[:, l][:, None]), axis=1)
        out[:, c] = acc / L
    return out


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
                f"test={len(test_codes)} L={cfg.L_projections} workers={n_workers} "
                f"d_feat={N_FEATURES}")

    # Features
    t0 = time.time()
    X_train = extract_features_parallel(train_codes, n_workers=n_workers, desc="phi(train)")
    X_val   = extract_features_parallel(val_codes,   n_workers=n_workers, desc="phi(val)")
    X_test  = extract_features_parallel(test_codes,  n_workers=n_workers, desc="phi(test)")
    for X in (X_train, X_val, X_test):
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    feat_time = time.time() - t0

    # Standardise per-feature using training mean/std (zero variance -> 1).
    mu = X_train.mean(axis=0); sd = X_train.std(axis=0); sd[sd == 0] = 1.0
    X_train_z = (X_train - mu) / sd
    X_val_z   = (X_val   - mu) / sd
    X_test_z  = (X_test  - mu) / sd

    # Group training features by class
    class_train_features = []
    for c in range(cfg.n_cls):
        sel = (train_labels == c)
        if sel.sum() == 0:
            # fall back to all-train to avoid empty class (rare).
            class_train_features.append(X_train_z)
        else:
            class_train_features.append(X_train_z[sel])
    logger.info(f"[setup] per-class train sizes: " +
                ", ".join(str(len(c)) for c in class_train_features))

    # Val pass (SW)
    t0 = time.time()
    val_preds, val_info = attribute_sw_batch(
        X_val_z, class_train_features, L=cfg.L_projections,
        seed=cfg.seed, n_workers=n_workers,
    )
    val_time = time.time() - t0
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                        cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]
    logger.info(f"[val] macro={val_macro:.4f} time={val_time:.0f}s "
                f"proj_var={val_info['mean_proj_variance']:.4f}")

    # Test pass (SW, L=100)
    t0 = time.time()
    test_preds, test_info = attribute_sw_batch(
        X_test_z, class_train_features, L=cfg.L_projections,
        seed=cfg.seed, n_workers=n_workers,
    )
    test_time = time.time() - t0
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro
    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} time={test_time:.0f}s")

    # Control: nearest-centroid (F1)
    nc_preds = attribute_nearest_centroid(X_test_z, class_train_features)
    nc_macro = float(f1_score(test_labels, nc_preds, average="macro", zero_division=0))
    logger.info(f"[F1] nearest-centroid macro={nc_macro:.4f} (sw - nc = {test_macro - nc_macro:+.4f})")

    # F3: stability — Spearman of class-distance ranking at L=100 vs L=25
    # Re-compute the SW distance matrices using a tractable subsample of the test set.
    t0 = time.time()
    n_stab = min(200, X_test_z.shape[0])
    rng = np.random.default_rng(cfg.seed)
    stab_idx = rng.choice(X_test_z.shape[0], n_stab, replace=False)
    X_stab = X_test_z[stab_idx]
    dist_L_big   = _per_query_sw_scores(X_stab, class_train_features,
                                        L=cfg.L_projections,       seed=cfg.seed)
    dist_L_small = _per_query_sw_scores(X_stab, class_train_features,
                                        L=cfg.L_projections_small, seed=cfg.seed + 7)
    # Per-query Spearman across classes, averaged.
    spearmans = []
    for i in range(n_stab):
        a = dist_L_big[i]; b = dist_L_small[i]
        if np.std(a) > 0 and np.std(b) > 0:
            rho, _ = spearmanr(a, b)
            if not np.isnan(rho):
                spearmans.append(float(rho))
    spearman_mean = float(np.mean(spearmans)) if spearmans else 0.0
    stab_time = time.time() - t0
    logger.info(f"[F3] Spearman(L=100 vs L=25) over {n_stab} queries = {spearman_mean:.4f} "
                f"time={stab_time:.0f}s")

    ts_met["falsifier_F1_nearest_centroid_macro"]  = nc_macro
    ts_met["falsifier_F1_sw_minus_nc"]             = test_macro - nc_macro
    ts_met["falsifier_F2_mean_proj_variance"]      = test_info["mean_proj_variance"]
    ts_met["falsifier_F2_per_class_proj_variance"] = test_info["per_class_proj_variance"]
    ts_met["falsifier_F3_spearman_L100_vs_L25"]    = spearman_mean
    ts_met["falsifier_F3_n_queries_used"]          = int(n_stab)

    return {
        "tag": tag, "method": "SCATTERPLOT",
        "note": ("Sliced-Wasserstein attribution between per-class distributions "
                 "of stylometric features. CPU-only; no neural encoder. "
                 "Inspired by Bonneel 2015 and Kolouri 2019."),
        "enc": f"sw-L{cfg.L_projections}-d{N_FEATURES}", "bench": cfg.benchmark,
        "frac": cfg.frac, "L_projections": cfg.L_projections,
        "n_features": int(N_FEATURES),
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "nearest_mean_macro": nc_macro,
        "sw_macro": test_macro,
        "sw_minus_nc": test_macro - nc_macro,
        "projection_variance_at_L100": test_info["mean_proj_variance"],
        "spearman_L100_vs_L25": spearman_mean,
        "feat_time_sec": feat_time, "val_time_sec": val_time, "test_time_sec": test_time,
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
            tag = f"exp146_scatterplot_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"sw-nc={res['sw_minus_nc']:+.4f} "
                            f"spearman={res['spearman_L100_vs_L25']:.3f}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp146_scatterplot_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Benchmark':<12} {'Frac':>6} {'L':>5} {'Train':>8} {'Test':>8} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'NC-F1':>8} {'SW-NC':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'ProjVar':>10} {'Spearman':>10} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['L_projections']:>5d} "
              f"{r['train_size_after_cap']:>8d} {r['test_size_after_cap']:>8d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['nearest_mean_macro']:>8.4f} "
              f"{r['sw_minus_nc']:>+8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['projection_variance_at_L100']:>10.4f} "
              f"{r['spearman_L100_vs_L25']:>10.4f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
