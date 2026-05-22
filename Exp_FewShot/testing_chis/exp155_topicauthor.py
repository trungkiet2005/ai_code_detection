# exp155 — TOPICAUTHOR
# =============================================================================
# NAME       : TOPICAUTHOR (LDA topic profile per author + KL attribution)
# REFERENCE  : Blei, Ng & Jordan 2003 ("Latent Dirichlet Allocation", JMLR).
#              Seroussi, Zukerman & Bohnert 2014 ("Authorship Attribution
#              with Topic Models", Computational Linguistics) — applied
#              LDA to natural-language authorship, NEVER to code.
#              Hoffman et al. 2010 (online VB for LDA, ICML).
# CLAIM      : Every author can be represented as a probability distribution
#              over K latent topics learned from the training corpus. Fit
#              ONE global LDA over hashed (1,2)-char-grams, project each
#              training sample to its K-simplex topic vector, then average
#              over each author's training samples to get a K-simplex
#              "topic profile" per author. At test time, project the query,
#              and attribute via minimum KL divergence to author profiles.
# EQUATION   : LDA: x | topic ~ Cat(beta_topic), topic | author ~ Dir(alpha)
#              theta(x) = E_q[topic | x]            (variational posterior)
#              theta(c) = (1/|D_c|) sum_{x in D_c} theta(x)
#              y_hat = argmin_c KL(theta(query) || theta(c))
# WHY NEW    : LDA topic models (Blei 2003) are foundational for unsupervised
#              text analysis. They have been applied to natural-language
#              authorship attribution (Seroussi 2014) but NEVER to AI-code
#              authorship. The "author = mean topic distribution" formulation
#              gives a probabilistic AND interpretable author signature.
# WOW HOOK   : "Every author is a probability distribution over latent
#              topics. We learn 16 topics over the training corpus, project
#              each author onto a 16-simplex, and attribute by KL divergence.
#              Authorship is a Dirichlet random variable."
# FALSIFIER  : (F1) Sweep K in {8, 16, 32} on val. The best K must NOT be
#              the largest (else LDA isn't compressing).
#              (F2) KL between topic distributions must beat L2 between
#              topic distributions by >= 0.003 macro-F1 (else KL is
#              decorative).
#              (F3) Topic-entropy per author H(theta(c)) — if all entropies
#              are within 0.05 of log(K) (uniform), profiles aren't
#              distinguishing authors.
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, re, math
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
from datasets import load_dataset
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp155_topicauthor")

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
# Stylometric features (kept for parity, unused by LDA path)
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
# Cfg
# =============================================================================

@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    frac: float = 0.20
    n_cls: int = 6
    seed: int = 42
    n_topics: int = 16
    n_features_hash: int = 4096
    ngram_range: tuple = (1, 2)
    lda_max_iter: int = 10
    train_cap_per_class: int = 300
    test_cap_per_class: int = 400
    val_cap_per_class: int = 200
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = 4000
    k_sweep: tuple = (8, 16, 32)


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
# LDA + KL helpers
# =============================================================================

_EPS = 1e-9


def _normalize_simplex(P):
    P = np.clip(P, _EPS, None)
    P = P / P.sum(axis=1, keepdims=True)
    return P


def _author_profiles(theta_train, labels, n_cls):
    K = theta_train.shape[1]
    prof = np.full((n_cls, K), 1.0 / K, dtype=np.float64)
    for c in range(n_cls):
        sel = (labels == c)
        if sel.sum() > 0:
            prof[c] = theta_train[sel].mean(axis=0)
    return _normalize_simplex(prof)


def _kl_attribute(theta_query, profiles):
    # theta_query: (n_query, K); profiles: (n_cls, K). Both rows on K-simplex.
    # KL(q || p) = sum q * (log q - log p). Lower is better.
    K = theta_query.shape[1]
    q = np.clip(theta_query, _EPS, None)
    p = np.clip(profiles, _EPS, None)
    log_q = np.log(q); log_p = np.log(p)
    # divergences[i, c] = sum_k q[i,k] * (log_q[i,k] - log_p[c,k])
    # = sum_k q[i,k] log_q[i,k] - q[i] @ log_p[c].T
    H_q = (q * log_q).sum(axis=1, keepdims=True)            # (n_query, 1)
    cross = q @ log_p.T                                       # (n_query, n_cls)
    div = H_q - cross
    return np.argmin(div, axis=1), div


def _l2_attribute(theta_query, profiles):
    # squared L2 between simplex points
    diff = theta_query[:, None, :] - profiles[None, :, :]
    div = (diff * diff).sum(axis=2)
    return np.argmin(div, axis=1), div


def _topic_entropy_per_class(profiles):
    p = np.clip(profiles, _EPS, None)
    return (-p * np.log(p)).sum(axis=1)


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

    logger.info(f"[setup] frac={cfg.frac} train={len(train_codes)} val={len(val_codes)} "
                f"test={len(test_codes)} n_topics_default={cfg.n_topics} "
                f"n_hash={cfg.n_features_hash} ngram={cfg.ngram_range}")

    # Build one hashing vectorizer for all K (n_features_hash is fixed)
    vec = HashingVectorizer(
        analyzer="char", ngram_range=tuple(cfg.ngram_range),
        n_features=cfg.n_features_hash, alternate_sign=False, norm=None,
    )
    t0 = time.time()
    Xtr = vec.transform(train_codes)
    Xvl = vec.transform(val_codes)
    Xts = vec.transform(test_codes)
    vec_time = time.time() - t0
    logger.info(f"[vec] shapes: train={Xtr.shape} val={Xvl.shape} test={Xts.shape} "
                f"nnz_train={Xtr.nnz} t={vec_time:.1f}s")

    # F1: K-sweep on val (KL attribution)
    k_sweep_val = {}
    k_sweep_models = {}
    for K in cfg.k_sweep:
        t0 = time.time()
        lda = LatentDirichletAllocation(
            n_components=K, learning_method="online", max_iter=cfg.lda_max_iter,
            batch_size=128, random_state=cfg.seed, n_jobs=1,
        )
        lda.fit(Xtr)
        theta_tr = _normalize_simplex(lda.transform(Xtr))
        theta_vl = _normalize_simplex(lda.transform(Xvl))
        prof = _author_profiles(theta_tr, train_labels, cfg.n_cls)
        vp, _ = _kl_attribute(theta_vl, prof)
        m = float(f1_score(val_labels, vp, average="macro", zero_division=0))
        k_sweep_val[str(K)] = m
        k_sweep_models[K] = (lda, prof, theta_tr)
        logger.info(f"[F1] K={K} val_macro={m:.4f} t={time.time()-t0:.1f}s")

    best_K = max(k_sweep_val.items(), key=lambda kv: kv[1])[0]
    best_K_i = int(best_K)
    largest_K = max(int(k) for k in k_sweep_val.keys())
    f1_best_is_largest = (best_K_i == largest_K)
    logger.info(f"[select] best_K={best_K_i} (largest_K={largest_K}; "
                f"best_is_largest={f1_best_is_largest})")

    lda, prof_train, theta_tr = k_sweep_models[best_K_i]

    # Final transform of val + test at best K
    theta_vl = _normalize_simplex(lda.transform(Xvl))
    theta_ts = _normalize_simplex(lda.transform(Xts))

    # Headline: KL attribution at best K
    val_preds, _ = _kl_attribute(theta_vl, prof_train)
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                        cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]
    logger.info(f"[val KL] macro={val_macro:.4f}")

    test_preds, _ = _kl_attribute(theta_ts, prof_train)
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro
    logger.info(f"[test KL] macro={test_macro:.4f} gap={gap:+.4f}")

    # F2: L2 attribution at the same K (on test) for comparison
    test_preds_l2, _ = _l2_attribute(theta_ts, prof_train)
    test_macro_l2 = float(f1_score(test_labels, test_preds_l2,
                                   average="macro", zero_division=0))
    kl_minus_l2 = test_macro - test_macro_l2
    logger.info(f"[F2] KL test={test_macro:.4f} L2 test={test_macro_l2:.4f} "
                f"KL-L2={kl_minus_l2:+.4f}")

    # F3: topic entropy per author
    H_per_class = _topic_entropy_per_class(prof_train).tolist()
    H_uniform = math.log(best_K_i)
    H_gap = [H_uniform - h for h in H_per_class]
    max_gap_to_uniform = float(max(H_gap)) if H_gap else 0.0
    all_within_005 = bool(all(abs(g) <= 0.05 for g in H_gap))
    logger.info(f"[F3] H_uniform={H_uniform:.4f} per_class_H="
                f"{['%.3f' % h for h in H_per_class]} "
                f"max_gap_to_uniform={max_gap_to_uniform:.4f}")

    ts_met["falsifier_F1_k_sweep_val_macros"]   = k_sweep_val
    ts_met["falsifier_F1_best_K_from_val"]      = int(best_K_i)
    ts_met["falsifier_F1_best_K_is_largest"]    = bool(f1_best_is_largest)
    ts_met["falsifier_F2_kl_test_macro"]        = float(test_macro)
    ts_met["falsifier_F2_l2_test_macro"]        = float(test_macro_l2)
    ts_met["falsifier_F2_kl_minus_l2_macro"]    = float(kl_minus_l2)
    ts_met["falsifier_F3_topic_entropy_per_author"] = H_per_class
    ts_met["falsifier_F3_log_K_uniform_entropy"]    = float(H_uniform)
    ts_met["falsifier_F3_max_gap_to_uniform"]       = float(max_gap_to_uniform)
    ts_met["falsifier_F3_all_within_0p05_of_uniform"] = bool(all_within_005)

    return {
        "tag": tag, "method": "TOPICAUTHOR",
        "note": ("LDA topic model on hashed (1,2)-char-grams; per-author mean "
                 "topic profile on K-simplex; KL-divergence attribution."),
        "enc": f"lda_K{best_K_i}_hash{cfg.n_features_hash}",
        "bench": cfg.benchmark, "frac": cfg.frac,
        "n_topics": int(best_K_i),
        "n_features_hash": int(cfg.n_features_hash),
        "ngram_range": list(cfg.ngram_range),
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "kl_test_macro": float(test_macro),
        "l2_test_macro": float(test_macro_l2),
        "kl_minus_l2_macro": float(kl_minus_l2),
        "k_sweep_val_macros": k_sweep_val,
        "best_K_is_largest": bool(f1_best_is_largest),
        "topic_entropy_per_author": H_per_class,
        "log_K_uniform_entropy": float(H_uniform),
        "all_within_0p05_of_uniform": bool(all_within_005),
        "vec_time_sec": vec_time,
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
            tag = f"exp155_topicauthor_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"K={res['n_topics']} KL-L2={res['kl_minus_l2_macro']:+.4f}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp155_topicauthor_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'K':>4} {'Train':>7} {'Test':>7} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'L2-F1':>8} {'KL-L2':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'Wall':>8}")
    print("-"*140)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['n_topics']:>4d} "
              f"{r['train_size_after_cap']:>7d} {r['test_size_after_cap']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['l2_test_macro']:>8.4f} {r['kl_minus_l2_macro']:>+8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} {r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
