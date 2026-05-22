# exp154 — NAIVEBAYES
# =============================================================================
# NAME       : NAIVEBAYES (Multinomial + Complement NB on hashed char n-grams)
# REFERENCE  : Manning, Raghavan & Schuetze 2008 ("Introduction to
#              Information Retrieval"), chapter 13. Rennie et al. 2003
#              ("Tackling the Poor Assumptions of Naive Bayes Text
#              Classifiers", ICML) — ComplementNB. McCallum & Nigam 1998
#              (NB for text classification).
# CLAIM      : The classical textbook baseline for text classification has
#              never been reported on AI-code authorship attribution.
#              Hash character (1,2,3)-grams of code into a 2^16 sparse
#              feature vector; fit Multinomial NB and Complement NB; pick
#              the better one on val. This is what every IR textbook says
#              to try first.
# EQUATION   : phi(code) = hash(char_n_grams(code, n in {1,2,3}))   # counts
#              Multinomial NB: p(c | x) propto p(c) * prod_t p(t | c)^{x_t}
#              Complement NB: p(c | x) propto p(c) * prod_t (1 - p(t | not_c))
#              y_hat = argmax_c log p(c | x)
# WHY NEW    : Every published baseline on CoDET-M4 / AICD / Droid uses a
#              transformer encoder, supervised contrastive, or stylometric
#              random forest. Multinomial NB on character n-grams — THE
#              textbook baseline since 1998 — has never been reported.
#              We close the gap.
# WOW HOOK   : "Before there were transformers, there was Naive Bayes on
#              character n-grams. We close the gap the field has been
#              ignoring and report what the simplest possible baseline
#              scores on a benchmark dominated by neural methods."
# FALSIFIER  : (F1) ComplementNB should help on AICD (12 classes, imbalanced)
#              more than CoDET (6 classes). Report
#              falsifier_F1_complement_vs_multinomial per benchmark.
#              (F2) Sweep alpha in {0.01, 0.1, 1.0, 10.0} on val;
#              alpha=1.0 (Laplace) must NOT be best at every slot.
#              (F3) Sweep ngram_range in {(1,1), (1,2), (1,3), (2,4)} on val;
#              chosen range must be picked on val, not test.
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
from datasets import load_dataset
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp154_naivebayes")

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
# Stylometric features (kept for parity even though NB uses char-ngrams)
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
    alpha: float = 1.0
    ngram_range: tuple = (1, 3)
    n_features_hash: int = 1 << 16
    train_cap_per_class: int = 300
    test_cap_per_class: int = 400
    val_cap_per_class: int = 200
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = 4000
    alpha_sweep: tuple = (0.01, 0.1, 1.0, 10.0)
    ngram_sweep: tuple = ((1, 1), (1, 2), (1, 3), (2, 4))


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
# NB train helpers
# =============================================================================

def _make_vec(ngram_range, n_features_hash):
    return HashingVectorizer(
        analyzer="char", ngram_range=tuple(ngram_range),
        n_features=n_features_hash, alternate_sign=False, norm=None,
    )


def train_nb(X_sparse, y, kind="multinomial", alpha=1.0):
    if kind == "complement":
        clf = ComplementNB(alpha=alpha)
    else:
        clf = MultinomialNB(alpha=alpha)
    clf.fit(X_sparse, y)
    return clf


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
                f"test={len(test_codes)} alpha={cfg.alpha} ngram={cfg.ngram_range} "
                f"n_hash={cfg.n_features_hash}")

    # F3: ngram-range sweep on val (fix alpha = default 1.0, kind = multinomial),
    # then F2: alpha sweep with chosen ngram_range. The chosen (kind, alpha, ngram)
    # combo is picked from val.
    ngram_sweep_val = {}
    for ng in cfg.ngram_sweep:
        vec = _make_vec(ng, cfg.n_features_hash)
        t0 = time.time()
        Xtr = vec.transform(train_codes); Xvl = vec.transform(val_codes)
        clf = train_nb(Xtr, train_labels, kind="multinomial", alpha=1.0)
        vp = clf.predict(Xvl)
        m = float(f1_score(val_labels, vp, average="macro", zero_division=0))
        ngram_sweep_val[str(ng)] = m
        logger.info(f"[F3] ngram={ng} val_macro={m:.4f} t={time.time()-t0:.1f}s")
    best_ngram_key = max(ngram_sweep_val.items(), key=lambda kv: kv[1])[0]
    # parse back tuple "(1, 3)" -> (1, 3)
    inner = best_ngram_key.strip("()").split(",")
    best_ngram = (int(inner[0]), int(inner[1]))
    logger.info(f"[select] best_ngram={best_ngram}")

    # F2: alpha sweep + F1: ComplementNB vs MultinomialNB with the chosen ngram.
    vec_best = _make_vec(best_ngram, cfg.n_features_hash)
    Xtr = vec_best.transform(train_codes)
    Xvl = vec_best.transform(val_codes)
    Xts = vec_best.transform(test_codes)
    logger.info(f"[vec] shapes: train={Xtr.shape} val={Xvl.shape} test={Xts.shape} "
                f"nnz_train={Xtr.nnz}")

    alpha_sweep_val_mnb = {}
    alpha_sweep_val_cnb = {}
    for a in cfg.alpha_sweep:
        clf_m = train_nb(Xtr, train_labels, kind="multinomial", alpha=a)
        clf_c = train_nb(Xtr, train_labels, kind="complement",  alpha=a)
        vm = float(f1_score(val_labels, clf_m.predict(Xvl), average="macro", zero_division=0))
        vc = float(f1_score(val_labels, clf_c.predict(Xvl), average="macro", zero_division=0))
        alpha_sweep_val_mnb[str(a)] = vm
        alpha_sweep_val_cnb[str(a)] = vc
        logger.info(f"[F2] alpha={a} mnb_val={vm:.4f} cnb_val={vc:.4f}")

    # Pick best (kind, alpha) on val
    best_mnb_alpha = max(alpha_sweep_val_mnb.items(), key=lambda kv: kv[1])
    best_cnb_alpha = max(alpha_sweep_val_cnb.items(), key=lambda kv: kv[1])
    if best_cnb_alpha[1] >= best_mnb_alpha[1]:
        best_kind = "complement"; best_alpha = float(best_cnb_alpha[0])
        best_val_macro_picker = best_cnb_alpha[1]
    else:
        best_kind = "multinomial"; best_alpha = float(best_mnb_alpha[0])
        best_val_macro_picker = best_mnb_alpha[1]
    logger.info(f"[select] best_kind={best_kind} best_alpha={best_alpha} "
                f"val={best_val_macro_picker:.4f}")

    t0 = time.time()
    clf = train_nb(Xtr, train_labels, kind=best_kind, alpha=best_alpha)
    train_time = time.time() - t0

    val_preds = clf.predict(Xvl)
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                        cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]
    logger.info(f"[val] macro={val_macro:.4f}")

    test_preds = clf.predict(Xts)
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro
    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f}")

    # F1: report MNB-test and CNB-test side by side at best_alpha.
    clf_m_only = train_nb(Xtr, train_labels, kind="multinomial", alpha=best_alpha)
    clf_c_only = train_nb(Xtr, train_labels, kind="complement",  alpha=best_alpha)
    mnb_test = float(f1_score(test_labels, clf_m_only.predict(Xts), average="macro", zero_division=0))
    cnb_test = float(f1_score(test_labels, clf_c_only.predict(Xts), average="macro", zero_division=0))
    f1_delta = abs(cnb_test - mnb_test)
    logger.info(f"[F1] mnb_test={mnb_test:.4f} cnb_test={cnb_test:.4f} |delta|={f1_delta:.4f}")

    ts_met["falsifier_F1_complement_vs_multinomial"] = float(f1_delta)
    ts_met["falsifier_F1_mnb_test_macro"]            = float(mnb_test)
    ts_met["falsifier_F1_cnb_test_macro"]            = float(cnb_test)
    ts_met["falsifier_F2_alpha_sweep_val_macros_mnb"] = alpha_sweep_val_mnb
    ts_met["falsifier_F2_alpha_sweep_val_macros_cnb"] = alpha_sweep_val_cnb
    ts_met["falsifier_F3_ngram_sweep_val_macros"]    = ngram_sweep_val
    ts_met["best_kind_from_val"]                     = best_kind
    ts_met["best_alpha_from_val"]                    = float(best_alpha)
    ts_met["best_ngram_from_val"]                    = list(best_ngram)

    return {
        "tag": tag, "method": "NAIVEBAYES",
        "note": ("Multinomial + Complement Naive Bayes on hashed character "
                 "n-grams. CPU-only; sklearn HashingVectorizer + MNB/CNB."),
        "enc": f"hash_char_ngrams_n{cfg.n_features_hash}", "bench": cfg.benchmark,
        "frac": cfg.frac, "alpha": float(best_alpha),
        "kind": best_kind, "ngram_range": list(best_ngram),
        "n_features_hash": int(cfg.n_features_hash),
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "mnb_test_macro": mnb_test, "cnb_test_macro": cnb_test,
        "complement_vs_multinomial_delta": float(f1_delta),
        "alpha_sweep_val_mnb": alpha_sweep_val_mnb,
        "alpha_sweep_val_cnb": alpha_sweep_val_cnb,
        "ngram_sweep_val": ngram_sweep_val,
        "train_time_sec": train_time,
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
            tag = f"exp154_naivebayes_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"kind={res['kind']} alpha={res['alpha']} "
                            f"ngram={res['ngram_range']}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp154_naivebayes_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Kind':>11} {'Alpha':>7} {'NGram':>8} "
          f"{'Train':>7} {'Test':>7} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'C-M':>7} {'Wall':>8}")
    print("-"*140)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['kind']:>11} "
              f"{r['alpha']:>7.2f} {str(r['ngram_range']):>8} "
              f"{r['train_size_after_cap']:>7d} {r['test_size_after_cap']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['complement_vs_multinomial_delta']:>+7.4f} "
              f"{r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
