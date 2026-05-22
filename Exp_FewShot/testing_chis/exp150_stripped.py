# exp150 — STRIPPED
# =============================================================================
# NAME       : STRIPPED (Comment/String/Import-Stripped Stylometric Attribution).
# REFERENCE  : Caliskan et al. 2015 (stylometric attribution, USENIX Security).
#              Breiman 2001 (Random Forests). This is a new ablation; no prior
#              code-attribution paper has audited comment/string/import leakage
#              on CoDET-M4 / AICD.
# CLAIM      : Strip every comment, every string literal, every import/include
#              statement, then collapse repeated whitespace, BEFORE computing
#              the 57-d stylometric features used by RFOREST (exp145). The
#              delta in Macro-F1 between full-strip and no-strip is the upper
#              bound on how much of the model's accuracy was riding on
#              non-code-style leakage (LLM disclaimer comments, signature
#              strings, repeated header boilerplate).
# EQUATION   : strip(code) = code with comments + strings + imports + ws collapsed
#              phi: stripped_code -> R^57   (stylometric features)
#              y_hat = RandomForest(phi(strip(code, mode='full')))
# WHY NEW    : ext_gptsniffer leads the leaderboard at 0.5421 but no published
#              baseline on CoDET-M4 / AICD has audited whether the signal is
#              real code structure or comment/string boilerplate. This is the
#              first stripping ablation on this benchmark.
# WOW HOOK   : "We strip every comment, string, and import from each sample
#              before features are computed. If accuracy survives, authorship
#              is real. If it collapses, the field has been measuring comment
#              style, not code style."
# FALSIFIER  : (F1) full-strip drop >= 0.05 vs no-strip means the model was
#              relying on comment/string leakage.
#              (F2) comments-only minus full-strip isolates the string+import
#              contribution.
#              (F3) per-language drop on CoDET-M4 (Python is comment-heavy):
#              does the drop concentrate in comment-heavy languages?
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
logger = logging.getLogger("exp150_stripped")

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
# Strip routines — the new object
# =============================================================================

_RE_PY_COMMENT  = re.compile(r"#[^\n]*")
_RE_C_LINE      = re.compile(r"//[^\n]*")
_RE_C_BLOCK     = re.compile(r"/\*[\s\S]*?\*/")
_RE_TRIPLE_DQ   = re.compile(r'"""[\s\S]*?"""')
_RE_TRIPLE_SQ   = re.compile(r"'''[\s\S]*?'''")
_RE_STRING_DQ   = re.compile(r'"[^"\n]*"')
_RE_STRING_SQ   = re.compile(r"'[^'\n]*'")
_RE_MULTI_NL    = re.compile(r"\n{3,}")

# First-token check for import-like lines (handles Python/JS/Java/C/Go).
_IMPORT_TOKENS = ("import", "from", "#include", "include", "using",
                  "package", "require", "export")


def _strip_imports(code):
    out_lines = []
    for ln in code.split("\n"):
        s = ln.strip()
        if not s:
            out_lines.append(ln); continue
        first = s.split(None, 1)[0].lower()
        if first in _IMPORT_TOKENS:
            continue
        # also handle "#include" without space (rare but happens)
        if first.startswith("#include"):
            continue
        out_lines.append(ln)
    return "\n".join(out_lines)


def strip_code(code, mode="full"):
    """Three modes:
       - 'none'           : no strip (control)
       - 'comments_only'  : strip comments only (keep strings + imports)
       - 'full'           : strip comments + strings + imports + collapse ws
    """
    if mode == "none":
        return code
    out = code
    # Always strip comments (both modes that strip)
    # Block comments first to avoid leaving '/' tails.
    out = _RE_C_BLOCK.sub("", out)
    out = _RE_C_LINE.sub("", out)
    out = _RE_PY_COMMENT.sub("", out)
    if mode == "comments_only":
        return out
    # full: also strip strings + imports + collapse newlines
    out = _RE_TRIPLE_DQ.sub('""', out)
    out = _RE_TRIPLE_SQ.sub("''", out)
    out = _RE_STRING_DQ.sub('""', out)
    out = _RE_STRING_SQ.sub("''", out)
    out = _strip_imports(out)
    out = _RE_MULTI_NL.sub("\n\n", out)
    return out


# =============================================================================
# Stylometric feature extraction (identical to exp145, applied to stripped code)
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


def _strip_one_full(code):
    return strip_code(code, mode="full")


def _strip_one_comments(code):
    return strip_code(code, mode="comments_only")


def strip_codes_parallel(codes, mode, n_workers=None, desc="strip"):
    if mode == "none":
        return list(codes)
    fn = _strip_one_full if mode == "full" else _strip_one_comments
    if n_workers is None:
        n_workers = max(1, mp.cpu_count() - 1)
    if n_workers == 1 or len(codes) < 64:
        return [fn(c) for c in tqdm(codes, desc=desc)]
    try:
        with mp.Pool(n_workers) as pool:
            return list(tqdm(pool.imap(fn, codes, chunksize=32),
                             total=len(codes), desc=desc))
    except Exception as e:
        logger.warning(f"[strip] multiprocessing failed ({e}); falling back to serial")
        return [fn(c) for c in tqdm(codes, desc=desc)]


# =============================================================================
# Random Forest training
# =============================================================================

def train_rf(X_train, y_train, n_estimators=200, max_depth=None, n_jobs=-1, seed=42):
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        n_jobs=n_jobs, random_state=seed, class_weight="balanced")
    rf.fit(X_train, y_train)
    return rf


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

def _run_one_mode(mode, train_codes, train_labels, val_codes, val_labels,
                  val_langs, val_sources, test_codes, test_labels,
                  test_langs, test_sources, cfg, sib_mask, dist_mat, n_workers):
    """Strip in `mode`, extract features, train RF, return val/test pack."""
    t0 = time.time()
    s_train = strip_codes_parallel(train_codes, mode, n_workers=n_workers,
                                   desc=f"strip[{mode}](train)")
    s_val   = strip_codes_parallel(val_codes,   mode, n_workers=n_workers,
                                   desc=f"strip[{mode}](val)")
    s_test  = strip_codes_parallel(test_codes,  mode, n_workers=n_workers,
                                   desc=f"strip[{mode}](test)")
    X_tr = extract_features_parallel(s_train, n_workers=n_workers, desc=f"phi[{mode}](train)")
    X_vl = extract_features_parallel(s_val,   n_workers=n_workers, desc=f"phi[{mode}](val)")
    X_ts = extract_features_parallel(s_test,  n_workers=n_workers, desc=f"phi[{mode}](test)")
    for X in (X_tr, X_vl, X_ts):
        np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    rf = train_rf(X_tr, train_labels, n_estimators=cfg.n_estimators,
                  max_depth=cfg.max_depth, n_jobs=n_workers, seed=cfg.seed)
    val_preds  = rf.predict(X_vl)
    test_preds = rf.predict(X_ts)
    val_pack   = eval_pack(val_preds, val_labels, val_langs, val_sources,
                           cfg.n_cls, sib_mask, dist_mat)
    test_pack  = eval_pack(test_preds, test_labels, test_langs, test_sources,
                           cfg.n_cls, sib_mask, dist_mat)
    dt = time.time() - t0
    logger.info(f"[mode={mode}] val={val_pack['overall']['macro_f1']:.4f} "
                f"test={test_pack['overall']['macro_f1']:.4f} time={dt:.0f}s")
    return val_pack, test_pack, dt


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

    results_per_mode = {}
    for mode in ("none", "comments_only", "full"):
        val_pack, test_pack, dt = _run_one_mode(
            mode, train_codes, train_labels, val_codes, val_labels,
            val_langs, val_sources, test_codes, test_labels,
            test_langs, test_sources, cfg, sib_mask, dist_mat, n_workers)
        results_per_mode[mode] = {
            "val_macro":  val_pack["overall"]["macro_f1"],
            "test_macro": test_pack["overall"]["macro_f1"],
            "val_pack":   val_pack,
            "test_pack":  test_pack,
            "wall_sec":   dt,
        }

    # Headline = full
    headline = results_per_mode["full"]
    val_macro  = headline["val_macro"]
    test_macro = headline["test_macro"]
    ts_met = headline["test_pack"]
    gap = val_macro - test_macro
    logger.info(f"[headline=full] val={val_macro:.4f} test={test_macro:.4f} gap={gap:+.4f}")

    # F1
    f1_drop = results_per_mode["none"]["test_macro"] - results_per_mode["full"]["test_macro"]
    # F2
    f2_delta = (results_per_mode["comments_only"]["test_macro"]
                - results_per_mode["full"]["test_macro"])
    # F3 — per-language drop full-minus-none on test (CoDET-M4 most informative)
    none_lang = results_per_mode["none"]["test_pack"]["per_language"]
    full_lang = results_per_mode["full"]["test_pack"]["per_language"]
    per_lang_drop = {}
    for L, none_d in none_lang.items():
        if L in full_lang:
            per_lang_drop[L] = float(none_d["macro_f1"] - full_lang[L]["macro_f1"])

    ts_met["falsifier_F1_strip_drop_vs_none"]      = float(f1_drop)
    ts_met["falsifier_F2_comments_only_minus_full"] = float(f2_delta)
    ts_met["falsifier_F3_per_language_full_minus_none"] = per_lang_drop
    ts_met["mode_none_test_macro"]          = float(results_per_mode["none"]["test_macro"])
    ts_met["mode_none_val_macro"]           = float(results_per_mode["none"]["val_macro"])
    ts_met["mode_comments_only_test_macro"] = float(results_per_mode["comments_only"]["test_macro"])
    ts_met["mode_comments_only_val_macro"]  = float(results_per_mode["comments_only"]["val_macro"])
    ts_met["mode_full_test_macro"]          = float(results_per_mode["full"]["test_macro"])
    ts_met["mode_full_val_macro"]           = float(results_per_mode["full"]["val_macro"])

    return {
        "tag": tag, "method": "STRIPPED",
        "note": ("Comment+string+import-stripped stylometric attribution. "
                 "Audits how much of the field's accuracy is comment/string leakage. "
                 "Headline reports mode='full' (everything stripped). CPU-only."),
        "enc": f"stylometric-d{N_FEATURES}-stripped", "bench": cfg.benchmark,
        "frac": cfg.frac, "n_estimators": cfg.n_estimators,
        "n_features": int(N_FEATURES),
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "mode_none_test_macro":          float(results_per_mode["none"]["test_macro"]),
        "mode_comments_only_test_macro": float(results_per_mode["comments_only"]["test_macro"]),
        "mode_full_test_macro":          float(results_per_mode["full"]["test_macro"]),
        "falsifier_F1_strip_drop_vs_none":       float(f1_drop),
        "falsifier_F2_comments_only_minus_full": float(f2_delta),
        "falsifier_F3_per_language_full_minus_none": per_lang_drop,
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
            tag = f"exp150_stripped_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} "
                            f"F1_strip_drop={res['falsifier_F1_strip_drop_vs_none']:+.4f} "
                            f"time={res['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp150_stripped_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'None':>8} {'Cmts':>8} {'Full':>8} {'F1drop':>9} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['mode_none_test_macro']:>8.4f} "
              f"{r['mode_comments_only_test_macro']:>8.4f} {r['mode_full_test_macro']:>8.4f} "
              f"{r['falsifier_F1_strip_drop_vs_none']:>+9.4f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
