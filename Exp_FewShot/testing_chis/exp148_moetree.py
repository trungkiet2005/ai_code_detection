# exp148 — MOETREE
# =============================================================================
# NAME       : MOETREE (Mixture-of-Decision-Tree Experts with Confidence-Gated
#              Routing over Stylometric Feature Partitions)
# REFERENCE  : Shazeer et al. 2017 "Outrageously Large Neural Networks: The
#              Sparsely-Gated Mixture-of-Experts Layer" (arXiv:1701.06538);
#              Fedus et al. 2022 Switch Transformer (arXiv:2101.03961).
#              Decision-tree experts are classical (Quinlan 1986). MoE with
#              tree experts over disjoint feature partitions is novel and
#              has NEVER been applied to authorship attribution.
# CLAIM      : Author identity is not uniform across feature types. Some
#              authors are distinctive in their indentation, others in their
#              operator choice, others in their comments. Attribution is a
#              ROUTING problem over feature-specialised experts.
# EQUATION   : p(y|x) = sum_{e=1}^E g_e(x) * p_e(y | x_{S_e})
#              where S_e is the feature subset for expert e,
#              g_e(x) = softmax(W h(x))_e, and h(x) is the per-expert
#              max-confidence stack (one scalar per expert).
# WHY NEW    : MoE is widely used in deep LLMs. Applying MoE with DECISION-
#              TREE experts over disjoint FEATURE-SUBSET partitions for
#              authorship attribution is novel — no prior code-attribution
#              paper uses MoE structure, and no prior MoE paper uses
#              decision trees as experts.
# WOW HOOK   : "Author identity is a routing problem. We learn a mixture-
#              of-decision-tree experts over disjoint stylometric feature
#              partitions, gated by per-query confidence. Different authors
#              have their signal in different feature subspaces — the gate
#              learns who needs what."
# FALSIFIER  : (F1) Each expert is USED: gate entropy across train sums
#              to > log(E)/2 (no collapse). Report falsifier_F1_gate_entropy
#              and falsifier_F1_expert_usage_fractions.
#              (F2) MoE composite >= best-single-expert composite by
#              >= 0.005 (mixing helps). Report
#              falsifier_F2_best_single_expert_macro and
#              falsifier_F2_moe_minus_best_single.
#              (F3) Per-class top-expert varies: at least 2 distinct
#              experts are top-1 for at least one class each. Report
#              falsifier_F3_n_top1_experts_distinct.
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, re, subprocess, importlib.util, warnings, glob
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
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp148_moetree")

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
# Stylometric feature extraction (copied/adapted from exp145_rforest)
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
    names += [f"line_len_{s}" for s in ["mean", "std", "max", "min", "q25", "q75"]]   # 6
    names += ["indent_mean", "indent_max", "indent_change_rate"]                       # 3
    names += ["id_len_mean", "id_len_std", "snake_ratio", "camel_ratio"]               # 4
    names += [f"op_{op}" for op in OPS]                                                # 25
    names += [f"kw_{k}" for k in KEYWORDS_15]                                          # 15
    names += ["comment_py_per_line", "comment_c_per_line",
              "comment_char_density", "comment_present"]                               # 4
    names += ["space_density", "tab_density", "punc_density", "alpha_ratio"]           # 4
    return names


FEATURE_NAMES = _build_feature_names()
N_FEATURES = len(FEATURE_NAMES)

# Expert feature partitions (5 disjoint subsets).
# Indices computed against FEATURE_NAMES order above.
_off_line     = 0
_off_indent   = _off_line   + 6
_off_id       = _off_indent + 3
_off_ops      = _off_id     + 4
_off_kw       = _off_ops    + len(OPS)
_off_comment  = _off_kw     + 15
_off_ws       = _off_comment + 4
_end          = _off_ws     + 4

assert _end == N_FEATURES, f"feature offset mismatch: {_end} vs {N_FEATURES}"

EXPERT_SUBSETS = {
    "line":     list(range(_off_line,    _off_line   + 6)),                          # 6
    "indent":   list(range(_off_indent,  _off_indent + 3)),                          # 3
    "id":       list(range(_off_id,      _off_id     + 4)),                          # 4
    "opskw":    list(range(_off_ops,     _off_kw     + 15)),                         # 25 + 15 = 40
    "commws":   list(range(_off_comment, _off_ws     + 4)),                          # 4 + 4 = 8
}
EXPERT_NAMES = list(EXPERT_SUBSETS.keys())
N_EXPERTS = len(EXPERT_NAMES)


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
    f_kw  = [float(len(re.findall(rf"\b{k}\b", code)) / max(len(identifiers), 1))
             for k in KEYWORDS_15]

    comments_py = re.findall(r"#[^\n]*", code)
    comments_c  = re.findall(r"//[^\n]*|/\*[\s\S]*?\*/", code)
    f_comment = [
        float(len(comments_py) / n_lines),
        float(len(comments_c) / n_lines),
        float(sum(len(c) for c in comments_py + comments_c) / n_chars),
        1.0 if comments_py or comments_c else 0.0,
    ]

    n_space = code.count(" ")
    n_tab   = code.count("\t")
    n_alpha = sum(1 for c in code if c.isalpha())
    n_punc  = sum(1 for c in code if not c.isalnum() and not c.isspace())
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
# Mixture-of-tree-experts core
# =============================================================================

def _train_experts(X_train: np.ndarray, y_train: np.ndarray,
                   max_depth: int, n_cls: int, seed: int) -> Dict[str, DecisionTreeClassifier]:
    experts = {}
    for name, idx in EXPERT_SUBSETS.items():
        Xe = X_train[:, idx]
        clf = DecisionTreeClassifier(max_depth=max_depth, random_state=seed,
                                     class_weight="balanced")
        clf.fit(Xe, y_train)
        experts[name] = clf
    return experts


def _expert_probas(experts: Dict[str, DecisionTreeClassifier], X: np.ndarray,
                   n_cls: int) -> np.ndarray:
    """Return (N, E, n_cls) array of per-expert per-class probabilities,
    padded to full n_cls in case an expert hasn't seen all classes."""
    N = X.shape[0]
    out = np.zeros((N, N_EXPERTS, n_cls), dtype=np.float32)
    for ei, name in enumerate(EXPERT_NAMES):
        clf = experts[name]
        Xe = X[:, EXPERT_SUBSETS[name]]
        p = clf.predict_proba(Xe)
        seen = clf.classes_
        for j, c in enumerate(seen):
            if 0 <= c < n_cls:
                out[:, ei, c] = p[:, j]
    return out


def _gate_features(probas: np.ndarray) -> np.ndarray:
    """Per-expert max-confidence stack: (N, E)."""
    return probas.max(axis=2)


def _gate_target(probas: np.ndarray, y: np.ndarray) -> np.ndarray:
    """For each sample, pick the expert whose argmax matches the true label
    with the highest confidence; if none match, pick the most confident expert."""
    N = probas.shape[0]
    targets = np.zeros(N, dtype=np.int64)
    for i in range(N):
        true_c = int(y[i])
        best_e, best_conf, fallback_e, fallback_conf = -1, -1.0, 0, -1.0
        for ei in range(N_EXPERTS):
            pred_c = int(np.argmax(probas[i, ei]))
            conf = float(probas[i, ei, pred_c])
            if conf > fallback_conf:
                fallback_e, fallback_conf = ei, conf
            if pred_c == true_c and conf > best_conf:
                best_e, best_conf = ei, conf
        targets[i] = best_e if best_e >= 0 else fallback_e
    return targets


def _train_gate(gate_feats: np.ndarray, gate_targets: np.ndarray,
                seed: int) -> LogisticRegression:
    """Train a softmax gate over expert IDs."""
    n_classes_seen = len(set(gate_targets.tolist()))
    if n_classes_seen < 2:
        # Cannot train; fallback to a uniform 'gate' (we'll detect this later).
        return None
    gate = LogisticRegression(max_iter=500, multi_class="multinomial",
                              C=1.0, random_state=seed)
    gate.fit(gate_feats, gate_targets)
    return gate


def _gate_weights(gate: LogisticRegression, gate_feats: np.ndarray) -> np.ndarray:
    """Return (N, E) softmax weights summing to 1 per row."""
    N = gate_feats.shape[0]
    W = np.full((N, N_EXPERTS), 1.0 / N_EXPERTS, dtype=np.float32)
    if gate is None:
        return W
    p = gate.predict_proba(gate_feats)  # (N, n_classes_seen)
    seen = gate.classes_
    out = np.full((N, N_EXPERTS), 1e-6, dtype=np.float32)
    for j, e in enumerate(seen):
        if 0 <= e < N_EXPERTS:
            out[:, e] = p[:, j]
    # Renormalise
    out = out / out.sum(axis=1, keepdims=True)
    return out


def _moe_predict(experts: Dict[str, DecisionTreeClassifier],
                 gate: LogisticRegression, X: np.ndarray, n_cls: int):
    """Return (preds, mix_probas, gate_weights)."""
    probas = _expert_probas(experts, X, n_cls)             # (N, E, n_cls)
    gf = _gate_features(probas)                            # (N, E)
    gw = _gate_weights(gate, gf)                           # (N, E)
    # mix[i, c] = sum_e gw[i, e] * probas[i, e, c]
    mix = np.einsum("ne,nec->nc", gw, probas)              # (N, n_cls)
    preds = np.argmax(mix, axis=1)
    return preds, mix, gw, probas


def _per_expert_predict(experts: Dict[str, DecisionTreeClassifier],
                        X: np.ndarray, n_cls: int) -> Dict[str, np.ndarray]:
    out = {}
    for name in EXPERT_NAMES:
        Xe = X[:, EXPERT_SUBSETS[name]]
        out[name] = experts[name].predict(Xe)
    return out


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
    test_cap_per_class:  int = 400
    val_cap_per_class:   int = 200
    max_depth: int = 8
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

    tr_data_frac   = stratified_subsample(tr_data, cfg.frac,                     seed=cfg.seed)
    tr_data_capped = stratified_subsample(tr_data_frac, cfg.train_cap_per_class, seed=cfg.seed + 1)
    ts_data_capped = stratified_subsample(ts_data, cfg.test_cap_per_class,       seed=cfg.seed + 2)
    vl_data_capped = stratified_subsample(vl_data, cfg.val_cap_per_class,        seed=cfg.seed + 3)

    train_codes  = [r["code"][:cfg.max_chars] for r in tr_data_capped]
    train_labels = np.array(list(tr_data_capped["label"]), dtype=np.int64)
    val_codes    = [r["code"][:cfg.max_chars] for r in vl_data_capped]
    val_labels   = np.array(list(vl_data_capped["label"]), dtype=np.int64)
    val_langs    = [r.get("language", "") or "" for r in vl_data_capped]
    val_sources  = [r.get("source", "") or ""   for r in vl_data_capped]
    test_codes   = [r["code"][:cfg.max_chars] for r in ts_data_capped]
    test_labels  = np.array(list(ts_data_capped["label"]), dtype=np.int64)
    test_langs   = [r.get("language", "") or "" for r in ts_data_capped]
    test_sources = [r.get("source", "") or ""   for r in ts_data_capped]

    n_workers = cfg.n_workers if cfg.n_workers > 0 else max(1, mp.cpu_count() - 1)
    logger.info(f"[setup] frac={cfg.frac} train={len(train_codes)} "
                f"val={len(val_codes)} test={len(test_codes)} n_cls={cfg.n_cls} "
                f"n_features={N_FEATURES} n_experts={N_EXPERTS}")

    # Extract features
    t0 = time.time()
    X_train = extract_features_parallel(train_codes, n_workers=n_workers, desc="phi(train)")
    X_val   = extract_features_parallel(val_codes,   n_workers=n_workers, desc="phi(val)")
    X_test  = extract_features_parallel(test_codes,  n_workers=n_workers, desc="phi(test)")
    feat_time = time.time() - t0
    logger.info(f"[feat] X_train={X_train.shape} X_val={X_val.shape} X_test={X_test.shape} "
                f"time={feat_time:.0f}s")

    # Train experts
    experts = _train_experts(X_train, train_labels, cfg.max_depth, cfg.n_cls, cfg.seed)

    # Train gate
    probas_tr = _expert_probas(experts, X_train, cfg.n_cls)
    gf_tr = _gate_features(probas_tr)
    gtgt = _gate_target(probas_tr, train_labels)
    gate = _train_gate(gf_tr, gtgt, seed=cfg.seed)

    # Val
    val_preds, val_mix, val_gw, val_probas = _moe_predict(experts, gate, X_val, cfg.n_cls)
    val_met = eval_pack(val_preds, val_labels.tolist(), val_langs, val_sources,
                        cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]
    logger.info(f"[val] macro={val_macro:.4f}")

    # Test
    t1 = time.time()
    test_preds, test_mix, test_gw, test_probas = _moe_predict(experts, gate, X_test, cfg.n_cls)
    test_time = time.time() - t1
    ts_met = eval_pack(test_preds, test_labels.tolist(), test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro
    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} time={test_time:.0f}s")

    # ---- F1: gate entropy and expert usage on train ----
    # gw on train
    train_gw = _gate_weights(gate, gf_tr)
    avg_usage = train_gw.mean(axis=0)  # (E,)
    avg_usage_dict = {EXPERT_NAMES[i]: float(avg_usage[i]) for i in range(N_EXPERTS)}
    # average per-sample entropy (in nats)
    eps = 1e-12
    ent_per_sample = -np.sum(train_gw * np.log(train_gw + eps), axis=1)
    mean_entropy = float(ent_per_sample.mean())
    max_entropy = float(np.log(N_EXPERTS))
    F1_pass = bool(mean_entropy > max_entropy / 2.0)

    # ---- F2: best single expert macro on test ----
    per_expert_preds_test = _per_expert_predict(experts, X_test, cfg.n_cls)
    expert_macros = {}
    for name, p in per_expert_preds_test.items():
        expert_macros[name] = float(f1_score(test_labels, p, average="macro", zero_division=0))
    best_single = max(expert_macros.values())
    best_single_name = max(expert_macros, key=expert_macros.get)
    F2_delta = test_macro - best_single
    F2_pass = bool(F2_delta >= 0.005)

    # ---- F3: per-class top expert variety ----
    # For each class, find the expert with highest per-class macro
    # using only test samples of that class.
    per_class_top_expert = {}
    for c in range(cfg.n_cls):
        sel = (test_labels == c)
        if sel.sum() == 0: continue
        best_e, best_acc = None, -1.0
        for name, p in per_expert_preds_test.items():
            acc = float((p[sel] == c).mean())
            if acc > best_acc:
                best_acc, best_e = acc, name
        per_class_top_expert[int(c)] = best_e
    n_distinct = len(set(per_class_top_expert.values()))
    F3_pass = bool(n_distinct >= 2)

    ts_met["falsifier_F1_gate_entropy"]            = mean_entropy
    ts_met["falsifier_F1_max_entropy"]             = max_entropy
    ts_met["falsifier_F1_expert_usage_fractions"]  = avg_usage_dict
    ts_met["falsifier_F1_pass"]                    = F1_pass
    ts_met["falsifier_F2_per_expert_macros"]       = expert_macros
    ts_met["falsifier_F2_best_single_expert_macro"]= float(best_single)
    ts_met["falsifier_F2_best_single_expert_name"] = best_single_name
    ts_met["falsifier_F2_moe_minus_best_single"]   = F2_delta
    ts_met["falsifier_F2_pass"]                    = F2_pass
    ts_met["falsifier_F3_per_class_top_expert"]    = {str(k): v for k, v in per_class_top_expert.items()}
    ts_met["falsifier_F3_n_top1_experts_distinct"] = int(n_distinct)
    ts_met["falsifier_F3_pass"]                    = F3_pass

    return {
        "tag": tag, "method": "MOETREE",
        "note": ("Mixture-of-decision-tree-experts over 5 disjoint stylometric "
                 "feature partitions, with a learned logistic-regression gate "
                 "over per-expert max-confidence. CPU-only."),
        "enc": f"stylometric({N_FEATURES}d)/{N_EXPERTS}experts",
        "bench": cfg.benchmark, "frac": cfg.frac,
        "max_depth": cfg.max_depth,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "falsifier_F1_gate_entropy":             mean_entropy,
        "falsifier_F1_expert_usage_fractions":   avg_usage_dict,
        "falsifier_F2_best_single_expert_macro": float(best_single),
        "falsifier_F2_best_single_expert_name":  best_single_name,
        "falsifier_F2_moe_minus_best_single":    F2_delta,
        "falsifier_F3_n_top1_experts_distinct":  int(n_distinct),
        "test_time_sec": test_time,
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
            tag = f"exp148_moetree_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"F1ent={res['falsifier_F1_gate_entropy']:.3f} "
                            f"F2={res['falsifier_F2_moe_minus_best_single']:+.4f} "
                            f"F3distinct={res['falsifier_F3_n_top1_experts_distinct']}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()

    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp148_moetree_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Train':>7} {'Test':>7} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'F1ent':>8} {'F2dlt':>8} {'F3#':>5} {'Best1Exp':>10} {'Wall':>8}")
    print("-" * 140)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['train_size_after_cap']:>7d} {r['test_size_after_cap']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} "
              f"{r['falsifier_F1_gate_entropy']:>8.3f} "
              f"{r['falsifier_F2_moe_minus_best_single']:>+8.4f} "
              f"{r['falsifier_F3_n_top1_experts_distinct']:>5d} "
              f"{r['falsifier_F2_best_single_expert_name']:>10} "
              f"{r['wall']:>8.0f}s")
    print("=" * 140)


if __name__ == "__main__":
    main()
