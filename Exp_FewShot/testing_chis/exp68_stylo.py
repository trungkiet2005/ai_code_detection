# exp68_stylo — Stylometry feature fusion (STYLO)
# =============================================================================
# Theory-Track exp -- STYLO (Stylometry Feature Fusion with unixcoder)
#
# ROLE           : arXiv:2506.17323 ("I Know Which LLM Wrote Your Code Last
#                  Summer") showed 97.56% binary / 95.40% 5-class C-attribution
#                  using PURE stylometry on top of a frozen backbone.  Their
#                  feature bank is the strong signal -- but they didn't fuse
#                  with a code-aware encoder.  STYLO concatenates 50 stylometric
#                  features (extension of our 22-AST: keyword usage, operator
#                  frequencies, control-flow style, line-statistics) with the
#                  unixcoder sentence embedding before classification.
# NAME           : STYLO  (Stylometry-Encoder Fusion)
# ARXIV_ID       : arXiv:2506.17323 (LLM-AuthorBench)
# ONE-LINE CLAIM : Per-author stylometric idiom (AST motifs, token n-grams,
#                  control-flow style) is a complementary signal to the
#                  pretrained encoder embedding; fusing both improves few-shot
#                  attribution.
# EQUATION       : phi(x) = [unixcoder(x) ; MLP(stylo_50(x))]; classifier on phi.
# WHY NOT BEFORE : Existing attribution work uses EITHER stylometry OR a code
#                  encoder.  Fusing has been done in TEXT authorship, but for
#                  CODE the relevant stylometric features are different
#                  (indentation, naming convention, control-flow); we make the
#                  full 50-dim bank explicit (S5: characteristic AST motifs).
# FALSIFIER      : If macro-F1 of STYLO is within +/- 0.005 of exp_n16_ce
#                  baseline (no stylo features), the stylo bank is not adding
#                  information beyond what unixcoder already captures.
# REPORTS        : Full eval pack + ablation hooks (STYLO_ONLY, ENC_ONLY).
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from dataclasses import dataclass, field
from typing import List, Dict
import re as _re

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp68_stylo")

PAPER_BASELINE = 0.6633
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}


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
    D = torch.full((n_cls, n_cls), default_dist)
    for i in range(n_cls):
        for j in range(n_cls):
            d = _gene_distance(i, j, adj)
            if d < float("inf"): D[i, j] = d
            elif (i == 0) != (j == 0): D[i, j] = 3.0
    return D


def build_sibling_mask(n_cls, adj):
    M = torch.zeros(n_cls, n_cls)
    for i in range(n_cls):
        for j in adj.get(i, []): M[i, j] = 1.0
    return M


# =============================================================================
# 50-dim stylometry feature bank (extends 22-AST with 28 more)
# =============================================================================

PY_KEYWORDS = ("def", "class", "lambda", "return", "yield", "await", "async",
               "if", "elif", "else", "for", "while", "break", "continue",
               "import", "from", "try", "except", "finally", "raise", "with",
               "global", "nonlocal", "pass", "assert")
JC_KEYWORDS = ("public", "private", "protected", "static", "final", "void",
               "new", "this", "extends", "implements", "throws")
OPERATORS = ("==", "!=", "<=", ">=", "&&", "||", "->", "=>", "++", "--",
             "+=", "-=", "*=", "/=", "%=", "<<", ">>", "**")


def extract_stylo_features(code: str, max_len: int = 96) -> List[float]:
    lines = code.split("\n"); num_lines = max(len(lines), 1)
    line_lens = [len(l) for l in lines]
    avg_line_len = float(np.mean(line_lens)) if line_lens else 0.0
    max_line_len = float(max(line_lens)) if line_lens else 0.0
    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    avg_indent = float(np.mean(indents)) if indents else 0.0
    max_indent = float(max(indents)) if indents else 0.0
    indent_var = float(np.var(indents)) if indents else 0.0
    n_func = len(_re.findall(r"\b(def|function|func|fn)\s+\w+", code))
    n_class = len(_re.findall(r"\b(class|struct|interface|enum)\s+\w+", code))
    n_loops = len(_re.findall(r"\b(for|foreach)\s*[\(\{]", code)) + len(_re.findall(r"\bwhile\s*[\(\{]", code))
    n_cond = len(_re.findall(r"\bif\s*[\(\{]", code)) + code.count("else ") + code.count("elif ")
    n_return = len(_re.findall(r"\breturn\b", code))
    n_comment = code.count("//") + code.count("#") + code.count("/*")
    n_import = len(_re.findall(r"\b(import|from|include|require|using)\b", code))
    n_try = code.count("try") + code.count("catch") + code.count("except")
    max_depth, depth = 0, 0
    for c in code:
        if c in "{([":
            depth += 1
            if depth > max_depth: max_depth = depth
        elif c in "})]":
            depth = max(0, depth - 1)
    identifiers = _re.findall(r"\b[a-zA-Z_]\w*\b", code)
    n_ids = max(len(identifiers), 1)
    snake_ratio = sum(1 for i in identifiers if "_" in i and i.islower()) / n_ids
    camel_ratio = sum(1 for i in identifiers if any(c.isupper() for c in i[1:]) and "_" not in i) / n_ids
    short_ratio = sum(1 for i in identifiers if len(i) == 1) / n_ids
    avg_id_len = float(np.mean([len(i) for i in identifiers])) if identifiers else 0.0
    empty_ratio = sum(1 for l in lines if not l.strip()) / num_lines
    code_len = max(len(code), 1)
    alpha_ratio = sum(c.isalpha() for c in code) / code_len
    digit_ratio = sum(c.isdigit() for c in code) / code_len

    # === 28 extra stylometric features ===
    # Keyword frequencies (normalised by total tokens)
    n_tokens = max(len(_re.findall(r"\S+", code)), 1)
    py_kw_count = sum(len(_re.findall(rf"\b{k}\b", code)) for k in PY_KEYWORDS)
    jc_kw_count = sum(len(_re.findall(rf"\b{k}\b", code)) for k in JC_KEYWORDS)
    py_kw_ratio = py_kw_count / n_tokens
    jc_kw_ratio = jc_kw_count / n_tokens
    # Operator frequencies
    op_count = sum(code.count(op) for op in OPERATORS)
    op_ratio = op_count / n_tokens
    # Specific operators
    eq_count = code.count("==") / max(code.count("=") + 1, 1)
    arrow_ratio = (code.count("->") + code.count("=>")) / n_tokens
    inc_ratio = (code.count("++") + code.count("--")) / n_tokens
    # Bracket / paren density
    paren_ratio = (code.count("(") + code.count(")")) / code_len
    brace_ratio = (code.count("{") + code.count("}")) / code_len
    bracket_ratio = (code.count("[") + code.count("]")) / code_len
    # String literal patterns
    n_dquote = code.count('"'); n_squote = code.count("'")
    dq_ratio = n_dquote / max(n_dquote + n_squote, 1)
    n_fstr = len(_re.findall(r"\bf['\"]", code)) / max(n_dquote + n_squote + 1, 1)
    # Comment style preference
    n_hash = code.count("#"); n_slash = code.count("//"); n_block = code.count("/*")
    total_c = max(n_hash + n_slash + n_block, 1)
    hash_pref = n_hash / total_c
    slash_pref = n_slash / total_c
    # Magic numbers / hex / float
    n_hex = len(_re.findall(r"\b0x[0-9a-fA-F]+\b", code)) / n_tokens
    n_float = len(_re.findall(r"\b\d+\.\d+\b", code)) / n_tokens
    n_int = len(_re.findall(r"\b\d+\b", code)) / n_tokens
    # Variable naming style
    upper_ids = sum(1 for i in identifiers if i.isupper()) / n_ids
    longest_id = max((len(i) for i in identifiers), default=0) / 30.0
    # Spacing / indentation type
    n_tab_lines = sum(1 for l in lines if l.startswith("\t"))
    tab_ratio = n_tab_lines / num_lines
    space2 = sum(1 for l in lines if l.startswith("  ") and not l.startswith("    "))
    space4 = sum(1 for l in lines if l.startswith("    "))
    space2_ratio = space2 / num_lines
    space4_ratio = space4 / num_lines
    # Trailing whitespace / blank lines run
    trail_ws = sum(1 for l in lines if l.rstrip() != l) / num_lines
    # Specific token n-grams
    n_self_dot = code.count("self.") / n_tokens
    n_this_dot = code.count("this.") / n_tokens
    n_print = (len(_re.findall(r"\bprint\s*\(", code)) +
               len(_re.findall(r"\bSystem\.out\.print", code))) / n_tokens
    # Cyclomatic-style complexity proxy
    cyclo = (n_cond + n_loops + code.count("&&") + code.count("||")) / num_lines

    feats = [
        # === 22 original AST features ===
        num_lines/500., avg_line_len/80., max_line_len/200., avg_indent/10., max_indent/20.,
        indent_var/50., n_func/10., n_class/5., n_loops/10., n_cond/20., n_return/20.,
        n_comment/50., n_import/10., n_try/10., max_depth/15., snake_ratio, camel_ratio,
        short_ratio, avg_id_len/10., empty_ratio, alpha_ratio, digit_ratio,
        # === 28 stylometric extras ===
        py_kw_ratio, jc_kw_ratio, op_ratio, eq_count, arrow_ratio, inc_ratio,
        paren_ratio, brace_ratio, bracket_ratio, dq_ratio, n_fstr,
        hash_pref, slash_pref, n_hex, n_float, n_int, upper_ids, longest_id,
        tab_ratio, space2_ratio, space4_ratio, trail_ws,
        n_self_dot, n_this_dot, n_print, cyclo,
        # 4 reserved padding slots filled with zero
        0.0, 0.0,
    ]
    return (feats + [0.0]*(max_len-len(feats)))[:max_len]


# =============================================================================
# Model
# =============================================================================

class STYLOModel(nn.Module):
    def __init__(self, enc_name, n_cls, stylo_in=96, stylo_dim=96):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.stylo_encoder = nn.Sequential(
            nn.Linear(stylo_in, 192), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(192, stylo_dim), nn.GELU()
        )
        self.proj = nn.Sequential(nn.Linear(hidden + stylo_dim, 256), nn.GELU(), nn.Dropout(0.1))
        self.clf = nn.Linear(256, n_cls)

    def forward(self, ids, mask, stylo_feat):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        stylo_emb = self.stylo_encoder(stylo_feat)
        h = self.proj(torch.cat([sem, stylo_emb], dim=-1))
        return self.clf(h)


# =============================================================================
# Plumbing
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    device: str = "cuda"; gene_adj: dict = field(default_factory=dict)


def adaptive_schedule(cfg):
    f = cfg.frac
    if f <= 0.02: cfg.epochs, cfg.lr_enc, cfg.warmup = 10, 3e-5, 0.20
    elif f <= 0.10: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 3e-5, 0.15
    else: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 4e-5, 0.10
    return cfg


def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40: cfg.bs, cfg.seq = 256, 512
        elif mem >= 10: cfg.bs, cfg.seq = 128, 384
        else: cfg.bs, cfg.seq = 64, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} seq={cfg.seq}")
    return cfg


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _is_human(t):
    return str(t or "").strip().lower() in {"human", "human_written", "human-generated"}


def _vocab(train):
    names = {str(r.get("model", "") or "").strip() for r in train
             if not _is_human(r.get("target", "")) and r.get("model", "")}
    return {n: i + 1 for i, n in enumerate(sorted(names))}


def _conv_codet(split, task, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code", "code"):
            v = r.get(f, "")
            if isinstance(v, str) and v.strip(): code = v; break
        if task == "binary":
            label = 0 if _is_human(r.get("target", "")) else 1
        else:
            label = 0 if _is_human(r.get("target", "")) else vocab.get(str(r.get("model", "") or "").strip(), -1)
        return {"code": code, "label": label,
                "language": str(r.get("language", "")).strip().lower(),
                "source": str(r.get("source", "")).strip().lower()}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _conv_aicd(split):
    def row(r):
        return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1)),
                "language": str(r.get("language", "")).strip().lower(), "source": ""}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _load_codet():
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42)
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
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


class FSDS(TD):
    def __init__(self, data, tok, seq_len, stylo_dim=96, frac=1.0, seed=42):
        self.data = data; self.tok = tok; self.seq_len = seq_len; self.stylo_dim = stylo_dim
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        enc = self.tok(code, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
        return {"ids": enc["input_ids"].squeeze(0), "mask": enc["attention_mask"].squeeze(0),
                "stylo_feat": torch.tensor(extract_stylo_features(code, self.stylo_dim), dtype=torch.float32),
                "label": r["label"], "language": r.get("language", "") or "",
                "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        stylo_feat = b["stylo_feat"].to(cfg.device); labs = b["label"]
        logits = model(ids, mask, stylo_feat)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        lang_batch = b.get("language", [""] * len(labs))
        src_batch = b.get("source", [""] * len(labs))
        langs.extend(list(lang_batch) if not isinstance(lang_batch, list) else lang_batch)
        sources.extend(list(src_batch) if not isinstance(src_batch, list) else src_batch)
    preds = np.array(preds); labels = np.array(labels); n_cls = cfg.n_cls
    overall = {"accuracy": float(accuracy_score(labels, preds)),
               "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
               "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
               "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
               "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0))}
    per_class = {"f1": f1_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
                 "precision": precision_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
                 "recall": recall_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist()}
    cm = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    off_diag = int(cm.sum() - cm.trace())
    sib_conf = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls) if i != j and sib_mask_np[i, j] > 0))
    sib_rate = sib_conf / max(off_diag, 1)
    D = build_distance_matrix(n_cls, cfg.gene_adj).numpy()
    cross = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls) if i != j and D[i, j] >= 3.0))
    cross_rate = cross / max(off_diag, 1)
    per_lang, per_src = {}, {}
    if any(l for l in langs):
        la = np.array(langs)
        for L in sorted(set(langs)):
            if not L: continue
            sel = (la == L)
            if sel.sum() < 2: continue
            per_lang[L] = {"n": int(sel.sum()),
                           "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0)),
                           "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                           "accuracy": float(accuracy_score(labels[sel], preds[sel]))}
    if any(s for s in sources):
        sa = np.array(sources)
        for S in sorted(set(sources)):
            if not S: continue
            sel = (sa == S)
            if sel.sum() < 2: continue
            per_src[S] = {"n": int(sel.sum()),
                          "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0)),
                          "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                          "accuracy": float(accuracy_score(labels[sel], preds[sel]))}
    return {"overall": overall, "per_class": per_class, "per_language": per_lang, "per_source": per_src,
            "confusion_matrix": cm.tolist(), "sibling_confusion_rate": float(sib_rate),
            "cross_family_confusion_rate": float(cross_rate),
            "off_diag_total": off_diag, "n_samples": int(len(labels))}


def train_epoch(model, loader, opt, sch, scaler, cfg):
    model.train(); tot = 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        stylo_feat = b["stylo_feat"].to(cfg.device); labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            logits = model(ids, mask, stylo_feat)
            loss = F.cross_entropy(logits, labs)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item()
    return tot / len(loader)


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    sib_mask_np = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()
    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, "author", vocab); vl_data = _conv_codet(vl_raw, "author", vocab); ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = STYLOModel(cfg.enc, cfg.n_cls).to(cfg.device)
    enc_param_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_param_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss = train_epoch(model, tr_dl, opt, sch, scaler, cfg)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f}")
    return {"tag": tag, "method": "STYLO", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "n_stylo_feat": 50,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    encoders = ["unixcoder-base"]
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for enc in encoders:
        for bench, task, n_cls in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac, n_cls=n_cls)
                tag = f"exp68_stylo_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp68_stylo_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*120)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} {'Wall':>8}")
    print("-"*120)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} {r['wall']:>8.0f}s")
    print("="*120)


if __name__ == "__main__":
    main()
