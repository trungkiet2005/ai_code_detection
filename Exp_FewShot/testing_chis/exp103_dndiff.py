# exp103 — DENOISE-v3
# NAME       : DENOISE-v3 (Conditional Style-Preserving Denoising for Authorship)
# REFERENCE  : new (inspired by BART arXiv:1910.13461, conditional diffusion arXiv:2105.05233)
# CLAIM      : Style is *what cannot be denoised away*. If we corrupt ONLY non-style
#              attributes (mask variable names but preserve their count and casing
#              convention; rename methods but preserve their length-class; jitter
#              whitespace but preserve indent depth), then the encoder learns to
#              recover the corrupted attribute CONDITIONAL on style features it
#              already has. This forces explicit factorisation: style features must
#              survive the corruption, non-style features must be recoverable.
# EQUATION   : x_corr = T_nonstyle(x)         # corrupt only non-style attributes
#              c_style = phi_style(x)         # style-conditioning vector (from clean)
#              L_dn = CE(decoder(phi(x_corr), c_style), x_tokens)
#              + lambda_inv * 1 - cos(phi_style(x), phi_style(x_corr))     # style invariance
# WHY NEW    : No prior code-attribution paper uses CONDITIONAL denoising where the
#              conditioning is the very signal we want to extract. The corruption is
#              explicitly designed to be ORTHOGONAL to style (mask names but keep
#              count: style-preserving; rename methods but keep length: style-preserving).
# WOW HOOK   : "Style is what cannot be denoised away — we corrupt everything except style,
#              and let the encoder discover the residual that is identity."
# FALSIFIER  : (F1) If mean cos(phi_style(x), phi_style(x_corr)) < 0.85 on test, the
#              corruption broke style → corruption is not style-preserving. (F2) If
#              composite ≤ DENOISE-v1 (exp102), conditional denoising adds nothing
#              over plain denoising — the conditioning is decorative.
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
import re as _re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

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
logger = logging.getLogger("exp103_dndiff")

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
# Style-preserving corruption
# =============================================================================

_IDENT_RE = _re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_PY_DEF_RE = _re.compile(r"\bdef\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")
_C_DEF_RE = _re.compile(r"\b(?:void|int|float|double|char|long|short|bool|auto|static|public|private|protected)\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(")

_PY_KEYWORDS = {
    "False", "None", "True", "and", "as", "assert", "async", "await", "break",
    "class", "continue", "def", "del", "elif", "else", "except", "finally",
    "for", "from", "global", "if", "import", "in", "is", "lambda", "nonlocal",
    "not", "or", "pass", "raise", "return", "try", "while", "with", "yield",
    "self", "cls", "print", "len", "range", "int", "str", "float", "bool",
    "list", "dict", "set", "tuple", "void", "char", "double", "long", "short",
    "public", "private", "protected", "static", "auto", "const", "extern",
    "main", "return", "if", "else", "while", "for", "do", "switch", "case",
    "default", "break", "continue", "function", "var", "let", "new",
    "delete", "this", "throw", "try", "catch", "finally", "include", "define",
}


def _ident_mask(code: str, rng: random.Random) -> Tuple[str, bool]:
    """Mode 1: mask 50% of identifiers (KEEP count + first-letter casing).
    Each masked identifier becomes <MASK<i>_U> or <MASK<i>_L>."""
    matches = list(_IDENT_RE.finditer(code))
    cands = [m for m in matches if m.group() not in _PY_KEYWORDS and len(m.group()) > 1]
    if not cands:
        return code, False
    n_pick = max(1, len(cands) // 2)
    picked = set(id(m) for m in rng.sample(cands, min(n_pick, len(cands))))
    out = []
    last = 0
    counter = 0
    for m in matches:
        out.append(code[last:m.start()])
        if id(m) in picked:
            tag = "U" if m.group()[0].isupper() else "L"
            out.append(f"<MASK{counter}_{tag}>")
            counter += 1
        else:
            out.append(m.group())
        last = m.end()
    out.append(code[last:])
    return "".join(out), counter > 0


def _ws_jitter_preserve_indent(code: str, rng: random.Random) -> Tuple[str, bool]:
    """Mode 2: whitespace jitter around operators inside lines (KEEP leading indent)."""
    ops = "+-*/%=<>,;()[]{}|&^~!"
    lines = code.split("\n")
    fired = False
    new_lines = []
    for line in lines:
        # Split leading whitespace from rest
        i = 0
        while i < len(line) and line[i] in (" ", "\t"):
            i += 1
        lead = line[:i]
        body = line[i:]
        new_body_chars = []
        for c in body:
            new_body_chars.append(c)
            if c in ops and rng.random() < 0.25:
                new_body_chars.append(" ")
                fired = True
        new_lines.append(lead + "".join(new_body_chars))
    return "\n".join(new_lines), fired


def _method_rename_len_preserve(code: str, rng: random.Random) -> Tuple[str, bool]:
    """Mode 3: rename method definitions with same-length generic name."""
    def gen_name(L: int) -> str:
        # Generic alphabetic name of length L starting with letter
        if L < 1:
            return "x"
        first = chr(ord("a") + rng.randint(0, 25))
        rest = "".join(chr(ord("a") + rng.randint(0, 25)) if rng.random() < 0.5
                       else chr(ord("0") + rng.randint(0, 9)) for _ in range(L - 1))
        return first + rest

    fired = [False]

    def repl_py(m):
        name = m.group(1)
        if name in _PY_KEYWORDS or len(name) < 2:
            return m.group(0)
        if rng.random() < 0.5:
            fired[0] = True
            return m.group(0).replace(name, gen_name(len(name)), 1)
        return m.group(0)

    new = _PY_DEF_RE.sub(repl_py, code)
    if not fired[0]:
        new = _C_DEF_RE.sub(repl_py, new)
    return new, fired[0]


def style_preserving_corrupt(code: str, language: str, rng: random.Random) -> Tuple[str, str]:
    """Pick one of 3 corruption modes; if fail, fallback to ws-jitter (guaranteed fire)."""
    modes = ["ident_mask", "ws_jitter", "method_rename"]
    rng.shuffle(modes)
    for m in modes:
        try:
            if m == "ident_mask":
                new, fired = _ident_mask(code, rng)
            elif m == "ws_jitter":
                new, fired = _ws_jitter_preserve_indent(code, rng)
            else:
                new, fired = _method_rename_len_preserve(code, rng)
        except Exception:
            continue
        if fired:
            return new, m
    # Guaranteed fallback: force a single space insertion around the first operator
    ops = set("+-*/%=<>,;()[]{}|&^~!")
    for i, c in enumerate(code):
        if c in ops:
            return code[:i+1] + " " + code[i+1:], "fallback_ws"
    return code + "\n", "fallback_newline"


# =============================================================================
# Data plumbing
# =============================================================================

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_dn: float = 0.5; lambda_inv: float = 0.5
    style_dim: int = 128
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
        # Two forward passes + decoder over full sequence => reduce bs by 1.5x
        if mem >= 40: cfg.bs, cfg.seq = 84, 512
        elif mem >= 10: cfg.bs, cfg.seq = 42, 384
        else: cfg.bs, cfg.seq = 20, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} (reduced 1.5x for dual-forward+decoder) seq={cfg.seq}")
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


# =============================================================================
# Tokenisation — faithful UniXcoder protocol
# =============================================================================

def _tokenize(code, tokenizer, max_len):
    toks = tokenizer.tokenize(" ".join(code.split()))[:max_len - 4]
    toks = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + toks + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_len - len(ids))
    return ids[:max_len]


# =============================================================================
# Dataset
# =============================================================================

class FSDS_DnDiff(TD):
    """Returns: ids0/mask0 (clean), ids1/mask1 (corrupted), label, language, source, corrupt_mode."""
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42, do_corrupt=True):
        self.data = data; self.tok = tok; self.seq_len = seq_len; self.do_corrupt = do_corrupt
        self.seed = seed
        self.pad_id = tok.pad_token_id
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_DnDiff] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        lang = r.get("language", "") or ""
        ids0 = _tokenize(code, self.tok, self.seq_len)
        mask0 = [1 if t != self.pad_id else 0 for t in ids0]
        if self.do_corrupt:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            code_c, mode = style_preserving_corrupt(code, lang, rng)
            ids1 = _tokenize(code_c, self.tok, self.seq_len)
            mask1 = [1 if t != self.pad_id else 0 for t in ids1]
        else:
            ids1 = ids0; mask1 = mask0; mode = "none"
        return {"ids0": torch.tensor(ids0, dtype=torch.long),
                "mask0": torch.tensor(mask0, dtype=torch.long),
                "ids1": torch.tensor(ids1, dtype=torch.long),
                "mask1": torch.tensor(mask1, dtype=torch.long),
                "label": r["label"],
                "language": lang,
                "source": r.get("source", "") or "",
                "corrupt_mode": mode}


# =============================================================================
# Model
# =============================================================================

class DnDiffModel(nn.Module):
    def __init__(self, enc_name, n_cls, vocab_size, style_dim=128):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.style_proj = nn.Sequential(nn.Linear(hidden, 256), nn.GELU(), nn.Dropout(0.1),
                                        nn.Linear(256, style_dim))
        self.dec = nn.Linear(hidden + style_dim, vocab_size)
        self.clf = nn.Linear(style_dim, n_cls)
        self.style_dim = style_dim
        self.vocab_size = vocab_size
        self.n_cls = n_cls

    def encode_style(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return F.normalize(self.style_proj(sem), dim=-1)

    def encode_tokens(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        return out.last_hidden_state

    def encode_full(self, ids, mask):
        """Return (style_code, classifier_logits) for eval path."""
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        style = F.normalize(self.style_proj(sem), dim=-1)
        return style, self.clf(style)

    def forward(self, ids_clean, mask_clean, ids_corr, mask_corr, labels=None):
        style_clean = self.encode_style(ids_clean, mask_clean)
        style_corr = self.encode_style(ids_corr, mask_corr)
        tok_corr = self.encode_tokens(ids_corr, mask_corr)
        B, L, H = tok_corr.shape
        style_b = style_clean.unsqueeze(1).expand(B, L, self.style_dim)
        dec_in = torch.cat([tok_corr, style_b], dim=-1)
        logits_dn = self.dec(dec_in)
        logits_cls = self.clf(style_clean)
        return logits_dn, logits_cls, style_clean, style_corr


# =============================================================================
# Train / Eval
# =============================================================================

def train_epoch(model, loader, opt, sch, scaler, cfg, tok):
    model.train()
    tot_ce, tot_dn, tot_inv = 0.0, 0.0, 0.0
    mode_counts: Dict[str, int] = {}
    for b in tqdm(loader, desc="Train"):
        ids_c = b["ids0"].to(cfg.device); mask_c = b["mask0"].to(cfg.device)
        ids_x = b["ids1"].to(cfg.device); mask_x = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        for m in b.get("corrupt_mode", []):
            mode_counts[m] = mode_counts.get(m, 0) + 1
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            logits_dn, logits_cls, style_c, style_x = model(ids_c, mask_c, ids_x, mask_x, labs)
            B, L, V = logits_dn.shape
            loss_dn = F.cross_entropy(logits_dn.view(B*L, V), ids_c.view(B*L), ignore_index=tok.pad_token_id)
            loss_inv = 1.0 - (style_c * style_x).sum(dim=-1).mean()
            loss_ce = F.cross_entropy(logits_cls, labs)
            loss = cfg.lambda_dn * loss_dn + cfg.lambda_inv * loss_inv + loss_ce
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        tot_ce += loss_ce.item(); tot_dn += loss_dn.item(); tot_inv += loss_inv.item()
    n = max(1, len(loader))
    return tot_ce / n, tot_dn / n, tot_inv / n, mode_counts


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=False):
    model.eval()
    preds, labels, langs, sources = [], [], [], []
    inv_cos_list = []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        style0, logits = model.encode_full(ids0, mask0)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            ids1 = b["ids1"].to(cfg.device); mask1 = b["mask1"].to(cfg.device)
            style1, _ = model.encode_full(ids1, mask1)
            sim = (style0 * style1).sum(dim=-1).detach().cpu().float().numpy()
            inv_cos_list.extend(sim.tolist())
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
    cross = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls) if i != j and dist_mat_cpu[i, j] >= 3.0))
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
    out = {"overall": overall, "per_class": per_class, "per_language": per_lang, "per_source": per_src,
           "confusion_matrix": cm.tolist(), "sibling_confusion_rate": float(sib_rate),
           "cross_family_confusion_rate": float(cross_rate),
           "off_diag_total": off_diag, "n_samples": int(len(labels))}
    if collect_falsifier:
        out["falsifier"] = {
            "mean_view_cos_F1": float(np.mean(inv_cos_list)) if inv_cos_list else 0.0,
            "std_view_cos": float(np.std(inv_cos_list)) if inv_cos_list else 0.0,
            "n_eval": len(inv_cos_list),
        }
    return out


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat_t = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist_mat_cpu = dist_mat_t.numpy()
    sib_mask_np = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()
    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, "author", vocab); vl_data = _conv_codet(vl_raw, "author", vocab); ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    tr_ds = FSDS_DnDiff(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_corrupt=True)
    vl_ds = FSDS_DnDiff(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_corrupt=True)
    ts_ds = FSDS_DnDiff(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_corrupt=True)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"lambda_dn={cfg.lambda_dn} lambda_inv={cfg.lambda_inv}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    vocab_size = tok.vocab_size
    model = DnDiffModel(cfg.enc, cfg.n_cls, vocab_size, cfg.style_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    cum_mode_counts: Dict[str, int] = {}
    final_loss_dn = 0.0
    for epoch in range(cfg.epochs):
        ce, dn, inv, mode_counts = train_epoch(model, tr_dl, opt, sch, scaler, cfg, tok)
        for m, c in mode_counts.items():
            cum_mode_counts[m] = cum_mode_counts.get(m, 0) + c
        final_loss_dn = dn
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] ce={ce:.4f} dn={dn:.4f} inv={inv:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=True)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    fa = ts_met["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"view_cos={fa['mean_view_cos_F1']:.3f}")
    total_modes = max(sum(cum_mode_counts.values()), 1)
    mode_dist = {m: c / total_modes for m, c in cum_mode_counts.items()}
    return {"tag": tag, "method": "DENOISE-v3", "upstream": "novel",
            "note": "Conditional style-preserving denoising; corruption orthogonal to style",
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_dn": cfg.lambda_dn, "lambda_inv": cfg.lambda_inv,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "mean_view_cos_test": float(fa["mean_view_cos_F1"]),
            "final_loss_dn": float(final_loss_dn),
            "corrupt_mode_distribution": mode_dist,
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
                tag = f"exp103_dndiff_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"view_cos={res['mean_view_cos_test']:.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp103_dndiff_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'view_cos':>10} {'Wall':>8}")
    print("-"*140)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['mean_view_cos_test']:>10.3f} {r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
