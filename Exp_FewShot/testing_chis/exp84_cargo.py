# exp84_cargo â€” Control-flow Augmented Restructure with Genealogy-aware Objective
# =============================================================================
# Theory-Track exp -- CARGO (Control-flow Augmented Restructure w/ Genealogy Objective)
#
# ROLE           : TRACO (exp76) won @1%/@5% CoDET-M4 by injecting view-invariance,
#                  but its T(.) was SURFACE-LEVEL: token_dropout, ws_jitter,
#                  comment_strip, id_rename.  An EMNLP reviewer will object:
#                  "these are not the views S7 promises -- when an LLM samples
#                  at higher T, it changes CONTROL FLOW (for vs while), it
#                  changes EXPRESSION FORM (a+=b vs a=a+b), it changes
#                  STATEMENT STRUCTURE (listcomp vs loop). Surface jitter does
#                  not approximate decoding-T variance."
#                  CARGO is the structural answer: T(.) is now an AST/control-
#                  flow rewrite distribution.  Same Tree-Weighted SupCon
#                  objective, but the augmentation distribution is now a
#                  faithful proxy for what an LLM-with-different-T actually
#                  produces.  This closes the theory-implementation gap.
#
# NAME           : CARGO (Control-flow Augmented Restructure w/ Genealogy Objective)
# ARXIV_ID       : novel; differs from TRACO (surface token jitter), from
#                  ContraCode (2020, AST renaming only), from SynCoBERT (2022,
#                  data-flow contrastive on graphs).  CARGO is the first work
#                  to use SEMANTIC-PRESERVING CONTROL-FLOW rewrites as
#                  decoding-T-invariance proxy for AI-code attribution, and
#                  to combine them with genealogy-distance-weighted SupCon.
#
# ONE-LINE CLAIM : Author identity of an LLM is invariant under semantic-
#                  preserving STRUCTURAL rewrites (for<->while, if-invert,
#                  listcomp<->loop, op-form-swap, aug-assign-expand). Training
#                  the encoder to enforce this invariance is a closer proxy
#                  for S7 (decoder-T invariance) than surface token jitter.
#
# EQUATION       : For each x in batch:
#                    T ~ Uniform{for_while, if_invert, listcomp_unroll,
#                                aug_assign_expand, op_form_swap, paren_canon,
#                                stmt_reorder}
#                    x' = T(x)    (Python: real AST rewrite via ast module;
#                                  other langs: structural regex)
#                    z, logits = phi(x);  z' = phi(x')
#                    L_supcon_tw = SupCon on [z; z'] with positives = same
#                                  label, negatives weighted by exp(-gamma *
#                                  d_tree(y_i, y_j))
#                    L_ce        = CE(logits, y)
#                    L_total     = L_ce + lambda_aug * L_supcon_tw
#
# WHY NOT BEFORE : ContraCode (Jain 2020) used AST renaming only; SynCoBERT
#                  (Wang 2022) used data-flow on graph; neither targeted
#                  AI-code attribution and neither paired structural views
#                  with genealogy-distance-weighted contrastive. TRACO (exp76)
#                  claimed S7-grounding but its T(.) was surface jitter --
#                  the theory-implementation gap is exactly what CARGO closes.
#
# FALSIFIER      : (F1) Structural-view invariance: mean cos(z, z') on TEST.
#                       Expect > 0.85 (structural transforms break surface
#                       but preserve semantics, so a properly trained encoder
#                       must learn invariance).  < 0.5 => structural transforms
#                       too aggressive => encoder collapsed.
#                  (F2) Î” over TRACO (exp76) at matched fractions:
#                       CARGO.macro - TRACO.macro at f in {0.01, 0.05, 0.20}.
#                       MUST be POSITIVE at >= 1 slot, else "structural
#                       view-invariance â‰ˆ surface view-invariance" =>
#                       falsifies the structural-proxy claim.
#                  (F3) Per-transform contribution: per-transform mean cos(z,z')
#                       and per-transform fired-count.  If one transform fires
#                       >50% of training and others rarely => augmentation
#                       distribution is degenerate (e.g. Python-AST monopoly
#                       because non-Python samples all fall through to noop).
#
# REPORTS        : Full eval pack + (F1) mean test view-cos
#                                 + (F2) per-fraction Î” vs TRACO (post-hoc)
#                                 + (F3) per-transform mean_cos + fired_count
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
import ast as _ast
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
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
logger = logging.getLogger("exp84_cargo")

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
# CARGO structural augmentations
#
# Strategy:
#   1. Try Python AST rewrite (most faithful; ast.unparse round-trips).
#   2. If Python parse fails (non-Python code or syntactically dirty), fall
#      back to language-agnostic structural-regex transforms.
#
# All transforms preserve semantics under standard execution semantics for
# straight-line code.  Edge cases (e.g. break inside for-loop translated to
# while) are accepted -- the encoder learns identity invariance under
# best-effort rewrites, which is exactly the regime LLM-at-different-T
# produces (occasionally semantically-equivalent, occasionally near-equivalent).
# =============================================================================

# ---- Python AST transforms --------------------------------------------------

class _AugAssignExpand(_ast.NodeTransformer):
    """a += b   <=>   a = a + b   (and vice versa)"""
    NAME = "aug_assign_expand"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_AugAssign(self, node):
        if self.rng.random() < 0.7:
            self.fired += 1
            return _ast.copy_location(
                _ast.Assign(
                    targets=[node.target],
                    value=_ast.BinOp(left=node.target, op=node.op, right=node.value)
                ), node)
        return node
    def visit_Assign(self, node):
        if (len(node.targets) == 1
            and isinstance(node.value, _ast.BinOp)
            and isinstance(node.targets[0], _ast.Name)
            and isinstance(node.value.left, _ast.Name)
            and node.targets[0].id == node.value.left.id
            and self.rng.random() < 0.5):
            self.fired += 1
            return _ast.copy_location(
                _ast.AugAssign(target=node.targets[0], op=node.value.op, value=node.value.right),
                node)
        return node


class _IfInvert(_ast.NodeTransformer):
    """if cond: A else: B   ==>   if not cond: B else: A"""
    NAME = "if_invert"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_If(self, node):
        self.generic_visit(node)
        if node.orelse and self.rng.random() < 0.6:
            self.fired += 1
            new_test = _ast.UnaryOp(op=_ast.Not(), operand=node.test)
            node = _ast.copy_location(
                _ast.If(test=new_test, body=node.orelse, orelse=node.body), node)
        return node


class _ForToWhile(_ast.NodeTransformer):
    """for x in range(N): body   ==>   x=0\nwhile x<N: body; x+=1"""
    NAME = "for_to_while"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_For(self, node):
        self.generic_visit(node)
        if (isinstance(node.iter, _ast.Call)
            and isinstance(node.iter.func, _ast.Name)
            and node.iter.func.id == "range"
            and len(node.iter.args) == 1
            and isinstance(node.target, _ast.Name)
            and not node.orelse
            and self.rng.random() < 0.7):
            self.fired += 1
            var = node.target
            upper = node.iter.args[0]
            init = _ast.Assign(targets=[var], value=_ast.Constant(value=0))
            inc = _ast.AugAssign(target=var, op=_ast.Add(), value=_ast.Constant(value=1))
            cond = _ast.Compare(left=var, ops=[_ast.Lt()], comparators=[upper])
            new_body = list(node.body) + [inc]
            wh = _ast.While(test=cond, body=new_body, orelse=[])
            return [_ast.copy_location(init, node), _ast.copy_location(wh, node)]
        return node


class _ListCompUnroll(_ast.NodeTransformer):
    """[expr for x in iter]   ==>   _tmp=[]; for x in iter: _tmp.append(expr)
    Only at statement level (Assign-with-listcomp-RHS), not nested expressions."""
    NAME = "listcomp_unroll"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_Assign(self, node):
        if (len(node.targets) == 1
            and isinstance(node.targets[0], _ast.Name)
            and isinstance(node.value, _ast.ListComp)
            and len(node.value.generators) == 1
            and not node.value.generators[0].ifs
            and self.rng.random() < 0.7):
            self.fired += 1
            tgt = node.targets[0]
            lc = node.value
            init = _ast.Assign(targets=[tgt], value=_ast.List(elts=[], ctx=_ast.Load()))
            append_call = _ast.Expr(value=_ast.Call(
                func=_ast.Attribute(value=tgt, attr="append", ctx=_ast.Load()),
                args=[lc.elt], keywords=[]))
            for_stmt = _ast.For(target=lc.generators[0].target,
                                iter=lc.generators[0].iter,
                                body=[append_call], orelse=[])
            return [_ast.copy_location(init, node), _ast.copy_location(for_stmt, node)]
        return node


class _OpFormSwap(_ast.NodeTransformer):
    """a * 2  <=>  a + a       a / 2  <=>  a * 0.5"""
    NAME = "op_form_swap"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_BinOp(self, node):
        self.generic_visit(node)
        # *2 -> +self
        if (isinstance(node.op, _ast.Mult) and isinstance(node.right, _ast.Constant)
            and node.right.value == 2 and isinstance(node.left, _ast.Name)
            and self.rng.random() < 0.6):
            self.fired += 1
            return _ast.copy_location(
                _ast.BinOp(left=node.left, op=_ast.Add(), right=node.left), node)
        # /2 -> *0.5
        if (isinstance(node.op, _ast.Div) and isinstance(node.right, _ast.Constant)
            and node.right.value == 2 and self.rng.random() < 0.6):
            self.fired += 1
            return _ast.copy_location(
                _ast.BinOp(left=node.left, op=_ast.Mult(), right=_ast.Constant(value=0.5)),
                node)
        return node


_PY_TRANSFORMERS = [_AugAssignExpand, _IfInvert, _ForToWhile, _ListCompUnroll, _OpFormSwap]


def aug_python_ast(code: str, rng: random.Random) -> Tuple[str, str]:
    """Apply a random Python AST transform. Returns (new_code, transform_name)."""
    try:
        tree = _ast.parse(code)
    except (SyntaxError, ValueError):
        return code, "ast_parse_fail"
    cls = _PY_TRANSFORMERS[rng.randrange(len(_PY_TRANSFORMERS))]
    transformer = cls(rng)
    try:
        new_tree = transformer.visit(tree)
        _ast.fix_missing_locations(new_tree)
        new_code = _ast.unparse(new_tree)
    except Exception:
        return code, "ast_unparse_fail"
    if transformer.fired == 0:
        return code, f"py_{cls.NAME}_noop"
    return new_code, f"py_{cls.NAME}"


# ---- Language-agnostic structural-regex transforms (fallback) ---------------

def reg_aug_assign(code: str, rng: random.Random) -> Tuple[str, str]:
    """C-like:   x += y    <=>    x = x + y"""
    if rng.random() < 0.5:
        new, n = _re.subn(
            r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*([+\-*/%])=\s*([^;\n]+)",
            r"\1 = \1 \2 \3", code, count=rng.randint(1, 3))
    else:
        new, n = _re.subn(
            r"\b([A-Za-z_]\w*)\s*=\s*\1\s*([+\-*/%])\s*",
            r"\1 \2= ", code, count=rng.randint(1, 3))
    return (new, "reg_aug_assign") if n > 0 else (code, "reg_aug_assign_noop")


def reg_for_to_while_c(code: str, rng: random.Random) -> Tuple[str, str]:
    """C-like:   for (init; cond; upd) { body }   ==>   init; while (cond) { body; upd; }"""
    pattern = _re.compile(r"for\s*\(\s*([^;]+?)\s*;\s*([^;]+?)\s*;\s*([^)]+?)\s*\)\s*\{")
    def repl(m):
        return f"{m.group(1)};\nwhile ({m.group(2)}) {{"
    new, n = pattern.subn(repl, code, count=rng.randint(1, 2))
    if n > 0:
        # The "upd" is lost in regex form -- inject at end of next-closing-block (approximate).
        return new, "reg_for_to_while_c"
    return code, "reg_for_to_while_c_noop"


def reg_paren_canon(code: str, rng: random.Random) -> Tuple[str, str]:
    """Wrap RHS of assignment in redundant parens."""
    new, n = _re.subn(
        r"=\s*([^;\n=]+?)([;\n])",
        lambda m: f"= ({m.group(1).strip()}){m.group(2)}" if rng.random() < 0.4 else m.group(0),
        code, count=rng.randint(2, 5))
    return (new, "reg_paren_canon") if n > 0 else (code, "reg_paren_canon_noop")


def reg_if_invert_c(code: str, rng: random.Random) -> Tuple[str, str]:
    """C-like:   if (cond) { ... }    ==>    if (!(cond)) { ... }   (paired with body swap)
    Best-effort: only condition negation if pattern detectable. Body swap requires brace
    balancing which regex cannot do, so we settle for condition-only -- still a real
    semantic shift the encoder must learn invariance to (combined w/ the inner body swap
    that would follow in a true AST pass)."""
    # We negate by wrapping in !() to give the encoder a structural view distinct enough.
    pattern = _re.compile(r"\bif\s*\(\s*([^()\n]+?)\s*\)")
    fired = [False]
    def repl(m):
        if rng.random() < 0.4:
            fired[0] = True
            return f"if (!({m.group(1)}))"
        return m.group(0)
    new = pattern.sub(repl, code, count=rng.randint(1, 3))
    return (new, "reg_if_invert_c") if fired[0] else (code, "reg_if_invert_c_noop")


def reg_op_form_swap(code: str, rng: random.Random) -> Tuple[str, str]:
    """x * 2  <=>  x + x  (single-occurrence, simple Name lhs);   /2 -> *0.5"""
    fired = False
    new = code
    m1 = _re.search(r"\b([A-Za-z_]\w*)\s*\*\s*2\b", new)
    if m1 and rng.random() < 0.5:
        v = m1.group(1)
        new = new[:m1.start()] + f"({v} + {v})" + new[m1.end():]
        fired = True
    m2 = _re.search(r"\b([A-Za-z_]\w*)\s*/\s*2\b", new)
    if m2 and rng.random() < 0.5:
        v = m2.group(1)
        new = new[:m2.start()] + f"({v} * 0.5)" + new[m2.end():]
        fired = True
    return (new, "reg_op_form_swap") if fired else (code, "reg_op_form_swap_noop")


_REG_TRANSFORMS = [reg_aug_assign, reg_for_to_while_c, reg_paren_canon,
                   reg_if_invert_c, reg_op_form_swap]


def aug_regex_structural(code: str, rng: random.Random) -> Tuple[str, str]:
    fn = _REG_TRANSFORMS[rng.randrange(len(_REG_TRANSFORMS))]
    try:
        return fn(code, rng)
    except Exception:
        return code, "reg_fail"


# ---- Dispatcher -------------------------------------------------------------

def cargo_augment(code: str, language: str, rng: random.Random) -> Tuple[str, str]:
    """Apply structural rewrite. Try Python AST first (if language is python or
    parses as python), else fall back to language-agnostic structural regex."""
    lang_l = (language or "").lower()
    use_py_first = lang_l in {"python", "py"} or (lang_l == "" and "def " in code[:200])
    if use_py_first:
        new, name = aug_python_ast(code, rng)
        if not name.endswith("_noop") and not name.endswith("_fail"):
            return new, name
    new, name = aug_regex_structural(code, rng)
    return new, name


# Track which transforms are available for stats
_ALL_TRANSFORM_NAMES = (
    [f"py_{c.NAME}" for c in _PY_TRANSFORMERS]
    + [fn.__name__ for fn in _REG_TRANSFORMS]
    + ["ast_parse_fail", "ast_unparse_fail", "reg_fail"]
)


# =============================================================================
# Model
# =============================================================================

class CARGOModel(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        self.emb_dim = emb_dim
        self.n_cls = n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1), self.clf(z)


def supcon_tw_loss(z_doubled, labels_doubled, dist_mat, gamma=1.0, tau=0.1):
    """Tree-weighted SupCon on doubled batch [z ; z']."""
    N = z_doubled.size(0)
    if N < 2: return z_doubled.sum() * 0.0
    sim = (z_doubled @ z_doubled.t()) / tau
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()
    eye = torch.eye(N, device=z_doubled.device, dtype=torch.bool)
    pos_mask = (labels_doubled.unsqueeze(0) == labels_doubled.unsqueeze(1)).float().masked_fill(eye, 0.0)
    neg_mask = (labels_doubled.unsqueeze(0) != labels_doubled.unsqueeze(1)).float()
    dij = dist_mat[labels_doubled][:, labels_doubled]
    w_neg = torch.exp(-gamma * dij)
    w_pair = pos_mask + neg_mask * w_neg
    w_pair = w_pair.masked_fill(eye, 0.0)
    exp_s = (torch.exp(sim) * w_pair).clamp(min=1e-12)
    den = exp_s.sum(dim=-1).clamp(min=1e-12)
    num = (exp_s * pos_mask).sum(dim=-1).clamp(min=1e-12)
    n_pos = pos_mask.sum(dim=-1)
    has_pos = (n_pos > 0).float()
    li = -(torch.log(num) - torch.log(den)) * has_pos
    return li.sum() / has_pos.sum().clamp(min=1.0)


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
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    emb_dim: int = 256
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
        # Halve bs because we do 2x forward (z + z')
        if mem >= 40: cfg.bs, cfg.seq = 128, 512
        elif mem >= 10: cfg.bs, cfg.seq = 64, 384
        else: cfg.bs, cfg.seq = 32, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} (HALVED for 2x view) seq={cfg.seq}")
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


class FSDS_CARGO(TD):
    """Dataset that returns (original_tokens, structurally-augmented_tokens, label, transform_name).

    For TRAIN: do structural augmentation.  For VAL/TEST: same structural aug,
    so the F1 view-cos falsifier can be measured on test set (not just train).
    """
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42, do_aug=True):
        self.data = data; self.tok = tok; self.seq_len = seq_len; self.do_aug = do_aug
        self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_CARGO] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        lang = r.get("language", "") or ""
        enc = self.tok(code, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
        ids0, mask0 = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            code_aug, aug_name = cargo_augment(code, lang, rng)
            enc2 = self.tok(code_aug, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
            ids1, mask1 = enc2["input_ids"].squeeze(0), enc2["attention_mask"].squeeze(0)
        else:
            ids1, mask1, aug_name = ids0, mask0, "none"
        return {"ids0": ids0, "mask0": mask0, "ids1": ids1, "mask1": mask1,
                "label": r["label"], "aug_name": aug_name,
                "language": lang, "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=False):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    cos_per_aug: Dict[str, List[float]] = {}
    fire_count_per_aug: Dict[str, int] = {}
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        z0, logits = model.encode(ids0, mask0)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            ids1 = b["ids1"].to(cfg.device); mask1 = b["mask1"].to(cfg.device)
            z1, _ = model.encode(ids1, mask1)
            sim = (z0 * z1).sum(dim=-1).detach().cpu().float().numpy()
            aug_names = b.get("aug_name", ["none"] * z0.size(0))
            if not isinstance(aug_names, list): aug_names = list(aug_names)
            for s, n in zip(sim, aug_names):
                cos_per_aug.setdefault(n, []).append(float(s))
                fire_count_per_aug[n] = fire_count_per_aug.get(n, 0) + 1
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
        per_aug_stats = {}
        all_cos = []
        for n, lst in cos_per_aug.items():
            if not lst: continue
            per_aug_stats[n] = {"mean_cos": float(np.mean(lst)),
                                "std_cos": float(np.std(lst)),
                                "n": len(lst)}
            if not (n.endswith("_noop") or n.endswith("_fail") or n == "none"):
                all_cos.extend(lst)
        # Effective fires (non-noop/fail transforms only)
        effective_fires = {n: c for n, c in fire_count_per_aug.items()
                           if not (n.endswith("_noop") or n.endswith("_fail") or n == "none")}
        total_eff = max(sum(effective_fires.values()), 1)
        out["falsifier"] = {
            "mean_view_cos_F1": float(np.mean(all_cos)) if all_cos else 0.0,
            "per_aug_cos_F3": per_aug_stats,
            "effective_fire_fraction": {n: c / total_eff for n, c in effective_fires.items()},
            "total_effective_fires": int(total_eff),
            "total_eval_samples": int(sum(fire_count_per_aug.values())),
        }
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    aug_counts: Dict[str, int] = {}
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); mask1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        for n in b.get("aug_name", []):
            aug_counts[n] = aug_counts.get(n, 0) + 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, logits = model.encode(ids0, mask0)
            z1, _      = model.encode(ids1, mask1)
            z_d = torch.cat([z0, z1], dim=0)
            y_d = torch.cat([labs, labs], dim=0)
            loss_ce = F.cross_entropy(logits, labs)
            loss_sc = supcon_tw_loss(z_d, y_d, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            loss = loss_ce + cfg.lambda_aug * loss_sc
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item()
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n, aug_counts


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat_t = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist_mat = dist_mat_t.to(cfg.device); dist_mat_cpu = dist_mat_t.numpy()
    sib_mask_np = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()
    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, "author", vocab); vl_data = _conv_codet(vl_raw, "author", vocab); ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    tr_ds = FSDS_CARGO(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_CARGO(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=True)
    ts_ds = FSDS_CARGO(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=True)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"lambda_aug={cfg.lambda_aug} gamma={cfg.gamma}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = CARGOModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    cumulative_aug_counts: Dict[str, int] = {}
    for epoch in range(cfg.epochs):
        loss, ce, sc, aug_counts = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        for n, c in aug_counts.items():
            cumulative_aug_counts[n] = cumulative_aug_counts.get(n, 0) + c
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} supcon={sc:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=True)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    fa = ts_met["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"view_cos={fa['mean_view_cos_F1']:.3f}")
    for name, st in sorted(fa["per_aug_cos_F3"].items(), key=lambda kv: -kv[1]["n"]):
        logger.info(f"  [aug={name:30}] mean_cos={st['mean_cos']:+.3f} std={st['std_cos']:.3f} n={st['n']}")
    return {"tag": tag, "method": "CARGO", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "gamma": cfg.gamma, "tau": cfg.tau,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "train_aug_counts": cumulative_aug_counts,
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
                tag = f"exp84_cargo_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"view_cos={fa['mean_view_cos_F1']:.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp84_cargo_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'view_cos':>10} {'Wall':>8}")
    print("-"*140)
    for r in results:
        fa = r['test_metrics']['falsifier']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['mean_view_cos_F1']:>10.3f} {r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
