# exp107 — ECHO
# NAME       : ECHO (Ensemble disagreement as pseudo-label oracle)
# REFERENCE  : new; co-training (Blum & Mitchell 1998), tri-training
#              (Zhou & Li 2005) — but using DISAGREEMENT as positive
#              signal instead of confident agreement.
# CLAIM      : The training set is too small at 1%. The UNLABELED pool
#              (the rest of the training data, with labels HIDDEN) is
#              large. Train K=3 TRACO models with different augmentation
#              seeds. On the unlabeled pool, where the K models AGREE,
#              we have a confident pseudo-label. Add high-agreement
#              samples to the labeled pool and retrain. We never invoke
#              the ground truth on the unlabeled samples.
# EQUATION   : K=3 TRACO models {phi_k}, each trained on the same
#              labeled pool with different augmentation seeds.
#              For each unlabeled x:
#                votes = [argmax phi_k(x) for k in 1..K]
#                if all(v == votes[0] for v in votes):
#                    pseudo_label x with votes[0]
#                    add (x, votes[0]) to labeled pool
#              Retrain K models on enlarged pool. Repeat R=2 rounds.
# WHY NEW    : TRACOD (exp80) used EMA momentum and collapsed. ECHO uses
#              ensemble DISAGREEMENT signal: when the ensemble agrees,
#              we trust the label; when it disagrees, we don't. This
#              filters pseudo-labels by ensemble consensus, not by single-
#              model confidence.
# WOW HOOK   : "We mine the ensemble's agreement as a free labeling
#              oracle. The training set GROWS during training, without
#              any human annotation, simply by collecting cases where
#              three models concur."
# FALSIFIER  : (F1) Round 2 pseudo-label pool size > 5% of unlabeled
#              pool (signal exists). (F2) Pseudo-label accuracy > 0.90
#              on the agreed subset (high agreement = high accuracy).
#              (F3) Composite > best single TRACO model by >= 0.005.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math, copy
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
logger = logging.getLogger("exp107_echo")

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
# Augmentation (copied from exp84_cargo.py)
# =============================================================================
import ast as _ast
import re as _re


class _AugAssignExpand(_ast.NodeTransformer):
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
    NAME = "op_form_swap"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if (isinstance(node.op, _ast.Mult) and isinstance(node.right, _ast.Constant)
            and node.right.value == 2 and isinstance(node.left, _ast.Name)
            and self.rng.random() < 0.6):
            self.fired += 1
            return _ast.copy_location(
                _ast.BinOp(left=node.left, op=_ast.Add(), right=node.left), node)
        if (isinstance(node.op, _ast.Div) and isinstance(node.right, _ast.Constant)
            and node.right.value == 2 and self.rng.random() < 0.6):
            self.fired += 1
            return _ast.copy_location(
                _ast.BinOp(left=node.left, op=_ast.Mult(), right=_ast.Constant(value=0.5)),
                node)
        return node


_PY_TRANSFORMERS = [_AugAssignExpand, _IfInvert, _ForToWhile, _ListCompUnroll, _OpFormSwap]


def aug_python_ast(code: str, rng: random.Random) -> Tuple[str, str]:
    try:
        _ast.parse(code)
    except (SyntaxError, ValueError):
        return code, "ast_parse_fail"
    order = list(range(len(_PY_TRANSFORMERS)))
    rng.shuffle(order)
    last_name = ""
    for idx in order:
        cls = _PY_TRANSFORMERS[idx]
        transformer = cls(rng)
        try:
            new_tree = transformer.visit(_ast.parse(code))
            _ast.fix_missing_locations(new_tree)
            new_code = _ast.unparse(new_tree)
        except Exception:
            last_name = f"py_{cls.NAME}_unparse_fail"
            continue
        if transformer.fired > 0:
            return new_code, f"py_{cls.NAME}"
        last_name = f"py_{cls.NAME}_noop"
    return code, last_name or "ast_all_noop"


def reg_aug_assign(code: str, rng: random.Random) -> Tuple[str, str]:
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
    pattern = _re.compile(r"for\s*\(\s*([^;]+?)\s*;\s*([^;]+?)\s*;\s*([^)]+?)\s*\)\s*\{")
    def repl(m):
        return f"{m.group(1)};\nwhile ({m.group(2)}) {{"
    new, n = pattern.subn(repl, code, count=rng.randint(1, 2))
    if n > 0:
        return new, "reg_for_to_while_c"
    return code, "reg_for_to_while_c_noop"


def reg_paren_canon(code: str, rng: random.Random) -> Tuple[str, str]:
    new, n = _re.subn(
        r"=\s*([^;\n=]+?)([;\n])",
        lambda m: f"= ({m.group(1).strip()}){m.group(2)}" if rng.random() < 0.4 else m.group(0),
        code, count=rng.randint(2, 5))
    return (new, "reg_paren_canon") if n > 0 else (code, "reg_paren_canon_noop")


def reg_if_invert_c(code: str, rng: random.Random) -> Tuple[str, str]:
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
    order = list(range(len(_REG_TRANSFORMS)))
    rng.shuffle(order)
    for idx in order:
        try:
            new, name = _REG_TRANSFORMS[idx](code, rng)
        except Exception:
            continue
        if not (name.endswith("_noop") or name.endswith("_fail")):
            return new, name
    return code, "reg_all_noop"


def fallback_ws_normalize(code: str, rng: random.Random) -> Tuple[str, str]:
    ops = "+-*/%=<>,;()[]{}|&^~!"
    out = []
    fired = False
    for c in code:
        out.append(c)
        if c in ops and rng.random() < 0.20:
            out.append(" ")
            fired = True
    if not fired and code:
        return code + "\n", "fallback_newline"
    return "".join(out), "fallback_ws_norm"


def cargo_augment(code: str, language: str, rng: random.Random) -> Tuple[str, str]:
    lang_l = (language or "").lower()
    use_py_first = lang_l in {"python", "py"} or (lang_l == "" and "def " in code[:200])
    if use_py_first:
        new, name = aug_python_ast(code, rng)
        if not (name.endswith("_noop") or name.endswith("_fail")
                or name == "ast_all_noop" or name == "ast_parse_fail"):
            return new, name
    new, name = aug_regex_structural(code, rng)
    if name != "reg_all_noop" and not name.endswith("_fail"):
        return new, name
    return fallback_ws_normalize(code, rng)


def supcon_tw_loss(z_doubled, labels_doubled, dist_mat, gamma=1.0, tau=0.1):
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
# Model
# =============================================================================

class TRACOModel(nn.Module):
    """Standard TRACO architecture (copy from exp84)."""
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        self.emb_dim = emb_dim
        self.n_cls = n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1), self.clf(z)


# =============================================================================
# Plumbing
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 128; seq: int = 512; epochs: int = 6
    lr_enc: float = 3e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    emb_dim: int = 256
    K: int = 3              # ensemble size
    R: int = 2              # rounds of pseudo-labeling
    unlabeled_mult: int = 10  # unlabeled pool size = unlabeled_mult * frac of full train
    device: str = "cuda"; gene_adj: dict = field(default_factory=dict)


def adaptive_schedule(cfg, round_idx=0):
    """Per-round schedule. Round 0 = full schedule; later rounds shrink to keep cost bounded."""
    f = cfg.frac
    if round_idx == 0:
        if f <= 0.02: cfg.epochs, cfg.lr_enc, cfg.warmup = 10, 3e-5, 0.20
        elif f <= 0.10: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 3e-5, 0.15
        else: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 4e-5, 0.10
    elif round_idx == 1:
        cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 3e-5, 0.10
    else:
        cfg.epochs, cfg.lr_enc, cfg.warmup = 4, 3e-5, 0.10
    return cfg


def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
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


# =============================================================================
# Label-pool dataset
#
# We hold the FULL training set in memory.  We maintain:
#   - labeled_indices : list[int] indices into full train + their TRUE labels
#   - pseudo_indices  : list[(int, int)] indices + their PSEUDO labels (assigned by ensemble agreement)
#   - unlabeled_pool  : list[int] candidate indices that may yet be pseudo-labeled
# At each round, the labeled training data = labeled_indices (true labels)
#   UNION pseudo_indices (pseudo labels).
# =============================================================================

class IndexedFSDS(TD):
    """Dataset over arbitrary indices of the full train data, with per-sample labels
    (potentially pseudo) supplied externally.

    items: list of (idx_into_data, label).
    """
    def __init__(self, full_data, items, tok, seq_len, seed=42, do_aug=True):
        self.full_data = full_data
        self.items = items
        self.tok = tok; self.seq_len = seq_len
        self.do_aug = do_aug; self.seed = seed

    def __len__(self): return len(self.items)

    def __getitem__(self, i):
        idx, label = self.items[i]
        r = self.full_data[idx]
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
                "label": int(label), "aug_name": aug_name,
                "language": lang, "source": r.get("source", "") or "",
                "_data_idx": int(idx)}


class EvalFSDS(TD):
    """Val/test dataset (no aug)."""
    def __init__(self, data, tok, seq_len):
        self.data = data; self.tok = tok; self.seq_len = seq_len

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        lang = r.get("language", "") or ""
        enc = self.tok(code, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
        ids0, mask0 = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        return {"ids0": ids0, "mask0": mask0, "ids1": ids0, "mask1": mask0,
                "label": int(r["label"]), "aug_name": "none",
                "language": lang, "source": r.get("source", "") or ""}


def stratified_label_split(full_data, frac, n_cls, unlabeled_mult, seed=42):
    """Stratified sample by class.
    Returns:
        labeled_idx: per-class fraction `frac` of full_data
        unlabeled_idx: subsample of remaining, sized ~ unlabeled_mult * len(labeled_idx) (capped at remaining)
    """
    rng = random.Random(seed)
    labels = list(full_data["label"])
    labeled_idx, unlabeled_idx = [], []
    for lbl in range(n_cls):
        idx = [i for i, x in enumerate(labels) if x == lbl]
        rng.shuffle(idx)
        n_lab = min(max(1, int(len(idx) * frac)), len(idx))
        labeled_idx.extend(idx[:n_lab])
        unlabeled_idx.extend(idx[n_lab:])
    rng.shuffle(unlabeled_idx)
    target_unlab = min(len(unlabeled_idx), unlabeled_mult * len(labeled_idx))
    unlabeled_idx = unlabeled_idx[:target_unlab]
    logger.info(f"[split] labeled={len(labeled_idx)} unlabeled_pool={len(unlabeled_idx)} "
                f"(unlabeled_mult={unlabeled_mult})")
    return labeled_idx, unlabeled_idx


# =============================================================================
# Eval pack
# =============================================================================

@torch.no_grad()
def eval_pack_one(model, loader, cfg, sib_mask_np, dist_mat_cpu):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        _, logits = model.encode(ids0, mask0)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        lang_batch = b.get("language", [""] * len(labs))
        src_batch = b.get("source", [""] * len(labs))
        langs.extend(list(lang_batch) if not isinstance(lang_batch, list) else lang_batch)
        sources.extend(list(src_batch) if not isinstance(src_batch, list) else src_batch)
    return _summarize_preds(np.array(preds), np.array(labels), np.array(langs), np.array(sources),
                            cfg.n_cls, sib_mask_np, dist_mat_cpu)


@torch.no_grad()
def eval_pack_ensemble(models, loader, cfg, sib_mask_np, dist_mat_cpu):
    """Ensemble eval: average softmax across K models, then argmax."""
    for m in models: m.eval()
    all_preds_logits = None
    labels, langs, sources = [], [], []
    for b in tqdm(loader, desc="Eval-ens"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        probs_sum = None
        for m in models:
            _, lg = m.encode(ids0, mask0)
            p = F.softmax(lg, dim=-1)
            probs_sum = p if probs_sum is None else probs_sum + p
        probs_sum = probs_sum / len(models)
        preds_b = probs_sum.argmax(dim=-1).cpu().numpy()
        if all_preds_logits is None:
            all_preds_logits = preds_b
        else:
            all_preds_logits = np.concatenate([all_preds_logits, preds_b])
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        lang_batch = b.get("language", [""] * len(labs))
        src_batch = b.get("source", [""] * len(labs))
        langs.extend(list(lang_batch) if not isinstance(lang_batch, list) else lang_batch)
        sources.extend(list(src_batch) if not isinstance(src_batch, list) else src_batch)
    return _summarize_preds(np.array(all_preds_logits), np.array(labels),
                            np.array(langs), np.array(sources),
                            cfg.n_cls, sib_mask_np, dist_mat_cpu)


def _summarize_preds(preds, labels, langs, sources, n_cls, sib_mask_np, dist_mat_cpu):
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
    if langs.size > 0 and any(l for l in langs):
        for L in sorted(set(langs.tolist())):
            if not L: continue
            sel = (langs == L)
            if sel.sum() < 2: continue
            per_lang[L] = {"n": int(sel.sum()),
                           "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0)),
                           "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                           "accuracy": float(accuracy_score(labels[sel], preds[sel]))}
    if sources.size > 0 and any(s for s in sources):
        for S in sorted(set(sources.tolist())):
            if not S: continue
            sel = (sources == S)
            if sel.sum() < 2: continue
            per_src[S] = {"n": int(sel.sum()),
                          "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0)),
                          "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                          "accuracy": float(accuracy_score(labels[sel], preds[sel]))}
    return {"overall": overall, "per_class": per_class, "per_language": per_lang, "per_source": per_src,
            "confusion_matrix": cm.tolist(), "sibling_confusion_rate": float(sib_rate),
            "cross_family_confusion_rate": float(cross_rate),
            "off_diag_total": off_diag, "n_samples": int(len(labels))}


# =============================================================================
# Train one TRACO model on (labeled + pseudo-labeled) pool
# =============================================================================

def train_epoch_traco(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); mask1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
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
    n = max(1, len(loader))
    return tot/n, ce_s/n, sc_s/n


def train_one_traco(cfg, full_tr_data, items, vl_dl, dist_mat, sib_mask_np, dist_mat_cpu, model_seed, tok):
    """Train one TRACO model on the given items (list of (idx, label))."""
    set_seed(model_seed)
    tr_ds = IndexedFSDS(full_tr_data, items, tok, cfg.seq, seed=model_seed, do_aug=True)
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    total_steps = max(1, len(tr_dl)) * cfg.epochs
    model = TRACOModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, ce, sc = train_epoch_traco(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        vm = eval_pack_one(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = vm["overall"]["macro_f1"]; hist.append(v)
        logger.info(f"[member seed={model_seed} e{epoch+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val, hist


# =============================================================================
# Ensemble agreement on unlabeled pool
# =============================================================================

@torch.no_grad()
def ensemble_agree(models, full_tr_data, unlabeled_idx, cfg, tok):
    """Return:
        agreed: list of (idx, agreed_label) where all K models predict same
        true_labels_on_agreed: list of true labels (for accuracy report only; never used in training)
    """
    # Build a fake dataset over unlabeled indices, dummy label = 0
    dummy_items = [(i, 0) for i in unlabeled_idx]
    ds = IndexedFSDS(full_tr_data, dummy_items, tok, cfg.seq, seed=cfg.seed + 99, do_aug=False)
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    dl = DataLoader(ds, shuffle=False, **loader_cfg)

    K = len(models)
    preds_per_model: List[List[int]] = [[] for _ in range(K)]
    data_indices: List[int] = []
    for b in tqdm(dl, desc="Ens-agree"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        di = b["_data_idx"].cpu().tolist() if torch.is_tensor(b["_data_idx"]) else list(b["_data_idx"])
        data_indices.extend(di)
        for k, m in enumerate(models):
            m.eval()
            _, lg = m.encode(ids0, mask0)
            preds_per_model[k].extend(lg.argmax(dim=-1).cpu().tolist())

    agreed = []
    true_labels_on_agreed = []
    full_labels = list(full_tr_data["label"])
    for j, idx in enumerate(data_indices):
        votes = [preds_per_model[k][j] for k in range(K)]
        if len(set(votes)) == 1:
            agreed.append((idx, votes[0]))
            true_labels_on_agreed.append(full_labels[idx])
    return agreed, true_labels_on_agreed


# =============================================================================
# run_exp
# =============================================================================

def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg)
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

    vl_ds = EvalFSDS(vl_data, tok, cfg.seq)
    ts_ds = EvalFSDS(ts_data, tok, cfg.seq)
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    # Stratified split
    labeled_idx, unlabeled_idx = stratified_label_split(
        tr_data, cfg.frac, cfg.n_cls, cfg.unlabeled_mult, seed=cfg.seed)
    full_labels = list(tr_data["label"])
    # items: list of (idx, label) — initially only true-labeled
    labeled_items: List[Tuple[int, int]] = [(i, full_labels[i]) for i in labeled_idx]
    pseudo_items: List[Tuple[int, int]] = []

    # Per-round bookkeeping
    pseudo_pool_size_per_round: List[int] = []
    pseudo_label_accuracy_per_round: List[float] = []
    member_val_macros_per_round: List[List[float]] = []
    member_test_macros_per_round: List[List[float]] = []
    ensemble_val_per_round: List[float] = []
    ensemble_test_per_round: List[float] = []
    round_size_per_round: List[int] = []
    val_hist_round0: List[float] = []

    models: List[nn.Module] = []
    K = cfg.K
    R = cfg.R

    for r in range(R + 1):
        cfg_r = copy.deepcopy(cfg)
        cfg_r = adaptive_schedule(cfg_r, round_idx=r)
        items_r = labeled_items + pseudo_items
        round_size_per_round.append(len(items_r))
        logger.info(f"[round {r}] training K={K} TRACO models on |items|={len(items_r)} "
                    f"(true_labeled={len(labeled_items)} pseudo={len(pseudo_items)}) "
                    f"epochs={cfg_r.epochs}")
        # Free previous round's models before training new ones
        models = []
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()

        # Train K members with different seeds
        member_val_macros: List[float] = []
        for k in range(K):
            model_seed = cfg.seed + 1000 * (r + 1) + k * 13
            m, vmac, hist = train_one_traco(cfg_r, tr_data, items_r, vl_dl,
                                            dist_mat, sib_mask_np, dist_mat_cpu,
                                            model_seed, tok)
            models.append(m)
            member_val_macros.append(vmac)
            if r == 0 and k == 0:
                val_hist_round0 = hist
        member_val_macros_per_round.append(member_val_macros)

        # Per-member test
        member_test_macros: List[float] = []
        member_test_packs: List[dict] = []
        for k, m in enumerate(models):
            tm = eval_pack_one(m, ts_dl, cfg_r, sib_mask_np, dist_mat_cpu)
            member_test_macros.append(tm["overall"]["macro_f1"])
            member_test_packs.append(tm)
        member_test_macros_per_round.append(member_test_macros)

        # Ensemble val & test
        ens_val_pack = eval_pack_ensemble(models, vl_dl, cfg_r, sib_mask_np, dist_mat_cpu)
        ens_test_pack = eval_pack_ensemble(models, ts_dl, cfg_r, sib_mask_np, dist_mat_cpu)
        ensemble_val_per_round.append(ens_val_pack["overall"]["macro_f1"])
        ensemble_test_per_round.append(ens_test_pack["overall"]["macro_f1"])
        logger.info(f"[round {r}] members_val={[f'{x:.4f}' for x in member_val_macros]} "
                    f"ens_val={ens_val_pack['overall']['macro_f1']:.4f} "
                    f"ens_test={ens_test_pack['overall']['macro_f1']:.4f}")

        # Pseudo-label step (skip after final round)
        if r < R:
            # Determine currently un-pseudo-labeled indices in unlabeled_idx
            used_idx_set = {idx for idx, _ in pseudo_items}
            candidate_idx = [i for i in unlabeled_idx if i not in used_idx_set]
            logger.info(f"[round {r}] mining ensemble agreement on |candidates|={len(candidate_idx)}")
            agreed, true_on_agreed = ensemble_agree(models, tr_data, candidate_idx, cfg_r, tok)
            # Accuracy of pseudo-labels (vs ground truth, for reporting only — never used to filter)
            if agreed:
                acc_pl = float(np.mean([al == tl for (_, al), tl in zip(agreed, true_on_agreed)]))
            else:
                acc_pl = 0.0
            pseudo_pool_size_per_round.append(len(agreed))
            pseudo_label_accuracy_per_round.append(acc_pl)
            logger.info(f"[round {r}] agreed={len(agreed)} ({len(agreed)/max(1,len(candidate_idx)):.3f} "
                        f"of candidates), pseudo-acc={acc_pl:.4f}")
            # Add agreed to pseudo_items
            pseudo_items = pseudo_items + agreed

    # Final ensemble metrics (last round)
    final_test_pack = eval_pack_ensemble(models, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    final_val_pack = eval_pack_ensemble(models, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
    ensemble_macro = final_test_pack["overall"]["macro_f1"]
    ensemble_val = final_val_pack["overall"]["macro_f1"]
    single_best_test_macro = max(member_test_macros_per_round[-1])
    single_best_val_macro = max(member_val_macros_per_round[-1])
    gap = ensemble_val - ensemble_macro

    fa = {
        "pseudo_pool_size_per_round_F1": pseudo_pool_size_per_round,
        "unlabeled_pool_size": len(unlabeled_idx),
        "pseudo_pool_fraction_per_round": [
            x / max(1, len(unlabeled_idx)) for x in pseudo_pool_size_per_round],
        "pseudo_label_accuracy_per_round_F2": pseudo_label_accuracy_per_round,
        "single_best_test_macro": float(single_best_test_macro),
        "single_best_val_macro": float(single_best_val_macro),
        "ensemble_minus_single_best_F3": float(ensemble_macro - single_best_test_macro),
        "member_val_macros_per_round": member_val_macros_per_round,
        "member_test_macros_per_round": member_test_macros_per_round,
        "ensemble_val_per_round": ensemble_val_per_round,
        "ensemble_test_per_round": ensemble_test_per_round,
        "round_size_per_round": round_size_per_round,
        "K": K, "R": R,
    }
    final_test_pack["falsifier"] = fa

    logger.info(f"[final] ens_val={ensemble_val:.4f} ens_test={ensemble_macro:.4f} "
                f"single_best_test={single_best_test_macro:.4f} "
                f"delta_ens-single={ensemble_macro - single_best_test_macro:+.4f} "
                f"pseudo_pool_per_round={pseudo_pool_size_per_round} "
                f"pseudo_acc_per_round={[f'{x:.3f}' for x in pseudo_label_accuracy_per_round]}")

    return {"tag": tag, "method": "ECHO",
            "note": f"K={K} ensemble + R={R} rounds of agreement-based pseudo-labeling",
            "enc": cfg.enc, "bench": cfg.benchmark, "frac": cfg.frac,
            "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "K": K, "R": R, "unlabeled_mult": cfg.unlabeled_mult,
            "val_macro": ensemble_val, "macro": ensemble_macro,
            "weighted": final_test_pack["overall"]["weighted_f1"],
            "acc": final_test_pack["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": ensemble_macro - PAPER_BASELINE,
            "single_best_test_macro": float(single_best_test_macro),
            "single_best_val_macro": float(single_best_val_macro),
            "test_metrics": final_test_pack,
            "val_history": val_hist_round0,
            "ensemble_val_per_round": ensemble_val_per_round,
            "ensemble_test_per_round": ensemble_test_per_round,
            "pseudo_pool_size_per_round": pseudo_pool_size_per_round,
            "pseudo_label_accuracy_per_round": pseudo_label_accuracy_per_round,
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
                tag = f"exp107_echo_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"single_best={res['single_best_test_macro']:.4f} "
                                f"delta={fa['ensemble_minus_single_best_F3']:+.4f} "
                                f"pseudo_pool={fa['pseudo_pool_size_per_round_F1']} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp107_echo_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*170)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'SingleBest':>11} {'d(ens-sb)':>10} {'PseudoR1':>10} {'PseudoAccR1':>12} {'Wall':>8}")
    print("-"*170)
    for r in results:
        fa = r['test_metrics']['falsifier']
        ps_r1 = fa['pseudo_pool_size_per_round_F1'][0] if fa['pseudo_pool_size_per_round_F1'] else 0
        pa_r1 = fa['pseudo_label_accuracy_per_round_F2'][0] if fa['pseudo_label_accuracy_per_round_F2'] else 0.0
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['single_best_test_macro']:>11.4f} {fa['ensemble_minus_single_best_F3']:>+10.4f} "
              f"{ps_r1:>10d} {pa_r1:>12.4f} {r['wall']:>8.0f}s")
    print("="*170)


if __name__ == "__main__":
    main()
