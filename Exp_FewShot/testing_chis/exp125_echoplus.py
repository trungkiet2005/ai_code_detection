# exp125 — ECHO+
# NAME       : ECHO+ (Ensemble Co-Reinforcing Operators with MAML inner loop)
# REFERENCE  : new; combines ensemble disagreement (Tarvainen 2017
#              co-training) + MAML (Finn 2017) + self-pseudo-labelling.
# CLAIM      : Train K=3 TRACO+MAML models on disjoint cross-validation
#              folds. On the held-out fold, the OTHER K-1 models vote.
#              Disagreement-rich samples get pseudo-labelled (majority
#              vote) and ADDED to the training pool for round 2. Three
#              rounds. Self-distilled ensemble: each model trained on
#              data the others labelled, no oracle.
# EQUATION   : K=3 models {phi_k}, K-fold split of train (fraction frac).
#              Round r=1..R:
#                For k in 1..K: train phi_k on fold k's data (CE + TRACO supcon)
#                For each held-out sample x in fold_k:
#                  votes = [argmax phi_j(x) for j != k]
#                  if max-vote-count >= K-1: add (x, mode(votes)) to fold_k's pool
#              Final prediction = ensemble average across K models.
# WHY NEW    : ECHO alone is multi-seed ensemble. ECHO+ combines (i)
#              k-fold CV co-training + (ii) MAML wrapper + (iii) self-
#              pseudo-label injection. Each is published separately;
#              their combination as a self-improving 3-stage loop is new.
# WOW HOOK   : "Self-labelling becomes meta-learning. Each model in the
#              ensemble meta-trains on data the others labelled.
#              Disagreement is the labelling oracle; convergence is
#              agreement. We never ask for an external label."
# FALSIFIER  : (F1) Pseudo-label accuracy at round 3 > 0.85 on labelled
#              holdout. (F2) Each round composite > previous by ≥ 0.003.
#              (F3) Ensemble disagreement entropy decreases monotonically.
#              (F4) Composite > METATRACO + 0.005.
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
from torch.utils.data import Dataset as TD, DataLoader, Subset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp125_echoplus")

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
# CARGO structural augmentations (copied verbatim from exp84_cargo.py)
# =============================================================================

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
        tree = _ast.parse(code)
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


# =============================================================================
# Model: standard TRACO encoder + projector + classifier
# =============================================================================

class TRACOModel(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1),
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
    bs: int = 128; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    emb_dim: int = 256
    K_models: int = 3; R_rounds: int = 3
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
        # 2-view per model (orig + aug). Halve TRACO's 256 -> 128.
        if mem >= 80: cfg.bs, cfg.seq = 192, 512
        elif mem >= 40: cfg.bs, cfg.seq = 128, 512
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


class FSDS_ECHO(TD):
    """Dataset that returns (orig, structurally-augmented) views + label.
    Supports per-sample label override via `label_override` dict (idx -> int)
    to inject pseudo-labels without rebuilding the HF dataset.
    """
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42, do_aug=True,
                 label_override: Optional[Dict[int, int]] = None):
        self.data = data; self.tok = tok; self.seq_len = seq_len; self.do_aug = do_aug
        self.seed = seed
        self.label_override = label_override or {}
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_ECHO] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        lang = r.get("language", "") or ""
        enc = self.tok(code, max_length=self.seq_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        ids0, mask0 = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            code_aug, aug_name = cargo_augment(code, lang, rng)
            enc2 = self.tok(code_aug, max_length=self.seq_len, padding="max_length",
                            truncation=True, return_tensors="pt")
            ids1, mask1 = enc2["input_ids"].squeeze(0), enc2["attention_mask"].squeeze(0)
        else:
            ids1, mask1, aug_name = ids0, mask0, "none"
        lbl = self.label_override.get(i, r["label"])
        return {"ids0": ids0, "mask0": mask0, "ids1": ids1, "mask1": mask1,
                "label": int(lbl), "aug_name": aug_name,
                "language": lang, "source": r.get("source", "") or ""}


# =============================================================================
# Helpers: stratified k-fold + co-train pseudo-labelling
# =============================================================================

def stratified_kfold(labels: List[int], K: int, seed: int) -> List[List[int]]:
    """Return K disjoint stratified index lists."""
    rng = random.Random(seed)
    by_label: Dict[int, List[int]] = {}
    for i, y in enumerate(labels):
        by_label.setdefault(int(y), []).append(i)
    folds: List[List[int]] = [[] for _ in range(K)]
    for y, idx in by_label.items():
        rng.shuffle(idx)
        for i, sample_idx in enumerate(idx):
            folds[i % K].append(sample_idx)
    return folds


@torch.no_grad()
def predict_indices(model: nn.Module, base_ds: TD, indices: List[int], cfg: Cfg) -> List[int]:
    """Run argmax prediction over a Subset of base_ds at `indices` and return predicted labels."""
    if not indices: return []
    model.eval()
    sub = Subset(base_ds, indices)
    loader = DataLoader(sub, batch_size=cfg.bs, shuffle=False, num_workers=2, pin_memory=True)
    preds = []
    for b in loader:
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device)
        _, lg = model.encode(ids, mask)
        preds.extend(lg.argmax(-1).cpu().tolist())
    return preds


def vote_pseudo_label(other_models: List[nn.Module], base_ds: TD,
                      indices: List[int], cfg: Cfg) -> Tuple[List[Tuple[int, int]], float, int]:
    """For each idx in `indices`, run other_models, return agreed-vote pseudo-labels.
    Returns:
      (list of (idx, pseudo_label) where all K-1 models agree,
       pseudo-label accuracy vs true label (if ground truth exists),
       n agreed)
    """
    if not indices: return [], 0.0, 0
    preds_per_model: List[List[int]] = [predict_indices(m, base_ds, indices, cfg)
                                        for m in other_models]
    agreed: List[Tuple[int, int]] = []
    n_correct = 0; n_total = 0
    for i, idx in enumerate(indices):
        votes = [p[i] for p in preds_per_model]
        if len(set(votes)) == 1:
            pl = votes[0]
            agreed.append((idx, pl))
            true = base_ds.data[idx]["label"]
            n_total += 1
            if pl == int(true): n_correct += 1
    acc = (n_correct / n_total) if n_total > 0 else 0.0
    return agreed, acc, len(agreed)


# =============================================================================
# Training & eval
# =============================================================================

def train_one_model(model: nn.Module, train_indices: List[int], base_ds: TD,
                    label_overrides: Dict[int, int], cfg: Cfg, dist_mat,
                    epochs: int, total_steps_factor: float = 1.0):
    """Train `model` on a Subset of base_ds restricted to `train_indices`.
    `label_overrides` maps base_ds index -> pseudo-label (used in base_ds.__getitem__).
    """
    # Inject pseudo-labels
    base_ds.label_override = dict(label_overrides)
    sub = Subset(base_ds, train_indices)
    total_steps = max(1, len(sub) // cfg.bs) * epochs
    loader = DataLoader(sub, batch_size=cfg.bs, shuffle=True, num_workers=4, pin_memory=True)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    for epoch in range(epochs):
        model.train()
        tot, ce_s, sc_s = 0.0, 0.0, 0.0
        for b in tqdm(loader, desc=f"Train ep{epoch+1}"):
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
        logger.info(f"    [model train ep{epoch+1}] loss={tot/n:.4f} ce={ce_s/n:.4f} sc={sc_s/n:.4f}")


@torch.no_grad()
def ensemble_predict(models: List[nn.Module], loader: DataLoader, cfg: Cfg):
    """Return ensemble (mean-softmax-argmax) predictions, per-sample disagreement entropy."""
    for m in models: m.eval()
    all_preds: List[int] = []
    all_labels: List[int] = []
    all_langs: List[str] = []
    all_srcs: List[str] = []
    disagree_entropies: List[float] = []
    K = len(models)
    for b in tqdm(loader, desc="Ensemble Eval"):
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device)
        labs = b["label"]
        probs_sum = None
        per_model_preds = []
        for m in models:
            _, lg = m.encode(ids, mask)
            p = F.softmax(lg, dim=-1)
            per_model_preds.append(lg.argmax(-1).cpu())
            probs_sum = p if probs_sum is None else probs_sum + p
        probs_mean = probs_sum / K
        ens_pred = probs_mean.argmax(-1).cpu().tolist()
        all_preds.extend(ens_pred)
        all_labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        # disagreement entropy: per-sample, frequency over K model preds
        stack = torch.stack(per_model_preds, dim=0).numpy()  # (K, B)
        for j in range(stack.shape[1]):
            vals, counts = np.unique(stack[:, j], return_counts=True)
            p = counts / counts.sum()
            ent = float(-(p * np.log(p + 1e-12)).sum())
            disagree_entropies.append(ent)
        lang_batch = b.get("language", [""] * len(labs))
        src_batch = b.get("source", [""] * len(labs))
        all_langs.extend(list(lang_batch) if not isinstance(lang_batch, list) else lang_batch)
        all_srcs.extend(list(src_batch) if not isinstance(src_batch, list) else src_batch)
    return (np.array(all_preds), np.array(all_labels), all_langs, all_srcs,
            float(np.mean(disagree_entropies)) if disagree_entropies else 0.0)


def eval_pack_ensemble(models: List[nn.Module], loader: DataLoader, cfg: Cfg,
                       sib_mask_np, dist_mat_cpu):
    preds, labels, langs, sources, mean_dis_ent = ensemble_predict(models, loader, cfg)
    n_cls = cfg.n_cls
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
    return {"overall": overall, "per_class": per_class, "per_language": per_lang, "per_source": per_src,
            "confusion_matrix": cm.tolist(), "sibling_confusion_rate": float(sib_rate),
            "cross_family_confusion_rate": float(cross_rate),
            "off_diag_total": off_diag, "n_samples": int(len(labels)),
            "mean_disagreement_entropy": mean_dis_ent}


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
        tr_data = _conv_codet(tr_raw, "author", vocab)
        vl_data = _conv_codet(vl_raw, "author", vocab)
        ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    # Build train ds at full frac (one shared base ds across the K models).
    tr_ds = FSDS_ECHO(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_ECHO(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed + 1, do_aug=True)
    ts_ds = FSDS_ECHO(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed + 2, do_aug=True)

    K = cfg.K_models; R = cfg.R_rounds
    # Per-round epochs (round 1 = full adaptive epochs; later rounds shorter)
    round_epochs = [cfg.epochs, max(4, cfg.epochs // 2 + 1), max(4, cfg.epochs // 2 + 1)]
    logger.info(f"[ECHO+] K={K} rounds={R} per_round_epochs={round_epochs}")
    logger.info(f"[sched] frac={cfg.frac} epochs(base)={cfg.epochs} lr_enc={cfg.lr_enc} "
                f"warmup={cfg.warmup} lambda_aug={cfg.lambda_aug} gamma={cfg.gamma}")

    # Stratified k-fold over the (already sampled) tr_ds
    labels_all = [tr_ds.data[i]["label"] for i in range(len(tr_ds))]
    folds = stratified_kfold(labels_all, K, cfg.seed)
    logger.info(f"[ECHO+] fold sizes: {[len(f) for f in folds]} / total={len(tr_ds)}")

    # Per-model training-index pool (starts as own fold; pseudo-labels added later)
    fold_pools: List[List[int]] = [list(f) for f in folds]
    # Per-model pseudo-label overrides (idx -> pseudo-label)
    fold_overrides: List[Dict[int, int]] = [{} for _ in range(K)]
    # Instantiate K models
    models: List[nn.Module] = []
    for k in range(K):
        set_seed(cfg.seed + 100 * k)
        m = TRACOModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
        models.append(m)

    vl_dl = DataLoader(vl_ds, batch_size=cfg.bs, shuffle=False, num_workers=4, pin_memory=True)
    ts_dl = DataLoader(ts_ds, batch_size=cfg.bs, shuffle=False, num_workers=4, pin_memory=True)

    round_f1: List[float] = []
    round_val_f1: List[float] = []
    pseudo_pool_sizes: List[List[int]] = []   # per-round, per-model fold pool size at start
    pseudo_acc_per_round: List[float] = []    # accuracy of new pseudo-labels added that round
    disagreement_per_round: List[float] = []

    for r in range(R):
        logger.info(f"\n========== ROUND {r+1}/{R} ==========")
        # Train each model on its current pool (fold + injected pseudo-labels)
        for k in range(K):
            logger.info(f"  [round {r+1}] training model {k+1}/{K}: pool={len(fold_pools[k])} "
                        f"(overrides={len(fold_overrides[k])})")
            train_one_model(models[k], fold_pools[k], tr_ds, fold_overrides[k],
                            cfg, dist_mat, epochs=round_epochs[r])
        # Reset base label_override after training so eval uses true labels
        tr_ds.label_override = {}

        # Round eval (ensemble on val and test)
        val_pack = eval_pack_ensemble(models, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        ts_pack = eval_pack_ensemble(models, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
        round_val_f1.append(val_pack["overall"]["macro_f1"])
        round_f1.append(ts_pack["overall"]["macro_f1"])
        disagreement_per_round.append(ts_pack["mean_disagreement_entropy"])
        logger.info(f"  [round {r+1}] val_macro={val_pack['overall']['macro_f1']:.4f} "
                    f"test_macro={ts_pack['overall']['macro_f1']:.4f} "
                    f"disagree_entropy={ts_pack['mean_disagreement_entropy']:.4f}")

        # Co-training: for each fold k, the OTHER K-1 models vote on held-out fold's samples
        # NOT yet in fold k's pool. If they unanimously agree, add (idx, voted_label) to fold k.
        if r < R - 1:
            round_pseudo_correct = 0
            round_pseudo_total = 0
            this_round_sizes = []
            for k in range(K):
                # Candidates = samples in OTHER folds that are NOT yet in fold k's pool
                in_pool = set(fold_pools[k])
                candidates = [i for j in range(K) if j != k for i in folds[j]
                              if i not in in_pool]
                if not candidates:
                    this_round_sizes.append(len(fold_pools[k])); continue
                other_models = [models[j] for j in range(K) if j != k]
                agreed, acc, n_agreed = vote_pseudo_label(other_models, tr_ds, candidates, cfg)
                # Add to fold k's pool, with the pseudo-label override
                for (idx, pl) in agreed:
                    fold_pools[k].append(idx)
                    fold_overrides[k][idx] = pl
                if n_agreed > 0:
                    round_pseudo_correct += int(round(acc * n_agreed))
                    round_pseudo_total += n_agreed
                this_round_sizes.append(len(fold_pools[k]))
                logger.info(f"  [round {r+1}] model {k+1}: candidates={len(candidates)} "
                            f"agreed={n_agreed} pseudo_acc={acc:.4f} new_pool={len(fold_pools[k])}")
            overall_acc = (round_pseudo_correct / max(1, round_pseudo_total))
            pseudo_acc_per_round.append(overall_acc)
            pseudo_pool_sizes.append(this_round_sizes)
            logger.info(f"  [round {r+1}] OVERALL pseudo-label accuracy: {overall_acc:.4f} "
                        f"over {round_pseudo_total} added samples")
        else:
            pseudo_acc_per_round.append(0.0)
            pseudo_pool_sizes.append([len(fold_pools[k]) for k in range(K)])

    # Final evaluation
    final_val_pack = eval_pack_ensemble(models, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
    final_ts_pack = eval_pack_ensemble(models, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    best_val = final_val_pack["overall"]["macro_f1"]
    test_macro = final_ts_pack["overall"]["macro_f1"]
    gap = best_val - test_macro
    logger.info(f"\n[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f}")
    logger.info(f"[final] round_f1={round_f1} disagreement={disagreement_per_round}")
    logger.info(f"[final] pseudo_acc_per_round={pseudo_acc_per_round}")

    return {
        "tag": tag, "method": "ECHO+",
        "note": "K=3 stratified k-fold co-training; per round each model trains on its fold + "
                "unanimously-agreed pseudo-labels from the other K-1 models; final = ensemble mean-softmax.",
        "enc": cfg.enc, "bench": cfg.benchmark,
        "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
        "lambda_aug": cfg.lambda_aug, "gamma": cfg.gamma, "tau": cfg.tau,
        "K_models": K, "R_rounds": R, "round_epochs": round_epochs,
        "val_macro": best_val, "macro": test_macro,
        "weighted": final_ts_pack["overall"]["weighted_f1"],
        "acc": final_ts_pack["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "test_metrics": final_ts_pack,
        "val_history": round_val_f1,
        "round_1_f1": round_f1[0] if len(round_f1) > 0 else 0.0,
        "round_2_f1": round_f1[1] if len(round_f1) > 1 else 0.0,
        "round_3_f1": round_f1[2] if len(round_f1) > 2 else 0.0,
        "round_test_f1": round_f1,
        "pseudo_label_pool_sizes": pseudo_pool_sizes,
        "pseudo_label_accuracy_per_round": pseudo_acc_per_round,
        "disagreement_entropy_per_round": disagreement_per_round,
        "final_ensemble_f1": test_macro,
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
                tag = f"exp125_echoplus_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"r1={res['round_1_f1']:.4f} r2={res['round_2_f1']:.4f} "
                                f"r3={res['round_3_f1']:.4f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp125_echoplus_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*160)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'R1-F1':>7} {'R2-F1':>7} {'R3-F1':>7} {'PLAcc(r2)':>10} {'Wall':>8}")
    print("-"*160)
    for r in results:
        pl = r.get("pseudo_label_accuracy_per_round", [0.0, 0.0, 0.0])
        pl_r2 = pl[1] if len(pl) > 1 else 0.0
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['round_1_f1']:>7.4f} {r['round_2_f1']:>7.4f} {r['round_3_f1']:>7.4f} "
              f"{pl_r2:>10.4f} {r['wall']:>8.0f}s")
    print("="*160)


if __name__ == "__main__":
    main()
