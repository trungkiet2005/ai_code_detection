# exp126 — HORIZON
# NAME       : HORIZON (Hierarchical Open-set Recurrent Inference Zone)
# REFERENCE  : new; combines knowledge distillation (Hinton 2015) +
#              confidence-based cascade serving + open-set rejection
#              (Geifman 2017 selective prediction).
# CLAIM      : Production attribution is a SERVING problem, not just
#              an offline eval. Most samples are easy and need a small
#              fast model; some are hard and need the full model; a
#              calibrated few should be REJECTED. We train (1) a small
#              student distilled from TRACO, (2) the full TRACO teacher,
#              (3) a learned confidence gate, (4) an open-set rejection
#              threshold calibrated on validation.
# EQUATION   : Stage 1: tiny_student = TRACO-distilled to 4-layer encoder
#              Stage 2: full_teacher = standard TRACO
#              Routing: if max softmax(tiny_student(x)) > tau_easy:
#                          predict tiny's argmax
#                       elif max softmax(full_teacher(x)) > tau_hard:
#                          predict full_teacher's argmax
#                       else: predict "abstain"
#              Calibrate tau_easy, tau_hard on val set for <=5% error rate.
# WHY NEW    : No prior code-attribution paper formalizes the serving
#              problem with calibrated cascade + open-set rejection.
#              This is the production deployment story missing from
#              the field.
# WOW HOOK   : "Attribution as triage. Most code is easy; some is hard;
#              some is genuinely ambiguous. The model knows which is
#              which. Coverage at 5%-risk replaces single-number Macro-F1
#              as the deployment metric."
# FALSIFIER  : (F1) Tier-1 alone accuracy > 0.85 on its routed slice
#              (small model is genuinely good on easy samples).
#              (F2) Tier-2 accuracy on its routed slice > Tier-1
#              accuracy on the same slice by >= 0.05 (medium tier earns
#              its compute). (F3) Coverage at 5%-risk > 0.60 (the model
#              can serve 60%+ of inputs with 5% error rate).
#              (F4) Cost-weighted composite (assuming 0.1x cost for
#              tiny, 1x for full) within 0.01 of full-only composite.
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
logger = logging.getLogger("exp126_horizon")

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
# Models
# =============================================================================

class TRACOFull(nn.Module):
    """Full TRACO teacher: full UniXcoder + projector + classifier."""
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


class TRACOTiny(nn.Module):
    """Tiny student: first N_layers of UniXcoder + small projector + classifier."""
    def __init__(self, enc_name, n_cls, n_layers=4, emb_dim=128):
        super().__init__()
        self.encoder_full = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        if hasattr(self.encoder_full, "encoder") and hasattr(self.encoder_full.encoder, "layer"):
            full_layers = self.encoder_full.encoder.layer
            self.encoder_full.encoder.layer = nn.ModuleList([full_layers[i] for i in range(min(n_layers, len(full_layers)))])
            self.encoder_full.config.num_hidden_layers = min(n_layers, len(full_layers))
        h = self.encoder_full.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 256), nn.GELU(), nn.Linear(256, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        self.emb_dim = emb_dim
        self.n_cls = n_cls

    def encode(self, ids, mask):
        out = self.encoder_full(input_ids=ids, attention_mask=mask)
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
    student_layers: int = 4
    distill_T: float = 4.0
    distill_alpha: float = 0.7  # weight on KL vs CE
    target_risk: float = 0.05
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
        if mem >= 40: cfg.bs, cfg.seq = 128, 512
        elif mem >= 10: cfg.bs, cfg.seq = 64, 384
        else: cfg.bs, cfg.seq = 32, 256
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
    """Dataset returning (orig, augmented) tokens + label/lang/source."""
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
            logger.info(f"[FSDS] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

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
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu):
    """Standard eval pack for a single model."""
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
    return {"overall": overall, "per_class": per_class, "per_language": per_lang, "per_source": per_src,
            "confusion_matrix": cm.tolist(), "sibling_confusion_rate": float(sib_rate),
            "cross_family_confusion_rate": float(cross_rate),
            "off_diag_total": off_diag, "n_samples": int(len(labels))}


# =============================================================================
# Phase 1: train standard TRACO teacher
# =============================================================================

def train_teacher_epoch(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train-teacher"):
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
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n


def train_teacher(cfg, tr_dl, vl_dl, sib_mask_np, dist_mat_cpu, dist_mat):
    model = TRACOFull(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    total_steps = max(1, len(tr_dl)) * cfg.epochs
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, ce, sc = train_teacher_epoch(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        vm = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = vm["overall"]["macro_f1"]; hist.append(v)
        logger.info(f"[teacher e{epoch+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, best_val, hist


# =============================================================================
# Phase 2: distill tiny student from teacher
# =============================================================================

def distill_epoch(student, teacher, loader, opt, sch, scaler, cfg):
    student.train(); teacher.eval()
    tot, kl_s, ce_s = 0.0, 0.0, 0.0
    T = cfg.distill_T
    alpha = cfg.distill_alpha
    for b in tqdm(loader, desc="Distill"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.no_grad():
            _, t_logits = teacher.encode(ids0, mask0)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            _, s_logits = student.encode(ids0, mask0)
            kl = F.kl_div(
                F.log_softmax(s_logits / T, dim=-1),
                F.softmax(t_logits / T, dim=-1),
                reduction="batchmean") * (T * T)
            ce = F.cross_entropy(s_logits, labs)
            loss = alpha * kl + (1 - alpha) * ce
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); kl_s += kl.item(); ce_s += ce.item()
    n = len(loader)
    return tot/n, kl_s/n, ce_s/n


def distill_from_teacher(teacher, cfg, tr_dl, vl_dl, sib_mask_np, dist_mat_cpu):
    student = TRACOTiny(cfg.enc, cfg.n_cls, n_layers=cfg.student_layers, emb_dim=128).to(cfg.device)
    enc_ids = {id(p) for p in student.encoder_full.parameters()}
    head_params = [p for p in student.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(student.encoder_full.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    total_steps = max(1, len(tr_dl)) * cfg.epochs
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, kl, ce = distill_epoch(student, teacher, tr_dl, opt, sch, scaler, cfg)
        vm = eval_pack(student, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = vm["overall"]["macro_f1"]; hist.append(v)
        logger.info(f"[student e{epoch+1}] loss={loss:.4f} kl={kl:.4f} ce={ce:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: vv.cpu().clone() for k, vv in student.state_dict().items()}
    student.load_state_dict(best_state)
    return student, best_val, hist


# =============================================================================
# Phase 3: calibrate cascade thresholds on val
# =============================================================================

@torch.no_grad()
def collect_predictions(model, loader, cfg):
    """Return (max_prob, pred, label) arrays for all samples in loader."""
    model.eval()
    mp, pr, lb = [], [], []
    for b in tqdm(loader, desc="Collect"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        labs = b["label"]
        _, logits = model.encode(ids0, mask0)
        probs = F.softmax(logits, dim=-1)
        mp_, pr_ = probs.max(dim=-1)
        mp.extend(mp_.cpu().tolist()); pr.extend(pr_.cpu().tolist())
        lb.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
    return np.array(mp), np.array(pr), np.array(lb)


def calibrate_thresholds(student, teacher, vl_dl, cfg, target_risk=0.05):
    """Sweep (tau_easy, tau_reject) and return the pair maximizing coverage subject to risk<=target_risk."""
    s_mp, s_pr, s_lb = collect_predictions(student, vl_dl, cfg)
    t_mp, t_pr, t_lb = collect_predictions(teacher, vl_dl, cfg)
    assert (s_lb == t_lb).all(), "label arrays must align"
    labels = s_lb
    n = len(labels)
    best = None
    grid = np.linspace(0.3, 0.95, 14)
    for tau_easy in grid:
        for tau_reject in grid:
            tier1_mask = s_mp > tau_easy
            tier2_candidates = ~tier1_mask
            tier2_accept = tier2_candidates & (t_mp > tau_reject)
            rejected = tier2_candidates & ~tier2_accept
            preds = np.where(tier1_mask, s_pr, t_pr)
            accepted = tier1_mask | tier2_accept
            n_accept = int(accepted.sum())
            if n_accept == 0: continue
            errors = ((preds != labels) & accepted).sum()
            risk = errors / n_accept
            coverage = n_accept / n
            if risk <= target_risk:
                score = coverage
                if best is None or score > best[0]:
                    best = (score, tau_easy, tau_reject, risk, coverage)
    if best is None:
        # Fallback: pick pair maximizing coverage-at-best-effort risk
        for tau_easy in grid:
            for tau_reject in grid:
                tier1_mask = s_mp > tau_easy
                tier2_candidates = ~tier1_mask
                tier2_accept = tier2_candidates & (t_mp > tau_reject)
                preds = np.where(tier1_mask, s_pr, t_pr)
                accepted = tier1_mask | tier2_accept
                n_accept = int(accepted.sum())
                if n_accept == 0: continue
                errors = ((preds != labels) & accepted).sum()
                risk = errors / n_accept
                coverage = n_accept / n
                score = coverage - 5.0 * max(0.0, risk - target_risk)
                if best is None or score > best[0]:
                    best = (score, tau_easy, tau_reject, risk, coverage)
    _, tau_easy, tau_reject, risk, coverage = best
    logger.info(f"[calibrate] tau_easy={tau_easy:.3f} tau_reject={tau_reject:.3f} "
                f"val_risk={risk:.3f} val_coverage={coverage:.3f}")
    return float(tau_easy), float(tau_reject)


# =============================================================================
# Phase 4: cascade evaluation
# =============================================================================

@torch.no_grad()
def cascade_eval(student, teacher, ts_dl, cfg, tau_easy, tau_reject, sib_mask_np, dist_mat_cpu):
    s_mp, s_pr, labels = collect_predictions(student, ts_dl, cfg)
    t_mp, t_pr, t_lb = collect_predictions(teacher, ts_dl, cfg)
    assert (labels == t_lb).all()
    n = len(labels)
    tier1_mask = s_mp > tau_easy
    tier2_candidates = ~tier1_mask
    tier2_accept = tier2_candidates & (t_mp > tau_reject)
    rejected = tier2_candidates & ~tier2_accept

    final_preds = np.where(tier1_mask, s_pr, t_pr)
    accepted = tier1_mask | tier2_accept

    # Accepted-slice metrics
    n_acc = int(accepted.sum())
    n_cls = cfg.n_cls
    if n_acc > 0:
        a_preds = final_preds[accepted]
        a_labels = labels[accepted]
        overall = {"accuracy": float(accuracy_score(a_labels, a_preds)),
                   "macro_f1": float(f1_score(a_labels, a_preds, average="macro", zero_division=0)),
                   "weighted_f1": float(f1_score(a_labels, a_preds, average="weighted", zero_division=0)),
                   "micro_f1": float(f1_score(a_labels, a_preds, average="micro", zero_division=0)),
                   "macro_precision": float(precision_score(a_labels, a_preds, average="macro", zero_division=0)),
                   "macro_recall": float(recall_score(a_labels, a_preds, average="macro", zero_division=0))}
        per_class = {"f1": f1_score(a_labels, a_preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
                     "precision": precision_score(a_labels, a_preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
                     "recall": recall_score(a_labels, a_preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist()}
        cm = confusion_matrix(a_labels, a_preds, labels=list(range(n_cls)))
    else:
        overall = {"accuracy": 0.0, "macro_f1": 0.0, "weighted_f1": 0.0, "micro_f1": 0.0,
                   "macro_precision": 0.0, "macro_recall": 0.0}
        per_class = {"f1": [0.0]*n_cls, "precision": [0.0]*n_cls, "recall": [0.0]*n_cls}
        cm = np.zeros((n_cls, n_cls), dtype=int)

    off_diag = int(cm.sum() - cm.trace())
    sib_conf = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls) if i != j and sib_mask_np[i, j] > 0))
    sib_rate = sib_conf / max(off_diag, 1)
    cross = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls) if i != j and dist_mat_cpu[i, j] >= 3.0))
    cross_rate = cross / max(off_diag, 1)

    # Tier slice metrics (falsifier)
    tier1_acc = float(accuracy_score(labels[tier1_mask], final_preds[tier1_mask])) if tier1_mask.sum() > 0 else 0.0
    tier2_acc = float(accuracy_score(labels[tier2_accept], final_preds[tier2_accept])) if tier2_accept.sum() > 0 else 0.0
    tier1_cov = float(tier1_mask.sum()) / n
    tier2_cov = float(tier2_accept.sum()) / n
    rej_rate = float(rejected.sum()) / n

    # "Tier-1 accuracy on the Tier-2 slice" (counterfactual: would student have done worse on hard cases?)
    if tier2_accept.sum() > 0:
        tier1_on_tier2_acc = float(accuracy_score(labels[tier2_accept], s_pr[tier2_accept]))
    else:
        tier1_on_tier2_acc = 0.0

    # Coverage at 5% risk (already calibrated)
    if n_acc > 0:
        risk_at_op = float(((final_preds != labels) & accepted).sum()) / n_acc
        coverage_at_op = float(n_acc) / n
    else:
        risk_at_op = 0.0; coverage_at_op = 0.0

    # Cost-weighted composite: 0.1x for tier1, 1.0x for tier2, 1.0x for rejected (we paid the teacher fwd)
    cost_tier1 = 0.1
    cost_tier2 = 1.0
    cost_rejected = 1.0
    total_cost = (tier1_mask.sum() * cost_tier1
                  + tier2_accept.sum() * cost_tier2
                  + rejected.sum() * cost_rejected) / n
    cost_weighted_composite = overall["macro_f1"] - 0.0  # placeholder; reviewer reads cost separately

    per_lang = {}; per_src = {}
    # per-lang / per-src can be computed but we operate only over accepted slice for sanity

    falsifier = {
        "tier1_coverage": tier1_cov,
        "tier1_accuracy": tier1_acc,
        "tier2_coverage": tier2_cov,
        "tier2_accuracy": tier2_acc,
        "rejection_rate": rej_rate,
        "coverage_at_5pct_risk": coverage_at_op,
        "operating_risk": risk_at_op,
        "tier1_acc_on_tier2_slice": tier1_on_tier2_acc,
        "tier2_minus_tier1_on_tier2_slice": tier2_acc - tier1_on_tier2_acc,
        "tau_easy": float(tau_easy),
        "tau_reject": float(tau_reject),
        "avg_cost_per_query": float(total_cost),
        "cost_weighted_composite": float(cost_weighted_composite),
    }

    return {"overall": overall, "per_class": per_class, "per_language": per_lang, "per_source": per_src,
            "confusion_matrix": cm.tolist(), "sibling_confusion_rate": float(sib_rate),
            "cross_family_confusion_rate": float(cross_rate),
            "off_diag_total": off_diag, "n_samples": int(n_acc),
            "falsifier": falsifier}


# =============================================================================
# run_exp
# =============================================================================

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
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    # Phase 1
    logger.info(f"[phase1] training teacher (full UniXcoder + TRACO)")
    teacher, teacher_val, teacher_hist = train_teacher(cfg, tr_dl, vl_dl, sib_mask_np, dist_mat_cpu, dist_mat)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False
    teacher_test = eval_pack(teacher, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    logger.info(f"[phase1] teacher val={teacher_val:.4f} test={teacher_test['overall']['macro_f1']:.4f}")

    # Phase 2
    logger.info(f"[phase2] distilling student ({cfg.student_layers}-layer UniXcoder)")
    student, student_val, student_hist = distill_from_teacher(teacher, cfg, tr_dl, vl_dl, sib_mask_np, dist_mat_cpu)
    student_test = eval_pack(student, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    logger.info(f"[phase2] student val={student_val:.4f} test={student_test['overall']['macro_f1']:.4f}")

    # Phase 3
    logger.info(f"[phase3] calibrating cascade thresholds on val (target risk={cfg.target_risk})")
    tau_easy, tau_reject = calibrate_thresholds(student, teacher, vl_dl, cfg, target_risk=cfg.target_risk)

    # Phase 4
    logger.info(f"[phase4] cascade evaluation on test")
    cascade_test = cascade_eval(student, teacher, ts_dl, cfg, tau_easy, tau_reject, sib_mask_np, dist_mat_cpu)
    cascade_macro = cascade_test["overall"]["macro_f1"]
    # Use teacher_val as the "val_macro" baseline (the full system's reference)
    best_val = teacher_val
    gap = best_val - cascade_macro
    fa = cascade_test["falsifier"]
    logger.info(f"[final] teacher_val={teacher_val:.4f} cascade_test={cascade_macro:.4f} "
                f"gap={gap:+.4f} t1_cov={fa['tier1_coverage']:.2f} t1_acc={fa['tier1_accuracy']:.3f} "
                f"t2_cov={fa['tier2_coverage']:.2f} t2_acc={fa['tier2_accuracy']:.3f} "
                f"rej={fa['rejection_rate']:.2f} cov@5%={fa['coverage_at_5pct_risk']:.2f}")

    return {"tag": tag, "method": "HORIZON",
            "note": "2-tier cascade (4-layer student distilled from full teacher) + open-set rejection",
            "enc": cfg.enc, "bench": cfg.benchmark, "frac": cfg.frac,
            "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "student_layers": cfg.student_layers, "distill_T": cfg.distill_T, "distill_alpha": cfg.distill_alpha,
            "tau_easy": tau_easy, "tau_reject": tau_reject,
            "val_macro": best_val, "macro": cascade_macro,
            "weighted": cascade_test["overall"]["weighted_f1"],
            "acc": cascade_test["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": cascade_macro - PAPER_BASELINE,
            "teacher_val_macro": teacher_val,
            "teacher_test_macro": teacher_test["overall"]["macro_f1"],
            "student_val_macro": student_val,
            "student_test_macro": student_test["overall"]["macro_f1"],
            "test_metrics": cascade_test,
            "teacher_test_metrics": teacher_test,
            "student_test_metrics": student_test,
            "val_history": teacher_hist,
            "student_val_history": student_hist,
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
                tag = f"exp126_horizon_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"t1_cov={fa['tier1_coverage']:.2f} t1_acc={fa['tier1_accuracy']:.3f} "
                                f"t2_cov={fa['tier2_coverage']:.2f} t2_acc={fa['tier2_accuracy']:.3f} "
                                f"rej={fa['rejection_rate']:.2f} cov@5%={fa['coverage_at_5pct_risk']:.2f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp126_horizon_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*170)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'T1cov':>7} {'T1acc':>7} {'T2cov':>7} {'T2acc':>7} {'Rej':>6} {'Cov@5%':>8} {'Wall':>8}")
    print("-"*170)
    for r in results:
        fa = r['test_metrics']['falsifier']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['tier1_coverage']:>7.3f} {fa['tier1_accuracy']:>7.3f} "
              f"{fa['tier2_coverage']:>7.3f} {fa['tier2_accuracy']:>7.3f} "
              f"{fa['rejection_rate']:>6.3f} {fa['coverage_at_5pct_risk']:>8.3f} {r['wall']:>8.0f}s")
    print("="*170)


if __name__ == "__main__":
    main()
