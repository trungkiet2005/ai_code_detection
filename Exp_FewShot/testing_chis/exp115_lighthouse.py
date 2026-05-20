# exp115 — LIGHTHOUSE
# NAME       : LIGHTHOUSE (Designed class geometry via ETF + sibling repulsion)
# REFERENCE  : new; equiangular tight frame theory (Papyan 2020 neural
#              collapse) + sibling boost from this paper's genealogy
#              prior.
# CLAIM      : The standard learnable K-class weight matrix W is the
#              wrong abstraction. We FIX W to be an Equiangular Tight
#              Frame (maximally spread on unit sphere), PLUS we add an
#              extra repulsion between SIBLING pairs so sibling vectors
#              are MORE separated than they would be in a vanilla ETF.
#              The encoder learns to PROJECT samples onto this hand-
#              designed geometry. No classifier weights to learn.
# EQUATION   : W_ETF = ETF(K, d_emb)  # equiangular tight frame, K×d
#              For each sibling pair (i, j) in adj:
#                tangent = random unit vector ⊥ W_ETF[i]
#                W[i] = normalize(W_ETF[i] + δ · tangent)
#                W[j] = normalize(W_ETF[j] - δ · tangent)
#              W is FROZEN. logits = cos(z, W) per sample.
#              L = CE(softmax(logits/τ), y) + λ · TRACO_supcon(z, y)
# WHY NEW    : ETF-Simplex (exp_n09 in this lab) used vanilla ETF and
#              underperformed at low data. LIGHTHOUSE adds the genealogy
#              prior into the ETF construction itself — sibling separation
#              is HARD-CODED into the target geometry, not learned.
#              No prior code-attribution method designs the classifier
#              geometry around the family tree.
# WOW HOOK   : "We don't learn the class embeddings — we DESIGN them.
#              The classifier is a LIGHTHOUSE: K beacons pre-arranged
#              on the unit sphere, with siblings pushed slightly further
#              apart. The encoder just learns to point at the right beacon."
# FALSIFIER  : (F1) cos(z, W_y) > cos(z, W_y') for y ≠ y' on test
#              (geometry matched). (F2) Replacing fixed W with learnable
#              W loses ≥ 0.01 at 1% slot. (F3) Composite > TRACO + 0.005.
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
logger = logging.getLogger("exp115_lighthouse")

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
# CARGO structural augmentations (copied verbatim from exp84 for 2-view contrast)
# =============================================================================

class _AugAssignExpand(_ast.NodeTransformer):
    NAME = "aug_assign_expand"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_AugAssign(self, node):
        if self.rng.random() < 0.7:
            self.fired += 1
            return _ast.copy_location(
                _ast.Assign(targets=[node.target],
                            value=_ast.BinOp(left=node.target, op=node.op, right=node.value)), node)
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
                _ast.AugAssign(target=node.targets[0], op=node.value.op, value=node.value.right), node)
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
                _ast.BinOp(left=node.left, op=_ast.Mult(), right=_ast.Constant(value=0.5)), node)
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


def reg_aug_assign(code, rng):
    if rng.random() < 0.5:
        new, n = _re.subn(r"\b([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)?)\s*([+\-*/%])=\s*([^;\n]+)",
                          r"\1 = \1 \2 \3", code, count=rng.randint(1, 3))
    else:
        new, n = _re.subn(r"\b([A-Za-z_]\w*)\s*=\s*\1\s*([+\-*/%])\s*",
                          r"\1 \2= ", code, count=rng.randint(1, 3))
    return (new, "reg_aug_assign") if n > 0 else (code, "reg_aug_assign_noop")


def reg_for_to_while_c(code, rng):
    pattern = _re.compile(r"for\s*\(\s*([^;]+?)\s*;\s*([^;]+?)\s*;\s*([^)]+?)\s*\)\s*\{")
    def repl(m):
        return f"{m.group(1)};\nwhile ({m.group(2)}) {{"
    new, n = pattern.subn(repl, code, count=rng.randint(1, 2))
    if n > 0:
        return new, "reg_for_to_while_c"
    return code, "reg_for_to_while_c_noop"


def reg_paren_canon(code, rng):
    new, n = _re.subn(r"=\s*([^;\n=]+?)([;\n])",
        lambda m: f"= ({m.group(1).strip()}){m.group(2)}" if rng.random() < 0.4 else m.group(0),
        code, count=rng.randint(2, 5))
    return (new, "reg_paren_canon") if n > 0 else (code, "reg_paren_canon_noop")


def reg_if_invert_c(code, rng):
    pattern = _re.compile(r"\bif\s*\(\s*([^()\n]+?)\s*\)")
    fired = [False]
    def repl(m):
        if rng.random() < 0.4:
            fired[0] = True
            return f"if (!({m.group(1)}))"
        return m.group(0)
    new = pattern.sub(repl, code, count=rng.randint(1, 3))
    return (new, "reg_if_invert_c") if fired[0] else (code, "reg_if_invert_c_noop")


def reg_op_form_swap(code, rng):
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


def aug_regex_structural(code, rng):
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


def fallback_ws_normalize(code, rng):
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


def cargo_augment(code, language, rng):
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
# LIGHTHOUSE: ETF + sibling repulsion + frozen classifier
# =============================================================================

def build_etf(K, d, sigma=1.0):
    """Equiangular Tight Frame: K row vectors in R^d, maximally equiangular.
    Construction: take M = sqrt(K/(K-1)) * (I - 11^T/K) which is K×K with
    rank K-1, then embed/extend into R^d. Each row of M has norm sqrt((K-1)/K)
    before scaling; after L2-normalization they form an equiangular set with
    pairwise cosine = -1/(K-1)."""
    if d < K - 1:
        raise ValueError(f"ETF requires d >= K-1; got d={d}, K={K}")
    M = torch.eye(K) - torch.ones(K, K) / K
    M = M * math.sqrt(K / (K - 1))
    # Normalize rows so each lives on the unit sphere
    W = F.normalize(M, dim=-1)
    if d > K:
        pad = torch.zeros(K, d - K)
        W = torch.cat([W, pad], dim=1)
    elif d < K:
        W = W[:, :d]
    W = F.normalize(W, dim=-1)
    return W * sigma


def add_sibling_repulsion(W, gene_adj, delta=0.3, seed=42):
    """For each sibling pair (i, j) with i < j: push them slightly further apart
    along a random tangent direction perpendicular to W[i]. Both rows are
    re-normalized to unit sphere. delta controls the strength of repulsion;
    delta=0 reduces to vanilla ETF."""
    rng = torch.Generator(); rng.manual_seed(seed)
    K, d = W.shape
    W_new = W.clone()
    pairs_processed = []
    for i in range(K):
        for j in gene_adj.get(i, []):
            if i < j:
                wi = W_new[i]
                tangent = torch.randn(d, generator=rng)
                # Project out the component along wi
                tangent = tangent - (tangent @ wi) * wi
                norm = tangent.norm().clamp(min=1e-8)
                tangent = tangent / norm
                W_new[i] = F.normalize(W_new[i] + delta * tangent, dim=0)
                W_new[j] = F.normalize(W_new[j] - delta * tangent, dim=0)
                pairs_processed.append((i, j))
    return W_new, pairs_processed


class LIGHTHOUSEModel(nn.Module):
    def __init__(self, enc_name, n_cls, gene_adj, emb_dim=256, delta=0.3, tau=0.1, seed=42):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(512, emb_dim))
        # Build fixed ETF + sibling repulsion
        W_etf = build_etf(n_cls, emb_dim, sigma=1.0)
        W_etf, sib_pairs = add_sibling_repulsion(W_etf, gene_adj, delta=delta, seed=seed)
        self.register_buffer("W", W_etf)
        self.sib_pairs = sib_pairs
        self.tau_cos = tau
        self.n_cls = n_cls
        self.emb_dim = emb_dim
        self.delta = delta

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = F.normalize(self.proj(sem), dim=-1)
        # Cosine logits to fixed W (W is already unit-norm)
        logits = (z @ self.W.t()) / self.tau_cos
        return z, logits


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
    bs: int = 128; seq: int = 512; epochs: int = 6
    lr_enc: float = 3e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    emb_dim: int = 256
    delta: float = 0.3
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
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} (HALVED for 2-view) seq={cfg.seq}")
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
        enc = self.tok("<encoder_only>" + code, max_length=self.seq_len,
                       padding="max_length", truncation=True, return_tensors="pt")
        ids0, mask0 = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            code_aug, aug_name = cargo_augment(code, lang, rng)
            enc2 = self.tok("<encoder_only>" + code_aug, max_length=self.seq_len,
                            padding="max_length", truncation=True, return_tensors="pt")
            ids1, mask1 = enc2["input_ids"].squeeze(0), enc2["attention_mask"].squeeze(0)
        else:
            ids1, mask1, aug_name = ids0, mask0, "none"
        return {"ids0": ids0, "mask0": mask0, "ids1": ids1, "mask1": mask1,
                "label": r["label"], "aug_name": aug_name,
                "language": lang, "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=False):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    # ETF falsifier accumulators
    cos_correct, cos_wrong, correct_margin = [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        labs_dev = labs.to(cfg.device) if torch.is_tensor(labs) else torch.tensor(labs, device=cfg.device)
        z0, logits = model.encode(ids0, mask0)
        pred = logits.argmax(dim=-1)
        preds.extend(pred.cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            # cos(z, W) directly (without tau scaling)
            cos_zw = z0 @ model.W.t()                       # (B, K)
            B = z0.size(0)
            idx = torch.arange(B, device=z0.device)
            cos_y = cos_zw[idx, labs_dev]                   # cos(z, W_y)
            # masked max for non-y
            mask_off = torch.ones_like(cos_zw, dtype=torch.bool)
            mask_off[idx, labs_dev] = False
            cos_nonY = cos_zw.masked_fill(~mask_off, -1e9).max(dim=-1).values
            correct = (pred == labs_dev)
            wrong_idx = ~correct
            cos_correct.append(cos_y[correct].detach().cpu().float().numpy())
            margin = (cos_y - cos_nonY)[correct].detach().cpu().float().numpy()
            correct_margin.append(margin)
            # Wrong: argmax-predicted (not-y) cosine
            if wrong_idx.any():
                cos_wrong_pred = cos_zw[idx, pred][wrong_idx]
                cos_wrong.append(cos_wrong_pred.detach().cpu().float().numpy())
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
        c_all = np.concatenate(cos_correct) if cos_correct else np.array([0.0])
        w_all = np.concatenate(cos_wrong) if cos_wrong else np.array([0.0])
        m_all = np.concatenate(correct_margin) if correct_margin else np.array([0.0])
        # Geometry diagnostics: pairwise cosines between W rows
        W_np = model.W.detach().cpu().float().numpy()
        K = W_np.shape[0]
        pairwise = W_np @ W_np.T
        off = pairwise[np.triu_indices(K, k=1)]
        # Sibling vs non-sibling W-row cosines
        sib_cos, nonsib_cos = [], []
        for i in range(K):
            for j in range(i+1, K):
                if sib_mask_np[i, j] > 0 or sib_mask_np[j, i] > 0:
                    sib_cos.append(float(pairwise[i, j]))
                else:
                    nonsib_cos.append(float(pairwise[i, j]))
        out["falsifier"] = {
            "etf_cosine_correct_mean": float(c_all.mean()),
            "etf_cosine_correct_std": float(c_all.std()),
            "etf_cosine_wrong_mean": float(w_all.mean()),
            "etf_cosine_wrong_std": float(w_all.std()),
            "etf_correct_margin_mean": float(m_all.mean()),
            "etf_correct_margin_std": float(m_all.std()),
            "W_pairwise_cos_mean": float(off.mean()),
            "W_pairwise_cos_min": float(off.min()),
            "W_pairwise_cos_max": float(off.max()),
            "W_sibling_cos_mean": float(np.mean(sib_cos)) if sib_cos else 0.0,
            "W_nonsibling_cos_mean": float(np.mean(nonsib_cos)) if nonsib_cos else 0.0,
            "n_sibling_pairs": len(sib_cos),
            "delta_used": float(model.delta),
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
            z1, _ = model.encode(ids1, mask1)
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
        tr_data = _conv_codet(tr_raw, "author", vocab)
        vl_data = _conv_codet(vl_raw, "author", vocab)
        ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"delta={cfg.delta} lambda_aug={cfg.lambda_aug}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = LIGHTHOUSEModel(cfg.enc, cfg.n_cls, gene_adj=cfg.gene_adj,
                            emb_dim=cfg.emb_dim, delta=cfg.delta, tau=cfg.tau,
                            seed=cfg.seed).to(cfg.device)
    logger.info(f"[geometry] {len(model.sib_pairs)} sibling-repulsion pair(s) applied: {model.sib_pairs}")
    # Optim: encoder + proj only (W is buffer, no params)
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
    fa = ts_met.get("falsifier", {})
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"cos_correct={fa.get('etf_cosine_correct_mean', 0.0):+.3f} "
                f"margin={fa.get('etf_correct_margin_mean', 0.0):+.3f} "
                f"W_sib_cos={fa.get('W_sibling_cos_mean', 0.0):+.3f} "
                f"W_nonsib_cos={fa.get('W_nonsibling_cos_mean', 0.0):+.3f}")
    return {"tag": tag, "method": "LIGHTHOUSE",
            "note": f"Frozen ETF classifier with sibling repulsion delta={cfg.delta}; W is a buffer not trained",
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "delta": cfg.delta, "lambda_aug": cfg.lambda_aug, "gamma": cfg.gamma, "tau": cfg.tau,
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
                tag = f"exp115_lighthouse_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics'].get('falsifier', {})
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"cos_correct={fa.get('etf_cosine_correct_mean', 0.0):+.3f} "
                                f"margin={fa.get('etf_correct_margin_mean', 0.0):+.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp115_lighthouse_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*160)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'CosOk':>8} {'Margin':>8} {'Wsib':>8} {'Wnsib':>8} {'Wall':>8}")
    print("-"*160)
    for r in results:
        fa = r['test_metrics'].get('falsifier', {})
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa.get('etf_cosine_correct_mean', 0.0):>+8.3f} {fa.get('etf_correct_margin_mean', 0.0):>+8.3f} "
              f"{fa.get('W_sibling_cos_mean', 0.0):>+8.3f} {fa.get('W_nonsibling_cos_mean', 0.0):>+8.3f} "
              f"{r['wall']:>8.0f}s")
    print("="*160)


if __name__ == "__main__":
    main()
