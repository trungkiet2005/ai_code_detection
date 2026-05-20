# exp130 — MoSAE
# NAME       : MoSAE (Mixture-of-Style-Adapter Experts: sparse MoE for code authorship)
# REFERENCE  : new for code attribution; Switch Transformer (Fedus 2022,
#              arXiv:2101.03961), Mixtral-of-Experts (Jiang 2024,
#              arXiv:2401.04088), LoRA (Hu 2021, arXiv:2106.09685).
#              MoE has NEVER been used for code authorship attribution.
# CLAIM      : Authorship is not one classifier — it is a routing policy
#              over K LATENT STYLE EXPERTS. Each expert is a low-rank
#              LoRA adapter on top of the shared UniXcoder body. A
#              learned router (small MLP conditioned on the global pooled
#              embedding) selects the top-2 experts per sample. Expert
#              specialisation emerges from supervision; identity is which
#              experts fire for a given snippet.
# EQUATION   : H = encoder(x)          # shared body
#              z = mean_pool(H)
#              router_logits = MLP(z)  # (B, K)
#              top_k = argtopk(router_logits, k=2)
#              g = softmax_top_k(router_logits)  # gates only over chosen experts
#              z_expert_k = LoRA_k(H) -> mean_pool
#              z_final = sum_k g_k * z_expert_k
#              loss = CE(clf(z_final), y) + λ_aux * router_balance_loss
# WHY NEW    : MoE is a 2024 trend in LLMs (Mixtral, DeepSeek-MoE) but
#              has never been applied to code authorship. We adapt the
#              sparse-routing paradigm to a small encoder model and show
#              that style emerges as expert-routing patterns, NOT as a
#              single softmax weight.
# WOW HOOK   : "Authorship is not a label — it is a routing POLICY over
#              K latent style experts. Two experts fire per sample; the
#              author identity is encoded in WHICH TWO."
# FALSIFIER  : (F1) Expert utilization entropy > 0.5 * log(K) — no
#              dead experts (sparse routing actually uses all experts
#              over the training set). (F2) Per-sample top-2 expert
#              consistency: same-class samples route to same expert pair
#              >= 60% of the time. (F3) Composite > TRACO + 0.005.
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
logger = logging.getLogger("exp130_mosae")

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
# CARGO structural augmentations (copied from exp84 — used as 2-view data
# augmentation for the supervised contrastive loss in MoSAE training).
# =============================================================================

class _AugAssignExpand(_ast.NodeTransformer):
    NAME = "aug_assign_expand"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_AugAssign(self, node):
        if self.rng.random() < 0.7:
            self.fired += 1
            return _ast.copy_location(
                _ast.Assign(targets=[node.target],
                            value=_ast.BinOp(left=node.target, op=node.op, right=node.value)),
                node)
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
    def repl(m): return f"{m.group(1)};\nwhile ({m.group(2)}) {{"
    new, n = pattern.subn(repl, code, count=rng.randint(1, 2))
    return (new, "reg_for_to_while_c") if n > 0 else (code, "reg_for_to_while_c_noop")


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
# MoSAE model: shared UniXcoder body + K LoRA experts + learned top-k router
# =============================================================================


class LoRAAdapter(nn.Module):
    """Low-rank residual adapter: h -> h + scale * up(down(h))."""
    def __init__(self, hidden_size, rank=16):
        super().__init__()
        self.down = nn.Linear(hidden_size, rank, bias=False)
        self.up = nn.Linear(rank, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.up.weight)  # zero init => initial pass = identity
        self.scale = 1.0 / rank

    def forward(self, h):
        return h + self.scale * self.up(self.down(h))


class MoSAE(nn.Module):
    """Shared UniXcoder body + K style-experts (LoRA) + learned top-k router."""
    def __init__(self, enc_name, n_cls, K=6, top_k=2, emb_dim=256, lora_rank=16):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.K = K
        self.top_k = top_k
        self.hidden = h
        self.experts = nn.ModuleList([LoRAAdapter(h, rank=lora_rank) for _ in range(K)])
        self.router = nn.Sequential(
            nn.Linear(h, h // 2), nn.GELU(),
            nn.Linear(h // 2, K))
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        self.emb_dim = emb_dim
        self.n_cls = n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        H = out.last_hidden_state                                # (B, L, h)
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(1).clamp(min=1)
        h_pool = (H * mask_f).sum(1) / denom                     # (B, h) — for routing
        router_logits = self.router(h_pool)                      # (B, K)
        topk_vals, topk_idx = router_logits.topk(self.top_k, dim=-1)  # (B, top_k)
        topk_gates = F.softmax(topk_vals, dim=-1)                # (B, top_k)
        B = H.size(0)
        # Compute all K expert outputs (K small, dense is fine).
        expert_outs = []
        for k in range(self.K):
            h_k = self.experts[k](H)
            z_k = (h_k * mask_f).sum(1) / denom                  # (B, h)
            expert_outs.append(z_k)
        expert_outs = torch.stack(expert_outs, dim=1)            # (B, K, h)
        z_final = torch.zeros(B, self.hidden, device=H.device, dtype=expert_outs.dtype)
        ar = torch.arange(B, device=H.device)
        for kp in range(self.top_k):
            idx = topk_idx[:, kp]                                # (B,)
            gate = topk_gates[:, kp]                             # (B,)
            chosen = expert_outs[ar, idx]                        # (B, h)
            z_final = z_final + gate.unsqueeze(-1) * chosen
        z = F.normalize(self.proj(z_final), dim=-1)
        return z, self.clf(z), router_logits, topk_idx, topk_gates


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


def router_balance_loss(router_logits: torch.Tensor, top_k: int) -> torch.Tensor:
    """Switch-Transformer-style load-balancing loss.

      L_bal = K * sum_e (P_e * F_e),
      P_e = mean over batch of softmax(router_logits)[e],
      F_e = mean over batch of 1{e in top-k} / top_k.

    Minimised when P and F are both uniform 1/K → balance loss → 1.
    """
    probs = F.softmax(router_logits, dim=-1)                 # (B, K)
    p_expert = probs.mean(dim=0)                             # (K,)
    _, topk_idx = router_logits.topk(top_k, dim=-1)
    one_hot = torch.zeros_like(probs)
    one_hot.scatter_(1, topk_idx, 1.0 / top_k)
    f_expert = one_hot.mean(dim=0)                           # (K,)
    K = router_logits.size(-1)
    return K * (p_expert * f_expert).sum()


# =============================================================================
# Plumbing
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 64; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    lambda_balance: float = 0.01
    K: int = 6; top_k: int = 2; lora_rank: int = 16
    emb_dim: int = 256
    device: str = "cuda"; gene_adj: dict = field(default_factory=dict)


def adaptive_schedule(cfg):
    f = cfg.frac
    if f <= 0.02: cfg.epochs, cfg.lr_enc, cfg.warmup = 10, 3e-5, 0.20
    elif f <= 0.10: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 3e-5, 0.15
    else: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 4e-5, 0.10
    return cfg


def _hw(cfg):
    """MoSAE is heavier: 2-view + K=6 LoRA expert forwards inside each pass.
    Be conservative on bs to leave VRAM headroom."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 80: cfg.bs, cfg.seq = 64, 512   # 2-view + K experts → conservative
        elif mem >= 40: cfg.bs, cfg.seq = 48, 512
        elif mem >= 10: cfg.bs, cfg.seq = 32, 384
        else: cfg.bs, cfg.seq = 16, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} (2-view + K={cfg.K} experts) seq={cfg.seq}")
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


class FSDS_MOSAE(TD):
    """2-view dataset: (orig, CARGO-augmented) -> supcon contrast."""
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
            logger.info(f"[FSDS_MOSAE] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

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
    # Routing accounting for the F1/F2 falsifier.
    expert_use = torch.zeros(model.K)
    # Per-class storage of top-2 pairs (sorted tuple) to compute consistency.
    per_class_pairs: Dict[int, List[Tuple[int, int]]] = {}
    per_class_first_expert: Dict[int, List[int]] = {}
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        z0, logits, _, topk_idx, _ = model.encode(ids0, mask0)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        lbl_list = labs.tolist() if torch.is_tensor(labs) else list(labs)
        labels.extend(lbl_list)
        # Expert routing stats
        topk_cpu = topk_idx.detach().cpu()
        for bi, lbl in enumerate(lbl_list):
            pair = tuple(sorted([int(topk_cpu[bi, kk].item()) for kk in range(model.top_k)]))
            per_class_pairs.setdefault(int(lbl), []).append(pair)
            per_class_first_expert.setdefault(int(lbl), []).append(pair[0])
            for kk in range(model.top_k):
                expert_use[int(topk_cpu[bi, kk].item())] += 1
        lang_batch = b.get("language", [""] * len(lbl_list))
        src_batch = b.get("source", [""] * len(lbl_list))
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
        # F1: expert utilization entropy (normalised).
        total_routes = expert_use.sum().clamp(min=1.0)
        p = (expert_use / total_routes).clamp(min=1e-12)
        H = -(p * p.log()).sum().item()
        H_norm = H / math.log(model.K) if model.K > 1 else 0.0
        # F2: same-class top-2 pair consistency (mode pair share).
        same_class_route_consistency = 0.0
        n_classes_seen = 0
        per_class_top_expert: List[int] = []
        for c in range(n_cls):
            pairs = per_class_pairs.get(c, [])
            if not pairs:
                per_class_top_expert.append(-1)
                continue
            from collections import Counter
            counts = Counter(pairs)
            top_pair, top_count = counts.most_common(1)[0]
            same_class_route_consistency += top_count / len(pairs)
            n_classes_seen += 1
            # Most-routed single expert for this class
            single_counts = Counter(per_class_first_expert.get(c, []))
            per_class_top_expert.append(single_counts.most_common(1)[0][0]
                                        if single_counts else -1)
        if n_classes_seen > 0:
            same_class_route_consistency /= n_classes_seen
        out["falsifier"] = {
            "expert_utilization": expert_use.tolist(),
            "expert_utilization_entropy": float(H_norm),
            "per_class_top_expert": per_class_top_expert,
            "same_class_route_consistency": float(same_class_route_consistency),
            "K": int(model.K), "top_k": int(model.top_k),
        }
    return out


def train_epoch_mosae(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train()
    tot, ce_s, sc_s, bal_s = 0.0, 0.0, 0.0, 0.0
    expert_use = torch.zeros(model.K)
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); mask1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, logits, r_logits0, topk_idx0, _ = model.encode(ids0, mask0)
            z1, _, r_logits1, _, _ = model.encode(ids1, mask1)
            z_d = torch.cat([z0, z1], dim=0); y_d = torch.cat([labs, labs], dim=0)
            loss_ce = F.cross_entropy(logits, labs)
            loss_sc = supcon_tw_loss(z_d, y_d, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            loss_bal = router_balance_loss(r_logits0, model.top_k) + \
                       router_balance_loss(r_logits1, model.top_k)
            loss = loss_ce + cfg.lambda_aug * loss_sc + cfg.lambda_balance * loss_bal
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        with torch.no_grad():
            for idx in topk_idx0.flatten().detach().cpu().tolist():
                expert_use[idx] += 1
        tot += loss.item(); ce_s += loss_ce.item()
        sc_s += loss_sc.item(); bal_s += loss_bal.item()
    n = max(1, len(loader))
    return tot/n, ce_s/n, sc_s/n, bal_s/n, expert_use


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
    tr_ds = FSDS_MOSAE(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_MOSAE(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS_MOSAE(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"lambda_aug={cfg.lambda_aug} lambda_bal={cfg.lambda_balance} K={cfg.K} top_k={cfg.top_k}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = MoSAE(cfg.enc, cfg.n_cls, K=cfg.K, top_k=cfg.top_k,
                  emb_dim=cfg.emb_dim, lora_rank=cfg.lora_rank).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    train_expert_use_total = torch.zeros(model.K)
    for epoch in range(cfg.epochs):
        loss, ce, sc, bal, ep_expert_use = train_epoch_mosae(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        train_expert_use_total += ep_expert_use
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} supcon={sc:.4f} bal={bal:.4f} val={v:.4f} "
                    f"expert_use={ep_expert_use.tolist()}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=True)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    fa = ts_met["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"expert_entropy={fa['expert_utilization_entropy']:.3f} "
                f"route_consistency={fa['same_class_route_consistency']:.3f}")
    return {"tag": tag, "method": "MoSAE",
            "note": "Mixture-of-Style-Adapter Experts (K=6, top-2 sparse routing, LoRA rank=16)",
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "lambda_balance": cfg.lambda_balance,
            "gamma": cfg.gamma, "tau": cfg.tau,
            "K": cfg.K, "top_k": cfg.top_k, "lora_rank": cfg.lora_rank,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "train_expert_use_total": train_expert_use_total.tolist(),
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
                tag = f"exp130_mosae_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"entropy={fa['expert_utilization_entropy']:.3f} "
                                f"route_cons={fa['same_class_route_consistency']:.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp130_mosae_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'Entropy':>9} {'RouteCons':>10} {'Wall':>8}")
    print("-"*150)
    for r in results:
        fa = r['test_metrics']['falsifier']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['expert_utilization_entropy']:>9.3f} {fa['same_class_route_consistency']:>10.3f} "
              f"{r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
