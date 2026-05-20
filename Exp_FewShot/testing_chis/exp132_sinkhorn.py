# exp132 — SINKHORN
# NAME       : SINKHORN (Optimal Transport attribution via Sinkhorn distance)
# REFERENCE  : new for code attribution; Sinkhorn-Knopp algorithm (Cuturi 2013,
#              arXiv:1306.0895), Optimal Transport in deep learning
#              (Peyré & Cuturi 2019, FnT). NEVER applied to AI-code authorship.
# CLAIM      : Standard attribution computes COSINE to a single class
#              prototype (point-level matching). This collapses the
#              within-class variation: every author's training samples
#              form a DISTRIBUTION in embedding space, not a single point.
#              The correct attribution should be DISTRIBUTION-LEVEL:
#              compute the SINKHORN (entropic OT) distance between the
#              query and each author's training distribution. The
#              minimum-cost transport plan reveals the author.
# EQUATION   : Build per-author "bank" of M=32 representative embeddings.
#              For query q with embedding z(q):
#                For each candidate author c:
#                  - Compute cost matrix C[i,j] = ||z(q) - bank_c[j]||^2
#                    (treating q as 1 point and bank_c as M points)
#                  - Solve entropic OT: min_P <C, P> - ε·H(P)
#                    subject to row/col marginals
#                  - Sinkhorn distance = <C, P*>
#                y_hat = argmin_c Sinkhorn(q, bank_c)
#              Train: TRACO supcon (encoder learns clusters) + small
#              bank-update step every epoch (k-means refit on training embeddings)
# WHY NEW    : Prototypical networks (Snell 2017) used class MEAN, which
#              is a point estimate. SINKHORN keeps M=32 anchor points per
#              author and computes DISTRIBUTIONAL distance via OT.
#              No prior code-attribution paper uses optimal transport.
# WOW HOOK   : "Attribution is not nearest-neighbor — it is distribution
#              MATCHING. The author whose training distribution best
#              transports to the query wins. Optimal transport replaces
#              cosine."
# FALSIFIER  : (F1) Per-author bank diversity: mean pairwise cosine
#              within a bank < 0.95 (banks ARE diverse, not collapsed).
#              (F2) Sinkhorn F1 > parametric softmax F1 by >= 0.005
#              (OT inference beats softmax). (F3) Removing bank
#              (reduce to nearest-mean) loses >= 0.01 at 1% slot.
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
logger = logging.getLogger("exp132_sinkhorn")

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
# CARGO structural augmentations (reused as the view generator for SupCon)
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


# =============================================================================
# SINKHORN: optimal-transport attribution
# =============================================================================

class SinkhornModel(nn.Module):
    """TRACO encoder + per-author bank of M embeddings.

    Forward returns the standard encoded embedding + softmax logits (as fallback
    / training signal). The OT inference is done OUTSIDE this module, at eval
    time, by computing per-class Sinkhorn distances to the bank.
    """
    def __init__(self, enc_name, n_cls, emb_dim=256, M=32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        # Per-author bank: (n_cls, M, emb_dim). NOT a parameter; updated externally.
        self.register_buffer("bank", torch.zeros(n_cls, M, emb_dim))
        self.register_buffer("bank_valid", torch.zeros(n_cls, dtype=torch.bool))
        self.M = M
        self.emb_dim = emb_dim
        self.n_cls = n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        z = F.normalize(z, dim=-1)
        return z, self.clf(z)


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


def sinkhorn_distance(z_query, bank, epsilon=0.05):
    """Entropic-OT Sinkhorn distance from each query point to a single bank.

    z_query : (B, d)   — treated as B independent 1-point distributions.
    bank    : (M, d)   — uniform distribution over M anchors.

    Returns (B,) of effective Sinkhorn distances. The closed form for a
    single source point under entropic regularisation is
        dist(q) = -epsilon * (logsumexp(-||q - b_j||^2 / epsilon) - log M)
    which is the softmin-weighted average squared distance.
    """
    # Cost: ||q - b||^2  (since both are unit-normalised on the sphere this is
    # equivalent to 2 - 2*<q, b> up to a constant — we keep the squared form
    # so the formula is correct for un-normalised banks too).
    C = (z_query.unsqueeze(1) - bank.unsqueeze(0)).pow(2).sum(-1)   # (B, M)
    log_K = -C / max(epsilon, 1e-6)                                  # (B, M)
    log_alpha = torch.logsumexp(log_K, dim=-1)                       # (B,)
    M = bank.shape[0]
    dist = -epsilon * (log_alpha - math.log(max(M, 1)))
    return dist


def per_class_sinkhorn(z_query, bank, bank_valid, epsilon=0.05):
    """Compute Sinkhorn distance from each query to each class bank.

    z_query    : (B, d)
    bank       : (n_cls, M, d)
    bank_valid : (n_cls,) boolean — which classes have populated banks

    Returns (B, n_cls). Use argmin for prediction. Unfilled classes get +inf.
    """
    B = z_query.size(0)
    n_cls = bank.shape[0]
    dists = torch.full((B, n_cls), float("inf"), device=z_query.device, dtype=z_query.dtype)
    for c in range(n_cls):
        if not bool(bank_valid[c].item()):
            continue
        dists[:, c] = sinkhorn_distance(z_query, bank[c], epsilon=epsilon)
    return dists


# Tiny torch-only k-means fallback so the file works even if sklearn import is heavy.
def _torch_kmeans(x: torch.Tensor, k: int, n_iter: int = 25, seed: int = 0):
    """x: (N, d) tensor on any device. Returns centroids (k, d)."""
    N, d = x.shape
    g = torch.Generator(device=x.device).manual_seed(seed)
    if N <= k:
        c = x.clone()
        if N < k:
            pad = x[:1].expand(k - N, d)
            c = torch.cat([c, pad], dim=0)
        return c
    idx = torch.randperm(N, generator=g, device=x.device)[:k]
    centers = x[idx].clone()
    for _ in range(n_iter):
        d2 = (x.unsqueeze(1) - centers.unsqueeze(0)).pow(2).sum(-1)  # (N, k)
        a = d2.argmin(dim=-1)
        new_centers = centers.clone()
        for j in range(k):
            sel = (a == j)
            if sel.sum() > 0:
                new_centers[j] = x[sel].mean(dim=0)
        shift = (new_centers - centers).pow(2).sum().item()
        centers = new_centers
        if shift < 1e-6:
            break
    return centers


@torch.no_grad()
def refit_banks(model, loader, cfg, n_cls, M):
    """Re-fit per-class banks via k-means on training embeddings.

    For each class: collect all training embeddings of that class, k-means them
    into M centroids, store as bank. Returns per-class embedding lists for
    diagnostic use (bank diversity falsifier F1).
    """
    try:
        from sklearn.cluster import KMeans  # noqa
        have_sklearn = True
    except Exception:
        have_sklearn = False

    model.eval()
    per_class_embs: Dict[int, List[np.ndarray]] = {c: [] for c in range(n_cls)}
    for b in tqdm(loader, desc="Refit-Bank"):
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device)
        labs = b["label"]
        z, _ = model.encode(ids, mask)
        z_cpu = z.detach().cpu().float().numpy()
        labs_l = labs.tolist() if torch.is_tensor(labs) else list(labs)
        for i, l in enumerate(labs_l):
            if 0 <= int(l) < n_cls:
                per_class_embs[int(l)].append(z_cpu[i])
    new_bank = model.bank.clone().fill_(0)
    new_valid = torch.zeros(n_cls, dtype=torch.bool, device=model.bank.device)
    for c in range(n_cls):
        embs = per_class_embs[c]
        if len(embs) == 0:
            continue
        if len(embs) == 1:
            v = torch.tensor(embs[0], device=new_bank.device, dtype=new_bank.dtype)
            v = F.normalize(v, dim=-1)
            for j in range(M):
                new_bank[c, j] = v
            new_valid[c] = True
            continue
        embs_arr = np.stack(embs).astype(np.float32)
        n_clusters = min(M, len(embs))
        if have_sklearn:
            try:
                km = KMeans(n_clusters=n_clusters, n_init=3, random_state=cfg.seed)
                km.fit(embs_arr)
                centers = torch.tensor(km.cluster_centers_, device=new_bank.device, dtype=new_bank.dtype)
            except Exception:
                xt = torch.tensor(embs_arr, device=new_bank.device, dtype=new_bank.dtype)
                centers = _torch_kmeans(xt, n_clusters, n_iter=20, seed=cfg.seed + c)
        else:
            xt = torch.tensor(embs_arr, device=new_bank.device, dtype=new_bank.dtype)
            centers = _torch_kmeans(xt, n_clusters, n_iter=20, seed=cfg.seed + c)
        centers = F.normalize(centers, dim=-1)
        if n_clusters < M:
            pad = centers[0].unsqueeze(0).expand(M - n_clusters, -1)
            centers = torch.cat([centers, pad], dim=0)
        new_bank[c] = centers
        new_valid[c] = True
    model.bank.copy_(new_bank)
    model.bank_valid.copy_(new_valid)
    # Report bank diversity: mean pairwise cosine within each class bank.
    diversities = []
    for c in range(n_cls):
        if not bool(new_valid[c].item()):
            continue
        b = new_bank[c]  # (M, d) already normalised
        sim = (b @ b.t())
        eye = torch.eye(b.shape[0], device=b.device, dtype=torch.bool)
        off = sim.masked_fill(eye, 0.0)
        denom = b.shape[0] * (b.shape[0] - 1)
        if denom > 0:
            mean_off = off.sum().item() / denom
            diversities.append(mean_off)
    diversity_mean = float(np.mean(diversities)) if diversities else 1.0
    return per_class_embs, diversity_mean


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
    M: int = 32           # bank size per class
    epsilon: float = 0.05  # Sinkhorn entropic reg
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


class FSDS_SINKHORN(TD):
    """Dataset that returns (original_tokens, structurally-augmented_tokens, label)."""
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
            logger.info(f"[FSDS_SINKHORN] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

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
    return {"overall": overall, "per_class": per_class, "per_language": per_lang,
            "per_source": per_src, "confusion_matrix": cm.tolist(),
            "sibling_confusion_rate": float(sib_rate),
            "cross_family_confusion_rate": float(cross_rate),
            "off_diag_total": off_diag, "n_samples": int(len(labels))}


@torch.no_grad()
def eval_pack_sinkhorn(model, loader, cfg, sib_mask_np, dist_mat_cpu,
                       use_sinkhorn: bool = True):
    """Both parametric softmax AND Sinkhorn-OT predictions reported.

      - macro_main : standard parametric softmax (from clf head)
      - macro      : Sinkhorn-OT prediction (argmin over class banks) — HEADLINE
    """
    model.eval()
    preds_main, preds_sink, labels, langs, sources = [], [], [], [], []
    have_valid_bank = use_sinkhorn and bool(model.bank_valid.any().item())
    for b in tqdm(loader, desc="Eval-Sinkhorn"):
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device); labs = b["label"]
        z, logits = model.encode(ids, mask)
        preds_main.extend(logits.argmax(dim=-1).cpu().tolist())
        if have_valid_bank:
            dists = per_class_sinkhorn(z.float(), model.bank.float(),
                                       model.bank_valid, epsilon=cfg.epsilon)
            preds_sink.extend(dists.argmin(dim=-1).cpu().tolist())
        else:
            preds_sink.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        lang_batch = b.get("language", [""] * len(labs))
        src_batch = b.get("source", [""] * len(labs))
        langs.extend(list(lang_batch) if not isinstance(lang_batch, list) else lang_batch)
        sources.extend(list(src_batch) if not isinstance(src_batch, list) else src_batch)
    preds_main = np.array(preds_main); preds_sink = np.array(preds_sink); labels = np.array(labels)
    n_cls = cfg.n_cls
    sum_sink = _summarize_preds(preds_sink, labels, langs, sources, n_cls, sib_mask_np, dist_mat_cpu)
    sum_main = _summarize_preds(preds_main, labels, langs, sources, n_cls, sib_mask_np, dist_mat_cpu)
    sum_sink["overall"]["macro_f1_main"] = sum_main["overall"]["macro_f1"]
    sum_sink["overall"]["accuracy_main"] = sum_main["overall"]["accuracy"]
    sum_sink["overall"]["weighted_f1_main"] = sum_main["overall"]["weighted_f1"]
    sum_sink["overall"]["delta_sink_minus_main"] = (sum_sink["overall"]["macro_f1"]
                                                    - sum_main["overall"]["macro_f1"])
    sum_sink["softmax_summary"] = sum_main
    sum_sink["used_sinkhorn"] = bool(have_valid_bank)
    return sum_sink


def train_epoch_sinkhorn(model, loader, opt, sch, scaler, cfg, dist_mat):
    """Standard TRACO training; banks are updated externally between epochs."""
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
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n


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
    tr_ds = FSDS_SINKHORN(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    tr_ds_noaug = FSDS_SINKHORN(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=False)
    # Pin the sampled indices so refit sees the same subset.
    tr_ds_noaug.data = tr_ds.data
    vl_ds = FSDS_SINKHORN(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS_SINKHORN(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"lambda_aug={cfg.lambda_aug} gamma={cfg.gamma} M={cfg.M} epsilon={cfg.epsilon}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    tr_dl_noaug = DataLoader(tr_ds_noaug, shuffle=False, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = SinkhornModel(cfg.enc, cfg.n_cls, cfg.emb_dim, M=cfg.M).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    best_diversity = 1.0
    for epoch in range(cfg.epochs):
        loss, ce, sc = train_epoch_sinkhorn(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        # Refit banks AFTER each training epoch so the OT evaluator sees the
        # encoder's current geometry.
        _, diversity = refit_banks(model, tr_dl_noaug, cfg, cfg.n_cls, cfg.M)
        val_met = eval_pack_sinkhorn(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu,
                                     use_sinkhorn=True)
        v = val_met["overall"]["macro_f1"]
        v_main = val_met["overall"]["macro_f1_main"]
        val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} supcon={sc:.4f} "
                    f"val_sink={v:.4f} val_main={v_main:.4f} bank_div={diversity:.3f}")
        if v > best_val:
            best_val = v
            best_diversity = diversity
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    ts_met = eval_pack_sinkhorn(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, use_sinkhorn=True)
    test_macro_sink = ts_met["overall"]["macro_f1"]
    test_macro_main = ts_met["overall"]["macro_f1_main"]
    gap = best_val - test_macro_sink
    delta = ts_met["overall"]["delta_sink_minus_main"]
    logger.info(f"[final] val={best_val:.4f} test_sink={test_macro_sink:.4f} "
                f"test_main={test_macro_main:.4f} delta={delta:+.4f} gap={gap:+.4f} "
                f"bank_div={best_diversity:.3f}")
    return {"tag": tag, "method": "SINKHORN", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "gamma": cfg.gamma, "tau": cfg.tau,
            "M": cfg.M, "epsilon": cfg.epsilon,
            "val_macro": best_val, "macro": test_macro_sink,
            "macro_main": test_macro_main,
            "delta_sink_minus_main": delta,
            "bank_diversity_mean": best_diversity,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro_sink - PAPER_BASELINE,
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
                tag = f"exp132_sinkhorn_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test_sink={res['macro']:.4f} "
                                f"test_main={res['macro_main']:.4f} "
                                f"delta={res['delta_sink_minus_main']:+.4f} "
                                f"({res['dpaper']:+.4f}) gap={res['val_test_gap']:+.4f} "
                                f"bank_div={res['bank_diversity_mean']:.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp132_sinkhorn_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} "
          f"{'Sink-F1':>8} {'Main-F1':>8} {'Delta':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'BankDiv':>9} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['macro_main']:>8.4f} "
              f"{r['delta_sink_minus_main']:>+8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['bank_diversity_mean']:>9.3f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
