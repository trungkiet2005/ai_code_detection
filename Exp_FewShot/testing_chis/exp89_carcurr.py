# exp89_carcurr â€” CARGO Difficulty-Annealing Curriculum on transform pool
# =============================================================================
# NAME           : CARCURR (CARgo CURRiculum on transform difficulty)
# ARXIV_ID       : novel
# ONE-LINE CLAIM : Structural rewrites have INTRINSIC DIFFICULTY (paren_canon
#                  preserves nearly all tokens; for-to-while changes statement
#                  structure entirely).  Training under all difficulties from
#                  step 0 collapses the encoder; an EASY -> HARD curriculum on
#                  the transform pool yields strictly stronger structural
#                  invariance than uniform sampling.
# EQUATION       : Difficulty rank d_k for transform T_k (fixed a priori from
#                  pre-pilot view_cos with a frozen encoder):
#                    easy   = {paren_canon, op_form_swap, reg_paren_canon, reg_op_form_swap}
#                    medium = {aug_assign_expand, listcomp_unroll, reg_aug_assign}
#                    hard   = {if_invert, for_to_while, reg_if_invert_c, reg_for_to_while_c}
#                  Phase schedule (epochs):  phase 1 = easy only,
#                                             phase 2 = easy + medium,
#                                             phase 3+ = easy + medium + hard.
#                  Otherwise identical to CARGO (CE + lambda_aug * SupCon_TW).
# WHY NOT BEFORE : Curricula in NLP (CL on sequence length, on difficulty of
#                  task) and in vision (RandAugment with magnitude scheduling)
#                  exist, but no prior work CURRICULA OVER STRUCTURAL CODE
#                  REWRITES.  Anti-pattern check: this is NOT "apply curriculum
#                  to X" -- the curriculum is over a DOMAIN-SPECIFIC transform
#                  pool keyed to AST structural depth.
# FALSIFIER      : (F1) Per-epoch macro_F1 monotone INCREASING despite the
#                       transform-pool harder-set-being-introduced.
#                       Drop at phase transition => curriculum failed.
#                  (F2) Final view_cos higher than CARGO at same fraction
#                       (better structural invariance learned).
#                  (F3) Difficulty rank validated post-hoc: per-transform
#                       view_cos on TEST.  Spearman correlation with the
#                       a priori easy<medium<hard ordering should be > 0.7.
# REPORTS        : Full eval pack + (F1) per-epoch macro_F1 history
#                                 + (F2) final view_cos
#                                 + (F3) per-transform view_cos rank
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
import ast as _ast
import re as _re
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
for _p in ("numpy", "torch", "datasets", "transformers", "scikit-learn", "tqdm"):
    _ensure(_p)

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
logger = logging.getLogger("exp89_carcurr")

PAPER_BASELINE = 0.6633
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}


def _gd(u, v, adj):
    if u == v: return 0.0
    q = [(u, 0)]; seen = {u}
    while q:
        c, d = q.pop(0)
        for nb in adj.get(c, []):
            if nb == v: return d + 1.0
            if nb not in seen: seen.add(nb); q.append((nb, d + 1))
    return float("inf")

def build_distance_matrix(n, adj, default=4.0):
    D = torch.full((n, n), default)
    for i in range(n):
        for j in range(n):
            d = _gd(i, j, adj)
            if d < float("inf"): D[i, j] = d
            elif (i == 0) != (j == 0): D[i, j] = 3.0
    return D

def build_sibling_mask(n, adj):
    M = torch.zeros(n, n)
    for i in range(n):
        for j in adj.get(i, []): M[i, j] = 1.0
    return M


# ---- CARGO transforms with explicit difficulty tier --------------------------

class _AugAssignExpand(_ast.NodeTransformer):
    NAME = "aug_assign_expand"; TIER = "medium"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_AugAssign(self, node):
        if self.rng.random() < 0.7:
            self.fired += 1
            return _ast.copy_location(_ast.Assign(targets=[node.target],
                value=_ast.BinOp(left=node.target, op=node.op, right=node.value)), node)
        return node

class _IfInvert(_ast.NodeTransformer):
    NAME = "if_invert"; TIER = "hard"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_If(self, node):
        self.generic_visit(node)
        if node.orelse and self.rng.random() < 0.6:
            self.fired += 1
            return _ast.copy_location(_ast.If(
                test=_ast.UnaryOp(op=_ast.Not(), operand=node.test),
                body=node.orelse, orelse=node.body), node)
        return node

class _ForToWhile(_ast.NodeTransformer):
    NAME = "for_to_while"; TIER = "hard"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_For(self, node):
        self.generic_visit(node)
        if (isinstance(node.iter, _ast.Call) and isinstance(node.iter.func, _ast.Name)
            and node.iter.func.id == "range" and len(node.iter.args) == 1
            and isinstance(node.target, _ast.Name) and not node.orelse
            and self.rng.random() < 0.7):
            self.fired += 1
            v = node.target; up = node.iter.args[0]
            init = _ast.Assign(targets=[v], value=_ast.Constant(value=0))
            inc = _ast.AugAssign(target=v, op=_ast.Add(), value=_ast.Constant(value=1))
            cond = _ast.Compare(left=v, ops=[_ast.Lt()], comparators=[up])
            wh = _ast.While(test=cond, body=list(node.body) + [inc], orelse=[])
            return [_ast.copy_location(init, node), _ast.copy_location(wh, node)]
        return node

class _ListCompUnroll(_ast.NodeTransformer):
    NAME = "listcomp_unroll"; TIER = "medium"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_Assign(self, node):
        if (len(node.targets) == 1 and isinstance(node.targets[0], _ast.Name)
            and isinstance(node.value, _ast.ListComp) and len(node.value.generators) == 1
            and not node.value.generators[0].ifs and self.rng.random() < 0.7):
            self.fired += 1
            tgt = node.targets[0]; lc = node.value
            init = _ast.Assign(targets=[tgt], value=_ast.List(elts=[], ctx=_ast.Load()))
            ap = _ast.Expr(value=_ast.Call(
                func=_ast.Attribute(value=tgt, attr="append", ctx=_ast.Load()),
                args=[lc.elt], keywords=[]))
            fs = _ast.For(target=lc.generators[0].target, iter=lc.generators[0].iter,
                          body=[ap], orelse=[])
            return [_ast.copy_location(init, node), _ast.copy_location(fs, node)]
        return node

class _OpFormSwap(_ast.NodeTransformer):
    NAME = "op_form_swap"; TIER = "easy"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_BinOp(self, node):
        self.generic_visit(node)
        if (isinstance(node.op, _ast.Mult) and isinstance(node.right, _ast.Constant)
            and node.right.value == 2 and isinstance(node.left, _ast.Name)
            and self.rng.random() < 0.6):
            self.fired += 1
            return _ast.copy_location(_ast.BinOp(left=node.left, op=_ast.Add(), right=node.left), node)
        return node


def _apply_py(cls, code, rng):
    try: tree = _ast.parse(code)
    except (SyntaxError, ValueError): return code, False
    tr = cls(rng)
    try:
        new = tr.visit(tree); _ast.fix_missing_locations(new)
        out = _ast.unparse(new)
    except Exception: return code, False
    return (out, True) if tr.fired else (code, False)


def reg_aug_assign(code, rng):
    new, n = _re.subn(r"\b([A-Za-z_]\w*)\s*([+\-*/%])=\s*([^;\n]+)",
                      r"\1 = \1 \2 \3", code, count=rng.randint(1, 3))
    return (new, n > 0)

def reg_paren_canon(code, rng):
    new, n = _re.subn(r"=\s*([^;\n=]+?)([;\n])",
        lambda m: f"= ({m.group(1).strip()}){m.group(2)}" if rng.random() < 0.4 else m.group(0),
        code, count=rng.randint(2, 5))
    return (new, n > 0)

def reg_op_form_swap_c(code, rng):
    fired = False; new = code
    m = _re.search(r"\b([A-Za-z_]\w*)\s*\*\s*2\b", new)
    if m and rng.random() < 0.5:
        v = m.group(1); new = new[:m.start()] + f"({v} + {v})" + new[m.end():]; fired = True
    return (new, fired)

def reg_if_invert_c(code, rng):
    fired = [False]
    def r(m):
        if rng.random() < 0.4: fired[0] = True; return f"if (!({m.group(1)}))"
        return m.group(0)
    new = _re.compile(r"\bif\s*\(\s*([^()\n]+?)\s*\)").sub(r, code, count=rng.randint(1, 3))
    return (new, fired[0])

def reg_for_to_while_c(code, rng):
    new, n = _re.compile(r"for\s*\(\s*([^;]+?)\s*;\s*([^;]+?)\s*;\s*([^)]+?)\s*\)\s*\{").subn(
        lambda m: f"{m.group(1)};\nwhile ({m.group(2)}) {{", code, count=rng.randint(1, 2))
    return (new, n > 0)


# Tiered pool: (name, tier, callable). Curriculum filters by tier.
TIERED_POOL: List[Tuple[str, str, callable]] = [
    # easy: token-equivalent perturbations
    ("py_op_form_swap",     "easy",   lambda c, r: _apply_py(_OpFormSwap, c, r)),
    ("reg_paren_canon",     "easy",   reg_paren_canon),
    ("reg_op_form_swap_c",  "easy",   reg_op_form_swap_c),
    # medium: statement-form swaps
    ("py_aug_assign_expand","medium", lambda c, r: _apply_py(_AugAssignExpand, c, r)),
    ("py_listcomp_unroll",  "medium", lambda c, r: _apply_py(_ListCompUnroll, c, r)),
    ("reg_aug_assign",      "medium", reg_aug_assign),
    # hard: control-flow restructures
    ("py_if_invert",        "hard",   lambda c, r: _apply_py(_IfInvert, c, r)),
    ("py_for_to_while",     "hard",   lambda c, r: _apply_py(_ForToWhile, c, r)),
    ("reg_if_invert_c",     "hard",   reg_if_invert_c),
    ("reg_for_to_while_c",  "hard",   reg_for_to_while_c),
]


def _curr_pool_for_phase(phase: int):
    """phase 0 -> easy only;  phase 1 -> easy+medium;  phase >=2 -> all"""
    if phase <= 0: tiers = {"easy"}
    elif phase == 1: tiers = {"easy", "medium"}
    else: tiers = {"easy", "medium", "hard"}
    return [(n, t, fn) for n, t, fn in TIERED_POOL if t in tiers]


def curr_augment(code, lang, rng, phase: int) -> Tuple[str, str, str]:
    pool = _curr_pool_for_phase(phase)
    if not pool: return code, "noop", "none"
    name, tier, fn = pool[rng.randrange(len(pool))]
    try:
        out = fn(code, rng)
        new, fired = out if isinstance(out, tuple) else (out, True)
    except Exception:
        return code, f"{name}_fail", tier
    return (new, name, tier) if fired else (code, f"{name}_noop", tier)


# ---- Model + loss -----------------------------------------------------------

class CARCURRModel(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls); self.emb_dim = emb_dim; self.n_cls = n_cls

    def encode(self, ids, mask):
        o = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (o.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1), self.clf(z)


def supcon_tw_loss(z, y, dist, gamma=1.0, tau=0.1):
    N = z.size(0)
    if N < 2: return z.sum() * 0.0
    sim = (z @ z.t()) / tau
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()
    eye = torch.eye(N, device=z.device, dtype=torch.bool)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)).float().masked_fill(eye, 0.0)
    neg = (y.unsqueeze(0) != y.unsqueeze(1)).float()
    dij = dist[y][:, y]
    w = pos + neg * torch.exp(-gamma * dij)
    w = w.masked_fill(eye, 0.0)
    es = (torch.exp(sim) * w).clamp(min=1e-12)
    num = (es * pos).sum(-1).clamp(min=1e-12)
    den = es.sum(-1).clamp(min=1e-12)
    has = (pos.sum(-1) > 0).float()
    return (-(torch.log(num) - torch.log(den)) * has).sum() / has.sum().clamp(min=1.0)


# ---- Plumbing ---------------------------------------------------------------

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
    emb_dim: int = 256; device: str = "cuda"; gene_adj: dict = field(default_factory=dict)
    # Curriculum: phase 0 for epochs [0, e1), phase 1 for [e1, e2), phase 2 onward
    # e1, e2 set in adaptive_schedule based on total epochs
    phase_breakpoints: Tuple[int, int] = (2, 4)


def adaptive_schedule(c):
    if c.frac <= 0.02:
        c.epochs, c.lr_enc, c.warmup = 10, 3e-5, 0.20
        c.phase_breakpoints = (3, 6)
    elif c.frac <= 0.10:
        c.epochs, c.lr_enc, c.warmup = 6, 3e-5, 0.15
        c.phase_breakpoints = (2, 4)
    else:
        c.epochs, c.lr_enc, c.warmup = 6, 4e-5, 0.10
        c.phase_breakpoints = (2, 4)
    return c


def phase_of_epoch(epoch: int, breakpoints: Tuple[int, int]) -> int:
    e1, e2 = breakpoints
    if epoch < e1: return 0
    if epoch < e2: return 1
    return 2


def _hw(c):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40: c.bs, c.seq = 128, 512
        elif mem >= 10: c.bs, c.seq = 64, 384
        else: c.bs, c.seq = 32, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={c.bs} (2x view) seq={c.seq}")
    return c

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _is_human(t): return str(t or "").strip().lower() in {"human", "human_written", "human-generated"}

def _vocab(tr):
    names = {str(r.get("model", "") or "").strip() for r in tr
             if not _is_human(r.get("target", "")) and r.get("model", "")}
    return {n: i + 1 for i, n in enumerate(sorted(names))}

def _conv_codet(s, t, vocab):
    def row(r):
        code = next((r.get(f, "") for f in ("cleaned_code", "code")
                     if isinstance(r.get(f, ""), str) and r.get(f, "").strip()), "")
        lbl = (0 if _is_human(r.get("target", ""))
               else (1 if t == "binary"
                     else vocab.get(str(r.get("model", "") or "").strip(), -1)))
        return {"code": code, "label": lbl,
                "language": str(r.get("language", "")).strip().lower(),
                "source": str(r.get("source", "")).strip().lower()}
    return s.map(row, remove_columns=s.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _conv_aicd(s):
    def row(r):
        return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1)),
                "language": str(r.get("language", "")).strip().lower(), "source": ""}
    return s.map(row, remove_columns=s.column_names).filter(
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
    tn = {"t1": "T1", "t2": "T2", "t3": "T3"}.get(task.lower())
    if tn is None: raise ValueError(f"[aicd] bad '{task}'")
    p = os.path.join(KAGGLE_AICD, tn)
    if not os.path.isdir(p): raise FileNotFoundError(f"[aicd] STRICT: {tn} not found")
    pf = sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: no parquet")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0: return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


# Phase is stored as a module-level int updated by train loop before each epoch.
_CUR_PHASE: int = 0


class FSDS_CARCURR(TD):
    def __init__(self, data, tok, seq, frac=1.0, seed=42, do_aug=True):
        self.data = data; self.tok = tok; self.seq = seq; self.do_aug = do_aug; self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_CARCURR] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]; lang = r.get("language", "") or ""
        e0 = self.tok(code, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        ids0, m0 = e0["input_ids"].squeeze(0), e0["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            new, name, tier = curr_augment(code, lang, rng, _CUR_PHASE)
            e1 = self.tok(new, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
            ids1, m1 = e1["input_ids"].squeeze(0), e1["attention_mask"].squeeze(0)
        else:
            ids1, m1, name, tier = ids0, m0, "none", "none"
        return {"ids0": ids0, "mask0": m0, "ids1": ids1, "mask1": m1,
                "label": r["label"], "aug_name": name, "aug_tier": tier,
                "language": lang, "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_cpu, collect_falsifier=False):
    model.eval(); preds, labels = [], []
    cos_by_name: Dict[str, List[float]] = {}
    cos_by_tier: Dict[str, List[float]] = {}
    all_cos = []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device); labs = b["label"]
        z0, lg = model.encode(ids0, m0)
        preds.extend(lg.argmax(-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            ids1 = b["ids1"].to(cfg.device); m1 = b["mask1"].to(cfg.device)
            z1, _ = model.encode(ids1, m1)
            sim = (z0 * z1).sum(-1).detach().cpu().float().numpy()
            names = b.get("aug_name", []); tiers = b.get("aug_tier", [])
            names = list(names) if not isinstance(names, list) else names
            tiers = list(tiers) if not isinstance(tiers, list) else tiers
            for s, nm, ti in zip(sim, names, tiers):
                cos_by_name.setdefault(nm, []).append(float(s))
                cos_by_tier.setdefault(ti, []).append(float(s))
                all_cos.append(float(s))
    preds = np.array(preds); labels = np.array(labels); n = cfg.n_cls
    ov = {"accuracy": float(accuracy_score(labels, preds)),
          "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
          "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0))}
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
    od = int(cm.sum() - cm.trace())
    sib = int(sum(cm[i, j] for i in range(n) for j in range(n) if i != j and sib_mask_np[i, j] > 0))
    out = {"overall": ov, "confusion_matrix": cm.tolist(),
           "sibling_confusion_rate": float(sib / max(od, 1)),
           "off_diag_total": od, "n_samples": int(len(labels))}
    if collect_falsifier:
        per_name = {nm: {"mean_cos": float(np.mean(lst)), "n": len(lst)}
                    for nm, lst in cos_by_name.items() if lst}
        per_tier = {ti: {"mean_cos": float(np.mean(lst)), "n": len(lst)}
                    for ti, lst in cos_by_tier.items() if lst}
        out["falsifier"] = {
            "mean_view_cos_F2": float(np.mean(all_cos)) if all_cos else 0.0,
            "per_transform_cos_F3": per_name,
            "per_tier_cos_F3": per_tier,
        }
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg, dist):
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    tier_counts: Dict[str, int] = {}
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); m1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        for ti in b.get("aug_tier", []):
            tier_counts[ti] = tier_counts.get(ti, 0) + 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, lg = model.encode(ids0, m0)
            z1, _ = model.encode(ids1, m1)
            zd = torch.cat([z0, z1], 0); yd = torch.cat([labs, labs], 0)
            loss_ce = F.cross_entropy(lg, labs)
            loss_sc = supcon_tw_loss(zd, yd, dist, gamma=cfg.gamma, tau=cfg.tau)
            loss = loss_ce + cfg.lambda_aug * loss_sc
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item()
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n, tier_counts


def run_exp(cfg, tag):
    global _CUR_PHASE
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dmt = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist = dmt.to(cfg.device); dist_cpu = dmt.numpy()
    sib_np = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()
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
    tr_ds = FSDS_CARCURR(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_CARCURR(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=True)
    ts_ds = FSDS_CARCURR(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=True)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} wu={cfg.warmup} "
                f"l={cfg.lambda_aug} phase_bp={cfg.phase_breakpoints}")
    # num_workers=0 so _CUR_PHASE update is visible to dataset workers
    lc = dict(batch_size=cfg.bs, num_workers=0, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = CARCURRModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh = 0.0, None, []
    phase_history, tier_count_history = [], []
    for ep in range(cfg.epochs):
        _CUR_PHASE = phase_of_epoch(ep, cfg.phase_breakpoints)
        phase_history.append(_CUR_PHASE)
        loss, ce, sc, tcs = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist)
        tier_count_history.append(tcs)
        vm = eval_pack(model, vl_dl, cfg, sib_np, dist_cpu)
        v = vm["overall"]["macro_f1"]; vh.append(v)
        logger.info(f"[ep{ep+1}/phase{_CUR_PHASE}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} "
                    f"val={v:.4f} tier_counts={tcs}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    # Run eval falsifier under FULL pool (phase 2) so we see cos for every transform
    _CUR_PHASE = 2
    model.load_state_dict(best_state)
    tm = eval_pack(model, ts_dl, cfg, sib_np, dist_cpu, collect_falsifier=True)
    test = tm["overall"]["macro_f1"]; gap = best_val - test
    fa = tm["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f} "
                f"cos={fa['mean_view_cos_F2']:.3f} per_tier={fa['per_tier_cos_F3']}")
    return {"tag": tag, "method": "CARCURR", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "gamma": cfg.gamma, "tau": cfg.tau,
            "phase_breakpoints": list(cfg.phase_breakpoints),
            "val_macro": best_val, "macro": test, "weighted": tm["overall"]["weighted_f1"],
            "acc": tm["overall"]["accuracy"], "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "test_metrics": tm, "val_history": vh, "phase_history": phase_history,
            "tier_count_history": tier_count_history,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp89_carcurr_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time() - t0, 1); results.append(r)
                fa = r["test_metrics"]["falsifier"]
                logger.info(f"[{tag}] test={r['macro']:.4f} ({r['dpaper']:+.4f}) gap={r['val_test_gap']:+.4f} "
                            f"cos={fa['mean_view_cos_F2']:.3f} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp89_carcurr_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 140)
    print(f"{'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9} {'view_cos':>10} {'Wall':>8}")
    print("-" * 140)
    for r in results:
        fa = r["test_metrics"]["falsifier"]
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['mean_view_cos_F2']:>10.3f} {r['wall']:>8.0f}s")
    print("=" * 140)


if __name__ == "__main__":
    main()
