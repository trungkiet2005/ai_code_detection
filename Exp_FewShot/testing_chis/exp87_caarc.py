# exp87_caarc â€” CARGO Adaptive Reward Curriculum: per-transform self-weighting
# =============================================================================
# NAME           : CAARC (CARgo Adaptive Reward Curriculum)
# ARXIV_ID       : novel
# ONE-LINE CLAIM : Different structural transforms carry different attribution
#                  signal; the encoder should LEARN which transform to weight
#                  more via online reward proportional to the contrastive
#                  uniformity gap (pos_cos - neg_cos) per transform.
# EQUATION       : For each transform T_k:
#                    r_k = EMA[ mean_pos_cos(T_k) - mean_neg_cos(T_k) ]
#                  Sampling probability over CARGO pool:
#                    p_k = softmax(r_k / temp_sampler)
#                  Loss (same as CARGO):
#                    L = L_ce + lambda_aug * SupCon_TW([z; z'])
#                  Reward update each batch from per-sample transform tag.
# WHY NOT BEFORE : Contrastive-augmentation curricula in vision (RandAugment,
#                  TrivialAugment) pick transforms uniformly or by hand-set
#                  policy.  CAARC is the first to condition the policy on
#                  ONLINE CONTRASTIVE REWARD specific to code attribution.
#                  Differs from MoCo's negative-sample re-weighting (which
#                  weights samples, not transform-distributions).
# FALSIFIER      : (F1) Final p_k distribution: must be NON-UNIFORM (KL >
#                       0.1 from uniform).  Uniform => reward signal not
#                       informative => method is no-op vs. CARGO.
#                  (F2) Rank correlation: rank(p_k after training) vs.
#                       rank(per-transform F1 in CARGO ablation).
#                       Spearman > 0.5 => the reward really tracks utility.
#                  (F3) Delta vs CARGO @ matched fracs: must be positive
#                       >= 1 slot, else adaptive weighting wastes capacity.
# REPORTS        : Full eval pack + (F1) final p_k distribution
#                                 + (F2) per-transform mean reward r_k
#                                 + (F3) sampling counts per transform
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
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
logger = logging.getLogger("exp87_caarc")

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


# ---- CARGO transforms with explicit named pool ------------------------------

class _AugAssignExpand(_ast.NodeTransformer):
    NAME = "aug_assign_expand"
    def __init__(self, rng): self.rng = rng; self.fired = 0
    def visit_AugAssign(self, node):
        if self.rng.random() < 0.7:
            self.fired += 1
            return _ast.copy_location(_ast.Assign(targets=[node.target],
                value=_ast.BinOp(left=node.target, op=node.op, right=node.value)), node)
        return node

class _IfInvert(_ast.NodeTransformer):
    NAME = "if_invert"
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
    NAME = "for_to_while"
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
    NAME = "listcomp_unroll"
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
    NAME = "op_form_swap"
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

def reg_if_invert_c(code, rng):
    fired = [False]
    def r(m):
        if rng.random() < 0.4: fired[0] = True; return f"if (!({m.group(1)}))"
        return m.group(0)
    new = _re.compile(r"\bif\s*\(\s*([^()\n]+?)\s*\)").sub(r, code, count=rng.randint(1, 3))
    return (new, fired[0])


# Named pool: (name, callable) -- transform_idx -> name for reward tracking
NAMED_POOL: List[Tuple[str, callable]] = [
    ("py_aug_assign_expand", lambda c, r: _apply_py(_AugAssignExpand, c, r)),
    ("py_if_invert",         lambda c, r: _apply_py(_IfInvert, c, r)),
    ("py_for_to_while",      lambda c, r: _apply_py(_ForToWhile, c, r)),
    ("py_listcomp_unroll",   lambda c, r: _apply_py(_ListCompUnroll, c, r)),
    ("py_op_form_swap",      lambda c, r: _apply_py(_OpFormSwap, c, r)),
    ("reg_aug_assign",       reg_aug_assign),
    ("reg_paren_canon",      reg_paren_canon),
    ("reg_if_invert_c",      reg_if_invert_c),
]

# Global state: per-transform reward EMA + count.  Updated post-batch by train loop.
class TransformPolicy:
    def __init__(self, pool, ema=0.95, temp=0.5):
        self.names = [p[0] for p in pool]; self.K = len(pool); self.fns = [p[1] for p in pool]
        self.rewards = np.zeros(self.K, dtype=np.float32)
        self.counts = np.zeros(self.K, dtype=np.int64)
        self.ema = ema; self.temp = temp
        self.lock = None  # multi-worker safe? we'll accept eventual consistency
    def sample_idx(self, rng):
        p = self.probs()
        # rng-aware draw
        u = rng.random()
        cdf = 0.0
        for k in range(self.K):
            cdf += p[k]
            if u <= cdf: return k
        return self.K - 1
    def probs(self):
        x = self.rewards / max(self.temp, 1e-6)
        x = x - x.max()
        e = np.exp(x); return (e / e.sum()).astype(np.float32)
    def update(self, idx, r):
        self.rewards[idx] = self.ema * self.rewards[idx] + (1 - self.ema) * r
        self.counts[idx] += 1

_POLICY: TransformPolicy = TransformPolicy(NAMED_POOL)


def caarc_augment(code, lang, rng) -> Tuple[str, int, str]:
    """Returns (new_code, transform_idx, name).  Falls through (returns idx=-1)
    if no transform fires; train loop treats those as anchor==view."""
    idx = _POLICY.sample_idx(rng)
    name, fn = _POLICY.names[idx], _POLICY.fns[idx]
    try:
        out = fn(code, rng)
        new, fired = out if isinstance(out, tuple) else (out, True)
    except Exception:
        return code, -1, f"{name}_fail"
    if not fired:
        return code, -1, f"{name}_noop"
    return new, idx, name


# ---- Model + loss -----------------------------------------------------------

class CAARCModel(nn.Module):
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
    policy_ema: float = 0.95; policy_temp: float = 0.5
    emb_dim: int = 256; device: str = "cuda"; gene_adj: dict = field(default_factory=dict)


def adaptive_schedule(c):
    if c.frac <= 0.02: c.epochs, c.lr_enc, c.warmup = 10, 3e-5, 0.20
    elif c.frac <= 0.10: c.epochs, c.lr_enc, c.warmup = 6, 3e-5, 0.15
    else: c.epochs, c.lr_enc, c.warmup = 6, 4e-5, 0.10
    return c

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


class FSDS_CAARC(TD):
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
            logger.info(f"[FSDS_CAARC] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]; lang = r.get("language", "") or ""
        e0 = self.tok(code, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        ids0, m0 = e0["input_ids"].squeeze(0), e0["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            new, idx, name = caarc_augment(code, lang, rng)
            e1 = self.tok(new, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
            ids1, m1 = e1["input_ids"].squeeze(0), e1["attention_mask"].squeeze(0)
        else:
            ids1, m1, idx, name = ids0, m0, -1, "none"
        return {"ids0": ids0, "mask0": m0, "ids1": ids1, "mask1": m1,
                "label": r["label"], "t_idx": int(idx), "t_name": name,
                "language": lang, "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_cpu, collect_falsifier=False):
    model.eval(); preds, labels = [], []
    cos_overall = []
    cos_by_idx: Dict[int, List[float]] = {}
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device); labs = b["label"]
        z0, lg = model.encode(ids0, m0)
        preds.extend(lg.argmax(-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            ids1 = b["ids1"].to(cfg.device); m1 = b["mask1"].to(cfg.device)
            z1, _ = model.encode(ids1, m1)
            sim = (z0 * z1).sum(-1).detach().cpu().float().numpy()
            cos_overall.extend(sim.tolist())
            for s, k in zip(sim, b["t_idx"].tolist() if torch.is_tensor(b["t_idx"]) else list(b["t_idx"])):
                cos_by_idx.setdefault(int(k), []).append(float(s))
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
        per_t = {}
        for k, lst in cos_by_idx.items():
            nm = _POLICY.names[k] if 0 <= k < _POLICY.K else f"idx_{k}"
            per_t[nm] = {"mean_cos": float(np.mean(lst)), "n": len(lst)}
        out["falsifier"] = {
            "mean_view_cos_F1": float(np.mean(cos_overall)) if cos_overall else 0.0,
            "per_transform_cos_F2": per_t,
            "policy_probs_F1": _POLICY.probs().tolist(),
            "policy_rewards_F2": _POLICY.rewards.tolist(),
            "policy_counts_F3": _POLICY.counts.tolist(),
            "policy_names": list(_POLICY.names),
        }
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg, dist):
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); m1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        t_idx = b["t_idx"]  # tensor or list of transform indices
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
        # ---- POLICY REWARD UPDATE ----
        with torch.no_grad():
            pos_cos = (z0 * z1).sum(-1).float().cpu().numpy()      # per-sample
            sim_mat = (z0 @ z1.t()).float().cpu().numpy()           # cross sims
            # negative cos: average sim to OTHER samples (off-diagonal)
            np.fill_diagonal(sim_mat, 0.0)
            B = sim_mat.shape[0]
            neg_cos = sim_mat.sum(axis=1) / max(B - 1, 1)
            rewards = pos_cos - neg_cos
            t_idx_list = t_idx.tolist() if torch.is_tensor(t_idx) else list(t_idx)
            for k_i, r_i in zip(t_idx_list, rewards):
                if 0 <= int(k_i) < _POLICY.K:
                    _POLICY.update(int(k_i), float(r_i))
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item()
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n


def run_exp(cfg, tag):
    global _POLICY
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    # reset policy per-run to avoid cross-run contamination
    _POLICY = TransformPolicy(NAMED_POOL, ema=cfg.policy_ema, temp=cfg.policy_temp)
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
    tr_ds = FSDS_CAARC(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_CAARC(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=True)
    ts_ds = FSDS_CAARC(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=True)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} wu={cfg.warmup} "
                f"l={cfg.lambda_aug} ema={cfg.policy_ema} temp={cfg.policy_temp}")
    # NOTE: num_workers=0 so policy state is shared (workers fork it independently otherwise).
    # We accept slower data loading for correctness of the reward signal.
    lc = dict(batch_size=cfg.bs, num_workers=0, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = CAARCModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh = 0.0, None, []
    policy_hist: List[List[float]] = []
    for ep in range(cfg.epochs):
        loss, ce, sc = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist)
        vm = eval_pack(model, vl_dl, cfg, sib_np, dist_cpu)
        v = vm["overall"]["macro_f1"]; vh.append(v)
        policy_hist.append(_POLICY.probs().tolist())
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} val={v:.4f}")
        logger.info(f"  policy probs: {dict(zip(_POLICY.names, [f'{p:.2f}' for p in _POLICY.probs()]))}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    tm = eval_pack(model, ts_dl, cfg, sib_np, dist_cpu, collect_falsifier=True)
    test = tm["overall"]["macro_f1"]; gap = best_val - test
    fa = tm["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f} "
                f"cos={fa['mean_view_cos_F1']:.3f}")
    # F1 KL-from-uniform on policy
    p = np.array(fa["policy_probs_F1"]); p = np.clip(p, 1e-8, 1.0)
    kl_unif = float(np.sum(p * np.log(p * len(p))))
    return {"tag": tag, "method": "CAARC", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "gamma": cfg.gamma, "tau": cfg.tau,
            "policy_ema": cfg.policy_ema, "policy_temp": cfg.policy_temp,
            "val_macro": best_val, "macro": test, "weighted": tm["overall"]["weighted_f1"],
            "acc": tm["overall"]["accuracy"], "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "test_metrics": tm, "val_history": vh, "policy_history": policy_hist,
            "policy_kl_from_uniform": kl_unif,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp87_caarc_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time() - t0, 1); results.append(r)
                logger.info(f"[{tag}] test={r['macro']:.4f} ({r['dpaper']:+.4f}) gap={r['val_test_gap']:+.4f} "
                            f"KL_unif={r['policy_kl_from_uniform']:.3f} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp87_caarc_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 140)
    print(f"{'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9} {'KL_unif':>9} {'Wall':>8}")
    print("-" * 140)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['policy_kl_from_uniform']:>9.3f} {r['wall']:>8.0f}s")
    print("=" * 140)


if __name__ == "__main__":
    main()
