# exp88_caghost â€” CARGO Ghost-Sibling: k-view hard mining over genealogy graph
# =============================================================================
# NAME           : CAGHOST (CARgo GHost-sibling hard mining)
# ARXIV_ID       : novel; extends Wei et al "Hard Negative Mining via Contrastive Learning"
#                  by conditioning the HARD-NEGATIVE pool on GENEALOGY (siblings only).
# ONE-LINE CLAIM : The HARDEST attribution mistakes are between SIBLINGS in the
#                  generator family tree (S1, S5).  Mining HARDEST sibling
#                  negatives + HARDEST same-author positives across K augmented
#                  views explicitly attacks this confusion class, beyond what
#                  uniform SupCon over the batch can express.
# EQUATION       : For anchor x with label y, generate K=3 views {x_k = T_k(x)}.
#                  Embeddings: {z, z_1, z_2, z_3}.  In doubled batch,
#                    hardest positive   p* = argmin_{j in same-label, j != anchor} cos(z, z_j)
#                    hardest sib-neg    n* = argmax_{j: sib_mask[y, y_j] = 1} cos(z, z_j)
#                  Triplet-style loss:
#                    L_mine = relu( margin + cos(z, n*) - cos(z, p*) )
#                  Combined:
#                    L_total = L_ce + lambda_sc * SupCon_TW + lambda_m * L_mine
# WHY NOT BEFORE : Hard-negative mining is standard in metric learning (FaceNet,
#                  arXiv 1503.03832) but never conditioned on the GENEALOGY
#                  GRAPH of generators.  Sibling-conditional mining defines a
#                  per-anchor hard-negative POOL that vanilla SupCon cannot,
#                  because in SupCon all wrong-class samples are equal.
# FALSIFIER      : (F1) sibling_confusion_rate (test) AFTER training:
#                       MUST be < CARGO sibling_confusion_rate at the same
#                       fraction.  Equal or higher => mining no-op.
#                  (F2) cos(z, n*) average AFTER training: should be CLOSE TO
#                       cos(z, p*) only at training start; gap should WIDEN
#                       during training.  No-widening = mining failed.
#                  (F3) per-class F1 lift on the most-confused sibling pair
#                       (CoDET: classes {1, 3} = GPT/CodeLlama; AICD: each
#                       triple). Lift > 0 = the right signal.
# REPORTS        : Full eval pack + (F1) sib_confusion_rate post-training
#                                 + (F2) hardest pos/neg cos gap history
#                                 + (F3) per-sibling-pair F1
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
logger = logging.getLogger("exp88_caghost")

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


# ---- CARGO transforms -------------------------------------------------------

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

_PY_T = [_AugAssignExpand, _IfInvert, _ForToWhile, _OpFormSwap]

def aug_python_ast(code, rng):
    try: tree = _ast.parse(code)
    except (SyntaxError, ValueError): return code, "ast_parse_fail"
    cls = _PY_T[rng.randrange(len(_PY_T))]
    tr = cls(rng)
    try:
        new = tr.visit(tree); _ast.fix_missing_locations(new)
        return (_ast.unparse(new), f"py_{cls.NAME}") if tr.fired else (code, f"py_{cls.NAME}_noop")
    except Exception: return code, "ast_unparse_fail"

def reg_aug_assign(code, rng):
    new, n = _re.subn(r"\b([A-Za-z_]\w*)\s*([+\-*/%])=\s*([^;\n]+)",
                      r"\1 = \1 \2 \3", code, count=rng.randint(1, 3))
    return (new, "reg_aug_assign") if n > 0 else (code, "reg_aug_assign_noop")

def reg_paren_canon(code, rng):
    new, n = _re.subn(r"=\s*([^;\n=]+?)([;\n])",
        lambda m: f"= ({m.group(1).strip()}){m.group(2)}" if rng.random() < 0.4 else m.group(0),
        code, count=rng.randint(2, 5))
    return (new, "reg_paren_canon") if n > 0 else (code, "reg_paren_canon_noop")

def reg_if_invert_c(code, rng):
    fired = [False]
    def r(m):
        if rng.random() < 0.4: fired[0] = True; return f"if (!({m.group(1)}))"
        return m.group(0)
    new = _re.compile(r"\bif\s*\(\s*([^()\n]+?)\s*\)").sub(r, code, count=rng.randint(1, 3))
    return (new, "reg_if_invert_c") if fired[0] else (code, "reg_if_invert_c_noop")

_REG = [reg_aug_assign, reg_paren_canon, reg_if_invert_c]

def cargo_augment(code, lang, rng):
    lang_l = (lang or "").lower()
    if lang_l in {"python", "py"} or (lang_l == "" and "def " in code[:200]):
        new, name = aug_python_ast(code, rng)
        if not name.endswith("_noop") and not name.endswith("_fail"):
            return new, name
    fn = _REG[rng.randrange(len(_REG))]
    try: return fn(code, rng)
    except Exception: return code, "reg_fail"


# ---- Model + losses ---------------------------------------------------------

class CAGHOSTModel(nn.Module):
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


def ghost_mine_loss(z, y, sib_mask, margin=0.2):
    """Per-anchor:
       p* = argmin_{j != i, y_j = y_i} cos(z_i, z_j)        (hardest positive)
       n* = argmax_{j: sib_mask[y_i, y_j] = 1} cos(z_i, z_j)  (hardest sibling neg)
       L_mine = mean_i relu(margin + cos(z_i, n*_i) - cos(z_i, p*_i))
    Only counts anchors that have at least one positive AND one sibling negative."""
    N = z.size(0)
    if N < 2: return z.sum() * 0.0, 0.0, 0.0
    sim = z @ z.t()
    eye = torch.eye(N, device=z.device, dtype=torch.bool)
    pos = (y.unsqueeze(0) == y.unsqueeze(1)) & (~eye)
    sib = sib_mask[y][:, y].bool() & (~eye)
    # hardest positive: min sim where pos==True, else +inf
    sim_pos = sim.masked_fill(~pos, float("inf"))
    pmin, _ = sim_pos.min(dim=-1)
    # hardest sibling neg: max sim where sib==True, else -inf
    sim_sib = sim.masked_fill(~sib, float("-inf"))
    nmax, _ = sim_sib.max(dim=-1)
    valid = pos.any(-1) & sib.any(-1)
    if valid.sum() == 0: return z.sum() * 0.0, 0.0, 0.0
    pmin_v = pmin[valid]; nmax_v = nmax[valid]
    loss = F.relu(margin + nmax_v - pmin_v).mean()
    return loss, pmin_v.mean().item(), nmax_v.mean().item()


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
    lambda_sc: float = 0.4; lambda_mine: float = 0.4; margin: float = 0.2
    gamma: float = 1.0; tau: float = 0.1
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


class FSDS_CAGHOST(TD):
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
            logger.info(f"[FSDS_CAGHOST] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]; lang = r.get("language", "") or ""
        e0 = self.tok(code, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        ids0, m0 = e0["input_ids"].squeeze(0), e0["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            new, name = cargo_augment(code, lang, rng)
            e1 = self.tok(new, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
            ids1, m1 = e1["input_ids"].squeeze(0), e1["attention_mask"].squeeze(0)
        else:
            ids1, m1, name = ids0, m0, "none"
        return {"ids0": ids0, "mask0": m0, "ids1": ids1, "mask1": m1,
                "label": r["label"], "aug_name": name,
                "language": lang, "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_cpu, collect_falsifier=False):
    model.eval(); preds, labels = [], []
    cos_list = []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device); labs = b["label"]
        z0, lg = model.encode(ids0, m0)
        preds.extend(lg.argmax(-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            ids1 = b["ids1"].to(cfg.device); m1 = b["mask1"].to(cfg.device)
            z1, _ = model.encode(ids1, m1)
            cos_list.extend((z0 * z1).sum(-1).detach().cpu().float().numpy().tolist())
    preds = np.array(preds); labels = np.array(labels); n = cfg.n_cls
    ov = {"accuracy": float(accuracy_score(labels, preds)),
          "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
          "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0))}
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
    od = int(cm.sum() - cm.trace())
    sib = int(sum(cm[i, j] for i in range(n) for j in range(n) if i != j and sib_mask_np[i, j] > 0))
    # per-class F1
    per_class_f1 = f1_score(labels, preds, average=None, zero_division=0, labels=list(range(n))).tolist()
    out = {"overall": ov, "confusion_matrix": cm.tolist(),
           "sibling_confusion_rate": float(sib / max(od, 1)),
           "off_diag_total": od, "n_samples": int(len(labels)),
           "per_class_f1": per_class_f1}
    if collect_falsifier:
        # F3: per-sibling-pair F1 (CoDET only meaningfully has {1,3}; AICD has multiple)
        sib_pair_f1 = {}
        for i in range(n):
            for j in range(i + 1, n):
                if sib_mask_np[i, j] > 0:
                    pair_lbl_mask = (labels == i) | (labels == j)
                    if pair_lbl_mask.sum() < 2: continue
                    sub_l = labels[pair_lbl_mask]; sub_p = preds[pair_lbl_mask]
                    sub_p = np.where(np.isin(sub_p, [i, j]), sub_p, -1)
                    sib_pair_f1[f"{i}_{j}"] = float(
                        f1_score(sub_l, sub_p, labels=[i, j], average="macro", zero_division=0))
        out["falsifier"] = {
            "mean_view_cos_F1_view": float(np.mean(cos_list)) if cos_list else 0.0,
            "sibling_confusion_rate_F1": float(sib / max(od, 1)),
            "sibling_pair_f1_F3": sib_pair_f1,
        }
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg, dist, sib_mask):
    model.train(); tot, ce_s, sc_s, m_s = 0.0, 0.0, 0.0, 0.0
    pmin_hist, nmax_hist = [], []
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); m1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, lg = model.encode(ids0, m0)
            z1, _ = model.encode(ids1, m1)
            zd = torch.cat([z0, z1], 0); yd = torch.cat([labs, labs], 0)
            loss_ce = F.cross_entropy(lg, labs)
            loss_sc = supcon_tw_loss(zd, yd, dist, gamma=cfg.gamma, tau=cfg.tau)
            loss_m, pmin_, nmax_ = ghost_mine_loss(zd.float(), yd, sib_mask, margin=cfg.margin)
            loss = loss_ce + cfg.lambda_sc * loss_sc + cfg.lambda_mine * loss_m
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item(); m_s += float(loss_m.item())
        pmin_hist.append(pmin_); nmax_hist.append(nmax_)
    n = len(loader)
    return (tot/n, ce_s/n, sc_s/n, m_s/n,
            float(np.mean(pmin_hist) if pmin_hist else 0.0),
            float(np.mean(nmax_hist) if nmax_hist else 0.0))


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dmt = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist = dmt.to(cfg.device); dist_cpu = dmt.numpy()
    sib_t = build_sibling_mask(cfg.n_cls, cfg.gene_adj).to(cfg.device)
    sib_np = sib_t.cpu().numpy()
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
    tr_ds = FSDS_CAGHOST(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_CAGHOST(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=True)
    ts_ds = FSDS_CAGHOST(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=True)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} wu={cfg.warmup} "
                f"lsc={cfg.lambda_sc} lmine={cfg.lambda_mine} margin={cfg.margin}")
    lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = CAGHOSTModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh = 0.0, None, []
    pmin_per_ep, nmax_per_ep = [], []
    for ep in range(cfg.epochs):
        loss, ce, sc, mn, pmin_, nmax_ = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist, sib_t)
        vm = eval_pack(model, vl_dl, cfg, sib_np, dist_cpu)
        v = vm["overall"]["macro_f1"]; vh.append(v)
        pmin_per_ep.append(pmin_); nmax_per_ep.append(nmax_)
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} mine={mn:.4f} "
                    f"pmin={pmin_:.3f} nmax={nmax_:.3f} val={v:.4f}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    tm = eval_pack(model, ts_dl, cfg, sib_np, dist_cpu, collect_falsifier=True)
    test = tm["overall"]["macro_f1"]; gap = best_val - test
    fa = tm["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f} "
                f"sib_conf={fa['sibling_confusion_rate_F1']:.3f} "
                f"sib_pair_f1={fa['sibling_pair_f1_F3']}")
    return {"tag": tag, "method": "CAGHOST", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_sc": cfg.lambda_sc, "lambda_mine": cfg.lambda_mine, "margin": cfg.margin,
            "gamma": cfg.gamma, "tau": cfg.tau,
            "val_macro": best_val, "macro": test, "weighted": tm["overall"]["weighted_f1"],
            "acc": tm["overall"]["accuracy"], "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "test_metrics": tm, "val_history": vh,
            "pmin_per_epoch": pmin_per_ep, "nmax_per_epoch": nmax_per_ep,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp88_caghost_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time() - t0, 1); results.append(r)
                fa = r["test_metrics"]["falsifier"]
                logger.info(f"[{tag}] test={r['macro']:.4f} ({r['dpaper']:+.4f}) gap={r['val_test_gap']:+.4f} "
                            f"sib_conf={fa['sibling_confusion_rate_F1']:.3f} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp88_caghost_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 140)
    print(f"{'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9} {'SibConf':>9} {'Wall':>8}")
    print("-" * 140)
    for r in results:
        fa = r["test_metrics"]["falsifier"]
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['sibling_confusion_rate_F1']:>9.3f} {r['wall']:>8.0f}s")
    print("=" * 140)


if __name__ == "__main__":
    main()
