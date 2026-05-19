# exp86_carmix â€” Dual-distribution view-invariance (CARGO structural + TRACO surface)
# =============================================================================
# NAME           : CARMIX (CARGO âŠ• TRACO dual-distribution invariance)
# ARXIV_ID       : novel
# ONE-LINE CLAIM : Surface-level (token-jitter) and structural-level (AST/CFG-
#                  rewrite) view distributions encode COMPLEMENTARY invariances
#                  of LLM-author identity; the encoder must satisfy BOTH.
# EQUATION       : x_s = T_struct(x);   T_struct in CARGO transform pool
#                  x_t = T_surf  (x);   T_surf   in TRACO transform pool
#                  z, z_s, z_t = phi(x), phi(x_s), phi(x_t)
#                  L_struct = SupCon_TW([z; z_s])
#                  L_surf   = SupCon_TW([z; z_t])
#                  L_total  = L_ce + lambda_s * L_struct + lambda_t * L_surf
# WHY NOT BEFORE : No prior work studies the INTERACTION of two augmentation
#                  distributions for code attribution.  TRACO (surface) and
#                  CARGO (structural) are studied in isolation; CARMIX is the
#                  first joint formulation.  This is the direct empirical
#                  test of "structural views are ORTHOGONAL to surface views".
# FALSIFIER      : (F1) Per-view cosine: cos(z, z_struct) and cos(z, z_surf).
#                       Both > 0.85 = both invariances learned.  One << other
#                       = encoder is biased to one distribution.
#                  (F2) Orthogonality test (post-hoc ablation): macro_f1 at
#                       (lambda_s=1.0, lambda_t=0)   = "CARGO-only run"
#                       (lambda_s=0,   lambda_t=1.0) = "TRACO-only run"
#                       (lambda_s=0.5, lambda_t=0.5) = "CARMIX joint"
#                       JOINT > max(CARGO-only, TRACO-only) at >=1 fraction
#                       => distributions are ORTHOGONAL signal-wise (positive
#                       result).  Equal or worse => surface and structural
#                       are SUBSTITUTES (negative result; still publishable).
#                  (F3) cos(z_s, z_t) on TEST: if > 0.95 the two augmented
#                       views collapse to same point => distributions are
#                       not orthogonal in representation space (failure mode).
# REPORTS        : Full eval pack + (F1) cos(z,z_s), cos(z,z_t)
#                                 + (F2) by-design only main joint run (post-hoc compare)
#                                 + (F3) cos(z_s, z_t)
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
logger = logging.getLogger("exp86_carmix")

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


# ---- CARGO structural transforms --------------------------------------------

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

_PY_TRANSFORMERS = [_AugAssignExpand, _IfInvert, _ForToWhile, _OpFormSwap]

def aug_python_ast(code, rng):
    try: tree = _ast.parse(code)
    except (SyntaxError, ValueError): return code, "ast_parse_fail"
    cls = _PY_TRANSFORMERS[rng.randrange(len(_PY_TRANSFORMERS))]
    tr = cls(rng)
    try:
        new = tr.visit(tree); _ast.fix_missing_locations(new)
        out = _ast.unparse(new)
    except Exception: return code, "ast_unparse_fail"
    return (out, f"py_{cls.NAME}") if tr.fired else (code, f"py_{cls.NAME}_noop")

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

_REG_STRUCT = [reg_aug_assign, reg_paren_canon, reg_if_invert_c]

def cargo_struct(code, lang, rng):
    lang_l = (lang or "").lower()
    if lang_l in {"python", "py"} or (lang_l == "" and "def " in code[:200]):
        new, name = aug_python_ast(code, rng)
        if not name.endswith("_noop") and not name.endswith("_fail"): return new, name
    fn = _REG_STRUCT[rng.randrange(len(_REG_STRUCT))]
    try: return fn(code, rng)
    except Exception: return code, "reg_fail"


# ---- TRACO surface transforms (inlined from exp76) --------------------------

_RESERVED = {"if", "else", "elif", "for", "while", "do", "return", "def", "function",
    "class", "struct", "enum", "import", "from", "include", "public", "private", "void",
    "new", "this", "extends", "implements", "try", "catch", "except", "finally", "with",
    "in", "of", "is", "not", "and", "or", "as", "True", "False", "None", "null",
    "true", "false", "self", "int", "float", "double", "char", "long", "bool", "string",
    "var", "let", "const"}

def aug_token_dropout(code, rng, p=0.1):
    tokens = _re.split(r"(\s+|[^\w\s])", code)
    out = []
    for t in tokens:
        if t.strip() and t.strip() not in _RESERVED and not t.isspace():
            if rng.random() < p: out.append(" "); continue
        out.append(t)
    return "".join(out), "tok_dropout"

def aug_id_rename(code, rng, max_renames=8):
    ids = set(_re.findall(r"\b[a-zA-Z_]\w{2,}\b", code))
    ids = [i for i in ids if i not in _RESERVED and not i[0].isdigit()]
    if not ids: return code, "id_rename_noop"
    n = min(max_renames, len(ids)); chosen = rng.sample(ids, n)
    new = code
    for k, orig in enumerate(chosen):
        new = _re.sub(rf"\b{_re.escape(orig)}\b", f"v{k}", new)
    return new, "id_rename"

def aug_ws_jitter(code, rng, p=0.15):
    ops = ["+", "-", "*", "/", "%", "=", "<", ">", ",", ";"]
    out = []
    for c in code:
        out.append(c)
        if c in ops and rng.random() < p: out.append(" ")
    return "".join(out), "ws_jitter"

def aug_comment_strip(code, rng):
    new = _re.sub(r"/\*[\s\S]*?\*/", "", code)
    new = _re.sub(r"//[^\n]*", "", new)
    new = _re.sub(r"#[^\n]*", "", new)
    return new, "comment_strip"

_SURF = [aug_token_dropout, aug_id_rename, aug_ws_jitter, aug_comment_strip]

def traco_surf(code, lang, rng):
    fn = _SURF[rng.randrange(len(_SURF))]
    try: return fn(code, rng)
    except Exception: return code, "surf_fail"


# ---- Model + loss -----------------------------------------------------------

class CARMIXModel(nn.Module):
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
    lambda_s: float = 0.35; lambda_t: float = 0.35; gamma: float = 1.0; tau: float = 0.1
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
        if mem >= 40: c.bs, c.seq = 96, 512
        elif mem >= 10: c.bs, c.seq = 48, 384
        else: c.bs, c.seq = 24, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={c.bs} (3x view) seq={c.seq}")
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


class FSDS_CARMIX(TD):
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
            logger.info(f"[FSDS_CARMIX] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]; lang = r.get("language", "") or ""
        e0 = self.tok(code, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        ids0, m0 = e0["input_ids"].squeeze(0), e0["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            xs, ns = cargo_struct(code, lang, rng)
            xt, nt = traco_surf(code, lang, rng)
            es = self.tok(xs, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
            et = self.tok(xt, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
            ids_s, m_s = es["input_ids"].squeeze(0), es["attention_mask"].squeeze(0)
            ids_t, m_t = et["input_ids"].squeeze(0), et["attention_mask"].squeeze(0)
        else:
            ids_s, m_s, ids_t, m_t, ns, nt = ids0, m0, ids0, m0, "none", "none"
        return {"ids0": ids0, "mask0": m0, "ids_s": ids_s, "mask_s": m_s,
                "ids_t": ids_t, "mask_t": m_t, "label": r["label"],
                "name_s": ns, "name_t": nt, "language": lang, "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_cpu, collect_falsifier=False):
    model.eval(); preds, labels = [], []
    cos_s, cos_t, cos_st = [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device); labs = b["label"]
        z0, lg = model.encode(ids0, m0)
        preds.extend(lg.argmax(-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            ids_s = b["ids_s"].to(cfg.device); m_s = b["mask_s"].to(cfg.device)
            ids_t = b["ids_t"].to(cfg.device); m_t = b["mask_t"].to(cfg.device)
            zs, _ = model.encode(ids_s, m_s); zt, _ = model.encode(ids_t, m_t)
            cos_s.extend((z0 * zs).sum(-1).detach().cpu().float().numpy().tolist())
            cos_t.extend((z0 * zt).sum(-1).detach().cpu().float().numpy().tolist())
            cos_st.extend((zs * zt).sum(-1).detach().cpu().float().numpy().tolist())
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
        out["falsifier"] = {
            "mean_cos_struct_F1": float(np.mean(cos_s)) if cos_s else 0.0,
            "mean_cos_surf_F1": float(np.mean(cos_t)) if cos_t else 0.0,
            "mean_cos_struct_vs_surf_F3": float(np.mean(cos_st)) if cos_st else 0.0,
        }
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg, dist):
    model.train(); tot, ce_s, ss_s, st_s = 0.0, 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device)
        ids_s = b["ids_s"].to(cfg.device); m_s = b["mask_s"].to(cfg.device)
        ids_t = b["ids_t"].to(cfg.device); m_t = b["mask_t"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, lg = model.encode(ids0, m0)
            zs, _ = model.encode(ids_s, m_s)
            zt, _ = model.encode(ids_t, m_t)
            zd_s = torch.cat([z0, zs], 0); yd = torch.cat([labs, labs], 0)
            zd_t = torch.cat([z0, zt], 0)
            loss_ce = F.cross_entropy(lg, labs)
            loss_ss = supcon_tw_loss(zd_s, yd, dist, gamma=cfg.gamma, tau=cfg.tau)
            loss_st = supcon_tw_loss(zd_t, yd, dist, gamma=cfg.gamma, tau=cfg.tau)
            loss = loss_ce + cfg.lambda_s * loss_ss + cfg.lambda_t * loss_st
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); ss_s += loss_ss.item(); st_s += loss_st.item()
    n = len(loader)
    return tot/n, ce_s/n, ss_s/n, st_s/n


def run_exp(cfg, tag):
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
    tr_ds = FSDS_CARMIX(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_CARMIX(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=True)
    ts_ds = FSDS_CARMIX(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=True)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} wu={cfg.warmup} "
                f"ls={cfg.lambda_s} lt={cfg.lambda_t}")
    lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = CARMIXModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh = 0.0, None, []
    for ep in range(cfg.epochs):
        loss, ce, ss, st = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist)
        vm = eval_pack(model, vl_dl, cfg, sib_np, dist_cpu)
        v = vm["overall"]["macro_f1"]; vh.append(v)
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce={ce:.4f} ss={ss:.4f} st={st:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    tm = eval_pack(model, ts_dl, cfg, sib_np, dist_cpu, collect_falsifier=True)
    test = tm["overall"]["macro_f1"]; gap = best_val - test
    fa = tm["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f} "
                f"cos_s={fa['mean_cos_struct_F1']:.3f} cos_t={fa['mean_cos_surf_F1']:.3f} "
                f"cos_s_vs_t={fa['mean_cos_struct_vs_surf_F3']:.3f}")
    return {"tag": tag, "method": "CARMIX", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_s": cfg.lambda_s, "lambda_t": cfg.lambda_t, "gamma": cfg.gamma, "tau": cfg.tau,
            "val_macro": best_val, "macro": test, "weighted": tm["overall"]["weighted_f1"],
            "acc": tm["overall"]["accuracy"], "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "test_metrics": tm, "val_history": vh,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp86_carmix_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time() - t0, 1); results.append(r)
                fa = r["test_metrics"]["falsifier"]
                logger.info(f"[{tag}] test={r['macro']:.4f} ({r['dpaper']:+.4f}) gap={r['val_test_gap']:+.4f} "
                            f"cos_s={fa['mean_cos_struct_F1']:.3f} cos_t={fa['mean_cos_surf_F1']:.3f} "
                            f"t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp86_carmix_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 140)
    print(f"{'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'cos_s':>8} {'cos_t':>8} {'cos_s/t':>9} {'Wall':>8}")
    print("-" * 140)
    for r in results:
        fa = r["test_metrics"]["falsifier"]
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['mean_cos_struct_F1']:>8.3f} {fa['mean_cos_surf_F1']:>8.3f} "
              f"{fa['mean_cos_struct_vs_surf_F3']:>9.3f} {r['wall']:>8.0f}s")
    print("=" * 140)


if __name__ == "__main__":
    main()
