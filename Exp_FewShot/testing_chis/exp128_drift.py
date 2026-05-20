# exp128 — DRIFT
# NAME       : DRIFT (Diffusion Refinement of embedding Identity through Time)
# REFERENCE  : new; score-based generative modelling (Song et al. 2021,
#              arXiv:2011.13456), DDPM (Ho 2020), classifier-free guidance
#              ideas re-purposed for discriminative attribution.
# CLAIM      : Authorship classification can be cast as a DIFFUSION
#              PROBLEM in embedding space. We train a score network
#              s_theta(z_t, t) that learns to denoise corrupted author
#              embeddings back to clean ones. At inference, the query
#              embedding is run through T denoising steps; the convergent
#              embedding lands closest to its author's basin of attraction
#              on the embedding manifold. Diffusion replaces softmax.
# EQUATION   : Pretext: train encoder to produce z = phi(x) in R^d_z
#              Add Gaussian noise: z_t = sqrt(alpha_t)*z + sqrt(1-alpha_t)*eps
#              Score net: s_theta(z_t, t, y) tries to predict eps
#              L_score = E[||s_theta(z_t, t, y) - eps||^2]   (y = class conditioning)
#              At test, classify by:
#                For each candidate y: run T DDIM steps from query_emb
#                with conditioning y; compute final reconstruction error.
#                argmin_y reconstruction_error(query, y)
# WHY NEW    : No prior code-attribution method uses diffusion. Generative
#              attribution (FLUX-style flow) has been proposed but never
#              implemented for code. DRIFT does it with a small score
#              network on top of the TRACO embedding — re-using existing
#              encoder, adding only a per-class score head.
# WOW HOOK   : "Authorship is not classification — it is denoising.
#              We train a score field for each author; at inference, the
#              query embedding DRIFTS through the field until it settles
#              into the basin of its author. The closest basin wins."
# FALSIFIER  : (F1) Score MSE on test held-out samples > 0.0 and < 0.5
#              (non-trivial denoising signal). (F2) DDIM-trajectory
#              prediction beats single-step argmax classifier by >= 0.005
#              at 1% slot. (F3) Removing class-conditioning (unconditional
#              score) loses >= 0.01 => conditioning is doing real work.
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
logger = logging.getLogger("exp128_drift")

PAPER_BASELINE = 0.6633
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


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
# CARGO structural augmentations (re-used for 2-view contrastive)
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


_PY_TRANSFORMERS = [_AugAssignExpand, _IfInvert, _ForToWhile]


def aug_python_ast(code: str, rng: random.Random) -> Tuple[str, str]:
    try:
        _ast.parse(code)
    except (SyntaxError, ValueError):
        return code, "ast_parse_fail"
    order = list(range(len(_PY_TRANSFORMERS)))
    rng.shuffle(order)
    for idx in order:
        cls = _PY_TRANSFORMERS[idx]
        transformer = cls(rng)
        try:
            new_tree = transformer.visit(_ast.parse(code))
            _ast.fix_missing_locations(new_tree)
            new_code = _ast.unparse(new_tree)
        except Exception:
            continue
        if transformer.fired > 0:
            return new_code, f"py_{cls.NAME}"
    return code, "ast_all_noop"


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


def reg_paren_canon(code: str, rng: random.Random) -> Tuple[str, str]:
    new, n = _re.subn(
        r"=\s*([^;\n=]+?)([;\n])",
        lambda m: f"= ({m.group(1).strip()}){m.group(2)}" if rng.random() < 0.4 else m.group(0),
        code, count=rng.randint(2, 5))
    return (new, "reg_paren_canon") if n > 0 else (code, "reg_paren_canon_noop")


_REG_TRANSFORMS = [reg_aug_assign, reg_paren_canon]


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
# DRIFT model: encoder + projector + classifier + score network
# =============================================================================

class DRIFT(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256, score_hidden=512, T=1000):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        # Score network: input = (z_t, t_embed, y_embed) -> predict eps of shape z_t
        self.t_embed = nn.Embedding(T, 64)
        self.y_embed = nn.Embedding(n_cls, 64)
        self.score_net = nn.Sequential(
            nn.Linear(emb_dim + 64 + 64, score_hidden),
            nn.GELU(),
            nn.Linear(score_hidden, score_hidden),
            nn.GELU(),
            nn.Linear(score_hidden, emb_dim),
        )
        self.emb_dim = emb_dim
        self.n_cls = n_cls
        # Noise schedule (linear beta from 0.0001 to 0.02)
        betas = torch.linspace(0.0001, 0.02, T)
        alphas = 1 - betas
        alpha_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", alpha_cumprod)
        self.T = T

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        z_n = F.normalize(z, dim=-1)
        return z_n, self.clf(z_n)

    def score(self, z_t, t, y):
        """Predict eps from noisy embedding z_t at timestep t conditioned on class y."""
        te = self.t_embed(t)  # (B, 64)
        ye = self.y_embed(y)  # (B, 64)
        inp = torch.cat([z_t, te, ye], dim=-1)
        return self.score_net(inp)

    def add_noise(self, z, t):
        """Forward diffusion: q(z_t | z_0)."""
        eps = torch.randn_like(z)
        a_cp = self.alpha_cumprod[t].unsqueeze(-1)  # (B, 1)
        z_t = torch.sqrt(a_cp) * z + torch.sqrt(1 - a_cp) * eps
        return z_t, eps

    def ddim_sample(self, z_query, y_candidate, n_steps=10):
        """Iteratively denoise starting from z_query under conditioning y_candidate."""
        z = z_query.clone()
        step_indices = torch.linspace(self.T - 1, 0, n_steps).long().to(z.device)
        for i in range(n_steps - 1):
            t = step_indices[i].unsqueeze(0).expand(z.size(0))
            t_next = step_indices[i + 1].unsqueeze(0).expand(z.size(0))
            eps_pred = self.score(z, t, y_candidate)
            a_cp = self.alpha_cumprod[t].unsqueeze(-1)
            a_cp_next = self.alpha_cumprod[t_next].unsqueeze(-1)
            # DDIM: predict z_0 then move toward t_next
            z0_pred = (z - torch.sqrt(1 - a_cp) * eps_pred) / torch.sqrt(a_cp).clamp(min=1e-6)
            z = torch.sqrt(a_cp_next) * z0_pred + torch.sqrt(1 - a_cp_next) * eps_pred
        return z


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

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 96; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    emb_dim: int = 256
    # DRIFT-specific
    lambda_score: float = 0.3
    ddim_steps: int = 10
    T_diffusion: int = 1000
    score_hidden: int = 512
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
        # DRIFT: 2-view encoder + score net. Halve relative to single-view; score net small.
        if mem >= 80: cfg.bs, cfg.seq = 96, 512
        elif mem >= 40: cfg.bs, cfg.seq = 64, 512
        elif mem >= 10: cfg.bs, cfg.seq = 32, 384
        else: cfg.bs, cfg.seq = 16, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} (DRIFT: 2-view + score) seq={cfg.seq}")
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


class FSDS_DRIFT(TD):
    """Dataset returning two views (original + CARGO structurally-augmented)."""
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
            logger.info(f"[FSDS_DRIFT] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def _tokenize(self, code):
        enc_text = "<encoder_only>" + code
        enc = self.tok(enc_text, max_length=self.seq_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        lang = r.get("language", "") or ""
        ids0, mask0 = self._tokenize(code)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            code_aug, aug_name = cargo_augment(code, lang, rng)
            ids1, mask1 = self._tokenize(code_aug)
        else:
            ids1, mask1, aug_name = ids0, mask0, "none"
        return {"ids0": ids0, "mask0": mask0, "ids1": ids1, "mask1": mask1,
                "label": r["label"], "aug_name": aug_name,
                "language": lang, "source": r.get("source", "") or ""}


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train()
    tot, ce_s, sc_s, score_s = 0.0, 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device)
        ids2 = b["ids1"].to(cfg.device); m2 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z, logits = model.encode(ids, mask)
            z2, _ = model.encode(ids2, m2)
            # Standard CE + tree-weighted SupCon (2-view CARGO)
            loss_ce = F.cross_entropy(logits, labs)
            z_d = torch.cat([z, z2], dim=0); y_d = torch.cat([labs, labs], dim=0)
            loss_sc = supcon_tw_loss(z_d, y_d, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            # Score loss: sample random t, add noise, predict eps with TRUE label conditioning
            t = torch.randint(0, model.T, (z.size(0),), device=z.device)
            # detach z so score head doesn't backprop through encoder; encoder is shaped by CE+supcon
            z_clean = z.detach().float()
            z_noisy, eps_true = model.add_noise(z_clean, t)
            eps_pred = model.score(z_noisy, t, labs)
            loss_score = ((eps_pred - eps_true) ** 2).mean()
            loss = loss_ce + cfg.lambda_aug * loss_sc + cfg.lambda_score * loss_score
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += float(loss.item()); ce_s += float(loss_ce.item())
        sc_s += float(loss_sc.item()); score_s += float(loss_score.item())
    n = max(len(loader), 1)
    return tot/n, ce_s/n, sc_s/n, score_s/n


@torch.no_grad()
def eval_pack_drift(model, loader, cfg, sib_mask_np, dist_mat_cpu):
    """Eval with BOTH the discriminative head (main) AND the DDIM-trajectory predictor (drift)."""
    model.eval()
    preds_main, preds_drift, labels, langs, sources = [], [], [], [], []
    score_mses = []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device); labs = b["label"]
        z, logits = model.encode(ids, mask)
        preds_main.extend(logits.argmax(-1).cpu().tolist())
        # Drift prediction: for each class, run DDIM trajectory, pick argmin reconstruction error
        B = z.size(0)
        z_f = z.float()
        drift_scores = torch.full((B, model.n_cls), float("-inf"), device=z.device)
        for c in range(model.n_cls):
            y_c = torch.full((B,), c, dtype=torch.long, device=z.device)
            z_denoised = model.ddim_sample(z_f, y_c, n_steps=cfg.ddim_steps)
            # Score = -||z_query - z_denoised||^2 (closer = better fit to class c)
            drift_scores[:, c] = -((z_f - z_denoised) ** 2).sum(-1)
        preds_drift.extend(drift_scores.argmax(-1).cpu().tolist())
        # Score MSE proxy at random t for monitoring (using true label)
        labs_t = labs.to(cfg.device) if torch.is_tensor(labs) else torch.tensor(list(labs), device=cfg.device)
        t = torch.randint(0, model.T, (z.size(0),), device=z.device)
        z_noisy, eps_true = model.add_noise(z_f, t)
        eps_pred = model.score(z_noisy, t, labs_t)
        score_mses.append(float(((eps_pred - eps_true) ** 2).mean().item()))
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        lang_batch = b.get("language", [""] * len(labs))
        src_batch = b.get("source", [""] * len(labs))
        langs.extend(list(lang_batch) if not isinstance(lang_batch, list) else lang_batch)
        sources.extend(list(src_batch) if not isinstance(src_batch, list) else src_batch)
    preds_main = np.array(preds_main); preds_drift = np.array(preds_drift); labels = np.array(labels)
    n_cls = cfg.n_cls
    # Drift = headline. Main = backup.
    preds = preds_drift
    overall = {"accuracy": float(accuracy_score(labels, preds)),
               "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
               "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
               "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
               "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0))}
    overall_main = {"accuracy": float(accuracy_score(labels, preds_main)),
                    "macro_f1": float(f1_score(labels, preds_main, average="macro", zero_division=0)),
                    "weighted_f1": float(f1_score(labels, preds_main, average="weighted", zero_division=0))}
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
    return {"overall": overall, "overall_main": overall_main,
            "per_class": per_class, "per_language": per_lang, "per_source": per_src,
            "confusion_matrix": cm.tolist(), "sibling_confusion_rate": float(sib_rate),
            "cross_family_confusion_rate": float(cross_rate),
            "off_diag_total": off_diag, "n_samples": int(len(labels)),
            "eval_score_mse_mean": float(np.mean(score_mses)) if score_mses else 0.0}


@torch.no_grad()
def eval_quick_main(model, loader, cfg):
    """Quick eval using only the discriminative head (used during finetune for early stopping)."""
    model.eval(); preds, labels = [], []
    for b in tqdm(loader, desc="ValQuick"):
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device); labs = b["label"]
        _, logits = model.encode(ids, mask)
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
    return float(f1_score(np.array(labels), np.array(preds), average="macro", zero_division=0))


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
    tr_ds = FSDS_DRIFT(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_DRIFT(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS_DRIFT(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"lambda_aug={cfg.lambda_aug} lambda_score={cfg.lambda_score} ddim_steps={cfg.ddim_steps}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = DRIFT(cfg.enc, cfg.n_cls, cfg.emb_dim, cfg.score_hidden, T=cfg.T_diffusion).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    score_history = []
    for epoch in range(cfg.epochs):
        loss, ce, sc, score_l = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        score_history.append(score_l)
        # Use quick main-head val for early stopping (DDIM is too slow per-epoch)
        v_main = eval_quick_main(model, vl_dl, cfg)
        val_hist.append(v_main)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} supcon={sc:.4f} "
                    f"score={score_l:.4f} val_main={v_main:.4f}")
        if v_main > best_val:
            best_val = v_main
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    # Final eval with BOTH heads
    ts_met = eval_pack_drift(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    # Headline = drift
    test_macro = ts_met["overall"]["macro_f1"]
    test_macro_main = ts_met["overall_main"]["macro_f1"]
    # Recompute best_val using drift on val for an apples-to-apples val/test gap
    # (val_hist tracked main; recompute drift-val on best model)
    val_met_drift = eval_pack_drift(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
    val_macro_drift = val_met_drift["overall"]["macro_f1"]
    gap = val_macro_drift - test_macro
    delta_drift_minus_main = test_macro - test_macro_main
    final_score_mse = ts_met["eval_score_mse_mean"]
    logger.info(f"[final] val_drift={val_macro_drift:.4f} test_drift={test_macro:.4f} "
                f"test_main={test_macro_main:.4f} delta={delta_drift_minus_main:+.4f} "
                f"gap={gap:+.4f} score_mse={final_score_mse:.5f}")
    return {"tag": tag, "method": "DRIFT",
            "note": "Diffusion Refinement of embedding Identity through Time (DDIM-trajectory attribution)",
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "lambda_score": cfg.lambda_score,
            "ddim_steps": cfg.ddim_steps, "T_diffusion": cfg.T_diffusion,
            "gamma": cfg.gamma, "tau": cfg.tau,
            "val_macro": val_macro_drift, "macro": test_macro,
            "macro_main": test_macro_main,
            "delta_drift_minus_main": float(delta_drift_minus_main),
            "score_loss_final": float(score_history[-1]) if score_history else float("nan"),
            "score_history": [float(x) for x in score_history],
            "eval_score_mse": float(final_score_mse),
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
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
                tag = f"exp128_drift_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test_drift={res['macro']:.4f} test_main={res['macro_main']:.4f} "
                                f"({res['dpaper']:+.4f}) gap={res['val_test_gap']:+.4f} "
                                f"score_mse={res['eval_score_mse']:.5f} time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp128_drift_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Main-F1':>8} {'dDrift':>8} {'Gap':>8} {'dPaper':>9} {'ScoreMSE':>10} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['macro_main']:>8.4f} "
              f"{r['delta_drift_minus_main']:>+8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['eval_score_mse']:>10.5f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
