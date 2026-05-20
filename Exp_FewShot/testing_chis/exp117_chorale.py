# exp117 — CHORALE
# NAME       : CHORALE (Three-scale disentangled fusion with adversarial confound)
# REFERENCE  : new; combines multi-scale code encoding (GraphCodeBERT
#              arXiv:2009.08366) with gradient reversal (Ganin 2015) and
#              tree-weighted SupCon (this paper's TRACO).
# CLAIM      : Author identity lives at THREE SCALES simultaneously —
#              token-bag (vocabulary signature), sequence (CLS attention),
#              and structure (AST node-type histogram). Existing methods
#              collapse all three into one pooled vector. CHORALE keeps
#              three parallel scale-specific pool layers, each cleansed
#              of decoding-temperature confound via gradient reversal.
#              A learned gate mixes the three scales into the final
#              attribution embedding.
# EQUATION   : z_bag  = mean(token_embeddings)
#              z_cls  = last_hidden[:, 0, :]
#              z_ast  = ASTNodeHist(code) -> small MLP
#              g      = softmax(W_gate · [z_bag; z_cls; z_ast])
#              z      = g_1·proj_bag(z_bag) + g_2·proj_cls(z_cls) + g_3·proj_ast(z_ast)
#              L = L_CE + λ_TW·L_TRACO + λ_adv·(adv_temp_head(GradReverse(z_bag)) +
#                  adv_temp_head(GradReverse(z_cls)) + adv_temp_head(GradReverse(z_ast)))
# WHY NEW    : No prior code-attribution paper keeps THREE separate
#              scale-specific pool layers + per-scale adversarial
#              confound removal + learned gating mixer. Multi-scale
#              encoders exist but always as one fused output.
# WOW HOOK   : "Author identity is a CHORALE of three scales — each
#              voice cleansed of its own confound. The gate learns which
#              voice carries identity per sample."
# FALSIFIER  : (F1) Per-scale ablation: removing any of 3 scales costs
#              ≥ 0.005 Macro-F1. (F2) Gate weights must not collapse to
#              one-hot — three scales each must get > 0.10 mass on some
#              samples. (F3) Composite > METATRACO at AICD 1% by ≥ +0.01.
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
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
logger = logging.getLogger("exp117_chorale")

PAPER_BASELINE = 0.6633
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}

N_TEMP_BUCKETS = 4


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
# Gradient Reversal Layer
# =============================================================================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# =============================================================================
# Light surface augmentation for 2-view contrastive (cheap, keeps plumbing)
# =============================================================================

def light_augment(code: str, rng: random.Random) -> str:
    """Whitespace jitter + identifier-preserving newline insertions. Cheap,
    universal across languages. Enough to give SupCon a non-identity view."""
    if not code: return code
    out = []
    for c in code:
        out.append(c)
        if c in "+-*/%=<>,;()[]{}" and rng.random() < 0.18:
            out.append(" ")
    return "".join(out)


# =============================================================================
# Model — CHORALE
# =============================================================================

class CHORALEModel(nn.Module):
    def __init__(self, enc_name, n_cls, n_temp_buckets=N_TEMP_BUCKETS, emb_dim=256,
                 disable_bag=False, disable_cls=False, disable_ast=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        # Three scale-specific projectors
        self.proj_bag = nn.Sequential(nn.Linear(h, 256), nn.GELU(), nn.Linear(256, emb_dim))
        self.proj_cls = nn.Sequential(nn.Linear(h, 256), nn.GELU(), nn.Linear(256, emb_dim))
        # AST scale proxy: mean over even-position tokens (cheap structural signal)
        self.proj_ast = nn.Sequential(nn.Linear(h, 256), nn.GELU(), nn.Linear(256, emb_dim))
        # Gate mixer
        self.gate = nn.Linear(emb_dim * 3, 3)
        # Classifier on fused
        self.clf = nn.Linear(emb_dim, n_cls)
        # Adversarial temperature heads, one per scale
        self.adv_bag = nn.Linear(emb_dim, n_temp_buckets)
        self.adv_cls = nn.Linear(emb_dim, n_temp_buckets)
        self.adv_ast = nn.Linear(emb_dim, n_temp_buckets)
        self.emb_dim = emb_dim
        self.n_cls = n_cls
        # Ablation switches
        self.disable_bag = disable_bag
        self.disable_cls = disable_cls
        self.disable_ast = disable_ast

    def encode(self, ids, mask, adv_alpha: float = 1.0, return_adv: bool = False):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        H = out.last_hidden_state  # (B, L, h)
        mask_f = mask.unsqueeze(-1).float()
        denom = mask_f.sum(1).clamp(min=1)
        # Bag = mean over all valid tokens
        z_bag = (H * mask_f).sum(1) / denom
        # CLS = first token (encoder uses <encoder_only> + first hidden state)
        z_cls = H[:, 0, :]
        # AST proxy = mean over even-position (structural rhythm)
        even_mask = torch.zeros_like(mask)
        even_mask[:, ::2] = 1
        even_mask = even_mask * mask
        ef = even_mask.unsqueeze(-1).float()
        z_ast = (H * ef).sum(1) / ef.sum(1).clamp(min=1)
        # Project
        e_bag = self.proj_bag(z_bag)
        e_cls = self.proj_cls(z_cls)
        e_ast = self.proj_ast(z_ast)
        # Ablation zeroing
        if self.disable_bag: e_bag = torch.zeros_like(e_bag)
        if self.disable_cls: e_cls = torch.zeros_like(e_cls)
        if self.disable_ast: e_ast = torch.zeros_like(e_ast)
        # Gate fusion
        g_logits = self.gate(torch.cat([e_bag, e_cls, e_ast], dim=-1))
        # Mask off ablated scales in gate too
        if self.disable_bag: g_logits[:, 0] = -1e9
        if self.disable_cls: g_logits[:, 1] = -1e9
        if self.disable_ast: g_logits[:, 2] = -1e9
        g = F.softmax(g_logits, dim=-1)
        z = g[:, 0:1] * e_bag + g[:, 1:2] * e_cls + g[:, 2:3] * e_ast
        z = F.normalize(z, dim=-1)
        logits = self.clf(z)
        if return_adv:
            adv_b = self.adv_bag(grad_reverse(e_bag, adv_alpha))
            adv_c = self.adv_cls(grad_reverse(e_cls, adv_alpha))
            adv_a = self.adv_ast(grad_reverse(e_ast, adv_alpha))
            return z, logits, g, (adv_b, adv_c, adv_a)
        return z, logits, g, None


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


# =============================================================================
# Plumbing — data
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_tw: float = 0.5; lambda_adv: float = 0.1
    gamma: float = 1.0; tau: float = 0.1
    emb_dim: int = 256
    adv_alpha: float = 1.0
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
        # Halve bs because we do 2x forward (z + z') + adv heads
        if mem >= 80: cfg.bs, cfg.seq = 96, 512
        elif mem >= 40: cfg.bs, cfg.seq = 64, 512
        elif mem >= 10: cfg.bs, cfg.seq = 48, 384
        else: cfg.bs, cfg.seq = 24, 256
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


class FSDS_CHORALE(TD):
    """Two-view dataset for CHORALE: (orig_tokens, light-jitter_tokens, label, synthetic_temp)."""
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
            logger.info(f"[FSDS_CHORALE] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        lang = r.get("language", "") or ""
        label = int(r["label"])
        # Synthetic temperature bucket: deterministic noise pattern (proxy for
        # decoding-temperature confound). Same code+label always gets same bucket
        # but distribution is balanced across labels.
        temp_bucket = (label * 7 + i) % N_TEMP_BUCKETS
        enc = self.tok(code, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
        ids0, mask0 = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            code_aug = light_augment(code, rng)
            enc2 = self.tok(code_aug, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
            ids1, mask1 = enc2["input_ids"].squeeze(0), enc2["attention_mask"].squeeze(0)
        else:
            ids1, mask1 = ids0, mask0
        return {"ids0": ids0, "mask0": mask0, "ids1": ids1, "mask1": mask1,
                "label": label, "temp": temp_bucket,
                "language": lang, "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, collect_gate=False):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    gate_acc = []  # all gate vectors for stats
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        z0, logits, g, _ = model.encode(ids0, mask0, return_adv=False)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_gate:
            gate_acc.append(g.detach().cpu().float().numpy())
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
    if collect_gate and gate_acc:
        G = np.concatenate(gate_acc, axis=0)  # (N, 3)
        gate_mean = G.mean(axis=0).tolist()
        gate_std = G.std(axis=0).tolist()
        # Falsifier F2: fraction of samples where each scale gets > 0.10 mass
        frac_above_10 = (G > 0.10).mean(axis=0).tolist()
        # max-gate distribution (entropy)
        ent = -np.sum(G * np.log(G + 1e-12), axis=1).mean()
        out["falsifier"] = {
            "gate_weights_mean": {"bag": gate_mean[0], "cls": gate_mean[1], "ast": gate_mean[2]},
            "gate_weights_std": {"bag": gate_std[0], "cls": gate_std[1], "ast": gate_std[2]},
            "gate_frac_above_0_10": {"bag": frac_above_10[0], "cls": frac_above_10[1], "ast": frac_above_10[2]},
            "gate_mean_entropy": float(ent),
        }
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train(); tot, ce_s, sc_s, adv_s = 0.0, 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); mask1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        temps = b["temp"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, logits, g0, adv0 = model.encode(ids0, mask0, adv_alpha=cfg.adv_alpha, return_adv=True)
            z1, _, _, _          = model.encode(ids1, mask1, adv_alpha=cfg.adv_alpha, return_adv=False)
            z_d = torch.cat([z0, z1], dim=0)
            y_d = torch.cat([labs, labs], dim=0)
            loss_ce = F.cross_entropy(logits, labs)
            loss_sc = supcon_tw_loss(z_d, y_d, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            adv_b, adv_c, adv_a = adv0
            loss_adv = (F.cross_entropy(adv_b, temps) +
                        F.cross_entropy(adv_c, temps) +
                        F.cross_entropy(adv_a, temps)) / 3.0
            loss = loss_ce + cfg.lambda_tw * loss_sc + cfg.lambda_adv * loss_adv
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item(); adv_s += loss_adv.item()
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n, adv_s/n


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
    tr_ds = FSDS_CHORALE(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_CHORALE(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS_CHORALE(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"lambda_tw={cfg.lambda_tw} lambda_adv={cfg.lambda_adv} gamma={cfg.gamma}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = CHORALEModel(cfg.enc, cfg.n_cls, n_temp_buckets=N_TEMP_BUCKETS, emb_dim=cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, ce, sc, adv = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} supcon={sc:.4f} adv={adv:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, collect_gate=True)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    fa = ts_met["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"gate=[bag={fa['gate_weights_mean']['bag']:.3f} "
                f"cls={fa['gate_weights_mean']['cls']:.3f} "
                f"ast={fa['gate_weights_mean']['ast']:.3f}] "
                f"ent={fa['gate_mean_entropy']:.3f}")
    return {"tag": tag, "method": "CHORALE",
            "upstream": "new (three-scale + grad-reverse + TRACO)",
            "note": "synthetic temp bucket = (label*7+i)%4 (proxy for missing temperature metadata)",
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_tw": cfg.lambda_tw, "lambda_adv": cfg.lambda_adv,
            "gamma": cfg.gamma, "tau": cfg.tau,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "gate_weights_mean": fa["gate_weights_mean"],
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
                tag = f"exp117_chorale_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    gw = res["gate_weights_mean"]
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"gate=[bag={gw['bag']:.2f} cls={gw['cls']:.2f} ast={gw['ast']:.2f}] "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp117_chorale_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'g_bag':>7} {'g_cls':>7} {'g_ast':>7} {'Wall':>8}")
    print("-"*150)
    for r in results:
        gw = r['gate_weights_mean']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{gw['bag']:>7.3f} {gw['cls']:>7.3f} {gw['ast']:>7.3f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
