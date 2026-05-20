# exp118 — PRISM
# NAME       : PRISM (Projection-onto-Refracted-Identity-Spectrum Manifold)
# REFERENCE  : new; combines Equiangular Tight Frame (Papyan 2020
#              neural collapse) + DCT spectral filtering + sibling
#              repulsion + TRACO supcon.
# CLAIM      : The standard learned-W K-way classifier is the wrong
#              choice. Replace it with (i) DCT pre-filter on per-token
#              hidden states (low-frequency = style; high-frequency
#              = decoding noise), (ii) FIXED ETF geometry as the
#              classifier with sibling-repulsion bonus baked in, and
#              (iii) encoder learns to refract code onto the prism.
# EQUATION   : H = encoder(x)  ∈ R^{B×L×d}
#              DCT_K(H) = first K=16 cosine coefficients along L axis
#              z = LayerNorm(flatten(DCT_K(H)) @ W_proj) ∈ R^256
#              W = ETF(n_cls, 256) + δ·SiblingRepulse(adj)   (no grad)
#              logits = cos(z, W)  → softmax for CE
#              L = CE(logits, y) + λ · TRACO_supcon(z, y, dist_mat)
# WHY NEW    : ETF-Simplex (exp_n09) used vanilla ETF; nobody added
#              spectral pre-filter OR sibling-repulsion bonus into
#              the FIXED classifier. The combination "designed geometry
#              + spectral filter + encoder learns projection" is new.
# WOW HOOK   : "We replace the classifier with a PRISM — hand-designed
#              geometry that encodes sibling repulsion. The encoder
#              learns to refract code onto the prism after filtering
#              out high-frequency decoder noise."
# FALSIFIER  : (F1) cos(z, W_y) > cos(z, W_y') for y ≠ y' on test.
#              (F2) F1 at K=16 ≈ F1 at K=L (full spectrum) within 0.005.
#              (F3) Replacing fixed W with learnable W loses ≥ 0.01 at 1%.
#              (F4) Composite > TRACO by ≥ +0.005.
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
logger = logging.getLogger("exp118_prism")

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
# ETF construction + sibling repulsion bonus
# =============================================================================

def build_etf(K: int, d: int, sigma: float = 1.0) -> torch.Tensor:
    """Equiangular Tight Frame: K row vectors in R^d, maximally equiangular.
    Following Papyan 2020 neural-collapse form:
        M = sqrt(K/(K-1)) * (I_K - (1/K) * 1·1^T)   (K×K, rank K-1)
    We embed this K×K matrix into R^d by zero-padding columns when d >= K,
    or by truncating when d < K. Rows are then unit-normalized.
    """
    M = torch.eye(K) - torch.ones(K, K) / K
    M = M * math.sqrt(K / max(K - 1, 1))
    if d >= K:
        W = torch.cat([M, torch.zeros(K, d - K)], dim=1)
    else:
        W = M[:, :d]
    W = F.normalize(W * sigma, dim=-1)
    return W


def apply_sibling_repulsion(W: torch.Tensor, adj: Dict[int, list], delta: float = 0.3,
                            seed: int = 12345) -> torch.Tensor:
    """For each sibling pair (i, j), push their ETF vectors apart along a
    random tangent direction, then re-normalize. delta controls magnitude.
    Operates on a clone — input W is not mutated."""
    W = W.clone()
    K, d = W.shape
    g = torch.Generator(); g.manual_seed(seed)
    for i in range(K):
        for j in adj.get(i, []):
            if i < j:
                tangent = torch.randn(d, generator=g)
                # project tangent to be orthogonal to W[i]
                tangent = tangent - (tangent @ W[i]) * W[i]
                if tangent.norm() < 1e-8:
                    continue
                tangent = F.normalize(tangent, dim=0) * delta
                W[i] = F.normalize(W[i] + tangent, dim=0)
                W[j] = F.normalize(W[j] - tangent, dim=0)
    return W


# =============================================================================
# Light surface augmentation for 2-view contrastive
# =============================================================================

def light_augment(code: str, rng: random.Random) -> str:
    if not code: return code
    out = []
    for c in code:
        out.append(c)
        if c in "+-*/%=<>,;()[]{}" and rng.random() < 0.18:
            out.append(" ")
    return "".join(out)


# =============================================================================
# Model — PRISM
# =============================================================================

class PrismModel(nn.Module):
    def __init__(self, enc_name, n_cls, gene_adj, emb_dim=256, K_dct=16, delta=0.3,
                 learnable_W=False, etf_seed=12345):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.K_dct = K_dct
        self.hidden = h
        self.proj = nn.Sequential(
            nn.Linear(K_dct * h, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, emb_dim),
        )
        self.norm = nn.LayerNorm(emb_dim)
        # Build FIXED ETF classifier with sibling repulsion bonus
        W = build_etf(n_cls, emb_dim, sigma=1.0)
        W = apply_sibling_repulsion(W, gene_adj, delta=delta, seed=etf_seed)
        if learnable_W:
            self.W_prism = nn.Parameter(W.clone())
        else:
            self.register_buffer("W_prism", W)
        self.learnable_W = learnable_W
        self.emb_dim = emb_dim
        self.n_cls = n_cls
        # Cached DCT basis (rebuilt on device first use)
        self._dct_basis = None
        self._dct_basis_L = -1

    def _get_dct_basis(self, L: int, device, dtype):
        if self._dct_basis is not None and self._dct_basis_L == L and self._dct_basis.device == device:
            return self._dct_basis
        n = torch.arange(L, device=device, dtype=torch.float32)
        k = torch.arange(self.K_dct, device=device, dtype=torch.float32)
        basis = torch.cos(math.pi * (n.unsqueeze(0) + 0.5) * k.unsqueeze(1) / L)  # (K_dct, L)
        self._dct_basis = basis
        self._dct_basis_L = L
        return basis

    def _dct_truncate(self, H: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """H: (B, L, h); mask: (B, L). Returns (B, K_dct * h)."""
        B, L, h = H.shape
        basis = self._get_dct_basis(L, H.device, H.dtype)  # (K_dct, L)
        H_masked = H * mask.unsqueeze(-1).to(H.dtype)
        # (K_dct, L) @ (B, L, h) -> (B, K_dct, h)
        out = torch.einsum('kl,blh->bkh', basis.to(H.dtype), H_masked) * math.sqrt(2.0 / L)
        return out.reshape(B, self.K_dct * h)

    def get_W(self) -> torch.Tensor:
        if self.learnable_W:
            return F.normalize(self.W_prism, dim=-1)
        return self.W_prism

    def encode(self, ids, mask, temperature: float = 1.0):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        H = out.last_hidden_state  # (B, L, h)
        z_dct = self._dct_truncate(H, mask)
        z = self.norm(self.proj(z_dct))
        z = F.normalize(z, dim=-1)
        W = self.get_W()  # (n_cls, emb_dim), unit-normalized
        # logits = cos(z, W) / temperature
        logits = (z @ W.t()) / temperature
        return z, logits


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
    lambda_tw: float = 0.5
    gamma: float = 1.0; tau: float = 0.1
    emb_dim: int = 256
    K_dct: int = 16
    delta: float = 0.3
    cos_temperature: float = 0.07
    learnable_W: bool = False
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


class FSDS_PRISM(TD):
    """Two-view dataset for PRISM: (orig_tokens, light-jitter_tokens, label)."""
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
            logger.info(f"[FSDS_PRISM] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        lang = r.get("language", "") or ""
        label = int(r["label"])
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
                "label": label, "language": lang, "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=False):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    cos_diag_correct = []  # cos(z, W_y) when prediction == y
    cos_offdiag_wrong = []  # cos(z, W_y_pred) when prediction != y, computed as cos(z, W_y') where y' is the model's guess
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        z0, logits = model.encode(ids0, mask0, temperature=cfg.cos_temperature)
        pred = logits.argmax(dim=-1)
        preds.extend(pred.cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            # cos(z, W_y) raw — without temperature scaling
            W = model.get_W()
            cos_full = z0 @ W.t()  # (B, K)
            labs_t = labs.to(cfg.device) if torch.is_tensor(labs) else torch.tensor(labs, device=cfg.device)
            B = z0.size(0)
            arange = torch.arange(B, device=cfg.device)
            cos_y = cos_full[arange, labs_t]  # cos(z, W_true)
            # For each sample take the max cos with any wrong class
            wrong_mask = (torch.arange(model.n_cls, device=cfg.device).unsqueeze(0) != labs_t.unsqueeze(1))
            cos_wrong_max = cos_full.masked_fill(~wrong_mask, -1e9).max(dim=-1).values
            cos_diag_correct.extend(cos_y.detach().cpu().float().tolist())
            cos_offdiag_wrong.extend(cos_wrong_max.detach().cpu().float().tolist())
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
    if collect_falsifier and cos_diag_correct:
        diag_arr = np.array(cos_diag_correct)
        off_arr = np.array(cos_offdiag_wrong)
        out["falsifier"] = {
            "etf_cosine_diag_mean": float(diag_arr.mean()),
            "etf_cosine_diag_std": float(diag_arr.std()),
            "etf_cosine_offdiag_mean": float(off_arr.mean()),
            "etf_cosine_offdiag_std": float(off_arr.std()),
            "etf_cosine_margin": float(diag_arr.mean() - off_arr.mean()),
            "K_dct": cfg.K_dct,
            "delta": cfg.delta,
            "learnable_W": cfg.learnable_W,
        }
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); mask1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, logits = model.encode(ids0, mask0, temperature=cfg.cos_temperature)
            z1, _      = model.encode(ids1, mask1, temperature=cfg.cos_temperature)
            z_d = torch.cat([z0, z1], dim=0)
            y_d = torch.cat([labs, labs], dim=0)
            loss_ce = F.cross_entropy(logits, labs)
            loss_sc = supcon_tw_loss(z_d, y_d, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            loss = loss_ce + cfg.lambda_tw * loss_sc
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
        tr_data = _conv_codet(tr_raw, "author", vocab); vl_data = _conv_codet(vl_raw, "author", vocab); ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    tr_ds = FSDS_PRISM(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_PRISM(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS_PRISM(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"lambda_tw={cfg.lambda_tw} K_dct={cfg.K_dct} delta={cfg.delta} "
                f"learnable_W={cfg.learnable_W}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = PrismModel(cfg.enc, cfg.n_cls, cfg.gene_adj, emb_dim=cfg.emb_dim,
                       K_dct=cfg.K_dct, delta=cfg.delta,
                       learnable_W=cfg.learnable_W, etf_seed=cfg.seed + 1000).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, ce, sc = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} supcon={sc:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=True)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    fa = ts_met["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"cos_diag={fa['etf_cosine_diag_mean']:+.3f} "
                f"cos_off={fa['etf_cosine_offdiag_mean']:+.3f} "
                f"margin={fa['etf_cosine_margin']:+.3f}")
    return {"tag": tag, "method": "PRISM",
            "upstream": "new (DCT + ETF + sibling-repulse + TRACO)",
            "note": f"K_dct={cfg.K_dct} delta={cfg.delta} learnable_W={cfg.learnable_W}",
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_tw": cfg.lambda_tw, "gamma": cfg.gamma, "tau": cfg.tau,
            "K_dct": cfg.K_dct, "delta": cfg.delta, "learnable_W": cfg.learnable_W,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "etf_cosine_diag_mean": fa["etf_cosine_diag_mean"],
            "etf_cosine_offdiag_mean": fa["etf_cosine_offdiag_mean"],
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
                tag = f"exp118_prism_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"cos_diag={res['etf_cosine_diag_mean']:+.3f} "
                                f"cos_off={res['etf_cosine_offdiag_mean']:+.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp118_prism_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'cos_diag':>10} {'cos_off':>10} {'margin':>9} {'Wall':>8}")
    print("-"*150)
    for r in results:
        diag = r['etf_cosine_diag_mean']; off = r['etf_cosine_offdiag_mean']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{diag:>+10.3f} {off:>+10.3f} {(diag-off):>+9.3f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
