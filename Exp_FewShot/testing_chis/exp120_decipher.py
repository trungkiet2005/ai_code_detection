# exp120 — DECIPHER
# NAME       : DECIPHER (Dual-Encoder Contrastive Inference, Phrased-as-fixed-point Estimator-Refined)
# REFERENCE  : new; combines adversarial co-training (SCHISM idea) +
#              fixed-point attention retrieval (GRAVITY idea) +
#              TRACO + gradient reversal (CHORUS).
# CLAIM      : Sibling discrimination requires a WEAKER signal than
#              cross-family discrimination. Train TWO encoders: A
#              pushes siblings APART (sibling-splitter), B pulls them
#              TOGETHER (anti-discriminator). The classifier reads
#              z_A − z_B. At inference, iteratively retrieve neighbours
#              in the (z_A−z_B) space until convergence.
# EQUATION   : z_A = phi_A(x), z_B = phi_B(x)
#              L_A = CE + λ·TRACO_supcon(z_A) with sibling neg weight 10×
#              L_B = 1 - mean cos(z_B^i, z_B^j) for sibling pairs (i,j) in batch
#              Classifier reads (z_A - z_B)
#              At test: z_0 = z_A(q) - z_B(q); iterate T steps:
#                       N_t = top-K(z_{t-1}, train_diff_emb)
#                       z_t = LayerNorm(z_{t-1} + attn(z_{t-1}, N_t))
#              y_hat = argmax of softmax classifier(z_T)
# WHY NEW    : First code-attribution method to use TWO encoders
#              adversarially + fixed-point retrieval inference. Neither
#              single piece is novel but the synthesis (subtract the
#              sibling-merger from the sibling-splitter, then refine via
#              retrieval) has not been done.
# WOW HOOK   : "We train a sibling-merger that we then SUBTRACT.
#              What survives is identity. At inference, the query
#              drifts through its own neighbourhood until it lands on
#              its author."
# FALSIFIER  : (F1) ||z_A - z_B|| on sibling pairs > ||z_A - z_B|| on
#              cross-family by 2x (interpretable residual).
#              (F2) Removing the adversarial step (only A) loses >= 0.01.
#              (F3) Removing T>1 retrieval loses >= 0.005.
#              (F4) Composite > METATRACO + 0.005.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

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
logger = logging.getLogger("exp120_decipher")


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
# Losses
# =============================================================================

def supcon_tw_loss(z_doubled, labels_doubled, dist_mat, gamma=1.0, tau=0.1):
    """Tree-weighted SupCon."""
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


def supcon_tw_loss_sibling_emphasis(z_doubled, labels_doubled, dist_mat, sib_mask,
                                    gamma=1.0, tau=0.1, sib_neg_scale=10.0):
    """SupCon variant where sibling negatives are emphasised (negative weight 10x).
    Implemented by scaling the contribution of sibling-pair distances down so the
    exponential weight on them becomes much larger (harder negatives)."""
    N = z_doubled.size(0)
    if N < 2: return z_doubled.sum() * 0.0
    sim = (z_doubled @ z_doubled.t()) / tau
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()
    eye = torch.eye(N, device=z_doubled.device, dtype=torch.bool)
    pos_mask = (labels_doubled.unsqueeze(0) == labels_doubled.unsqueeze(1)).float().masked_fill(eye, 0.0)
    neg_mask = (labels_doubled.unsqueeze(0) != labels_doubled.unsqueeze(1)).float()
    dij = dist_mat[labels_doubled][:, labels_doubled]
    w_neg = torch.exp(-gamma * dij)
    # Emphasise sibling negatives: scale their weight by sib_neg_scale
    sib_pair = sib_mask[labels_doubled][:, labels_doubled]
    w_neg = w_neg * (1.0 + (sib_neg_scale - 1.0) * sib_pair)
    w_pair = pos_mask + neg_mask * w_neg
    w_pair = w_pair.masked_fill(eye, 0.0)
    exp_s = (torch.exp(sim) * w_pair).clamp(min=1e-12)
    den = exp_s.sum(dim=-1).clamp(min=1e-12)
    num = (exp_s * pos_mask).sum(dim=-1).clamp(min=1e-12)
    n_pos = pos_mask.sum(dim=-1)
    has_pos = (n_pos > 0).float()
    li = -(torch.log(num) - torch.log(den)) * has_pos
    return li.sum() / has_pos.sum().clamp(min=1.0)


def sibling_merger_loss(z, labels, sib_mask):
    """Encoder-B objective: PULL siblings TOGETHER.
    L_B = 1 - mean cos(z[i], z[j]) for (i,j) such that sib_mask[y_i, y_j] = 1.
    If no sibling pairs in batch, returns 0."""
    N = z.size(0)
    if N < 2: return z.sum() * 0.0
    sim = z @ z.t()  # cosine sim since z is L2-normalised
    sib_pair = sib_mask[labels][:, labels]
    eye = torch.eye(N, device=z.device, dtype=torch.bool)
    sib_pair = sib_pair.masked_fill(eye, 0.0)
    n_pair = sib_pair.sum().clamp(min=1.0)
    mean_cos = (sim * sib_pair).sum() / n_pair
    return 1.0 - mean_cos


# =============================================================================
# Model
# =============================================================================

class DECIPHER(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder_A = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        self.encoder_B = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder_A.config.hidden_size
        self.proj_A = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(512, emb_dim))
        self.proj_B = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1),
                                    nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        # Fixed-point retrieval refinement
        self.attn_refine = nn.MultiheadAttention(emb_dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(emb_dim)
        self.emb_dim = emb_dim
        self.n_cls = n_cls

    def encode_A(self, ids, mask):
        out = self.encoder_A(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return F.normalize(self.proj_A(sem), dim=-1)

    def encode_B(self, ids, mask):
        out = self.encoder_B(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return F.normalize(self.proj_B(sem), dim=-1)

    def encode(self, ids, mask):
        z_A = self.encode_A(ids, mask)
        z_B = self.encode_B(ids, mask)
        z = z_A - z_B  # residual
        return z, z_A, z_B

    def classify(self, z):
        return self.clf(z)

    def refine(self, z_query, train_diff_bank, T=3, K=16):
        """At test: iteratively attend over training residuals."""
        if train_diff_bank is None or train_diff_bank.size(0) < 2:
            return z_query
        z = z_query
        Kuse = min(K, train_diff_bank.size(0))
        for _ in range(T):
            sim = z @ train_diff_bank.t()
            topk_idx = sim.topk(Kuse, dim=-1).indices  # (B, K)
            neighbors = train_diff_bank[topk_idx]      # (B, K, emb_dim)
            attended, _ = self.attn_refine(z.unsqueeze(1), neighbors, neighbors)
            z = self.norm(z + attended.squeeze(1))
        return z


# =============================================================================
# Plumbing
# =============================================================================

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_A_supcon: float = 0.5
    lambda_B_merge: float = 0.5
    sib_neg_scale: float = 10.0
    gamma: float = 1.0; tau: float = 0.1
    refine_T: int = 3
    refine_K: int = 16
    emb_dim: int = 256
    device: str = "cuda"
    gene_adj: dict = field(default_factory=dict)


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
        # TWO encoders -> halve bs again on top of 2-view halving from exp84.
        if mem >= 80: cfg.bs, cfg.seq = 96, 512
        elif mem >= 40: cfg.bs, cfg.seq = 64, 512
        elif mem >= 20: cfg.bs, cfg.seq = 32, 448
        elif mem >= 10: cfg.bs, cfg.seq = 24, 384
        else: cfg.bs, cfg.seq = 12, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} seq={cfg.seq} (TWO encoders)")
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
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42):
        self.data = data; self.tok = tok; self.seq_len = seq_len
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
        enc = self.tok(code, max_length=self.seq_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        ids, mask = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        return {"ids": ids, "mask": mask, "label": r["label"],
                "language": lang, "source": r.get("source", "") or ""}


# =============================================================================
# Eval (with optional refine + falsifier residual stats)
# =============================================================================

@torch.no_grad()
def build_diff_bank(model, loader, cfg, max_n=4096):
    """Encode training samples and store (z_A - z_B). Used for fixed-point retrieval."""
    model.eval()
    diffs, labs_all = [], []
    n_seen = 0
    for b in tqdm(loader, desc="Bank"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        z, _, _ = model.encode(ids, mask)
        diffs.append(z.detach())
        labs_all.append(b["label"])
        n_seen += z.size(0)
        if n_seen >= max_n: break
    bank = torch.cat(diffs, dim=0)[:max_n]
    bank_labels = torch.cat([l if torch.is_tensor(l) else torch.tensor(l) for l in labs_all], dim=0)[:max_n]
    logger.info(f"[bank] residual bank built: N={bank.size(0)} d={bank.size(1)}")
    return bank, bank_labels


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, diff_bank=None,
              refine_T=3, collect_falsifier=False, sib_mask_t=None):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    sib_norms, cross_norms = [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        z, z_A, z_B = model.encode(ids, mask)
        if diff_bank is not None and refine_T > 0:
            z_refined = model.refine(z, diff_bank, T=refine_T, K=cfg.refine_K)
        else:
            z_refined = z
        logits = model.classify(z_refined)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labs_l = labs.tolist() if torch.is_tensor(labs) else list(labs)
        labels.extend(labs_l)
        if collect_falsifier and sib_mask_t is not None:
            # Pairwise residual norms within batch
            B = z.size(0)
            if B >= 2:
                pair_norm = (z.unsqueeze(0) - z.unsqueeze(1)).norm(dim=-1)  # (B, B)
                labs_t = torch.tensor(labs_l, device=z.device, dtype=torch.long)
                sib_pair = sib_mask_t[labs_t][:, labs_t]
                eye = torch.eye(B, device=z.device, dtype=torch.bool)
                diff_lbl = (labs_t.unsqueeze(0) != labs_t.unsqueeze(1))
                cross_pair = diff_lbl & (~sib_pair.bool()) & (~eye)
                sib_mat = sib_pair.bool() & (~eye)
                if sib_mat.any(): sib_norms.extend(pair_norm[sib_mat].cpu().tolist())
                if cross_pair.any(): cross_norms.extend(pair_norm[cross_pair].cpu().tolist())
        lang_batch = b.get("language", [""] * len(labs_l))
        src_batch = b.get("source", [""] * len(labs_l))
        langs.extend(list(lang_batch) if not isinstance(lang_batch, list) else lang_batch)
        sources.extend(list(src_batch) if not isinstance(src_batch, list) else src_batch)
    preds = np.array(preds); labels = np.array(labels); n_cls = cfg.n_cls
    overall = {"accuracy": float(accuracy_score(labels, preds)),
               "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
               "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
               "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
               "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0))}
    per_class = {"f1": f1_score(labels, preds, average=None, zero_division=0,
                                 labels=list(range(n_cls))).tolist(),
                 "precision": precision_score(labels, preds, average=None, zero_division=0,
                                              labels=list(range(n_cls))).tolist(),
                 "recall": recall_score(labels, preds, average=None, zero_division=0,
                                        labels=list(range(n_cls))).tolist()}
    cm = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    off_diag = int(cm.sum() - cm.trace())
    sib_conf = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls)
                       if i != j and sib_mask_np[i, j] > 0))
    sib_rate = sib_conf / max(off_diag, 1)
    cross = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls)
                    if i != j and dist_mat_cpu[i, j] >= 3.0))
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
    out = {"overall": overall, "per_class": per_class, "per_language": per_lang,
           "per_source": per_src, "confusion_matrix": cm.tolist(),
           "sibling_confusion_rate": float(sib_rate),
           "cross_family_confusion_rate": float(cross_rate),
           "off_diag_total": off_diag, "n_samples": int(len(labels))}
    if collect_falsifier:
        sib_mean = float(np.mean(sib_norms)) if sib_norms else 0.0
        cross_mean = float(np.mean(cross_norms)) if cross_norms else 0.0
        out["falsifier"] = {
            "sibling_residual_norm_mean": sib_mean,
            "cross_family_residual_norm_mean": cross_mean,
            "sibling_over_cross_ratio_F1": (sib_mean / cross_mean) if cross_mean > 0 else 0.0,
            "n_sibling_pairs": len(sib_norms),
            "n_cross_pairs": len(cross_norms),
        }
    return out


# =============================================================================
# Train (alternating: encoder-A step, encoder-B step)
# =============================================================================

def train_epoch(model, loader, opt_A, opt_B, sch_A, sch_B, scaler_A, scaler_B,
                cfg, dist_mat, sib_mask_t):
    model.train()
    tot_A, tot_B, ce_s, sc_s, mg_s = 0.0, 0.0, 0.0, 0.0, 0.0
    n_A, n_B = 0, 0
    for i, b in enumerate(tqdm(loader, desc="Train")):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        if i % 2 == 0:
            # ---- Step A: train encoder_A + proj_A + clf with sibling-emphasised supcon ----
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(cfg.device == "cuda")):
                z_A = model.encode_A(ids, mask)
                with torch.no_grad():
                    z_B = model.encode_B(ids, mask)
                z = z_A - z_B
                logits = model.classify(z)
                loss_ce = F.cross_entropy(logits, labs)
                z_d = torch.cat([z_A, z_A], dim=0)
                y_d = torch.cat([labs, labs], dim=0)
                loss_sc = supcon_tw_loss_sibling_emphasis(
                    z_d, y_d, dist_mat, sib_mask_t,
                    gamma=cfg.gamma, tau=cfg.tau, sib_neg_scale=cfg.sib_neg_scale)
                loss_A = loss_ce + cfg.lambda_A_supcon * loss_sc
            scaler_A.scale(loss_A).backward()
            scaler_A.unscale_(opt_A)
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder_A.parameters()) + list(model.proj_A.parameters())
                + list(model.clf.parameters()) + list(model.attn_refine.parameters())
                + list(model.norm.parameters()), 1.0)
            scaler_A.step(opt_A); scaler_A.update(); opt_A.zero_grad(); sch_A.step()
            tot_A += loss_A.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item(); n_A += 1
        else:
            # ---- Step B: train encoder_B + proj_B with anti-supcon (sibling merger) ----
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                enabled=(cfg.device == "cuda")):
                z_B = model.encode_B(ids, mask)
                loss_mg = sibling_merger_loss(z_B, labs, sib_mask_t)
                loss_B = cfg.lambda_B_merge * loss_mg
            scaler_B.scale(loss_B).backward()
            scaler_B.unscale_(opt_B)
            torch.nn.utils.clip_grad_norm_(
                list(model.encoder_B.parameters()) + list(model.proj_B.parameters()), 1.0)
            scaler_B.step(opt_B); scaler_B.update(); opt_B.zero_grad(); sch_B.step()
            tot_B += loss_B.item(); mg_s += loss_mg.item(); n_B += 1
    return {"loss_A": tot_A / max(n_A, 1),
            "loss_B": tot_B / max(n_B, 1),
            "ce": ce_s / max(n_A, 1),
            "supcon": sc_s / max(n_A, 1),
            "merge": mg_s / max(n_B, 1)}


def run_exp(cfg, tag):
    set_seed(cfg.seed); cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat_t = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist_mat = dist_mat_t.to(cfg.device); dist_mat_cpu = dist_mat_t.numpy()
    sib_mask_t_cpu = build_sibling_mask(cfg.n_cls, cfg.gene_adj)
    sib_mask_np = sib_mask_t_cpu.numpy()
    sib_mask_t = sib_mask_t_cpu.to(cfg.device)
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
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} "
                f"warmup={cfg.warmup} lambdaA={cfg.lambda_A_supcon} "
                f"lambdaB={cfg.lambda_B_merge} sib_neg_scale={cfg.sib_neg_scale}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = DECIPHER(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    # ---- Two optimisers: A trains encoder_A + proj_A + clf + attn_refine; B trains encoder_B + proj_B ----
    paramsA_enc = list(model.encoder_A.parameters())
    paramsA_head = (list(model.proj_A.parameters())
                    + list(model.clf.parameters())
                    + list(model.attn_refine.parameters())
                    + list(model.norm.parameters()))
    paramsB_enc = list(model.encoder_B.parameters())
    paramsB_head = list(model.proj_B.parameters())
    opt_A = torch.optim.AdamW(
        [{"params": paramsA_enc, "lr": cfg.lr_enc},
         {"params": paramsA_head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    opt_B = torch.optim.AdamW(
        [{"params": paramsB_enc, "lr": cfg.lr_enc},
         {"params": paramsB_head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch_A = get_cosine_schedule_with_warmup(opt_A, max(1, int(total_steps * cfg.warmup // 2)),
                                            max(1, total_steps // 2))
    sch_B = get_cosine_schedule_with_warmup(opt_B, max(1, int(total_steps * cfg.warmup // 2)),
                                            max(1, total_steps // 2))
    scaler_A = GradScaler(); scaler_B = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        stats = train_epoch(model, tr_dl, opt_A, opt_B, sch_A, sch_B,
                            scaler_A, scaler_B, cfg, dist_mat, sib_mask_t)
        # Validation: skip refine for speed during training; just classifier on z = z_A - z_B
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu,
                            diff_bank=None, refine_T=0)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] lossA={stats['loss_A']:.4f} lossB={stats['loss_B']:.4f} "
                    f"ce={stats['ce']:.4f} supcon={stats['supcon']:.4f} "
                    f"merge={stats['merge']:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    # ---- Build residual bank from training data ----
    bank_dl = DataLoader(FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed),
                         shuffle=False, **loader_cfg)
    diff_bank, _ = build_diff_bank(model, bank_dl, cfg, max_n=4096)
    # ---- F3 ablation: T=1 vs T=cfg.refine_T accuracy on test ----
    ts_met_T0 = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu,
                          diff_bank=None, refine_T=0)
    ts_met_T1 = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu,
                          diff_bank=diff_bank, refine_T=1)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu,
                       diff_bank=diff_bank, refine_T=cfg.refine_T,
                       collect_falsifier=True, sib_mask_t=sib_mask_t)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    fa = ts_met["falsifier"]
    f3_T0 = ts_met_T0["overall"]["macro_f1"]
    f3_T1 = ts_met_T1["overall"]["macro_f1"]
    logger.info(f"[final] val={best_val:.4f} test(T={cfg.refine_T})={test_macro:.4f} "
                f"gap={gap:+.4f} T0={f3_T0:.4f} T1={f3_T1:.4f} "
                f"sib_norm={fa['sibling_residual_norm_mean']:.3f} "
                f"cross_norm={fa['cross_family_residual_norm_mean']:.3f} "
                f"ratio_F1={fa['sibling_over_cross_ratio_F1']:.3f}")
    return {"tag": tag, "method": "DECIPHER", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_A_supcon": cfg.lambda_A_supcon,
            "lambda_B_merge": cfg.lambda_B_merge,
            "sib_neg_scale": cfg.sib_neg_scale, "refine_T": cfg.refine_T,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"],
            "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "f3_macro_T0": f3_T0,
            "f3_macro_T1": f3_T1,
            "f3_macro_Tfull": test_macro,
            "f3_delta_T1_to_Tfull": test_macro - f3_T1,
            "sibling_residual_norm_mean": fa["sibling_residual_norm_mean"],
            "cross_family_residual_norm_mean": fa["cross_family_residual_norm_mean"],
            "sibling_over_cross_ratio_F1": fa["sibling_over_cross_ratio_F1"],
            "note": ("DECIPHER = encoder_A (sibling-splitter) - encoder_B (sibling-merger) "
                     "+ fixed-point retrieval refinement"),
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
                tag = f"exp120_decipher_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"ratio={res['sibling_over_cross_ratio_F1']:.3f} "
                                f"dT={res['f3_delta_T1_to_Tfull']:+.4f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp120_decipher_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*160)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'T0':>8} {'T1':>8} {'Ratio':>8} {'Wall':>8}")
    print("-"*160)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['f3_macro_T0']:>8.4f} {r['f3_macro_T1']:>8.4f} "
              f"{r['sibling_over_cross_ratio_F1']:>8.3f} {r['wall']:>8.0f}s")
    print("="*160)


if __name__ == "__main__":
    main()
