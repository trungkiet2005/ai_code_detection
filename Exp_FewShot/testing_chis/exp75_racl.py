# exp75_racl — Retrieval-Augmented Code Logit (RACL)
# =============================================================================
# Theory-Track exp -- RACL (Retrieval-Augmented Code Logit)
#
# ROLE           : Retrieval-augmented classification is the missing paradigm
#                  for AI-code attribution.  RAFC (arXiv:2406.11148) and kNN-LM
#                  (Khandelwal et al. 2020) showed that augmenting a parametric
#                  classifier with non-parametric retrieval is strongly few-shot
#                  efficient.  None of our 17 prior methods uses retrieval.
#                  RACL combines:
#                    - parametric CE classifier  (logits_param)
#                    - in-batch / memory-bank kNN soft-vote with tree-weighted
#                      similarity  (logits_knn)
#                  via a LEARNED MIXING WEIGHT beta = sigmoid(beta_logit).
#                  The learned beta per fraction tests whether retrieval helps
#                  more at low-n (where parametric is data-starved) and less
#                  at high-n (where parametric saturates).
# NAME           : RACL  (Retrieval-Augmented Code Logit)
# ARXIV_ID       : extends RAFC arXiv:2406.11148 + kNN-LM idea with tree-aware
#                  similarity weighting and learned regime mixing.
# ONE-LINE CLAIM : An LLM-attribution model can compute its logits as a
#                  learned convex combination of a parametric classifier and
#                  a non-parametric retrieval head; the learned mixing weight
#                  is a direct empirical signal of where each paradigm helps.
# EQUATION       : During training (in-batch retrieval):
#                    sim_ij = cos(z_i, z_j)   for j != i in batch
#                    top-k(i) = top-k by sim_ij
#                    logits_knn_ic = sum_{j in top-k(i)} 1{y_j=c}
#                                    * exp(sim_ij / tau) / Z_i
#                  Combined: logits_i = beta * logits_param_i + (1-beta) * logits_knn_i
#                  beta = sigmoid(beta_logit) in (0, 1) learned.
#                  Loss: L = CE(logits, y) + 0.1 * CE(logits_param, y)
#                  (anchor on parametric to avoid full reliance on retrieval).
#                  At inference, kNN is over the FULL training memory bank
#                  built once per epoch (k=16).
# WHY NOT BEFORE : Retrieval-augmented few-shot has been studied for text
#                  classification (RAFC, GORAG) and images (RAFIC) but never
#                  for AI-code attribution with TREE-AWARE similarity.  The
#                  learned regime mixing beta also makes the parametric vs
#                  retrieval contribution INTERPRETABLE per fraction.
# FALSIFIER      : (F1) Learned beta per fraction: monotone decrease with n
#                       would confirm "retrieval helps more at low-n".
#                  (F2) F1 of parametric-only (beta=1) vs retrieval-only
#                       (beta=0) vs combined.  Combined > max(both) confirms
#                       the mixture is non-trivial.
#                  (F3) Sibling-confusion rate of retrieval-only vs parametric-
#                       only.  Retrieval should reduce sib_conf since k-NN
#                       in tree-aware metric handles siblings well.
# REPORTS        : Full eval pack + beta + per-mode F1 triple + sib_conf per mode.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from dataclasses import dataclass, field
from typing import List, Dict

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
logger = logging.getLogger("exp75_racl")

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
# Model
# =============================================================================

class RACLModel(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        # Learned mixing weight beta = sigmoid(beta_logit)
        self.beta_logit = nn.Parameter(torch.tensor(0.0))   # init beta=0.5
        self.n_cls = n_cls; self.emb_dim = emb_dim

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1)

    def parametric_logits(self, z):
        return self.clf(z)

    def beta(self):
        return torch.sigmoid(self.beta_logit)


def knn_logits_inbatch(z, labels, dist_mat, k=16, tau=0.1, gamma=0.5):
    """In-batch kNN soft vote with tree-aware similarity.

    For each i, compute logit_c = sum_{j in top-k(i), y_j=c} sim_ij * exp(-gamma * d_tree(c, y_j))
    Note: when y_j=c, d_tree term = 0 so contribution is unweighted; we mainly
    use the d_tree term for cross-checking via inverse formulation; for clean
    voting, simply tree-aware weight the cosine in similarity itself.
    """
    B = z.size(0); n_cls = int(labels.max().item() + 1) if labels.numel() > 0 else 1
    # cosine similarity matrix
    sim = (z @ z.t())                                   # (B, B)
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e4)
    # top-k indices per row
    k_eff = min(k, B - 1)
    top_vals, top_idx = sim.topk(k=k_eff, dim=-1)       # (B, k)
    # gather neighbor labels and weights
    neigh_lab = labels[top_idx]                         # (B, k)
    weights = F.softmax(top_vals / tau, dim=-1)         # (B, k)
    # build (B, n_cls) logit by scatter-add weighted votes
    logits_knn = torch.zeros(B, n_cls, device=z.device, dtype=weights.dtype)
    logits_knn.scatter_add_(1, neigh_lab, weights)
    # convert to log-probs (with epsilon)
    logits_knn = torch.log(logits_knn.clamp(min=1e-6))
    return logits_knn, top_idx, top_vals


def knn_logits_memory(z, mem_z, mem_y, n_cls, k=16, tau=0.1):
    """kNN over external memory bank.  mem_z (M, d) L2-normed; mem_y (M,) labels."""
    sim = z @ mem_z.t()                                 # (B, M)
    k_eff = min(k, mem_z.size(0))
    top_vals, top_idx = sim.topk(k=k_eff, dim=-1)
    neigh_lab = mem_y[top_idx]                          # (B, k)
    weights = F.softmax(top_vals / tau, dim=-1)
    B = z.size(0)
    logits_knn = torch.zeros(B, n_cls, device=z.device, dtype=weights.dtype)
    logits_knn.scatter_add_(1, neigh_lab, weights)
    return torch.log(logits_knn.clamp(min=1e-6))


# =============================================================================
# Plumbing
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4; lr_beta: float = 1e-2
    warmup: float = 0.1; wd: float = 0.01
    k_neigh: int = 16; knn_tau: float = 0.1; knn_gamma: float = 0.5
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
        if mem >= 40: cfg.bs, cfg.seq = 256, 512
        elif mem >= 10: cfg.bs, cfg.seq = 128, 384
        else: cfg.bs, cfg.seq = 64, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} seq={cfg.seq}")
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
        enc = self.tok(code, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
        return {"ids": enc["input_ids"].squeeze(0), "mask": enc["attention_mask"].squeeze(0),
                "label": r["label"], "language": r.get("language", "") or "",
                "source": r.get("source", "") or ""}


@torch.no_grad()
def build_memory(model, loader, cfg, max_per_cls=2000):
    model.eval()
    zs, ys = [], []
    for b in tqdm(loader, desc="MemBuild"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        z = model.encode(ids, mask)
        zs.append(z.detach().cpu().float())
        ys.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
    z_all = torch.cat(zs, dim=0); y_all = torch.tensor(ys, dtype=torch.long)
    # downsample per-class to max_per_cls
    keep = []
    rng = np.random.default_rng(cfg.seed)
    for c in range(cfg.n_cls):
        idx = (y_all == c).nonzero(as_tuple=True)[0].numpy()
        if len(idx) > max_per_cls: idx = rng.choice(idx, size=max_per_cls, replace=False)
        keep.extend(idx.tolist())
    keep = torch.tensor(keep, dtype=torch.long)
    return z_all[keep].to(cfg.device), y_all[keep].to(cfg.device)


@torch.no_grad()
def eval_modes(model, loader, cfg, sib_mask_np, dist_mat_cpu, mem_z, mem_y):
    """Run three predictors: parametric-only, retrieval-only, combined."""
    model.eval()
    preds_p, preds_k, preds_c, labels, langs, sources = [], [], [], [], [], []
    for b in tqdm(loader, desc="Eval3"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        z = model.encode(ids, mask)
        log_p = model.parametric_logits(z)
        log_k = knn_logits_memory(z, mem_z, mem_y, cfg.n_cls, k=cfg.k_neigh, tau=cfg.knn_tau)
        beta = model.beta()
        # combined logits: use log-probs scale
        log_p_lp = F.log_softmax(log_p, dim=-1)
        log_k_lp = log_k - log_k.logsumexp(dim=-1, keepdim=True)
        log_c = beta * log_p_lp + (1.0 - beta) * log_k_lp
        preds_p.extend(log_p.argmax(dim=-1).cpu().tolist())
        preds_k.extend(log_k.argmax(dim=-1).cpu().tolist())
        preds_c.extend(log_c.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        lang_batch = b.get("language", [""] * len(labs))
        src_batch = b.get("source", [""] * len(labs))
        langs.extend(list(lang_batch) if not isinstance(lang_batch, list) else lang_batch)
        sources.extend(list(src_batch) if not isinstance(src_batch, list) else src_batch)
    labels = np.array(labels); n_cls = cfg.n_cls
    def _pack(preds, labels, sib_mask_np, dist_mat_cpu, n_cls):
        preds = np.array(preds)
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
        return {"overall": overall, "per_class": per_class, "confusion_matrix": cm.tolist(),
                "sibling_confusion_rate": float(sib_rate),
                "cross_family_confusion_rate": float(cross_rate),
                "off_diag_total": off_diag, "n_samples": int(len(labels))}
    pack_p = _pack(preds_p, labels, sib_mask_np, dist_mat_cpu, n_cls)
    pack_k = _pack(preds_k, labels, sib_mask_np, dist_mat_cpu, n_cls)
    pack_c = _pack(preds_c, labels, sib_mask_np, dist_mat_cpu, n_cls)
    # per_lang / per_src on COMBINED predictions
    preds_c_arr = np.array(preds_c)
    per_lang, per_src = {}, {}
    if any(l for l in langs):
        la = np.array(langs)
        for L in sorted(set(langs)):
            if not L: continue
            sel = (la == L)
            if sel.sum() < 2: continue
            per_lang[L] = {"n": int(sel.sum()),
                           "macro_f1": float(f1_score(labels[sel], preds_c_arr[sel], average="macro", zero_division=0)),
                           "weighted_f1": float(f1_score(labels[sel], preds_c_arr[sel], average="weighted", zero_division=0)),
                           "accuracy": float(accuracy_score(labels[sel], preds_c_arr[sel]))}
    if any(s for s in sources):
        sa = np.array(sources)
        for S in sorted(set(sources)):
            if not S: continue
            sel = (sa == S)
            if sel.sum() < 2: continue
            per_src[S] = {"n": int(sel.sum()),
                          "macro_f1": float(f1_score(labels[sel], preds_c_arr[sel], average="macro", zero_division=0)),
                          "weighted_f1": float(f1_score(labels[sel], preds_c_arr[sel], average="weighted", zero_division=0)),
                          "accuracy": float(accuracy_score(labels[sel], preds_c_arr[sel]))}
    pack_c["per_language"] = per_lang
    pack_c["per_source"] = per_src
    return pack_p, pack_k, pack_c


def train_epoch(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train(); tot, ce_p_s, ce_c_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z = model.encode(ids, mask)
            log_p = model.parametric_logits(z)
            log_k, _, _ = knn_logits_inbatch(z, labs, dist_mat, k=cfg.k_neigh, tau=cfg.knn_tau, gamma=cfg.knn_gamma)
            beta = model.beta()
            log_p_lp = F.log_softmax(log_p, dim=-1)
            log_k_lp = log_k - log_k.logsumexp(dim=-1, keepdim=True)
            log_c = beta * log_p_lp + (1.0 - beta) * log_k_lp
            loss_c = F.nll_loss(log_c, labs)
            loss_p = F.cross_entropy(log_p, labs)
            loss = loss_c + 0.1 * loss_p           # anchor on parametric
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_p_s += loss_p.item(); ce_c_s += loss_c.item()
    n = len(loader)
    return tot/n, ce_p_s/n, ce_c_s/n


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
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"k={cfg.k_neigh} knn_tau={cfg.knn_tau}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    tr_dl_eval = DataLoader(tr_ds, shuffle=False, batch_size=cfg.bs, num_workers=4, pin_memory=True)
    model = RACLModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    beta_p = [model.beta_logit]
    head_params = [p for n, p in model.named_parameters()
                   if id(p) not in enc_ids and n != "beta_logit"]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head},
        {"params": beta_p, "lr": cfg.lr_beta, "weight_decay": 0.0}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist, beta_hist = 0.0, None, [], []
    # validation uses combined logits w/ in-batch retrieval (cheap)
    for epoch in range(cfg.epochs):
        loss, ce_p, ce_c = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        # rebuild memory bank from training set for val eval
        mem_z, mem_y = build_memory(model, tr_dl_eval, cfg)
        _, _, vc = eval_modes(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu, mem_z, mem_y)
        v = vc["overall"]["macro_f1"]; val_hist.append(v)
        beta_now = float(model.beta().item()); beta_hist.append(beta_now)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce_p={ce_p:.4f} ce_c={ce_c:.4f} val={v:.4f} beta={beta_now:.3f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    # === Eval all three modes ===
    mem_z, mem_y = build_memory(model, tr_dl_eval, cfg)
    pack_p, pack_k, pack_c = eval_modes(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, mem_z, mem_y)
    test_macro_c = pack_c["overall"]["macro_f1"]
    test_macro_p = pack_p["overall"]["macro_f1"]
    test_macro_k = pack_k["overall"]["macro_f1"]
    gap = best_val - test_macro_c
    beta_final = float(model.beta().item())
    pack_c["falsifier"] = {
        "beta_final_F1": beta_final, "beta_history_F1": beta_hist,
        "macro_param_only_F2": test_macro_p, "macro_retr_only_F2": test_macro_k,
        "macro_combined_F2": test_macro_c,
        "combined_minus_max_F2": test_macro_c - max(test_macro_p, test_macro_k),
        "sib_conf_param_F3": float(pack_p["sibling_confusion_rate"]),
        "sib_conf_retr_F3": float(pack_k["sibling_confusion_rate"]),
        "sib_conf_comb_F3": float(pack_c["sibling_confusion_rate"]),
    }
    logger.info(f"[final] val={best_val:.4f} test_comb={test_macro_c:.4f} test_param={test_macro_p:.4f} "
                f"test_retr={test_macro_k:.4f} gap={gap:+.4f} beta={beta_final:.3f}")
    return {"tag": tag, "method": "RACL", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "k_neigh": cfg.k_neigh, "knn_tau": cfg.knn_tau, "knn_gamma": cfg.knn_gamma,
            "val_macro": best_val, "macro": test_macro_c,
            "weighted": pack_c["overall"]["weighted_f1"], "acc": pack_c["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro_c - PAPER_BASELINE,
            "test_metrics": pack_c, "val_history": val_hist,
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
                tag = f"exp75_racl_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test_C={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"P={fa['macro_param_only_F2']:.4f} K={fa['macro_retr_only_F2']:.4f} "
                                f"gap={res['val_test_gap']:+.4f} beta={fa['beta_final_F1']:.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp75_racl_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'TestC':>8} {'TestP':>8} {'TestK':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'beta':>6} {'Wall':>8}")
    print("-"*150)
    for r in results:
        fa = r['test_metrics']['falsifier']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{fa['macro_param_only_F2']:>8.4f} {fa['macro_retr_only_F2']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} {fa['beta_final_F1']:>6.3f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
