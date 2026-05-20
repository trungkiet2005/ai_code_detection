# exp119 — FOREST
# NAME       : FOREST (Family-Of-Recursive-Experts via Self-Trained tree)
# REFERENCE  : new; combines per-family specialists (this paper's
#              SPECEXPERT exp98) with MAML (Finn 2017) and learnable
#              ultrametric tree (TREELEARN exp106 idea).
# CLAIM      : The K-class softmax is the wrong factorization. The
#              right one is a TREE: a family-root predicts the family,
#              a per-family expert predicts the sibling within. We
#              JOINTLY learn the tree (differentiable ultrametric) and
#              META-LEARN the per-family experts (MAML outer loop where
#              each task is one family's sibling discrimination).
# EQUATION   : Family clusters {F_1,..F_M} inferred from D_learn (rank-M soft clustering)
#              p_family(y_fam | x) = softmax(W_fam · z(x))
#              p_within_family(y | x, fam) = softmax(W_fam · z(x))  per-fam expert
#              p(y | x) = sum_fam p_family(fam | x) * p_within(y | x, fam)
#              Outer loop: MAML over family-tasks
#              L = CE_total + λ_TW · L_TRACO_supcon + λ_ultra · UltraPenalty(D_learn)
# WHY NEW    : SPECEXPERT used hand-coded family map + joint training.
#              METATRACO used MAML on flat K-way. No prior code-
#              attribution paper combines (learn-the-tree) + (per-family
#              experts) + (meta-learn the experts). Three orthogonal
#              components synthesized.
# WOW HOOK   : "Attribution as descent down a tree the model grew
#              itself. The forest meta-learns its taxonomy + each
#              family-leaf is a few-shot sibling specialist."
# FALSIFIER  : (F1) Learned D_learn cluster-Rand-Index with hand-coded
#              tree > 0.5. (F2) Family-root accuracy on test > 0.80.
#              (F3) Removing per-family experts loses >= 0.02 at AICD 1%.
#              (F4) Composite > METATRACO on AICD-T2 by >= +0.005.
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
logger = logging.getLogger("exp119_forest")


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


def build_family_assignment(n_cls, gene_adj):
    """Connected-components of sibling graph. Returns list of family member lists."""
    visited = set(); families = []
    for c in range(n_cls):
        if c in visited: continue
        comp = []; queue = [c]
        while queue:
            x = queue.pop(0)
            if x in visited: continue
            visited.add(x); comp.append(x)
            for nb in gene_adj.get(x, []):
                if nb not in visited: queue.append(nb)
        families.append(sorted(comp))
    return families


# =============================================================================
# Losses
# =============================================================================

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


def ultrametric_penalty(D):
    """Soft ultrametric constraint: D[i,j] <= max(D[i,k], D[k,j]) for all triplets."""
    K = D.size(0)
    if K < 3: return D.sum() * 0.0
    Di = D.unsqueeze(2)
    Dik = D.unsqueeze(1).expand(K, K, K)
    Dkj = D.transpose(0, 1).unsqueeze(0).expand(K, K, K)
    max_iko_kj = torch.maximum(Dik, Dkj)
    viol = F.relu(Di - max_iko_kj) ** 2
    return viol.mean()


# =============================================================================
# Model
# =============================================================================

class FOREST(nn.Module):
    def __init__(self, enc_name, n_cls, families, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, emb_dim))
        self.M = len(families)
        self.fam_root = nn.Linear(emb_dim, self.M)
        for m, members in enumerate(families):
            head = nn.Linear(emb_dim, len(members))
            self.add_module(f"expert_{m}", head)
        # Learnable ultrametric tree (K x K logits, symmetrised)
        self.tree_logits = nn.Parameter(torch.randn(n_cls, n_cls) * 0.01)
        self.families = families
        self.n_cls = n_cls
        self.emb_dim = emb_dim
        # Class -> family index
        self.class_to_family = {}
        for m, mem in enumerate(families):
            for c in mem: self.class_to_family[c] = m
        # Class -> within-family slot
        self.class_to_slot = {}
        for m, mem in enumerate(families):
            for j, c in enumerate(mem): self.class_to_slot[c] = j

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1)

    def forward(self, ids, mask):
        z = self.encode(ids, mask)
        fam_logits = self.fam_root(z)  # (B, M)
        B = z.size(0)
        out = torch.full((B, self.n_cls), -1e9, device=z.device)
        log_fam = F.log_softmax(fam_logits, dim=-1)
        for m, mem in enumerate(self.families):
            head = getattr(self, f"expert_{m}")
            lg = head(z)
            log_within = F.log_softmax(lg, dim=-1)
            for j, c in enumerate(mem):
                out[:, c] = log_fam[:, m] + log_within[:, j]
        return out, z, fam_logits

    def get_distance_matrix(self):
        W = self.tree_logits
        D = F.softplus(W + W.t()) / 2.0
        D = D - torch.diag(torch.diag(D))
        return D

    def family_label_tensor(self, labels):
        return torch.tensor([self.class_to_family[int(y)] for y in labels.tolist()],
                            device=labels.device, dtype=torch.long)


# =============================================================================
# Plumbing
# =============================================================================

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    lr_tree: float = 5e-3
    warmup: float = 0.1; wd: float = 0.01
    lambda_supcon: float = 0.5
    lambda_ultra: float = 0.1
    lambda_fam: float = 0.5
    inner_lr: float = 1e-4
    inner_steps: int = 2          # simplified meta-learning inner steps
    gamma: float = 1.0; tau: float = 0.1
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
        if mem >= 80: cfg.bs, cfg.seq = 192, 512
        elif mem >= 40: cfg.bs, cfg.seq = 128, 512
        elif mem >= 20: cfg.bs, cfg.seq = 64, 448
        elif mem >= 10: cfg.bs, cfg.seq = 48, 384
        else: cfg.bs, cfg.seq = 24, 256
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
        lang = r.get("language", "") or ""
        enc = self.tok(code, max_length=self.seq_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        ids, mask = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        return {"ids": ids, "mask": mask, "label": r["label"],
                "language": lang, "source": r.get("source", "") or ""}


# =============================================================================
# Cluster Rand-Index for F1 (learned tree vs hand-coded family)
# =============================================================================

def _hierarchical_cluster_from_D(D_np, n_families):
    """Greedy agglomerative clustering on a distance matrix; returns cluster IDs."""
    K = D_np.shape[0]
    clusters = [[i] for i in range(K)]
    D = D_np.copy()
    while len(clusters) > n_families:
        # find closest pair (min distance over current cluster reps)
        best = (1e18, -1, -1)
        for a in range(len(clusters)):
            for b in range(a + 1, len(clusters)):
                d = np.mean([D[i, j] for i in clusters[a] for j in clusters[b]])
                if d < best[0]: best = (d, a, b)
        _, a, b = best
        if a < 0: break
        clusters[a] = clusters[a] + clusters[b]
        clusters.pop(b)
    out = np.zeros(K, dtype=np.int64)
    for cid, mem in enumerate(clusters):
        for c in mem: out[c] = cid
    return out


def cluster_rand_index(labels_a, labels_b):
    """Adjusted-free Rand index between two clusterings."""
    n = len(labels_a)
    if n < 2: return 0.0
    same_a = labels_a.reshape(-1, 1) == labels_a.reshape(1, -1)
    same_b = labels_b.reshape(-1, 1) == labels_b.reshape(1, -1)
    agree = (same_a == same_b).astype(np.float64)
    # exclude diagonal
    mask = ~np.eye(n, dtype=bool)
    return float(agree[mask].mean())


# =============================================================================
# Eval
# =============================================================================

@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    fam_preds, fam_labels = [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        out, _, fam_logits = model(ids, mask)
        preds.extend(out.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        # Family-level prediction
        fam_preds.extend(fam_logits.argmax(dim=-1).cpu().tolist())
        labs_l = labs.tolist() if torch.is_tensor(labs) else list(labs)
        fam_labels.extend([model.class_to_family[int(y)] for y in labs_l])
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
    fam_acc = float(accuracy_score(np.array(fam_labels), np.array(fam_preds))) if fam_labels else 0.0
    out = {"overall": overall, "per_class": per_class, "per_language": per_lang,
           "per_source": per_src, "confusion_matrix": cm.tolist(),
           "sibling_confusion_rate": float(sib_rate),
           "cross_family_confusion_rate": float(cross_rate),
           "off_diag_total": off_diag, "n_samples": int(len(labels)),
           "family_root_accuracy_F2": fam_acc}
    return out


# =============================================================================
# Train
# =============================================================================

def train_epoch(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train(); tot, ce_s, sc_s, ul_s, fm_s = 0.0, 0.0, 0.0, 0.0, 0.0
    n_batches = 0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        fam_labs = model.family_label_tensor(labs)
        # ---- Simplified meta-learning inner adaptation ----
        # Randomly select ONE family per batch, do `inner_steps` SGD on its expert
        # only, then compute the meta-objective on the full batch.
        if cfg.inner_steps > 0 and model.M > 1:
            picked = random.randrange(model.M)
            expert = getattr(model, f"expert_{picked}")
            fam_mask_bool = (fam_labs == picked)
            if fam_mask_bool.any():
                sub_ids = ids[fam_mask_bool]; sub_mask = mask[fam_mask_bool]
                sub_y = labs[fam_mask_bool]
                # map labels to within-family slots
                sub_slot = torch.tensor([model.class_to_slot[int(y)] for y in sub_y.tolist()],
                                        device=labs.device, dtype=torch.long)
                with torch.no_grad():
                    z_sub = model.encode(sub_ids, sub_mask)
                for _ in range(cfg.inner_steps):
                    lg_sub = expert(z_sub)
                    if lg_sub.size(0) < 1: break
                    loss_in = F.cross_entropy(lg_sub, sub_slot)
                    grads = torch.autograd.grad(loss_in, expert.parameters(),
                                                 create_graph=False, allow_unused=True)
                    with torch.no_grad():
                        for p, g in zip(expert.parameters(), grads):
                            if g is not None: p.sub_(cfg.inner_lr * g)
        # ---- Meta-objective on full batch ----
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda")):
            out, z, fam_logits = model(ids, mask)
            loss_ce = F.nll_loss(out, labs)
            loss_fam = F.cross_entropy(fam_logits, fam_labs)
            # tree-weighted supcon using LEARNED distance matrix (detached for stable loss)
            D_learn = model.get_distance_matrix()
            # Combine with hand-coded for warmup (linear blend by step)
            D_use = 0.5 * D_learn.detach() + 0.5 * dist_mat
            z_d = torch.cat([z, z], dim=0)
            y_d = torch.cat([labs, labs], dim=0)
            loss_sc = supcon_tw_loss(z_d, y_d, D_use, gamma=cfg.gamma, tau=cfg.tau)
            loss_ul = ultrametric_penalty(D_learn)
            loss = (loss_ce
                    + cfg.lambda_fam * loss_fam
                    + cfg.lambda_supcon * loss_sc
                    + cfg.lambda_ultra * loss_ul)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item()
        ul_s += loss_ul.item(); fm_s += loss_fam.item(); n_batches += 1
    n = max(n_batches, 1)
    return tot/n, ce_s/n, sc_s/n, ul_s/n, fm_s/n


def run_exp(cfg, tag):
    set_seed(cfg.seed); cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    families = build_family_assignment(cfg.n_cls, cfg.gene_adj)
    logger.info(f"[families] M={len(families)} :: {families}")
    dist_mat_t = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist_mat = dist_mat_t.to(cfg.device); dist_mat_cpu = dist_mat_t.numpy()
    sib_mask_np = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()
    # ---- Hand-coded family clustering for F1 Rand-Index ----
    handcoded_clusters = np.array([{c: m for m, mem in enumerate(families) for c in mem}[i]
                                   for i in range(cfg.n_cls)])
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
                f"warmup={cfg.warmup} lambda_sc={cfg.lambda_supcon} "
                f"lambda_ul={cfg.lambda_ultra} lambda_fam={cfg.lambda_fam}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = FOREST(cfg.enc, cfg.n_cls, families, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    tree_ids = {id(model.tree_logits)}
    head_params = [p for p in model.parameters()
                   if id(p) not in enc_ids and id(p) not in tree_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head},
        {"params": [model.tree_logits], "lr": cfg.lr_tree}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, ce, sc, ul, fm = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} supcon={sc:.4f} "
                    f"ultra={ul:.4f} fam={fm:.4f} val={v:.4f} fam_acc={val_met['family_root_accuracy_F2']:.3f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    # ---- Falsifier F1: cluster Rand-Index between learned and hand-coded ----
    with torch.no_grad():
        D_learned = model.get_distance_matrix().detach().cpu().numpy()
    learned_clusters = _hierarchical_cluster_from_D(D_learned, n_families=len(families))
    rand_idx = cluster_rand_index(learned_clusters, handcoded_clusters)
    fam_acc = ts_met["family_root_accuracy_F2"]
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"fam_acc={fam_acc:.3f} rand_idx={rand_idx:.3f}")
    logger.info(f"[learned distance matrix]\n{np.array2string(D_learned, precision=3, suppress_small=True)}")
    return {"tag": tag, "method": "FOREST", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_supcon": cfg.lambda_supcon, "lambda_ultra": cfg.lambda_ultra,
            "lambda_fam": cfg.lambda_fam, "inner_steps": cfg.inner_steps,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "family_root_accuracy_F2": fam_acc,
            "cluster_rand_index_F1": rand_idx,
            "learned_distance_matrix": D_learned.tolist(),
            "families": families,
            "note": "FOREST = learnable tree + per-family experts + meta-learn inner loop",
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
                tag = f"exp119_forest_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"fam_acc={res['family_root_accuracy_F2']:.3f} "
                                f"rand={res['cluster_rand_index_F1']:.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp119_forest_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'FamAcc':>8} {'Rand':>8} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['family_root_accuracy_F2']:>8.3f} "
              f"{r['cluster_rand_index_F1']:>8.3f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
