# exp71_geneprint — Tri-Channel Disentangled Genealogy-Print (HERO CANDIDATE)
# =============================================================================
# Theory-Track exp -- GENEPRINT (Tri-Channel Disentangled Genealogy-Print)
#
# ROLE           : Synthesises the insights from the entire Round 1+2 slate:
#                  TKL (sibling weight DECAYS with n), RACO (contrastive weight
#                  GROWS with n), DTKE (dual-tree breaks AICD saturation),
#                  DECOFP (decoding fingerprint helps low-n AICD), PERPSIG (K
#                  virtual decoders can be orthogonal), STYLO (surface
#                  features alone already captured by unixcoder).
#
#                  KEY INSIGHT NOBODY HAS EXPLOITED: the author identity of an
#                  AI-generated code sample factorises as the OUTER PRODUCT of
#                  THREE orthogonal information channels:
#                    (a) Topology  -- where on the genealogy tree            (S1)
#                    (b) Decoding  -- T / top-p / repetition fingerprint     (S2)
#                    (c) Motif     -- per-author AST / token recurrence      (S5)
#                  Every prior method in the slate hits ONE channel.  None
#                  enforce that the three channels carry MUTUALLY DISTINCT
#                  information.  GENEPRINT does, by splitting the encoder
#                  output into z = [z_T; z_D; z_M] with channel-specific
#                  supervision and an HSIC disentanglement penalty.
# NAME           : GENEPRINT  (Tri-Channel Disentangled Genealogy-Print)
# ARXIV_ID       : novel; relates to disentangled-rep (arXiv:2604.21300) but
#                  ours is THREE explicit channels with S-fact grounding,
#                  not the standard content/style two-channel.
# ONE-LINE CLAIM : Author identity of AI-generated code = outer product of
#                  (topology, decoding-style, motif); enforcing the three
#                  channels to be mutually independent in the encoder output
#                  yields strictly more attribution information than any
#                  single channel.
# EQUATION       : z = phi(x) in R^768, split into z_T, z_D, z_M in R^256 each
#                  L_T  = MSE( (1 - cos(z_T_i, z_T_j))/2 ,  d_tree(y_i,y_j)/D_max )
#                  L_D  = MSE( g(z_D),  [rep_entropy(x), TTR(x), burst(x)] )
#                  L_M  = SupCon(z_M, y)             (vanilla, NO tree weight)
#                  L_disen = HSIC(z_T, z_D) + HSIC(z_T, z_M) + HSIC(z_D, z_M)
#                  Total: L = CE(W [z_T;z_D;z_M], y)
#                           + lambda_T L_T + lambda_D L_D + lambda_M L_M
#                           + beta L_disen
# WHY NOT BEFORE : Disentangled representations have been studied for content/
#                  style (text authorship) but never AS A FACTORISATION GROUNDED
#                  IN PROBLEM-SPECIFIC S-FACTS for AI-code attribution.  The
#                  three S-facts (S1/S2/S5) are independently established but
#                  no method has forced them into orthogonal encoder channels
#                  with HSIC, while letting the classifier consume the full
#                  concat.
# FALSIFIER      : Three independent rigorous hooks:
#                  (F1) Per-channel zero-out: zero z_T (or z_D, z_M) at TEST,
#                       measure macro-F1 drop.  If drop of any single channel
#                       equals total drop, decomposition is degenerate.
#                  (F2) HSIC matrix at end of training: if max off-diagonal
#                       HSIC > 0.1, disentanglement failed.
#                  (F3) Spearman correlation between d_Z(z_T_i, z_T_j) and
#                       d_tree(y_i, y_j) on test: if < 0.3, z_T is not
#                       capturing genealogy.
# REPORTS        : full eval pack + (F1) zero-out F1 triple
#                                 + (F2) final HSIC matrix
#                                 + (F3) Spearman_zT_tree
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from dataclasses import dataclass, field
from typing import List, Dict
import re as _re
from collections import Counter

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("scikit-learn"); _ensure("tqdm"); _ensure("scipy")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from scipy.stats import spearmanr
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp71_geneprint")

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


def extract_decoding_surrogate(code: str) -> List[float]:
    tokens = _re.findall(r"\w+|[^\w\s]", code)
    n = max(len(tokens), 1)
    if n >= 4:
        ngrams = [tuple(tokens[i:i+4]) for i in range(n - 3)]
        counts = Counter(ngrams); total = sum(counts.values())
        p = np.array(list(counts.values())) / total
        H = float(-(p * np.log(p + 1e-12)).sum())
        H_max = math.log(max(total, 1) + 1e-12)
        rep_entropy = H / max(H_max, 1e-6)
    else: rep_entropy = 0.0
    identifiers = _re.findall(r"\b[a-zA-Z_]\w*\b", code)
    ttr = len(set(identifiers)) / max(len(identifiers), 1)
    if n >= 100:
        ws = 50; n_w = n // ws; freqs = []
        for w in range(n_w):
            chunk = tokens[w*ws:(w+1)*ws]; c = Counter(chunk)
            top = sum(v for _, v in c.most_common(3)) / ws
            freqs.append(top)
        burst = float(np.std(freqs)) if freqs else 0.0
    else: burst = 0.0
    return [min(max(rep_entropy, 0.0), 1.0),
            min(max(ttr, 0.0), 1.0),
            min(max(burst * 5.0, 0.0), 1.0)]


# =============================================================================
# Model — 3-channel split
# =============================================================================

class GENEPRINTModel(nn.Module):
    def __init__(self, enc_name, n_cls, ch_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.shared = nn.Sequential(nn.Linear(hidden, 768), nn.GELU(), nn.Dropout(0.1))
        self.proj_T = nn.Sequential(nn.Linear(768, ch_dim), nn.GELU())
        self.proj_D = nn.Sequential(nn.Linear(768, ch_dim), nn.GELU())
        self.proj_M = nn.Sequential(nn.Linear(768, ch_dim), nn.GELU())
        self.dec_head = nn.Sequential(
            nn.Linear(ch_dim, 128), nn.GELU(),
            nn.Linear(128, 3), nn.Sigmoid()
        )
        self.clf = nn.Linear(3 * ch_dim, n_cls)
        self.ch_dim = ch_dim

    def forward(self, ids, mask, zero_channel=None):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        h = self.shared(sem)
        z_T = self.proj_T(h)
        z_D = self.proj_D(h)
        z_M = self.proj_M(h)
        d_hat = self.dec_head(z_D)
        # for falsifier F1: zero one channel at inference time
        if zero_channel == "T": z_T = torch.zeros_like(z_T)
        if zero_channel == "D": z_D = torch.zeros_like(z_D)
        if zero_channel == "M": z_M = torch.zeros_like(z_M)
        z_cat = torch.cat([z_T, z_D, z_M], dim=-1)
        logits = self.clf(z_cat)
        return logits, z_T, z_D, z_M, d_hat


# =============================================================================
# Channel-specific losses
# =============================================================================

def topology_loss(z_T, labels, dist_mat, D_max=4.0):
    """Tree-isometry: cos-distance(z_T_i, z_T_j) should track d_tree(y_i, y_j)."""
    B = z_T.size(0)
    if B < 2: return z_T.sum() * 0.0
    z_n = F.normalize(z_T, dim=-1)
    cos = z_n @ z_n.t()                                  # (B, B) in [-1, 1]
    z_dist = (1.0 - cos) / 2.0                           # in [0, 1]
    tree_dist = dist_mat[labels][:, labels] / D_max      # in [0, 1]
    eye = torch.eye(B, device=z_T.device, dtype=torch.bool)
    mask = ~eye
    return F.mse_loss(z_dist[mask], tree_dist[mask])


def supcon_loss(z_M, labels, tau=0.1):
    """Vanilla SupCon on motif channel; no tree weighting."""
    B = z_M.size(0)
    if B < 2: return z_M.sum() * 0.0
    z = F.normalize(z_M, dim=-1)
    sim = (z @ z.t()) / tau
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    exp_s = torch.exp(sim).masked_fill(eye, 0.0)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().masked_fill(eye, 0.0)
    n_pos = pos_mask.sum(dim=-1)
    has_pos = (n_pos > 0).float()
    num = (exp_s * pos_mask).sum(dim=-1).clamp(min=1e-12)
    den = exp_s.sum(dim=-1).clamp(min=1e-12)
    li = -(torch.log(num) - torch.log(den)) * has_pos
    return li.sum() / has_pos.sum().clamp(min=1.0)


def rbf_kernel(X, sigma):
    """Pairwise RBF kernel matrix of X (B, D)."""
    sq = torch.cdist(X, X, p=2) ** 2
    return torch.exp(-sq / (2.0 * sigma * sigma))


def hsic(X, Y, sigma=1.0):
    """Hilbert-Schmidt Independence Criterion (biased estimator)."""
    B = X.size(0)
    if B < 4: return X.sum() * 0.0
    Kx = rbf_kernel(X, sigma)
    Ky = rbf_kernel(Y, sigma)
    H = torch.eye(B, device=X.device) - (1.0 / B)
    return torch.trace(Kx @ H @ Ky @ H) / ((B - 1) ** 2)


def disentangle_loss(z_T, z_D, z_M, sigma=1.0):
    return hsic(z_T, z_D, sigma) + hsic(z_T, z_M, sigma) + hsic(z_D, z_M, sigma)


# =============================================================================
# Plumbing (same as Round 2 template)
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
    # channel loss weights
    lambda_T: float = 0.3      # topology MSE
    lambda_D: float = 0.3      # decoding MSE
    lambda_M: float = 0.3      # motif SupCon
    beta: float = 0.05         # HSIC disentanglement
    hsic_sigma: float = 1.0
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
                "dec_true": torch.tensor(extract_decoding_surrogate(code), dtype=torch.float32),
                "label": r["label"], "language": r.get("language", "") or "",
                "source": r.get("source", "") or ""}


# =============================================================================
# Eval (with falsifier hooks)
# =============================================================================

@torch.no_grad()
def _classify_only(model, loader, cfg, zero_channel=None):
    model.eval(); preds, labels = [], []
    for b in tqdm(loader, desc=f"Eval(zero={zero_channel})"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        logits, _, _, _, _ = model(ids, mask, zero_channel=zero_channel)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
    return float(f1_score(labels, preds, average="macro", zero_division=0,
                          labels=list(range(cfg.n_cls))))


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=False):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    z_T_all, z_D_all, z_M_all, lab_all = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        logits, z_T, z_D, z_M, _ = model(ids, mask)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            z_T_all.append(z_T.detach().cpu().float())
            z_D_all.append(z_D.detach().cpu().float())
            z_M_all.append(z_M.detach().cpu().float())
            lab_all.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
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
    if collect_falsifier and z_T_all:
        zT = torch.cat(z_T_all, dim=0); zD = torch.cat(z_D_all, dim=0); zM = torch.cat(z_M_all, dim=0)
        labs_arr = np.array(lab_all)
        # (F3) Spearman between cos-distance in z_T space and d_tree on a subsample
        rng = np.random.default_rng(0)
        N = zT.size(0); n_sub = min(2000, N)
        idx = rng.choice(N, size=n_sub, replace=False)
        zT_s = F.normalize(zT[idx], dim=-1)
        cos_s = (zT_s @ zT_s.t()).numpy()
        d_emb = (1.0 - cos_s) / 2.0
        d_tre = dist_mat_cpu[labs_arr[idx]][:, labs_arr[idx]] / 4.0
        tri = np.triu_indices(n_sub, k=1)
        rho, _ = spearmanr(d_emb[tri], d_tre[tri])
        # (F2) HSIC (use small batch for memory)
        sub = min(512, N); idx2 = rng.choice(N, size=sub, replace=False)
        zT2, zD2, zM2 = zT[idx2], zD[idx2], zM[idx2]
        with torch.no_grad():
            hTD = float(hsic(zT2, zD2).item())
            hTM = float(hsic(zT2, zM2).item())
            hDM = float(hsic(zD2, zM2).item())
        out["falsifier"] = {
            "spearman_zT_dtree_F3": float(rho) if not np.isnan(rho) else 0.0,
            "hsic_TD_F2": hTD, "hsic_TM_F2": hTM, "hsic_DM_F2": hDM,
            "n_subsample_F3": int(n_sub), "n_subsample_F2": int(sub),
        }
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train()
    tot, ce_s, lt_s, ld_s, lm_s, dz_s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        labs = b["label"].to(cfg.device); dec_true = b["dec_true"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            logits, z_T, z_D, z_M, d_hat = model(ids, mask)
            loss_ce = F.cross_entropy(logits, labs)
            loss_T  = topology_loss(z_T, labs, dist_mat)
            loss_D  = F.mse_loss(d_hat, dec_true)
            loss_M  = supcon_loss(z_M, labs)
            loss_dz = disentangle_loss(z_T.float(), z_D.float(), z_M.float(), cfg.hsic_sigma)
            loss = (loss_ce
                    + cfg.lambda_T * loss_T
                    + cfg.lambda_D * loss_D
                    + cfg.lambda_M * loss_M
                    + cfg.beta * loss_dz)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item()
        lt_s += loss_T.item(); ld_s += loss_D.item(); lm_s += loss_M.item(); dz_s += loss_dz.item()
    n = len(loader)
    return tot/n, ce_s/n, lt_s/n, ld_s/n, lm_s/n, dz_s/n


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat_t = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist_mat = dist_mat_t.to(cfg.device)
    dist_mat_cpu = dist_mat_t.numpy()
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
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} "
                f"warmup={cfg.warmup} lT={cfg.lambda_T} lD={cfg.lambda_D} lM={cfg.lambda_M} beta={cfg.beta}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = GENEPRINTModel(cfg.enc, cfg.n_cls).to(cfg.device)
    enc_param_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_param_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, ce, lt, ld, lm, dz = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} lT={lt:.4f} lD={ld:.4f} lM={lm:.4f} dz={dz:.6f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=True)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    # F1 falsifier: per-channel zero-out at test
    f1_full = test_macro
    f1_zero_T = _classify_only(model, ts_dl, cfg, zero_channel="T")
    f1_zero_D = _classify_only(model, ts_dl, cfg, zero_channel="D")
    f1_zero_M = _classify_only(model, ts_dl, cfg, zero_channel="M")
    ts_met["falsifier"]["zero_out_F1"] = {
        "full": f1_full, "zero_T": f1_zero_T, "zero_D": f1_zero_D, "zero_M": f1_zero_M,
        "drop_T": f1_full - f1_zero_T, "drop_D": f1_full - f1_zero_D, "drop_M": f1_full - f1_zero_M,
    }
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f}")
    logger.info(f"[falsifier] Spearman(z_T, d_tree)={ts_met['falsifier']['spearman_zT_dtree_F3']:.3f} "
                f"HSIC(TD)={ts_met['falsifier']['hsic_TD_F2']:.4f} "
                f"HSIC(TM)={ts_met['falsifier']['hsic_TM_F2']:.4f} "
                f"HSIC(DM)={ts_met['falsifier']['hsic_DM_F2']:.4f}")
    logger.info(f"[falsifier] zero_T drop={f1_full-f1_zero_T:+.4f}  zero_D drop={f1_full-f1_zero_D:+.4f}  zero_M drop={f1_full-f1_zero_M:+.4f}")
    return {"tag": tag, "method": "GENEPRINT", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_T": cfg.lambda_T, "lambda_D": cfg.lambda_D, "lambda_M": cfg.lambda_M, "beta": cfg.beta,
            "val_macro": best_val, "macro": test_macro,
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
                tag = f"exp71_geneprint_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"rho_T={fa['spearman_zT_dtree_F3']:.3f} "
                                f"dropT={fa['zero_out_F1']['drop_T']:+.3f} "
                                f"dropD={fa['zero_out_F1']['drop_D']:+.3f} "
                                f"dropM={fa['zero_out_F1']['drop_M']:+.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp71_geneprint_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'rho_T':>7} {'dropT':>7} {'dropD':>7} {'dropM':>7} {'Wall':>8}")
    print("-"*150)
    for r in results:
        fa = r['test_metrics']['falsifier']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['spearman_zT_dtree_F3']:>+7.3f} "
              f"{fa['zero_out_F1']['drop_T']:>+7.3f} {fa['zero_out_F1']['drop_D']:>+7.3f} "
              f"{fa['zero_out_F1']['drop_M']:>+7.3f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
