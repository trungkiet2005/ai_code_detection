# exp77_cronos — Cross-Regime Orchestrated Network via S-Fact aux heads (CRONOS)
# =============================================================================
# Theory-Track exp -- CRONOS (Cross-Regime Orchestrated Network)
#
# ROLE           : GENEPRINT (exp71) failed because it FORCED a channel split.
#                  CRONOS keeps the encoder output SHARED -- the classifier
#                  sees the full z.  We add THREE auxiliary task heads, each
#                  grounded in a distinct S-fact, that REGULARIZE the encoder
#                  to capture intertwined signals from genealogy + decoding +
#                  sibling discrimination:
#                    Head A (S1): pairwise tree-distance regressor on (z_i, z_j)
#                    Head B (S2): decoding-fingerprint regressor on z
#                    Head C (S9): pairwise sibling-vs-cross binary classifier
#                  All aux gradients flow through the same encoder.  The aux
#                  heads are NOT used at inference.
# NAME           : CRONOS  (Cross-Regime Orchestrated Network)
# ARXIV_ID       : novel; multi-task auxiliary co-training is generic, but
#                  the S-fact grounding of three specific aux tasks for
#                  AI-code attribution is new.
# ONE-LINE CLAIM : Single-encoder multi-task co-training with S1+S2+S9 aux
#                  heads regularises the representation more effectively than
#                  any channel decomposition, because the classifier consumes
#                  the FULL representation.
# EQUATION       : z = phi(x)
#                  logits = clf(z)
#                  Head A (B^2 -> 1): t_ij_hat = MLP_A([z_i; z_j])
#                                     L_A = MSE(t_ij_hat, d_tree(y_i, y_j) / D_max)
#                  Head B (B -> 3):    d_hat = MLP_B(z)
#                                     L_B = MSE(d_hat, surrogate(x))
#                  Head C (B^2 -> 1): s_ij_hat = sigmoid(MLP_C([z_i; z_j]))
#                                     L_C = BCE(s_ij_hat, 1{sibling(y_i, y_j)})
#                  Total: L = L_CE + lambda_A L_A + lambda_B L_B + lambda_C L_C
# WHY NOT BEFORE : Multi-task auxiliary co-training has been used in NLP but
#                  with TASK-AGNOSTIC heads (e.g. NSP, MLM).  CRONOS's aux
#                  heads are PROBLEM-SPECIFIC -- each predicts a measurable
#                  S-fact of AI-code attribution.  Unlike GENEPRINT, no channel
#                  split: aux heads are STRUCTURAL REGULARIZERS, not output
#                  factors.
# FALSIFIER      : (F1) Per-aux-head ablation: train with lambda_A=0 (or B/C)
#                       and measure macro-F1 drop.  >+0.005 = real signal.
#                  (F2) Aux task accuracy at end: MSE(t_ij_hat, d_tree) on test
#                       and BCE(s_ij_hat, sibling-truth) accuracy.  If aux
#                       tasks DO NOT converge, the regularization is fake.
#                  (F3) Convergence: aux loss curves -- monotone-decreasing
#                       confirms encoder is genuinely learning aux tasks.
# REPORTS        : Full eval pack + per-aux test loss + aux convergence.
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
logger = logging.getLogger("exp77_cronos")

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
# Model
# =============================================================================

class CRONOSModel(nn.Module):
    def __init__(self, enc_name, n_cls, z_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, z_dim))
        self.clf = nn.Linear(z_dim, n_cls)
        # Head A (S1): pair tree distance regressor on [z_i; z_j]
        self.head_A = nn.Sequential(nn.Linear(2 * z_dim, 128), nn.GELU(),
                                    nn.Linear(128, 1), nn.Sigmoid())
        # Head B (S2): decoding fingerprint regressor (3-dim)
        self.head_B = nn.Sequential(nn.Linear(z_dim, 128), nn.GELU(),
                                    nn.Linear(128, 3), nn.Sigmoid())
        # Head C (S9): pair sibling-vs-cross binary classifier on [z_i; z_j]
        self.head_C = nn.Sequential(nn.Linear(2 * z_dim, 128), nn.GELU(),
                                    nn.Linear(128, 1))
        self.z_dim = z_dim; self.n_cls = n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.proj(sem)

    def forward(self, ids, mask):
        z = self.encode(ids, mask)
        logits = self.clf(z)
        return logits, z


def head_A_loss(model, z, labels, dist_mat, D_max=4.0):
    """Pairwise tree-distance regression. Sample B//2 pairs per batch."""
    B = z.size(0)
    if B < 2: return z.sum() * 0.0
    n_pairs = max(2, B // 2)
    idx_i = torch.randint(0, B, (n_pairs,), device=z.device)
    idx_j = torch.randint(0, B, (n_pairs,), device=z.device)
    keep = idx_i != idx_j
    idx_i = idx_i[keep]; idx_j = idx_j[keep]
    if idx_i.numel() < 2: return z.sum() * 0.0
    z_pair = torch.cat([z[idx_i], z[idx_j]], dim=-1)
    target = dist_mat[labels[idx_i], labels[idx_j]] / D_max
    pred = model.head_A(z_pair).squeeze(-1)
    return F.mse_loss(pred, target)


def head_B_loss(model, z, dec_true):
    pred = model.head_B(z)
    return F.mse_loss(pred, dec_true)


def head_C_loss(model, z, labels, sib_mask):
    """Pairwise sibling binary classifier. Sample B//2 pairs, target = 1 if siblings."""
    B = z.size(0)
    if B < 2: return z.sum() * 0.0
    n_pairs = max(2, B // 2)
    idx_i = torch.randint(0, B, (n_pairs,), device=z.device)
    idx_j = torch.randint(0, B, (n_pairs,), device=z.device)
    keep = idx_i != idx_j
    idx_i = idx_i[keep]; idx_j = idx_j[keep]
    if idx_i.numel() < 2: return z.sum() * 0.0
    z_pair = torch.cat([z[idx_i], z[idx_j]], dim=-1)
    target = sib_mask[labels[idx_i], labels[idx_j]]  # 1.0 sibling, 0.0 else
    pred_logit = model.head_C(z_pair).squeeze(-1)
    return F.binary_cross_entropy_with_logits(pred_logit, target)


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
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_A: float = 0.3; lambda_B: float = 0.2; lambda_C: float = 0.2
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


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        logits, _ = model(ids, mask)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
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
    return {"overall": overall, "per_class": per_class, "per_language": per_lang, "per_source": per_src,
            "confusion_matrix": cm.tolist(), "sibling_confusion_rate": float(sib_rate),
            "cross_family_confusion_rate": float(cross_rate),
            "off_diag_total": off_diag, "n_samples": int(len(labels))}


def train_epoch(model, loader, opt, sch, scaler, cfg, dist_mat, sib_mask):
    model.train(); tot, ce_s, A_s, B_s, C_s = 0.0, 0.0, 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        labs = b["label"].to(cfg.device); dec_true = b["dec_true"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            logits, z = model(ids, mask)
            loss_ce = F.cross_entropy(logits, labs)
            loss_A = head_A_loss(model, z, labs, dist_mat)
            loss_B = head_B_loss(model, z, dec_true)
            loss_C = head_C_loss(model, z, labs, sib_mask)
            loss = loss_ce + cfg.lambda_A * loss_A + cfg.lambda_B * loss_B + cfg.lambda_C * loss_C
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item()
        A_s += loss_A.item(); B_s += loss_B.item(); C_s += loss_C.item()
    n = len(loader)
    return tot/n, ce_s/n, A_s/n, B_s/n, C_s/n


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat_t = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist_mat = dist_mat_t.to(cfg.device); dist_mat_cpu = dist_mat_t.numpy()
    sib_mask_t = build_sibling_mask(cfg.n_cls, cfg.gene_adj)
    sib_mask = sib_mask_t.to(cfg.device); sib_mask_np = sib_mask_t.numpy()
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
                f"lA={cfg.lambda_A} lB={cfg.lambda_B} lC={cfg.lambda_C}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = CRONOSModel(cfg.enc, cfg.n_cls).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist, aux_hist = 0.0, None, [], []
    for epoch in range(cfg.epochs):
        loss, ce, A, B, C = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist_mat, sib_mask)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        aux_hist.append({"A": A, "B": B, "C": C})
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} A={A:.4f} B={B:.4f} C={C:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    ts_met["falsifier"] = {"aux_history_F3": aux_hist, "final_aux_F2": aux_hist[-1] if aux_hist else {}}
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f}")
    return {"tag": tag, "method": "CRONOS", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_A": cfg.lambda_A, "lambda_B": cfg.lambda_B, "lambda_C": cfg.lambda_C,
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
                tag = f"exp77_cronos_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp77_cronos_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*120)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} {'Wall':>8}")
    print("-"*120)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} {r['wall']:>8.0f}s")
    print("="*120)


if __name__ == "__main__":
    main()
