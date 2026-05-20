# exp127 — JEPA
# NAME       : JEPA (Joint Embedding Predictive Architecture for Code Authorship)
# REFERENCE  : new for code; I-JEPA (Assran et al. 2023, arXiv:2301.08243),
#              V-JEPA (Bardes et al. 2024, arXiv:2404.08471); foundational
#              vision paper by Yann LeCun group. Never applied to code
#              attribution before.
# CLAIM      : The right pretext task for code authorship is NOT
#              token-level MLM (which optimises for surface reconstruction).
#              The right pretext is to predict the EMBEDDING of a masked
#              code span from the context — in REPRESENTATION SPACE, not
#              token space. The predictor never has to decode back to text;
#              it only has to predict what the encoder WOULD HAVE produced.
#              This forces the encoder to represent abstract style
#              properties that survive masking, exactly the property we
#              want for attribution.
# EQUATION   : x = code; mask spans M = {[a_i, b_i]}
#              z_full = encoder(x)                       # (L, d)
#              z_context = encoder(x_masked)             # (L, d), masked spans zeroed
#              z_target = stop_grad(z_full[M])           # targets, no grad
#              z_pred = predictor(z_context, M)          # predict embeddings at masked positions
#              L_jepa = ||z_pred - z_target||^2_2
#              Fine-tune: classifier head on encoder mean-pool
#              L_total = L_jepa (pretext) -> L_CE + lambda*TRACO_supcon (finetune)
# WHY NEW    : Code attribution has never used a JEPA-style objective.
#              CodeBERT / UniXcoder use token MLM; this paper's TRACO uses
#              contrastive on full embeddings. JEPA is the first to train
#              the encoder to predict its OWN OUTPUT at masked positions —
#              a self-prediction objective that does NOT require a decoder.
# WOW HOOK   : "We don't predict tokens. We predict EMBEDDINGS. The
#              encoder learns to imagine its own output at positions it
#              cannot see — a self-prediction task that captures abstract
#              style without ever decoding back to text."
# FALSIFIER  : (F1) JEPA-pretrained encoder + TRACO finetune beats
#              MLM-pretrained UniXcoder + TRACO finetune by >= 0.005 at 1%.
#              (F2) The predictor's MSE on held-out spans is non-trivial
#              (< 0.1, but > 0.001) — i.e., it actually predicts something
#              non-zero but not perfect.
#              (F3) Removing the JEPA pretext (random init predictor)
#              loses >= 0.005 at 1% — proves pretext is doing real work.
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
logger = logging.getLogger("exp127_jepa")

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
# JEPA model
# =============================================================================

class JEPAModel(nn.Module):
    """Joint Embedding Predictive Architecture for code.

    Context encoder (UniXcoder) produces context embeddings z_context.
    Target encoder (EMA copy of context encoder) produces target embeddings z_target.
    Predictor (small Transformer) predicts target embeddings from context embeddings.
    """
    def __init__(self, enc_name, n_cls, emb_dim=256, predictor_dim=384):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        # Target encoder = EMA copy of context encoder
        self.target_encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        # Small predictor: 2-layer transformer in predictor_dim space
        encoder_layer = nn.TransformerEncoderLayer(d_model=predictor_dim, nhead=6, batch_first=True,
                                                   dim_feedforward=predictor_dim*2, activation="gelu")
        self.predictor = nn.TransformerEncoder(encoder_layer, num_layers=2)
        self.context_to_pred = nn.Linear(h, predictor_dim)
        self.pred_to_target = nn.Linear(predictor_dim, h)
        # Mask token in predictor space
        self.mask_token = nn.Parameter(torch.randn(1, 1, predictor_dim) * 0.02)
        # Classification head (for finetune phase)
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        self.emb_dim = emb_dim
        self.h = h
        self.n_cls = n_cls

    @torch.no_grad()
    def update_target(self, momentum=0.996):
        """EMA update of target encoder from context encoder."""
        for p_t, p_c in zip(self.target_encoder.parameters(), self.encoder.parameters()):
            p_t.data.mul_(momentum).add_(p_c.data, alpha=1 - momentum)

    def encode_context(self, ids, mask, masked_positions=None):
        """Encode with masked-out positions zeroed.
        masked_positions: (B, L) boolean mask of which positions to mask."""
        out = self.encoder(input_ids=ids, attention_mask=mask)
        h_ctx = out.last_hidden_state  # (B, L, h)
        if masked_positions is not None:
            keep = (~masked_positions).unsqueeze(-1).float()
            h_ctx = h_ctx * keep
        return h_ctx

    def encode_target(self, ids, mask):
        with torch.no_grad():
            out = self.target_encoder(input_ids=ids, attention_mask=mask)
            return out.last_hidden_state

    def predict_masked(self, h_ctx, masked_positions, attention_mask):
        """Predict target embeddings at masked positions from context."""
        B, L, _ = h_ctx.shape
        x = self.context_to_pred(h_ctx)  # (B, L, predictor_dim)
        mask_tok = self.mask_token.expand(B, L, -1)
        x = torch.where(masked_positions.unsqueeze(-1), mask_tok, x)
        key_padding_mask = (attention_mask == 0)  # True = padding
        x = self.predictor(x, src_key_padding_mask=key_padding_mask)
        return self.pred_to_target(x)  # (B, L, h)

    def encode_for_clf(self, ids, mask):
        """Standard encoder for classification (no masking)."""
        out = self.encoder(input_ids=ids, attention_mask=mask)
        h = out.last_hidden_state
        mask_f = mask.unsqueeze(-1).float()
        sem = (h * mask_f).sum(1) / mask_f.sum(1).clamp(min=1)
        z = F.normalize(self.proj(sem), dim=-1)
        return z, self.clf(z)


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
# JEPA masking strategy
# =============================================================================

def jepa_random_mask(attention_mask, mask_ratio=0.4, span_len_min=4, span_len_max=12):
    """Generate random span masks. For each sample, mask `mask_ratio` fraction of
    non-pad tokens via random spans of length [min, max]."""
    B, L = attention_mask.shape
    masked = torch.zeros_like(attention_mask, dtype=torch.bool)
    for b in range(B):
        valid_len = int(attention_mask[b].sum().item())
        if valid_len < span_len_max * 2: continue
        target_n_masked = int(valid_len * mask_ratio)
        n_masked = 0
        attempts = 0
        while n_masked < target_n_masked and attempts < 20:
            attempts += 1
            span_len = random.randint(span_len_min, span_len_max)
            start = random.randint(0, valid_len - span_len)
            end = start + span_len
            new_count = int(masked[b, start:end].sum().item())
            if new_count == 0:
                masked[b, start:end] = True
                n_masked += span_len
    return masked


# =============================================================================
# Plumbing
# =============================================================================

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 64; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    emb_dim: int = 256
    # JEPA-specific
    pretext_epochs: int = 3
    mask_ratio: float = 0.4
    predictor_dim: int = 384
    ema_momentum: float = 0.996
    lr_pretext: float = 1e-4
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
        # JEPA: 3 encoder passes (context+target during pretext, encoder during finetune)
        # be conservative
        if mem >= 80: cfg.bs, cfg.seq = 64, 512
        elif mem >= 40: cfg.bs, cfg.seq = 48, 512
        elif mem >= 10: cfg.bs, cfg.seq = 24, 384
        else: cfg.bs, cfg.seq = 12, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} (JEPA: 2x encoders) seq={cfg.seq}")
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


class FSDS_JEPA(TD):
    """Dataset returning UniXcoder-tokenized code with language and source metadata."""
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42):
        self.data = data; self.tok = tok; self.seq_len = seq_len
        self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_JEPA] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def _tokenize(self, code):
        # UniXcoder convention: prepend <encoder_only> for encoder-only tasks
        enc_text = "<encoder_only>" + code
        enc = self.tok(enc_text, max_length=self.seq_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        return enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        lang = r.get("language", "") or ""
        ids0, mask0 = self._tokenize(code)
        return {"ids0": ids0, "mask0": mask0, "label": r["label"],
                "language": lang, "source": r.get("source", "") or ""}


# =============================================================================
# JEPA pretext training
# =============================================================================

def jepa_pretext_epoch(model, loader, opt, cfg):
    """One epoch of JEPA self-prediction pretext."""
    model.train()
    tot_mse, n_batch = 0.0, 0
    for b in tqdm(loader, desc="Pretext"):
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device)
        masked_pos = jepa_random_mask(mask, mask_ratio=cfg.mask_ratio).to(cfg.device)
        # If no positions masked in this batch, skip
        if masked_pos.sum() == 0:
            continue
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            h_ctx = model.encode_context(ids, mask, masked_pos)
            h_tgt = model.encode_target(ids, mask)
            h_pred = model.predict_masked(h_ctx, masked_pos, mask)
            # MSE only at masked positions, normalised by num masked positions * hidden dim
            mp = masked_pos.unsqueeze(-1).float()
            denom = mp.sum().clamp(min=1.0) * h_tgt.size(-1)
            mse = ((h_pred - h_tgt) ** 2 * mp).sum() / denom
        opt.zero_grad()
        mse.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        opt.step()
        model.update_target(momentum=cfg.ema_momentum)
        tot_mse += float(mse.item()); n_batch += 1
    return tot_mse / max(n_batch, 1)


# =============================================================================
# Fine-tune (CE + TRACO-style supcon, no augmentation -- single view)
# =============================================================================

def train_epoch_finetune(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Finetune"):
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z, logits = model.encode_for_clf(ids, mask)
            loss_ce = F.cross_entropy(logits, labs)
            # Single-view supcon: positives = same label, no augmentation needed
            loss_sc = supcon_tw_loss(z, labs, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            loss = loss_ce + cfg.lambda_aug * loss_sc
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item()
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device); labs = b["label"]
        _, logits = model.encode_for_clf(ids, mask)
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
    tr_ds = FSDS_JEPA(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS_JEPA(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1)
    ts_ds = FSDS_JEPA(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2)
    logger.info(f"[sched] frac={cfg.frac} pretext_ep={cfg.pretext_epochs} finetune_ep={cfg.epochs} "
                f"lr_enc={cfg.lr_enc} warmup={cfg.warmup} mask_ratio={cfg.mask_ratio}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = JEPAModel(cfg.enc, cfg.n_cls, cfg.emb_dim, cfg.predictor_dim).to(cfg.device)

    # ===== Phase 1: JEPA pretext =====
    pretext_params = (list(model.encoder.parameters())
                      + list(model.predictor.parameters())
                      + list(model.context_to_pred.parameters())
                      + list(model.pred_to_target.parameters())
                      + [model.mask_token])
    opt_pretext = torch.optim.AdamW(pretext_params, lr=cfg.lr_pretext, weight_decay=cfg.wd)
    pretext_mse_history = []
    for ep in range(cfg.pretext_epochs):
        mse = jepa_pretext_epoch(model, tr_dl, opt_pretext, cfg)
        pretext_mse_history.append(mse)
        logger.info(f"[pretext epoch {ep+1}/{cfg.pretext_epochs}] mse={mse:.5f}")
    final_pretext_mse = pretext_mse_history[-1] if pretext_mse_history else float("nan")

    # ===== Phase 2: TRACO-style fine-tune =====
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters()
                   if id(p) not in enc_ids and p.requires_grad
                   and not any(id(p) == id(tp) for tp in model.target_encoder.parameters())]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, ce, sc = train_epoch_finetune(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[finetune epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} supcon={sc:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"pretext_mse={final_pretext_mse:.5f}")
    return {"tag": tag, "method": "JEPA",
            "note": "Joint Embedding Predictive Architecture for code (2-phase: JEPA pretext + TRACO finetune)",
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "pretext_epochs": cfg.pretext_epochs, "mask_ratio_used": cfg.mask_ratio,
            "ema_momentum": cfg.ema_momentum, "predictor_dim": cfg.predictor_dim,
            "lambda_aug": cfg.lambda_aug, "gamma": cfg.gamma, "tau": cfg.tau,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "jepa_final_pretext_mse": float(final_pretext_mse),
            "jepa_pretext_mse_history": [float(x) for x in pretext_mse_history],
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
                tag = f"exp127_jepa_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"pretext_mse={res['jepa_final_pretext_mse']:.5f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp127_jepa_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'PretextMSE':>12} {'Wall':>8}")
    print("-"*140)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['jepa_final_pretext_mse']:>12.5f} {r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
