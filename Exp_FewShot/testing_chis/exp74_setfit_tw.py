# exp74_setfit_tw — SetFit with Tree-Weighted negative sampling (SETFIT-TW)
# =============================================================================
# Theory-Track exp -- SETFIT-TW (SetFit + Tree-Weighted negatives)
#
# ROLE           : SetFit (Tunstall et al., arXiv:2209.11055) was the few-shot
#                  text-classification SOTA in 2022 -- 8 labelled examples per
#                  class competing with full-data RoBERTa.  It is two-stage:
#                    Stage 1: contrastive Siamese on sentence pairs.
#                    Stage 2: classifier head trained on frozen embeddings.
#                  No prior work has adapted SetFit to code attribution with a
#                  TREE-WEIGHTED NEGATIVE SAMPLER -- forcing the contrastive
#                  stage to distinguish siblings (same family, different
#                  generator) at higher loss weight than cross-family pairs.
# NAME           : SETFIT-TW  (SetFit with Tree-Weighted negatives)
# ARXIV_ID       : arXiv:2209.11055 + tree-weighted pair sampling (S1).
# ONE-LINE CLAIM : Sibling-weighted Siamese pretraining followed by a frozen-
#                  encoder linear head produces few-shot embeddings that
#                  generalise across regimes without further fine-tuning.
# EQUATION       : Stage 1 (epochs 1..N-1):
#                    L_supcon_tw = SupCon with negatives re-weighted by
#                                  w_neg(i,j) = exp(-gamma * d_tree(y_i, y_j))
#                                  so siblings are hard negatives.
#                  Stage 2 (last 2 epochs):
#                    freeze encoder; train W in (d, n_cls) only.
#                    L_CE = CE( W z, y )
# WHY NOT BEFORE : SetFit was designed for TEXT and uses flat-label negatives.
#                  The two-stage decoupling lets us cleanly study whether a
#                  TREE-WEIGHTED REPRESENTATION (stage 1) alone -- without
#                  fine-tuning the encoder during stage 2 -- is sufficient
#                  for AI-code attribution.
# FALSIFIER      : (F1) Stage-2 frozen-head F1 vs stage-1 nearest-prototype F1.
#                       If frozen-head F1 dominates, the contrastive
#                       embedding alone is incomplete.
#                  (F2) Pair-similarity AUC for positive vs negative pairs at
#                       end of stage 1.  AUC > 0.9 indicates strong Siamese
#                       separation.
#                  (F3) Sibling-confusion rate at test.  Tree-weighted negs
#                       should reduce this vs flat-weighted negs.
# REPORTS        : Full eval pack + (F1) frozen-head vs nearest-proto F1
#                                 + (F2) pair-AUC end-of-stage-1
#                                 + (F3) sibling_confusion_rate (already in pack)
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
                             recall_score, confusion_matrix, roc_auc_score)
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp74_setfit_tw")

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

class SETFITTWModel(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        self.emb_dim = emb_dim
        self.n_cls = n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1)

    def forward(self, ids, mask):
        z = self.encode(ids, mask)
        return self.clf(z), z


def supcon_tw_loss(z, labels, dist_mat, gamma=1.0, tau=0.1):
    """SupCon with negatives weighted by exp(-gamma * d_tree).

    Sibling negatives get HIGH weight -> harder.  Cross-family negatives
    get LOW weight -> already easy.
    """
    B = z.size(0)
    if B < 2: return z.sum() * 0.0
    sim = (z @ z.t()) / tau
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()
    eye = torch.eye(B, device=z.device, dtype=torch.bool)
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)).float().masked_fill(eye, 0.0)
    neg_mask = (labels.unsqueeze(0) != labels.unsqueeze(1)).float()
    dij = dist_mat[labels][:, labels]
    w_neg = torch.exp(-gamma * dij)                          # (B, B) sibling > cross
    w_pair = pos_mask + neg_mask * w_neg                     # pos w=1, neg w=tree-decayed
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

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    gamma: float = 1.0; tau: float = 0.1
    stage2_epochs: int = 2   # last 2 epochs: freeze encoder, train clf head
    emb_dim: int = 256
    device: str = "cuda"; gene_adj: dict = field(default_factory=dict)


def adaptive_schedule(cfg):
    f = cfg.frac
    if f <= 0.02: cfg.epochs, cfg.lr_enc, cfg.warmup = 10, 3e-5, 0.20
    elif f <= 0.10: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 3e-5, 0.15
    else: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 4e-5, 0.10
    cfg.stage2_epochs = max(2, cfg.epochs // 3)
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
def _all_embeddings(model, loader, cfg):
    model.eval()
    zs, ys = [], []
    for b in tqdm(loader, desc="Embed"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        z = model.encode(ids, mask)
        zs.append(z.detach().cpu().float())
        ys.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
    return torch.cat(zs, dim=0).numpy(), np.array(ys)


def _nearest_proto_predict(z_train, y_train, z_test, n_cls):
    """Compute class-prototype as mean of train embeddings, predict by nearest."""
    protos = np.stack([z_train[y_train == c].mean(0) if (y_train == c).sum() > 0
                       else np.zeros(z_train.shape[1]) for c in range(n_cls)], axis=0)
    d = -(z_test @ protos.T)                  # negative cosine (cos = z @ proto since L2-normed-ish)
    return d.argmin(axis=-1), protos


def _pair_auc(z, y, max_pairs=20000, seed=0):
    rng = np.random.default_rng(seed)
    N = z.shape[0]
    if N < 4: return 0.5
    n = min(max_pairs, N * 2)
    i = rng.integers(0, N, size=n); j = rng.integers(0, N, size=n)
    keep = i != j
    i = i[keep]; j = j[keep]
    sim = (z[i] * z[j]).sum(axis=-1)
    same = (y[i] == y[j]).astype(np.int32)
    if same.sum() == 0 or same.sum() == len(same): return 0.5
    try: return float(roc_auc_score(same, sim))
    except Exception: return 0.5


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=False, train_emb=None):
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
    out = {"overall": overall, "per_class": per_class, "per_language": per_lang, "per_source": per_src,
           "confusion_matrix": cm.tolist(), "sibling_confusion_rate": float(sib_rate),
           "cross_family_confusion_rate": float(cross_rate),
           "off_diag_total": off_diag, "n_samples": int(len(labels))}
    if collect_falsifier and train_emb is not None:
        z_tr, y_tr = train_emb
        z_te, y_te = _all_embeddings(model, loader, cfg)
        proto_preds, _ = _nearest_proto_predict(z_tr, y_tr, z_te, n_cls)
        proto_f1 = float(f1_score(y_te, proto_preds, average="macro", zero_division=0, labels=list(range(n_cls))))
        auc = _pair_auc(z_te, y_te)
        out["falsifier"] = {
            "frozen_head_F1_macro": overall["macro_f1"],
            "nearest_proto_F1_macro": proto_f1,
            "delta_head_vs_proto_F1": overall["macro_f1"] - proto_f1,
            "pair_AUC_F2": auc,
            "sib_conf_rate_F3": float(sib_rate),
        }
    return out


def train_stage1(model, loader, opt, sch, scaler, cfg, dist_mat):
    """Stage 1: SupCon-TW + light CE (so clf head doesn't lag)."""
    model.train(); tot, sc_s, ce_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Stage1"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            logits, z = model(ids, mask)
            loss_sc = supcon_tw_loss(z, labs, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            loss_ce = F.cross_entropy(logits, labs)
            loss = loss_sc + 0.3 * loss_ce
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); sc_s += loss_sc.item(); ce_s += loss_ce.item()
    n = len(loader)
    return tot/n, sc_s/n, ce_s/n


def train_stage2(model, loader, opt, sch, scaler, cfg):
    """Stage 2: freeze encoder + proj, train ONLY clf head."""
    for p in model.encoder.parameters(): p.requires_grad = False
    for p in model.proj.parameters(): p.requires_grad = False
    model.encoder.eval()                                       # frozen BN/Dropout off
    model.proj.eval()
    model.clf.train()
    tot, ce_s = 0.0, 0.0
    for b in tqdm(loader, desc="Stage2"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            with torch.no_grad():
                z = model.encode(ids, mask)
            logits = model.clf(z)
            loss = F.cross_entropy(logits, labs)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.clf.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss.item()
    return tot / len(loader)


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
    total_steps_s1 = max(1, len(tr_ds) // cfg.bs) * max(1, cfg.epochs - cfg.stage2_epochs)
    total_steps_s2 = max(1, len(tr_ds) // cfg.bs) * cfg.stage2_epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} (stage1={cfg.epochs - cfg.stage2_epochs}, "
                f"stage2={cfg.stage2_epochs}) lr_enc={cfg.lr_enc} warmup={cfg.warmup} gamma={cfg.gamma}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = SETFITTWModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    # Stage 1 optimizer (encoder + all heads)
    opt_s1 = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch_s1 = get_cosine_schedule_with_warmup(opt_s1, max(1, int(total_steps_s1 * cfg.warmup)), total_steps_s1)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    # === Stage 1 ===
    for epoch in range(cfg.epochs - cfg.stage2_epochs):
        loss, sc, ce = train_stage1(model, tr_dl, opt_s1, sch_s1, scaler, cfg, dist_mat)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[s1 epoch {epoch+1}] loss={loss:.4f} supcon={sc:.4f} ce={ce:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    # === Stage 2 (freeze encoder/proj, retrain clf head) ===
    # Re-init clf head for clean Stage 2
    nn.init.kaiming_uniform_(model.clf.weight, a=math.sqrt(5))
    if model.clf.bias is not None: nn.init.zeros_(model.clf.bias)
    opt_s2 = torch.optim.AdamW([{"params": model.clf.parameters(), "lr": 5e-3}], weight_decay=0.0)
    sch_s2 = get_cosine_schedule_with_warmup(opt_s2, max(1, int(total_steps_s2 * 0.1)), total_steps_s2)
    for epoch in range(cfg.stage2_epochs):
        loss = train_stage2(model, tr_dl, opt_s2, sch_s2, scaler, cfg)
        # for eval we need model.eval(); train_stage2 already left encoder/proj in eval
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[s2 epoch {epoch+1}] head_loss={loss:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    # === Eval with falsifier ===
    z_tr, y_tr = _all_embeddings(model, tr_dl, cfg)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu,
                        collect_falsifier=True, train_emb=(z_tr, y_tr))
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    fa = ts_met["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"proto={fa['nearest_proto_F1_macro']:.4f} "
                f"d_head-proto={fa['delta_head_vs_proto_F1']:+.4f} "
                f"pair_AUC={fa['pair_AUC_F2']:.3f}")
    return {"tag": tag, "method": "SETFIT-TW", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "stage2_epochs": cfg.stage2_epochs,
            "lr_enc": cfg.lr_enc, "gamma": cfg.gamma, "tau": cfg.tau,
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
                tag = f"exp74_setfittw_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"proto={fa['nearest_proto_F1_macro']:.4f} "
                                f"AUC={fa['pair_AUC_F2']:.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp74_setfittw_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'Proto':>8} {'AUC':>7} {'Wall':>8}")
    print("-"*140)
    for r in results:
        fa = r['test_metrics']['falsifier']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['nearest_proto_F1_macro']:>8.4f} {fa['pair_AUC_F2']:>7.3f} {r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
