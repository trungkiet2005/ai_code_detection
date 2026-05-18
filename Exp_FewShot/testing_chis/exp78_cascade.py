# exp78_cascade — Hierarchical Family-then-Sibling Cascade Decoding (CASCADE)
# =============================================================================
# Theory-Track exp -- CASCADE (Hierarchical Family-then-Sibling Decoding)
#
# ROLE           : All prior methods classify via a flat softmax over n_cls.
#                  CASCADE factorises the decoding probabilistically along the
#                  genealogy tree (S1) at the DECODING level (not loss-reshape,
#                  not feature engineering, not prototype constraint):
#                    p(y = k | x) = p_family(f(k) | x) * p_sibling(s(k) | f(k), x)
#                  The model has TWO heads: a family classifier and a within-
#                  family sibling classifier.  At inference, the full class
#                  probability is the product.  This is mathematically clean
#                  and the family/sibling accuracy can be analysed separately
#                  -- a signal nobody else extracts.
# NAME           : CASCADE  (Hierarchical Family-then-Sibling Decoding)
# ARXIV_ID       : Hierarchical Softmax (Morin & Bengio 2005) generalised to
#                  LLM-genealogy attribution.  Hierarchical decoding has been
#                  used for word embeddings but NOT for LLM-attribution where
#                  the hierarchy is a known external graph.
# ONE-LINE CLAIM : AI-code attribution factorises along genealogy as
#                  P(class) = P(family) * P(sibling | family);  explicit
#                  hierarchical decoding captures this factorisation more
#                  effectively than any loss/feature/contrastive modifier.
# EQUATION       : f(k) = family index of class k
#                  s(k) = within-family index of class k
#                  p_family(f | x) = softmax_f( W_F z )         (n_F-way)
#                  p_sibling(s | f, x) = softmax_s( W_S[f] z * mask[f] )
#                  p(y = k | x) = p_family(f(k) | x) * p_sibling(s(k) | f(k), x)
#                  L = CE_family(p_family, f(y))
#                      + lambda_s * CE_sibling(p_sibling[f(y)], s(y))
#                  (sibling CE conditional on TRUE family at train time)
# WHY NOT BEFORE : Hierarchical softmax is standard for LARGE-vocab language
#                  modelling.  Applying it to AI-code attribution where the
#                  hierarchy is THE GENEALOGY TREE OF THE GENERATORS is novel.
#                  Prior tree-aware attribution methods (LASCL, HCAL, our TKL/
#                  GSCE/SCR) treat the tree as a LOSS WEIGHT prior, not as the
#                  DECODING STRUCTURE itself.
# FALSIFIER      : (F1) Family-only acc: argmax p_family on test.  If this
#                       is at "ceiling" (e.g. 0.85+ on AICD), siblings are
#                       the sole bottleneck.
#                  (F2) Sibling conditional acc: p_sibling | TRUE family.
#                       Measures how much info is in sibling discrimination
#                       given the family is known.  Low => the encoder cannot
#                       distinguish siblings even with hint.
#                  (F3) Product accuracy: argmax_{k} p(y=k|x).  Should equal
#                       (or exceed) flat CE baseline.
# REPORTS        : Full eval pack + (F1) family-acc, (F2) sibling-cond-acc,
#                                   (F3) joint-acc.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

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
logger = logging.getLogger("exp78_cascade")

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


def compute_family_mapping(n_cls, adj):
    """Union-find on sibling adjacency -> family decomposition.

    Returns:
      family_of_class: list[int] of length n_cls, value in [0, n_families)
      sib_idx: list[int] of length n_cls, value in [0, max_sib)
      n_families: int
      max_sib: int
      fam_classes: dict[fam -> sorted list of class ids]
    """
    parent = list(range(n_cls))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py
    for i in range(n_cls):
        for j in adj.get(i, []):
            union(i, j)
    roots = sorted({find(i) for i in range(n_cls)})
    root_to_fam = {r: idx for idx, r in enumerate(roots)}
    family_of_class = [root_to_fam[find(i)] for i in range(n_cls)]
    fam_classes = {f: [] for f in range(len(roots))}
    for c in range(n_cls):
        fam_classes[family_of_class[c]].append(c)
    for f in fam_classes:
        fam_classes[f].sort()
    sib_idx = [0] * n_cls
    for c in range(n_cls):
        f = family_of_class[c]
        sib_idx[c] = fam_classes[f].index(c)
    max_sib = max(len(v) for v in fam_classes.values())
    return family_of_class, sib_idx, len(roots), max_sib, fam_classes


# =============================================================================
# Model
# =============================================================================

class CASCADEModel(nn.Module):
    def __init__(self, enc_name, n_cls, n_families, max_sib, z_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, z_dim))
        self.head_F = nn.Linear(z_dim, n_families)
        # Sibling head per family: produce (n_families * max_sib) logits, reshape to (n_families, max_sib)
        self.head_S = nn.Linear(z_dim, n_families * max_sib)
        self.n_cls = n_cls
        self.n_families = n_families
        self.max_sib = max_sib
        self.z_dim = z_dim

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.proj(sem)

    def forward(self, ids, mask):
        z = self.encode(ids, mask)
        log_F = self.head_F(z)                                  # (B, n_F)
        log_S = self.head_S(z).view(-1, self.n_families, self.max_sib)  # (B, n_F, max_sib)
        return log_F, log_S, z


def cascade_class_logprob(log_F, log_S, family_of_class, sib_idx, sib_mask):
    """Compute log p(y=k|x) for all k via product decomposition.

    log_F: (B, n_F)   raw family logits
    log_S: (B, n_F, max_sib) raw sibling logits per family
    sib_mask: (n_F, max_sib) 1 where sibling slot exists, 0 otherwise
    Returns: (B, n_cls) full class log-probabilities (consistent: sums to 1 across k).
    """
    B = log_F.size(0)
    n_cls = len(family_of_class)
    log_p_F = F.log_softmax(log_F, dim=-1)                       # (B, n_F)
    # Mask out non-existent sibling slots with -inf BEFORE softmax
    log_S_masked = log_S + (sib_mask.log().unsqueeze(0))          # (B, n_F, max_sib), -inf where no sibling
    log_p_S = F.log_softmax(log_S_masked, dim=-1)                # (B, n_F, max_sib)
    # Assemble class log-prob via family + sibling
    fam_t = torch.tensor(family_of_class, device=log_F.device, dtype=torch.long)
    sib_t = torch.tensor(sib_idx, device=log_F.device, dtype=torch.long)
    log_p_F_per_k = log_p_F[:, fam_t]                            # (B, n_cls)
    log_p_S_per_k = log_p_S[:, fam_t, sib_t]                     # (B, n_cls)
    return log_p_F_per_k + log_p_S_per_k                          # (B, n_cls)


def cascade_loss(log_F, log_S, labels, family_of_class_t, sib_idx_t, sib_mask, lambda_s=1.0):
    """CE on family + conditional CE on sibling given true family."""
    fam_target = family_of_class_t[labels]                       # (B,)
    sib_target = sib_idx_t[labels]                               # (B,)
    L_F = F.cross_entropy(log_F, fam_target)
    log_S_masked = log_S + sib_mask.log().unsqueeze(0)
    # For each sample, select the row corresponding to its true family
    B = log_S.size(0)
    log_S_sel = log_S_masked[torch.arange(B, device=log_S.device), fam_target]  # (B, max_sib)
    L_S = F.cross_entropy(log_S_sel, sib_target)
    return L_F + lambda_s * L_S, L_F, L_S


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
    lambda_s: float = 1.0
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
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu,
              family_of_class, sib_idx, sib_mask, family_of_class_t, sib_idx_t):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    fam_preds, fam_correct = [], []
    sib_cond_correct = []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        log_F, log_S, _ = model(ids, mask)
        log_p_class = cascade_class_logprob(log_F, log_S, family_of_class, sib_idx, sib_mask)
        preds.extend(log_p_class.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        # falsifier hooks
        labs_t = labs.to(cfg.device) if torch.is_tensor(labs) else torch.tensor(labs, device=cfg.device)
        fam_t = family_of_class_t[labs_t]
        sib_t = sib_idx_t[labs_t]
        fam_pred = log_F.argmax(dim=-1)
        fam_correct.extend((fam_pred == fam_t).cpu().tolist())
        fam_preds.extend(fam_pred.cpu().tolist())
        # sibling conditional on TRUE family
        log_S_masked = log_S + sib_mask.log().unsqueeze(0)
        log_S_sel = log_S_masked[torch.arange(log_S.size(0), device=log_S.device), fam_t]
        sib_pred = log_S_sel.argmax(dim=-1)
        sib_cond_correct.extend((sib_pred == sib_t).cpu().tolist())
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
            "off_diag_total": off_diag, "n_samples": int(len(labels)),
            "falsifier": {
                "family_acc_F1": float(np.mean(fam_correct)) if fam_correct else 0.0,
                "sibling_cond_acc_F2": float(np.mean(sib_cond_correct)) if sib_cond_correct else 0.0,
                "joint_acc_F3": float(np.mean(preds == labels)),
            }}


def train_epoch(model, loader, opt, sch, scaler, cfg,
                family_of_class_t, sib_idx_t, sib_mask):
    model.train(); tot, lf_s, ls_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            log_F, log_S, _ = model(ids, mask)
            loss, L_F, L_S = cascade_loss(log_F, log_S, labs, family_of_class_t, sib_idx_t, sib_mask, lambda_s=cfg.lambda_s)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); lf_s += L_F.item(); ls_s += L_S.item()
    n = len(loader)
    return tot/n, lf_s/n, ls_s/n


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat_t = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist_mat = dist_mat_t.to(cfg.device); dist_mat_cpu = dist_mat_t.numpy()
    sib_mask_np = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()
    fam_of_class, sib_idx, n_families, max_sib, fam_classes = compute_family_mapping(cfg.n_cls, cfg.gene_adj)
    logger.info(f"[hier] n_cls={cfg.n_cls} n_families={n_families} max_sib={max_sib}")
    logger.info(f"[hier] family_of_class={fam_of_class}")
    logger.info(f"[hier] sib_idx={sib_idx}")
    # sib_mask for sibling head: (n_families, max_sib)
    sib_mask = torch.zeros(n_families, max_sib, device=cfg.device)
    for f, cls_list in fam_classes.items():
        for s_idx in range(len(cls_list)):
            sib_mask[f, s_idx] = 1.0
    family_of_class_t = torch.tensor(fam_of_class, device=cfg.device, dtype=torch.long)
    sib_idx_t = torch.tensor(sib_idx, device=cfg.device, dtype=torch.long)
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
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} lambda_s={cfg.lambda_s}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = CASCADEModel(cfg.enc, cfg.n_cls, n_families, max_sib).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, L_F, L_S = train_epoch(model, tr_dl, opt, sch, scaler, cfg, family_of_class_t, sib_idx_t, sib_mask)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu,
                            fam_of_class, sib_idx, sib_mask, family_of_class_t, sib_idx_t)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        fa = val_met["falsifier"]
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} L_F={L_F:.4f} L_S={L_S:.4f} val={v:.4f} "
                    f"fam_acc={fa['family_acc_F1']:.3f} sib_cond={fa['sibling_cond_acc_F2']:.3f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu,
                       fam_of_class, sib_idx, sib_mask, family_of_class_t, sib_idx_t)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    fa = ts_met["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"fam_acc={fa['family_acc_F1']:.3f} sib_cond={fa['sibling_cond_acc_F2']:.3f}")
    return {"tag": tag, "method": "CASCADE", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_s": cfg.lambda_s, "n_families": n_families, "max_sib": max_sib,
            "family_of_class": fam_of_class, "sib_idx": sib_idx,
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
                tag = f"exp78_cascade_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"fam={fa['family_acc_F1']:.3f} sib_cond={fa['sibling_cond_acc_F2']:.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp78_cascade_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'famAcc':>8} {'sibCond':>8} {'Wall':>8}")
    print("-"*140)
    for r in results:
        fa = r['test_metrics']['falsifier']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['family_acc_F1']:>8.3f} {fa['sibling_cond_acc_F2']:>8.3f} {r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
