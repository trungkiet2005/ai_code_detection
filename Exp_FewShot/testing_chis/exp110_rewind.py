# exp110 — REWIND
# NAME       : REWIND (Backwards distillation: 20% teacher → 1% student via attention)
# REFERENCE  : new; combines Hinton 2015 knowledge distillation with
#              attention transfer (Zagoruyko 2017) and few-shot setting.
# CLAIM      : The 1% slot is a sample-efficiency problem. The 20% slot
#              is already saturated. Train a TRACO teacher on 20% data,
#              then DISTILL to a student on 1% only — transferring not
#              just soft labels but PER-LAYER ATTENTION PATTERNS. The
#              student inherits "where to look" from the teacher's 20%
#              experience without needing the data.
# EQUATION   : Phase 1: teacher_θ = TRACO on 20% data (full)
#              Phase 2: student_φ on 1% data, init from teacher_θ
#                L = L_TRACO + α·KL(student_logits/T, teacher_logits/T) +
#                    β·sum_layers ||A_s^l - A_t^l||^2 + γ·(1 - cos(z_s, z_t))
#                A_l = sum_heads of |last_attention_layer_output|^2 (attention map)
# WHY NEW    : TRACOD (exp80) used EMA momentum and collapsed. REWIND uses
#              a STATIC teacher trained on MORE data with attention transfer
#              that no prior code-attribution method has tried.
# WOW HOOK   : "Train backwards. The teacher knows what 20% looks like.
#              The 1% student inherits not weights but VIEWPOINT —
#              specifically the teacher's attention pattern over tokens."
# FALSIFIER  : (F1) Student F1 @ 1% > TRACO @ 1% by ≥ 0.02. (F2) Attention
#              map distance ||A_s - A_t|| drops below 0.1 during training.
#              (F3) Removing attention transfer loses ≥ 0.01 at 1%.
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math, copy
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
logger = logging.getLogger("exp110_rewind")

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
# Model with attention extraction hook
# =============================================================================

class REWINDModel(nn.Module):
    """TRACO-style model that can return per-layer attention maps."""
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        self.emb_dim = emb_dim
        self.n_cls = n_cls

    def encode(self, ids, mask, return_attn=False):
        out = self.encoder(input_ids=ids, attention_mask=mask,
                           output_attentions=return_attn)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z_pre = self.proj(sem)
        z = F.normalize(z_pre, dim=-1)
        logits = self.clf(z)
        if return_attn:
            # attentions: tuple length=n_layers, each (B, n_heads, L, L)
            # Per-layer attention map: average over heads + average over query positions => (B, L)
            attn_maps = []
            for a in out.attentions:
                a_mean = a.mean(dim=1)          # (B, L, L) average over heads
                a_map = a_mean.mean(dim=1)      # (B, L) average over query positions
                attn_maps.append(a_map)
            return z, logits, attn_maps
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
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    # Distillation hyperparams
    alpha_kl: float = 1.0; beta_attn: float = 0.5; gamma_repr: float = 0.5
    dist_T: float = 4.0
    teacher_frac: float = 0.20
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
        # Halve again because: student forward + teacher forward + attention storage
        if mem >= 40: cfg.bs, cfg.seq = 96, 512
        elif mem >= 10: cfg.bs, cfg.seq = 48, 384
        else: cfg.bs, cfg.seq = 24, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} (teacher+student+attn) seq={cfg.seq}")
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
    """Simple dataset: single-view tokenization. Sub-samples by frac per-class."""
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42):
        self.data = data; self.tok = tok; self.seq_len = seq_len
        self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed)
            max_lbl = max(self.data["label"]) + 1
            keep = []
            for lbl in range(max_lbl):
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                if not idx: continue
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
        ids0, mask0 = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        return {"ids0": ids0, "mask0": mask0, "label": r["label"],
                "language": lang, "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        z0, logits = model.encode(ids0, mask0)
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


# =============================================================================
# Training
# =============================================================================

def train_teacher_epoch(model, loader, opt, sch, scaler, cfg, dist_mat):
    """Standard TRACO-style training (no distillation)."""
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="TeacherTrain"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, logits = model.encode(ids0, mask0)
            loss_ce = F.cross_entropy(logits, labs)
            # SupCon on single view (label-only positives)
            z_d = torch.cat([z0, z0], dim=0)
            y_d = torch.cat([labs, labs], dim=0)
            loss_sc = supcon_tw_loss(z_d, y_d, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            loss = loss_ce + cfg.lambda_aug * loss_sc
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item()
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n


def train_student_epoch(student, teacher, loader, opt, sch, scaler, cfg, dist_mat, attn_dist_tracker):
    """Train student with distillation losses against frozen teacher."""
    student.train(); teacher.eval()
    tot, ce_s, sc_s, kl_s, attn_s, repr_s = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    T = cfg.dist_T
    for b in tqdm(loader, desc="StudentTrain"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.no_grad():
            z_t, lg_t, attn_t = teacher.encode(ids0, mask0, return_attn=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z_s, lg_s, attn_s_list = student.encode(ids0, mask0, return_attn=True)
            loss_ce = F.cross_entropy(lg_s, labs)
            z_d = torch.cat([z_s, z_s], dim=0)
            y_d = torch.cat([labs, labs], dim=0)
            loss_sc = supcon_tw_loss(z_d, y_d, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            # KL distillation on logits
            loss_kl = F.kl_div(F.log_softmax(lg_s / T, dim=-1),
                               F.softmax(lg_t.float() / T, dim=-1),
                               reduction='batchmean') * (T * T)
            # Attention transfer: per-layer L2 on attention maps
            loss_attn = torch.zeros((), device=cfg.device, dtype=lg_s.dtype)
            n_layers = min(len(attn_s_list), len(attn_t))
            for li in range(n_layers):
                a_s = attn_s_list[li]
                a_t = attn_t[li]
                # Mask out PAD positions for fair comparison
                m = mask0.float()
                diff = (a_s - a_t.to(a_s.dtype)) * m
                loss_attn = loss_attn + diff.pow(2).sum(dim=-1).mean()
            loss_attn = loss_attn / max(n_layers, 1)
            # Representation alignment (cosine)
            loss_repr = 1.0 - (z_s * z_t.to(z_s.dtype)).sum(-1).mean()
            loss = (loss_ce + cfg.lambda_aug * loss_sc
                    + cfg.alpha_kl * loss_kl
                    + cfg.beta_attn * loss_attn
                    + cfg.gamma_repr * loss_repr)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item()
        kl_s += loss_kl.item(); attn_s += loss_attn.item(); repr_s += loss_repr.item()
        attn_dist_tracker.append(float(loss_attn.detach().item()))
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n, kl_s/n, attn_s/n, repr_s/n


def train_phase1_teacher(cfg, tag, tr_data, vl_data, ts_data, dist_mat, dist_mat_cpu, sib_mask_np, tok):
    """Phase 1: train teacher on cfg.teacher_frac (default 20%) data."""
    logger.info(f"[Phase 1] Training teacher on {cfg.teacher_frac*100:.0f}% data...")
    t_cfg = copy.deepcopy(cfg)
    t_cfg.frac = cfg.teacher_frac
    t_cfg = adaptive_schedule(t_cfg)
    tr_ds = FSDS(tr_data, tok, t_cfg.seq, frac=t_cfg.frac, seed=t_cfg.seed)
    vl_ds = FSDS(vl_data, tok, t_cfg.seq, frac=1.0, seed=t_cfg.seed+1)
    total_steps = max(1, len(tr_ds) // t_cfg.bs) * t_cfg.epochs
    loader_cfg = dict(batch_size=t_cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    teacher = REWINDModel(t_cfg.enc, t_cfg.n_cls, t_cfg.emb_dim).to(t_cfg.device)
    enc_ids = {id(p) for p in teacher.encoder.parameters()}
    head_params = [p for p in teacher.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(teacher.encoder.parameters()), "lr": t_cfg.lr_enc},
        {"params": head_params, "lr": t_cfg.lr_head}], weight_decay=t_cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * t_cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state = 0.0, None
    val_hist = []
    for epoch in range(t_cfg.epochs):
        loss, ce, sc = train_teacher_epoch(teacher, tr_dl, opt, sch, scaler, t_cfg, dist_mat)
        val_met = eval_pack(teacher, vl_dl, t_cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[teacher-ep {epoch+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: vv.cpu().clone() for k, vv in teacher.state_dict().items()}
    teacher.load_state_dict(best_state)
    # Evaluate teacher on test for reporting
    ts_ds = FSDS(ts_data, tok, t_cfg.seq, frac=1.0, seed=t_cfg.seed+2)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    teacher_test = eval_pack(teacher, ts_dl, t_cfg, sib_mask_np, dist_mat_cpu)
    teacher_macro = teacher_test["overall"]["macro_f1"]
    logger.info(f"[Phase 1 done] teacher_val={best_val:.4f} teacher_test={teacher_macro:.4f}")
    return teacher, best_val, teacher_macro, val_hist, teacher_test


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg)
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

    # Phase 1: train teacher (always on cfg.teacher_frac = 20% by default)
    teacher, teacher_val, teacher_macro_at_20pct, teacher_val_hist, teacher_test_metrics = \
        train_phase1_teacher(cfg, tag, tr_data, vl_data, ts_data, dist_mat, dist_mat_cpu, sib_mask_np, tok)
    teacher.eval()
    for p in teacher.parameters(): p.requires_grad = False

    # If cfg.frac == teacher_frac, the teacher IS the student — report teacher directly.
    if abs(cfg.frac - cfg.teacher_frac) < 1e-6:
        logger.info(f"[short-circuit] student_frac == teacher_frac ({cfg.frac}); "
                    f"returning teacher metrics directly.")
        return {"tag": tag, "method": "REWIND", "note": "student=teacher (same frac)",
                "enc": cfg.enc, "bench": cfg.benchmark,
                "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
                "alpha_kl": cfg.alpha_kl, "beta_attn": cfg.beta_attn, "gamma_repr": cfg.gamma_repr,
                "dist_T": cfg.dist_T, "teacher_frac": cfg.teacher_frac,
                "val_macro": teacher_val,
                "macro": teacher_macro_at_20pct,
                "weighted": teacher_test_metrics["overall"]["weighted_f1"],
                "acc": teacher_test_metrics["overall"]["accuracy"],
                "val_test_gap": teacher_val - teacher_macro_at_20pct,
                "dpaper": teacher_macro_at_20pct - PAPER_BASELINE,
                "test_metrics": teacher_test_metrics,
                "val_history": teacher_val_hist,
                "teacher_macro_at_20pct": teacher_macro_at_20pct,
                "final_attention_distance": 0.0,
                "student_vs_teacher_at_same_frac": 0.0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    # Phase 2: train student on cfg.frac with distillation
    cfg = adaptive_schedule(cfg)
    logger.info(f"[Phase 2] Training student on {cfg.frac*100:.0f}% data with distillation...")
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched-student] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} "
                f"warmup={cfg.warmup} alpha_kl={cfg.alpha_kl} beta_attn={cfg.beta_attn} "
                f"gamma_repr={cfg.gamma_repr}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    # Initialize student from teacher weights (this is the "REWIND" inheritance)
    student = REWINDModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    student.load_state_dict({k: v.to(cfg.device) for k, v in teacher.state_dict().items()})

    enc_ids = {id(p) for p in student.encoder.parameters()}
    head_params = [p for p in student.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(student.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    attn_dist_history = []
    for epoch in range(cfg.epochs):
        loss, ce, sc, kl, at, rp = train_student_epoch(
            student, teacher, tr_dl, opt, sch, scaler, cfg, dist_mat, attn_dist_history)
        val_met = eval_pack(student, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[student-ep {epoch+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} "
                    f"kl={kl:.4f} attn={at:.4f} repr={rp:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: vv.cpu().clone() for k, vv in student.state_dict().items()}
    student.load_state_dict(best_state)
    ts_met = eval_pack(student, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    final_attn_dist = float(np.mean(attn_dist_history[-max(1, len(attn_dist_history)//10):])) \
                      if attn_dist_history else 0.0
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"teacher@20%={teacher_macro_at_20pct:.4f} final_attn_dist={final_attn_dist:.4f}")
    return {"tag": tag, "method": "REWIND",
            "note": f"teacher@{cfg.teacher_frac:.0%} -> student@{cfg.frac:.0%} via attn+kl+repr distill",
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "alpha_kl": cfg.alpha_kl, "beta_attn": cfg.beta_attn, "gamma_repr": cfg.gamma_repr,
            "dist_T": cfg.dist_T, "teacher_frac": cfg.teacher_frac,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "teacher_macro_at_20pct": teacher_macro_at_20pct,
            "teacher_val_history": teacher_val_hist,
            "final_attention_distance": final_attn_dist,
            "student_vs_teacher_at_same_frac": test_macro - teacher_macro_at_20pct,
            "attn_dist_history_tail": attn_dist_history[-50:] if attn_dist_history else [],
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
                tag = f"exp110_rewind_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"teacher@20%={res['teacher_macro_at_20pct']:.4f} "
                                f"attn_dist={res['final_attention_distance']:.4f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp110_rewind_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Encoder':<20} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'Tch@20%':>9} {'AttnDist':>10} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['enc']:<20} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['teacher_macro_at_20pct']:>9.4f} "
              f"{r['final_attention_distance']:>10.4f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
