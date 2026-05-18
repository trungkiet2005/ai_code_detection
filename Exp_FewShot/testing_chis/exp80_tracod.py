# exp80_tracod — TRACO with EMA Teacher Self-Distillation (TRACOD)
# =============================================================================
# Theory-Track exp -- TRACOD (TRACO + EMA Teacher Distillation)
#
# ROLE           : TRACO (exp76) enforces S7 invariance via in-batch contrastive
#                  on doubled views.  In-batch contrastive is limited to the
#                  current batch's diversity; an EMA teacher gives a GLOBAL,
#                  temporally-smoothed target distribution.  TRACOD adds DINO-
#                  style self-distillation: a slow-moving teacher network
#                  produces soft class targets that the student matches across
#                  views.  Combined with TRACO's contrastive on the embedding,
#                  TRACOD has invariance signals at BOTH the embedding level
#                  (TRACO) and the class distribution level (distillation).
# NAME           : TRACOD  (TRACO + Distillation)
# ARXIV_ID       : extends DINO (Caron et al., arXiv:2104.14294) which uses
#                  EMA teacher self-distillation in self-supervised vision;
#                  TRACOD adapts the paradigm to supervised AI-code attribution
#                  with TRACO's S7-grounded code augmentations and a
#                  classifier (not projection head).
# ONE-LINE CLAIM : Embedding-level (TRACO) and class-distribution-level (DINO)
#                  invariance signals are complementary; both should be
#                  enforced under semantic-preserving code augmentations.
# EQUATION       : x' = T(x)  (TRACO augmentation)
#                  z_s, p_s   = student(x)
#                  z_s', p_s' = student(x')
#                  with torch.no_grad():
#                    p_t  = softmax((teacher(x).logits - C) / tau_t)
#                    p_t' = softmax((teacher(x').logits - C) / tau_t)
#                  L_distill = CE(p_t', log_softmax(p_s))
#                              + CE(p_t,  log_softmax(p_s'))
#                  L_contrast = SupCon-TW([z_s; z_s'], [y; y])
#                  L_total = L_CE_student + lambda_aug * L_contrast
#                            + lambda_distill * L_distill
#                  theta_teacher = m * theta_teacher + (1-m) * theta_student
#                  C = running mean of teacher logits (centering, prevents collapse)
# WHY NOT BEFORE : DINO requires NO labels (uses self-supervision via centering).
#                  TRACOD uses LABELS (CE on student) PLUS DINO-style targets
#                  for cross-view consistency.  No prior work combines DINO
#                  self-distillation with code-augmentation-contrastive for
#                  attribution.
# FALSIFIER      : (F1) Teacher-student agreement: fraction of test samples
#                       where argmax(student) == argmax(teacher).  > 0.95 =
#                       convergence achieved.
#                  (F2) Mean KL(teacher || student) on test.  Small = good.
#                  (F3) View-cos like TRACO: encoder invariance.
# REPORTS        : Full eval pack + (F1) ts-agreement, (F2) mean-KL, (F3) view-cos.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math, copy
from dataclasses import dataclass, field
from typing import List, Dict, Tuple
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
logger = logging.getLogger("exp80_tracod")

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
# Augmentation (reuse TRACO's transforms)
# =============================================================================

_RESERVED = {"if","else","elif","for","while","do","return","def","function","func",
             "fn","class","struct","interface","enum","import","from","include",
             "require","using","public","private","protected","static","final",
             "void","new","this","extends","implements","throws","try","catch",
             "except","finally","raise","with","in","of","is","not","and","or",
             "as","True","False","None","null","true","false","self",
             "int","float","double","char","long","short","bool","string","str",
             "var","let","const"}


def aug_token_dropout(code, rng, p=0.1):
    tokens = _re.split(r"(\s+|[^\w\s])", code)
    out = []
    for t in tokens:
        if t.strip() and t.strip() not in _RESERVED and not t.isspace():
            if rng.random() < p: out.append(" "); continue
        out.append(t)
    return "".join(out)


def aug_id_rename(code, rng, max_renames=8):
    ids = set(_re.findall(r"\b[a-zA-Z_]\w{2,}\b", code))
    ids = [i for i in ids if i not in _RESERVED and not i[0].isdigit()]
    if not ids: return code
    n = min(max_renames, len(ids))
    chosen = rng.sample(ids, n)
    new = code
    for k, orig in enumerate(chosen):
        new = _re.sub(rf"\b{_re.escape(orig)}\b", f"v{k}", new)
    return new


def aug_ws_jitter(code, rng, p=0.15):
    ops = ["+", "-", "*", "/", "%", "=", "<", ">", ",", ";"]
    out_chars = []
    for c in code:
        out_chars.append(c)
        if c in ops and rng.random() < p: out_chars.append(" ")
    return "".join(out_chars)


def aug_comment_strip(code, rng):
    code = _re.sub(r"/\*[\s\S]*?\*/", "", code)
    code = _re.sub(r"//[^\n]*", "", code)
    code = _re.sub(r"#[^\n]*", "", code)
    return code


_AUG_TABLE = [("token_dropout", aug_token_dropout),
              ("id_rename", aug_id_rename),
              ("ws_jitter", aug_ws_jitter),
              ("comment_strip", aug_comment_strip)]


def augment(code, rng):
    name, fn = _AUG_TABLE[rng.randrange(len(_AUG_TABLE))]
    try: return fn(code, rng), name
    except Exception: return code, "noop"


# =============================================================================
# Model
# =============================================================================

class TRACODNet(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1), self.clf(z)


def supcon_tw_loss(z_doubled, labels_doubled, dist_mat, gamma=1.0, tau=0.1):
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
    has_pos = (pos_mask.sum(dim=-1) > 0).float()
    li = -(torch.log(num) - torch.log(den)) * has_pos
    return li.sum() / has_pos.sum().clamp(min=1.0)


@torch.no_grad()
def ema_update(student, teacher, m):
    for sp, tp in zip(student.parameters(), teacher.parameters()):
        tp.data.mul_(m).add_(sp.data, alpha=1.0 - m)


def distill_loss(student_logits, teacher_logits, center, tau_s=0.1, tau_t=0.04):
    """DINO-style distillation: teacher's softmax (sharpened+centered) targets student."""
    with torch.no_grad():
        t_logits = (teacher_logits - center) / tau_t
        p_t = F.softmax(t_logits, dim=-1)
    log_p_s = F.log_softmax(student_logits / tau_s, dim=-1)
    return -(p_t * log_p_s).sum(dim=-1).mean()


# =============================================================================
# Plumbing
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 128; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; lambda_distill: float = 0.3
    gamma: float = 1.0; tau: float = 0.1
    momentum: float = 0.996; center_m: float = 0.9
    tau_s: float = 0.1; tau_t: float = 0.04
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
        # bs/2 due to teacher + 2-view = effectively 3x forward
        if mem >= 40: cfg.bs, cfg.seq = 96, 512
        elif mem >= 10: cfg.bs, cfg.seq = 48, 384
        else: cfg.bs, cfg.seq = 24, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} (HALVED+ for teacher+2views) seq={cfg.seq}")
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


class FSDS_AUG(TD):
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
            logger.info(f"[FSDS_AUG] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        enc = self.tok(code, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
        ids0, mask0 = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            code_aug, _ = augment(code, rng)
            enc2 = self.tok(code_aug, max_length=self.seq_len, padding="max_length", truncation=True, return_tensors="pt")
            ids1, mask1 = enc2["input_ids"].squeeze(0), enc2["attention_mask"].squeeze(0)
        else:
            ids1, mask1 = ids0, mask0
        return {"ids0": ids0, "mask0": mask0, "ids1": ids1, "mask1": mask1,
                "label": r["label"], "language": r.get("language", "") or "",
                "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(student, teacher, loader, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=False):
    student.eval()
    if teacher is not None: teacher.eval()
    preds, labels, langs, sources = [], [], [], []
    cos_list, kl_list, agree_list = [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        z0, logits_s = student.encode(ids0, mask0)
        preds.extend(logits_s.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            ids1 = b["ids1"].to(cfg.device); mask1 = b["mask1"].to(cfg.device)
            z1, _ = student.encode(ids1, mask1)
            sim = (z0 * z1).sum(dim=-1).cpu().float().numpy().tolist()
            cos_list.extend(sim)
            if teacher is not None:
                _, logits_t = teacher.encode(ids0, mask0)
                p_s = F.softmax(logits_s, dim=-1)
                p_t = F.softmax(logits_t, dim=-1)
                kl = (p_t * (p_t.clamp(min=1e-12).log() - p_s.clamp(min=1e-12).log())).sum(dim=-1)
                kl_list.extend(kl.cpu().float().numpy().tolist())
                agree = (logits_s.argmax(dim=-1) == logits_t.argmax(dim=-1)).float()
                agree_list.extend(agree.cpu().numpy().tolist())
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
    if collect_falsifier:
        out["falsifier"] = {
            "view_cos_F3": float(np.mean(cos_list)) if cos_list else 0.0,
            "mean_KL_teacher_student_F2": float(np.mean(kl_list)) if kl_list else 0.0,
            "ts_agreement_F1": float(np.mean(agree_list)) if agree_list else 0.0,
        }
    return out


def train_epoch(student, teacher, loader, opt, sch, scaler, cfg, dist_mat, center):
    student.train(); teacher.eval()
    tot, ce_s, sc_s, dl_s = 0.0, 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); mask1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, logits_s0 = student.encode(ids0, mask0)
            z1, logits_s1 = student.encode(ids1, mask1)
            with torch.no_grad():
                _, logits_t0 = teacher.encode(ids0, mask0)
                _, logits_t1 = teacher.encode(ids1, mask1)
                # update center as EMA of teacher logits batch-mean
                batch_center = 0.5 * (logits_t0.mean(0) + logits_t1.mean(0)).detach()
                center.mul_(cfg.center_m).add_(batch_center, alpha=1.0 - cfg.center_m)
            z_d = torch.cat([z0, z1], dim=0)
            y_d = torch.cat([labs, labs], dim=0)
            loss_ce = F.cross_entropy(logits_s0, labs)
            loss_sc = supcon_tw_loss(z_d, y_d, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            # cross-view distillation: teacher view 0 -> student view 1 and vice versa
            loss_d_a = distill_loss(logits_s1, logits_t0.detach(), center.detach(), cfg.tau_s, cfg.tau_t)
            loss_d_b = distill_loss(logits_s0, logits_t1.detach(), center.detach(), cfg.tau_s, cfg.tau_t)
            loss_distill = 0.5 * (loss_d_a + loss_d_b)
            loss = loss_ce + cfg.lambda_aug * loss_sc + cfg.lambda_distill * loss_distill
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        ema_update(student, teacher, cfg.momentum)
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item(); dl_s += loss_distill.item()
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n, dl_s/n


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
    tr_ds = FSDS_AUG(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_AUG(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=True)
    ts_ds = FSDS_AUG(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=True)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"lambda_aug={cfg.lambda_aug} lambda_distill={cfg.lambda_distill} m={cfg.momentum}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    student = TRACODNet(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    teacher = copy.deepcopy(student).to(cfg.device)
    for p in teacher.parameters(): p.requires_grad = False
    center = torch.zeros(cfg.n_cls, device=cfg.device)
    enc_ids = {id(p) for p in student.encoder.parameters()}
    head_params = [p for p in student.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(student.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, ce, sc, dl = train_epoch(student, teacher, tr_dl, opt, sch, scaler, cfg, dist_mat, center)
        val_met = eval_pack(student, teacher, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} distill={dl:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in student.state_dict().items()}
    student.load_state_dict(best_state)
    ts_met = eval_pack(student, teacher, ts_dl, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=True)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    fa = ts_met["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"view_cos={fa['view_cos_F3']:.3f} ts_agree={fa['ts_agreement_F1']:.3f} "
                f"meanKL={fa['mean_KL_teacher_student_F2']:.4f}")
    return {"tag": tag, "method": "TRACOD", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "lambda_distill": cfg.lambda_distill,
            "momentum": cfg.momentum, "tau_s": cfg.tau_s, "tau_t": cfg.tau_t,
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
                tag = f"exp80_tracod_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"view_cos={fa['view_cos_F3']:.3f} ts_agree={fa['ts_agreement_F1']:.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp80_tracod_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'viewCos':>8} {'TS_agr':>8} {'Wall':>8}")
    print("-"*140)
    for r in results:
        fa = r['test_metrics']['falsifier']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['view_cos_F3']:>8.3f} {fa['ts_agreement_F1']:>8.3f} {r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
