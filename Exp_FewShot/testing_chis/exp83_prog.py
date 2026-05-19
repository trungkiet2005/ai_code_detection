# exp83_prog — Progressive Family→Sibling Curriculum Fine-tune (PROG)
# =============================================================================
# Theory-Track exp -- PROG (Progressive Family→Sibling Curriculum)
#
# ROLE           : CASCADE (exp78) showed family-acc is the bottleneck.
#                  PROG implements a 3-phase training curriculum that first
#                  forces the encoder to optimise family discrimination, then
#                  expands to full-class discrimination, then adds TRACO
#                  augmentation views.  Unlike CASCADE which factorises the
#                  DECODING (probabilistic product), PROG factorises the
#                  TRAINING SCHEDULE.  The encoder's first gradient signals
#                  are family-only -- so the representation is anchored to
#                  the harder (per CASCADE) discrimination first.
# NAME           : PROG  (Progressive Family→Sibling Curriculum)
# ARXIV_ID       : extends curriculum-learning (Bengio 2009) by specifically
#                  using a LABEL-TREE-derived curriculum schedule (coarse
#                  family first, fine sibling later).  Novel for code
#                  attribution.
# ONE-LINE CLAIM : Aligning the training-schedule curriculum to the empirical
#                  bottleneck (family-acc per CASCADE) lifts attribution
#                  more than any post-hoc loss term.
# EQUATION       : Phase 1 (epochs 0 .. e1):
#                    head: family-classifier only
#                    L_1 = CE_family(W_F z, family(y))
#                  Phase 2 (epochs e1 .. e2):
#                    head: class-classifier (initialised from family-CE pretrain)
#                    L_2 = CE_class(W z, y)
#                  Phase 3 (epochs e2 .. N):
#                    L_3 = CE_class + lambda_aug * SupCon([z; z'], [y; y])
# WHY NOT BEFORE : Curriculum learning has been applied to text/vision but the
#                  curriculum is usually difficulty-based.  PROG defines the
#                  curriculum by LABEL-TREE GRANULARITY, anchored in CASCADE's
#                  empirical family-bottleneck finding.
# FALSIFIER      : (F1) Phase-end macro-F1 progression: Phase 1 (family-only)
#                       → Phase 2 (class) → Phase 3 (class + aug).  Monotone
#                       increase = curriculum genuinely helps.
#                  (F2) Compare PROG-final test F1 vs joint-only training
#                       (TRACO).  +0.005 = curriculum-grounded gain.
#                  (F3) Family-acc at end of Phase 1: pretrained encoder
#                       should produce high family-acc before class fine-tune.
# REPORTS        : Full eval pack + (F1)(F2)(F3) + phase-end val/test metrics.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from dataclasses import dataclass, field
from typing import List, Dict
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
logger = logging.getLogger("exp83_prog")

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
    parent = list(range(n_cls))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        px, py = find(x), find(y)
        if px != py: parent[px] = py
    for i in range(n_cls):
        for j in adj.get(i, []): union(i, j)
    roots = sorted({find(i) for i in range(n_cls)})
    root_to_fam = {r: idx for idx, r in enumerate(roots)}
    family_of_class = [root_to_fam[find(i)] for i in range(n_cls)]
    return family_of_class, len(roots)


# =============================================================================
# Augmentations
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
    tokens = _re.split(r"(\s+|[^\w\s])", code); out = []
    for t in tokens:
        if t.strip() and t.strip() not in _RESERVED and not t.isspace():
            if rng.random() < p: out.append(" "); continue
        out.append(t)
    return "".join(out)


def aug_id_rename(code, rng, max_renames=8):
    ids = set(_re.findall(r"\b[a-zA-Z_]\w{2,}\b", code))
    ids = [i for i in ids if i not in _RESERVED and not i[0].isdigit()]
    if not ids: return code
    chosen = rng.sample(ids, min(max_renames, len(ids)))
    new = code
    for k, orig in enumerate(chosen):
        new = _re.sub(rf"\b{_re.escape(orig)}\b", f"v{k}", new)
    return new


def aug_ws_jitter(code, rng, p=0.15):
    ops = ["+","-","*","/","%","=","<",">",",",";"]
    out = []
    for c in code:
        out.append(c)
        if c in ops and rng.random() < p: out.append(" ")
    return "".join(out)


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

class PROGModel(nn.Module):
    def __init__(self, enc_name, n_cls, n_families, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, emb_dim))
        self.head_F = nn.Linear(emb_dim, n_families)
        self.head_C = nn.Linear(emb_dim, n_cls)
        self.n_cls = n_cls; self.n_families = n_families; self.emb_dim = emb_dim

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.proj(sem)


def supcon_view_loss(z_doubled, labels_doubled, tau=0.1):
    N = z_doubled.size(0)
    if N < 2: return z_doubled.sum() * 0.0
    z_n = F.normalize(z_doubled, dim=-1)
    sim = (z_n @ z_n.t()) / tau
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()
    eye = torch.eye(N, device=z_doubled.device, dtype=torch.bool)
    pos_mask = (labels_doubled.unsqueeze(0) == labels_doubled.unsqueeze(1)).float().masked_fill(eye, 0.0)
    exp_s = torch.exp(sim).masked_fill(eye, 0.0)
    den = exp_s.sum(dim=-1).clamp(min=1e-12)
    num = (exp_s * pos_mask).sum(dim=-1).clamp(min=1e-12)
    has_pos = (pos_mask.sum(dim=-1) > 0).float()
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
    bs: int = 128; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; tau: float = 0.1
    phase1_frac: float = 0.33; phase2_frac: float = 0.66        # epoch fractions for phases
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
        if mem >= 40: cfg.bs, cfg.seq = 128, 512
        elif mem >= 10: cfg.bs, cfg.seq = 64, 384
        else: cfg.bs, cfg.seq = 32, 256
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
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, family_of_class_t=None):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    fam_preds = []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        z = model.encode(ids0, mask0)
        logits = model.head_C(z)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if family_of_class_t is not None:
            log_F = model.head_F(z)
            fam_preds.extend(log_F.argmax(dim=-1).cpu().tolist())
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
    if family_of_class_t is not None and fam_preds:
        fam_preds = np.array(fam_preds)
        true_fam = family_of_class_t.cpu().numpy()[labels]
        out["family_acc"] = float(np.mean(fam_preds == true_fam))
    return out


def train_step_phase1(model, b, cfg, family_of_class_t, scaler, opt, sch):
    ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device)
    labs = b["label"].to(cfg.device)
    fam_target = family_of_class_t[labs]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
        z = model.encode(ids, mask)
        log_F = model.head_F(z)
        loss = F.cross_entropy(log_F, fam_target)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
    return loss.item()


def train_step_phase2(model, b, cfg, scaler, opt, sch):
    ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device)
    labs = b["label"].to(cfg.device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
        z = model.encode(ids, mask)
        log_C = model.head_C(z)
        loss = F.cross_entropy(log_C, labs)
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
    return loss.item()


def train_step_phase3(model, b, cfg, scaler, opt, sch):
    ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
    ids1 = b["ids1"].to(cfg.device); mask1 = b["mask1"].to(cfg.device)
    labs = b["label"].to(cfg.device)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
        z0 = model.encode(ids0, mask0)
        z1 = model.encode(ids1, mask1)
        log_C = model.head_C(z0)
        z_d = torch.cat([z0, z1], dim=0)
        y_d = torch.cat([labs, labs], dim=0)
        loss_ce = F.cross_entropy(log_C, labs)
        loss_sc = supcon_view_loss(z_d, y_d, tau=cfg.tau)
        loss = loss_ce + cfg.lambda_aug * loss_sc
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
    return loss.item()


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat_t = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist_mat_cpu = dist_mat_t.numpy()
    sib_mask_np = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()
    fam_of_class, n_families = compute_family_mapping(cfg.n_cls, cfg.gene_adj)
    family_of_class_t = torch.tensor(fam_of_class, device=cfg.device, dtype=torch.long)
    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, "author", vocab); vl_data = _conv_codet(vl_raw, "author", vocab); ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    tr_ds = FSDS_AUG(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_AUG(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS_AUG(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    # Phase boundaries
    e1 = max(1, int(cfg.epochs * cfg.phase1_frac))
    e2 = max(e1 + 1, int(cfg.epochs * cfg.phase2_frac))
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} (P1=[0..{e1}), P2=[{e1}..{e2}), P3=[{e2}..{cfg.epochs}))")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = PROGModel(cfg.enc, cfg.n_cls, n_families, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist, phase_marks = 0.0, None, [], []
    for epoch in range(cfg.epochs):
        if epoch < e1:
            phase = 1; phase_name = "P1_family"
            losses = [train_step_phase1(model, b, cfg, family_of_class_t, scaler, opt, sch) for b in tqdm(tr_dl, desc=phase_name)]
        elif epoch < e2:
            phase = 2; phase_name = "P2_class"
            losses = [train_step_phase2(model, b, cfg, scaler, opt, sch) for b in tqdm(tr_dl, desc=phase_name)]
        else:
            phase = 3; phase_name = "P3_traco"
            losses = [train_step_phase3(model, b, cfg, scaler, opt, sch) for b in tqdm(tr_dl, desc=phase_name)]
        mean_loss = float(np.mean(losses)) if losses else 0.0
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu, family_of_class_t)
        v = val_met["overall"]["macro_f1"]
        fam_acc_v = val_met.get("family_acc", 0.0)
        val_hist.append(v); phase_marks.append(phase)
        logger.info(f"[epoch {epoch+1}/{cfg.epochs} {phase_name}] loss={mean_loss:.4f} val_classF1={v:.4f} val_famAcc={fam_acc_v:.3f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, family_of_class_t)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    ts_met["falsifier"] = {
        "phase_marks_F1": phase_marks,
        "val_hist_F1": val_hist,
        "family_acc_F3": ts_met.get("family_acc", 0.0),
    }
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} fam_acc={ts_met.get('family_acc', 0):.3f}")
    return {"tag": tag, "method": "PROG", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "phase1_frac": cfg.phase1_frac, "phase2_frac": cfg.phase2_frac,
            "n_families": n_families,
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
                tag = f"exp83_prog_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"fam_acc={fa['family_acc_F3']:.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp83_prog_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*130)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} "
          f"{'dPaper':>9} {'famAcc':>8} {'Wall':>8}")
    print("-"*130)
    for r in results:
        fa = r['test_metrics']['falsifier']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['family_acc_F3']:>8.3f} {r['wall']:>8.0f}s")
    print("="*130)


if __name__ == "__main__":
    main()
