# exp105 — ACTIVE
# NAME       : ACTIVE (BALD/Entropy Active-Learning Loop for Few-Shot Code Attribution)
# REFERENCE  : new (BALD arXiv:1112.5745, MC-Dropout arXiv:1506.02142,
#              Active learning survey arXiv:2009.00236)
# CLAIM      : The 1% regime is not a sample-efficiency problem — it is a SAMPLE-SELECTION
#              problem. Random 1% leaves the model unsure about most of the unlabeled pool.
#              An entropy-based active loop that asks an oracle to label the highest-
#              uncertainty samples first will beat random sampling at matched budget.
# EQUATION   : Round r:
#                model_r = train(L_r U U_r_acquired)
#                u(x) = H[p(y|x; model_r)]  for x in pool
#                or u(x) = BALD via MC-dropout: mutual_info between predictions and weights
#                acquire top-K from pool by u; add to L
#              Final composite over (initial-frac to final-frac) acquisition curve.
# WHY NEW    : No prior AI code-attribution paper uses active learning. All compete on
#              equally-sized random subsets. We compare random N% vs active M% with M < N
#              and show M can match or beat N.
# WOW HOOK   : "The right 1% beats random 5%. Code attribution at extreme few-shot is not
#              a sample-efficiency problem — it's a sample-selection problem."
# FALSIFIER  : (F1) If random N% >= active N% at matched final size, active learning
#              has no marginal value -> falsified. (F2) If the BALD/entropy acquisition
#              correlates < 0.1 with eventual loss reduction, the uncertainty signal is
#              not informative.
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
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
from torch.utils.data import Dataset as TD, DataLoader, Subset
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp105_active")

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
# Data plumbing
# =============================================================================

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    emb_dim: int = 256
    mode: str = "active"           # "active" or "random"
    inner_epochs: int = 2           # epochs in inner acquisition rounds
    n_rounds: int = 3               # number of acquisition rounds (active only)
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


# =============================================================================
# Tokenisation — faithful UniXcoder protocol
# =============================================================================

def _tokenize(code, tokenizer, max_len):
    toks = tokenizer.tokenize(" ".join(code.split()))[:max_len - 4]
    toks = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + toks + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_len - len(ids))
    return ids[:max_len]


# =============================================================================
# Dataset
# =============================================================================

class FSDS_Full(TD):
    """Tokenizes full dataset (no subsampling). Subsampling done via Subset
    with explicit indices for active-learning bookkeeping."""
    def __init__(self, data, tok, seq_len):
        self.data = data; self.tok = tok; self.seq_len = seq_len
        self.pad_id = tok.pad_token_id

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        ids = _tokenize(code, self.tok, self.seq_len)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "label": r["label"],
                "language": r.get("language", "") or "",
                "source": r.get("source", "") or ""}


# =============================================================================
# Model — TRACO-style: encoder + projector + linear head
# =============================================================================

class TRACOStyleModel(nn.Module):
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
        return F.normalize(z, dim=-1), self.clf(z)


# =============================================================================
# Stratified sampling helper
# =============================================================================

def stratified_sample_indices(labels: List[int], frac: float, seed: int) -> List[int]:
    rng = random.Random(seed)
    by_lbl: Dict[int, List[int]] = {}
    for i, l in enumerate(labels):
        by_lbl.setdefault(int(l), []).append(i)
    keep = []
    for l, idx_list in by_lbl.items():
        k = min(max(1, int(len(idx_list) * frac)), len(idx_list))
        keep.extend(rng.sample(idx_list, k))
    return keep


# =============================================================================
# Acquisition: top-K entropy with per-class balancing
# =============================================================================

@torch.no_grad()
def entropy_acquire(model, full_ds: FSDS_Full, pool_indices: List[int], k: int,
                    device: str, pad_id: int, n_cls: int, bs: int = 128) -> List[int]:
    """Return up to k pool indices with highest entropy under the model.
    Balance acquisitions across predicted classes to avoid class collapse."""
    model.eval()
    sub = Subset(full_ds, pool_indices)
    loader = DataLoader(sub, batch_size=bs, shuffle=False, num_workers=2)
    ents: List[float] = []
    preds: List[int] = []
    for b in tqdm(loader, desc="Acquire"):
        ids = b["input_ids"].to(device)
        mask = ids.ne(pad_id).long()
        out = model.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = model.proj(sem)
        logits = model.clf(z)
        p = F.softmax(logits, dim=-1)
        ent = -(p * (p + 1e-12).log()).sum(dim=-1)
        ents.extend(ent.cpu().float().tolist())
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
    # Balance by predicted class: take top-(k/n_cls) per class
    per_cls = max(1, k // n_cls)
    chosen: List[int] = []
    for c in range(n_cls):
        cand = [(ents[i], pool_indices[i]) for i in range(len(pool_indices)) if preds[i] == c]
        cand.sort(key=lambda x: -x[0])
        chosen.extend([idx for _, idx in cand[:per_cls]])
    # Fill rest with highest-entropy regardless of class if short
    if len(chosen) < k:
        remaining = [(ents[i], pool_indices[i]) for i in range(len(pool_indices))
                     if pool_indices[i] not in set(chosen)]
        remaining.sort(key=lambda x: -x[0])
        chosen.extend([idx for _, idx in remaining[:k - len(chosen)]])
    return chosen[:k]


# =============================================================================
# Train / eval helpers
# =============================================================================

def train_loop(model, indices: List[int], full_ds: FSDS_Full, cfg: Cfg, n_epochs: int,
               lr_enc: float, warmup: float, pad_id: int) -> nn.Module:
    """Train CE-only on the supplied indices."""
    sub = Subset(full_ds, indices)
    loader = DataLoader(sub, batch_size=cfg.bs, shuffle=True,
                        num_workers=4, pin_memory=True)
    total_steps = max(1, len(sub) // cfg.bs) * max(1, n_epochs)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * warmup)), total_steps)
    scaler = GradScaler()
    model.train()
    for ep in range(n_epochs):
        for b in tqdm(loader, desc=f"InnerTrain ep{ep+1}"):
            ids = b["input_ids"].to(cfg.device)
            mask = ids.ne(pad_id).long()
            labs = b["label"].to(cfg.device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
                z, logits = model.encode(ids, mask)
                loss = F.cross_entropy(logits, labs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
    return model


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, pad_id):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["input_ids"].to(cfg.device); mask = ids.ne(pad_id).long(); labs = b["label"]
        z, logits = model.encode(ids, mask)
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
# Main run_exp
# =============================================================================

def run_exp(cfg: Cfg, tag: str) -> dict:
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat_t = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist_mat_cpu = dist_mat_t.numpy()
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
    pad_id = tok.pad_token_id
    full_tr_ds = FSDS_Full(tr_data, tok, cfg.seq)
    vl_ds = FSDS_Full(vl_data, tok, cfg.seq)
    ts_ds = FSDS_Full(ts_data, tok, cfg.seq)
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    all_labels = list(tr_data["label"])

    logger.info(f"[sched] mode={cfg.mode} frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup}")

    acquisition_log: List[Dict] = []
    if cfg.mode == "active":
        # Seed = frac / (n_rounds + 1) per class; acquire same amount per round.
        seed_frac = cfg.frac / (cfg.n_rounds + 1)
        seed_indices = stratified_sample_indices(all_labels, seed_frac, seed=cfg.seed)
        pool_set = set(range(len(all_labels))) - set(seed_indices)
        pool_indices = sorted(pool_set)
        logger.info(f"[active] seed_frac={seed_frac:.4f} seed_n={len(seed_indices)} pool_n={len(pool_indices)}")
        cur_indices = list(seed_indices)
        # Per-round acquisition budget (k samples per round)
        per_round_n = int(seed_frac * len(all_labels))
        for round_i in range(cfg.n_rounds):
            model = TRACOStyleModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
            model = train_loop(model, cur_indices, full_tr_ds, cfg,
                               n_epochs=cfg.inner_epochs, lr_enc=cfg.lr_enc, warmup=cfg.warmup,
                               pad_id=pad_id)
            k = per_round_n
            new_idx = entropy_acquire(model, full_tr_ds, pool_indices, k, cfg.device, pad_id, cfg.n_cls,
                                      bs=cfg.bs)
            acquisition_log.append({"round": round_i, "n_acquired": len(new_idx),
                                    "cur_total": len(cur_indices) + len(new_idx)})
            logger.info(f"[active round {round_i+1}] acquired {len(new_idx)} -> total {len(cur_indices)+len(new_idx)}")
            new_set = set(new_idx)
            cur_indices = cur_indices + list(new_idx)
            pool_indices = [i for i in pool_indices if i not in new_set]
            del model
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
        final_indices = cur_indices
    else:
        # Random: just sample frac up-front
        final_indices = stratified_sample_indices(all_labels, cfg.frac, seed=cfg.seed)
        logger.info(f"[random] sampled {len(final_indices)} ({cfg.frac*100:.1f}%)")

    # Final training pass
    final_model = TRACOStyleModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    sub = Subset(full_tr_ds, final_indices)
    loader = DataLoader(sub, batch_size=cfg.bs, shuffle=True,
                        num_workers=4, pin_memory=True)
    total_steps = max(1, len(sub) // cfg.bs) * cfg.epochs
    enc_ids = {id(p) for p in final_model.encoder.parameters()}
    head_params = [p for p in final_model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(final_model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        final_model.train()
        ce_sum = 0.0
        for b in tqdm(loader, desc=f"FinalTrain ep{epoch+1}"):
            ids = b["input_ids"].to(cfg.device); mask = ids.ne(pad_id).long()
            labs = b["label"].to(cfg.device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
                z, logits = final_model.encode(ids, mask)
                loss = F.cross_entropy(logits, labs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(final_model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
            ce_sum += loss.item()
        val_met = eval_pack(final_model, vl_dl, cfg, sib_mask_np, dist_mat_cpu, pad_id)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] ce={ce_sum/max(1,len(loader)):.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in final_model.state_dict().items()}
    if best_state is not None:
        final_model.load_state_dict(best_state)
    ts_met = eval_pack(final_model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, pad_id)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    logger.info(f"[final {cfg.mode}] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f}")
    return {"tag": tag, "method": "ACTIVE", "upstream": "novel",
            "note": f"BALD/entropy active-learning loop; mode={cfg.mode}",
            "mode": cfg.mode,
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "n_rounds": cfg.n_rounds if cfg.mode == "active" else 0,
            "inner_epochs": cfg.inner_epochs if cfg.mode == "active" else 0,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "acquisition_log": acquisition_log,
            "final_n_samples": len(final_indices),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    encoders = ["unixcoder-base"]
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    modes = ["active", "random"]
    results = []
    for enc in encoders:
        for bench, task, n_cls in benchmarks:
            for frac in fracs:
                for mode in modes:
                    cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac, n_cls=n_cls, mode=mode)
                    tag = f"ext_active_{bench}_f{frac}_{mode}"
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
    with open(os.path.join(out_dir, "exp105_active_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    # Pretty-print: side-by-side active vs random per slot
    print("\n" + "="*150)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Mode':<8} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'NSamp':>7} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['mode']:<8} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['final_n_samples']:>7d} {r['wall']:>8.0f}s")
    print("="*150)
    # Side-by-side delta (active - random) per (bench, frac)
    print("\nACTIVE vs RANDOM (delta = active - random):")
    print("-"*100)
    by_slot: Dict[Tuple[str, float], Dict[str, dict]] = {}
    for r in results:
        slot = (r["bench"], r["frac"])
        by_slot.setdefault(slot, {})[r["mode"]] = r
    for (bench, frac), pair in sorted(by_slot.items()):
        if "active" in pair and "random" in pair:
            a = pair["active"]["macro"]; rnd = pair["random"]["macro"]
            print(f"{bench:<12} frac={frac:>5.0%}  active={a:.4f}  random={rnd:.4f}  delta={a-rnd:+.4f}")
    print("="*100)


if __name__ == "__main__":
    main()
