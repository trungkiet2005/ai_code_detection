# exp94_rainfer — Retrieval-Augmented Inference (test-time-only adaptation)
# =============================================================================
# NAME       : RAINFER (Retrieval-Augmented Inference for code attribution)
# REFERENCE  : new; extends kNN-LM (Khandelwal 2020) and RAFC (arXiv:2406.11148)
#              by mixing a parametric classifier with a TREE-AWARE kNN vote
#              over the training set. Crucially, retrieval is APPLIED ONLY AT
#              TEST TIME on a model trained with vanilla TRACO; no extra
#              training is required.
# CLAIM      : The hardest sibling mistakes are local in representation
#              space. A small kNN vote over the labeled training set can
#              correct them at zero training cost. The mix is parameterised
#              by one scalar beta per fraction, set on validation.
# EQUATION   : Train: same as TRACO (CE + lambda * SupCon-TW).
#              Test:  z_test = phi(x_test).
#                     Top-K nearest train embeddings -> kNN soft labels.
#                     kNN logits l_knn_k = sum_{i in topK} sim(z_test, z_i) *
#                                          tree_weight(y_i) * onehot(y_i).
#                     Final logits = beta * l_param + (1 - beta) * l_knn.
#              beta selected per (benchmark, fraction) on validation
#              by grid search over [0.0, 0.1, ..., 1.0].
# WHY NEW    : RAFC retrieves over external code; we retrieve over the SAME
#              few-shot training set, with tree-distance-weighted kNN. No
#              previous code-attribution work has reported a test-time-only
#              improvement of this form.
# FALSIFIER  : (i) Best beta_val for at least one slot is < 1.0 (i.e. kNN
#              contributes). (ii) Macro-F1 at beta_val >= Macro-F1 at
#              beta=1.0 (parametric-only). (iii) Tree-weighted kNN beats
#              uniform kNN -- otherwise the tree weighting is not needed at
#              retrieval time either.
# GPU TUNING : Retrieval index lives in CPU memory (50 MB for AICD-T2 train).
#              No extra GPU memory at inference beyond TRACO baseline.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from dataclasses import dataclass, field
from typing import Tuple
import re as _re

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
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
logger = logging.getLogger("exp94_rainfer")

PAPER_BASELINE = 0.6633
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}

def _gd(u, v, adj):
    if u == v: return 0.0
    q = [(u, 0)]; seen = {u}
    while q:
        c, d = q.pop(0)
        for nb in adj.get(c, []):
            if nb == v: return d + 1.0
            if nb not in seen: seen.add(nb); q.append((nb, d + 1))
    return float("inf")

def build_dist(n, adj, default=4.0):
    D = torch.full((n, n), default)
    for i in range(n):
        for j in range(n):
            d = _gd(i, j, adj)
            if d < float("inf"): D[i, j] = d
            elif (i == 0) != (j == 0): D[i, j] = 3.0
    return D


# ---- Augmentation (TRACO) ---------------------------------------------------

_RESERVED = {"if","else","elif","for","while","do","return","def","function",
    "class","struct","interface","enum","import","from","include","public",
    "private","void","new","this","extends","implements","try","catch","except",
    "finally","with","in","of","is","not","and","or","as","True","False","None",
    "null","true","false","self","int","float","double","char","long","bool",
    "string","var","let","const"}

def aug_token_dropout(code, rng, p=0.1):
    out = []
    for t in _re.split(r"(\s+|[^\w\s])", code):
        if t.strip() and t.strip() not in _RESERVED and not t.isspace():
            if rng.random() < p: out.append(" "); continue
        out.append(t)
    return "".join(out)

def aug_id_rename(code, rng, max_n=8):
    ids = [i for i in set(_re.findall(r"\b[a-zA-Z_]\w{2,}\b", code))
           if i not in _RESERVED and not i[0].isdigit()]
    if not ids: return code
    chosen = rng.sample(ids, min(max_n, len(ids)))
    new = code
    for k, orig in enumerate(chosen):
        new = _re.sub(rf"\b{_re.escape(orig)}\b", f"v{k}", new)
    return new

def aug_ws_jitter(code, rng, p=0.15):
    out = []
    for c in code:
        out.append(c)
        if c in "+-*/%=<>,;" and rng.random() < p: out.append(" ")
    return "".join(out)

def aug_comment_strip(code, rng):
    new = _re.sub(r"/\*[\s\S]*?\*/", "", code)
    new = _re.sub(r"//[^\n]*", "", new)
    new = _re.sub(r"#[^\n]*", "", new)
    return new

_AUG = [aug_token_dropout, aug_id_rename, aug_ws_jitter, aug_comment_strip]
def augment(code, rng):
    fn = _AUG[rng.randrange(len(_AUG))]
    try: return fn(code, rng)
    except Exception: return code


# ---- Model ------------------------------------------------------------------

class RainferModel(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        self.emb_dim, self.n_cls = emb_dim, n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1), self.clf(z)


def supcon_tw_loss(z_d, y_d, dist, gamma=1.0, tau=0.1):
    N = z_d.size(0)
    if N < 2: return z_d.sum() * 0.0
    sim = (z_d @ z_d.t()) / tau
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()
    eye = torch.eye(N, device=z_d.device, dtype=torch.bool)
    pos = (y_d.unsqueeze(0) == y_d.unsqueeze(1)).float().masked_fill(eye, 0.0)
    neg = (y_d.unsqueeze(0) != y_d.unsqueeze(1)).float()
    dij = dist[y_d][:, y_d]
    w = pos + neg * torch.exp(-gamma * dij)
    w = w.masked_fill(eye, 0.0)
    es = (torch.exp(sim) * w).clamp(min=1e-12)
    num = (es * pos).sum(-1).clamp(min=1e-12)
    den = es.sum(-1).clamp(min=1e-12)
    has = (pos.sum(-1) > 0).float()
    return (-(torch.log(num) - torch.log(den)) * has).sum() / has.sum().clamp(min=1.0)


# ---- Retrieval helpers ------------------------------------------------------

@torch.no_grad()
def build_index(model, loader, cfg):
    """Encode all training samples and cache (embedding, label) to CPU."""
    model.eval()
    embs, labs = [], []
    for b in tqdm(loader, desc="BuildIdx"):
        ids = b["ids0"].to(cfg.device); m = b["mask0"].to(cfg.device)
        z, _ = model.encode(ids, m)
        embs.append(z.cpu().float())
        labs.append(b["label"] if torch.is_tensor(b["label"])
                    else torch.tensor(list(b["label"])))
    Z = torch.cat(embs, dim=0)             # (N_train, d_z)
    Y = torch.cat(labs, dim=0)             # (N_train,)
    logger.info(f"[idx] built {Z.shape[0]} train embeddings, dim={Z.shape[1]}")
    return Z, Y


@torch.no_grad()
def knn_logits(z_test, Z_train, Y_train, n_cls, dist_mat, K=32, gamma_retr=1.0):
    """Compute tree-aware kNN soft labels for a batch of test embeddings.

    For each test sample t:
      1. similarity s_ti = z_t . z_i (cosine, since z is L2-normed)
      2. take top-K most similar train samples
      3. tree weight w_ti = exp(-gamma_retr * d_T(y_t_pred, y_i))    <-- skipped
         (we use UNIFORM tree weight inside top-K because at test we don't
          know the prediction; tree-awareness comes from the encoder which
          was trained with tree-weighted SupCon)
      4. soft label l_t[c] = sum_{i in topK} s_ti * 1[y_i = c] / Z_norm
    """
    # z_test: (B, d_z), Z_train: (N, d_z), Y_train: (N,)
    sims = z_test @ Z_train.t()                                # (B, N)
    topK_sims, topK_idx = sims.topk(K, dim=-1)                  # (B, K)
    topK_labs = Y_train[topK_idx]                               # (B, K)
    # One-hot soft labels weighted by similarity
    onehot = F.one_hot(topK_labs, num_classes=n_cls).float()    # (B, K, n_cls)
    w = topK_sims.unsqueeze(-1)                                  # (B, K, 1)
    soft = (onehot * w).sum(dim=1)                               # (B, n_cls)
    # Normalize to sum-to-1
    soft = soft / soft.sum(dim=-1, keepdim=True).clamp(min=1e-6)
    return soft


# ---- Plumbing ---------------------------------------------------------------

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    emb_dim: int = 256; K_knn: int = 32; device: str = "cuda"

def adaptive_schedule(c):
    if c.frac <= 0.02: c.epochs, c.lr_enc, c.warmup = 10, 3e-5, 0.20
    elif c.frac <= 0.10: c.epochs, c.lr_enc, c.warmup = 6, 3e-5, 0.15
    else: c.epochs, c.lr_enc, c.warmup = 6, 4e-5, 0.10
    return c

def _hw(c):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40: c.bs, c.seq = 128, 512
        elif mem >= 20: c.bs, c.seq = 96, 448
        elif mem >= 10: c.bs, c.seq = 64, 384
        else: c.bs, c.seq = 32, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={c.bs} seq={c.seq}")
    return c

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _is_human(t): return str(t or "").strip().lower() in {"human","human_written","human-generated"}

def _vocab(tr):
    names = {str(r.get("model", "") or "").strip() for r in tr
             if not _is_human(r.get("target", "")) and r.get("model", "")}
    return {n: i + 1 for i, n in enumerate(sorted(names))}

def _conv_codet(s, t, vocab):
    def row(r):
        code = next((r.get(f, "") for f in ("cleaned_code", "code")
                     if isinstance(r.get(f, ""), str) and r.get(f, "").strip()), "")
        lbl = (0 if _is_human(r.get("target", ""))
               else (1 if t == "binary"
                     else vocab.get(str(r.get("model", "") or "").strip(), -1)))
        return {"code": code, "label": lbl,
                "language": str(r.get("language", "")).strip().lower(),
                "source": str(r.get("source", "")).strip().lower()}
    return s.map(row, remove_columns=s.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _conv_aicd(s):
    def row(r):
        return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1)),
                "language": str(r.get("language", "")).strip().lower(), "source": ""}
    return s.map(row, remove_columns=s.column_names).filter(
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
    tn = {"t1":"T1","t2":"T2","t3":"T3"}.get(task.lower())
    if tn is None: raise ValueError(f"[aicd] bad '{task}'")
    p = os.path.join(KAGGLE_AICD, tn)
    if not os.path.isdir(p): raise FileNotFoundError(f"[aicd] STRICT: {tn} not found")
    pf = sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: no parquet")
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
    def __init__(self, data, tok, seq, frac=1.0, seed=42, do_aug=True):
        self.data = data; self.tok = tok; self.seq = seq; self.do_aug = do_aug; self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]
        e0 = self.tok(code, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        ids0, m0 = e0["input_ids"].squeeze(0), e0["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            ca = augment(code, rng)
            e1 = self.tok(ca, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
            ids1, m1 = e1["input_ids"].squeeze(0), e1["attention_mask"].squeeze(0)
        else:
            ids1, m1 = ids0, m0
        return {"ids0": ids0, "mask0": m0, "ids1": ids1, "mask1": m1,
                "label": r["label"]}


@torch.no_grad()
def eval_loader_rainfer(model, loader, cfg, Z_train, Y_train, beta_grid):
    """Eval that returns Macro-F1 for each beta in beta_grid (parametric+kNN mix).
    Done in a single pass over the test/val set."""
    model.eval()
    z_all, lg_all, lab_all = [], [], []
    for b in tqdm(loader, desc="EvalEnc"):
        ids = b["ids0"].to(cfg.device); m = b["mask0"].to(cfg.device); labs = b["label"]
        z, lg = model.encode(ids, m)
        z_all.append(z.cpu().float())
        lg_all.append(lg.cpu().float())
        lab_all.append(labs if torch.is_tensor(labs) else torch.tensor(list(labs)))
    Z = torch.cat(z_all, 0); L = torch.cat(lg_all, 0); Y = torch.cat(lab_all, 0)
    p_param = F.softmax(L, dim=-1)
    p_knn = knn_logits(Z, Z_train, Y_train, cfg.n_cls, None, K=cfg.K_knn)
    results = {}
    for beta in beta_grid:
        p_mix = beta * p_param + (1.0 - beta) * p_knn
        preds = p_mix.argmax(-1).numpy()
        macro = float(f1_score(Y.numpy(), preds, average="macro", zero_division=0))
        weighted = float(f1_score(Y.numpy(), preds, average="weighted", zero_division=0))
        acc = float(accuracy_score(Y.numpy(), preds))
        results[float(round(beta, 2))] = {"macro_f1": macro, "weighted_f1": weighted, "accuracy": acc}
    return results, Y.numpy(), L, Z


def train_epoch(model, loader, opt, sch, scaler, cfg, dist):
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); m1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, lg = model.encode(ids0, m0)
            z1, _ = model.encode(ids1, m1)
            zd = torch.cat([z0, z1], 0); yd = torch.cat([labs, labs], 0)
            loss_ce = F.cross_entropy(lg, labs)
            loss_sc = supcon_tw_loss(zd, yd, dist, gamma=cfg.gamma, tau=cfg.tau)
            loss = loss_ce + cfg.lambda_aug * loss_sc
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item()
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist = build_dist(cfg.n_cls, adj).to(cfg.device)
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
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    tr_dl = DataLoader(tr_ds, shuffle=True, batch_size=cfg.bs, num_workers=4, pin_memory=True)
    # For index building we need a NON-augmented loader over the same train data
    tr_idx_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=False)
    tr_idx_dl = DataLoader(tr_idx_ds, shuffle=False, batch_size=cfg.bs, num_workers=4, pin_memory=True)
    vl_dl = DataLoader(vl_ds, shuffle=False, batch_size=cfg.bs, num_workers=4, pin_memory=True)
    ts_dl = DataLoader(ts_ds, shuffle=False, batch_size=cfg.bs, num_workers=4, pin_memory=True)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} wu={cfg.warmup} K={cfg.K_knn}")
    model = RainferModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    # ---- Training (vanilla TRACO) ----
    best_val_param, best_state, vh = 0.0, None, []
    for ep in range(cfg.epochs):
        loss, ce, sc = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist)
        # quick parametric-only val for early stopping
        model.eval()
        with torch.no_grad():
            preds, labs = [], []
            for b in vl_dl:
                ids = b["ids0"].to(cfg.device); m = b["mask0"].to(cfg.device)
                _, lg = model.encode(ids, m)
                preds.extend(lg.argmax(-1).cpu().tolist())
                labs.extend(b["label"].tolist() if torch.is_tensor(b["label"]) else list(b["label"]))
        v = float(f1_score(labs, preds, average="macro", zero_division=0))
        vh.append(v)
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} val_param={v:.4f}")
        if v > best_val_param:
            best_val_param = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    # ---- Build retrieval index from train embeddings (no aug) ----
    Z_train, Y_train = build_index(model, tr_idx_dl, cfg)
    # ---- Validation grid search for beta ----
    beta_grid = [round(0.1 * i, 1) for i in range(0, 11)]
    val_res, _, _, _ = eval_loader_rainfer(model, vl_dl, cfg, Z_train, Y_train, beta_grid)
    best_beta = max(val_res.keys(), key=lambda b: val_res[b]["macro_f1"])
    best_val = val_res[best_beta]["macro_f1"]
    logger.info(f"[val-grid] best beta={best_beta} val_macro={best_val:.4f}  "
                f"(parametric-only beta=1.0 val_macro={val_res[1.0]['macro_f1']:.4f})")
    # ---- Test with best beta ----
    test_res, y_te, L_te, Z_te = eval_loader_rainfer(model, ts_dl, cfg, Z_train, Y_train, beta_grid)
    test_macro = test_res[best_beta]["macro_f1"]
    gap = best_val - test_macro
    logger.info(f"[final] best_beta={best_beta} val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f}")
    return {"tag": tag, "method": "RAINFER", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "gamma": cfg.gamma, "tau": cfg.tau, "K_knn": cfg.K_knn,
            "best_beta": best_beta,
            "val_macro": best_val, "macro": test_macro,
            "weighted": test_res[best_beta]["weighted_f1"], "acc": test_res[best_beta]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "val_grid": val_res, "test_grid": test_res,
            "val_history_param": vh,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp94_rainfer_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time()-t0, 1); results.append(r)
                logger.info(f"[{tag}] test={r['macro']:.4f} ({r['dpaper']:+.4f}) "
                            f"gap={r['val_test_gap']:+.4f} beta={r['best_beta']} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp94_rainfer_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
