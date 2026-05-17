# exp73_tapa — Tree-Anchored Prototypical Attribution (TAPA)
# =============================================================================
# Theory-Track exp -- TAPA (Tree-Anchored Prototypical Attribution)
#
# ROLE           : Prototype-based few-shot attribution is the missing paradigm
#                  in our portfolio.  We adapt Prototypical Networks (Snell
#                  et al., 2017, arXiv:1703.05175) by adding TWO problem-
#                  specific inductive biases:
#                    1. Multi-layer encoder pooling (LIGHT, arXiv:2503.00958):
#                       different transformer layers capture different
#                       stylistic signal.  Learnable softmax weight per layer.
#                    2. Tree-anchored prototype geometry: pairwise prototype
#                       distances are constrained to MATCH the genealogy tree
#                       distance via auxiliary MSE loss.  Prototypes are not
#                       free -- they live on a tree-shaped manifold.
# NAME           : TAPA  (Tree-Anchored Prototypical Attribution)
# ARXIV_ID       : extends Snell 2017 + LIGHT 2025 with a tree-isometry
#                  constraint specific to LLM-genealogy.
# ONE-LINE CLAIM : Few-shot LLM-code attribution is solved by prototypes
#                  whose geometry is anchored to the generative-model
#                  genealogy and whose encoder pools layers adaptively.
# EQUATION       : alpha_l = softmax(layer_weights)   l = 1..L
#                  h(x) = sum_l alpha_l * pool(H_l(x))
#                  z = phi(h(x))                        in R^d
#                  mu_k  EMA prototype, updated each step from batch members
#                  d_ik = ||z_i - mu_k||^2
#                  Classification: p(y=k | x) = softmax_k(-d_ik / exp(log_tau))
#                  Tree-anchor:  L_tree = MSE( ||mu_i - mu_j|| ,
#                                              scale * d_tree(i, j) )
#                  Total:  L = L_CE_proto  +  lambda_tree * L_tree
# WHY NOT BEFORE : Prototypical networks for code authorship exist
#                  (AuthAttLyzer-V2 arXiv:2406.19896) but they do NOT use
#                  multi-layer pooling and do NOT constrain prototype geometry
#                  to a known label tree.  The combination is the novelty.
# FALSIFIER      : (F1) Learned layer weights: max_l alpha_l - min_l alpha_l.
#                       If close to 0, all layers contribute equally -- no
#                       multi-layer effect.
#                  (F2) Spearman(||mu_i - mu_j||, d_tree(i,j)) at end.
#                       If < 0.7, tree anchor failed.
#                  (F3) Prototype margin: mean inter-class proto distance
#                       MINUS mean intra-class z-to-mu distance.  Positive
#                       large value = prototypes well-separated.
# REPORTS        : Full eval pack + (F1) per-layer alpha values
#                                 + (F2) proto-tree Spearman
#                                 + (F3) prototype margin
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
logger = logging.getLogger("exp73_tapa")

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

class TAPAModel(nn.Module):
    def __init__(self, enc_name, n_cls, pool_dim=256, n_layers_pool=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.n_layers_pool = n_layers_pool
        # Learnable softmax weights over last n_layers_pool transformer layers
        self.layer_weights = nn.Parameter(torch.zeros(n_layers_pool))
        self.proj = nn.Sequential(nn.Linear(hidden, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, pool_dim))
        # Online prototype memory (EMA-updated, not autograd)
        self.register_buffer("prototypes", torch.zeros(n_cls, pool_dim))
        self.register_buffer("proto_init", torch.zeros(n_cls, dtype=torch.bool))
        # learnable temperature
        self.log_tau = nn.Parameter(torch.tensor(0.0))
        self.n_cls = n_cls
        self.pool_dim = pool_dim

    def get_layer_weights(self):
        return F.softmax(self.layer_weights, dim=0)

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask, output_hidden_states=True)
        # hidden_states is tuple of (n_layers+1) tensors, each (B, T, D)
        hs = out.hidden_states
        last_k = hs[-self.n_layers_pool:]
        stacked = torch.stack(last_k, dim=0)             # (K, B, T, D)
        w = self.get_layer_weights().view(-1, 1, 1, 1)
        pooled = (stacked * w).sum(dim=0)                # (B, T, D)
        m = mask.unsqueeze(-1).float()
        sem = (pooled * m).sum(dim=1) / m.sum(dim=1).clamp(min=1)
        return self.proj(sem)

    def forward(self, ids, mask):
        z = self.encode(ids, mask)
        # squared euclidean to prototypes
        d2 = torch.cdist(z, self.prototypes, p=2) ** 2   # (B, n_cls)
        logits = -d2 / torch.exp(self.log_tau)
        return logits, z

    @torch.no_grad()
    def update_prototypes(self, z, labels, ema=0.9):
        z_det = z.detach().float()
        for c in range(self.n_cls):
            m = (labels == c)
            if m.sum() < 1: continue
            batch_proto = z_det[m].mean(dim=0)
            if not bool(self.proto_init[c].item()):
                self.prototypes[c] = batch_proto
                self.proto_init[c] = True
            else:
                self.prototypes[c] = ema * self.prototypes[c] + (1.0 - ema) * batch_proto


def tree_anchor_loss(prototypes, dist_mat, D_max=4.0):
    """MSE between pairwise prototype L2-distance and tree distance (both normalized)."""
    n = prototypes.size(0)
    if n < 2: return prototypes.sum() * 0.0
    d_p = torch.cdist(prototypes, prototypes, p=2)
    eye = torch.eye(n, device=prototypes.device, dtype=torch.bool)
    mask = ~eye
    d_p_mx = d_p[mask].max().clamp(min=1e-6)
    d_p_norm = d_p[mask] / d_p_mx
    d_t_norm = dist_mat[mask] / D_max
    return F.mse_loss(d_p_norm, d_t_norm)


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
    lambda_tree: float = 0.5
    proto_ema: float = 0.9
    n_layers_pool: int = 4
    pool_dim: int = 256
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
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=False):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    z_per_lab = {}
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        logits, z = model(ids, mask)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_falsifier:
            z_cpu = z.detach().cpu().float().numpy()
            for li, zi in zip(labs.tolist() if torch.is_tensor(labs) else list(labs), z_cpu):
                z_per_lab.setdefault(int(li), []).append(zi)
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
        with torch.no_grad():
            protos = model.prototypes.detach().cpu().float()
            d_p = torch.cdist(protos, protos).numpy()
        n = n_cls; tri = np.triu_indices(n, k=1)
        rho, _ = spearmanr(d_p[tri], dist_mat_cpu[tri])
        # prototype margin: mean inter-class - mean intra-class
        mean_inter = float(d_p[tri].mean())
        intra_dists = []
        for c, zs in z_per_lab.items():
            zs = np.array(zs)
            if len(zs) < 2: continue
            mu = protos[c].numpy()
            intra_dists.extend(np.linalg.norm(zs - mu, axis=-1).tolist())
        mean_intra = float(np.mean(intra_dists)) if intra_dists else 0.0
        proto_margin = mean_inter - mean_intra
        alpha = model.get_layer_weights().detach().cpu().float().tolist()
        out["falsifier"] = {
            "layer_alpha_F1": alpha,
            "layer_alpha_range_F1": float(max(alpha) - min(alpha)),
            "proto_spearman_F2": float(rho) if not np.isnan(rho) else 0.0,
            "mean_inter_proto_dist_F3": mean_inter,
            "mean_intra_proto_dist_F3": mean_intra,
            "proto_margin_F3": float(proto_margin),
        }
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg, dist_mat):
    model.train(); tot, ce_s, tr_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            logits, z = model(ids, mask)
            loss_ce = F.cross_entropy(logits, labs)
            loss_tr = tree_anchor_loss(model.prototypes, dist_mat)
            loss = loss_ce + cfg.lambda_tree * loss_tr
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        # EMA prototype update AFTER optimizer step (no autograd)
        model.update_prototypes(z, labs, ema=cfg.proto_ema)
        tot += loss.item(); ce_s += loss_ce.item(); tr_s += loss_tr.item()
    n = len(loader)
    return tot/n, ce_s/n, tr_s/n


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
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} warmup={cfg.warmup} "
                f"lambda_tree={cfg.lambda_tree} ema={cfg.proto_ema} n_layers_pool={cfg.n_layers_pool}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = TAPAModel(cfg.enc, cfg.n_cls, cfg.pool_dim, cfg.n_layers_pool).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids and p.requires_grad]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, ce, tr = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist_mat)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        alpha = model.get_layer_weights().detach().cpu().float().tolist()
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} tree={tr:.4f} val={v:.4f} "
                    f"alpha={[f'{a:.2f}' for a in alpha]}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu, collect_falsifier=True)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    fa = ts_met["falsifier"]
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"proto_rho={fa['proto_spearman_F2']:.3f} margin={fa['proto_margin_F3']:.3f} "
                f"alpha_range={fa['layer_alpha_range_F1']:.3f}")
    return {"tag": tag, "method": "TAPA", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_tree": cfg.lambda_tree, "proto_ema": cfg.proto_ema,
            "n_layers_pool": cfg.n_layers_pool, "pool_dim": cfg.pool_dim,
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
                tag = f"exp73_tapa_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    fa = res['test_metrics']['falsifier']
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"rho_proto={fa['proto_spearman_F2']:+.3f} "
                                f"margin={fa['proto_margin_F3']:+.3f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp73_tapa_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*140)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'rho':>7} {'margin':>8} {'a_rng':>7} {'Wall':>8}")
    print("-"*140)
    for r in results:
        fa = r['test_metrics']['falsifier']
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{fa['proto_spearman_F2']:>+7.3f} {fa['proto_margin_F3']:>+8.3f} "
              f"{fa['layer_alpha_range_F1']:>7.3f} {r['wall']:>8.0f}s")
    print("="*140)


if __name__ == "__main__":
    main()
