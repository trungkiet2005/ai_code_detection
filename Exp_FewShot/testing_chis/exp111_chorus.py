# exp111 — CHORUS
# NAME       : CHORUS (Author identity as residual after subtracting decoding-temperature)
# REFERENCE  : new; gradient reversal layer (Ganin 2015, DANN 2016)
#              repurposed for orthogonalizing two confounds in code
#              generation.
# CLAIM      : LLM-generated code has TWO confounded sources of variation:
#              author identity (the model) and decoding parameters
#              (temperature, top-p). A classifier that ALSO captures
#              temperature is leaking signal. Train the encoder to be
#              SIMULTANEOUSLY good at author AND BAD at predicting
#              temperature (via gradient reversal). The residual
#              representation is temperature-invariant author identity.
# EQUATION   : z = phi(x)
#              L_main = CE(W_author · z, y_author)
#              L_aux  = CE(W_temp · GradReverse(z), y_temp)
#              L = L_main + λ_TW·L_TRACO(z, y_author) + λ_adv·L_aux
#              At test, use only W_author.
# WHY NEW    : DECOFP (exp70) predicted temperature as auxiliary task.
#              CHORUS does the OPPOSITE — explicitly UN-predicts it from
#              z via gradient reversal. Inverts the sign of the signal.
# WOW HOOK   : "Author identity is what is LEFT OVER after we remove
#              decoding temperature. We disentangle by negative gradient
#              and verify by probing the cleaned representation."
# FALSIFIER  : (F1) After training, frozen-z linear probe Macro-F1 on
#              temperature < 0.30 (chance is 1/T_buckets=0.25).
#              (F2) Composite > TRACO by ≥ +0.005 at AICD-T2 (where
#              decoding variance is biggest). (F3) Removing the
#              gradient-reversal head loses ≥ 0.005.
#
# NOTE on temperature proxy: the CoDET-M4 / AICD-T2 datasets do NOT expose
# the actual decoding temperature used by the generator. We compute a
# *deterministic, code-derived* proxy: the normalized token-entropy of
# the sample, quartile-bucketed into 4 categories. High token entropy
# correlates with high temperature (more random sampling -> more lexical
# diversity per-author). This is an approximation; if the adversarial
# regularization works on a noisy proxy, it should work better on a real
# temperature label. The mechanism is the contribution; the proxy is the
# best we can do without re-generating the corpus.
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math, collections
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
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp111_chorus")

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
# Temperature proxy (deterministic, code-derived)
# =============================================================================

def temperature_proxy_bucket(code: str, n_buckets: int = 4) -> int:
    """Approximate temperature bucket from normalized token-entropy.

    High entropy ≈ many distinct tokens with flat distribution ≈ likely sampled
    at higher temperature. Normalized to [0, 1] by dividing by log(N_tokens)
    (= maximum possible entropy if every token were unique). Then bucketed.

    Deterministic per sample.
    """
    if not code or len(code) < 10: return 0
    toks = code.split()
    if len(toks) < 3: return 0
    counts = collections.Counter(toks)
    total = sum(counts.values())
    probs = [c / total for c in counts.values()]
    ent = -sum(p * math.log(p + 1e-12) for p in probs)
    max_ent = math.log(max(2, len(toks)))
    norm_ent = ent / max_ent if max_ent > 0 else 0.0
    norm_ent = max(0.0, min(1.0, norm_ent))
    return min(n_buckets - 1, int(norm_ent * n_buckets))


# =============================================================================
# Gradient-reversal layer (DANN, Ganin 2015)
# =============================================================================

class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.alpha * grad_output, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# =============================================================================
# Model
# =============================================================================

class CHORUSModel(nn.Module):
    def __init__(self, enc_name, n_cls, n_temp_buckets=4, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1),
                                  nn.Linear(512, emb_dim))
        self.clf_main = nn.Linear(emb_dim, n_cls)
        # Temperature predictor head; sees gradient-reversed z.
        self.clf_temp = nn.Sequential(nn.Linear(emb_dim, 128), nn.GELU(),
                                      nn.Dropout(0.1), nn.Linear(128, n_temp_buckets))
        self.emb_dim = emb_dim
        self.n_cls = n_cls
        self.n_temp_buckets = n_temp_buckets

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = F.normalize(self.proj(sem), dim=-1)
        return z

    def forward(self, ids, mask, alpha=1.0):
        z = self.encode(ids, mask)
        logits_main = self.clf_main(z)
        z_rev = grad_reverse(z, alpha)
        logits_temp = self.clf_temp(z_rev)
        return z, logits_main, logits_temp

    def encode_with_logits(self, ids, mask):
        """Inference: only main classifier (temperature head ignored)."""
        z = self.encode(ids, mask)
        return z, self.clf_main(z)


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
    lambda_adv: float = 0.3
    n_temp_buckets: int = 4
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
        # Halve bs because we keep a 2-view doubled buffer for SupCon
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


class FSDS_CHORUS(TD):
    """Dataset with temperature proxy bucket attached per sample."""
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42, n_temp_buckets=4):
        self.data = data; self.tok = tok; self.seq_len = seq_len
        self.seed = seed; self.n_temp_buckets = n_temp_buckets
        if frac < 1.0:
            rng = random.Random(seed)
            max_lbl = max(self.data["label"]) + 1
            keep = []
            for lbl in range(max_lbl):
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                if not idx: continue
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_CHORUS] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        lang = r.get("language", "") or ""
        temp_bucket = temperature_proxy_bucket(code, n_buckets=self.n_temp_buckets)
        enc = self.tok(code, max_length=self.seq_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        ids0, mask0 = enc["input_ids"].squeeze(0), enc["attention_mask"].squeeze(0)
        return {"ids0": ids0, "mask0": mask0, "label": r["label"],
                "temp_bucket": int(temp_bucket),
                "language": lang, "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device); labs = b["label"]
        z, logits = model.encode_with_logits(ids0, mask0)
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


@torch.no_grad()
def probe_temperature_invariance(model, test_loader, cfg):
    """Train a linear probe on FROZEN z to predict temp_bucket.
    Lower Macro-F1 = better disentanglement.
    Falsifier F1: probe macro_f1 should be < 0.30."""
    model.eval()
    embs, temp_labs, author_labs = [], [], []
    for b in tqdm(test_loader, desc="ProbeExtract"):
        ids = b["ids0"].to(cfg.device); mask = b["mask0"].to(cfg.device)
        z = model.encode(ids, mask)
        embs.append(z.float().cpu())
        tb = b["temp_bucket"]
        temp_labs.extend(tb.tolist() if torch.is_tensor(tb) else list(tb))
        la = b["label"]
        author_labs.extend(la.tolist() if torch.is_tensor(la) else list(la))
    if not embs:
        return {"probe_temperature_macro_f1": 0.0, "n_probe_samples": 0,
                "temp_bucket_distribution": {}}
    X = torch.cat(embs, 0).numpy()
    y = np.array(temp_labs)
    counts = collections.Counter(y.tolist())
    dist = {int(k): int(v) for k, v in counts.items()}
    # Need at least 2 classes with >= 2 samples for cross-val to be meaningful
    valid_classes = [k for k, v in counts.items() if v >= 3]
    if len(valid_classes) < 2:
        logger.warning(f"[probe] Insufficient temperature variation; classes={dist}")
        return {"probe_temperature_macro_f1": float('nan'),
                "n_probe_samples": int(len(y)),
                "temp_bucket_distribution": dist,
                "note": "insufficient temp_bucket variation for probe"}
    try:
        clf = LogisticRegression(max_iter=300, C=1.0, multi_class='auto', solver='lbfgs')
        scores = cross_val_score(clf, X, y, cv=3, scoring='f1_macro')
        probe_f1 = float(scores.mean())
        probe_std = float(scores.std())
    except Exception as e:
        logger.warning(f"[probe] failed: {e}")
        probe_f1 = float('nan'); probe_std = float('nan')
    return {"probe_temperature_macro_f1": probe_f1,
            "probe_temperature_macro_f1_std": probe_std,
            "n_probe_samples": int(len(y)),
            "temp_bucket_distribution": dist}


def train_epoch(model, loader, opt, sch, scaler, cfg, dist_mat, epoch_progress_iter,
                total_steps_for_alpha):
    """Train one epoch with adversarial temperature head.
    alpha (gradient-reversal strength) is annealed via DANN schedule:
        alpha = 2 / (1 + exp(-10 * progress)) - 1
    where progress runs from 0 to 1 over the full training run."""
    model.train()
    tot, ce_s, sc_s, adv_s = 0.0, 0.0, 0.0, 0.0
    adv_correct, adv_total = 0, 0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); mask0 = b["mask0"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        tb = b["temp_bucket"]
        temp_labs = tb.to(cfg.device) if torch.is_tensor(tb) else torch.tensor(list(tb), device=cfg.device)
        step = next(epoch_progress_iter)
        progress = min(1.0, step / max(1, total_steps_for_alpha))
        alpha = 2.0 / (1.0 + math.exp(-10.0 * progress)) - 1.0
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z, logits_main, logits_temp = model(ids0, mask0, alpha=alpha)
            loss_ce = F.cross_entropy(logits_main, labs)
            # SupCon on doubled view (label-only positives; no augmentation here so use z twice)
            z_d = torch.cat([z, z], dim=0)
            y_d = torch.cat([labs, labs], dim=0)
            loss_sc = supcon_tw_loss(z_d, y_d, dist_mat, gamma=cfg.gamma, tau=cfg.tau)
            loss_adv = F.cross_entropy(logits_temp, temp_labs)
            loss = loss_ce + cfg.lambda_aug * loss_sc + cfg.lambda_adv * loss_adv
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item()
        sc_s += loss_sc.item(); adv_s += loss_adv.item()
        with torch.no_grad():
            pred_t = logits_temp.argmax(dim=-1)
            adv_correct += (pred_t == temp_labs).sum().item()
            adv_total += temp_labs.size(0)
    n = len(loader)
    adv_acc = adv_correct / max(adv_total, 1)
    return tot/n, ce_s/n, sc_s/n, adv_s/n, adv_acc


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
    tr_ds = FSDS_CHORUS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed,
                        n_temp_buckets=cfg.n_temp_buckets)
    vl_ds = FSDS_CHORUS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1,
                        n_temp_buckets=cfg.n_temp_buckets)
    ts_ds = FSDS_CHORUS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2,
                        n_temp_buckets=cfg.n_temp_buckets)
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} "
                f"warmup={cfg.warmup} lambda_adv={cfg.lambda_adv} "
                f"n_temp_buckets={cfg.n_temp_buckets}")
    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    model = CHORUSModel(cfg.enc, cfg.n_cls, n_temp_buckets=cfg.n_temp_buckets,
                        emb_dim=cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()
    best_val, best_state, val_hist = 0.0, None, []
    adv_acc_hist = []
    # Global step counter (across epochs) for alpha schedule
    def _step_gen():
        s = 0
        while True:
            s += 1
            yield s
    step_iter = _step_gen()
    for epoch in range(cfg.epochs):
        loss, ce, sc, adv, adv_acc = train_epoch(model, tr_dl, opt, sch, scaler, cfg,
                                                  dist_mat, step_iter, total_steps)
        adv_acc_hist.append(adv_acc)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} "
                    f"adv={adv:.4f} adv_acc={adv_acc:.3f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    # Falsifier F1: train linear probe on frozen z for temp prediction
    probe_info = probe_temperature_invariance(model, ts_dl, cfg)
    probe_f1 = probe_info.get("probe_temperature_macro_f1", float('nan'))
    chance = 1.0 / cfg.n_temp_buckets
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"probe_temp_F1={probe_f1:.4f} (chance={chance:.3f}) "
                f"final_adv_acc={adv_acc_hist[-1] if adv_acc_hist else 0:.3f}")
    return {"tag": tag, "method": "CHORUS",
            "note": ("adversarial gradient-reversal on entropy-bucketed temperature proxy; "
                     "encoder trained to be BAD at temp prediction; verified via frozen-z probe"),
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "lambda_adv": cfg.lambda_adv,
            "gamma": cfg.gamma, "tau": cfg.tau, "n_temp_buckets": cfg.n_temp_buckets,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "probe_temperature_macro_f1": probe_f1,
            "probe_chance_level": chance,
            "probe_info": probe_info,
            "adv_acc_history": adv_acc_hist,
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
                tag = f"exp111_chorus_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"probe_F1={res['probe_temperature_macro_f1']:.4f} "
                                f"(chance={res['probe_chance_level']:.3f}) "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp111_chorus_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Encoder':<20} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'ProbeF1':>9} {'Chance':>8} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['enc']:<20} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['probe_temperature_macro_f1']:>9.4f} "
              f"{r['probe_chance_level']:>8.3f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
