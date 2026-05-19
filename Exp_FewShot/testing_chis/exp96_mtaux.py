# exp96_mtaux — Multi-Task Auxiliary Genealogy Heads (representation shaping)
# =============================================================================
# NAME       : MTAUX (Multi-Task with auxiliary genealogy-aware heads)
# REFERENCE  : new; differs from GENEPRINT (exp71, FAILED) by NOT enforcing
#              orthogonality. Instead, three heads SHARE the encoder and
#              their gradients shape a single representation that the
#              classification head also reads. Differs from TRACO by
#              replacing the contrastive doubled batch with three small
#              regression / classification auxiliary tasks on the SAME
#              forward pass.
# CLAIM      : The encoder learns better attribution representations when
#              its gradient signal carries explicit information about the
#              data-generating process (the genealogy tree distance, the
#              decoding-temperature proxy, and the sibling-pair relation),
#              even if these tasks are never consumed at test time. This is
#              the "auxiliary task shaping" hypothesis from multi-task
#              learning literature, adapted to few-shot code attribution.
# EQUATION   : Single encoder phi, four heads:
#                h_clf    : phi(x) -> K-class softmax (CE on label)
#                h_dist   : (phi(x), phi(x_pair)) -> regress d_T(y, y_pair)
#                h_sib    : (phi(x), phi(x_pair)) -> binary "are siblings?"
#                h_decode : phi(x) -> regress proxy decoder-T value
#                           (proxy = std-dev of repeated-token rate)
#              Loss: L = L_ce + a*L_dist + b*L_sib + c*L_decode
#              At inference only h_clf is used; the other heads exist only
#              to shape phi during training.
# WHY NEW    : GENEPRINT explicitly SPLIT the representation into channels
#              and failed because the classifier ignored channels. MTAUX
#              shares a single representation and lets the auxiliary tasks
#              regularize WITHOUT forcing channel-level decomposition.
# FALSIFIER  : (i) Macro-F1 strictly greater than CE-only baseline (exp65)
#                  at extreme few-shot.
#              (ii) Aux losses converge to non-trivial values (not constant).
#              (iii) Per-class lift on sibling-pair classes (1, 3 on
#                   CoDET-M4) > lift on non-sibling classes.
# GPU TUNING : Within-batch pairs constructed without a 2x forward; the
#              pair is (x_i, x_{(i+1) mod B}) so single-view forward suffices.
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
logger = logging.getLogger("exp96_mtaux")

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

def build_sib_mask(n, adj):
    M = torch.zeros(n, n)
    for i in range(n):
        for j in adj.get(i, []): M[i, j] = 1.0
    return M


def repeated_token_rate(code: str) -> float:
    """Proxy for decoder-T: lower temperature -> more repeated tokens.
    We compute (n_unique / n_tokens) on whitespace-split tokens.
    Higher value -> more diverse -> higher implied T."""
    toks = code.split()
    if not toks: return 1.0
    return len(set(toks)) / max(1, len(toks))


# ---- Model ------------------------------------------------------------------

class MTAUXModel(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, emb_dim))
        # 4 heads, all reading from the same emb_dim representation
        self.clf = nn.Linear(emb_dim, n_cls)
        self.head_dist = nn.Sequential(nn.Linear(2 * emb_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_sib = nn.Sequential(nn.Linear(2 * emb_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.head_dec = nn.Sequential(nn.Linear(emb_dim, 64), nn.GELU(), nn.Linear(64, 1))
        self.emb_dim, self.n_cls = emb_dim, n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1), self.clf(z)

    def aux(self, z_a, z_b):
        pair = torch.cat([z_a, z_b], dim=-1)
        return self.head_dist(pair).squeeze(-1), self.head_sib(pair).squeeze(-1)

    def decoder_t_head(self, z):
        return self.head_dec(z).squeeze(-1)


# ---- Plumbing ---------------------------------------------------------------

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    a_dist: float = 0.10; b_sib: float = 0.10; c_decode: float = 0.05
    emb_dim: int = 256; device: str = "cuda"

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
        # Single-view forward (no contrastive doubled batch). Can use larger bs.
        if mem >= 40: c.bs, c.seq = 192, 512
        elif mem >= 20: c.bs, c.seq = 128, 448
        elif mem >= 10: c.bs, c.seq = 96, 384
        else: c.bs, c.seq = 48, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={c.bs} seq={c.seq} (single-view, 4 aux heads)")
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
                "source": str(r.get("source", "")).strip().lower(),
                "tdiv": repeated_token_rate(code)}
    return s.map(row, remove_columns=s.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _conv_aicd(s):
    def row(r):
        code = str(r.get("code", "")).strip()
        return {"code": code, "label": int(r.get("label", -1)),
                "language": str(r.get("language", "")).strip().lower(), "source": "",
                "tdiv": repeated_token_rate(code)}
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


class FSDS_MT(TD):
    def __init__(self, data, tok, seq, frac=1.0, seed=42):
        self.data = data; self.tok = tok; self.seq = seq; self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_MT] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]
        e = self.tok(code, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        return {"ids": e["input_ids"].squeeze(0), "mask": e["attention_mask"].squeeze(0),
                "label": r["label"], "tdiv": float(r.get("tdiv", 0.5))}


@torch.no_grad()
def eval_pack(model, loader, cfg):
    model.eval(); preds, labels = [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device); labs = b["label"]
        _, lg = model.encode(ids, mask)
        preds.extend(lg.argmax(-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
    preds = np.array(preds); labels = np.array(labels); n = cfg.n_cls
    ov = {"accuracy": float(accuracy_score(labels, preds)),
          "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
          "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
          "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
          "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0))}
    pcf1 = f1_score(labels, preds, average=None, zero_division=0, labels=list(range(n))).tolist()
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
    return {"overall": ov, "per_class_f1": pcf1,
            "confusion_matrix": cm.tolist(), "n_samples": int(len(labels))}


def train_epoch(model, loader, opt, sch, scaler, cfg, dist, sib_mask):
    model.train(); tot, ce_s, dl_s, sib_s, dec_s = 0.0, 0.0, 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        tdiv = b["tdiv"].to(cfg.device).float()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z, lg = model.encode(ids, mask)
            # Pair construction: roll by one -> (z_i, z_{i+1}) per sample
            z_pair = torch.roll(z, shifts=1, dims=0)
            y_pair = torch.roll(labs, shifts=1, dims=0)
            d_target = dist[labs][torch.arange(labs.size(0), device=cfg.device), y_pair]  # scalar per pair
            sib_target = sib_mask[labs][torch.arange(labs.size(0), device=cfg.device), y_pair]
            d_pred, sib_pred = model.aux(z, z_pair)
            dec_pred = model.decoder_t_head(z)
            loss_ce = F.cross_entropy(lg, labs)
            loss_dist = F.smooth_l1_loss(d_pred, d_target.float())
            loss_sib = F.binary_cross_entropy_with_logits(sib_pred, sib_target.float())
            loss_dec = F.smooth_l1_loss(dec_pred, tdiv)
            loss = loss_ce + cfg.a_dist * loss_dist + cfg.b_sib * loss_sib + cfg.c_decode * loss_dec
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); dl_s += loss_dist.item()
        sib_s += loss_sib.item(); dec_s += loss_dec.item()
    n = len(loader)
    return tot/n, ce_s/n, dl_s/n, sib_s/n, dec_s/n


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist = build_dist(cfg.n_cls, adj).to(cfg.device)
    sib_mask = build_sib_mask(cfg.n_cls, adj).to(cfg.device)
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
    tr_ds = FSDS_MT(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS_MT(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1)
    ts_ds = FSDS_MT(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} wu={cfg.warmup} "
                f"a={cfg.a_dist} b={cfg.b_sib} c={cfg.c_decode}")
    lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = MTAUXModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh = 0.0, None, []
    for ep in range(cfg.epochs):
        loss, ce, dl, sb, dc = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist, sib_mask)
        vm = eval_pack(model, vl_dl, cfg)
        v = vm["overall"]["macro_f1"]; vh.append(v)
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce={ce:.4f} dist={dl:.4f} sib={sb:.4f} dec={dc:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    tm = eval_pack(model, ts_dl, cfg)
    test = tm["overall"]["macro_f1"]; gap = best_val - test
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f}")
    return {"tag": tag, "method": "MTAUX", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "a_dist": cfg.a_dist, "b_sib": cfg.b_sib, "c_decode": cfg.c_decode,
            "val_macro": best_val, "macro": test,
            "weighted": tm["overall"]["weighted_f1"], "acc": tm["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "test_metrics": tm, "val_history": vh,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp96_mtaux_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time()-t0, 1); results.append(r)
                logger.info(f"[{tag}] test={r['macro']:.4f} ({r['dpaper']:+.4f}) gap={r['val_test_gap']:+.4f} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp96_mtaux_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
