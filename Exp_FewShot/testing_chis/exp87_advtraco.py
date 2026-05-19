# exp87 — ADVTRACO: Adversarial min-max view-augmentation for code attribution
# =============================================================================
# NAME       : ADVTRACO (Adversarial TRACO)
# REFERENCE  : new; combines Madry-style adversarial training
#              (arXiv:1706.06083) with TRACO's view-augmentation contrastive.
# CLAIM      : Augmentation strength matters, but the RIGHT strength is
#              data-dependent. Tracker insight: CARGO v2's guaranteed-fire
#              ws-normalization fallback REGRESSED below v1 because the view
#              was too gentle, while the original TRACO surface pool
#              over-augments simple snippets. Adversarial selection of the
#              HARDEST view per sample solves both problems jointly.
# EQUATION   : For each sample x, embed phi(x). For each augmentation
#              T in pool, compute z_t = phi(T(x)). Adversary picks the
#              MAX-DISTANCE positive: T*(x) = argmax_T 1 - cos(z, z_t).
#              Final loss: L = L_ce(W z, y) + lambda * SupCon_TW([z; T*(x)]).
#              The encoder learns invariance to the WORST view, not an
#              average view.
# WHY NEW    : View augmentation in TRACO is uniform-random over pool.
#              Adversarial selection is standard in image robustness but
#              has not, to our knowledge, been used inside a tree-weighted
#              code-attribution contrastive setup. Min-max contrastive
#              SimCLR-style exists for images (Kim 2020 advCL) but not for
#              code authorship.
# FALSIFIER  : (i) Macro-F1 strictly > TRACO at extreme few-shot.
#              (ii) Per-sample selected augmentation distribution is
#                  non-uniform (each sample picks the augmentation that
#                  hits its weakest invariance).
#              (iii) Adversarial-view cos < mean-view cos by at least 0.05.
# GPU TUNING : K=4 augmentation candidates per sample. Inner adversary
#              uses no-grad scoring (cheap), only the selected view goes
#              through the gradient-bearing backward pass.
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
for _p in ("numpy", "torch", "datasets", "transformers", "scikit-learn", "tqdm"): _ensure(_p)
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
logger = logging.getLogger("exp87_advtraco")

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

# 4-augmentation pool (same as TRACO)
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
_AUG_FNS = [aug_token_dropout, aug_id_rename, aug_ws_jitter, aug_comment_strip]
_AUG_NAMES = ["tok_drop", "id_rename", "ws_jit", "com_strip"]


class ADVTRACOModel(nn.Module):
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


# Plumbing (compact, shared idiom with exp76)
@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    K_aug: int = 4
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
        # K+1 views per batch in adversary scoring => smaller bs
        if mem >= 40: c.bs, c.seq = 64, 512
        elif mem >= 20: c.bs, c.seq = 48, 448
        elif mem >= 10: c.bs, c.seq = 32, 384
        else: c.bs, c.seq = 16, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={c.bs} seq={c.seq} (K+1 views, K={c.K_aug})")
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


class FSDS_ADV(TD):
    """Dataset returning K=4 candidate augmented views per sample. The
    adversary at training time picks the WORST one (highest distance to
    original embedding)."""
    def __init__(self, data, tok, seq, frac=1.0, seed=42, do_aug=True, K=4):
        self.data = data; self.tok = tok; self.seq = seq; self.do_aug = do_aug
        self.seed = seed; self.K = K
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_ADV] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]
        e0 = self.tok(code, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        ids0, m0 = e0["input_ids"].squeeze(0), e0["attention_mask"].squeeze(0)
        # K candidate views (one per augmentation in the pool)
        ids_views, mask_views = [], []
        rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
        if self.do_aug:
            for k in range(self.K):
                fn = _AUG_FNS[k % len(_AUG_FNS)]
                try: ca = fn(code, rng)
                except Exception: ca = code
                ek = self.tok(ca, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
                ids_views.append(ek["input_ids"].squeeze(0))
                mask_views.append(ek["attention_mask"].squeeze(0))
        else:
            for k in range(self.K):
                ids_views.append(ids0); mask_views.append(m0)
        # Stack views: (K, L)
        ids_v = torch.stack(ids_views, dim=0)
        m_v = torch.stack(mask_views, dim=0)
        return {"ids0": ids0, "mask0": m0, "ids_v": ids_v, "mask_v": m_v,
                "label": r["label"]}


@torch.no_grad()
def eval_macro(model, loader, cfg):
    model.eval(); preds, labels = [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids0"].to(cfg.device); m = b["mask0"].to(cfg.device); labs = b["label"]
        _, lg = model.encode(ids, m)
        preds.extend(lg.argmax(-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
    preds = np.array(preds); labels = np.array(labels); n = cfg.n_cls
    return {"macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
            "accuracy": float(accuracy_score(labels, preds)),
            "confusion_matrix": confusion_matrix(labels, preds, labels=list(range(n))).tolist()}


def train_epoch(model, loader, opt, sch, scaler, cfg, dist):
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    adv_pick_counts = [0, 0, 0, 0]; cos_picked = []
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device)
        ids_v = b["ids_v"].to(cfg.device); m_v = b["mask_v"].to(cfg.device)  # B, K, L
        labs = b["label"].to(cfg.device)
        B, K, L = ids_v.shape
        with torch.no_grad():
            # Score all K views, pick the adversarial worst (lowest cosine to original)
            z0_score, _ = model.encode(ids0, m0)
            ids_flat = ids_v.view(B * K, L); m_flat = m_v.view(B * K, L)
            z_all, _ = model.encode(ids_flat, m_flat)
            z_all = z_all.view(B, K, -1)                        # B, K, D
            sims = (z_all * z0_score.unsqueeze(1)).sum(-1)       # B, K
            worst = sims.argmin(dim=-1)                          # B
            for w in worst.cpu().tolist(): adv_pick_counts[int(w)] += 1
            cos_picked.append(sims.gather(-1, worst.unsqueeze(-1)).squeeze(-1).mean().item())
        # Now do gradient-bearing forward on (original, adversarial-picked view)
        idx = torch.arange(B, device=cfg.device)
        ids1 = ids_v[idx, worst]; m1 = m_v[idx, worst]
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
    return tot/n, ce_s/n, sc_s/n, adv_pick_counts, float(np.mean(cos_picked)) if cos_picked else 0.0


def run_exp(cfg, tag):
    set_seed(cfg.seed); cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
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
    tr_ds = FSDS_ADV(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True, K=cfg.K_aug)
    vl_ds = FSDS_ADV(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False, K=1)
    ts_ds = FSDS_ADV(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False, K=1)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} wu={cfg.warmup} K={cfg.K_aug}")
    lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = ADVTRACOModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh, adv_history = 0.0, None, [], []
    for ep in range(cfg.epochs):
        loss, ce, sc, picks, cos_p = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist)
        vm = eval_macro(model, vl_dl, cfg)
        v = vm["macro_f1"]; vh.append(v); adv_history.append({"picks": picks, "cos_picked": cos_p})
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} val={v:.4f} "
                    f"adv_picks={picks} cos_picked={cos_p:.3f}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    tm = eval_macro(model, ts_dl, cfg)
    test = tm["macro_f1"]; gap = best_val - test
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f}")
    return {"tag": tag, "method": "ADVTRACO", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_aug": cfg.lambda_aug, "gamma": cfg.gamma, "tau": cfg.tau, "K_aug": cfg.K_aug,
            "val_macro": best_val, "macro": test, "weighted": tm["weighted_f1"],
            "acc": tm["accuracy"], "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "test_metrics": tm, "val_history": vh, "adv_history": adv_history,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp87_advtraco_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time()-t0, 1); results.append(r)
                logger.info(f"[{tag}] test={r['macro']:.4f} ({r['dpaper']:+.4f}) gap={r['val_test_gap']:+.4f} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}"); import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp87_advtraco_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
