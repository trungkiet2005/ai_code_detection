# exp99 — UNCERTTRACO: Uncertainty-aware selective prediction
# =============================================================================
# NAME       : UNCERTTRACO (Uncertainty-aware TRACO with selective prediction)
# REFERENCE  : new; combines MC-dropout (Gal & Ghahramani 2016) and selective
#              prediction (Geifman 2017) with TRACO for deployable attribution.
# CLAIM      : The headline Macro-F1 saturates near 0.488 on AICD 20pct (DTKE).
#              For deployment a calibrated REJECTION OPTION is more useful
#              than a marginally higher F1. We train TRACO with MC-dropout
#              enabled at test, run K=20 stochastic forward passes, and
#              report (Coverage, Risk) curves with the standard selective-
#              prediction metric Coverage@5%Risk.
# EQUATION   : For each test sample, run K stochastic forward passes ->
#              ensemble probability p_bar; predictive entropy H = -sum p log p;
#              accept if H < tau, else REJECT.
#              Metrics: Coverage(tau), Risk(tau) = error-rate on accepted.
#              Report: Macro-F1 at 100% coverage (vanilla TRACO),
#                       Coverage at 5% error-rate (selective),
#                       AUC of risk-coverage curve.
# WHY NEW    : No published code-attribution paper reports selective
#              prediction metrics. This is the deployment story missing
#              from the saturation discussion in §7.
# FALSIFIER  : Coverage@5%Risk > 0.50 at 20pct AICD-T2 means we can serve
#              >50% of inputs with only 5% error -- the saturated band is
#              irrelevant for deployment if we abstain on the uncertain ones.
# =============================================================================
from __future__ import annotations
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"
import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from dataclasses import dataclass, field
from typing import Tuple, Dict
import re as _re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
for _p in ("numpy","torch","datasets","transformers","scikit-learn","tqdm"): _ensure(_p)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp99_uncerttraco")

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

_RESERVED = {"if","else","elif","for","while","do","return","def","function","class",
    "struct","interface","enum","import","from","include","public","private","void",
    "new","this","extends","implements","try","catch","except","finally","with","in",
    "of","is","not","and","or","as","True","False","None","null","true","false","self",
    "int","float","double","char","long","bool","string","var","let","const"}
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
    chosen = rng.sample(ids, min(max_n, len(ids))); new = code
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
    try: return _AUG[rng.randrange(len(_AUG))](code, rng)
    except Exception: return code


class UncertTracoModel(nn.Module):
    """TRACO encoder with HIGHER dropout in the projector (0.3 instead of 0.1)
    so MC-dropout at test produces meaningful uncertainty."""
    def __init__(self, enc_name, n_cls, emb_dim=256, drop=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(drop), nn.Linear(512, emb_dim), nn.Dropout(drop))
        self.clf = nn.Linear(emb_dim, n_cls)
        self.emb_dim, self.n_cls = emb_dim, n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1), self.clf(z)

    @torch.no_grad()
    def mc_predict(self, ids, mask, K: int = 20):
        """Run K stochastic forward passes (dropout ENABLED). Return mean
        softmax and predictive entropy."""
        # Enable dropout layers even in eval()
        for m in self.modules():
            if isinstance(m, nn.Dropout): m.train()
        probs = []
        for _ in range(K):
            _, lg = self.encode(ids, mask)
            probs.append(F.softmax(lg.float(), dim=-1))
        # Restore eval mode
        self.eval()
        P = torch.stack(probs, dim=0)               # K, B, C
        p_bar = P.mean(dim=0)                        # B, C
        entropy = -(p_bar * torch.log(p_bar.clamp(min=1e-12))).sum(-1)
        return p_bar, entropy


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


def coverage_at_risk(entropy: np.ndarray, correct: np.ndarray, risk_target: float = 0.05):
    """Find the entropy threshold that yields at most `risk_target` error
    on accepted samples, and return the coverage (fraction accepted)."""
    order = np.argsort(entropy)
    correct_sorted = correct[order]
    cum_correct = np.cumsum(correct_sorted)
    cum_total = np.arange(1, len(correct_sorted) + 1)
    risk_curve = 1.0 - cum_correct / cum_total
    # Find largest coverage point where risk <= target.
    mask = risk_curve <= risk_target
    if mask.sum() == 0: return 0.0
    last_ok = int(np.argmax(np.cumsum(mask)))
    coverage = (last_ok + 1) / len(correct_sorted)
    return float(coverage)


@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    drop: float = 0.3; K_mc: int = 20
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
        if mem >= 40: c.bs, c.seq = 128, 512
        elif mem >= 20: c.bs, c.seq = 96, 448
        elif mem >= 10: c.bs, c.seq = 64, 384
        else: c.bs, c.seq = 32, 256
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
        return {"code": code, "label": lbl}
    return s.map(row, remove_columns=s.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)
def _conv_aicd(s):
    def row(r): return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1))}
    return s.map(row, remove_columns=s.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)
def _load_codet():
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42); s2 = s["train"].train_test_split(test_size=1/9, seed=42)
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
    s = ds.train_test_split(test_size=0.1, seed=42); s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


class FSDS(TD):
    def __init__(self, data, tok, seq, frac=1.0, seed=42, do_aug=True):
        self.data = data; self.tok = tok; self.seq = seq; self.do_aug = do_aug; self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed); labels = list(range(max(self.data["label"]) + 1)); keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
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
        return {"ids0": ids0, "mask0": m0, "ids1": ids1, "mask1": m1, "label": r["label"]}


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
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = UncertTracoModel(cfg.enc, cfg.n_cls, cfg.emb_dim, drop=cfg.drop).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh = 0.0, None, []
    for ep in range(cfg.epochs):
        loss, ce, sc = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist)
        model.eval()
        with torch.no_grad():
            preds, labs_v = [], []
            for b in vl_dl:
                ids = b["ids0"].to(cfg.device); m = b["mask0"].to(cfg.device)
                _, lg = model.encode(ids, m)
                preds.extend(lg.argmax(-1).cpu().tolist())
                labs_v.extend(b["label"].tolist() if torch.is_tensor(b["label"]) else list(b["label"]))
        v = float(f1_score(labs_v, preds, average="macro", zero_division=0)); vh.append(v)
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    # ---- TEST with MC-dropout selective prediction ----
    model.eval()
    all_pbar = []; all_entropy = []; all_labels = []
    with torch.no_grad():
        for b in tqdm(ts_dl, desc="MC-Eval"):
            ids = b["ids0"].to(cfg.device); m = b["mask0"].to(cfg.device)
            p_bar, ent = model.mc_predict(ids, m, K=cfg.K_mc)
            all_pbar.append(p_bar.cpu().float())
            all_entropy.append(ent.cpu().float())
            all_labels.append(b["label"] if torch.is_tensor(b["label"]) else torch.tensor(list(b["label"])))
    P = torch.cat(all_pbar, 0).numpy(); E = torch.cat(all_entropy, 0).numpy()
    Y = torch.cat(all_labels, 0).numpy()
    preds = P.argmax(axis=-1); correct = (preds == Y).astype(np.int64)
    macro = float(f1_score(Y, preds, average="macro", zero_division=0))
    cov_5 = coverage_at_risk(E, correct, risk_target=0.05)
    cov_10 = coverage_at_risk(E, correct, risk_target=0.10)
    cov_20 = coverage_at_risk(E, correct, risk_target=0.20)
    gap = best_val - macro
    logger.info(f"[final] val={best_val:.4f} test={macro:.4f} gap={gap:+.4f} "
                f"Coverage@5%Risk={cov_5:.3f} @10%={cov_10:.3f} @20%={cov_20:.3f}")
    return {"tag": tag, "method": "UNCERTTRACO", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "drop": cfg.drop, "K_mc": cfg.K_mc,
            "val_macro": best_val, "macro": macro,
            "coverage_at_5pct_risk": cov_5,
            "coverage_at_10pct_risk": cov_10,
            "coverage_at_20pct_risk": cov_20,
            "val_test_gap": gap, "dpaper": macro - PAPER_BASELINE,
            "val_history": vh, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]; results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp99_uncerttraco_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time()-t0, 1); results.append(r)
                logger.info(f"[{tag}] test={r['macro']:.4f} cov5={r['coverage_at_5pct_risk']:.3f} "
                            f"cov10={r['coverage_at_10pct_risk']:.3f} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}"); import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp99_uncerttraco_results.json"), "w") as f: json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
