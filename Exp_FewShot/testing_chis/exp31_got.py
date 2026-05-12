"""
================================================================================
Theory-Track exp -- Genealogical Optimal Transport (GOT):
Wasserstein loss with tree-metric ground cost for attribution.

ARXIV_ID      : Villani 2008 OT theory; Cuturi 2013 Sinkhorn (1306.0895);
                Alvarez-Melis 2018 structured prediction OT (1802.04395)
NAME          : Genealogical Optimal Transport (GOT)
ONE-LINE CLAIM: Using the genealogy tree distance as the ground metric in
                Wasserstein loss makes misclassification cost proportional to
                genealogical proximity — sibling errors cost less than cross-family.
EQUATION      : W_T(p, q) = inf_{γ∈Π(p,q)} Σ_{i,j} γ_{ij} · d_tree(i,j)
                Ground cost: C[i,j] = shortest-path distance on genealogy tree.
PROPERTY      : Standard CE treats all errors equally (cost = 1). GOT's ground
                metric makes the loss landscape genealogy-aware: codellama→nxcode
                (d_tree=2) costs 1/3 of codellama→human (d_tree=6).
WHY NOT BEFORE: OT has been used for domain adaptation (Courty 2017) and
                class-imbalance (Caron 2020), but never with a genealogical
                ground metric for code attribution. The "cost of being wrong"
                is structured by model ancestry — this is new.
FALSIFIER     : If GOT does not reduce sibling-pair confusion rate more than CE,
                the genealogical ground metric adds no signal.
================================================================================

exp31_got.py — Genealogical Optimal Transport for few-shot AI-code attribution.
Protocol: fraction-based (1% / 5% / 20%), unixcoder-base only.
"""
from __future__ import annotations


# === KAGGLE PATHS ===
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_DROID = "/kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/data"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from collections import defaultdict
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

from torch.cuda.amp import autocast, GradScaler

def _autocast_ctx(dev):
    return autocast(enabled=(dev.type == "cuda"))
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp31")

# === CONSTANTS ===
HIER_FAM = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
PAPER_BASELINE = 0.6633

# =============================================================================
# GENEALOGY TREE DISTANCE MATRIX
# =============================================================================
def _build_tree_distance(n_cls, hier_fam=None):
    """Build tree distance matrix from genealogy.
    
    CoDET-M4 tree (6 classes):
        root
        ├── human (0)
        └── AI
            ├── gpt-family: gpt(1), nxcode(4)
            ├── llama3.1 (2)
            ├── codellama (3)
            └── qwen1.5 (5)
    
    For AICD-T2 (12 classes): use flat distance (all distance=2, same=0).
    """
    C = torch.zeros(n_cls, n_cls)
    if hier_fam is not None and n_cls == 6:
        # Build from HIER_FAM: same=0, same-family=2, cross-family=4, human-vs-AI=6
        for i in range(n_cls):
            for j in range(n_cls):
                if i == j:
                    C[i, j] = 0.0
                elif hier_fam.get(i) == hier_fam.get(j):
                    C[i, j] = 2.0  # same family (e.g. gpt <-> nxcode)
                elif (i == 0) != (j == 0):
                    C[i, j] = 6.0  # human vs AI
                else:
                    C[i, j] = 4.0  # cross-family AI
    else:
        # Flat: all different classes have distance 1
        C = 1.0 - torch.eye(n_cls)
    # Normalize to [0, 1]
    if C.max() > 0:
        C = C / C.max()
    return C

# =============================================================================
# SINKHORN DIVERGENCE (differentiable OT)
# =============================================================================
def sinkhorn_loss(log_probs, targets, cost_matrix, eps=0.05, n_iter=50):
    """Compute Sinkhorn divergence between predicted distribution and target.
    
    Args:
        log_probs: (batch, K) log-softmax predictions
        targets: (batch,) integer labels
        cost_matrix: (K, K) ground cost matrix
        eps: entropic regularization
        n_iter: Sinkhorn iterations
    
    Returns:
        Scalar Sinkhorn loss
    """
    K = log_probs.size(1)
    device = log_probs.device
    
    # Predicted probabilities
    pred = log_probs.exp()  # (batch, K)
    
    # Target one-hot
    tgt = F.one_hot(targets, K).float()  # (batch, K)
    
    # Cost matrix
    C = cost_matrix.to(device)  # (K, K)
    
    # Kernel
    K_mat = (-C / eps).exp()  # (K, K)
    
    # Sinkhorn iterations (vectorized over batch)
    # For each sample: compute OT between pred[i] and tgt[i]
    # Simplified: use the expected transport cost
    # W ≈ Σ_j pred[i,j] * C[j, target[i]]
    # This is the "soft" Wasserstein loss
    
    # More principled: use the dual form
    u = torch.ones_like(pred)  # (batch, K)
    
    for _ in range(n_iter):
        v = tgt / (u @ K_mat + 1e-8)  # (batch, K)
        u = pred / (v @ K_mat.T + 1e-8)  # (batch, K)
    
    # Transport plan
    pi = u.unsqueeze(2) * K_mat.unsqueeze(0) * v.unsqueeze(1)  # (batch, K, K)
    
    # Transport cost
    loss = (pi * C.unsqueeze(0)).sum(dim=(1, 2)).mean()
    
    return loss


# === DATA LOADING (same as exp_n09) ===
@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    enc: str = "unixcoder-base"
    frac: float = 0.05
    n_cls: int = 6
    seed: int = 42
    bs: int = 256
    seq: int = 512
    epochs: int = 3
    lr_enc: float = 2e-5
    lr_head: float = 1e-4
    wd: float = 0.01
    warmup: float = 0.1
    device: str = "cuda"
    # GOT specific
    got_weight: float = 0.3       # Weight of Sinkhorn loss
    sinkhorn_eps: float = 0.05    # Entropic regularization
    sinkhorn_iter: int = 50       # Sinkhorn iterations

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.n_cls = 6 if self.task == "author" else 2
        elif self.benchmark == "aicd_t2":
            self.n_cls = 12; self.task = "t2"
        elif self.benchmark in ("droid_t3",):
            self.n_cls = 3; self.task = "t3"
        elif self.benchmark == "droid_t4":
            self.n_cls = 4; self.task = "t4"

def _hw(cfg: Cfg) -> Cfg:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40: cfg.bs, cfg.seq = 256, 512
        elif mem >= 10: cfg.bs, cfg.seq = 128, 384
        else: cfg.bs, cfg.seq = 64, 256
    return cfg

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _is_human(t): return str(t or "").strip().lower() in {"human","human_written","human-generated"}
def _vocab(train):
    names = {str(r.get("model","") or "").strip() for r in train
             if not _is_human(r.get("target","")) and r.get("model","")}
    return {n:i+1 for i,n in enumerate(sorted(names))}

def _conv_codet(split, task, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code","code"):
            v = r.get(f,"")
            if isinstance(v,str) and v.strip(): code = v; break
        if task == "binary": label = 0 if _is_human(r.get("target","")) else 1
        else:
            if _is_human(r.get("target","")): label = 0
            else: label = vocab.get(str(r.get("model","") or "").strip(), -1)
        return {"code":code,"label":label,"lang":str(r.get("language","")).strip().lower()}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"]>=0 and len(x["code"].strip())>0)

def _conv_aicd(split):
    def row(r): return {"code":str(r.get("code","")).strip(),"label":int(r.get("label",-1)),"lang":str(r.get("language","")).strip().lower()}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"]>=0 and len(x["code"].strip())>0)

def _load_codet():
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split","")).lower()=="train")
        vl = ds.filter(lambda x: str(x.get("split","")).lower() in {"val","validation","dev"})
        ts = ds.filter(lambda x: str(x.get("split","")).lower()=="test")
    else:
        s = ds.train_test_split(test_size=0.1, seed=42)
        s2 = s["train"].train_test_split(test_size=1/9, seed=42)
        return s2["train"], s2["test"], s["test"]
    return tr, vl, ts

def _load_aicd(task):
    """Load AICD-Bench -- STRICT: only loads the requested task dir, NO fallback."""
    task_map = {"t1": "T1", "t2": "T2", "t3": "T3"}
    task_name = task_map.get(task.lower(), None)
    if task_name is None:
        raise ValueError(f"[aicd] Unknown task '{task}'. Must be one of: t1, t2, t3.")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(f"[aicd] STRICT: {task_name} dir not found at {task_path}. NO fallback.")
    parquet_files = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"[aicd] STRICT: No parquet files in {task_path}. NO fallback.")
    logger.info(f"[aicd] Loading {task_name} from {task_path} ({len(parquet_files)} files)")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    if "split" in ds.column_names:
        try:
            tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
            vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
            ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
            if len(tr) > 0 and len(vl) > 0 and len(ts) > 0:
                return tr, vl, ts
        except Exception: pass
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]

def _preflight_check():
    logger.info("=" * 60)
    logger.info("[PREFLIGHT] Starting data validation...")
    all_ok = True
    for bench_name, task_arg in [("codet_m4", None), ("aicd_t2", "t2")]:
        try:
            if task_arg is None:
                tr, vl, ts = _load_codet()
                vocab = _vocab(tr)
                tr_d = _conv_codet(tr, "author", vocab)
            else:
                tr, vl, ts = _load_aicd(task_arg)
                tr_d = _conv_aicd(tr)
            logger.info(f"[PREFLIGHT] {bench_name}: Train={len(tr_d):,}")
            if len(tr_d) == 0: all_ok = False
        except Exception as e:
            logger.error(f"[PREFLIGHT] ❌ {bench_name}: {e}"); all_ok = False
    if not all_ok: raise RuntimeError("[PREFLIGHT] Dataset validation failed.")
    logger.info("[PREFLIGHT] ✅ All datasets OK")

class FSDS(TD):
    def __init__(self, hf, tok, max_len):
        self.hf = hf; self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.hf)
    def __getitem__(self, i):
        r = self.hf[i]
        enc = self.tok(r["code"], max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        return {"ids":enc["input_ids"].squeeze(0),"mask":enc["attention_mask"].squeeze(0),"label":int(r["label"])}

def collate(b):
    return {"ids":torch.stack([x["ids"] for x in b]),"mask":torch.stack([x["mask"] for x in b]),
            "labels":torch.tensor([x["label"] for x in b], dtype=torch.long)}

def build_dls(cfg: Cfg):
    set_seed(cfg.seed)
    enc_path = os.path.join(KAGGLE_MODELS, cfg.enc)
    tok = AutoTokenizer.from_pretrained(enc_path, local_files_only=True)
    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw) if cfg.task == "author" else {}
        tr_d = _conv_codet(tr_raw, cfg.task, vocab)
        vl_d = _conv_codet(vl_raw, cfg.task, vocab)
        ts_d = _conv_codet(ts_raw, cfg.task, vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd(cfg.task)
        tr_d = _conv_aicd(tr_raw); vl_d = _conv_aicd(vl_raw); ts_d = _conv_aicd(ts_raw)

    by_cls = defaultdict(list)
    for i, lab in enumerate(tr_d["label"]): by_cls[int(lab)].append(i)
    rng = random.Random(cfg.seed)
    chosen = []
    for cls in range(cfg.n_cls):
        pool = by_cls.get(cls, [])
        n = max(1, int(round(len(pool) * cfg.frac))) if pool else 0
        chosen.extend(rng.sample(pool, min(n, len(pool))) if pool else [])
    logger.info(f"[data] {cfg.enc} | {cfg.benchmark} | frac={cfg.frac} | n_train={len(chosen)}")
    rng.shuffle(chosen)
    tr_d = tr_d.select(chosen)
    def ld(ds, shuf):
        return DataLoader(FSDS(ds, tok, cfg.seq), batch_size=cfg.bs, shuffle=shuf, num_workers=4, collate_fn=collate, pin_memory=True)
    return ld(tr_d, True), ld(vl_d, False), ld(ts_d, False)

# =============================================================================
# MODEL: Standard encoder + linear head (novel part is in the LOSS)
# =============================================================================
class GOTNet(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        enc_path = os.path.join(KAGGLE_MODELS, cfg.enc)
        self.enc = AutoModel.from_pretrained(enc_path, local_files_only=True)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.head = nn.Linear(h, cfg.n_cls)

    def forward(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask)
        emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        logits = self.head(self.drop(emb))
        return {"logits": logits, "emb": emb}

    def groups(self):
        return [{"params": self.enc.parameters(), "lr": self.cfg.lr_enc, "weight_decay": self.cfg.wd},
                {"params": self.head.parameters(), "lr": self.cfg.lr_head, "weight_decay": self.cfg.wd}]

def class_w(loader, n):
    c = np.zeros(n)
    for b in loader:
        for l in b["labels"].tolist(): c[l] += 1
    c = np.maximum(c, 1); w = 1.0 / c
    return torch.tensor(w/w.sum()*n, dtype=torch.float32)

@torch.no_grad()
def eval_m(model, loader, dev):
    model.eval(); ps, ls = [], []
    for b in loader:
        with _autocast_ctx(dev): logits = model(b["ids"].to(dev), b["mask"].to(dev))["logits"]
        ps.extend(logits.argmax(1).cpu().tolist()); ls.extend(b["labels"].tolist())
    return {"macro": f1_score(ls, ps, average="macro", zero_division=0),
            "weighted": f1_score(ls, ps, average="weighted", zero_division=0),
            "acc": accuracy_score(ls, ps),
            "per_class_f1": f1_score(ls, ps, average=None, zero_division=0).tolist()}

def train(cfg: Cfg, tr_dl, vl_dl, ts_dl):
    dev = torch.device(cfg.device)
    model = GOTNet(cfg).to(dev)
    w = class_w(tr_dl, cfg.n_cls).to(dev)
    opt = torch.optim.AdamW(model.groups())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr_enc, cfg.lr_head],
        steps_per_epoch=len(tr_dl), epochs=cfg.epochs, pct_start=cfg.warmup)
    scaler = GradScaler(enabled=(dev.type == "cuda"))
    best_val, best_state = 0, None

    # Build tree distance matrix
    hier = HIER_FAM if cfg.benchmark == "codet_m4" else None
    cost_matrix = _build_tree_distance(cfg.n_cls, hier).to(dev)
    logger.info(f"[GOT] Cost matrix:\n{cost_matrix}")

    train_history, val_history = [], []

    for ep in range(cfg.epochs):
        model.train(); total_loss = 0; n_steps = 0
        pbar = tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.epochs}", leave=False)
        for b in pbar:
            ids, mask, labs = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)
            with _autocast_ctx(dev):
                out = model(ids, mask)
                logits = out["logits"]
                ce_loss = F.cross_entropy(logits, labs, weight=w)
                # GOT: Sinkhorn loss with tree-metric ground cost
                log_probs = F.log_softmax(logits, dim=-1)
                got_loss = sinkhorn_loss(log_probs, labs, cost_matrix, 
                                         eps=cfg.sinkhorn_eps, n_iter=cfg.sinkhorn_iter)
                loss = ce_loss + cfg.got_weight * got_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            total_loss += loss.item(); n_steps += 1
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "got": f"{got_loss.item():.4f}"})
        train_history.append({"epoch": ep+1, "loss": total_loss/max(n_steps,1)})
        vr = eval_m(model, vl_dl, dev)
        val_history.append({"epoch": ep+1, "macro": vr["macro"]})
        if vr["macro"] > best_val:
            best_val = vr["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    res = eval_m(model, ts_dl, dev)
    res["train_history"] = train_history
    res["val_history"] = val_history
    return res

def main():
    logger.info("[PREFLIGHT] Running dataset validation...")
    _preflight_check()
    encoders = ["unixcoder-base"]
    benchmarks = [("codet_m4","author"), ("aicd_t2","t2")]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for enc in encoders:
        for bench, task in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac); cfg = _hw(cfg)
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                tag = f"exp31_got_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                tr_dl, vl_dl, ts_dl = build_dls(cfg)
                res = train(cfg, tr_dl, vl_dl, ts_dl)
                elapsed = time.time() - t0
                row = {"tag": tag, "enc": enc, "bench": bench, "frac": frac,
                       "macro": round(res["macro"], 6), "weighted": round(res["weighted"], 6),
                       "acc": round(res["acc"], 6), "dpaper": round(res["macro"] - PAPER_BASELINE, 6),
                       "got_weight": cfg.got_weight, "sinkhorn_eps": cfg.sinkhorn_eps,
                       "per_class_f1": res["per_class_f1"],
                       "train_history": res["train_history"], "val_history": res["val_history"],
                       "wall": round(elapsed, 1), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
                results.append(row)
                logger.info(f"[{tag}] MacroF1={res['macro']:.4f} ({res['macro']-PAPER_BASELINE:+.4f} vs paper) time={elapsed:.0f}s")
                del tr_dl, vl_dl, ts_dl
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/exp31_got_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*100)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Macro-F1':>10} {'dPaper':>10} {'Weighted':>10} {'Wall':>8}")
    print("-"*100)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['macro']:>10.4f} {r['dpaper']:>+10.4f} {r['weighted']:>10.4f} {r['wall']:>8.0f}s")
    print("="*100)
    print("\n[OK] GOT experiments complete!")

if __name__ == "__main__":
    main()
