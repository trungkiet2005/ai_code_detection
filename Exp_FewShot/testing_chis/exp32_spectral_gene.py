"""
================================================================================
Theory-Track exp -- Spectral Genealogy Embedding (SGE):
Laplacian eigenvectors of the genealogy graph as target coordinate system.

ARXIV_ID      : Belkin 2003 Laplacian Eigenmaps; Von Luxburg 2007 spectral
                clustering tutorial; Spielman 2012 spectral graph theory.
NAME          : Spectral Genealogy Embedding (SGE)
ONE-LINE CLAIM: The eigenvectors of the genealogy graph Laplacian provide a smooth,
                low-dimensional target coordinate system where sibling generators
                are naturally close — aligning embeddings to this basis gives a
                continuous relaxation of the discrete tree prior.
EQUATION      : L_gene = D - A  (graph Laplacian of genealogy tree)
                {v_1, ..., v_K} = eigenvectors of L_gene sorted by eigenvalue
                L_sge = ||Z·Z^T - V·V^T||_F  (Frobenius alignment)
PROPERTY      : The Fiedler vector (2nd eigenvector) of L_gene naturally cuts
                the tree into families. Higher eigenvectors refine within families.
                This spectral basis is smoother than one-hot and richer than HIER_FAM.
WHY NOT BEFORE: Graph Laplacian spectral methods have been applied to DATA graphs
                (GNN, spectral clustering) but never to the LABEL structure graph.
                Our label graph IS the genealogy tree — its spectrum IS the attribution basis.
FALSIFIER     : If spectral embedding alignment does not improve family-group
                accuracy over flat one-hot, the spectral structure is not informative.
================================================================================

exp32_spectral_gene.py — Spectral Genealogy Embedding for few-shot AI-code attribution.
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
logger = logging.getLogger("exp32")

HIER_FAM = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
PAPER_BASELINE = 0.6633

# =============================================================================
# SPECTRAL GENEALOGY: Build Laplacian eigenvectors from tree
# =============================================================================
def _build_spectral_target(n_cls, hier_fam=None):
    """Build spectral target from genealogy graph Laplacian.
    
    Returns V (K, K) matrix of Laplacian eigenvectors as target coordinates.
    The Gram matrix V·V^T encodes genealogical similarity.
    """
    # Build adjacency matrix from genealogy
    A = torch.zeros(n_cls, n_cls)
    if hier_fam is not None and n_cls == 6:
        for i in range(n_cls):
            for j in range(n_cls):
                if i == j: continue
                if hier_fam.get(i) == hier_fam.get(j):
                    A[i, j] = 1.0   # same family = strong edge
                elif (i == 0) != (j == 0):
                    A[i, j] = 0.1   # human vs AI = weak edge
                else:
                    A[i, j] = 0.5   # cross-family AI = medium edge
    else:
        # For AICD-T2 (12 classes): simple nearest-neighbor graph
        A = 0.5 * torch.ones(n_cls, n_cls)
        A.fill_diagonal_(0)
    
    # Degree matrix and normalized Laplacian
    D = torch.diag(A.sum(dim=1))
    D_inv_sqrt = torch.diag(1.0 / (A.sum(dim=1).sqrt() + 1e-8))
    L_norm = torch.eye(n_cls) - D_inv_sqrt @ A @ D_inv_sqrt
    
    # Eigendecomposition
    eigenvalues, eigenvectors = torch.linalg.eigh(L_norm)
    
    # V = eigenvectors (columns sorted by eigenvalue, ascending)
    # The Fiedler vector (2nd column) naturally separates families
    logger.info(f"[SGE] Laplacian eigenvalues: {eigenvalues.tolist()}")
    
    return eigenvectors  # (K, K)


# === Standard data loading infrastructure ===
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
    sge_weight: float = 0.3  # Weight for spectral alignment loss

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.n_cls = 6 if self.task == "author" else 2
        elif self.benchmark == "aicd_t2":
            self.n_cls = 12; self.task = "t2"
        elif self.benchmark in ("droid_t3",): self.n_cls = 3; self.task = "t3"
        elif self.benchmark == "droid_t4": self.n_cls = 4; self.task = "t4"

def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True
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
    names = {str(r.get("model","") or "").strip() for r in train if not _is_human(r.get("target","")) and r.get("model","")}
    return {n:i+1 for i,n in enumerate(sorted(names))}
def _conv_codet(split, task, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code","code"):
            v = r.get(f,""); 
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
        s = ds.train_test_split(test_size=0.1, seed=42); s2 = s["train"].train_test_split(test_size=1/9, seed=42)
        return s2["train"], s2["test"], s["test"]
    return tr, vl, ts
def _load_aicd(task):
    task_map = {"t1": "T1", "t2": "T2", "t3": "T3"}
    task_name = task_map.get(task.lower(), None)
    if task_name is None: raise ValueError(f"[aicd] Unknown task '{task}'.")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path): raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found at {task_path}.")
    pf = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: No parquet in {task_path}.")
    logger.info(f"[aicd] Loading {task_name} from {task_path} ({len(pf)} files)")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        try:
            tr = ds.filter(lambda x: str(x.get("split","")).lower()=="train")
            vl = ds.filter(lambda x: str(x.get("split","")).lower() in {"val","validation","dev"})
            ts = ds.filter(lambda x: str(x.get("split","")).lower()=="test")
            if len(tr)>0 and len(vl)>0 and len(ts)>0: return tr, vl, ts
        except: pass
    s = ds.train_test_split(test_size=0.1, seed=42); s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]

def _preflight_check():
    logger.info("="*60); logger.info("[PREFLIGHT] Validating..."); all_ok = True
    for bn, ta in [("codet_m4", None), ("aicd_t2", "t2")]:
        try:
            if ta is None: tr,_,_ = _load_codet(); td = _conv_codet(tr, "author", _vocab(tr))
            else: tr,_,_ = _load_aicd(ta); td = _conv_aicd(tr)
            logger.info(f"[PREFLIGHT] {bn}: {len(td):,}")
            if len(td)==0: all_ok = False
        except Exception as e: logger.error(f"[PREFLIGHT] ❌ {bn}: {e}"); all_ok = False
    if not all_ok: raise RuntimeError("[PREFLIGHT] Failed.")
    logger.info("[PREFLIGHT] ✅ OK")

class FSDS(TD):
    def __init__(self, hf, tok, ml): self.hf=hf; self.tok=tok; self.ml=ml
    def __len__(self): return len(self.hf)
    def __getitem__(self, i):
        r = self.hf[i]; enc = self.tok(r["code"], max_length=self.ml, truncation=True, padding="max_length", return_tensors="pt")
        return {"ids":enc["input_ids"].squeeze(0),"mask":enc["attention_mask"].squeeze(0),"label":int(r["label"])}
def collate(b):
    return {"ids":torch.stack([x["ids"] for x in b]),"mask":torch.stack([x["mask"] for x in b]),
            "labels":torch.tensor([x["label"] for x in b], dtype=torch.long)}

def build_dls(cfg):
    set_seed(cfg.seed); tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    if cfg.benchmark == "codet_m4":
        tr_r,vl_r,ts_r = _load_codet(); v = _vocab(tr_r) if cfg.task=="author" else {}
        tr_d,vl_d,ts_d = _conv_codet(tr_r,cfg.task,v),_conv_codet(vl_r,cfg.task,v),_conv_codet(ts_r,cfg.task,v)
    else:
        tr_r,vl_r,ts_r = _load_aicd(cfg.task); tr_d,vl_d,ts_d = _conv_aicd(tr_r),_conv_aicd(vl_r),_conv_aicd(ts_r)
    by_cls = defaultdict(list)
    for i,lab in enumerate(tr_d["label"]): by_cls[int(lab)].append(i)
    rng = random.Random(cfg.seed); chosen = []
    for c in range(cfg.n_cls):
        pool = by_cls.get(c,[]); n = max(1,int(round(len(pool)*cfg.frac))) if pool else 0
        chosen.extend(rng.sample(pool, min(n,len(pool))) if pool else [])
    logger.info(f"[data] {cfg.enc}|{cfg.benchmark}|frac={cfg.frac}|n={len(chosen)}")
    rng.shuffle(chosen); tr_d = tr_d.select(chosen)
    def ld(ds,sh): return DataLoader(FSDS(ds,tok,cfg.seq),batch_size=cfg.bs,shuffle=sh,num_workers=4,collate_fn=collate,pin_memory=True)
    return ld(tr_d,True), ld(vl_d,False), ld(ts_d,False)

# =============================================================================
# MODEL
# =============================================================================
class SGENet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        enc_path = os.path.join(KAGGLE_MODELS, cfg.enc)
        self.enc = AutoModel.from_pretrained(enc_path, local_files_only=True)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.proj = nn.Linear(h, cfg.n_cls)  # Project to spectral dimension
        self.head = nn.Linear(h, cfg.n_cls)

    def forward(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask)
        emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z_spectral = self.proj(emb)  # (batch, K) — spectral coordinates
        logits = self.head(self.drop(emb))
        return {"logits": logits, "emb": emb, "z_spectral": z_spectral}

    def groups(self):
        return [{"params": self.enc.parameters(), "lr": self.cfg.lr_enc, "weight_decay": self.cfg.wd},
                {"params": list(self.proj.parameters()) + list(self.head.parameters()),
                 "lr": self.cfg.lr_head, "weight_decay": self.cfg.wd}]

def class_w(loader, n):
    c = np.zeros(n)
    for b in loader:
        for l in b["labels"].tolist(): c[l] += 1
    c = np.maximum(c,1); w = 1.0/c; return torch.tensor(w/w.sum()*n, dtype=torch.float32)

@torch.no_grad()
def eval_m(model, loader, dev):
    model.eval(); ps, ls = [], []
    for b in loader:
        with _autocast_ctx(dev): logits = model(b["ids"].to(dev), b["mask"].to(dev))["logits"]
        ps.extend(logits.argmax(1).cpu().tolist()); ls.extend(b["labels"].tolist())
    return {"macro":f1_score(ls,ps,average="macro",zero_division=0),
            "weighted":f1_score(ls,ps,average="weighted",zero_division=0),
            "acc":accuracy_score(ls,ps),
            "per_class_f1":f1_score(ls,ps,average=None,zero_division=0).tolist()}

def train(cfg, tr_dl, vl_dl, ts_dl):
    dev = torch.device(cfg.device)
    model = SGENet(cfg).to(dev)
    w = class_w(tr_dl, cfg.n_cls).to(dev)
    opt = torch.optim.AdamW(model.groups())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr_enc, cfg.lr_head],
        steps_per_epoch=len(tr_dl), epochs=cfg.epochs, pct_start=cfg.warmup)
    scaler = GradScaler(enabled=(dev.type=="cuda"))
    best_val, best_state = 0, None

    # Build spectral target
    hier = HIER_FAM if cfg.benchmark == "codet_m4" else None
    V = _build_spectral_target(cfg.n_cls, hier).to(dev)  # (K, K) eigenvectors
    target_gram = V @ V.T  # (K, K) spectral Gram matrix
    logger.info(f"[SGE] Target Gram matrix diagonal: {target_gram.diag().tolist()}")

    train_history, val_history = [], []
    for ep in range(cfg.epochs):
        model.train(); total_loss = 0; n_steps = 0
        pbar = tqdm(tr_dl, desc=f"Ep {ep+1}/{cfg.epochs}", leave=False)
        for b in pbar:
            ids, mask, labs = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)
            with _autocast_ctx(dev):
                out = model(ids, mask)
                ce_loss = F.cross_entropy(out["logits"], labs, weight=w)
                # SGE: align embedding Gram matrix to spectral target
                z = F.normalize(out["z_spectral"], dim=-1)  # (batch, K)
                gram = z @ z.T  # (batch, batch)
                # Build target Gram from labels
                tgt = target_gram[labs][:, labs]  # (batch, batch)
                sge_loss = F.mse_loss(gram, tgt)
                loss = ce_loss + cfg.sge_weight * sge_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            total_loss += loss.item(); n_steps += 1
            pbar.set_postfix({"loss":f"{loss.item():.4f}","sge":f"{sge_loss.item():.4f}"})
        train_history.append({"epoch":ep+1,"loss":total_loss/max(n_steps,1)})
        vr = eval_m(model, vl_dl, dev)
        val_history.append({"epoch":ep+1,"macro":vr["macro"]})
        if vr["macro"] > best_val:
            best_val = vr["macro"]; best_state = {k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    res = eval_m(model, ts_dl, dev); res["train_history"]=train_history; res["val_history"]=val_history
    return res

def main():
    logger.info("[PREFLIGHT] Running dataset validation..."); _preflight_check()
    encoders = ["unixcoder-base"]; benchmarks = [("codet_m4","author"),("aicd_t2","t2")]; fracs = [0.01,0.05,0.20]
    results = []
    for enc in encoders:
        for bench,task in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench,task=task,enc=enc,frac=frac); cfg = _hw(cfg)
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                tag = f"exp32_sge_{enc}_{bench}_f{frac}"; logger.info(f"=== {tag} ===")
                t0 = time.time(); tr_dl,vl_dl,ts_dl = build_dls(cfg)
                res = train(cfg, tr_dl, vl_dl, ts_dl); elapsed = time.time() - t0
                row = {"tag":tag,"enc":enc,"bench":bench,"frac":frac,
                       "macro":round(res["macro"],6),"weighted":round(res["weighted"],6),
                       "acc":round(res["acc"],6),"dpaper":round(res["macro"]-PAPER_BASELINE,6),
                       "per_class_f1":res["per_class_f1"],
                       "train_history":res["train_history"],"val_history":res["val_history"],
                       "wall":round(elapsed,1),"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
                results.append(row)
                logger.info(f"[{tag}] MacroF1={res['macro']:.4f} ({res['macro']-PAPER_BASELINE:+.4f}) time={elapsed:.0f}s")
                del tr_dl,vl_dl,ts_dl; import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    os.makedirs("results", exist_ok=True)
    with open("results/exp32_sge_results.json","w") as f: json.dump(results,f,indent=2)
    print("\n"+"="*100)
    print(f"{'Enc':<22} {'Bench':<12} {'Frac':>6} {'Macro':>10} {'dPaper':>10} {'W-F1':>10} {'Wall':>8}")
    print("-"*100)
    for r in results: print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['macro']:>10.4f} {r['dpaper']:>+10.4f} {r['weighted']:>10.4f} {r['wall']:>8.0f}s")
    print("="*100+"\n[OK] SGE done!")

if __name__ == "__main__": main()
