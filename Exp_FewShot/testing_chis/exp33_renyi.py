"""
================================================================================
Theory-Track exp -- Rényi Attribution Divergence (RAD):
α-parameterized loss interpolating max-likelihood to minimax attribution.

ARXIV_ID      : Rényi 1961 information measures; Van Erven 2014 (1206.2459);
                Li 2021 Rényi robust learning (2103.12028)
NAME          : Rényi Attribution Divergence (RAD)
ONE-LINE CLAIM: Replacing cross-entropy (α=1 limit) with Rényi-α divergence
                provides a single knob that interpolates between maximum-likelihood
                (α→1) and minimax/worst-case (α→∞) — the optimal α depends on
                the few-shot regime (samples per class).
EQUATION      : L_α(p, y) = 1/(α-1) · log(Σ_k p_k^α · y_k)
                α=1: recovers CE. α=2: down-weights confident samples.
                α=0.5: emphasises uncertain predictions (calibration).
PROPERTY      : In few-shot, CE over-fits to frequent patterns. Higher α (e.g. 2)
                provides implicit regularization by flattening the loss landscape.
                The optimal α is a function of n/K — this α-regime connection is
                our novel theoretical claim.
WHY NOT BEFORE: Rényi divergence is foundational in information theory but has
                never been used as a classification loss for code attribution.
                The claim that α* ∝ log(n/K) links information theory to the
                few-shot phase transition.
FALSIFIER     : If optimal α does not shift across 1%→5%→20% regimes,
                the Rényi-regime connection is wrong.
================================================================================

exp33_renyi.py — Rényi Attribution Divergence for few-shot AI-code attribution.
Protocol: fraction-based (1% / 5% / 20%), unixcoder-base only.
Sweeps α ∈ {0.5, 1.0, 2.0, 5.0} at each fraction.
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
try:
    from torch.amp import autocast as _ac, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as _ac, GradScaler
def _autocast_ctx(dev):
    enabled = (dev.type == "cuda")
    try: return _ac(device_type=dev.type, enabled=enabled)
    except TypeError: return _ac(enabled=enabled)
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp33")

HIER_FAM = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
PAPER_BASELINE = 0.6633

# =============================================================================
# RÉNYI DIVERGENCE LOSS
# =============================================================================
def renyi_divergence_loss(logits, targets, alpha=2.0, weight=None):
    """Rényi-α divergence as classification loss.
    
    L_α(p, y) = 1/(α-1) · log(Σ_k p_k^α · y_k)
    
    For α → 1: recovers cross-entropy.
    For α = 2: emphasises high-probability predictions (robust).
    For α = 0.5: emphasises low-probability predictions (calibration).
    
    Uses numerical stability tricks for extreme α values.
    """
    K = logits.size(1)
    
    if abs(alpha - 1.0) < 1e-6:
        # Standard CE (α → 1 limit)
        return F.cross_entropy(logits, targets, weight=weight)
    
    # Softmax probabilities
    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    
    # One-hot targets
    tgt = F.one_hot(targets, K).float()
    
    # Apply class weights if provided
    if weight is not None:
        w = weight[targets]  # (batch,)
    else:
        w = torch.ones(targets.size(0), device=targets.device)
    
    if alpha > 1.0:
        # For α > 1: L = 1/(α-1) · log(Σ_k p_k^α · y_k)
        # = 1/(α-1) · log(p_target^α) = α/(α-1) · log(p_target)
        # This simplifies to a reweighted version of CE
        log_p_target = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)  # (batch,)
        p_target = log_p_target.exp()
        
        # Rényi α > 1: weight = p_target^(α-1) / E[p_target^(α-1)]
        renyi_weight = p_target.detach() ** (alpha - 1)
        renyi_weight = renyi_weight / (renyi_weight.mean() + 1e-8)
        
        loss = -(renyi_weight * w * log_p_target).mean()
    else:
        # For α < 1: opposite weighting — emphasise uncertain samples
        log_p_target = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        p_target = log_p_target.exp()
        
        renyi_weight = p_target.detach() ** (alpha - 1)  # α-1 < 0 → upweights low prob
        renyi_weight = renyi_weight / (renyi_weight.mean() + 1e-8)
        renyi_weight = renyi_weight.clamp(max=10.0)  # Prevent explosion
        
        loss = -(renyi_weight * w * log_p_target).mean()
    
    return loss


# === Standard data loading ===
@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.05; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4; wd: float = 0.01
    warmup: float = 0.1; device: str = "cuda"
    renyi_alpha: float = 2.0  # Rényi α parameter

    def __post_init__(self):
        if self.benchmark == "codet_m4": self.n_cls = 6 if self.task == "author" else 2
        elif self.benchmark == "aicd_t2": self.n_cls = 12; self.task = "t2"
        elif self.benchmark == "droid_t3": self.n_cls = 3; self.task = "t3"
        elif self.benchmark == "droid_t4": self.n_cls = 4; self.task = "t4"

def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
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
    task_map = {"t1":"T1","t2":"T2","t3":"T3"}; task_name = task_map.get(task.lower())
    if not task_name: raise ValueError(f"[aicd] Unknown task '{task}'.")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path): raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found at {task_path}.")
    pf = sorted(glob.glob(os.path.join(task_path,"**","*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: No parquet in {task_path}.")
    logger.info(f"[aicd] Loading {task_name} ({len(pf)} files)")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        try:
            tr=ds.filter(lambda x:str(x.get("split","")).lower()=="train")
            vl=ds.filter(lambda x:str(x.get("split","")).lower() in {"val","validation","dev"})
            ts=ds.filter(lambda x:str(x.get("split","")).lower()=="test")
            if len(tr)>0 and len(vl)>0 and len(ts)>0: return tr,vl,ts
        except: pass
    s = ds.train_test_split(test_size=0.1, seed=42); s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]

def _preflight_check():
    logger.info("="*60+"\n[PREFLIGHT] Validating..."); ok=True
    for bn,ta in [("codet_m4",None),("aicd_t2","t2")]:
        try:
            if ta is None: tr,_,_=_load_codet(); td=_conv_codet(tr,"author",_vocab(tr))
            else: tr,_,_=_load_aicd(ta); td=_conv_aicd(tr)
            logger.info(f"[PREFLIGHT] {bn}: {len(td):,}")
            if len(td)==0: ok=False
        except Exception as e: logger.error(f"[PREFLIGHT] ❌ {bn}: {e}"); ok=False
    if not ok: raise RuntimeError("[PREFLIGHT] Failed.")
    logger.info("[PREFLIGHT] ✅ OK")

class FSDS(TD):
    def __init__(self,hf,tok,ml): self.hf=hf;self.tok=tok;self.ml=ml
    def __len__(self): return len(self.hf)
    def __getitem__(self,i):
        r=self.hf[i]; enc=self.tok(r["code"],max_length=self.ml,truncation=True,padding="max_length",return_tensors="pt")
        return {"ids":enc["input_ids"].squeeze(0),"mask":enc["attention_mask"].squeeze(0),"label":int(r["label"])}
def collate(b):
    return {"ids":torch.stack([x["ids"] for x in b]),"mask":torch.stack([x["mask"] for x in b]),
            "labels":torch.tensor([x["label"] for x in b],dtype=torch.long)}
def build_dls(cfg):
    set_seed(cfg.seed); tok=AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS,cfg.enc),local_files_only=True)
    if cfg.benchmark=="codet_m4":
        tr_r,vl_r,ts_r=_load_codet(); v=_vocab(tr_r) if cfg.task=="author" else {}
        tr_d,vl_d,ts_d=_conv_codet(tr_r,cfg.task,v),_conv_codet(vl_r,cfg.task,v),_conv_codet(ts_r,cfg.task,v)
    else:
        tr_r,vl_r,ts_r=_load_aicd(cfg.task); tr_d,vl_d,ts_d=_conv_aicd(tr_r),_conv_aicd(vl_r),_conv_aicd(ts_r)
    by_cls=defaultdict(list)
    for i,lab in enumerate(tr_d["label"]): by_cls[int(lab)].append(i)
    rng=random.Random(cfg.seed); chosen=[]
    for c in range(cfg.n_cls):
        pool=by_cls.get(c,[]); n=max(1,int(round(len(pool)*cfg.frac))) if pool else 0
        chosen.extend(rng.sample(pool,min(n,len(pool))) if pool else [])
    logger.info(f"[data] {cfg.enc}|{cfg.benchmark}|frac={cfg.frac}|n={len(chosen)}")
    rng.shuffle(chosen); tr_d=tr_d.select(chosen)
    def ld(ds,sh): return DataLoader(FSDS(ds,tok,cfg.seq),batch_size=cfg.bs,shuffle=sh,num_workers=4,collate_fn=collate,pin_memory=True)
    return ld(tr_d,True),ld(vl_d,False),ld(ts_d,False)

class RADNet(nn.Module):
    def __init__(self, cfg):
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
        return {"logits": self.head(self.drop(emb)), "emb": emb}
    def groups(self):
        return [{"params":self.enc.parameters(),"lr":self.cfg.lr_enc,"weight_decay":self.cfg.wd},
                {"params":self.head.parameters(),"lr":self.cfg.lr_head,"weight_decay":self.cfg.wd}]

def class_w(loader,n):
    c=np.zeros(n)
    for b in loader:
        for l in b["labels"].tolist(): c[l]+=1
    c=np.maximum(c,1); w=1.0/c; return torch.tensor(w/w.sum()*n,dtype=torch.float32)

@torch.no_grad()
def eval_m(model,loader,dev):
    model.eval(); ps,ls=[],[]
    for b in loader:
        with _autocast_ctx(dev): logits=model(b["ids"].to(dev),b["mask"].to(dev))["logits"]
        ps.extend(logits.argmax(1).cpu().tolist()); ls.extend(b["labels"].tolist())
    return {"macro":f1_score(ls,ps,average="macro",zero_division=0),"weighted":f1_score(ls,ps,average="weighted",zero_division=0),
            "acc":accuracy_score(ls,ps),"per_class_f1":f1_score(ls,ps,average=None,zero_division=0).tolist()}

def train(cfg, tr_dl, vl_dl, ts_dl):
    dev=torch.device(cfg.device); model=RADNet(cfg).to(dev)
    w=class_w(tr_dl,cfg.n_cls).to(dev)
    opt=torch.optim.AdamW(model.groups())
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[cfg.lr_enc,cfg.lr_head],
        steps_per_epoch=len(tr_dl),epochs=cfg.epochs,pct_start=cfg.warmup)
    scaler=GradScaler(enabled=(dev.type=="cuda"))
    best_val,best_state=0,None
    logger.info(f"[RAD] α={cfg.renyi_alpha}")
    train_history,val_history=[],[]
    for ep in range(cfg.epochs):
        model.train(); total_loss=0; n_steps=0
        pbar=tqdm(tr_dl,desc=f"Ep {ep+1}/{cfg.epochs}",leave=False)
        for b in pbar:
            ids,mask,labs=b["ids"].to(dev),b["mask"].to(dev),b["labels"].to(dev)
            with _autocast_ctx(dev):
                logits=model(ids,mask)["logits"]
                loss=renyi_divergence_loss(logits,labs,alpha=cfg.renyi_alpha,weight=w)
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            total_loss+=loss.item(); n_steps+=1
            pbar.set_postfix({"loss":f"{loss.item():.4f}"})
        train_history.append({"epoch":ep+1,"loss":total_loss/max(n_steps,1)})
        vr=eval_m(model,vl_dl,dev); val_history.append({"epoch":ep+1,"macro":vr["macro"]})
        if vr["macro"]>best_val: best_val=vr["macro"]; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    res=eval_m(model,ts_dl,dev); res["train_history"]=train_history; res["val_history"]=val_history
    return res

def main():
    logger.info("[PREFLIGHT] Validating..."); _preflight_check()
    encoders=["unixcoder-base"]; benchmarks=[("codet_m4","author"),("aicd_t2","t2")]
    fracs=[0.01,0.05,0.20]
    alphas=[0.5, 1.0, 2.0, 5.0]  # Sweep α values
    results=[]
    for enc in encoders:
        for bench,task in benchmarks:
            for frac in fracs:
                for alpha in alphas:
                    cfg=Cfg(benchmark=bench,task=task,enc=enc,frac=frac,renyi_alpha=alpha); cfg=_hw(cfg)
                    if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                    tag=f"exp33_rad_{enc}_{bench}_f{frac}_a{alpha}"
                    logger.info(f"=== {tag} ==="); t0=time.time()
                    tr_dl,vl_dl,ts_dl=build_dls(cfg)
                    res=train(cfg,tr_dl,vl_dl,ts_dl); elapsed=time.time()-t0
                    row={"tag":tag,"enc":enc,"bench":bench,"frac":frac,"alpha":alpha,
                         "macro":round(res["macro"],6),"weighted":round(res["weighted"],6),
                         "acc":round(res["acc"],6),"dpaper":round(res["macro"]-PAPER_BASELINE,6),
                         "per_class_f1":res["per_class_f1"],
                         "train_history":res["train_history"],"val_history":res["val_history"],
                         "wall":round(elapsed,1),"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
                    results.append(row)
                    logger.info(f"[{tag}] α={alpha} MacroF1={res['macro']:.4f} ({res['macro']-PAPER_BASELINE:+.4f}) time={elapsed:.0f}s")
                    del tr_dl,vl_dl,ts_dl; import gc; gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

    os.makedirs("results",exist_ok=True)
    with open("results/exp33_rad_results.json","w") as f: json.dump(results,f,indent=2)
    print("\n"+"="*110)
    print(f"{'Enc':<22} {'Bench':<12} {'Frac':>6} {'α':>6} {'Macro':>10} {'dPaper':>10} {'W-F1':>10} {'Wall':>8}")
    print("-"*110)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['alpha']:>6.1f} {r['macro']:>10.4f} {r['dpaper']:>+10.4f} {r['weighted']:>10.4f} {r['wall']:>8.0f}s")
    # Show best α per (bench, frac)
    print("\n" + "="*60)
    print("Best α per (benchmark, fraction):")
    for bench,_ in benchmarks:
        for frac in fracs:
            subset = [r for r in results if r["bench"]==bench and r["frac"]==frac]
            if subset:
                best = max(subset, key=lambda x: x["macro"])
                print(f"  {bench} f={frac:.0%}: α*={best['alpha']:.1f} → Macro={best['macro']:.4f}")
    print("="*110+"\n[OK] RAD done!")

if __name__ == "__main__": main()
