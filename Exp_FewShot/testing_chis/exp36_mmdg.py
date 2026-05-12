"""
================================================================================
Theory-Track exp -- MMD with Genealogical Kernel (MMDG):
Maximum Mean Discrepancy with tree-structured kernel for distributional attribution.

ARXIV_ID      : Gretton 2012 MMD (JMLR 13:723-773); Muandet 2017 kernel mean
                embedding (1605.09522); Li 2015 generative moment matching (1502.02761)
NAME          : Maximum Mean Discrepancy with Genealogical Kernel (MMDG)
ONE-LINE CLAIM: Using the genealogy tree to define a characteristic kernel for MMD
                creates a distributional test that measures whether same-family
                code embeddings are drawn from the same distribution — a stronger
                signal than point-wise classification.
EQUATION      : k_gene(y_i, y_j) = exp(-d_tree(y_i, y_j)² / (2σ²))
                MMD²(F_k, F_l) = E[k(z_i,z_j)] - 2E[k(z_i,z_l)] + E[k(z_l,z_m)]
                L = CE + λ · Σ_{(k,l) related} MMD²(F_k, F_l)
PROPERTY      : MMD with genealogical kernel tests whether sibling generators
                produce distinguishable code distributions. Maximising inter-family
                MMD while minimising intra-family variance respects genealogical distance.
WHY NOT BEFORE: MMD has been used for distribution matching (GAN, domain adaptation)
                but never with a kernel defined by label genealogy. Our kernel
                k_gene makes the two-sample test genealogy-aware.
FALSIFIER     : If MMDG does not improve sibling separation (sibling-pair F1)
                over CE baseline, the genealogical kernel is not characteristic.
================================================================================

exp36_mmdg.py — MMD with Genealogical Kernel for few-shot AI-code attribution.
Protocol: fraction-based (1% / 5% / 20%), unixcoder-base only.
"""

# === KAGGLE PATHS ===
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_DROID = "/kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/data"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

from __future__ import annotations
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
logger = logging.getLogger("exp36")

HIER_FAM = {0:0, 1:1, 2:2, 3:3, 4:1, 5:4}
PAPER_BASELINE = 0.6633

# =============================================================================
# GENEALOGICAL KERNEL AND MMD
# =============================================================================
def _build_gene_kernel(n_cls, hier_fam=None, sigma=1.0):
    """Build genealogy kernel: k_gene(i,j) = exp(-d_tree(i,j)² / (2σ²)).
    
    This kernel defines "how similar" two classes are genealogically.
    Used to weight the MMD computation between class distributions.
    """
    K_gene = torch.zeros(n_cls, n_cls)
    if hier_fam is not None and n_cls == 6:
        for i in range(n_cls):
            for j in range(n_cls):
                if i == j: d = 0.0
                elif hier_fam.get(i) == hier_fam.get(j): d = 1.0
                elif (i == 0) != (j == 0): d = 3.0
                else: d = 2.0
                K_gene[i, j] = (-d**2 / (2 * sigma**2))
    else:
        for i in range(n_cls):
            for j in range(n_cls):
                d = 0.0 if i == j else 1.0
                K_gene[i, j] = (-d**2 / (2 * sigma**2))
    K_gene = K_gene.exp()
    return K_gene

def genealogy_mmd_loss(embeddings, labels, n_cls, gene_kernel, rbf_sigma=1.0):
    """Compute genealogy-weighted MMD between class distributions.
    
    For each pair of classes (i,j), compute MMD²(F_i, F_j) and weight
    by the genealogical kernel: sibling pairs get higher weight.
    
    Objective: MAXIMISE inter-class MMD for genealogically distant classes,
    MINIMISE it for same class (intra-class tightness).
    """
    device = embeddings.device
    batch_size = embeddings.size(0)
    
    if batch_size < 2:
        return torch.tensor(0.0, device=device)
    
    # RBF kernel for embeddings: k(z_i, z_j) = exp(-||z_i - z_j||²/(2σ²))
    dists = torch.cdist(embeddings, embeddings, p=2)  # (batch, batch)
    K_rbf = (-dists**2 / (2 * rbf_sigma**2)).exp()  # (batch, batch)
    
    total_mmd = torch.tensor(0.0, device=device)
    n_pairs = 0
    
    for ci in range(n_cls):
        mask_i = (labels == ci)
        n_i = mask_i.sum().item()
        if n_i < 2: continue
        
        # Intra-class MMD (should be low = tight clusters)
        K_ii = K_rbf[mask_i][:, mask_i]
        intra_mmd = K_ii.mean()  # Higher = tighter (we want this)
        
        for cj in range(ci + 1, n_cls):
            mask_j = (labels == cj)
            n_j = mask_j.sum().item()
            if n_j < 1: continue
            
            # Inter-class: MMD²(F_i, F_j) = E[k(z_i,z_i')] + E[k(z_j,z_j')] - 2E[k(z_i,z_j)]
            K_jj = K_rbf[mask_j][:, mask_j]
            K_ij = K_rbf[mask_i][:, mask_j]
            
            mmd_sq = K_ii.mean() + K_jj.mean() - 2 * K_ij.mean()
            
            # Weight by genealogical kernel: sibling pairs get higher weight
            # For siblings (high gene_kernel), we want HIGHER MMD (more separation)
            gene_weight = gene_kernel[ci, cj].to(device)
            
            # Loss = -gene_weight * MMD² (negative because we want to MAXIMISE separation)
            total_mmd -= gene_weight * mmd_sq
            n_pairs += 1
    
    return total_mmd / max(n_pairs, 1)


# === Standard data loading ===
@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.05; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4; wd: float = 0.01
    warmup: float = 0.1; device: str = "cuda"
    mmd_weight: float = 0.1   # Weight for MMD loss
    rbf_sigma: float = 1.0    # RBF kernel bandwidth
    gene_sigma: float = 1.0   # Genealogy kernel bandwidth

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
        tr=ds.filter(lambda x:str(x.get("split","")).lower()=="train")
        vl=ds.filter(lambda x:str(x.get("split","")).lower() in {"val","validation","dev"})
        ts=ds.filter(lambda x:str(x.get("split","")).lower()=="test")
    else:
        s=ds.train_test_split(test_size=0.1,seed=42); s2=s["train"].train_test_split(test_size=1/9,seed=42)
        return s2["train"],s2["test"],s["test"]
    return tr,vl,ts
def _load_aicd(task):
    task_map={"t1":"T1","t2":"T2","t3":"T3"}; tn=task_map.get(task.lower())
    if not tn: raise ValueError(f"[aicd] Unknown '{task}'.")
    tp=os.path.join(KAGGLE_AICD,tn)
    if not os.path.isdir(tp): raise FileNotFoundError(f"[aicd] STRICT: {tn} not found.")
    pf=sorted(glob.glob(os.path.join(tp,"**","*.parquet"),recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: No parquet in {tp}.")
    logger.info(f"[aicd] Loading {tn} ({len(pf)} files)")
    ds=load_dataset("parquet",data_files=pf,split="train")
    if "split" in ds.column_names:
        try:
            tr=ds.filter(lambda x:str(x.get("split","")).lower()=="train")
            vl=ds.filter(lambda x:str(x.get("split","")).lower() in {"val","validation","dev"})
            ts=ds.filter(lambda x:str(x.get("split","")).lower()=="test")
            if len(tr)>0 and len(vl)>0 and len(ts)>0: return tr,vl,ts
        except: pass
    s=ds.train_test_split(test_size=0.1,seed=42); s2=s["train"].train_test_split(test_size=1/9,seed=42)
    return s2["train"],s2["test"],s["test"]
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

class MMDGNet(nn.Module):
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
    dev=torch.device(cfg.device); model=MMDGNet(cfg).to(dev)
    w=class_w(tr_dl,cfg.n_cls).to(dev)
    opt=torch.optim.AdamW(model.groups())
    sched=torch.optim.lr_scheduler.OneCycleLR(opt,max_lr=[cfg.lr_enc,cfg.lr_head],
        steps_per_epoch=len(tr_dl),epochs=cfg.epochs,pct_start=cfg.warmup)
    scaler=GradScaler(enabled=(dev.type=="cuda"))
    best_val,best_state=0,None
    hier = HIER_FAM if cfg.benchmark == "codet_m4" else None
    gene_kernel = _build_gene_kernel(cfg.n_cls, hier, cfg.gene_sigma)
    logger.info(f"[MMDG] Genealogy kernel:\n{gene_kernel}")
    train_history,val_history=[],[]
    for ep in range(cfg.epochs):
        model.train(); total_loss=0; n_steps=0
        pbar=tqdm(tr_dl,desc=f"Ep {ep+1}/{cfg.epochs}",leave=False)
        for b in pbar:
            ids,mask,labs=b["ids"].to(dev),b["mask"].to(dev),b["labels"].to(dev)
            with _autocast_ctx(dev):
                out=model(ids,mask)
                ce_loss=F.cross_entropy(out["logits"],labs,weight=w)
                emb_norm=F.normalize(out["emb"],dim=-1)
                mmd_loss=genealogy_mmd_loss(emb_norm,labs,cfg.n_cls,gene_kernel,cfg.rbf_sigma)
                loss=ce_loss+cfg.mmd_weight*mmd_loss
            scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sched.step()
            total_loss+=loss.item(); n_steps+=1
            pbar.set_postfix({"loss":f"{loss.item():.4f}","mmd":f"{mmd_loss.item():.4f}"})
        train_history.append({"epoch":ep+1,"loss":total_loss/max(n_steps,1)})
        vr=eval_m(model,vl_dl,dev); val_history.append({"epoch":ep+1,"macro":vr["macro"]})
        if vr["macro"]>best_val: best_val=vr["macro"]; best_state={k:v.cpu().clone() for k,v in model.state_dict().items()}
    if best_state: model.load_state_dict(best_state)
    res=eval_m(model,ts_dl,dev); res["train_history"]=train_history; res["val_history"]=val_history
    return res

def main():
    logger.info("[PREFLIGHT] Validating..."); _preflight_check()
    encoders=["unixcoder-base"]; benchmarks=[("codet_m4","author"),("aicd_t2","t2")]; fracs=[0.01,0.05,0.20]
    results=[]
    for enc in encoders:
        for bench,task in benchmarks:
            for frac in fracs:
                cfg=Cfg(benchmark=bench,task=task,enc=enc,frac=frac); cfg=_hw(cfg)
                tag=f"exp36_mmdg_{enc}_{bench}_f{frac}"; logger.info(f"=== {tag} ===")
                t0=time.time(); tr_dl,vl_dl,ts_dl=build_dls(cfg)
                res=train(cfg,tr_dl,vl_dl,ts_dl); elapsed=time.time()-t0
                row={"tag":tag,"enc":enc,"bench":bench,"frac":frac,
                     "macro":round(res["macro"],6),"weighted":round(res["weighted"],6),
                     "acc":round(res["acc"],6),"dpaper":round(res["macro"]-PAPER_BASELINE,6),
                     "per_class_f1":res["per_class_f1"],
                     "train_history":res["train_history"],"val_history":res["val_history"],
                     "wall":round(elapsed,1),"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}
                results.append(row)
                logger.info(f"[{tag}] Macro={res['macro']:.4f} ({res['macro']-PAPER_BASELINE:+.4f}) t={elapsed:.0f}s")
                del tr_dl,vl_dl,ts_dl; import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    os.makedirs("results",exist_ok=True)
    with open("results/exp36_mmdg_results.json","w") as f: json.dump(results,f,indent=2)
    print("\n"+"="*100)
    for r in results: print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} Macro={r['macro']:.4f} ({r['dpaper']:+.4f}) t={r['wall']:.0f}s")
    print("="*100+"\n[OK] MMDG done!")

if __name__ == "__main__": main()
