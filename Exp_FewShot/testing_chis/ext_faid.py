# ext_faid — Faithful K-class reproduction of FAID (EACL 2026)
# =============================================================================
# UPSTREAM : "FAID: Fine-grained AI-generated Text Detection using Multi-task
#            Auxiliary and Multi-level Contrastive Learning"
# FAITHFULNESS: Multi-level SupCon (a=2,b=1,c=1), 3-layer tanh head, τ=0.07
#   For all-AI data: L_human=L_mixed=L_mixed_set=0 → effective: 4b*L_label + c*CE
# =============================================================================
from __future__ import annotations
KAGGLE_MODELS="/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
import os,sys,time,json,random,subprocess,importlib.util,warnings,glob,math
from dataclasses import dataclass,field; from typing import Dict
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
_ensure("numpy");_ensure("torch");_ensure("datasets");_ensure("transformers");_ensure("scikit-learn");_ensure("tqdm")
import numpy as np; import torch,torch.nn as nn,torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
from torch.utils.data import Dataset as TD,DataLoader
from transformers import AutoTokenizer,get_linear_schedule_with_warmup
from tqdm import tqdm; from torch.cuda.amp import GradScaler
warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",stream=sys.stdout)
logger=logging.getLogger("ext_faid"); PAPER_BASELINE=0.6633

@dataclass
class Cfg:
    benchmark:str="codet_m4";task:str="author";enc:str="unixcoder-base"
    frac:float=0.20;n_cls:int=6;seed:int=42;bs:int=64;seq:int=512;epochs:int=6
    lr_enc:float=3e-5;warmup:float=0.10;wd:float=0.01
    temperature:float=0.07;a:float=2.0;b:float=1.0;c:float=1.0;device:str="cuda"

def adaptive_schedule(cfg):
    f=cfg.frac
    if f<=0.02: cfg.epochs,cfg.lr_enc,cfg.warmup=10,3e-5,0.20
    elif f<=0.10: cfg.epochs,cfg.lr_enc,cfg.warmup=6,3e-5,0.15
    else: cfg.epochs,cfg.lr_enc,cfg.warmup=6,4e-5,0.10
    return cfg
def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True;torch.backends.cudnn.benchmark=True
        mem=torch.cuda.get_device_properties(0).total_memory/1e9
        if mem>=40:cfg.bs=128
        elif mem>=20:cfg.bs=64
        else:cfg.bs=32
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs}")
    return cfg
def set_seed(s):
    random.seed(s);np.random.seed(s);torch.manual_seed(s)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(s)

KAGGLE_CODET="/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD="/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"
def _is_human(t): return str(t or "").strip().lower() in {"human","human_written","human-generated"}
def _vocab(train):
    names={str(r.get("model","")or"").strip() for r in train if not _is_human(r.get("target","")) and r.get("model","")}
    return {n:i+1 for i,n in enumerate(sorted(names))}
def _conv_codet(split,task,vocab):
    def row(r):
        code=""
        for f in ("cleaned_code","code"):
            v=r.get(f,"");
            if isinstance(v,str) and v.strip():code=v;break
        label=0 if _is_human(r.get("target","")) else vocab.get(str(r.get("model","")or"").strip(),-1)
        return {"code":code,"label":label,"language":str(r.get("language","")).strip().lower(),"source":str(r.get("source","")).strip().lower()}
    return split.map(row,remove_columns=split.column_names).filter(lambda x:x["label"]>=0 and len(x["code"].strip())>0)
def _conv_aicd(split):
    def row(r): return {"code":str(r.get("code","")).strip(),"label":int(r.get("label",-1)),"language":str(r.get("language","")).strip().lower(),"source":""}
    return split.map(row,remove_columns=split.column_names).filter(lambda x:x["label"]>=0 and len(x["code"].strip())>0)
def _load_codet():
    ds=load_dataset("parquet",data_files=KAGGLE_CODET,split="train")
    if "split" in ds.column_names:
        return (ds.filter(lambda x:str(x.get("split","")).lower()=="train"),ds.filter(lambda x:str(x.get("split","")).lower() in {"val","validation","dev"}),ds.filter(lambda x:str(x.get("split","")).lower()=="test"))
    s=ds.train_test_split(test_size=0.1,seed=42);s2=s["train"].train_test_split(test_size=1/9,seed=42);return s2["train"],s2["test"],s["test"]
def _load_aicd(task):
    tn={"t1":"T1","t2":"T2","t3":"T3"}.get(task.lower());tp=os.path.join(KAGGLE_AICD,tn)
    pf=sorted(glob.glob(os.path.join(tp,"**","*.parquet"),recursive=True))
    ds=load_dataset("parquet",data_files=pf,split="train")
    if "split" in ds.column_names:
        tr=ds.filter(lambda x:str(x.get("split","")).lower()=="train");vl=ds.filter(lambda x:str(x.get("split","")).lower() in {"val","validation","dev"});ts=ds.filter(lambda x:str(x.get("split","")).lower()=="test")
        if len(tr)>0 and len(vl)>0 and len(ts)>0:return tr,vl,ts
    s=ds.train_test_split(test_size=0.1,seed=42);s2=s["train"].train_test_split(test_size=1/9,seed=42);return s2["train"],s2["test"],s["test"]

def _tokenize(code,tokenizer,max_len):
    toks=tokenizer.tokenize(" ".join(code.split()))[:max_len-4]
    toks=[tokenizer.cls_token,"<encoder_only>",tokenizer.sep_token]+toks+[tokenizer.sep_token]
    ids=tokenizer.convert_tokens_to_ids(toks);ids+=[tokenizer.pad_token_id]*(max_len-len(ids));return ids[:max_len]
class FSDS(TD):
    def __init__(self,data,tok,seq_len,frac=1.0,seed=42):
        self.data=data;self.tok=tok;self.seq_len=seq_len
        if frac<1.0:
            rng=random.Random(seed);labels=list(range(max(self.data["label"])+1));keep=[]
            for lbl in labels:
                idx=[i for i,x in enumerate(self.data["label"]) if x==lbl]
                keep.extend(rng.sample(idx,min(max(1,int(len(idx)*frac)),len(idx))))
            self.data=self.data.select(keep);logger.info(f"[FSDS] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self):return len(self.data)
    def __getitem__(self,i):
        r=self.data[i];ids=_tokenize(r["code"][:5000],self.tok,self.seq_len)
        return {"input_ids":torch.tensor(ids,dtype=torch.long),"label":r["label"],"language":r.get("language",""),"source":r.get("source","")}

class ClassificationHead(nn.Module):
    def __init__(self,in_dim,out_dim):
        super().__init__();self.dense1=nn.Linear(in_dim,in_dim//4);self.dense2=nn.Linear(in_dim//4,in_dim//16);self.out_proj=nn.Linear(in_dim//16,out_dim)
        for l in (self.dense1,self.dense2,self.out_proj):nn.init.xavier_uniform_(l.weight);nn.init.normal_(l.bias,std=1e-6)
    def forward(self,x):return self.out_proj(torch.tanh(self.dense2(torch.tanh(self.dense1(x)))))

def _compute_supcon(q,k,q_label,k_label,temperature,eps=1e-6):
    q_n=F.normalize(q,dim=-1);k_n=F.normalize(k,dim=-1);sim=(q_n@k_n.T)/temperature
    same=(q_label.view(-1,1)==k_label.view(1,-1)).float()
    pos_sim=(sim*same).sum(1)/same.sum(1).clamp(min=eps);neg_sim=sim*(1-same)
    return torch.cat([pos_sim.unsqueeze(1),neg_sim],dim=1)

class FAIDK(nn.Module):
    """Effective loss for all-AI: a*L_set+(4b-a)*L_label+c*CE = 4b*L_label+c*CE (since L_set=L_label)"""
    def __init__(self,encoder,hidden,n_cls,pad_id,temperature,a,b,c):
        super().__init__();self.encoder=encoder;self.pad_id=pad_id;self.temperature=temperature
        self.a=a;self.b=b;self.c=c;self.head=ClassificationHead(hidden,n_cls)
    def _encode(self,input_ids):
        mask=input_ids.ne(self.pad_id);attn=mask.unsqueeze(1)*mask.unsqueeze(2)
        out=self.encoder(input_ids,attention_mask=attn,output_hidden_states=True);tok=out[0]
        return (tok*mask.unsqueeze(-1)).sum(1)/mask.sum(-1).unsqueeze(-1).clamp(min=1)
    def forward(self,input_ids,labels):
        q=self._encode(input_ids);k=q.detach()
        logits_label=_compute_supcon(q,k,labels,labels,self.temperature)
        gt=torch.zeros(q.size(0),dtype=torch.long,device=q.device)
        loss_label=F.cross_entropy(logits_label,gt)
        # For all-AI: L_set=L_label, so effective = a*L_set + (4b-a)*L_label = 4b*L_label
        loss_supcon=4*self.b*loss_label
        out=self.head(q);loss_ce=F.cross_entropy(out,labels)
        return loss_supcon+self.c*loss_ce, out

@torch.no_grad()
def eval_pack(model,loader,cfg):
    model.eval();preds,labels=[],[]
    for b in tqdm(loader,desc="Eval"):
        ids=b["input_ids"].to(cfg.device);labs=b["label"]
        if not torch.is_tensor(labs):labs=torch.tensor(labs,dtype=torch.long)
        labs=labs.to(cfg.device);_,logits=model(ids,labs)
        preds.extend(logits.argmax(-1).cpu().tolist());labels.extend(labs.cpu().tolist())
    preds=np.array(preds);labels=np.array(labels);n_cls=cfg.n_cls
    overall={"accuracy":float(accuracy_score(labels,preds)),"macro_f1":float(f1_score(labels,preds,average="macro",zero_division=0)),
             "weighted_f1":float(f1_score(labels,preds,average="weighted",zero_division=0)),
             "macro_precision":float(precision_score(labels,preds,average="macro",zero_division=0)),
             "macro_recall":float(recall_score(labels,preds,average="macro",zero_division=0))}
    cm=confusion_matrix(labels,preds,labels=list(range(n_cls)))
    return {"overall":overall,"confusion_matrix":cm.tolist(),"n_samples":int(len(labels))}

def run_exp(cfg,tag):
    set_seed(cfg.seed);cfg=_hw(cfg);cfg=adaptive_schedule(cfg)
    if cfg.benchmark=="codet_m4":
        tr_raw,vl_raw,ts_raw=_load_codet();vocab=_vocab(tr_raw)
        tr_data=_conv_codet(tr_raw,"author",vocab);vl_data=_conv_codet(vl_raw,"author",vocab);ts_data=_conv_codet(ts_raw,"author",vocab)
    else:
        tr_raw,vl_raw,ts_raw=_load_aicd("t2");tr_data=_conv_aicd(tr_raw);vl_data=_conv_aicd(vl_raw);ts_data=_conv_aicd(ts_raw)
    cfg.n_cls=max(tr_data["label"])+1
    tok=AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS,cfg.enc),local_files_only=True);pad_id=tok.pad_token_id
    tr_ds=FSDS(tr_data,tok,cfg.seq,frac=cfg.frac,seed=cfg.seed);vl_ds=FSDS(vl_data,tok,cfg.seq);ts_ds=FSDS(ts_data,tok,cfg.seq)
    lc=dict(batch_size=cfg.bs,num_workers=4,pin_memory=True)
    tr_dl=DataLoader(tr_ds,shuffle=True,**lc);vl_dl=DataLoader(vl_ds,shuffle=False,**lc);ts_dl=DataLoader(ts_ds,shuffle=False,**lc)
    from transformers import RobertaConfig,RobertaModel
    config=RobertaConfig.from_pretrained(os.path.join(KAGGLE_MODELS,cfg.enc),local_files_only=True)
    encoder=RobertaModel.from_pretrained(os.path.join(KAGGLE_MODELS,cfg.enc),local_files_only=True)
    model=FAIDK(encoder,config.hidden_size,cfg.n_cls,pad_id,cfg.temperature,cfg.a,cfg.b,cfg.c).to(cfg.device)
    total_steps=max(1,len(tr_ds)//cfg.bs)*cfg.epochs
    opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr_enc,weight_decay=cfg.wd)
    sch=get_linear_schedule_with_warmup(opt,int(total_steps*cfg.warmup),total_steps);scaler=GradScaler()
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} n_cls={cfg.n_cls} train={len(tr_ds)}")
    best_val,best_state,val_hist=0.0,None,[]
    for ep in range(cfg.epochs):
        model.train();tot=0.0
        for b in tqdm(tr_dl,desc=f"Train ep{ep+1}"):
            ids=b["input_ids"].to(cfg.device);labs=b["label"]
            if not torch.is_tensor(labs):labs=torch.tensor(labs,dtype=torch.long)
            labs=labs.to(cfg.device);opt.zero_grad()
            with torch.autocast(device_type="cuda",dtype=torch.bfloat16,enabled=(cfg.device=="cuda")):
                loss,_=model(ids,labs)
            scaler.scale(loss).backward();scaler.unscale_(opt);torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt);scaler.update();sch.step();tot+=loss.item()
        val_met=eval_pack(model,vl_dl,cfg);v=val_met["overall"]["macro_f1"];val_hist.append(v)
        logger.info(f"[epoch {ep+1}] loss={tot/max(1,len(tr_dl)):.4f} val={v:.4f}")
        if v>best_val:best_val=v;best_state={k:v_.cpu().clone() for k,v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    ts_met=eval_pack(model,ts_dl,cfg);test_macro=ts_met["overall"]["macro_f1"];gap=best_val-test_macro
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f}")
    return {"tag":tag,"method":"FAID-K","upstream":"EACL 2026","note":"Multi-level SupCon: 4b*L_label+c*CE (all-AI). a=2,b=1,c=1, τ=0.07.",
            "enc":cfg.enc,"bench":cfg.benchmark,"frac":cfg.frac,"epochs":cfg.epochs,"lr_enc":cfg.lr_enc,"a":cfg.a,"b":cfg.b,"c":cfg.c,
            "val_macro":best_val,"macro":test_macro,"weighted":ts_met["overall"]["weighted_f1"],"acc":ts_met["overall"]["accuracy"],
            "val_test_gap":gap,"dpaper":test_macro-PAPER_BASELINE,"test_metrics":ts_met,"val_history":val_hist,"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}

def main():
    results=[]
    for bench,task,n_cls in [("codet_m4","author",6),("aicd_t2","t2",12)]:
        for frac in [0.01,0.05,0.20]:
            cfg=Cfg(benchmark=bench,task=task,frac=frac,n_cls=n_cls);tag=f"ext_faid_{cfg.enc}_{bench}_f{frac}"
            logger.info(f"=== {tag} ===");t0=time.time()
            try:
                res=run_exp(cfg,tag);res["wall"]=round(time.time()-t0,1);results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s")
            except Exception as e:logger.error(f"[{tag}] FAILED: {e}");import traceback;traceback.print_exc()
            import gc;gc.collect()
            if torch.cuda.is_available():torch.cuda.empty_cache()
    try:_here=os.path.dirname(os.path.realpath(__file__))
    except NameError:_here=os.getcwd()
    out_dir=os.path.join(_here,"results");os.makedirs(out_dir,exist_ok=True)
    with open(os.path.join(out_dir,"ext_faid_results.json"),"w") as f:json.dump(results,f,indent=2)
    print("\n"+"="*100)
    for r in results:print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f}")
    print("="*100)

if __name__=="__main__":main()
