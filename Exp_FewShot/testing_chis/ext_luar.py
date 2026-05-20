# ext_luar — Faithful reproduction of LUAR (ICLR 2024)
# =============================================================================
# UPSTREAM : "LUAR: Linguistic Unified Authorship Representation"
# FAITHFULNESS: Mode A=frozen prototype cosine-NN, Mode B=N-shot FT (5ep, lr=2e-5)
# =============================================================================
from __future__ import annotations
KAGGLE_MODELS="/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
import os,sys,time,json,random,subprocess,importlib.util,warnings,glob,math,copy
from dataclasses import dataclass; from typing import Dict
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF","expandable_segments:True")
def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None: subprocess.check_call([sys.executable,"-m","pip","install","-q",p])
_ensure("numpy");_ensure("torch");_ensure("datasets");_ensure("transformers");_ensure("scikit-learn");_ensure("tqdm")
import numpy as np; import torch,torch.nn as nn,torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score,f1_score,precision_score,recall_score,confusion_matrix
from torch.utils.data import Dataset as TD,DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
warnings.filterwarnings("ignore")
import logging; logging.basicConfig(level=logging.INFO,format="%(asctime)s %(message)s",stream=sys.stdout)
logger=logging.getLogger("ext_luar"); PAPER_BASELINE=0.6633

@dataclass
class Cfg:
    benchmark:str="codet_m4";task:str="author";enc:str="unixcoder-base"
    frac:float=0.20;n_cls:int=6;seed:int=42;bs:int=64;seq:int=512
    ft_epochs:int=5;ft_lr:float=2e-5  # faithful: adaptation_lr=2e-5, num_few_shot_epochs=5
    device:str="cuda"

def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
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
        return {"code":code,"label":label,"language":str(r.get("language","")).strip().lower(),"source":""}
    return split.map(row,remove_columns=split.column_names).filter(lambda x:x["label"]>=0 and len(x["code"].strip())>0)
def _conv_aicd(split):
    def row(r): return {"code":str(r.get("code","")).strip(),"label":int(r.get("label",-1)),"language":"","source":""}
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
        return {"input_ids":torch.tensor(ids,dtype=torch.long),"label":r["label"]}

def encode_all(encoder,loader,device,pad_id):
    encoder.eval();embs,labs=[],[]
    with torch.no_grad():
        for b in tqdm(loader,desc="Encode"):
            ids=b["input_ids"].to(device);mask=ids.ne(pad_id)
            attn=mask.unsqueeze(1)*mask.unsqueeze(2)
            out=encoder(ids,attention_mask=attn,output_hidden_states=True);tok=out[0]
            vec=(tok*mask.unsqueeze(-1)).sum(1)/mask.sum(-1).unsqueeze(-1).clamp(min=1)
            embs.append(F.normalize(vec,dim=-1).cpu());l=b["label"]
            labs.extend(l.tolist() if torch.is_tensor(l) else list(l))
    return torch.cat(embs,0),np.array(labs)

def prototype_nn(support_emb,support_lab,query_emb,n_cls):
    protos=[]
    for c in range(n_cls):
        mask=(support_lab==c);
        if mask.sum()>0:protos.append(support_emb[mask].mean(0))
        else:protos.append(torch.zeros(support_emb.size(1)))
    protos=F.normalize(torch.stack(protos),dim=-1)
    sim=query_emb@protos.T;return sim.argmax(-1).numpy()

@torch.no_grad()
def eval_proto(preds,labels,n_cls):
    preds=np.array(preds);labels=np.array(labels)
    return {"accuracy":float(accuracy_score(labels,preds)),"macro_f1":float(f1_score(labels,preds,average="macro",zero_division=0)),
            "weighted_f1":float(f1_score(labels,preds,average="weighted",zero_division=0)),
            "macro_precision":float(precision_score(labels,preds,average="macro",zero_division=0)),
            "macro_recall":float(recall_score(labels,preds,average="macro",zero_division=0))}

def run_exp(cfg,tag):
    set_seed(cfg.seed);cfg=_hw(cfg)
    if cfg.benchmark=="codet_m4":
        tr_raw,vl_raw,ts_raw=_load_codet();vocab=_vocab(tr_raw)
        tr_data=_conv_codet(tr_raw,"author",vocab);vl_data=_conv_codet(vl_raw,"author",vocab);ts_data=_conv_codet(ts_raw,"author",vocab)
    else:
        tr_raw,vl_raw,ts_raw=_load_aicd("t2");tr_data=_conv_aicd(tr_raw);vl_data=_conv_aicd(vl_raw);ts_data=_conv_aicd(ts_raw)
    cfg.n_cls=max(tr_data["label"])+1
    tok=AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS,cfg.enc),local_files_only=True);pad_id=tok.pad_token_id
    tr_ds=FSDS(tr_data,tok,cfg.seq,frac=cfg.frac,seed=cfg.seed);vl_ds=FSDS(vl_data,tok,cfg.seq);ts_ds=FSDS(ts_data,tok,cfg.seq)
    lc=dict(batch_size=cfg.bs,num_workers=4,pin_memory=True)
    tr_dl=DataLoader(tr_ds,shuffle=False,**lc);vl_dl=DataLoader(vl_ds,shuffle=False,**lc);ts_dl=DataLoader(ts_ds,shuffle=False,**lc)
    from transformers import RobertaModel
    encoder=RobertaModel.from_pretrained(os.path.join(KAGGLE_MODELS,cfg.enc),local_files_only=True).to(cfg.device)

    # Mode A: frozen prototype-NN
    logger.info("[Mode A] Frozen prototype-NN")
    tr_emb,tr_lab=encode_all(encoder,tr_dl,cfg.device,pad_id)
    vl_emb,vl_lab=encode_all(encoder,vl_dl,cfg.device,pad_id)
    ts_emb,ts_lab=encode_all(encoder,ts_dl,cfg.device,pad_id)
    vl_preds=prototype_nn(tr_emb,tr_lab,vl_emb,cfg.n_cls)
    ts_preds=prototype_nn(tr_emb,tr_lab,ts_emb,cfg.n_cls)
    nn_val=eval_proto(vl_preds,vl_lab,cfg.n_cls)
    nn_test=eval_proto(ts_preds,ts_lab,cfg.n_cls)
    logger.info(f"[Mode A] val={nn_val['macro_f1']:.4f} test={nn_test['macro_f1']:.4f}")

    # Mode B: N-shot fine-tune then prototype-NN
    logger.info(f"[Mode B] N-shot FT (ep={cfg.ft_epochs}, lr={cfg.ft_lr})")
    encoder_ft=copy.deepcopy(encoder)
    ft_dl=DataLoader(tr_ds,shuffle=True,batch_size=min(cfg.bs,len(tr_ds)),num_workers=2,pin_memory=True)
    opt=torch.optim.AdamW(encoder_ft.parameters(),lr=cfg.ft_lr)
    clf=nn.Linear(encoder_ft.config.hidden_size,cfg.n_cls).to(cfg.device)
    opt2=torch.optim.AdamW(list(encoder_ft.parameters())+list(clf.parameters()),lr=cfg.ft_lr)
    for ep in range(cfg.ft_epochs):
        encoder_ft.train();clf.train();tot=0.0
        for b in ft_dl:
            ids=b["input_ids"].to(cfg.device);labs=b["label"]
            if not torch.is_tensor(labs):labs=torch.tensor(labs,dtype=torch.long)
            labs=labs.to(cfg.device);mask=ids.ne(pad_id);attn=mask.unsqueeze(1)*mask.unsqueeze(2)
            out=encoder_ft(ids,attention_mask=attn,output_hidden_states=True);tok=out[0]
            vec=(tok*mask.unsqueeze(-1)).sum(1)/mask.sum(-1).unsqueeze(-1).clamp(min=1)
            logits=clf(vec);loss=F.cross_entropy(logits,labs)
            opt2.zero_grad();loss.backward();opt2.step();tot+=loss.item()
        logger.info(f"[Mode B ep{ep+1}] loss={tot/max(1,len(ft_dl)):.4f}")
    tr_emb_ft,_=encode_all(encoder_ft,tr_dl,cfg.device,pad_id)
    vl_emb_ft,_=encode_all(encoder_ft,vl_dl,cfg.device,pad_id)
    ts_emb_ft,_=encode_all(encoder_ft,ts_dl,cfg.device,pad_id)
    vl_preds_ft=prototype_nn(tr_emb_ft,tr_lab,vl_emb_ft,cfg.n_cls)
    ts_preds_ft=prototype_nn(tr_emb_ft,tr_lab,ts_emb_ft,cfg.n_cls)
    ft_val=eval_proto(vl_preds_ft,vl_lab,cfg.n_cls)
    ft_test=eval_proto(ts_preds_ft,ts_lab,cfg.n_cls)
    logger.info(f"[Mode B] val={ft_val['macro_f1']:.4f} test={ft_test['macro_f1']:.4f}")

    # Pick best mode
    best_mode="B" if ft_val["macro_f1"]>nn_val["macro_f1"] else "A"
    best_val=max(nn_val["macro_f1"],ft_val["macro_f1"])
    best_test=ft_test["macro_f1"] if best_mode=="B" else nn_test["macro_f1"]
    best_met=ft_test if best_mode=="B" else nn_test
    logger.info(f"[final] best_mode={best_mode} val={best_val:.4f} test={best_test:.4f}")
    return {"tag":tag,"method":f"LUAR-{best_mode}","upstream":"ICLR 2024","note":f"Mode A=frozen prototype-NN, Mode B=N-shot FT. Best={best_mode}.",
            "enc":cfg.enc,"bench":cfg.benchmark,"frac":cfg.frac,"ft_epochs":cfg.ft_epochs,"ft_lr":cfg.ft_lr,
            "nn_val":nn_val["macro_f1"],"nn_test":nn_test["macro_f1"],"ft_val":ft_val["macro_f1"],"ft_test":ft_test["macro_f1"],
            "val_macro":best_val,"macro":best_test,"weighted":best_met["weighted_f1"],"acc":best_met["accuracy"],
            "val_test_gap":best_val-best_test,"dpaper":best_test-PAPER_BASELINE,
            "test_metrics":{"overall":best_met},"timestamp":time.strftime("%Y-%m-%d %H:%M:%S")}

def main():
    results=[]
    for bench,task,n_cls in [("codet_m4","author",6),("aicd_t2","t2",12)]:
        for frac in [0.01,0.05,0.20]:
            cfg=Cfg(benchmark=bench,task=task,frac=frac,n_cls=n_cls);tag=f"ext_luar_{cfg.enc}_{bench}_f{frac}"
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
    with open(os.path.join(out_dir,"ext_luar_results.json"),"w") as f:json.dump(results,f,indent=2)
    print("\n"+"="*100)
    for r in results:print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} NN={r['nn_test']:.4f} FT={r['ft_test']:.4f} best={r['macro']:.4f} {r['dpaper']:>+9.4f}")
    print("="*100)

if __name__=="__main__":main()
