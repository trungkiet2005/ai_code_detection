# exp161 — KSWEEP  (E2: the collapse curve)
# =============================================================================
# NAME       : KSWEEP (Attribution accuracy vs number-of-generators K).
# REFERENCE  : new (D1 proposal, related_work/PROPOSAL_D1_sinkclass.md).
#              Same plain-CE UniXcoder attributor as ext_gptsniffer
#              (GPTSniffer-K, Nguyen et al., JSS 2024).
# CLAIM      : Attribution difficulty is not fixed — it is governed by K. As we
#              add generator classes, Macro-F1 and accuracy of the plain-CE
#              attributor decay monotonically along one curve, the empirical
#              shadow of the log K / C_min sample-complexity law: more mutually
#              confusable generators -> more collapse.
# EQUATION   : record ( K, macro_F1(K), acc(K), {classes} ); the acc-vs-K curve
#              is the E2 collapse curve the theory (Fano: err >= 1-(I+log2)/log K)
#              must reproduce after normalising the x-axis by C_min (see exp162).
# WHY NEW    : No code-attribution paper reports a controlled generator-count
#              sweep with fixed data budget; K is always taken as the fixed
#              benchmark cardinality, never treated as the independent variable.
# WOW HOOK   : "Attribution accuracy is a falling curve in the number of
#              generators, not a benchmark constant — and every modality's
#              curve is the same curve once you rescale by generator distance."
# FALSIFIER  : If Macro-F1 is flat in K (|slope| < 0.01 F1 per added generator,
#              averaged over subset draws) on BOTH CoDET-M4 and AICD-T2, there
#              is no collapse curve and E2 fails.
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from dataclasses import dataclass
from typing import List, Dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
import torch
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp161")

KAGGLE_MODELS = os.environ.get("MODELS_DIR", "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models")
KAGGLE_CODET  = os.environ.get("CODET_PARQUET", "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet")
KAGGLE_AICD   = os.environ.get("AICD_DIR", "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench")

# --- data loading (identical to ext_gptsniffer.py) ---------------------------
def _is_human(t):
    return str(t or "").strip().lower() in {"human", "human_written", "human-generated"}

def _vocab(train):
    names = {str(r.get("model", "") or "").strip() for r in train
             if not _is_human(r.get("target", "")) and r.get("model", "")}
    return {n: i + 1 for i, n in enumerate(sorted(names))}

def _conv_codet(split, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code", "code"):
            v = r.get(f, "")
            if isinstance(v, str) and v.strip(): code = v; break
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

def _load_aicd(task="t2"):
    task_name = {"t1": "T1", "t2": "T2", "t3": "T3"}.get(task.lower())
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path): raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found")
    pf = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError("[aicd] STRICT: No parquet files")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0: return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]

# --- dataset -----------------------------------------------------------------
class FSDS(TD):
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42, cap=None, n_labels=None):
        self.data = data; self.tok = tok; self.seq_len = seq_len
        labels_all = list(self.data["label"])
        nl = (max(labels_all) + 1) if n_labels is None else n_labels
        if frac < 1.0 or cap is not None:
            rng = random.Random(seed); keep = []
            for lbl in range(nl):
                idx = [i for i, x in enumerate(labels_all) if x == lbl]
                if not idx: continue
                k = len(idx) if frac >= 1.0 else max(1, int(len(idx) * frac))
                if cap is not None: k = min(k, cap)
                keep.extend(rng.sample(idx, min(k, len(idx))))
            self.data = self.data.select(sorted(keep))

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        enc = self.tok(r["code"][:5000], padding="max_length", max_length=self.seq_len, truncation=True)
        return {"input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "label": int(r["label"])}

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; enc: str = "unixcoder-base"
    frac: float = 0.20; seed: int = 42
    bs: int = 32; seq: int = 512; epochs: int = 6
    lr_enc: float = 5e-5; warmup: float = 0.10; wd: float = 0.01
    device: str = "cuda"; smoke: bool = False

def adaptive_schedule(cfg):
    f = cfg.frac
    if f <= 0.02: cfg.epochs, cfg.lr_enc, cfg.warmup = 10, 5e-5, 0.20
    elif f <= 0.10: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 5e-5, 0.15
    else: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 5e-5, 0.10
    if cfg.smoke: cfg.epochs = 1
    return cfg

def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True; torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        cfg.bs = 64 if mem >= 40 else (32 if mem >= 20 else 16)
    else:
        cfg.device = "cpu"; cfg.bs = 8
    return cfg

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

@torch.no_grad()
def _predict(model, loader, cfg):
    model.eval(); preds, labels = [], []
    for b in loader:
        ids = b["input_ids"].to(cfg.device); mask = b["attention_mask"].to(cfg.device)
        out = model(input_ids=ids, attention_mask=mask)
        preds.extend(out.logits.argmax(-1).cpu().tolist())
        labs = b["label"]; labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
    return np.array(preds), np.array(labels)

def train_ce(tr_ds, vl_ds, n_cls, cfg):
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(KAGGLE_MODELS, cfg.enc), num_labels=n_cls, local_files_only=True).to(cfg.device)
    ldr = dict(batch_size=cfg.bs, num_workers=2, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **ldr); vl_dl = DataLoader(vl_ds, shuffle=False, **ldr)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_enc, weight_decay=cfg.wd)
    sch = get_linear_schedule_with_warmup(opt, int(total * cfg.warmup), total)
    scaler = GradScaler(enabled=(cfg.device == "cuda"))
    best_val, best_state = -1.0, None
    for ep in range(cfg.epochs):
        model.train()
        for b in tqdm(tr_dl, desc=f"ep{ep+1}", leave=False):
            ids = b["input_ids"].to(cfg.device); mask = b["attention_mask"].to(cfg.device)
            labs = b["label"]; labs = labs if torch.is_tensor(labs) else torch.tensor(labs)
            labs = labs.long().to(cfg.device)
            opt.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
                out = model(input_ids=ids, attention_mask=mask, labels=labs); loss = out.loss
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sch.step()
        vp, vl = _predict(model, vl_dl, cfg)
        v = float(f1_score(vl, vp, average="macro", zero_division=0))
        if v > best_val:
            best_val = v; best_state = {k: t.cpu().clone() for k, t in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    return model, best_val

def _remap(data, keep_labels):
    contig2orig = sorted(keep_labels)
    orig2contig = {o: i for i, o in enumerate(contig2orig)}
    kept = data.filter(lambda x: x["label"] in orig2contig)
    kept = kept.map(lambda x: {"label": orig2contig[x["label"]]})
    return kept, contig2orig

def run_bench(cfg, human_class, k_list, repeats):
    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet(); vocab = _vocab(tr_raw)
        tr = _conv_codet(tr_raw, vocab); vl = _conv_codet(vl_raw, vocab); ts = _conv_codet(ts_raw, vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr = _conv_aicd(tr_raw); vl = _conv_aicd(vl_raw); ts = _conv_aicd(ts_raw)
    all_labels = sorted(set(tr["label"]))
    generators = [c for c in all_labels if c != human_class]
    n_gen = len(generators)
    k_list = [k for k in k_list if 2 <= k <= n_gen]
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    cap = 40 if cfg.smoke else None
    logger.info(f"[{cfg.benchmark}] generators={generators} human={human_class} k_list={k_list} repeats={repeats}")
    records = []
    for K in k_list:
        for rep in range(repeats):
            rng = random.Random(cfg.seed * 1000 + K * 10 + rep)
            sel_gen = sorted(rng.sample(generators, K))
            keep = ([human_class] if human_class is not None else []) + sel_gen
            n_cls = len(keep)
            tr_k, c2o = _remap(tr, keep); vl_k, _ = _remap(vl, keep); ts_k, _ = _remap(ts, keep)
            tr_ds = FSDS(tr_k, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, cap=cap, n_labels=n_cls)
            vl_ds = FSDS(vl_k, tok, cfg.seq, frac=1.0, seed=cfg.seed + 1, cap=cap, n_labels=n_cls)
            ts_ds = FSDS(ts_k, tok, cfg.seq, frac=1.0, seed=cfg.seed + 2, cap=cap, n_labels=n_cls)
            logger.info(f"--- K={K} rep={rep} n_cls={n_cls} classes={keep} train={len(tr_ds)} ---")
            model, best_val = train_ce(tr_ds, vl_ds, n_cls, cfg)
            ts_dl = DataLoader(ts_ds, shuffle=False, batch_size=cfg.bs, num_workers=2, pin_memory=True)
            tp, tl = _predict(model, ts_dl, cfg)
            macro = float(f1_score(tl, tp, average="macro", zero_division=0))
            acc = float(accuracy_score(tl, tp))
            per_class = f1_score(tl, tp, average=None, zero_division=0, labels=list(range(n_cls))).tolist()
            rec = {"tag": f"exp161_{cfg.benchmark}_K{K}_r{rep}", "bench": cfg.benchmark, "frac": cfg.frac,
                   "K_generators": K, "include_human": human_class is not None, "n_cls_total": n_cls,
                   "classes_orig": keep, "selected_generators": sel_gen,
                   "macro": macro, "acc": acc, "val_macro": best_val,
                   "per_class_f1": per_class, "contig2orig": c2o,
                   "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
            logger.info(f"  K={K} rep={rep}: macro={macro:.4f} acc={acc:.4f} val={best_val:.4f}")
            records.append(rec)
            del model
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    # per-K aggregate summary
    for K in k_list:
        rr = [r for r in records if r["K_generators"] == K]
        if not rr: continue
        summ = {"tag": f"exp161_{cfg.benchmark}_K{K}_MEAN", "bench": cfg.benchmark, "frac": cfg.frac,
                "K_generators": K, "n_repeats": len(rr),
                "macro_mean": float(np.mean([r["macro"] for r in rr])),
                "macro_std": float(np.std([r["macro"] for r in rr])),
                "acc_mean": float(np.mean([r["acc"] for r in rr])),
                "acc_std": float(np.std([r["acc"] for r in rr]))}
        records.append(summ)
    return records

def main():
    bench = os.environ.get("BENCH", "codet_m4")
    frac = float(os.environ.get("FRAC", "0.20"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    seed = int(os.environ.get("SEED", "42"))
    repeats = int(os.environ.get("REPEATS", "2"))
    include_human = os.environ.get("INCLUDE_HUMAN", "1") == "1"
    kl_env = os.environ.get("KLIST", "")
    benches = [bench] if bench != "all" else ["codet_m4", "aicd_t2"]
    all_records = []
    for b in benches:
        cfg = Cfg(benchmark=b, frac=frac, seed=seed, smoke=smoke)
        cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
        human_class = (0 if b == "codet_m4" else None) if include_human else None
        if kl_env:
            k_list = [int(x) for x in kl_env.split(",") if x.strip()]
        else:
            k_list = [2, 3] if smoke else ([2, 3, 4, 5] if b == "codet_m4" else [2, 4, 6, 8, 10, 12])
        if smoke: repeats = 1
        try:
            all_records.extend(run_bench(cfg, human_class, k_list, repeats))
        except Exception as e:
            logger.error(f"[{b}] FAILED: {e}")
            import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp161_ksweep_results.json"), "w") as f:
        json.dump(all_records, f, indent=2)
    logger.info(f"[done] wrote {len(all_records)} records")

if __name__ == "__main__":
    main()
