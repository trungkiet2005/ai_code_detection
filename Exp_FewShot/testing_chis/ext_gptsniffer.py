# ext_gptsniffer — Faithful K-class reproduction of GPTSniffer (JSS 2024)
# =============================================================================
# NAME       : GPTSniffer-K  (K-class author attribution adaptation)
# UPSTREAM   : Nguyen et al., "GPTSniffer: A CodeBERT-based classifier to detect
#              source code written by ChatGPT", Journal of Systems and Software,
#              Volume 214, 2024.
#              https://github.com/MDEGroup/GPTSniffer
# FAITHFULNESS:
#   Architecture (gptsniffer.py):
#     - Encoder: microsoft/codebert-base (AutoModelForSequenceClassification)
#       → We swap to unixcoder-base to match our protocol; keep same arch.
#     - Loss: CE only (Trainer with standard HuggingFace cross-entropy)
#     - Head: Linear(hidden, 2) binary → Linear(hidden, K) K-class
#     - Tokenisation: standard RoBERTa subword, max_length=512, padding='max_length'
#       (gptsniffer.py line 68: tokenizer.encode_plus)
#     - Optimizer: AdamW, lr=5e-5, warmup_steps=500, weight_decay=0.01 (original)
#   Original training: 12 epochs, batch_size=32, lr=5e-5
#   Our adaptation: swap encoder (codebert → unixcoder), K-class head,
#     apply RAS schedule, AMP bf16, data from CoDET-M4 / AICD-T2.
# WHAT CHANGES vs original:
#   - Encoder: microsoft/codebert-base → unixcoder-base (protocol parity)
#   - Head: Linear(768, 2) → Linear(768, K)
#   - Loss: CE only (UNCHANGED — this is GPTSniffer's key simplicity claim)
#   - Data: their ChatGPT vs Human binary → CoDET-M4 K-class few-shot fractions
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from dataclasses import dataclass, field
from typing import List, Dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("ext_gptsniffer")

PAPER_BASELINE = 0.6633

# =============================================================================
# Config
# =============================================================================
@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 32; seq: int = 512; epochs: int = 6
    lr_enc: float = 5e-5          # faithful: gptsniffer.py line 104
    warmup: float = 0.10; wd: float = 0.01  # faithful: weight_decay=0.01
    device: str = "cuda"

def adaptive_schedule(cfg):
    f = cfg.frac
    if f <= 0.02: cfg.epochs, cfg.lr_enc, cfg.warmup = 10, 5e-5, 0.20
    elif f <= 0.10: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 5e-5, 0.15
    else: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 5e-5, 0.10
    return cfg

def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40: cfg.bs = 64
        elif mem >= 20: cfg.bs = 32
        else: cfg.bs = 16
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} seq={cfg.seq}")
    return cfg

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# =============================================================================
# Data loading (identical to exp84_cargo pattern)
# =============================================================================
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD  = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

def _is_human(t):
    return str(t or "").strip().lower() in {"human", "human_written", "human-generated"}

def _vocab(train):
    names = {str(r.get("model", "") or "").strip() for r in train
             if not _is_human(r.get("target", "")) and r.get("model", "")}
    return {n: i + 1 for i, n in enumerate(sorted(names))}

def _conv_codet(split, task, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code", "code"):
            v = r.get(f, "")
            if isinstance(v, str) and v.strip(): code = v; break
        if task == "binary":
            label = 0 if _is_human(r.get("target", "")) else 1
        else:
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

def _load_aicd(task):
    task_name = {"t1": "T1", "t2": "T2", "t3": "T3"}.get(task.lower())
    if task_name is None: raise ValueError(f"[aicd] Unknown task '{task}'")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path): raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found")
    pf = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: No parquet files")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0: return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]

# =============================================================================
# Dataset
# =============================================================================
class FSDS(TD):
    """Faithful to gptsniffer.py CodeDataset — standard encode_plus."""
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42):
        self.data = data; self.tok = tok; self.seq_len = seq_len
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        # Faithful to gptsniffer.py line 68:
        # tokenizer.encode_plus(code, padding='max_length', max_length=512, truncation=True)
        enc = self.tok.encode_plus(code, padding="max_length", max_length=self.seq_len, truncation=True)
        return {"input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "label": r["label"],
                "language": r.get("language", ""),
                "source": r.get("source", "")}

# =============================================================================
# Eval (rich metrics, same as exp84_cargo)
# =============================================================================
@torch.no_grad()
def eval_pack(model, loader, cfg):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["input_ids"].to(cfg.device)
        mask = b["attention_mask"].to(cfg.device)
        out = model(input_ids=ids, attention_mask=mask)
        preds.extend(out.logits.argmax(-1).cpu().tolist())
        labs = b["label"]
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        langs.extend(list(b.get("language", [""] * len(labs))))
        sources.extend(list(b.get("source", [""] * len(labs))))
    preds = np.array(preds); labels = np.array(labels); n_cls = cfg.n_cls
    overall = {"accuracy": float(accuracy_score(labels, preds)),
               "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
               "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
               "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
               "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0))}
    per_class = {"f1": f1_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
                 "precision": precision_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
                 "recall": recall_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist()}
    cm = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    per_lang, per_src = {}, {}
    if any(l for l in langs):
        la = np.array(langs)
        for L in sorted(set(langs)):
            if not L: continue
            sel = (la == L)
            if sel.sum() < 2: continue
            per_lang[L] = {"n": int(sel.sum()),
                           "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0)),
                           "accuracy": float(accuracy_score(labels[sel], preds[sel]))}
    return {"overall": overall, "per_class": per_class, "per_language": per_lang,
            "per_source": per_src, "confusion_matrix": cm.tolist(), "n_samples": int(len(labels))}

# =============================================================================
# Train
# =============================================================================
def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, "author", vocab)
        vl_data = _conv_codet(vl_raw, "author", vocab)
        ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
    cfg.n_cls = max(tr_data["label"]) + 1
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2)

    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    # Faithful to gptsniffer.py: AutoModelForSequenceClassification + CE
    model = AutoModelForSequenceClassification.from_pretrained(
        os.path.join(KAGGLE_MODELS, cfg.enc), num_labels=cfg.n_cls, local_files_only=True
    ).to(cfg.device)

    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    warmup_steps = int(total_steps * cfg.warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_enc, weight_decay=cfg.wd)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler = GradScaler()

    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr={cfg.lr_enc} "
                f"warmup={cfg.warmup} n_cls={cfg.n_cls} train={len(tr_ds)}")

    best_val, best_state, val_hist = 0.0, None, []
    for ep in range(cfg.epochs):
        model.train(); tot_loss = 0.0
        for b in tqdm(tr_dl, desc=f"Train ep{ep+1}"):
            ids = b["input_ids"].to(cfg.device)
            mask = b["attention_mask"].to(cfg.device)
            labs = b["label"]
            if not torch.is_tensor(labs): labs = torch.tensor(labs, dtype=torch.long)
            labs = labs.to(cfg.device)
            optimizer.zero_grad()
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
                out = model(input_ids=ids, attention_mask=mask, labels=labs)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()
            tot_loss += loss.item()
        avg_loss = tot_loss / max(1, len(tr_dl))
        val_met = eval_pack(model, vl_dl, cfg)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {ep+1}] loss={avg_loss:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg)
    test_macro = ts_met["overall"]["macro_f1"]; gap = best_val - test_macro
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f}")

    return {"tag": tag, "method": "GPTSniffer-K",
            "upstream": "JSS 2024 (Nguyen et al.)",
            "note": "CE-only head (AutoModelForSequenceClassification). Faithful: lr=5e-5, wd=0.01, encode_plus.",
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "test_metrics": ts_met, "val_history": val_hist,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

# =============================================================================
# Main
# =============================================================================
def main():
    encoders = ["unixcoder-base"]
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for enc in encoders:
        for bench, task, n_cls in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac, n_cls=n_cls)
                tag = f"ext_gptsniffer_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ext_gptsniffer_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*100)
    print(f"{'Enc':<22} {'Bench':<12} {'Frac':>6} {'Ep':>4} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9} {'Wall':>8}")
    print("-"*100)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} "
              f"{r['wall']:>8.0f}s")
    print("="*100)

if __name__ == "__main__":
    main()
