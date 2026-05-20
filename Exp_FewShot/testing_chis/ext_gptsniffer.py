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
#     - Optimizer: AdamW, lr=5e-5, warmup_steps=500, weight_decay=0.01 (original)
#   Original training: 12 epochs, batch_size=32, lr=5e-5
#   Our adaptation: swap encoder (codebert → unixcoder), K-class head,
#     apply RAS schedule, AMP bf16, data from CoDET-M4 / AICD-T2.
# WHAT CHANGES vs original:
#   - Encoder: microsoft/codebert-base → unixcoder-base (protocol parity)
#   - Head: Linear(768, 2) → Linear(768, K)
#   - Loss: CE only (UNCHANGED — this is GPTSniffer's key simplicity claim)
#   - Schedule: 12ep/lr=5e-5/warmup=500 → RAS schedule (3 regime levels)
#     NOTE: RAS is more favorable to few-shot. Original GPTSniffer used full data.
#   - Data: their ChatGPT vs Human binary → CoDET-M4 K-class few-shot fractions
# NOTE ON ENCODER SWAP:
#   CodeBERT uses standard RoBERTa tokenisation (no <encoder_only> token).
#   UniXcoder requires CLS <encoder_only> SEP ... SEP format.
#   We keep CodeBERT tokenisation for UniXcoder to stay close to gptsniffer.py line 68
#   (standard encode_plus, max_length=512, padding='max_length', truncation=True).
# =============================================================================
from __future__ import annotations
import os, sys, time, json, random, subprocess, importlib.util
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
for _p in ("numpy", "torch", "datasets", "transformers", "scikit-learn", "tqdm"):
    _ensure(_p)

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset as TD, DataLoader
from sklearn.metrics import f1_score

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"
PAPER_F1      = 0.6633

import logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger("ext_gptsniffer")

@dataclass
class Cfg:
    benchmark: str  = "codet_m4"
    task:      str  = "author_iid"
    frac:      float = 0.20
    n_cls:     int  = 6
    seed:      int  = 42
    seq:       int  = 512          # max_length=512 from gptsniffer.py line 68
    bs:        int  = 32           # batch_size=32 from gptsniffer.py line 87
    epochs:    int  = 6
    lr_enc:    float = 5e-5        # learning_rate=5e-5 from gptsniffer.py line 104 (original)
    warmup:    float = 0.10
    device:    str  = "cuda"

    def __post_init__(self):
        # RAS schedule
        if self.frac <= 0.02:
            self.epochs, self.lr_enc, self.warmup = 10, 5e-5, 0.20
        elif self.frac <= 0.10:
            self.epochs, self.lr_enc, self.warmup = 6, 5e-5, 0.15
        else:
            self.epochs, self.lr_enc, self.warmup = 6, 5e-5, 0.10
        # original uses lr=5e-5; we keep it across all fractions (faithful)
        try:
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if mem >= 40: self.bs = 64
            elif mem >= 20: self.bs = 32
            else: self.bs = 16
        except: pass

# ── Data loading ──────────────────────────────────────────────────────────────
def _load_codet(cfg):
    import pandas as pd
    df = pd.read_parquet(KAGGLE_CODET)
    label_col = "author_id" if cfg.task == "author_iid" else "author_ood_id"
    df = df[df[label_col].notna()].copy()
    df["label"] = df[label_col].astype(int)
    cfg.n_cls = df["label"].nunique()
    splits = {}
    for sp in ("train","val","test"):
        sub = df[df["split"]==sp].copy()
        if sp == "train":
            rng = random.Random(cfg.seed)
            keep = []
            for lbl in sub["label"].unique():
                idx = sub[sub["label"]==lbl].index.tolist()
                keep.extend(rng.sample(idx, max(1, int(len(idx)*cfg.frac))))
            sub = sub.loc[keep]
        splits[sp] = sub[["code","label"]].to_dict("records")
    return splits

def _load_aicd(cfg):
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(KAGGLE_AICD, "T2"))
    splits = {}
    for sp, key in [("train","train"),("val","validation"),("test","test")]:
        data = [{"code": r["code"], "label": int(r["label"])} for r in ds[key]]
        if sp == "train":
            rng = random.Random(cfg.seed)
            by_cls = {}
            for d in data:
                by_cls.setdefault(d["label"],[]).append(d)
            keep = []
            for lst in by_cls.values():
                keep.extend(rng.sample(lst, max(1, int(len(lst)*cfg.frac))))
            data = keep
        splits[sp] = data
    cfg.n_cls = len({d["label"] for d in splits["train"]})
    return splits

def load_data(cfg):
    return _load_codet(cfg) if cfg.benchmark == "codet_m4" else _load_aicd(cfg)

# ── Dataset: faithful to gptsniffer.py tokenize (standard encode_plus) ───────
class CodeDataset(TD):
    """
    Faithful to gptsniffer.py CodeDataset.__getitem__ (lines 68-71):
      tokenizer.encode_plus(code, padding='max_length', max_length=512, truncation=True)
    """
    def __init__(self, recs, tokenizer, seq):
        self.recs = recs
        self.tok  = tokenizer
        self.seq  = seq

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        enc = self.tok.encode_plus(
            r["code"],
            padding="max_length",
            max_length=self.seq,
            truncation=True,
        )
        return (
            torch.tensor(enc["input_ids"],      dtype=torch.long),
            torch.tensor(enc["attention_mask"], dtype=torch.long),
            torch.tensor(r["label"],            dtype=torch.long),
        )

# ── Model: CodeBERT-style AutoModelForSequenceClassification, K-class ─────────
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for ids, mask, y in loader:
            ids, mask, y = ids.to(device), mask.to(device), y.to(device)
            out = model(input_ids=ids, attention_mask=mask)
            preds.extend(out.logits.argmax(-1).cpu().tolist())
            labs.extend(y.cpu().tolist())
    return f1_score(labs, preds, average="macro", zero_division=0)

def train_one(cfg, splits):
    from transformers import (AutoTokenizer,
                               AutoModelForSequenceClassification,
                               get_linear_schedule_with_warmup)
    from torch.cuda.amp import GradScaler, autocast

    model_path = os.path.join(KAGGLE_MODELS, "unixcoder-base")
    # Use AutoTokenizer (standard RoBERTa subword, faithful to gptsniffer line 48)
    tokenizer  = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    # AutoModelForSequenceClassification = CE head (gptsniffer lines 49, 113)
    model      = AutoModelForSequenceClassification.from_pretrained(
        model_path, num_labels=cfg.n_cls, local_files_only=True
    ).to(cfg.device)

    tr_ds = CodeDataset(splits["train"], tokenizer, cfg.seq)
    vl_ds = CodeDataset(splits["val"],   tokenizer, cfg.seq)
    ts_ds = CodeDataset(splits["test"],  tokenizer, cfg.seq)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.bs,    shuffle=True,  num_workers=2, pin_memory=True)
    vl_dl = DataLoader(vl_ds, batch_size=cfg.bs*2,  shuffle=False, num_workers=2)
    ts_dl = DataLoader(ts_ds, batch_size=cfg.bs*2,  shuffle=False, num_workers=2)

    total_steps  = cfg.epochs * len(tr_dl)
    warmup_steps = int(total_steps * cfg.warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_enc, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler()

    best_val, best_test = 0.0, 0.0
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        for ids, mask, labs in tr_dl:
            ids, mask, labs = ids.to(cfg.device), mask.to(cfg.device), labs.to(cfg.device)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                out  = model(input_ids=ids, attention_mask=mask, labels=labs)
                loss = out.loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()

        val_f1 = evaluate(model, vl_dl, cfg.device)
        if val_f1 >= best_val:
            best_val  = val_f1
            best_test = evaluate(model, ts_dl, cfg.device)
        logger.info(f"[ep{ep+1}] val={val_f1:.4f} best_test={best_test:.4f}")

    return {
        "tag":          f"ext_gptsniffer_unixcoder-base_{cfg.benchmark}_f{cfg.frac}",
        "method":       "GPTSniffer-K",
        "upstream":     "JSS 2024 (Nguyen et al.)",
        "note":         "CE-only head (AutoModelForSequenceClassification). Encoder: codebert-base → unixcoder-base.",
        "enc":          "unixcoder-base",
        "bench":        cfg.benchmark,
        "frac":         cfg.frac,
        "epochs":       cfg.epochs,
        "lr_enc":       cfg.lr_enc,
        "val_macro":    best_val,
        "macro":        best_test,
        "val_test_gap": best_val - best_test,
        "dpaper":       best_test - PAPER_F1,
        "wall":         round(time.time() - t0, 1),
    }

def main():
    results = []
    for bench, task, n_cls in [
        ("codet_m4", "author_iid", 6),
        ("aicd_t2",  "model_family", 12),
    ]:
        for frac in [0.01, 0.05, 0.20]:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            splits = load_data(cfg)
            logger.info(f"\n{'='*60}\n{bench} frac={frac} n_cls={cfg.n_cls} "
                        f"train={len(splits['train'])}\n{'='*60}")
            rec = train_one(cfg, splits)
            results.append(rec)
            logger.info(f"  val={rec['val_macro']:.4f}  test={rec['macro']:.4f}  "
                        f"gap={rec['val_test_gap']:+.4f}  Δpaper={rec['dpaper']:+.4f}")

    out_path = "results/ext_gptsniffer_results.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved → {out_path}")
    print(f"\n{'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9}")
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} "
              f"{r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f}")

if __name__ == "__main__":
    main()
