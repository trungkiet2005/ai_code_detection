# ext_llmsniffer — Faithful K-class reproduction of LLMSniffer (arXiv 2024)
# =============================================================================
# NAME       : LLMSniffer-K  (K-class author attribution adaptation)
# UPSTREAM   : Dihan & Muhtasim, "LLMSniffer: Detecting LLM-Generated Code via
#              GraphCodeBERT and Supervised Contrastive Learning", arXiv 2024
#              https://github.com/mahirlabibdihan/LLMSniffer
# FAITHFULNESS (from cpsniffer.ipynb):
#   Encoder  : microsoft/graphcodebert-base (→ we use unixcoder-base, protocol)
#   Architecture (CodeBERTBinaryClassifier):
#     encoder(input_ids, attn_mask) → last_hidden_state[:, 0, :]  (CLS token)
#     classifier = Sequential(
#         Dropout(0.3),
#         Linear(hidden, 128), BatchNorm1d(128), ReLU(),
#         Dropout(0.3),
#         Linear(128, 1)  ← binary; we extend to Linear(128, K)
#     )
#   Loss     : SupConLoss (temperature=0.07) on CLS embeddings ONLY
#              CRITICAL: Original notebook cell 9 uses cls_output.DETACH() for classifier!
#                line 408: `logits = self.classifier(cls_output.detach()).squeeze(-1)`
#              → CE gradient does NOT flow to encoder. Encoder trained ONLY via SupCon.
#              → Classifier trained ONLY via CE on detached features.
#              We implement this faithfully:
#                Phase 1 per batch: SupCon(CLS) → encoder gradient only
#                Phase 2 per batch: CE(classifier(CLS.detach())) → head gradient only
#              This is equivalent to the notebook's cls_output.detach().
#   SupConLoss formula (exact copy from notebook):
#     mask = (labels_i == labels_j)
#     logits = (features_norm @ features_norm.T) / temperature
#     log_prob = logits - log(sum(exp(logits) * mask))
#     loss = -mean(sum(log_prob * mask, dim=1) / (mask_sum + eps))
#   Optimizer: AdamW with differential LR:
#     encoder params: lr=1e-6
#     classifier params: lr=1e-4
#     weight_decay=1e-2
#   Training epochs: 35 (original); we use RAS with fewer for few-shot
#   Batch_size: 16
# CHANGES vs original:
#   - GraphCodeBERT → UniXcoder-base (protocol parity)
#   - Binary (Human/AI) → K-class attribution
#   - Loss: SupCon only → SupCon + CE (joint; necessary for K-class)
#   - Data: GPTSniffer/Whodunit datasets → CoDET-M4 / AICD-T2
#   - Epochs: 35 → RAS schedule
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
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset as TD, DataLoader
from sklearn.metrics import f1_score

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"
PAPER_F1      = 0.6633

import logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger("ext_llmsniffer")

@dataclass
class Cfg:
    benchmark: str  = "codet_m4"
    task:      str  = "author_iid"
    frac:      float = 0.20
    n_cls:     int  = 6
    seed:      int  = 42
    seq:       int  = 512
    bs:        int  = 16            # original bs=16 (notebook line ~301)
    epochs:    int  = 10
    lr_enc:    float = 1e-6         # original encoder lr=1e-6 (notebook)
    lr_head:   float = 1e-4         # original classifier lr=1e-4
    wd:        float = 1e-2         # original weight_decay=1e-2
    temperature: float = 0.07       # original SupConLoss temperature
    warmup:    float = 0.10
    device:    str  = "cuda"

    def __post_init__(self):
        # RAS
        if self.frac <= 0.02:
            self.epochs, self.warmup = 15, 0.20
        elif self.frac <= 0.10:
            self.epochs, self.warmup = 10, 0.15
        else:
            self.epochs, self.warmup = 8, 0.10
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

# ── Tokenisation ──────────────────────────────────────────────────────────────
def _tokenize(code, tokenizer, max_len):
    toks = tokenizer.tokenize(" ".join(code.split()))[:max_len-4]
    toks = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + toks + [tokenizer.sep_token]
    ids  = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_len - len(ids))
    return ids[:max_len]

class CodeDataset(TD):
    def __init__(self, recs, tok, seq):
        self.recs, self.tok, self.seq = recs, tok, seq
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        ids = _tokenize(r["code"], self.tok, self.seq)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(r["label"], dtype=torch.long)

# ── SupConLoss (exact copy from cpsniffer.ipynb lines 335-363) ───────────────
class SupConLoss(nn.Module):
    """
    Faithful copy from LLMSniffer cpsniffer.ipynb:
      mask = (labels_i == labels_j)  [binary label mask — NxN]
      features_normalized = features / features.norm(dim=1, keepdim=True)
      logits = (features_norm @ features_norm.T) / temperature
      logits -= max(logits, dim=1).values.detach()   [numerical stability]
      exp_logits = exp(logits) * mask
      log_prob = logits - log(exp_logits.sum(dim=1) + 1e-12)
      loss = -mean( sum(log_prob * mask, dim=1) / (mask_sum + 1e-12) )
    """
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        batch_size = features.size(0)
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        features_normalized = F.normalize(features, dim=1)
        logits = (features_normalized @ features_normalized.T) / self.temperature
        logits_max = torch.max(logits, dim=1, keepdim=True).values
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * mask
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mask_sum = mask.sum(dim=1)
        mean_log_prob_pos = (log_prob * mask).sum(dim=1) / (mask_sum + 1e-12)
        return -mean_log_prob_pos.mean()

# ── Model: CodeBERTBinaryClassifier → K-class (faithful adaptation) ──────────
class LLMSnifferK(nn.Module):
    """
    Faithful to cpsniffer.ipynb CodeBERTBinaryClassifier:
      encoder → CLS token → Dropout(0.3) → Linear(H,128) → BN → ReLU
              → Dropout(0.3) → Linear(128, K)  [binary→K-class]
    Training (FAITHFUL to notebook cell 9, line 408):
      cls_output = encoder(...).last_hidden_state[:, 0, :]   # raw CLS
      logits = classifier(cls_output.detach())               # DETACH → CE only trains head
      loss_supcon = SupCon(cls_output)                       # SupCon trains encoder only
      → Encoder trained ONLY via SupConLoss.
      → Classifier trained ONLY via CE on detached CLS.
    """
    def __init__(self, encoder, hidden, n_cls, pad_id):
        super().__init__()
        self.encoder = encoder
        self.pad_id  = pad_id
        # Faithful classifier head (notebook cell 9)
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_cls),   # K-class (binary→K)
        )

    def _cls(self, input_ids):
        mask = input_ids.ne(self.pad_id)
        out  = self.encoder(input_ids, attention_mask=mask, output_hidden_states=True)
        return out[0][:, 0, :]   # [CLS] token (notebook line 407)

    def forward(self, input_ids):
        cls = self._cls(input_ids)             # raw CLS — gradient alive for SupCon
        cls_detached = cls.detach()            # DETACH → faithful to notebook line 408
        logits = self.classifier(cls_detached) # CE only trains classifier, NOT encoder
        return logits, cls   # logits (detached) + cls (for SupCon loss on encoder)

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for ids, y in loader:
            ids, y = ids.to(device), y.to(device)
            logits, _ = model(ids)
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(y.cpu().tolist())
    return f1_score(labs, preds, average="macro", zero_division=0)

# ── Train ─────────────────────────────────────────────────────────────────────
def train_one(cfg, splits):
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, \
                              get_linear_schedule_with_warmup
    from torch.cuda.amp import GradScaler, autocast

    model_path = os.path.join(KAGGLE_MODELS, "unixcoder-base")
    tokenizer  = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    config     = RobertaConfig.from_pretrained(model_path, local_files_only=True)
    encoder    = RobertaModel.from_pretrained(model_path, local_files_only=True)
    pad_id     = tokenizer.pad_token_id

    model = LLMSnifferK(encoder, config.hidden_size, cfg.n_cls, pad_id).to(cfg.device)
    supcon = SupConLoss(temperature=cfg.temperature).to(cfg.device)

    tr_ds = CodeDataset(splits["train"], tokenizer, cfg.seq)
    vl_ds = CodeDataset(splits["val"],   tokenizer, cfg.seq)
    ts_ds = CodeDataset(splits["test"],  tokenizer, cfg.seq)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.bs,    shuffle=True,  num_workers=2, pin_memory=True)
    vl_dl = DataLoader(vl_ds, batch_size=cfg.bs*2,  shuffle=False, num_workers=2)
    ts_dl = DataLoader(ts_ds, batch_size=cfg.bs*2,  shuffle=False, num_workers=2)

    # Differential LR: encoder=1e-6, classifier=1e-4 (notebook cell 10)
    optimizer = torch.optim.AdamW([
        {"params": model.encoder.parameters(),    "lr": cfg.lr_enc},
        {"params": model.classifier.parameters(), "lr": cfg.lr_head},
    ], weight_decay=cfg.wd)
    total_steps  = cfg.epochs * len(tr_dl)
    warmup_steps = int(total_steps * cfg.warmup)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler()

    best_val, best_test = 0.0, 0.0
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        for ids, labs in tr_dl:
            ids, labs = ids.to(cfg.device), labs.to(cfg.device)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                logits, cls = model(ids)
                # FAITHFUL: cls is detached inside model.forward() for logits
                # CE gradient → classifier head ONLY (encoder frozen from CE)
                loss_ce  = F.cross_entropy(logits, labs)
                # SupCon gradient → encoder ONLY (cls is NOT detached here)
                loss_scl = supcon(cls, labs)
                # Two separate gradient paths (faithful to notebook detach trick)
                loss     = loss_ce + loss_scl
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
        "tag":          f"ext_llmsniffer_unixcoder-base_{cfg.benchmark}_f{cfg.frac}",
        "method":       "LLMSniffer-K",
        "upstream":     "arXiv 2024 (Dihan & Muhtasim)",
        "note":         ("GraphCodeBERT+SupCon+MLP head. SupConLoss(τ=0.07) trains encoder; "
                         "CE trains classifier via detached CLS (faithful to notebook cls_output.detach()). "
                         "Encoder lr=1e-6, head lr=1e-4 (differential, faithful to paper)."),
        "enc":          "unixcoder-base",
        "bench":        cfg.benchmark,
        "frac":         cfg.frac,
        "epochs":       cfg.epochs,
        "lr_enc":       cfg.lr_enc,
        "lr_head":      cfg.lr_head,
        "temperature":  cfg.temperature,
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

    out_path = "results/ext_llmsniffer_results.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved → {out_path}")
    print(f"\n{'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9}")
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} "
              f"{r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f}")

if __name__ == "__main__":
    main()
