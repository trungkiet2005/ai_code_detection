# ext_codegptsensor — Faithful K-class reproduction of CodeGPTSensor (TOSEM 2025)
# =============================================================================
# NAME       : CodeGPTSensor-K  (multi-class extension)
# UPSTREAM   : Xu et al., "Distinguishing LLM-generated from Human-written Code
#              by Contrastive Learning", TOSEM 2025
#              https://github.com/doriscullen/CodeGPTSensor
# FAITHFULNESS:
#   - Architecture: UniXcoder → mean-pool → Linear(hidden, K) [was Linear(hidden,2)]
#   - Loss (from model.py line 40-53):
#       loss = CE(logits, y)                         # cross-entropy
#             + 0.1 * KL(view1, view2)               # R-Drop on two dropout views
#             + 0.2 * cosine_neg(embed, neg_embed)   # push away in-batch cross-class neg
#   - KL: symmetric KL between two dropout passes of SAME input (get_kl_loss)
#   - Cosine neg: cosine_embedding_loss with label=-1 (all are negatives)
#     In original: contrast_ids = a hard-negative code from a different sample.
#     K-class adaptation: contrast = random in-batch sample with different label.
#   - Encoder: UniXcoder-base (original also uses UniXcoder)
#   - Tokenisation: CLS <encoder_only> SEP ... SEP  (original format preserved)
# WHAT CHANGES vs original:
#   - head: Linear(hidden, 2) → Linear(hidden, K)
#   - contrast mining: from dataset-level JSONL pairs → in-batch cross-class pairs
#   - training protocol: our RAS schedule (not their fixed 5ep / lr=5e-5)
# =============================================================================
from __future__ import annotations
import os, sys, time, json, random, subprocess, importlib.util, warnings
from dataclasses import dataclass, field
from typing import List

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

# ── Kaggle paths ──────────────────────────────────────────────────────────────
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"
PAPER_F1      = 0.6633   # UniXcoder full-data CoDET-M4 reference

# ── Logging ───────────────────────────────────────────────────────────────────
import logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger("ext_codegptsensor")

# ── Config ────────────────────────────────────────────────────────────────────
@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task:      str = "author_iid"
    frac:      float = 0.20
    n_cls:     int = 6
    seed:      int = 42
    seq:       int = 512
    bs:        int = 64
    epochs:    int = 6
    lr_enc:    float = 3e-5
    warmup:    float = 0.10
    # CodeGPTSensor-specific
    lambda_kl:   float = 0.1    # KL R-Drop coefficient (from model.py line 53)
    lambda_cos:  float = 0.2    # cosine neg coefficient (from model.py line 53)
    device:    str = "cuda"

    def __post_init__(self):
        # Regime-adaptive schedule (same as all our experiments)
        if self.frac <= 0.02:
            self.epochs, self.lr_enc, self.warmup = 10, 3e-5, 0.20
        elif self.frac <= 0.10:
            self.epochs, self.lr_enc, self.warmup = 6, 3e-5, 0.15
        else:
            self.epochs, self.lr_enc, self.warmup = 6, 4e-5, 0.10
        # Auto batch-size from GPU memory
        try:
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if mem >= 40: self.bs = 128
            elif mem >= 20: self.bs = 64
            else: self.bs = 32
        except: pass
        logger.info(f"[cfg] frac={self.frac} ep={self.epochs} lr={self.lr_enc} bs={self.bs}")

# ── Data loading (identical to our other experiments) ─────────────────────────
def _load_codet(cfg: Cfg):
    import pandas as pd
    df = pd.read_parquet(KAGGLE_CODET)
    label_col = "author_id" if cfg.task == "author_iid" else "author_ood_id"
    df = df[df[label_col].notna()].copy()
    df["label"] = df[label_col].astype(int)
    cfg.n_cls = df["label"].nunique()
    splits = {}
    for sp in ("train","val","test"):
        sub = df[df["split"] == sp].copy()
        if sp == "train":
            rng = random.Random(cfg.seed)
            keep = []
            for lbl in sub["label"].unique():
                idx = sub[sub["label"]==lbl].index.tolist()
                keep.extend(rng.sample(idx, max(1, int(len(idx)*cfg.frac))))
            sub = sub.loc[keep]
        splits[sp] = sub[["code","label"]].to_dict("records")
    return splits

def _load_aicd(cfg: Cfg):
    from datasets import load_from_disk
    ds = load_from_disk(os.path.join(KAGGLE_AICD, "T2"))
    splits = {}
    for sp, key in [("train","train"),("val","validation"),("test","test")]:
        data = [{"code": r["code"], "label": int(r["label"])} for r in ds[key]]
        if sp == "train":
            rng = random.Random(cfg.seed)
            by_cls = {}
            for d in data:
                by_cls.setdefault(d["label"], []).append(d)
            keep = []
            for lst in by_cls.values():
                keep.extend(rng.sample(lst, max(1, int(len(lst)*cfg.frac))))
            data = keep
        splits[sp] = data
    cfg.n_cls = len({d["label"] for d in splits["train"]})
    return splits

def load_data(cfg: Cfg):
    if cfg.benchmark == "codet_m4":
        return _load_codet(cfg)
    return _load_aicd(cfg)

# ── Tokenisation (CodeGPTSensor format: CLS <encoder_only> SEP code SEP) ─────
def _tokenize(code: str, tokenizer, max_len: int) -> list:
    """Faithful to CodeGPTSensor convert_examples_to_features."""
    code_tok = tokenizer.tokenize(" ".join(code.split()))[:max_len - 4]
    toks = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + \
           code_tok + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_len - len(ids))
    return ids[:max_len]

class CodeDataset(TD):
    def __init__(self, records, tokenizer, seq):
        self.recs = records
        self.tok  = tokenizer
        self.seq  = seq

    def __len__(self): return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        ids = _tokenize(r["code"], self.tok, self.seq)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(r["label"], dtype=torch.long)

# ── Model: exact CodeGPTSensor architecture, K-class head ─────────────────────
def _get_xcode_vec(encoder, input_ids, pad_id):
    """Mean-pool over non-padding tokens (faithful to model.py get_xcode_vec)."""
    mask = input_ids.ne(pad_id)
    # UniXcoder uses causal mask like original: mask.unsqueeze(1) * mask.unsqueeze(2)
    attn_mask = mask.unsqueeze(1) * mask.unsqueeze(2)
    out = encoder(input_ids, attention_mask=attn_mask, output_hidden_states=True)
    token_emb = out[0]
    vec = (token_emb * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1).clamp(min=1)
    return vec

def _kl_loss(p, q):
    """Symmetric KL (get_kl_loss from model.py lines 64-77)."""
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none').sum()
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none').sum()
    return (p_loss + q_loss) / 2

def _cosine_neg_loss(vec, contrast_vec):
    """cosine_embedding_loss with label=-1 (all negatives, from model.py lines 58-61)."""
    labels = torch.full((vec.size(0),), -1, dtype=torch.float, device=vec.device)
    return F.cosine_embedding_loss(vec, contrast_vec, labels)

def build_in_batch_contrast(vec, y):
    """For K-class: pair each sample with a random OTHER-class sample in the batch."""
    B = vec.size(0)
    contrast = vec.clone()
    for i in range(B):
        other = [j for j in range(B) if y[j] != y[i]]
        if other:
            j = random.choice(other)
            contrast[i] = vec[j].detach()
    return contrast

class CodeGPTSensorK(nn.Module):
    def __init__(self, encoder, hidden, n_cls, pad_id):
        super().__init__()
        self.encoder   = encoder
        self.pad_id    = pad_id
        self.classifier = nn.Linear(hidden, n_cls)   # was Linear(hidden, 2)

    def forward(self, input_ids, labels=None, do_contrast=True):
        vec  = _get_xcode_vec(self.encoder, input_ids, self.pad_id)
        logits = self.classifier(vec)
        loss_ce = F.cross_entropy(logits, labels)

        if do_contrast:
            # KL R-Drop: two dropout forward passes
            vec2   = _get_xcode_vec(self.encoder, input_ids, self.pad_id)
            logits2 = self.classifier(vec2)
            loss_kl = _kl_loss(logits.float(), logits2.float())
            # Cosine neg: in-batch cross-class negatives
            contrast = build_in_batch_contrast(vec, labels)
            loss_cos  = _cosine_neg_loss(vec, contrast)
            # Final loss (coefficients from model.py line 53)
            loss = loss_ce + 0.1 * loss_kl + 0.2 * loss_cos
        else:
            loss = loss_ce

        return loss, logits

# ── Train / evaluate ──────────────────────────────────────────────────────────
def evaluate(model, loader, device, pad_id):
    model.eval()
    all_preds, all_labs = [], []
    with torch.no_grad():
        for ids, labs in loader:
            ids, labs = ids.to(device), labs.to(device)
            _, logits = model(ids, labs, do_contrast=False)
            all_preds.extend(logits.argmax(-1).cpu().tolist())
            all_labs.extend(labs.cpu().tolist())
    return f1_score(all_labs, all_preds, average="macro", zero_division=0)

def train_one(cfg: Cfg, splits):
    from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                               get_linear_schedule_with_warmup)
    from torch.cuda.amp import GradScaler, autocast

    model_path = os.path.join(KAGGLE_MODELS, "unixcoder-base")
    tokenizer  = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    config     = RobertaConfig.from_pretrained(model_path, local_files_only=True)
    encoder    = RobertaModel.from_pretrained(model_path, local_files_only=True)
    pad_id     = tokenizer.pad_token_id

    model = CodeGPTSensorK(encoder, config.hidden_size, cfg.n_cls, pad_id).to(cfg.device)

    tr_ds = CodeDataset(splits["train"], tokenizer, cfg.seq)
    vl_ds = CodeDataset(splits["val"],   tokenizer, cfg.seq)
    ts_ds = CodeDataset(splits["test"],  tokenizer, cfg.seq)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.bs, shuffle=True,  num_workers=2, pin_memory=True)
    vl_dl = DataLoader(vl_ds, batch_size=cfg.bs*2, shuffle=False, num_workers=2)
    ts_dl = DataLoader(ts_ds, batch_size=cfg.bs*2, shuffle=False, num_workers=2)

    total_steps   = cfg.epochs * len(tr_dl)
    warmup_steps  = int(total_steps * cfg.warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_enc, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler()

    best_val, best_test = 0.0, 0.0
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        for ids, labs in tr_dl:
            ids, labs = ids.to(cfg.device), labs.to(cfg.device)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                loss, _ = model(ids, labs, do_contrast=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()

        val_f1 = evaluate(model, vl_dl, cfg.device, pad_id)
        if val_f1 >= best_val:
            best_val = val_f1
            best_test = evaluate(model, ts_dl, cfg.device, pad_id)
        logger.info(f"[ep{ep+1}] val={val_f1:.4f} best_test={best_test:.4f}")

    return {
        "tag":          f"ext_cgpts_unixcoder-base_{cfg.benchmark}_f{cfg.frac}",
        "method":       "CodeGPTSensor-K",
        "upstream":     "TOSEM 2025 (Xu et al.)",
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

# ── Main ──────────────────────────────────────────────────────────────────────
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

    out_path = "results/ext_codegptsensor_results.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved → {out_path}")

    # Summary table
    print(f"\n{'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9}")
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} "
              f"{r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f}")

if __name__ == "__main__":
    main()
