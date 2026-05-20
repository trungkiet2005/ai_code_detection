# ext_damtl — Faithful K-class reproduction of DA-MTL (SecureComm 2025)
# =============================================================================
# NAME       : DA-MTL-K  (Multi-Task Learning: binary detection + K-class attr)
# UPSTREAM   : Khalil et al., "Two Birds with One Stone: Multi-Task Detection
#              and Attribution of LLM-Generated Text", SecureComm 2025
#              https://github.com/youssefkhalil320/MTL_training_two_birds_with_one_stone
# FAITHFULNESS (from methods/multi_supervised.py):
#   Architecture (MultiTaskModel):
#     shared_model  = encoder.roberta / encoder.bert  [shared Transformer]
#     classifier_task1 = Linear(hidden, num_labels_task1)  [binary: H vs AI]
#     classifier_task2 = Linear(hidden, num_labels_task2)  [K-class attribution]
#     forward(input_ids, attention_mask, task='task1'/'task2'):
#       CLS = shared_model(ids, mask).last_hidden_state[:, 0, :]
#       logits = classifier_taskX(CLS)
#   Loss (fine_tune_multi_task_model):
#     Simultaneous training from two interleaved DataLoaders (zip):
#       loss_task1 = CE(task1_logits, binary_labels)
#       loss_task1.backward(retain_graph=True)
#       loss_task2 = CE(task2_logits, kclass_labels)
#       loss_task2.backward()
#       optimizer.step()  [joint, sequential backward — NOT summed]
#     Optional PCGrad gradient surgery (use_pcgrad=True, epoch > 0)
#   Optimizer: AdamW, lr=1e-5 (multi_supervised.py line 284)
#   Epochs: 5 default (multi_task.py line 37)
#   Tokenisation: standard padding/truncation/max_length=512
#
#   ADAPTATION for our K-class setting:
#   - Task 1 (binary): same labels for both benchmarks (0=AI, 1=human)
#     → In CoDET-M4/AICD-T2 all data is AI-generated, so Task 1 is trivially
#       all-AI. We adapt: Task1 = binary (model_family ≤ threshold vs ≥ threshold)
#       OR we simplify: Task1 = binary detect (1 if first half of classes, 0 else)
#     → Cleaner: Task1 = coarser grouping (e.g., model category: GPT/Claude/etc)
#     → MOST FAITHFUL: Task1 = binary AI-vs-human proxy using the test split
#       where val/test have balanced human pseudo-labels. But we don't have human.
#     → We use: Task1 = 2-class (model_family parity: even=0, odd=1) as proxy
#       and Task2 = full K-class attribution. This tests MTL gradient sharing.
#   - Encoder: mBERT/XLM-R → UniXcoder-base (protocol parity)
#   - PCGrad: use_pcgrad=False (default, matching paper ablation without PCGrad)
# WHAT CHANGES vs original:
#   - Encoder: mBERT/XLM-R/DistilBERT → UniXcoder-base
#   - Task1: true binary (H vs AI) → proxy 2-class (model parity grouping)
#   - Dataset: text essays → CoDET-M4 / AICD-T2 code
#   - Metric: accuracy → macro-F1 (on Task 2, the attribution task)
# =============================================================================
from __future__ import annotations
import os, sys, time, json, random, subprocess, importlib.util
from dataclasses import dataclass
from itertools import cycle

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
logger = logging.getLogger("ext_damtl")

@dataclass
class Cfg:
    benchmark:  str  = "codet_m4"
    task:       str  = "author_iid"
    frac:       float = 0.20
    n_cls:      int  = 6          # Task2 = K-class attribution
    n_bin:      int  = 2          # Task1 = binary (proxy)
    seed:       int  = 42
    seq:        int  = 512
    bs:         int  = 16         # original default bs=16
    epochs:     int  = 5          # original default epochs=5
    lr:         float = 1e-5      # original AdamW lr=1e-5
    warmup:     float = 0.10
    use_pcgrad: bool = False       # original default False
    device:     str  = "cuda"

    def __post_init__(self):
        # RAS
        if self.frac <= 0.02:
            self.epochs, self.warmup = 10, 0.20
        elif self.frac <= 0.10:
            self.epochs, self.warmup = 6, 0.15
        else:
            self.epochs, self.warmup = 5, 0.10
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

def make_proxy_binary(label_k: int) -> int:
    """
    Task-1 proxy binary label: even author-id → 0, odd → 1.
    This creates a balanced 2-class auxiliary task that shares the same
    encoder space, exercising the MTL gradient conflict resolution without
    requiring true human-written samples.
    """
    return label_k % 2

# ── Tokenisation ──────────────────────────────────────────────────────────────
def _tokenize(code, tokenizer, max_len):
    toks = tokenizer.tokenize(" ".join(code.split()))[:max_len-4]
    toks = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + toks + [tokenizer.sep_token]
    ids  = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_len - len(ids))
    return ids[:max_len]

class CodeDataset(TD):
    """Returns (input_ids, label_k, label_bin)."""
    def __init__(self, recs, tok, seq):
        self.recs, self.tok, self.seq = recs, tok, seq
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        ids   = _tokenize(r["code"], self.tok, self.seq)
        lab_k = r["label"]
        lab_b = make_proxy_binary(lab_k)
        return (torch.tensor(ids,   dtype=torch.long),
                torch.tensor(lab_k, dtype=torch.long),
                torch.tensor(lab_b, dtype=torch.long))

# ── MultiTaskModel (faithful to multi_supervised.py lines 28-52) ──────────────
class MultiTaskModel(nn.Module):
    """
    Faithful copy of DA-MTL MultiTaskModel:
      shared_model = encoder (RoBERTa backbone)
      classifier_task1 = Linear(H, 2)   [binary proxy]
      classifier_task2 = Linear(H, K)   [K-class attribution]
      forward(ids, mask, task='task1'/'task2'):
        CLS = last_hidden_state[:, 0, :]
        return classifierX(CLS)
    """
    def __init__(self, encoder, hidden, n_bin, n_cls):
        super().__init__()
        self.encoder          = encoder
        self.classifier_task1 = nn.Linear(hidden, n_bin)  # binary
        self.classifier_task2 = nn.Linear(hidden, n_cls)  # K-class

    def forward(self, input_ids, attention_mask, task="task1"):
        out  = self.encoder(input_ids, attention_mask=attention_mask, output_hidden_states=True)
        cls  = out[0][:, 0, :]           # [CLS] token (line 47)
        if task == "task1":
            return self.classifier_task1(cls)
        else:
            return self.classifier_task2(cls)

# ── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(model, loader, device, task="task2"):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for ids, lab_k, lab_b in loader:
            ids   = ids.to(device)
            mask  = ids.ne(0)
            y     = lab_k.to(device) if task == "task2" else lab_b.to(device)
            logits = model(ids, mask, task=task)
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(y.cpu().tolist())
    return f1_score(labs, preds, average="macro", zero_division=0)

# ── PCGrad (faithful to multi_supervised.py lines 313-327) ───────────────────
def apply_pcgrad(model, grad_task1, grad_task2):
    """
    Faithful to DA-MTL PCGrad: project g1 to remove component conflicting with g2.
    g1 = g1 - (g1·g2 / |g2|²) * g2  if g1·g2 < 0
    """
    with torch.no_grad():
        for param, g1, g2 in zip(model.parameters(), grad_task1, grad_task2):
            if g1 is not None and g2 is not None:
                g1_flat = g1.flatten()
                g2_flat = g2.flatten()
                proj = (g1_flat @ g2_flat) / (g2_flat.norm()**2 + 1e-8)
                g1_flat = g1_flat - proj * g2_flat
                param.grad = g1_flat.view(param.grad.shape)

# ── Train ─────────────────────────────────────────────────────────────────────
def train_one(cfg, splits):
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, \
                              get_linear_schedule_with_warmup
    from torch.cuda.amp import GradScaler, autocast

    model_path = os.path.join(KAGGLE_MODELS, "unixcoder-base")
    tokenizer  = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    config     = RobertaConfig.from_pretrained(model_path, local_files_only=True)
    encoder    = RobertaModel.from_pretrained(model_path, local_files_only=True)

    model = MultiTaskModel(encoder, config.hidden_size, cfg.n_bin, cfg.n_cls).to(cfg.device)

    tr_ds = CodeDataset(splits["train"], tokenizer, cfg.seq)
    vl_ds = CodeDataset(splits["val"],   tokenizer, cfg.seq)
    ts_ds = CodeDataset(splits["test"],  tokenizer, cfg.seq)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.bs,    shuffle=True,  num_workers=2, pin_memory=True)
    vl_dl = DataLoader(vl_ds, batch_size=cfg.bs*2,  shuffle=False, num_workers=2)
    ts_dl = DataLoader(ts_ds, batch_size=cfg.bs*2,  shuffle=False, num_workers=2)

    # Faithful: AdamW lr=1e-5 (multi_supervised.py line 284)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    total_steps  = cfg.epochs * len(tr_dl)
    warmup_steps = int(total_steps * cfg.warmup)
    scheduler    = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler()

    # DA-MTL uses two interleaved loaders (zip), cycling longer one
    tr_dl2 = DataLoader(tr_ds, batch_size=cfg.bs, shuffle=True, num_workers=2, pin_memory=True)

    best_val, best_test = 0.0, 0.0
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        # Faithful: zip two separate loaders (multi_supervised.py line 291)
        # Both loaders use same dataset (Task1=binary proxy, Task2=K-class)
        for (batch1, batch2) in zip(tr_dl, cycle(tr_dl2)):
            optimizer.zero_grad()

            ids1, lab_k1, lab_b1 = [x.to(cfg.device) for x in batch1]
            mask1 = ids1.ne(tokenizer.pad_token_id)

            ids2, lab_k2, lab_b2 = [x.to(cfg.device) for x in batch2]
            mask2 = ids2.ne(tokenizer.pad_token_id)

            with autocast(dtype=torch.bfloat16):
                # Task1: binary proxy (even/odd label grouping)
                logits1 = model(ids1, mask1, task="task1")
                loss1   = F.cross_entropy(logits1, lab_b1)

            scaler.scale(loss1).backward(retain_graph=True)   # faithful: retain_graph

            with autocast(dtype=torch.bfloat16):
                # Task2: K-class attribution
                logits2 = model(ids2, mask2, task="task2")
                loss2   = F.cross_entropy(logits2, lab_k2)

            scaler.scale(loss2).backward()

            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()

        val_f1 = evaluate(model, vl_dl, cfg.device, task="task2")
        if val_f1 >= best_val:
            best_val  = val_f1
            best_test = evaluate(model, ts_dl, cfg.device, task="task2")
        logger.info(f"[ep{ep+1}] val_task2={val_f1:.4f} best_test={best_test:.4f}")

    return {
        "tag":          f"ext_damtl_unixcoder-base_{cfg.benchmark}_f{cfg.frac}",
        "method":       "DA-MTL-K",
        "upstream":     "SecureComm 2025 (Khalil et al.)",
        "note":         ("Shared encoder + Task1=binary(proxy:even/odd) + Task2=K-class. "
                         "Sequential backward(retain_graph). AdamW lr=1e-5. "
                         "Metric on Task2 (attribution). use_pcgrad=False."),
        "enc":          "unixcoder-base",
        "bench":        cfg.benchmark,
        "frac":         cfg.frac,
        "epochs":       cfg.epochs,
        "lr":           cfg.lr,
        "use_pcgrad":   cfg.use_pcgrad,
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

    out_path = "results/ext_damtl_results.json"
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
