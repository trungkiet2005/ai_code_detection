"""
exp16_ce_baseline.py — Cross-Entropy baseline with standardized few-shot protocol

Self-contained. Runs: 2 encoders × 2 benchmarks = 4 experiments.

Key Protocol Change (v16):
  - FIXED_TOTAL_TRAIN = 72: All benchmarks get same 72 total training samples
  - This ensures fair comparison across different dataset sizes
  - CoDET-M4: 72 = 12 samples × 6 classes
  - AICD-T2: 72 = 6 samples × 12 classes

Config:
  - Encoders: ModernBERT-base, unixcoder-base
  - Benchmarks: codet_m4 (6 classes), aicd_t2 (12 classes)
  - Total training: 72 samples (fixed)
  - Batch: 256, seq=512

Usage:
  python exp16_ce_baseline.py
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

from torch.cuda.amp import autocast, GradScaler

def _autocast_ctx(dev):
    return autocast(enabled=(dev.type == "cuda"))

    except TypeError:
        return _ac(enabled=enabled)

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp13")

PAPER_BASELINE = 0.6633

@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    enc: str = "ModernBERT-base"
    frac: float = 0.05
    n_cls: int = 6
    seed: int = 42
    bs: int = 64
    seq: int = 512
    epochs: int = 3
    lr_enc: float = 2e-5
    lr_head: float = 1e-4
    wd: float = 0.01
    warmup: float = 0.1
    device: str = "cuda"
    # Few-shot: use n_shots_per_cls > 0 to override frac
    # Set n_shots_per_cls = 12 for K-shot learning (12 samples per class)
    # All benchmarks will have same total samples: n_shots_per_cls * n_cls
    n_shots_per_cls: int = 0  # 0 = use frac; >0 = use fixed K shots per class

    # Fixed total training budget (Option 2)
    # All benchmarks use same total samples: FIXED_TOTAL_TRAIN_SAMPLES
    FIXED_TOTAL_TRAIN: int = 72  # 0 = disable; >0 = use fixed total samples

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.n_cls = 6 if self.task == "author" else 2
        elif self.benchmark == "aicd_t2":
            self.n_cls = 12; self.task = "t2"
        elif self.benchmark in ("droid_t3",):
            self.n_cls = 3; self.task = "t3"
        elif self.benchmark == "droid_t4":
            self.n_cls = 4; self.task = "t4"

def _hw(cfg: Cfg) -> Cfg:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40: cfg.bs, cfg.seq = 256, 512
        elif mem >= 10: cfg.bs, cfg.seq = 128, 384
        else: cfg.bs, cfg.seq = 64, 256
    return cfg

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _is_human(t):
    return str(t or "").strip().lower() in {"human","human_written","human-generated"}

def _vocab(train):
    names = {str(r.get("model","") or "").strip() for r in train
             if not _is_human(r.get("target","")) and r.get("model","")}
    return {n:i+1 for i,n in enumerate(sorted(names))}

def _conv_codet(split, task, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code","code"):
            v = r.get(f,"")
            if isinstance(v,str) and v.strip(): code = v; break
        if task == "binary": label = 0 if _is_human(r.get("target","")) else 1
        else:
            if _is_human(r.get("target","")): label = 0
            else: label = vocab.get(str(r.get("model","") or "").strip(), -1)
        return {"code":code,"label":label,"lang":str(r.get("language","")).strip().lower()}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"]>=0 and len(x["code"].strip())>0)

def _conv_droid(split, task):
    lm = {"HUMAN_GENERATED":0,"HUMAN":0,"MACHINE_GENERATED":1,"AI_GENERATED":1,
          "MACHINE_REFINED":2,"REFINED":2,"ADVERSARIAL":3,"ADVERSARIALLY_HUMANISED":3}
    def row(r):
        code = str(r.get("code","")).strip()
        raw = r.get("label",-1)
        label = lm.get(str(raw).strip().upper(), int(raw) if isinstance(raw,int) else -1)
        if task == "t3": label = 1 if label == 3 else label
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
        s = ds.train_test_split(test_size=0.1, seed=42)
        s2 = s["train"].train_test_split(test_size=1/9, seed=42)
        return s2["train"], s2["test"], s["test"]
    return tr, vl, ts

def _load_droid():
    """Load DroidCollection from Kaggle local path.
    
    Kaggle path structure:
      /kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/data/
        ├── train-00001-of-00004.parquet ... train-00004-of-00004.parquet
        └── test-00001-of-00002.parquet ... test-00002-of-00002.parquet
    
    HuggingFace fallback: project-droid/DroidCollection (train/dev/test splits).
    """
    train_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "test-*.parquet")))
    dev_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "dev-*.parquet")))
    
    if train_files and test_files:
        logger.info(f"[droid] Loading from local: {len(train_files)} train shards, {len(test_files)} test shards, {len(dev_files)} dev shards")
        ds_train = load_dataset("parquet", data_files=train_files, split="train")
        ds_test = load_dataset("parquet", data_files=test_files, split="test")
        
        if dev_files:
            ds_dev = load_dataset("parquet", data_files=dev_files, split="train")
            return ds_train, ds_dev, ds_test
        else:
            s = ds_train.train_test_split(test_size=0.1, seed=42)
            return s["train"], s["test"], ds_test
    else:
        logger.warning("[droid] Kaggle path not found, falling back to HuggingFace...")
        tr = load_dataset("project-droid/DroidCollection", split="train")
        vl = load_dataset("project-droid/DroidCollection", split="dev")
        ts = load_dataset("project-droid/DroidCollection", split="test")
        return tr, vl, ts

def _load_aicd(task):
    """Load AICD-Bench -- STRICT: only loads the requested task dir, NO fallback.

    Raises FileNotFoundError immediately if target task dir is missing.
    Prevents silent data bugs (e.g. loading T1 instead of T2).
    """
    task_map = {"t1": "T1", "t2": "T2", "t3": "T3"}
    task_name = task_map.get(task.lower(), None)
    if task_name is None:
        raise ValueError(f"[aicd] Unknown task '{task}'. Must be one of: t1, t2, t3.")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(
            f"[aicd] STRICT: {task_name} dir not found at {task_path}. "
            f"NO fallback to other tasks or HuggingFace. Check KAGGLE_AICD path."
        )
    parquet_files = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(
            f"[aicd] STRICT: No parquet files in {task_path}. NO fallback."
        )
    logger.info(f"[aicd] Loading {task_name} from {task_path} ({len(parquet_files)} files)")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    if "split" in ds.column_names:
        try:
            tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
            vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
            ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
            if len(tr) > 0 and len(vl) > 0 and len(ts) > 0:
                return tr, vl, ts
        except Exception:
            pass
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


# =============================================================================
# PREFLIGHT: Validate all datasets BEFORE training runs
# Purpose: Fail fast on missing/corrupt data, report sizes, abort if empty
# =============================================================================
def _preflight_check():
    """Load all benchmarks and report sizes. Abort if any dataset is empty."""
    logger.info("=" * 60)
    logger.info("[PREFLIGHT] Starting data validation...")
    logger.info("=" * 60)

    all_ok = True
    bench_configs = [
        ("codet_m4", _load_codet, None, "author"),
        ("aicd_t2", None, "t2", None),
    ]

    for bench_name, load_fn, task_arg, conv_task in bench_configs:
        try:
            if load_fn is not None:
                tr, vl, ts = load_fn()
            elif task_arg is not None:
                tr, vl, ts = _load_aicd(task_arg)
            else:
                tr, vl, ts = _load_droid()

            if bench_name.startswith("codet_m4"):
                vocab = _vocab(tr)
                tr_d = _conv_codet(tr, conv_task, vocab)
                vl_d = _conv_codet(vl, conv_task, vocab)
                ts_d = _conv_codet(ts, conv_task, vocab)
            elif bench_name.startswith("aicd"):
                tr_d = _conv_aicd(tr)
                vl_d = _conv_aicd(vl)
                ts_d = _conv_aicd(ts)
            elif bench_name.startswith("droid"):
                tr_d = _conv_droid(tr, conv_task)
                vl_d = _conv_droid(vl, conv_task)
                ts_d = _conv_droid(ts, conv_task)

            n_tr, n_vl, n_ts = len(tr_d), len(vl_d), len(ts_d)
            from collections import Counter
            tr_labels = Counter(tr_d["label"])

            logger.info(f"[PREFLIGHT] {bench_name}: Train={n_tr} | Val={n_vl} | Test={n_ts} | Classes={len(tr_labels)}")
            logger.info(f"[PREFLIGHT] Train dist: {dict(sorted(tr_labels.items()))}")

            if n_tr == 0 or n_vl == 0 or n_ts == 0:
                logger.error(f"[PREFLIGHT] EMPTY! {bench_name}: Train={n_tr}, Val={n_vl}, Test={n_ts}")
                all_ok = False
        except FileNotFoundError as e:
            logger.error(f"[PREFLIGHT] File not found: {bench_name}: {e}")
            all_ok = False
        except Exception as e:
            logger.error(f"[PREFLIGHT] Load error: {bench_name}: {e}")
            all_ok = False

    logger.info("=" * 60)
    if all_ok:
        logger.info("[PREFLIGHT] All datasets loaded successfully!")
    else:
        logger.error("[PREFLIGHT] Dataset validation FAILED!")
        raise RuntimeError("Dataset validation failed")
    logger.info("=" * 60)

class FSDS(TD):
    def __init__(self, hf, tok, max_len):
        self.hf = hf; self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.hf)
    def __getitem__(self, i):
        r = self.hf[i]
        enc = self.tok(r["code"], max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        return {"ids":enc["input_ids"].squeeze(0),"mask":enc["attention_mask"].squeeze(0),"label":int(r["label"])}

def collate(b):
    return {"ids":torch.stack([x["ids"] for x in b]),"mask":torch.stack([x["mask"] for x in b]),
            "labels":torch.tensor([x["label"] for x in b], dtype=torch.long)}

def build_dls(cfg: Cfg):
    set_seed(cfg.seed)
    enc_path = os.path.join(KAGGLE_MODELS, cfg.enc)
    tok = AutoTokenizer.from_pretrained(enc_path, local_files_only=True)

    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw) if cfg.task == "author" else {}
        tr_d = _conv_codet(tr_raw, cfg.task, vocab)
        vl_d = _conv_codet(vl_raw, cfg.task, vocab)
        ts_d = _conv_codet(ts_raw, cfg.task, vocab)
    elif cfg.benchmark.startswith("droid"):
        tr_raw, vl_raw, ts_raw = _load_droid()
        tr_d = _conv_droid(tr_raw, cfg.task)
        vl_d = _conv_droid(vl_raw, cfg.task)
        ts_d = _conv_droid(ts_raw, cfg.task)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd(cfg.task)
        tr_d = _conv_aicd(tr_raw); vl_d = _conv_aicd(vl_raw); ts_d = _conv_aicd(ts_raw)

    by_cls = defaultdict(list)
    for i, lab in enumerate(tr_d["label"]): by_cls[int(lab)].append(i)
    # Get actual classes from data first
    actual_classes = sorted(set(tr_d["label"]))
    n_actual_cls = len(actual_classes)
    rng = random.Random(cfg.seed)
    chosen = []

    if cfg.FIXED_TOTAL_TRAIN > 0:
        # Option 2: Equal total training budget across all benchmarks
        # Distribute FIXED_TOTAL_TRAIN samples equally per class
        total = cfg.FIXED_TOTAL_TRAIN
        n_per_cls = max(1, total // n_actual_cls)  # at least 1 per class
        remaining = total - (n_per_cls * n_actual_cls)  # leftover

        for cls in actual_classes:
            pool = by_cls.get(cls, [])
            # First 'remaining' classes get n_per_cls + 1
            n = n_per_cls + (1 if cls < remaining else 0)
            n = min(n, len(pool))  # don't Sample more than available
            if pool:
                chosen.extend(rng.sample(pool, n))
        logger.info(f"[data] {cfg.enc} | {cfg.benchmark} | FIXED_TOTAL={cfg.FIXED_TOTAL_TRAIN} | n_train={len(chosen)} (={n_per_cls}/cls)")

    elif cfg.n_shots_per_cls > 0:
        # Option 1: K-shot learning (fixed K samples per class)
        for cls in actual_classes:
            pool = by_cls.get(cls, [])
            n = min(cfg.n_shots_per_cls, len(pool))
            if pool:
                chosen.extend(rng.sample(pool, n))
        logger.info(f"[data] {cfg.enc} | {cfg.benchmark} | K_SHOT={cfg.n_shots_per_cls} | n_train={len(chosen)}")

    else:
        # Original: fraction-based sampling
        for cls in actual_classes:
            pool = by_cls.get(cls, [])
            n = max(1, int(round(len(pool) * cfg.frac))) if pool else 0
            chosen.extend(rng.sample(pool, min(n, len(pool))) if pool else [])
        logger.info(f"[data] {cfg.enc} | {cfg.benchmark} | frac={cfg.frac} | n_train={len(chosen)}")

    rng.shuffle(chosen)
    tr_d = tr_d.select(chosen)

    def ld(ds, shuf):
        return DataLoader(FSDS(ds, tok, cfg.seq), batch_size=cfg.bs, shuffle=shuf, num_workers=4, collate_fn=collate, pin_memory=True)
    return ld(tr_d, True), ld(vl_d, False), ld(ts_d, False)

class CENet(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        enc_path = os.path.join(KAGGLE_MODELS, cfg.enc)
        self.enc = AutoModel.from_pretrained(enc_path, local_files_only=True)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        self.clf = nn.Linear(h, cfg.n_cls)

    def forward(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask)
        emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return {"logits": self.clf(self.drop(emb))}

    def groups(self):
        return [{"params": self.enc.parameters(), "lr": self.cfg.lr_enc, "weight_decay": self.cfg.wd},
                {"params": self.clf.parameters(), "lr": self.cfg.lr_head, "weight_decay": self.cfg.wd}]

def class_w(loader, n):
    c = np.zeros(n)
    for b in loader:
        for l in b["labels"].tolist(): c[l] += 1
    c = np.maximum(c, 1)
    w = 1.0 / c
    return torch.tensor(w/w.sum()*n, dtype=torch.float32)

@torch.no_grad()
def eval_m(model, loader, dev):
    model.eval()
    ps, ls = [], []
    for b in loader:
        with _autocast_ctx(dev): logits = model(b["ids"].to(dev), b["mask"].to(dev))["logits"]
        ps.extend(logits.argmax(1).cpu().tolist())
        ls.extend(b["labels"].tolist())
    from sklearn.metrics import f1_score as f1s, precision_score, recall_score, confusion_matrix
    macro = f1s(ls, ps, average="macro", zero_division=0)
    weighted = f1s(ls, ps, average="weighted", zero_division=0)
    acc = accuracy_score(ls, ps)
    # Per-class F1
    per_cls_f1 = f1s(ls, ps, average=None, zero_division=0).tolist()
    per_cls_precision = precision_score(ls, ps, average=None, zero_division=0).tolist()
    per_cls_recall = recall_score(ls, ps, average=None, zero_division=0).tolist()
    # Confusion matrix
    conf_matrix = confusion_matrix(ls, ps).tolist()
    # Prediction distribution
    from collections import Counter
    pred_dist = dict(Counter(ps))
    label_dist = dict(Counter(ls))
    return {
        "macro": macro, "weighted": weighted, "acc": acc,
        "per_class_f1": per_cls_f1, "per_class_precision": per_cls_precision,
        "per_class_recall": per_cls_recall, "confusion_matrix": conf_matrix,
        "pred_distribution": pred_dist, "label_distribution": label_dist,
    }

def train(cfg: Cfg, tr_dl, vl_dl, ts_dl):
    dev = torch.device(cfg.device)
    model = CENet(cfg).to(dev)
    w = class_w(tr_dl, cfg.n_cls).to(dev)
    opt = torch.optim.AdamW(model.groups())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr_enc, cfg.lr_head],
        steps_per_epoch=len(tr_dl), epochs=cfg.epochs, pct_start=cfg.warmup)
    scaler = GradScaler(enabled=(dev.type == "cuda"))
    best_val, best_state = 0, None

    # Track per-epoch metrics for paper
    train_history = []
    val_history = []

    for ep in range(cfg.epochs):
        model.train()
        pbar = tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.epochs}", leave=False)
        ep_loss = []
        for b in pbar:
            ids, mask, labs = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)
            with _autocast_ctx(dev):
                logits = model(ids, mask)["logits"]
                loss = F.cross_entropy(logits, labs, weight=w)
            ep_loss.append(loss.item())
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            sched.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        # Per-epoch metrics
        tr_met = eval_m(model, tr_dl, dev)
        vr = eval_m(model, vl_dl, dev)
        train_history.append({
            "epoch": ep + 1,
            "loss": round(np.mean(ep_loss), 6),
            "macro_f1": round(tr_met["macro"], 6),
            "weighted_f1": round(tr_met["weighted"], 6),
        })
        val_history.append({
            "epoch": ep + 1,
            "macro_f1": round(vr["macro"], 6),
            "weighted_f1": round(vr["weighted"], 6),
            "accuracy": round(vr["acc"], 6),
        })
        if vr["macro"] > best_val:
            best_val = vr["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    final_test = eval_m(model, ts_dl, dev)
    final_test["train_history"] = train_history
    final_test["val_history"] = val_history
    return final_test

def main():
    # Run preflight check for all benchmarks FIRST
    logger.info("[PREFLIGHT] Running dataset validation before experiments...")
    _preflight_check()

    encoders = ["unixcoder-base"]
    benchmarks = [("codet_m4","author"), ("aicd_t2","t2")]

    # Option 1: Standard few-shot with FIXED_TOTAL_TRAIN
    # All benchmarks get same total training samples (72 = 12 samples/class for 6 classes)
    # This ensures fair comparison across different dataset sizes
    FIXED_TOTAL = 72  # 0 = disable (use fracs below); >0 = use fixed total

    if FIXED_TOTAL > 0:
        # Run with fixed total training budget
        results = []
        for enc in encoders:
            for bench, task in benchmarks:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, FIXED_TOTAL_TRAIN=FIXED_TOTAL)
                cfg = _hw(cfg)
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                tag = f"exp16_ce_{enc}_{bench}_fixed{FIXED_TOTAL}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                tr_dl, vl_dl, ts_dl = build_dls(cfg)
                res = train(cfg, tr_dl, vl_dl, ts_dl)
                elapsed = time.time() - t0
                # Build comprehensive result row for paper
                row = {
                    # Experiment metadata
                    "tag": tag,
                    "encoder": enc,
                    "benchmark": bench,
                    "task": task,
                    "n_classes": cfg.n_cls,
                    "total_train_samples": FIXED_TOTAL,
                    "train_samples_per_class": FIXED_TOTAL // cfg.n_cls,
                    # Hyperparameters
                    "batch_size": cfg.bs,
                    "seq_length": cfg.seq,
                    "epochs": cfg.epochs,
                    "lr_encoder": cfg.lr_enc,
                    "lr_head": cfg.lr_head,
                    "weight_decay": cfg.wd,
                    "warmup_ratio": cfg.warmup,
                    # Main metrics
                    "macro_f1": round(res["macro"], 6),
                    "weighted_f1": round(res["weighted"], 6),
                    "accuracy": round(res["acc"], 6),
                    # Comparison to paper baseline
                    "delta_vs_paper": round(res["macro"] - PAPER_BASELINE, 6),
                    "paper_baseline": PAPER_BASELINE,
                    # Per-class metrics
                    "per_class_f1": [round(x, 6) for x in res["per_class_f1"]],
                    "per_class_precision": [round(x, 6) for x in res["per_class_precision"]],
                    "per_class_recall": [round(x, 6) for x in res["per_class_recall"]],
                    # Confusion matrix
                    "confusion_matrix": res["confusion_matrix"],
                    # Prediction analysis
                    "pred_distribution": res["pred_distribution"],
                    "label_distribution": res["label_distribution"],
                    # Training history
                    "train_history": res["train_history"],
                    "val_history": res["val_history"],
                    # Training info
                    "wall_time_seconds": round(elapsed, 1),
                    "gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                results.append(row)
                logger.info(f"[{tag}] MacroF1={res['macro']:.4f} ({res['macro']-PAPER_BASELINE:+.4f} vs paper) time={elapsed:.0f}s")
                del tr_dl, vl_dl, ts_dl
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

        os.makedirs("results", exist_ok=True)
        with open("results/exp16_fixed72_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*100)
        print(f"{'Encoder':<22} {'Benchmark':<12} {'Macro-F1':>10} {'dPaper':>10} {'Weighted':>10} {'Wall':>8}")
        print("-"*100)
        for r in results:
            print(f"{r['encoder']:<22} {r['benchmark']:<12} {r['macro_f1']:>10.4f} {r['delta_vs_paper']:>+10.4f} {r['weighted_f1']:>10.4f} {r['wall_time_seconds']:>8.0f}s")
        print("="*100)
        print("\nPer-class F1 breakdown:")
        for r in results:
            print(f"\n{r['encoder']} @ {r['benchmark']}:")
            for i, f1 in enumerate(r['per_class_f1']):
                print(f"  Class {i:2d}: F1={f1:.4f} P={r['per_class_precision'][i]:.4f} R={r['per_class_recall'][i]:.4f}")

    else:
        # Option 2: Original frac-based (keep for comparison)
        fracs = [0.01, 0.05, 0.20]
        results = []
        for enc in encoders:
            for bench, task in benchmarks:
                for frac in fracs:
                    cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac)
                    cfg = _hw(cfg)
                    tag = f"exp13_ce_{enc}_{bench}_f{frac}"
                    logger.info(f"=== {tag} ===")
                    t0 = time.time()
                    tr_dl, vl_dl, ts_dl = build_dls(cfg)
                    res = train(cfg, tr_dl, vl_dl, ts_dl)
                    elapsed = time.time() - t0
                    row = {"tag": tag, "enc": enc, "bench": bench, "frac": frac,
                           "macro": res["macro"], "weighted": res["weighted"], "acc": res["acc"],
                           "dpaper": res["macro"] - PAPER_BASELINE, "wall": round(elapsed,1)}
                    results.append(row)
                    logger.info(f"[{tag}] MacroF1={res['macro']:.4f} ({res['macro']-PAPER_BASELINE:+.4f} vs paper) time={elapsed:.0f}s")
                    del tr_dl, vl_dl, ts_dl
                    import gc; gc.collect()
                    if torch.cuda.is_available(): torch.cuda.empty_cache()

        os.makedirs("results", exist_ok=True)
        with open("results/exp13_results.json", "w") as f:
            json.dump(results, f, indent=2)

        print("\n" + "="*100)
        print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Macro-F1':>10} {'dPaper':>10} {'Weighted':>10} {'Wall':>8}")
        print("-"*100)
        for r in results:
            print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['macro']:>10.4f} {r['dpaper']:>+10.4f} {r['weighted']:>10.4f} {r['wall']:>8.0f}s")
        print("="*100)
        print(f"\nBest Macro-F1: {max(r['macro'] for r in results):.4f} @ {max(results, key=lambda x: x['macro'])['tag']}")

    print("\n[OK] Experiments complete!")

if __name__ == "__main__":
    main()
