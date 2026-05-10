"""
exp_n10_vib.py — Variational Information Bottleneck for Data Efficiency

NAME : Variational Information Bottleneck (VIB)
ONE-LINE CLAIM : VIB adds a KL penalty that forces the representation
to forget source-specific noise while preserving generator signal,
achieving better generalization in few-shot regimes.
EQUATION : L_vib = CE(q(y|z), y) + β · KL(q(z|x) || p(z))
THEORY HOOK : Alemi et al. 2017 VIB; information-theoretic regularizer
that removes nuisance variables while keeping task-relevant signal.
WHY NOT BEFORE : Standard fine-tuning keeps all features, including
source-specific noise. VIB explicitly forgets this, critical for OOD.
FALSIFIER : If VIB does not improve generalization at frac=0.05,
then source noise is not the bottleneck.

Target: EMNLP Oral — Theory contribution (novel information-theoretic object).

Self-contained. Runs: 2 encoders × 4 benchmarks × 3 fractions = 24 experiments.

Config:
  - Encoders: ModernBERT-base, unixcoder-base
  - Benchmarks: codet_m4 (headline), aicd_t2 (stress), droid_t3, droid_t4
  - Fractions: 0.01, 0.05, 0.20
  - Batch: 256, seq=512

Usage:
  python exp_n10_vib.py
"""

# =============================================================================
# Theory-Track exp — Variational Information Bottleneck (VIB):
# information-theoretic regularization for source-invariant representations.
#
# NAME : Variational Information Bottleneck (VIB).
# ONE-LINE CLAIM : VIB adds a KL penalty forcing representations to forget
# source-specific noise while preserving generator signal.
# EQUATION : L_vib = CE(q(y|z), y) + β · KL(q(z|x) || p(z))
# where q(z|x) = N(μ(x), σ(x)) and p(z) = N(0, I).
# PROPERTY : The KL term regularizes representations to be compact and
# source-invariant. This is an information-theoretic implementation of
# the do(S)-invariance hypothesis from our causal model.
# WHY NOT BEFORE : Standard fine-tuning keeps all encoder features,
# including source-specific noise. VIB explicitly removes this.
# FALSIFIER : If VIB does not improve OOD generalization (GH split),
# then source noise is not the bottleneck.
# =============================================================================

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

def _autocast_ctx(dev: torch.device):
    enabled = (dev.type == "cuda")
    try:
        return _ac(device_type=dev.type, enabled=enabled)
    except TypeError:
        return _ac(enabled=enabled)

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp_n10")

# === CONSTANTS ===
HIER_FAM = {0:0,1:1,2:2,3:3,4:1,5:4}
PAPER_BASELINE = 0.6633

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

            # Convert to filtered data
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

            n_tr = len(tr_d)
            n_vl = len(vl_d)
            n_ts = len(ts_d)

            from collections import Counter
            tr_labels = Counter(tr_d["label"])
            vl_labels = Counter(vl_d["label"])
            ts_labels = Counter(ts_d["label"])

            logger.info(f"[PREFLIGHT] {bench_name}:")
            logger.info(f"  Train: {n_tr:,} | Val: {n_vl:,} | Test: {n_ts:,}")
            logger.info(f"  Train classes: {len(tr_labels)} | Val classes: {len(vl_labels)} | Test classes: {len(ts_labels)}")
            logger.info(f"  Train dist: {dict(sorted(tr_labels.items()))}")

            if n_tr == 0 or n_vl == 0 or n_ts == 0:
                logger.error(f"[PREFLIGHT] ❌ {bench_name}: EMPTY! Train={n_tr}, Val={n_vl}, Test={n_ts}")
                all_ok = False
            elif n_tr < 100:
                logger.warning(f"[PREFLIGHT] ⚠️ {bench_name}: Train={n_tr} is very small!")
        except FileNotFoundError as e:
            logger.error(f"[PREFLIGHT] ❌ {bench_name}: File not found: {e}")
            all_ok = False
        except Exception as e:
            logger.error(f"[PREFLIGHT] ❌ {bench_name}: Load error: {e}")
            all_ok = False

    logger.info("=" * 60)
    if all_ok:
        logger.info("[PREFLIGHT] ✅ All datasets loaded successfully!")
    else:
        logger.error("[PREFLIGHT] ❌ Dataset validation FAILED. Aborting.")
        raise RuntimeError("[PREFLIGHT] Dataset validation failed. Check logs above.")
    logger.info("=" * 60)

@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    enc: str = "ModernBERT-base"
    frac: float = 0.05
    n_cls: int = 6
    seed: int = 42
    bs: int = 256
    seq: int = 512
    epochs: int = 3
    lr_enc: float = 2e-5
    lr_head: float = 1e-4
    wd: float = 0.01
    vib_beta: float = 1e-3
    z_dim: int = 128
    warmup: float = 0.1
    device: str = "cuda"

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
    """Load AICD-Bench from Kaggle local path or HuggingFace.
    
    Kaggle path structure:
      /kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench/
        ├── T1/
        │   └── *.parquet
        ├── T2/
        │   └── *.parquet
        └── T3/
            └── *.parquet
    
    HuggingFace fallback: AICD-bench/AICD-Bench (requires internet).
    """
    cfg_map = {"t2":"T2","t3":"T3","t1":"T1"}
    task_name = cfg_map.get(task.upper().replace("T2","t2").replace("T3","t3").replace("T1","t1"), "T2")
    
    # Try local Kaggle path first
    local_base = KAGGLE_AICD
    task_dirs = ["T1", "T2", "T3"]
    
    for t in task_dirs:
        task_path = os.path.join(local_base, t)
        if os.path.isdir(task_path):
            # Find parquet files
            parquet_files = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
            if parquet_files:
                logger.info(f"[aicd] Loading from local: {task_path}")
                ds = load_dataset("parquet", data_files=parquet_files, split="train")
                # AICD has train/val/test splits
                if "validation" in ds.column_names or "val" in ds.column_names:
                    val_key = "validation" if "validation" in ds.column_names else "val"
                    return ds["train"], ds[val_key], ds["test"]
                else:
                    s = ds.train_test_split(test_size=0.1, seed=42)
                    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
                    return s2["train"], s2["test"], s["test"]
    
    # Try loading as single parquet
    parquet_files = sorted(glob.glob(os.path.join(local_base, "**", "*.parquet"), recursive=True))
    if parquet_files:
        logger.info(f"[aicd] Loading parquet files from: {local_base}")
        ds = load_dataset("parquet", data_files=parquet_files, split="train")
        if "validation" in ds.column_names or "val" in ds.column_names:
            val_key = "validation" if "validation" in ds.column_names else "val"
            return ds["train"], ds[val_key], ds["test"]
        else:
            s = ds.train_test_split(test_size=0.1, seed=42)
            s2 = s["train"].train_test_split(test_size=1/9, seed=42)
            return s2["train"], s2["test"], s["test"]
    
    # Fallback to HuggingFace (requires internet)
    logger.warning("[aicd] Local path not found, trying HuggingFace (requires internet)")
    return (load_dataset("AICD-bench/AICD-Bench", name=task_name, split=s) for s in ["train","validation","test"])

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
    rng = random.Random(cfg.seed)
    chosen = []
    for cls in range(cfg.n_cls):
        pool = by_cls.get(cls, [])
        n = max(1, int(round(len(pool) * cfg.frac))) if pool else 0
        chosen.extend(rng.sample(pool, min(n, len(pool))) if pool else [])
    rng.shuffle(chosen)
    tr_d = tr_d.select(chosen)
    logger.info(f"[data] {cfg.enc} | {cfg.benchmark} | frac={cfg.frac} | n_train={len(tr_d)}")

    def ld(ds, shuf):
        return DataLoader(FSDS(ds, tok, cfg.seq), batch_size=cfg.bs, shuffle=shuf, num_workers=4, collate_fn=collate, pin_memory=True)
    return ld(tr_d, True), ld(vl_d, False), ld(ts_d, False)

class VIBNet(nn.Module):
    """Variational Information Bottleneck network.
    
    Encodes inputs to stochastic latent variables with KL regularization,
    forcing representations to be compact and source-invariant.
    """
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        enc_path = os.path.join(KAGGLE_MODELS, cfg.enc)
        self.enc = AutoModel.from_pretrained(enc_path, local_files_only=True)
        h = self.enc.config.hidden_size
        self.drop = nn.Dropout(0.1)
        
        # VIB: stochastic encoder
        self.z_mean = nn.Linear(h, cfg.z_dim)
        self.z_logvar = nn.Linear(h, cfg.z_dim)
        
        # Classifier
        self.clf = nn.Linear(cfg.z_dim, cfg.n_cls)

    def forward(self, ids, mask, training=True):
        out = self.enc(input_ids=ids, attention_mask=mask)
        emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        
        # VIB: compute latent distribution
        mu = self.z_mean(emb)
        logvar = self.z_logvar(emb)
        std = torch.exp(0.5 * logvar)
        
        if training:
            # Reparameterization trick
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu  # Use mean at test time
        
        logits = self.clf(self.drop(z))
        return {"logits": logits, "z": z, "mu": mu, "logvar": logvar}

    def groups(self):
        return [{"params": self.enc.parameters(), "lr": self.cfg.lr_enc, "weight_decay": self.cfg.wd},
                {"params": list(self.z_mean.parameters()) + list(self.z_logvar.parameters()) + list(self.clf.parameters()),
                 "lr": self.cfg.lr_head, "weight_decay": self.cfg.wd}]

def vib_loss(logits, labels, mu, logvar, w, beta=1e-3):
    """VIB loss: cross-entropy + KL divergence.
    
    L_vib = CE(q(y|z), y) + β · KL(q(z|x) || p(z))
    
    The KL term encourages z to be close to the prior N(0, I),
    which regularizes the representation to forget source-specific info.
    """
    # Cross-entropy
    ce = F.cross_entropy(logits, labels, weight=w)
    
    # KL divergence: KL(N(μ,σ) || N(0,I))
    # = -0.5 * Σ (1 + log(σ²) - μ² - σ²)
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=-1).mean()
    
    return ce + beta * kl

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
        with _autocast_ctx(dev): logits = model(b["ids"].to(dev), b["mask"].to(dev), training=False)["logits"]
        ps.extend(logits.argmax(1).cpu().tolist())
        ls.extend(b["labels"].tolist())
    return {"macro": f1_score(ls, ps, average="macro", zero_division=0),
            "weighted": f1_score(ls, ps, average="weighted", zero_division=0),
            "acc": accuracy_score(ls, ps)}

def train(cfg: Cfg, tr_dl, vl_dl, ts_dl):
    dev = torch.device(cfg.device)
    model = VIBNet(cfg).to(dev)
    w = class_w(tr_dl, cfg.n_cls).to(dev)
    opt = torch.optim.AdamW(model.groups())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr_enc, cfg.lr_head],
        steps_per_epoch=len(tr_dl), epochs=cfg.epochs, pct_start=cfg.warmup)
    scaler = GradScaler(enabled=(dev.type == "cuda"))
    best_val, best_state = 0, None

    for ep in range(cfg.epochs):
        model.train()
        pbar = tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.epochs}", leave=False)
        for b in pbar:
            ids, mask, labs = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)
            with _autocast_ctx(dev):
                out = model(ids, mask, training=True)
                loss = vib_loss(out["logits"], labs, out["mu"], out["logvar"], w, cfg.vib_beta)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            sched.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        vr = eval_m(model, vl_dl, dev)
        if vr["macro"] > best_val:
            best_val = vr["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return eval_m(model, ts_dl, dev)

def main():
    # Run preflight check for all benchmarks FIRST
    logger.info("[PREFLIGHT] Running dataset validation before experiments...")
    _preflight_check()

    encoders = ["ModernBERT-base", "unixcoder-base"]
    benchmarks = [("codet_m4","author"), ("aicd_t2","t2")]
    fracs = [0.01, 0.05, 0.20]

    results = []
    for enc in encoders:
        for bench, task in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac)
                cfg = _hw(cfg)
                tag = f"exp_n10_vib_{enc}_{bench}_f{frac}"
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
    with open("results/exp_n10_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*100)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Macro-F1':>10} {'dPaper':>10} {'Weighted':>10} {'Wall':>8}")
    print("-"*100)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['macro']:>10.4f} {r['dpaper']:>+10.4f} {r['weighted']:>10.4f} {r['wall']:>8.0f}s")
    print("="*100)
    print(f"\nBest Macro-F1: {max(r['macro'] for r in results):.4f} @ {max(results, key=lambda x: x['macro'])['tag']}")

if __name__ == "__main__":
    main()
