"""
exp_n18_hier_supcon.py — HierTree + Supervised Contrastive (Exp18 migration)

Migrated from legacy/Exp_CodeDet/run_codet_m4_exp18_hiertree.py (70.55 F1)

Changes from legacy:
- unixcoder-base only (not ModernBERT)
- frac-based sampling (1%/5%/20%) instead of full data
- Simplified architecture (no AST/spectral features)
- codet_m4 + aicd_t2 only (not full benchmark suite)

Loss: L_ce + lambda_hier * L_hier + lambda_supcon * L_supcon

Config:
  - Encoder: unixcoder-base only
  - Benchmarks: codet_m4 (headline), aicd_t2 (stress)
  - Fractions: 0.01, 0.05, 0.20
  - Batch: 256, seq=512

Usage:
  python exp_n18_hier_supcon.py
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
from typing import Dict, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Bootstrap deps
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

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp_n18")

# === CONSTANTS ===
# Tree structure: class index → family index
# CoDET-M4 author classes: 0=human, 1=codellama, 2=gpt, 3=llama3.1, 4=nxcode, 5=qwen1.5
# Family grouping: human(0), llama-family(1,3), gpt(2), qwen-family(4,5)
HIER_FAM = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 3}  # class → family

PAPER_BASELINE = 0.6633

# === CONFIG ===
@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    enc: str = "unixcoder-base"
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
    # Loss weights
    lambda_hier: float = 0.4
    lambda_supcon: float = 0.12
    temperature: float = 0.07
    contrast_dim: int = 128

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.n_cls = 6 if self.task == "author" else 2
        elif self.benchmark == "aicd_t2":
            self.n_cls = 12
            self.task = "t2"
        elif self.benchmark in ("droid_t3",):
            self.n_cls = 3
            self.task = "t3"
        elif self.benchmark == "droid_t4":
            self.n_cls = 4
            self.task = "t4"

# === HARDWARE ===
def _hw(cfg: Cfg) -> Cfg:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40:
            cfg.bs, cfg.seq = 256, 512
        elif mem >= 10:
            cfg.bs, cfg.seq = 128, 384
        else:
            cfg.bs, cfg.seq = 64, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} seq={cfg.seq}")
    return cfg

def set_seed(s):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)

# === DATA ===
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
            if isinstance(v, str) and v.strip():
                code = v
                break
        if task == "binary":
            label = 0 if _is_human(r.get("target", "")) else 1
        else:
            if _is_human(r.get("target", "")):
                label = 0
            else:
                label = vocab.get(str(r.get("model", "") or "").strip(), -1)
        return {"code": code, "label": label, "lang": str(r.get("language", "")).strip().lower()}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _conv_droid(split, task):
    lm = {"HUMAN_GENERATED": 0, "HUMAN": 0, "MACHINE_GENERATED": 1, "AI_GENERATED": 1,
          "MACHINE_REFINED": 2, "REFINED": 2, "ADVERSARIAL": 3, "ADVERSARIALLY_HUMANISED": 3}

    def row(r):
        code = str(r.get("code", "")).strip()
        raw = r.get("label", -1)
        label = lm.get(str(raw).strip().upper(), int(raw) if isinstance(raw, int) else -1)
        if task == "t3":
            label = 1 if label == 3 else label
        return {"code": code, "label": label, "lang": str(r.get("language", "")).strip().lower()}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _conv_aicd(split):
    def row(r):
        return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1)),
                 "lang": str(r.get("language", "")).strip().lower()}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _load_codet():
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
    else:
        s = ds.train_test_split(test_size=0.1, seed=42)
        s2 = s["train"].train_test_split(test_size=1 / 9, seed=42)
        return s2["train"], s2["test"], s["test"]
    return tr, vl, ts

def _load_droid():
    train_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "test-*.parquet")))
    dev_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "dev-*.parquet")))

    if train_files and test_files:
        logger.info(f"[droid] Loading from local: {len(train_files)} train shards")
        ds_train = load_dataset("parquet", data_files=train_files, split="train")
        ds_test = load_dataset("parquet", data_files=test_files, split="train")

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
    """Load AICD-Bench -- STRICT: only loads the requested task dir, NO fallback."""
    task_map = {"t1": "T1", "t2": "T2", "t3": "T3"}
    task_name = task_map.get(task.lower(), None)
    if task_name is None:
        raise ValueError(f"[aicd] Unknown task '{task}'. Must be one of: t1, t2, t3.")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(
            f"[aicd] STRICT: {task_name} dir not found at {task_path}. "
            f"NO fallback to other tasks or HuggingFace."
        )
    parquet_files = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"[aicd] STRICT: No parquet files in {task_path}. NO fallback.")
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
    s2 = s["train"].train_test_split(test_size=1 / 9, seed=42)
    return s2["train"], s2["test"], s["test"]

# === DATASET ===
class FSDS(TD):
    def __init__(self, hf, tok, max_len):
        self.hf = hf
        self.tok = tok
        self.max_len = max_len

    def __len__(self):
        return len(self.hf)

    def __getitem__(self, i):
        r = self.hf[i]
        enc = self.tok(r["code"], max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        return {
            "ids": enc["input_ids"].squeeze(0),
            "mask": enc["attention_mask"].squeeze(0),
            "label": int(r["label"])
        }

def collate(b):
    return {
        "ids": torch.stack([x["ids"] for x in b]),
        "mask": torch.stack([x["mask"] for x in b]),
        "labels": torch.tensor([x["label"] for x in b], dtype=torch.long)
    }

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
    else:  # aicd_t2
        tr_raw, vl_raw, ts_raw = _load_aicd(cfg.task)
        tr_d = _conv_aicd(tr_raw)
        vl_d = _conv_aicd(vl_raw)
        ts_d = _conv_aicd(ts_raw)

    # Fraction-based sampling
    by_cls = defaultdict(list)
    for i, lab in enumerate(tr_d["label"]):
        by_cls[int(lab)].append(i)
    rng = random.Random(cfg.seed)
    chosen = []
    for cls in range(cfg.n_cls):
        pool = by_cls.get(cls, [])
        n = max(1, int(round(len(pool) * cfg.frac))) if pool else 0
        if pool:
            chosen.extend(rng.sample(pool, min(n, len(pool))))
    rng.shuffle(chosen)
    tr_d = tr_d.select(chosen)
    logger.info(f"[data] {cfg.enc} | {cfg.benchmark} | frac={cfg.frac} | n_train={len(tr_d)}")

    def ld(ds, shuf):
        return DataLoader(FSDS(ds, tok, cfg.seq), batch_size=cfg.bs, shuffle=shuf,
                          num_workers=4, collate_fn=collate, pin_memory=True)
    return ld(tr_d, True), ld(vl_d, False), ld(ts_d, False)

# === MODEL ===
class HierSupConNet(nn.Module):
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        enc_path = os.path.join(KAGGLE_MODELS, cfg.enc)
        self.enc = AutoModel.from_pretrained(enc_path, local_files_only=True)
        h = self.enc.config.hidden_size

        self.drop = nn.Dropout(0.1)
        self.clf = nn.Linear(h, cfg.n_cls)

        # Contrastive projection head for SupCon
        self.contrast_head = nn.Sequential(
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, cfg.contrast_dim)
        )

    def forward(self, ids, mask):
        out = self.enc(input_ids=ids, attention_mask=mask)
        # Mean pooling
        emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        emb = self.drop(emb)

        logits = self.clf(emb)
        z = F.normalize(self.contrast_head(emb), dim=-1)

        return {"logits": logits, "z": z, "emb": emb}

    def groups(self):
        return [
            {"params": self.enc.parameters(), "lr": self.cfg.lr_enc, "weight_decay": self.cfg.wd},
            {"params": list(self.clf.parameters()) + list(self.contrast_head.parameters()),
             "lr": self.cfg.lr_head, "weight_decay": self.cfg.wd}
        ]

# === LOSS FUNCTIONS ===

def supcon_loss(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al., NeurIPS 2020)."""
    if z.shape[0] < 2:
        return z.new_zeros(())

    z = F.normalize(z.float(), dim=-1)
    B = z.shape[0]
    device = z.device
    labels = labels.view(-1, 1)

    # Same label = positive pair
    mask = torch.eq(labels, labels.T).float().to(device)
    mask.fill_diagonal_(0)

    if mask.sum() < 1:
        return z.new_zeros(())

    sim = torch.mm(z, z.t()) / temperature
    logits_max, _ = sim.max(dim=1, keepdim=True)
    logits = sim - logits_max.detach()

    exp_logits = torch.exp(logits) * mask
    log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
    denom = mask.sum(1).clamp(min=1e-12)
    mean_log_prob_pos = (mask * log_prob).sum(1) / denom

    loss = -mean_log_prob_pos[mask.sum(1) > 0]
    return loss.mean() if loss.numel() > 0 else z.new_zeros(())

def hier_loss(z: torch.Tensor, labels: torch.Tensor, n_cls: int, margin: float = 0.3) -> torch.Tensor:
    """Hierarchical Affinity Tree Loss.

    Forces same-family samples closer than different-family samples.
    Family mapping from HIER_FAM.
    """
    if z.shape[0] < 4:
        return z.new_zeros(())

    B = z.shape[0]
    z_norm = F.normalize(z, p=2, dim=-1)
    cos_sim = torch.mm(z_norm, z_norm.t())
    dist = 1.0 - cos_sim  # distance in [0, 2]

    # Build family labels
    fam_labels = torch.tensor([HIER_FAM.get(int(y), int(y)) for y in labels.cpu()], device=z.device)

    loss = z.new_zeros(1).squeeze()
    count = 0

    for i in range(B):
        fi = fam_labels[i].item()
        # Same-family positives (excluding self)
        same_mask = (fam_labels == fi).float()
        same_mask[i] = 0
        # Different-family negatives
        diff_mask = (fam_labels != fi).float()

        if same_mask.sum() < 1 or diff_mask.sum() < 1:
            continue

        # Hardest positive (farthest same-family)
        d_pos = (dist[i] * same_mask).clamp(min=0).max()
        # Easiest negative (closest different-family)
        d_neg = (dist[i] * diff_mask + (1 - diff_mask) * 100).min()

        triplet = F.relu(d_pos - d_neg + margin)
        loss = loss + triplet
        count += 1

    return loss / max(count, 1)

def compute_loss(logits, labels, z, w, cfg: Cfg):
    """Combined loss: CE + HierTree + SupCon."""
    # Cross-entropy
    ce = F.cross_entropy(logits, labels, weight=w)

    # Hierarchical affinity loss
    h_loss = hier_loss(z, labels, cfg.n_cls, margin=0.3)

    # Supervised contrastive loss
    s_loss = supcon_loss(z, labels, cfg.temperature)

    total = ce + cfg.lambda_hier * h_loss + cfg.lambda_supcon * s_loss

    return {
        "total": total,
        "ce": ce.item(),
        "hier": h_loss.item(),
        "supcon": s_loss.item()
    }

# === TRAIN ===
def class_w(loader, n):
    c = np.zeros(n)
    for b in loader:
        for l in b["labels"].tolist():
            c[l] += 1
    c = np.maximum(c, 1)
    w = 1.0 / c
    return torch.tensor(w / w.sum() * n, dtype=torch.float32)

@torch.no_grad()
def eval_m(model, loader, dev):
    model.eval()
    ps, ls = [], []
    for b in loader:
        with autocast(enabled=(dev.type == "cuda")):
            out = model(b["ids"].to(dev), b["mask"].to(dev))
        ps.extend(out["logits"].argmax(1).cpu().tolist())
        ls.extend(b["labels"].tolist())
    return {
        "macro": f1_score(ls, ps, average="macro", zero_division=0),
        "weighted": f1_score(ls, ps, average="weighted", zero_division=0),
        "acc": accuracy_score(ls, ps)
    }

def train(cfg: Cfg, tr_dl, vl_dl, ts_dl):
    dev = torch.device(cfg.device)
    model = HierSupConNet(cfg).to(dev)
    w = class_w(tr_dl, cfg.n_cls).to(dev)
    opt = torch.optim.AdamW(model.groups())
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[cfg.lr_enc, cfg.lr_head],
                                                steps_per_epoch=len(tr_dl), epochs=cfg.epochs, pct_start=cfg.warmup)
    scaler = GradScaler(enabled=(dev.type == "cuda"))
    best_val, best_state = 0, None

    for ep in range(cfg.epochs):
        model.train()
        pbar = tqdm(tr_dl, desc=f"Epoch {ep + 1}/{cfg.epochs}", leave=False)
        ep_losses = {"total": 0, "ce": 0, "hier": 0, "supcon": 0}

        for b in pbar:
            ids, mask, labs = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)
            with autocast(enabled=(dev.type == "cuda")):
                out = model(ids, mask)
                losses = compute_loss(out["logits"], labs, out["z"], w, cfg)
                loss = losses["total"]

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()
            sched.step()

            for k in ep_losses:
                ep_losses[k] += losses.get(k, 0)
            pbar.set_postfix({k: f"{v / (pbar.n + 1):.4f}" for k, v in ep_losses.items()})

        vr = eval_m(model, vl_dl, dev)
        if vr["macro"] > best_val:
            best_val = vr["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            logger.info(f"[epoch {ep + 1}] New best val: {best_val:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return eval_m(model, ts_dl, dev)

# === MAIN ===
def main():
    encoders = ["unixcoder-base"]
    benchmarks = [("codet_m4", "author"), ("aicd_t2", "t2")]
    fracs = [0.01, 0.05, 0.20]

    results = []
    for enc in encoders:
        for bench, task in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac)
                cfg = _hw(cfg)
                tag = f"exp_n18_hsc_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                tr_dl, vl_dl, ts_dl = build_dls(cfg)
                res = train(cfg, tr_dl, vl_dl, ts_dl)
                elapsed = time.time() - t0
                row = {
                    "tag": tag, "enc": enc, "bench": bench, "frac": frac,
                    "macro": res["macro"], "weighted": res["weighted"], "acc": res["acc"],
                    "dpaper": res["macro"] - PAPER_BASELINE, "wall": round(elapsed, 1)
                }
                results.append(row)
                logger.info(f"[{tag}] MacroF1={res['macro']:.4f} ({res['macro'] - PAPER_BASELINE:+.4f} vs paper)")
                del tr_dl, vl_dl, ts_dl
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/exp_n18_hier_supcon_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Macro-F1':>10} {'dPaper':>10} {'Weighted':>10} {'Wall':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['macro']:>10.4f} "
              f"{r['dpaper']:>+10.4f} {r['weighted']:>10.4f} {r['wall']:>8.0f}s")
    print("=" * 100)
    print(f"\nBest Macro-F1: {max(r['macro'] for r in results):.4f}")

if __name__ == "__main__":
    main()
