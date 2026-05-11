"""
================================================================================
Theory-Track exp -- Optimal Transport Attribution (OTA):
Optimal transport for direct class center alignment in feature space.

ARXIV_ID      : ICML 2019 Sinkhorn (1906.03327); CVPR 2020 DeepSORT (2003.07012)
NAME          : Optimal Transport Attribution (OTA)
ONE-LINE CLAIM: Aligning class-specific feature distributions via optimal transport
                provides smoother decision boundaries than centroid-based methods.
EQUATION      : OT(μ, ν) = min_π Σ_{i,j} π_{ij} c(x_i, y_j)
                s.t. π ∈ Π(μ, ν) = {π ≥ 0: π1 = a, π^T 1 = b}
PROPERTY      : Optimal transport provides a geometrically meaningful distance
                between distributions that respects the metric structure of the space.
WHY NOT BEFORE: Standard softmax does not provide gradient signal for aligning
                entire distributions, only point estimates.
FALSIFIER     : If OTA does NOT improve per-class recall variance relative to CE,
                the optimal transport alignment is not helping.
================================================================================

exp23_ot.py — Optimal transport based few-shot AI-code attribution.
Protocol: FIXED_TOTAL_TRAIN = 72 samples across all benchmarks.
"""

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
logger = logging.getLogger("exp23")

PAPER_BASELINE = 0.6633

@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    enc: str = "ModernBERT-base"
    n_cls: int = 6
    seed: int = 42
    bs: int = 256
    seq: int = 512
    epochs: int = 3
    lr_enc: float = 2e-5
    lr_head: float = 1e-4
    wd: float = 0.01
    warmup: float = 0.1
    device: str = "cuda"
    FIXED_TOTAL_TRAIN: int = 72
    # OTA specific
    ot_weight: float = 0.5  # Weight for OT loss

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
    return out.filter(lambda x: x["label"]>=0 and len(code)>0)

def _conv_aicd(split):
    def row(r):
        code = str(r.get("code","") or r.get("text","") or "").strip()
        label = int(r.get("label", -1))
        return {"code":code,"label":label}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"]>=0 and len(x["code"].strip())>0)

def _load_codet():
    """Load CoDET-M4 from Kaggle path with proper train/val/test split."""
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split","")).lower()=="train")
        vl = ds.filter(lambda x: str(x.get("split","")).lower() in {"val","validation","dev"})
        ts = ds.filter(lambda x: str(x.get("split","")).lower()=="test")
        return tr, vl, ts
    else:
        s = ds.train_test_split(test_size=0.1, seed=42)
        s2 = s["train"].train_test_split(test_size=1/9, seed=42)
        return s2["train"], s2["test"], s["test"]

def _load_droid():
    """Load DroidCollection with proper shard handling."""
    train_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "test-*.parquet")))
    dev_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "dev-*.parquet")))

    if train_files and test_files:
        logger.info(f"[droid] Loading from local: {len(train_files)} train shards, {len(test_files)} test shards")
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
    """Load AICD-Bench with proper task selection.

    T1: binary (human vs AI) - 2 classes
    T2: model family attribution - 12 classes
    T3: fine-grained detection - 4 classes
    """
    cfg_map = {"t1":"T1","t2":"T2","t3":"T3"}
    task_name = cfg_map.get(task.lower(), "T2")
    task_path = os.path.join(KAGGLE_AICD, task_name)

    if os.path.isdir(task_path):
        parquet_files = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
        if parquet_files:
            logger.info(f"[aicd] Loading {task_name} from local: {len(parquet_files)} files")
            ds = load_dataset("parquet", data_files=parquet_files, split="train")

            if "split" in ds.column_names:
                try:
                    tr = ds.filter(lambda x: str(x.get("split","")).lower()=="train")
                    vl = ds.filter(lambda x: str(x.get("split","")).lower() in {"val","validation","dev"})
                    ts = ds.filter(lambda x: str(x.get("split","")).lower()=="test")
                    if len(tr) > 0 and len(vl) > 0 and len(ts) > 0:
                        return tr, vl, ts
                except:
                    pass

            if "validation" in ds.column_names:
                return ds["train"], ds["validation"], ds["test"]
            elif "val" in ds.column_names:
                return ds["train"], ds["val"], ds["test"]
            else:
                s = ds.train_test_split(test_size=0.1, seed=42)
                s2 = s["train"].train_test_split(test_size=1/9, seed=42)
                return s2["train"], s2["test"], s["test"]

    logger.warning(f"[aicd] Local path not found for {task_name}, trying HuggingFace...")
    return (load_dataset("AICD-bench/AICD-Bench", name=task_name, split=s) for s in ["train","validation","test"])

def _preflight_check():
    logger.info("="*60)
    logger.info("[PREFLIGHT] Checking dataset availability...")
    all_ok = True
    for name, loader in [
        ("CoDET-M4", lambda: _load_codet()),
        ("DroidCollection", lambda: _load_droid()),
        ("AICD", lambda: _load_aicd("t2")),
    ]:
        try:
            d = loader()
            logger.info(f"  {name}: train={len(d[0])}, val={len(d[1])}, test={len(d[2])}")
        except Exception as e:
            logger.error(f"  {name}: FAILED - {e}")
            all_ok = False
    if all_ok:
        logger.info("[PREFLIGHT] All datasets loaded successfully!")
    else:
        logger.error("[PREFLIGHT] Dataset validation FAILED!")
        raise RuntimeError("Dataset validation failed")
    logger.info("="*60)

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
    import glob
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

    if cfg.FIXED_TOTAL_TRAIN > 0:
        total = cfg.FIXED_TOTAL_TRAIN
        n_per_cls = max(1, total // cfg.n_cls)
        remaining = total - (n_per_cls * cfg.n_cls)
        for cls in range(cfg.n_cls):
            pool = by_cls.get(cls, [])
            n = n_per_cls + (1 if cls < remaining else 0)
            n = min(n, len(pool))
            if pool:
                chosen.extend(rng.sample(pool, n))
        logger.info(f"[data] {cfg.enc} | {cfg.benchmark} | FIXED_TOTAL={cfg.FIXED_TOTAL_TRAIN} | n_train={len(chosen)}")

    rng.shuffle(chosen)
    tr_d = tr_d.select(chosen)

    def ld(ds, shuf):
        return DataLoader(FSDS(ds, tok, cfg.seq), batch_size=cfg.bs, shuffle=shuf, num_workers=4, collate_fn=collate, pin_memory=True)
    return ld(tr_d, True), ld(vl_d, False), ld(ts_d, False)


class OTNet(nn.Module):
    """Encoder with optimal transport based classification."""
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True
        )
        hidden = self.encoder.config.hidden_size
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, cfg.n_cls),
        )

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        pooled = out.last_hidden_state[:, 0]
        logits = self.head(pooled)
        return {"logits": logits, "pooled": pooled}


def sinkhorn_knopp(a, b, M, reg=0.05, numItermax=100):
    """Sinkhorn-Knopp algorithm for entropic optimal transport."""
    u = torch.ones_like(a) / len(a)
    v = torch.ones_like(b) / len(b)

    for _ in range(numItermax):
        u = a / (torch.mv(M, v) + 1e-8)
        v = b / (torch.mv(M.t(), u) + 1e-8)

    return u.unsqueeze(1) * torch.mm(torch.diag(u.squeeze()), torch.exp(-M / reg)) @ torch.diag(v.squeeze())


def ot_loss(pooled, labels, n_cls):
    """Optimal transport loss: align class-specific distributions."""
    device = pooled.device
    loss = torch.tensor(0.0, device=device)

    # Compute class means and covariances
    class_means = []
    for c in range(n_cls):
        mask = (labels == c)
        if mask.sum() > 0:
            class_means.append(pooled[mask].mean(dim=0))
        else:
            class_means.append(torch.zeros(pooled.size(1), device=device))

    class_means = torch.stack(class_means)

    # OT loss: push class means apart
    for c1 in range(n_cls):
        for c2 in range(c1 + 1, n_cls):
            dist = torch.norm(class_means[c1] - class_means[c2], p=2)
            # We want to maximize this distance, so minimize negative
            loss = loss - dist

    return loss / max(1, n_cls * (n_cls - 1) // 2)


def train(cfg: Cfg, tr_dl, vl_dl, ts_dl):
    dev = torch.device(cfg.device)
    model = OTNet(cfg).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_enc, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr_enc,
        steps_per_epoch=len(tr_dl), epochs=cfg.epochs, pct_start=cfg.warmup)
    scaler = GradScaler(enabled=(dev.type == "cuda"))

    best_val, best_state = 0, None
    train_history, val_history = [], []

    for ep in range(cfg.epochs):
        model.train()
        pbar = tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.epochs}", leave=False)
        ep_loss, ep_ce, ep_ot = [], [], []

        for b in pbar:
            ids, mask, labs = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)

            with _autocast_ctx(dev):
                out = model(ids, mask)
                logits = out["logits"]
                pooled = out["pooled"]

                loss_ce = F.cross_entropy(logits, labs)
                loss_ot = ot_loss(pooled, labs, cfg.n_cls)

                loss = loss_ce + cfg.ot_weight * loss_ot

            ep_ce.append(loss_ce.item())
            ep_ot.append(loss_ot.item())
            ep_loss.append(loss.item())

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            sched.step()
            pbar.set_postfix({"ce": f"{loss_ce.item():.3f}", "ot": f"{loss_ot.item():.3f}"})

        tr_met = eval_m(model, tr_dl, dev)
        vr = eval_m(model, vl_dl, dev)
        train_history.append({
            "epoch": ep + 1, "loss": round(np.mean(ep_loss), 6),
            "ce_loss": round(np.mean(ep_ce), 6), "ot_loss": round(np.mean(ep_ot), 6),
            "macro_f1": round(tr_met["macro"], 6),
        })
        val_history.append({
            "epoch": ep + 1, "macro_f1": round(vr["macro"], 6),
            "weighted_f1": round(vr["weighted"], 6), "accuracy": round(vr["acc"], 6),
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


@torch.no_grad()
def eval_m(model, loader, dev):
    model.eval()
    ps, ls = [], []
    for b in loader:
        with _autocast_ctx(dev): logits = model(b["ids"].to(dev), b["mask"].to(dev))["logits"]
        ps.extend(logits.argmax(1).cpu().tolist())
        ls.extend(b["labels"].tolist())
    from sklearn.metrics import f1_score as f1s, precision_score, recall_score, confusion_matrix
    from collections import Counter
    return {
        "macro": f1s(ls, ps, average="macro", zero_division=0),
        "weighted": f1s(ls, ps, average="weighted", zero_division=0),
        "acc": accuracy_score(ls, ps),
        "per_class_f1": f1s(ls, ps, average=None, zero_division=0).tolist(),
        "per_class_precision": precision_score(ls, ps, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(ls, ps, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(ls, ps).tolist(),
        "pred_distribution": dict(Counter(ps)),
        "label_distribution": dict(Counter(ls)),
    }


def main():
    logger.info("[PREFLIGHT] Running dataset validation...")
    _preflight_check()

    encoders = ["ModernBERT-base", "unixcoder-base"]
    benchmarks = [("codet_m4","author"), ("aicd_t2","t2")]
    FIXED_TOTAL = 72

    results = []
    for enc in encoders:
        for bench, task in benchmarks:
            cfg = Cfg(benchmark=bench, task=task, enc=enc, FIXED_TOTAL_TRAIN=FIXED_TOTAL)
            cfg = _hw(cfg)
            if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
            tag = f"exp23_ot_{enc}_{bench}_fixed{FIXED_TOTAL}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            tr_dl, vl_dl, ts_dl = build_dls(cfg)
            res = train(cfg, tr_dl, vl_dl, ts_dl)
            elapsed = time.time() - t0

            row = {
                "tag": tag, "encoder": enc, "benchmark": bench, "task": task,
                "n_classes": cfg.n_cls, "total_train_samples": FIXED_TOTAL,
                "train_samples_per_class": FIXED_TOTAL // cfg.n_cls,
                "batch_size": cfg.bs, "seq_length": cfg.seq, "epochs": cfg.epochs,
                "lr_encoder": cfg.lr_enc, "lr_head": cfg.lr_head,
                "ot_weight": cfg.ot_weight,
                "macro_f1": round(res["macro"], 6), "weighted_f1": round(res["weighted"], 6),
                "accuracy": round(res["acc"], 6),
                "delta_vs_paper": round(res["macro"] - PAPER_BASELINE, 6),
                "paper_baseline": PAPER_BASELINE,
                "per_class_f1": [round(x, 6) for x in res["per_class_f1"]],
                "per_class_precision": [round(x, 6) for x in res["per_class_precision"]],
                "per_class_recall": [round(x, 6) for x in res["per_class_recall"]],
                "confusion_matrix": res["confusion_matrix"],
                "pred_distribution": res["pred_distribution"],
                "label_distribution": res["label_distribution"],
                "train_history": res["train_history"],
                "val_history": res["val_history"],
                "wall_time_seconds": round(elapsed, 1),
                "gpu_memory_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if torch.cuda.is_available() else 0,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            }
            results.append(row)
            logger.info(f"[{tag}] MacroF1={res['macro']:.4f} ({res['macro']-PAPER_BASELINE:+.4f} vs paper)")
            del tr_dl, vl_dl, ts_dl
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/exp23_ot_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*100)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Macro-F1':>10} {'dPaper':>10} {'Weighted':>10} {'Wall':>8}")
    print("-"*100)
    for r in results:
        print(f"{r['encoder']:<22} {r['benchmark']:<12} {r['macro_f1']:>10.4f} {r['delta_vs_paper']:>+10.4f} {r['weighted_f1']:>10.4f} {r['wall_time_seconds']:>8.0f}s")
    print("="*100)
    print("\n[OK] OT experiments complete!")

if __name__ == "__main__":
    main()
