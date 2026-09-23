# exp160 — LEAVEONEGEN  (E1: the Sink-Class experiment)
# =============================================================================
# NAME       : LEAVEONEGEN (Leave-One-Generator-Out sink-class measurement).
# REFERENCE  : new (D1 proposal, related_work/PROPOSAL_D1_sinkclass.md).
#              Empirical anchor for the "Sink-Class Law"; the harness is the
#              plain-CE UniXcoder attributor of ext_gptsniffer (GPTSniffer-K,
#              Nguyen et al., JSS 2024), our strongest faithful baseline.
# CLAIM      : A K-class attributor trained on K-1 seen generators does NOT
#              spread an unseen generator's samples uniformly over the seen
#              classes — it dumps them into a dominant "sink" class (the human
#              class where one exists), i.e. an unseen AI generator is read as
#              human / as one seen author far above the uniform 1/(K-1) rate.
# EQUATION   : SINK_RATE(g) = P( pred in sink-set | true = held-out gen g );
#              headline sink-set = {human}; also report empirical argmax sink,
#              routing entropy H, and routing concentration (max routing prob).
# WHY NEW    : No code-attribution paper measures WHERE an out-of-support
#              generator is routed under an open-world (held-out class) split;
#              prior OOD work reports only accuracy drop, never the sink target.
# WOW HOOK   : "Take any generator out of the training set and the detector
#              does not get confused — it confidently calls that unseen AI
#              'human'. Attribution failure is not noise, it has an address."
# FALSIFIER  : If, averaged over held-out generators, the sink-set routing rate
#              is within +/-2x of the uniform baseline 1/(K-1) (i.e. no class
#              disproportionately absorbs the held-out generator) on BOTH
#              CoDET-M4 and AICD-T2, the sink-class claim is falsified.
# =============================================================================
from __future__ import annotations

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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp160")

# -----------------------------------------------------------------------------
# Paths (env-overridable; Kaggle defaults, never hardcode a d:\ path)
# -----------------------------------------------------------------------------
KAGGLE_MODELS = os.environ.get("MODELS_DIR", "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models")
KAGGLE_CODET  = os.environ.get("CODET_PARQUET", "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet")
KAGGLE_AICD   = os.environ.get("AICD_DIR", "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench")

# -----------------------------------------------------------------------------
# Data loading — identical protocol to ext_gptsniffer.py
# -----------------------------------------------------------------------------
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

# -----------------------------------------------------------------------------
# Dataset — faithful encode_plus tokenisation, optional per-class fraction
# -----------------------------------------------------------------------------
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
            logger.info(f"[FSDS] kept {len(self.data)} (frac={frac} cap={cap})")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        enc = self.tok(r["code"][:5000], padding="max_length", max_length=self.seq_len, truncation=True)
        return {"input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long),
                "label": int(r["label"]), "language": r.get("language", "")}

# -----------------------------------------------------------------------------
# Config / schedule / hw  (reused from ext_gptsniffer)
# -----------------------------------------------------------------------------
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
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs}")
    else:
        cfg.device = "cpu"; cfg.bs = 8
    return cfg

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

# -----------------------------------------------------------------------------
# Train a plain-CE attributor on a given (already contiguous-labelled) split
# -----------------------------------------------------------------------------
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
        logger.info(f"  [ep{ep+1}] val_macro={v:.4f}")
        if v > best_val:
            best_val = v; best_state = {k: t.cpu().clone() for k, t in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    return model, best_val

# -----------------------------------------------------------------------------
# Leave-One-Generator-Out core
# -----------------------------------------------------------------------------
def _remap(data, keep_labels):
    """Filter dataset to keep_labels and remap to contiguous 0..K-1.
    Returns (new_dataset, contig2orig list)."""
    contig2orig = sorted(keep_labels)
    orig2contig = {o: i for i, o in enumerate(contig2orig)}
    kept = data.filter(lambda x: x["label"] in orig2contig)
    kept = kept.map(lambda x: {"label": orig2contig[x["label"]]})
    return kept, contig2orig

def run_bench(cfg, human_class):
    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet(); vocab = _vocab(tr_raw)
        tr = _conv_codet(tr_raw, vocab); vl = _conv_codet(vl_raw, vocab); ts = _conv_codet(ts_raw, vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr = _conv_aicd(tr_raw); vl = _conv_aicd(vl_raw); ts = _conv_aicd(ts_raw)
    all_labels = sorted(set(tr["label"]))
    n_cls = max(all_labels) + 1
    generators = [c for c in all_labels if c != human_class]
    if cfg.smoke: generators = generators[:2]
    logger.info(f"[{cfg.benchmark}] n_cls={n_cls} human={human_class} generators={generators}")
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    cap = 40 if cfg.smoke else None
    records = []
    for g in generators:
        keep = [c for c in all_labels if c != g]
        tr_k, c2o = _remap(tr, keep); vl_k, _ = _remap(vl, keep)
        K = len(keep)
        tr_ds = FSDS(tr_k, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, cap=cap, n_labels=K)
        vl_ds = FSDS(vl_k, tok, cfg.seq, frac=1.0, seed=cfg.seed + 1, cap=cap, n_labels=K)
        ts_ds = FSDS(ts, tok, cfg.seq, frac=1.0, seed=cfg.seed + 2, cap=cap, n_labels=n_cls)
        logger.info(f"--- held-out gen={g}  K={K}  train={len(tr_ds)} ---")
        model, best_val = train_ce(tr_ds, vl_ds, K, cfg)
        ts_dl = DataLoader(ts_ds, shuffle=False, batch_size=cfg.bs, num_workers=2, pin_memory=True)
        pred_contig, lab_orig = _predict(model, ts_dl, cfg)
        pred_orig = np.array([c2o[p] for p in pred_contig])
        # held-out generator routing
        held = (lab_orig == g)
        n_held = int(held.sum())
        routing = {}
        for c in keep:
            routing[int(c)] = float((pred_orig[held] == c).mean()) if n_held else 0.0
        uniform = 1.0 / max(1, K)
        sink_human = routing.get(int(human_class), None) if human_class is not None else None
        dom_class = int(max(routing, key=routing.get)) if routing else -1
        dom_rate = routing.get(dom_class, 0.0)
        probs = np.array([routing[c] for c in keep]) + 1e-12
        ent = float(-(probs * np.log(probs)).sum())
        # full confusion (rows = true orig incl held-out, cols = seen orig classes)
        cm = confusion_matrix(lab_orig, pred_orig, labels=all_labels)  # cols include held-out (all zero col)
        # seen-class test macro (exclude held-out samples)
        seen_mask = (lab_orig != g)
        seen_macro = float(f1_score(lab_orig[seen_mask], pred_orig[seen_mask], average="macro", zero_division=0)) if seen_mask.any() else 0.0
        rec = {"tag": f"exp160_{cfg.benchmark}_holdout{g}", "bench": cfg.benchmark, "frac": cfg.frac,
               "held_out_gen": int(g), "human_class": (int(human_class) if human_class is not None else None),
               "K_seen": K, "n_held_test": n_held, "uniform_baseline": uniform,
               "val_macro": best_val, "seen_test_macro": seen_macro,
               "routing_distribution": routing,
               "sink_rate_human": sink_human,
               "sink_rate_human_vs_uniform": (sink_human / uniform if sink_human is not None else None),
               "dominant_sink_class": dom_class, "dominant_sink_rate": dom_rate,
               "dominant_sink_vs_uniform": dom_rate / uniform,
               "routing_entropy": ent, "routing_max": float(probs.max()),
               "confusion_full": cm.tolist(), "class_labels": all_labels,
               "contig2orig": c2o, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
        logger.info(f"  gen={g}: sink_human={sink_human} dom={dom_class}@{dom_rate:.3f} "
                    f"(uniform={uniform:.3f}) seen_macro={seen_macro:.3f}")
        records.append(rec)
        del model
        import gc; gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
    # summary
    hs = [r["sink_rate_human"] for r in records if r["sink_rate_human"] is not None]
    ds = [r["dominant_sink_rate"] for r in records]
    summary = {"tag": f"exp160_{cfg.benchmark}_SUMMARY", "bench": cfg.benchmark, "frac": cfg.frac,
               "n_holdouts": len(records),
               "mean_sink_rate_human": (float(np.mean(hs)) if hs else None),
               "mean_dominant_sink_rate": float(np.mean(ds)) if ds else None,
               "mean_uniform_baseline": float(np.mean([r["uniform_baseline"] for r in records])) if records else None}
    logger.info(f"[SUMMARY {cfg.benchmark}] mean_sink_human={summary['mean_sink_rate_human']} "
                f"mean_dom_sink={summary['mean_dominant_sink_rate']}")
    return records + [summary]

def main():
    bench = os.environ.get("BENCH", "codet_m4")
    frac = float(os.environ.get("FRAC", "0.20"))
    smoke = os.environ.get("SMOKE", "0") == "1"
    seed = int(os.environ.get("SEED", "42"))
    hc_env = os.environ.get("HUMAN_CLASS", "")
    # CoDET-M4: class 0 = human. AICD-T2: no human class (all 12 are generators).
    if hc_env != "":
        human_class = None if int(hc_env) < 0 else int(hc_env)
    else:
        human_class = 0 if bench == "codet_m4" else None
    benches = [bench] if bench != "all" else ["codet_m4", "aicd_t2"]
    all_records = []
    for b in benches:
        cfg = Cfg(benchmark=b, frac=frac, seed=seed, smoke=smoke)
        cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
        hc = 0 if (b == "codet_m4" and hc_env == "") else human_class
        try:
            all_records.extend(run_bench(cfg, hc))
        except Exception as e:
            logger.error(f"[{b}] FAILED: {e}")
            import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp160_leaveonegen_results.json"), "w") as f:
        json.dump(all_records, f, indent=2)
    logger.info(f"[done] wrote {len(all_records)} records")

if __name__ == "__main__":
    main()
