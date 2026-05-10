"""
exp03_style.py — Style representation baseline

Published method: Few-shot detection via style representations
Style: code stylometry features + logistic regression

Self-contained. Runs: 4 benchmarks × 3 fractions = 12 experiments.

Config:
  - Benchmarks: codet_m4 (headline), aicd_t2 (stress), droid_t3, droid_t4
  - Fractions: 0.01, 0.05, 0.20

Usage:
  python exp03_style.py
"""

# === KAGGLE PATHS ===
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_DROID = "/kaggle/input/datasets/chiboiz/droid-collection/DroidCollection"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

from __future__ import annotations
import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("datasets"); _ensure("scikit-learn")

import numpy as np
from datasets import load_dataset
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp03")

HIER_FAM = {0:0,1:1,2:2,3:3,4:1,5:4}
PAPER_BASELINE = 0.6633

@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    n_cls: int = 6
    frac: float = 0.05
    seed: int = 42

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.n_cls = 6 if self.task == "author" else 2
        elif self.benchmark == "aicd_t2":
            self.n_cls = 12; self.task = "t2"
        elif self.benchmark in ("droid_t3",):
            self.n_cls = 3; self.task = "t3"
        elif self.benchmark == "droid_t4":
            self.n_cls = 2; self.task = "t4"

def set_seed(s):
    random.seed(s); np.random.seed(s)

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
        return {"code":code,"label":label}
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
        return {"code":code,"label":label}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"]>=0 and len(x["code"].strip())>0)

def _conv_aicd(split):
    def row(r): return {"code":str(r.get("code","")).strip(),"label":int(r.get("label",-1))}
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
    files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "**","*.parquet"), recursive=True))
    ds = load_dataset("parquet", data_files=files, split="train") if files else load_dataset(KAGGLE_DROID, split="train")
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]

def _load_aicd(task):
    cfg_map = {"t2":"T2","t3":"T3","t1":"T1"}
    if os.path.isdir(KAGGLE_AICD):
        return (load_dataset(KAGGLE_AICD, name=cfg_map.get(task,"T2"), split=s) for s in ["train","validation","test"])
    return (load_dataset("AICD-bench/AICD-Bench", name=cfg_map.get(task,"T2"), split=s) for s in ["train","validation","test"])

def _style_features(code: str) -> Dict[str, float]:
    lines = code.splitlines() or [code]
    stripped = [ln.strip() for ln in lines]
    tokens = code.replace("\n", " ").split()
    n_chars = max(1, len(code))
    n_lines = max(1, len(lines))
    n_tokens = max(1, len(tokens))

    keywords = ["for","while","if","else","return","class","public","static","def","import","try","except"]
    feats = {
        "chars": float(len(code)),
        "lines": float(n_lines),
        "tokens": float(n_tokens),
        "avg_line_len": float(sum(len(x) for x in lines) / n_lines),
        "avg_token_len": float(sum(len(x) for x in tokens) / n_tokens),
        "blank_ratio": float(sum(1 for x in stripped if not x) / n_lines),
        "indent_mean": float(sum(len(ln) - len(ln.lstrip(" ")) for ln in lines) / n_lines),
        "brace_rate": code.count("{") / n_chars,
        "paren_rate": code.count("(") / n_chars,
        "semicolon_rate": code.count(";") / n_chars,
        "comment_rate": (code.count("//") + code.count("#")) / n_chars,
        "underscore_rate": code.count("_") / n_chars,
        "digit_rate": sum(ch.isdigit() for ch in code) / n_chars,
    }
    low = " " + code.lower() + " "
    for kw in keywords:
        feats[f"kw_{kw}"] = low.count(f" {kw} ") / n_tokens
    return feats

def load_data(cfg: Cfg):
    set_seed(cfg.seed)

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
    logger.info(f"[data] style | {cfg.benchmark} | frac={cfg.frac} | n_train={len(tr_d)}")
    return tr_d, vl_d, ts_d

def main():
    benchmarks = [("codet_m4","author"), ("aicd_t2","t2"), ("droid_t3","t3"), ("droid_t4","t4")]
    fracs = [0.01, 0.05, 0.20]

    results = []
    for bench, task in benchmarks:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac)
            tag = f"exp03_style_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()

            tr_d, vl_d, ts_d = load_data(cfg)

            x_train = [_style_features(r["code"]) for r in tr_d]
            y_train = list(tr_d["label"])
            x_val = [_style_features(r["code"]) for r in vl_d]
            y_val = list(vl_d["label"])
            x_test = [_style_features(r["code"]) for r in ts_d]
            y_test = list(ts_d["label"])

            clf = make_pipeline(
                DictVectorizer(sparse=False),
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.seed, n_jobs=-1),
            )
            clf.fit(x_train, y_train)
            val_pred = clf.predict(x_val)
            test_pred = clf.predict(x_test)

            val_macro = f1_score(y_val, val_pred, average="macro", zero_division=0)
            test_macro = f1_score(y_test, test_pred, average="macro", zero_division=0)
            test_weighted = f1_score(y_test, test_pred, average="weighted", zero_division=0)
            test_acc = accuracy_score(y_test, test_pred)
            elapsed = time.time() - t0

            row = {"tag": tag, "enc": "style_repr", "bench": bench, "frac": frac,
                   "macro": test_macro, "weighted": test_weighted, "acc": test_acc,
                   "dpaper": test_macro - PAPER_BASELINE, "wall": round(elapsed,1)}
            results.append(row)
            logger.info(f"[{tag}] MacroF1={test_macro:.4f} ({test_macro-PAPER_BASELINE:+.4f} vs paper) time={elapsed:.0f}s")

            del tr_d, vl_d, ts_d

    os.makedirs("results", exist_ok=True)
    with open("results/exp03_results.json", "w") as f:
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
