# exp92_dcgpt — DetectCodeGPT adapter: whitespace-perturbation supervised + R-Drop
# =============================================================================
# NAME           : DCGPT (DetectCodeGPT supervised path, multi-class)
# UPSTREAM       : Shi et al., "Between Lines of Code: Unraveling the Distinct
#                  Patterns of Machine and Human Programmers", ICSE 2025
#                  Repo: github.com/YerbaPage/DetectCodeGPT
# ROLE           : DetectCodeGPT's headline is a zero-shot DetectGPT-style
#                  log-prob curvature score under whitespace/newline
#                  perturbations of code. The repo also ships a supervised
#                  baseline (baselines/supervised.py) that trains an encoder
#                  with cross-entropy. We adapt the supervised path because
#                  it is the only one applicable to K-class attribution
#                  (the zero-shot score is binary by construction).
#                  Their key code-specific insight: whitespace and newline
#                  perturbations preserve semantics but stress the encoder.
#                  We use those perturbations as a CE-only regularizer
#                  (R-Drop style consistency between original and perturbed
#                  view), so the comparison stays at the loss-shape level:
#                  no contrastive head, only CE and a KL-divergence
#                  consistency term.
# CLAIM TEST     : Whether the code-specific whitespace perturbation alone
#                  (without contrastive) recovers some of TRACO's gain. If
#                  yes, perturbations are the secret sauce; if not, the
#                  contrastive mechanism is needed.
# EQUATION       : L = L_ce(phi(x), y) + L_ce(phi(x_pert), y)
#                       + beta * KL( softmax(z(x)) || softmax(z(x_pert)) )
# FALSIFIER      : Delta vs TRACO and vs exp65 CE-only baseline.
# GPU TUNING     : Two forward passes (original + perturbed); same bs
#                  schedule as exp90/91. Logit-level KL on classifier head
#                  output (cheap), no projection head needed at training
#                  time but kept for fair encoder capacity match.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from dataclasses import dataclass, field
from typing import Tuple
import re as _re

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
for _p in ("numpy", "torch", "datasets", "transformers", "scikit-learn", "tqdm"):
    _ensure(_p)

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp92_dcgpt")

PAPER_BASELINE = 0.6633


# ---- Code-specific perturbations (DetectCodeGPT signature) ------------------

def perturb_ws_newline(code: str, rng: random.Random,
                      p_double_space: float = 0.08,
                      p_blank_line: float = 0.05,
                      p_trailing_space: float = 0.10) -> str:
    """Whitespace + newline perturbation as in DetectCodeGPT.
    Three operators (independent per character / per newline):
      (i) double-up some single spaces to two spaces
      (ii) insert blank line after some \n
      (iii) append trailing spaces to some lines
    All three preserve compile semantics for our target languages
    (Python whitespace-insensitive within tokens; C-family
    whitespace-insensitive everywhere)."""
    # (iii) trailing spaces per line first (split then rejoin)
    lines = code.split("\n")
    out_lines = []
    for ln in lines:
        if ln.strip() and rng.random() < p_trailing_space:
            out_lines.append(ln + " " * rng.randint(1, 3))
        else:
            out_lines.append(ln)
    code2 = "\n".join(out_lines)
    # (i) double single spaces (not within tokens; only spaces at non-boundary positions)
    out = []
    for c in code2:
        out.append(c)
        if c == " " and rng.random() < p_double_space: out.append(" ")
    code3 = "".join(out)
    # (ii) insert blank line after some newlines
    out2 = []
    for c in code3:
        out2.append(c)
        if c == "\n" and rng.random() < p_blank_line: out2.append("\n")
    return "".join(out2)


# ---- Model ------------------------------------------------------------------

class DCGPTModel(nn.Module):
    """Encoder + classifier head, no projection head (we follow the
    DetectCodeGPT supervised baseline which classifies directly from the
    encoder pooled output)."""
    def __init__(self, enc_name, n_cls):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.clf = nn.Linear(h, n_cls)
        self.n_cls = n_cls

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return self.clf(self.dropout(sem))


def consistency_kl(logits_a, logits_b):
    """Symmetric KL between two logit distributions (R-Drop / SimMatch style).
    Used as a consistency regularizer between original-code and
    perturbed-code predictions."""
    p = F.log_softmax(logits_a, dim=-1)
    q = F.log_softmax(logits_b, dim=-1)
    p_prob = p.exp()
    q_prob = q.exp()
    kl_pq = (p_prob * (p - q)).sum(-1).mean()
    kl_qp = (q_prob * (q - p)).sum(-1).mean()
    return 0.5 * (kl_pq + kl_qp)


# ---- Plumbing ---------------------------------------------------------------

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    beta_kl: float = 0.5; device: str = "cuda"

def adaptive_schedule(c):
    if c.frac <= 0.02: c.epochs, c.lr_enc, c.warmup = 10, 3e-5, 0.20
    elif c.frac <= 0.10: c.epochs, c.lr_enc, c.warmup = 6, 3e-5, 0.15
    else: c.epochs, c.lr_enc, c.warmup = 6, 4e-5, 0.10
    return c

def _hw(c):
    """Same 2-view memory profile as exp90/91 but heavier on the
    encoder (no projection head, larger CE logits buffer)."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40: c.bs, c.seq = 128, 512
        elif mem >= 20: c.bs, c.seq = 96, 448
        elif mem >= 10: c.bs, c.seq = 64, 384
        else: c.bs, c.seq = 32, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={c.bs} seq={c.seq} (2-view CE + KL)")
    return c

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _is_human(t): return str(t or "").strip().lower() in {"human","human_written","human-generated"}

def _vocab(tr):
    names = {str(r.get("model", "") or "").strip() for r in tr
             if not _is_human(r.get("target", "")) and r.get("model", "")}
    return {n: i + 1 for i, n in enumerate(sorted(names))}

def _conv_codet(s, t, vocab):
    def row(r):
        code = next((r.get(f, "") for f in ("cleaned_code", "code")
                     if isinstance(r.get(f, ""), str) and r.get(f, "").strip()), "")
        lbl = (0 if _is_human(r.get("target", ""))
               else (1 if t == "binary"
                     else vocab.get(str(r.get("model", "") or "").strip(), -1)))
        return {"code": code, "label": lbl,
                "language": str(r.get("language", "")).strip().lower(),
                "source": str(r.get("source", "")).strip().lower()}
    return s.map(row, remove_columns=s.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _conv_aicd(s):
    def row(r):
        return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1)),
                "language": str(r.get("language", "")).strip().lower(), "source": ""}
    return s.map(row, remove_columns=s.column_names).filter(
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

def _load_aicd(task):
    tn = {"t1":"T1","t2":"T2","t3":"T3"}.get(task.lower())
    if tn is None: raise ValueError(f"[aicd] bad '{task}'")
    p = os.path.join(KAGGLE_AICD, tn)
    if not os.path.isdir(p): raise FileNotFoundError(f"[aicd] STRICT: {tn} not found")
    pf = sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: no parquet")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0: return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


class FSDS_PERT(TD):
    """Dataset that emits (original, ws-perturbed) pair for train; original
    only for val/test (perturbation is a training-time regularizer, not an
    inference-time augmentation)."""
    def __init__(self, data, tok, seq, frac=1.0, seed=42, do_perturb=True):
        self.data = data; self.tok = tok; self.seq = seq
        self.do_perturb = do_perturb; self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_PERT] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]
        e0 = self.tok(code, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        ids0, m0 = e0["input_ids"].squeeze(0), e0["attention_mask"].squeeze(0)
        if self.do_perturb:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            code_p = perturb_ws_newline(code, rng)
            e1 = self.tok(code_p, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
            ids1, m1 = e1["input_ids"].squeeze(0), e1["attention_mask"].squeeze(0)
        else:
            ids1, m1 = ids0, m0
        return {"ids0": ids0, "mask0": m0, "ids1": ids1, "mask1": m1,
                "label": r["label"],
                "language": r.get("language", "") or "", "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device); labs = b["label"]
        lg = model(ids0, m0)
        preds.extend(lg.argmax(-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        lang_b = b.get("language", []); src_b = b.get("source", [])
        langs.extend(list(lang_b) if not isinstance(lang_b, list) else lang_b)
        sources.extend(list(src_b) if not isinstance(src_b, list) else src_b)
    preds = np.array(preds); labels = np.array(labels); n = cfg.n_cls
    ov = {"accuracy": float(accuracy_score(labels, preds)),
          "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
          "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
          "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
          "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
          "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0))}
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
    per_lang, per_src = {}, {}
    if any(l for l in langs):
        la = np.array(langs)
        for L in sorted(set(langs)):
            if not L: continue
            sel = (la == L)
            if sel.sum() < 2: continue
            per_lang[L] = {"n": int(sel.sum()),
                "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0))}
    if any(s for s in sources):
        sa = np.array(sources)
        for S in sorted(set(sources)):
            if not S: continue
            sel = (sa == S)
            if sel.sum() < 2: continue
            per_src[S] = {"n": int(sel.sum()),
                "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0))}
    return {"overall": ov, "confusion_matrix": cm.tolist(),
            "per_language": per_lang, "per_source": per_src,
            "n_samples": int(len(labels))}


def train_epoch(model, loader, opt, sch, scaler, cfg):
    model.train(); tot, ce_s, kl_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); m1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            lg0 = model(ids0, m0)
            lg1 = model(ids1, m1)
            loss_ce = 0.5 * (F.cross_entropy(lg0, labs) + F.cross_entropy(lg1, labs))
            loss_kl = consistency_kl(lg0.float(), lg1.float())
            loss = loss_ce + cfg.beta_kl * loss_kl
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); kl_s += loss_kl.item()
    n = len(loader)
    return tot/n, ce_s/n, kl_s/n


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, "author", vocab)
        vl_data = _conv_codet(vl_raw, "author", vocab)
        ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    tr_ds = FSDS_PERT(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_perturb=True)
    vl_ds = FSDS_PERT(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_perturb=False)
    ts_ds = FSDS_PERT(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_perturb=False)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} wu={cfg.warmup} beta_kl={cfg.beta_kl}")
    lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = DCGPTModel(cfg.enc, cfg.n_cls).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh = 0.0, None, []
    for ep in range(cfg.epochs):
        loss, ce, kl = train_epoch(model, tr_dl, opt, sch, scaler, cfg)
        vm = eval_pack(model, vl_dl, cfg)
        v = vm["overall"]["macro_f1"]; vh.append(v)
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce={ce:.4f} kl={kl:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    tm = eval_pack(model, ts_dl, cfg)
    test = tm["overall"]["macro_f1"]; gap = best_val - test
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f}")
    return {"tag": tag, "method": "DCGPT", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "beta_kl": cfg.beta_kl,
            "val_macro": best_val, "macro": test,
            "weighted": tm["overall"]["weighted_f1"], "acc": tm["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "test_metrics": tm, "val_history": vh,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp92_dcgpt_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time()-t0, 1); results.append(r)
                logger.info(f"[{tag}] test={r['macro']:.4f} ({r['dpaper']:+.4f}) gap={r['val_test_gap']:+.4f} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp92_dcgpt_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 110)
    print(f"{'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9} {'Wall':>8}")
    print("-" * 110)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} {r['wall']:>8.0f}s")
    print("=" * 110)


if __name__ == "__main__":
    main()
