# ext_llmsniffer — Faithful K-class reproduction of LLMSniffer (arXiv 2024)
# =============================================================================
# UPSTREAM    : "LLMSniffer: Detecting LLM-generated Code via GraphCodeBERT +
#               Supervised Contrastive Learning" — arXiv 2024
#
# FAITHFULNESS: Faithful to notebook cell 9 (lines 400-420):
#   - RobertaModel encoder (UniXcoder for protocol parity)
#   - CLS token (index 0 of last hidden state)
#   - CRITICAL: cls_output.detach() before classifier (line 408)
#     → CE gradient flows only through the classifier MLP head
#     → SupCon gradient flows only through the encoder
#   - Classifier: Sequential(Dropout(0.3), Linear(hidden,128), BatchNorm1d(128),
#                             ReLU(), Dropout(0.3), Linear(128, K))
#   - SupConLoss(τ=0.07) on un-detached CLS
#   - Total loss: CE(classifier(cls.detach()), labels) + SupCon(cls, labels)
#   - Differential LR: encoder=1e-6, head=1e-4
#   - 2D attention_mask: ids.ne(pad_id) (correct for newer transformers)
#
# Adaptive schedule (encoder lr is very small → more epochs needed):
#   frac ≤ 1%:  15 epochs
#   frac ≤ 5%:  10 epochs
#   frac ≤ 20%:  8 epochs
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from dataclasses import dataclass, field
from typing import Dict, List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from torch.utils.data import Dataset as TD, DataLoader
from transformers import (AutoTokenizer, RobertaConfig, RobertaModel,
                          get_cosine_schedule_with_warmup)
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("ext_llmsniffer")

# =============================================================================
# Shared constants
# =============================================================================

PAPER_BASELINE = 0.6633

GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD  = {i: [(i // 3) * 3 + j for j in range(3)
                       if (i // 3) * 3 + j != i] for i in range(12)}


def _gene_distance(u, v, adj):
    if u == v: return 0.0
    queue = [(u, 0)]; visited = {u}
    while queue:
        curr, d = queue.pop(0)
        for nb in adj.get(curr, []):
            if nb == v: return d + 1.0
            if nb not in visited:
                visited.add(nb); queue.append((nb, d + 1))
    return float("inf")


def build_distance_matrix(n_cls, adj, default_dist=4.0):
    D = torch.full((n_cls, n_cls), default_dist)
    for i in range(n_cls):
        for j in range(n_cls):
            d = _gene_distance(i, j, adj)
            if d < float("inf"): D[i, j] = d
            elif (i == 0) != (j == 0): D[i, j] = 3.0
    return D


def build_sibling_mask(n_cls, adj):
    M = torch.zeros(n_cls, n_cls)
    for i in range(n_cls):
        for j in adj.get(i, []): M[i, j] = 1.0
    return M


# =============================================================================
# Config & hardware helpers
# =============================================================================

@dataclass
class Cfg:
    benchmark:   str   = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac:        float = 0.20;        n_cls: int = 6;       seed: int = 42
    bs:          int   = 16;          seq:   int = 512;     epochs: int = 8
    lr_enc:      float = 1e-6;        lr_head: float = 1e-4
    warmup:      float = 0.10;        wd: float = 0.01
    temperature: float = 0.07
    device:      str   = "cuda"
    gene_adj:    dict  = field(default_factory=dict)


def adaptive_schedule(cfg):
    """Encoder LR is very small (1e-6) so more epochs are needed at low fractions."""
    f = cfg.frac
    if f <= 0.02:   cfg.epochs, cfg.warmup = 15, 0.20
    elif f <= 0.10: cfg.epochs, cfg.warmup = 10, 0.15
    else:           cfg.epochs, cfg.warmup = 8,  0.10
    return cfg


def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40:   cfg.bs = 64
        elif mem >= 20: cfg.bs = 32
        else:           cfg.bs = 16
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs}")
    return cfg


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


# =============================================================================
# Data loading
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD  = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


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
            if isinstance(v, str) and v.strip(): code = v; break
        label = 0 if _is_human(r.get("target", "")) else vocab.get(
            str(r.get("model", "") or "").strip(), -1)
        return {"code": code, "label": label,
                "language": str(r.get("language", "")).strip().lower(),
                "source":   str(r.get("source",   "")).strip().lower()}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _conv_aicd(split):
    def row(r):
        return {"code":     str(r.get("code",     "")).strip(),
                "label":    int(r.get("label",    -1)),
                "language": str(r.get("language", "")).strip().lower(),
                "source":   ""}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _load_codet():
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        return tr, vl, ts
    s  = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


def _load_aicd(task):
    task_name = {"t1": "T1", "t2": "T2", "t3": "T3"}.get(task.lower())
    if task_name is None: raise ValueError(f"[aicd] Unknown task '{task}'")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path): raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found")
    pf = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: No parquet files")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0: return tr, vl, ts
    s  = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


# =============================================================================
# Dataset — uses manual UniXcoder tokenisation protocol
# =============================================================================

def _tokenize(code, tokenizer, max_len):
    toks = tokenizer.tokenize(" ".join(code.split()))[:max_len - 4]
    toks = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + toks + [tokenizer.sep_token]
    ids  = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_len - len(ids))
    return ids[:max_len]


class FSDS(TD):
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42):
        self.data = data; self.tok = tok; self.seq_len = seq_len
        if frac < 1.0:
            rng    = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep   = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx) * frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS] Sampled {len(self.data)} samples ({frac * 100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r   = self.data[i]
        ids = _tokenize(r["code"][:5000], self.tok, self.seq_len)
        return {"input_ids": torch.tensor(ids, dtype=torch.long),
                "label":     r["label"],
                "language":  r.get("language", "") or "",
                "source":    r.get("source",   "") or ""}


# =============================================================================
# SupConLoss — faithful copy from cpsniffer.ipynb
# =============================================================================

class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        features_normalized = F.normalize(features, dim=1)
        logits = (features_normalized @ features_normalized.T) / self.temperature
        logits_max = torch.max(logits, dim=1, keepdim=True).values
        logits = logits - logits_max.detach()
        exp_logits = torch.exp(logits) * mask
        log_prob   = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)
        mask_sum   = mask.sum(dim=1)
        mean_log_prob_pos = (log_prob * mask).sum(dim=1) / (mask_sum + 1e-12)
        return -mean_log_prob_pos.mean()


# =============================================================================
# Model — faithful to LLMSniffer notebook
# =============================================================================

class LLMSnifferK(nn.Module):
    """Faithful reproduction of LLMSniffer notebook lines 400-420.

    Key: cls_output.detach() before classifier (line 408):
      - CE gradient flows ONLY through the classifier MLP
      - SupCon gradient flows ONLY through the encoder
    This is the defining architectural choice of LLMSniffer.
    """

    def __init__(self, encoder, hidden, n_cls, pad_id):
        super().__init__()
        self.encoder = encoder
        self.pad_id  = pad_id
        self.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(hidden, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, n_cls),
        )

    def _cls_token(self, input_ids):
        # 2D attention_mask: correct for newer transformers (fixes expand RuntimeError)
        mask = input_ids.ne(self.pad_id)
        out  = self.encoder(input_ids, attention_mask=mask, output_hidden_states=True)
        return out[0][:, 0, :]  # CLS token

    def forward(self, input_ids):
        cls = self._cls_token(input_ids)
        cls_detached = cls.detach()  # FAITHFUL: notebook line 408
        logits = self.classifier(cls_detached)
        return logits, cls  # logits for CE, cls (un-detached) for SupCon


# =============================================================================
# Evaluation — full eval_pack with sibling tracking
# =============================================================================

@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu):
    model.eval()
    preds, labels, langs, sources = [], [], [], []

    for b in tqdm(loader, desc="Eval"):
        ids  = b["input_ids"].to(cfg.device)
        labs = b["label"]
        if not torch.is_tensor(labs): labs = torch.tensor(labs, dtype=torch.long)
        labs = labs.to(cfg.device)
        logits, _ = model(ids)
        preds.extend(logits.argmax(-1).cpu().tolist())
        labels.extend(labs.cpu().tolist())
        langs.extend(list(b.get("language", [""] * len(labs))))
        sources.extend(list(b.get("source",   [""] * len(labs))))

    preds  = np.array(preds)
    labels = np.array(labels)
    n_cls  = cfg.n_cls

    overall = {
        "accuracy":        float(accuracy_score(labels, preds)),
        "macro_f1":        float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "weighted_f1":     float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "micro_f1":        float(f1_score(labels, preds, average="micro",    zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(labels, preds, average="macro",    zero_division=0)),
    }

    per_class = {
        "f1":        f1_score(labels, preds, average=None, zero_division=0,
                              labels=list(range(n_cls))).tolist(),
        "precision": precision_score(labels, preds, average=None, zero_division=0,
                                     labels=list(range(n_cls))).tolist(),
        "recall":    recall_score(labels, preds, average=None, zero_division=0,
                                  labels=list(range(n_cls))).tolist(),
    }

    cm       = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    off_diag = int(cm.sum() - cm.trace())

    sib_conf  = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls)
                        if i != j and sib_mask_np[i, j] > 0))
    sib_rate  = sib_conf / max(off_diag, 1)

    cross      = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls)
                         if i != j and dist_mat_cpu[i, j] >= 3.0))
    cross_rate = cross / max(off_diag, 1)

    per_lang, per_src = {}, {}
    if any(l for l in langs):
        la = np.array(langs)
        for L in sorted(set(langs)):
            if not L: continue
            sel = (la == L)
            if sel.sum() < 2: continue
            per_lang[L] = {
                "n":           int(sel.sum()),
                "macro_f1":    float(f1_score(labels[sel], preds[sel], average="macro",    zero_division=0)),
                "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                "accuracy":    float(accuracy_score(labels[sel], preds[sel])),
            }
    if any(s for s in sources):
        sa = np.array(sources)
        for S in sorted(set(sources)):
            if not S: continue
            sel = (sa == S)
            if sel.sum() < 2: continue
            per_src[S] = {
                "n":           int(sel.sum()),
                "macro_f1":    float(f1_score(labels[sel], preds[sel], average="macro",    zero_division=0)),
                "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                "accuracy":    float(accuracy_score(labels[sel], preds[sel])),
            }

    return {
        "overall":                     overall,
        "per_class":                   per_class,
        "per_language":                per_lang,
        "per_source":                  per_src,
        "confusion_matrix":            cm.tolist(),
        "sibling_confusion_rate":      float(sib_rate),
        "cross_family_confusion_rate": float(cross_rate),
        "off_diag_total":              off_diag,
        "n_samples":                   int(len(labels)),
    }


# =============================================================================
# Training loop
# =============================================================================

def train_epoch(model, loader, opt, sch, scaler, cfg, supcon_fn):
    model.train(); tot = 0.0
    for b in tqdm(loader, desc="Train"):
        ids  = b["input_ids"].to(cfg.device)
        labs = b["label"]
        if not torch.is_tensor(labs): labs = torch.tensor(labs, dtype=torch.long)
        labs = labs.to(cfg.device)
        opt.zero_grad()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda")):
            logits, cls = model(ids)
            # CE gradient → classifier only (cls detached inside model)
            loss_ce  = F.cross_entropy(logits, labs)
            # SupCon gradient → encoder only (cls is un-detached)
            loss_scl = supcon_fn(cls, labs)
            loss     = loss_ce + loss_scl
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sch.step()
        tot += loss.item()
    return tot / max(1, len(loader))


# =============================================================================
# Experiment runner
# =============================================================================

def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD

    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab   = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, "author", vocab)
        vl_data = _conv_codet(vl_raw, "author", vocab)
        ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw)
        vl_data = _conv_aicd(vl_raw)
        ts_data = _conv_aicd(ts_raw)

    cfg.n_cls = max(tr_data["label"]) + 1

    tok    = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    pad_id = tok.pad_token_id

    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0,      seed=cfg.seed + 1)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0,      seed=cfg.seed + 2)

    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True,  **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    config  = RobertaConfig.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    encoder = RobertaModel.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    model   = LLMSnifferK(encoder, config.hidden_size, cfg.n_cls, pad_id).to(cfg.device)

    supcon_fn = SupConLoss(temperature=cfg.temperature).to(cfg.device)

    # Differential LR: faithful to notebook
    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(),    "lr": cfg.lr_enc},
        {"params": model.classifier.parameters(), "lr": cfg.lr_head},
    ], weight_decay=cfg.wd)
    sch    = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()

    dist_mat_cpu = build_distance_matrix(cfg.n_cls, cfg.gene_adj).numpy()
    sib_mask_np  = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()

    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr_enc={cfg.lr_enc} lr_head={cfg.lr_head} "
                f"τ={cfg.temperature} n_cls={cfg.n_cls} train={len(tr_ds)}")

    best_val, best_state, val_hist = 0.0, None, []

    for ep in range(cfg.epochs):
        loss    = train_epoch(model, tr_dl, opt, sch, scaler, cfg, supcon_fn)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v       = val_met["overall"]["macro_f1"]
        val_hist.append(v)
        logger.info(f"[epoch {ep + 1}] loss={loss:.4f} val={v:.4f}")
        if v > best_val:
            best_val   = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met     = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu)
    test_macro = ts_met["overall"]["macro_f1"]
    gap        = best_val - test_macro
    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f}")

    return {
        "tag":          tag,
        "method":       "LLMSniffer-K",
        "upstream":     "arXiv 2024",
        "note":         (f"SupCon(τ={cfg.temperature})+CE(detach). "
                         "Encoder lr=1e-6 (SupCon only), head lr=1e-4 (CE only). "
                         "cls.detach() faithful to notebook line 408."),
        "enc":          cfg.enc,
        "bench":        cfg.benchmark,
        "frac":         cfg.frac,
        "epochs":       cfg.epochs,
        "lr_enc":       cfg.lr_enc,
        "lr_head":      cfg.lr_head,
        "temperature":  cfg.temperature,
        "val_macro":    best_val,
        "macro":        test_macro,
        "weighted":     ts_met["overall"]["weighted_f1"],
        "acc":          ts_met["overall"]["accuracy"],
        "val_test_gap": gap,
        "dpaper":       test_macro - PAPER_BASELINE,
        "test_metrics": ts_met,
        "val_history":  val_hist,
        "timestamp":    time.strftime("%Y-%m-%d %H:%M:%S"),
    }


# =============================================================================
# Entry point
# =============================================================================

def main():
    encoders   = ["unixcoder-base"]
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs      = [0.01, 0.05, 0.20]
    results    = []

    for enc in encoders:
        for bench, task, n_cls in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac, n_cls=n_cls)
                tag = f"ext_llmsniffer_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ext_llmsniffer_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 130)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'lr_enc':>8} {'lr_head':>8} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} {'Wall':>8}")
    print("-" * 130)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['lr_enc']:>8.0e} {r['lr_head']:>8.0e} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['wall']:>8.0f}s")
    print("=" * 130)


if __name__ == "__main__":
    main()
