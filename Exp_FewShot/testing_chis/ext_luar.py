# ext_luar — Faithful reproduction of LUAR (ICLR 2024)
# =============================================================================
# UPSTREAM    : "LUAR: Learning Universal Authorship Representations" — ICLR 2024
#               (from fewshot_iclr2024/ in external_baselines/luar/)
#
# FAITHFULNESS: Two modes:
#   Mode A — FROZEN encoder → prototype cosine nearest-neighbor classification.
#             No gradient update. Mean-pool over non-padding tokens, normalize.
#             Prototype = per-class mean of normalized train embeddings.
#             Classification = cosine-NN: argmax(query @ prototypes.T).
#
#   Mode B — N-shot fine-tune (5 epochs, lr=2e-5) with CE loss, then prototype-NN.
#             Same prototype-NN classification after fine-tuning.
#             Faithful to: adaptation_lr=2e-5, num_few_shot_epochs=5.
#
# Encoder:    RobertaModel (UniXcoder for protocol parity), mean-pool.
# Reports:    BOTH modes separately; picks best by val Macro-F1.
#
# STRUCTURAL DIFFERENCES from other ext_*.py files:
#   - No softmax classifier head (prototype-NN only)
#   - No adaptive_schedule (Mode B training is fixed)
#   - val_history = [] for Mode A; FT CE loss per epoch for Mode B
#   - eval_pack is replaced by eval_proto_metrics (precomputed embeddings)
#   - main() table shows both NN-Test and FT-Test columns
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math, copy
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
from transformers import AutoTokenizer, RobertaModel
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("ext_luar")

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
    benchmark: str   = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac:      float = 0.20;        n_cls: int = 6;       seed: int = 42
    bs:        int   = 64;          seq:   int = 512
    ft_epochs: int   = 5;           ft_lr: float = 2e-5   # faithful: LUAR adaptation params
    device:    str   = "cuda"
    gene_adj:  dict  = field(default_factory=dict)


def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40:   cfg.bs = 128
        elif mem >= 20: cfg.bs = 64
        else:           cfg.bs = 32
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
# LUAR-specific functions: encode_all, prototype_nn_predict
# =============================================================================

def encode_all(encoder, loader, device, pad_id):
    """Encode all samples; return (normalized_embeddings, labels_np, langs_list, sources_list)."""
    encoder.eval()
    embs, labs, langs, sources = [], [], [], []
    with torch.no_grad():
        for b in tqdm(loader, desc="Encode"):
            ids  = b["input_ids"].to(device)
            mask = ids.ne(pad_id)
            out  = encoder(ids, attention_mask=mask, output_hidden_states=True)
            tok  = out[0]
            # Mean-pool over non-padding tokens
            vec  = (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1).clamp(min=1)
            embs.append(F.normalize(vec, dim=-1).cpu())
            l = b["label"]
            labs.extend(l.tolist() if torch.is_tensor(l) else list(l))
            lang_batch = b.get("language", [""] * len(l))
            src_batch  = b.get("source",   [""] * len(l))
            langs.extend(list(lang_batch)   if not isinstance(lang_batch, list) else lang_batch)
            sources.extend(list(src_batch)  if not isinstance(src_batch,  list) else src_batch)
    return torch.cat(embs, 0), np.array(labs), langs, sources


def prototype_nn_predict(support_emb, support_lab, query_emb, n_cls):
    """Cosine nearest-neighbor classification via per-class prototype means."""
    protos = []
    for c in range(n_cls):
        mask = (support_lab == c)
        if mask.sum() > 0:
            protos.append(support_emb[mask].mean(0))
        else:
            protos.append(torch.zeros(support_emb.size(1)))
    protos = F.normalize(torch.stack(protos), dim=-1)
    sim    = query_emb @ protos.T
    return sim.argmax(-1).numpy()


# =============================================================================
# Evaluation — prototype-specific metrics with full sibling tracking
# =============================================================================

def eval_proto_metrics(preds, labels, n_cls, sib_mask_np, dist_mat_cpu,
                       langs=None, sources=None):
    """Compute full eval metrics on pre-computed prototype-NN predictions."""
    preds  = np.array(preds)
    labels = np.array(labels)

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
    if langs is not None and any(l for l in langs):
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
    if sources is not None and any(s for s in sources):
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
# Experiment runner
# =============================================================================

def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg)
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

    loader_cfg    = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl         = DataLoader(tr_ds, shuffle=False, **loader_cfg)
    vl_dl         = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl         = DataLoader(ts_ds, shuffle=False, **loader_cfg)
    ft_dl         = DataLoader(tr_ds, shuffle=True,
                               batch_size=min(cfg.bs, max(1, len(tr_ds))),
                               num_workers=2, pin_memory=True)

    dist_mat_cpu = build_distance_matrix(cfg.n_cls, cfg.gene_adj).numpy()
    sib_mask_np  = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()

    encoder = RobertaModel.from_pretrained(
        os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True
    ).to(cfg.device)

    # -------------------------------------------------------------------------
    # Mode A: Frozen prototype-NN (no training)
    # -------------------------------------------------------------------------
    logger.info("[Mode A] Frozen prototype cosine-NN")
    tr_emb, tr_lab, _,        _           = encode_all(encoder, tr_dl, cfg.device, pad_id)
    vl_emb, vl_lab, vl_langs, vl_sources  = encode_all(encoder, vl_dl, cfg.device, pad_id)
    ts_emb, ts_lab, ts_langs, ts_sources  = encode_all(encoder, ts_dl, cfg.device, pad_id)

    vl_preds_nn = prototype_nn_predict(tr_emb, tr_lab, vl_emb, cfg.n_cls)
    ts_preds_nn = prototype_nn_predict(tr_emb, tr_lab, ts_emb, cfg.n_cls)

    nn_val_met  = eval_proto_metrics(vl_preds_nn, vl_lab, cfg.n_cls, sib_mask_np, dist_mat_cpu,
                                     langs=vl_langs, sources=vl_sources)
    nn_test_met = eval_proto_metrics(ts_preds_nn, ts_lab, cfg.n_cls, sib_mask_np, dist_mat_cpu,
                                     langs=ts_langs, sources=ts_sources)

    nn_val_f1  = nn_val_met["overall"]["macro_f1"]
    nn_test_f1 = nn_test_met["overall"]["macro_f1"]
    logger.info(f"[Mode A] val={nn_val_f1:.4f} test={nn_test_f1:.4f}")

    # -------------------------------------------------------------------------
    # Mode B: N-shot fine-tune (CE loss) then prototype-NN
    # -------------------------------------------------------------------------
    logger.info(f"[Mode B] N-shot fine-tune ep={cfg.ft_epochs} lr={cfg.ft_lr}")
    encoder_ft = copy.deepcopy(encoder)
    clf_head   = nn.Linear(encoder_ft.config.hidden_size, cfg.n_cls).to(cfg.device)
    opt_ft     = torch.optim.AdamW(
        list(encoder_ft.parameters()) + list(clf_head.parameters()),
        lr=cfg.ft_lr,
    )

    ft_loss_hist = []
    for ep in range(cfg.ft_epochs):
        encoder_ft.train(); clf_head.train(); tot = 0.0
        for b in tqdm(ft_dl, desc=f"FT ep{ep + 1}"):
            ids  = b["input_ids"].to(cfg.device)
            labs = b["label"]
            if not torch.is_tensor(labs): labs = torch.tensor(labs, dtype=torch.long)
            labs = labs.to(cfg.device)
            mask = ids.ne(pad_id)
            out  = encoder_ft(ids, attention_mask=mask, output_hidden_states=True)
            tok  = out[0]
            vec  = (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1).clamp(min=1)
            logits = clf_head(vec)
            loss   = F.cross_entropy(logits, labs)
            opt_ft.zero_grad(); loss.backward(); opt_ft.step()
            tot += loss.item()
        ep_loss = tot / max(1, len(ft_dl))
        ft_loss_hist.append(ep_loss)
        logger.info(f"[Mode B ep{ep + 1}] loss={ep_loss:.4f}")

    # Re-encode with fine-tuned encoder
    tr_emb_ft, _, _, _ = encode_all(encoder_ft, tr_dl, cfg.device, pad_id)
    vl_emb_ft, _, _, _ = encode_all(encoder_ft, vl_dl, cfg.device, pad_id)
    ts_emb_ft, _, _, _ = encode_all(encoder_ft, ts_dl, cfg.device, pad_id)

    vl_preds_ft = prototype_nn_predict(tr_emb_ft, tr_lab, vl_emb_ft, cfg.n_cls)
    ts_preds_ft = prototype_nn_predict(tr_emb_ft, tr_lab, ts_emb_ft, cfg.n_cls)

    ft_val_met  = eval_proto_metrics(vl_preds_ft, vl_lab, cfg.n_cls, sib_mask_np, dist_mat_cpu,
                                     langs=vl_langs, sources=vl_sources)
    ft_test_met = eval_proto_metrics(ts_preds_ft, ts_lab, cfg.n_cls, sib_mask_np, dist_mat_cpu,
                                     langs=ts_langs, sources=ts_sources)

    ft_val_f1  = ft_val_met["overall"]["macro_f1"]
    ft_test_f1 = ft_test_met["overall"]["macro_f1"]
    logger.info(f"[Mode B] val={ft_val_f1:.4f} test={ft_test_f1:.4f}")

    # -------------------------------------------------------------------------
    # Pick best mode by val Macro-F1
    # -------------------------------------------------------------------------
    if ft_val_f1 > nn_val_f1:
        best_mode    = "B"
        best_val_f1  = ft_val_f1
        best_test_f1 = ft_test_f1
        best_met     = ft_test_met
        val_history  = ft_loss_hist  # FT CE loss per epoch
    else:
        best_mode    = "A"
        best_val_f1  = nn_val_f1
        best_test_f1 = nn_test_f1
        best_met     = nn_test_met
        val_history  = []  # Mode A has no training loop

    gap = best_val_f1 - best_test_f1
    logger.info(f"[final] best_mode={best_mode} val={best_val_f1:.4f} "
                f"test={best_test_f1:.4f} gap={gap:+.4f}")

    return {
        "tag":          tag,
        "method":       f"LUAR-{best_mode}",
        "upstream":     "ICLR 2024",
        "note":         (f"Mode A=frozen prototype cosine-NN. "
                         f"Mode B=N-shot FT (ep={cfg.ft_epochs}, lr={cfg.ft_lr}) + prototype-NN. "
                         f"Best by val F1 = Mode {best_mode}."),
        "enc":          cfg.enc,
        "bench":        cfg.benchmark,
        "frac":         cfg.frac,
        "ft_epochs":    cfg.ft_epochs,
        "ft_lr":        cfg.ft_lr,
        "nn_val":       float(nn_val_f1),
        "nn_test":      float(nn_test_f1),
        "ft_val":       float(ft_val_f1),
        "ft_test":      float(ft_test_f1),
        "best_mode":    best_mode,
        "val_macro":    float(best_val_f1),
        "macro":        float(best_test_f1),
        "weighted":     float(best_met["overall"]["weighted_f1"]),
        "acc":          float(best_met["overall"]["accuracy"]),
        "val_test_gap": float(gap),
        "dpaper":       float(best_test_f1 - PAPER_BASELINE),
        "test_metrics": best_met,
        "val_history":  val_history,
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
                tag = f"ext_luar_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] NN={res['nn_test']:.4f} FT={res['ft_test']:.4f} "
                                f"best({res['best_mode']})={res['macro']:.4f} "
                                f"({res['dpaper']:+.4f}) gap={res['val_test_gap']:+.4f} "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "ext_luar_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 140)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Mode':>6} "
          f"{'NN-Val':>8} {'NN-Test':>8} {'FT-Val':>8} {'FT-Test':>8} "
          f"{'Best':>8} {'dPaper':>9} {'Wall':>8}")
    print("-" * 140)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['best_mode']:>6} "
              f"{r['nn_val']:>8.4f} {r['nn_test']:>8.4f} "
              f"{r['ft_val']:>8.4f} {r['ft_test']:>8.4f} "
              f"{r['macro']:>8.4f} {r['dpaper']:>+9.4f} {r['wall']:>8.0f}s")
    print("=" * 140)


if __name__ == "__main__":
    main()
