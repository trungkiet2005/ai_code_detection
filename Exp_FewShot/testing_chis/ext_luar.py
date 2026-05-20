# ext_luar — Faithful reproduction of LUAR few-shot style detection (ICLR 2024)
# =============================================================================
# NAME       : LUAR-K  (K-class style-embedding + cosine-NN few-shot)
# UPSTREAM   : Rivera Soto et al., "Few-Shot Detection of Machine-Generated Text
#              using Style Representations", ICLR 2024
#              https://github.com/LLNL/LUAR/tree/main/fewshot_iclr2024
# FAITHFULNESS:
#   Inference paradigm (evaluate.py PURE_EMBEDDING_METHODS branch):
#     1. Extract style embeddings for SUPPORT set S (N shots per class)
#        using LUAR RoBERTa encoder → mean-pool episodes
#     2. Compute per-class prototype: mean of support embeddings
#        (fewshot_helper.py calculate_nn_metrics → prototype centroid)
#     3. Classify each QUERY by cosine similarity to closest prototype
#        (1-NN over class prototypes)
#     4. Report pAUC (partial AUC). We adapt to macro-F1 for parity with
#        our other experiments.
#   NO TRAINING in few-shot mode. The encoder is used frozen.
#   Metric: cosine similarity (NOT euclidean, NOT dot product)
#
#   ADAPTATION notes for code/K-class setting:
#   - Original: binary human-vs-machine, pAUC metric, text from Reddit/arXiv
#   - Ours: K-class author attribution, macro-F1 metric, code from CoDET-M4/AICD-T2
#   - Encoder: LUAR uses reddit-trained RoBERTa (CRUD checkpoint from GDrive).
#     We use UniXcoder-base (same as all our experiments) as the style encoder.
#     NOTE: UniXcoder is NOT style-trained; this is a significant deviation.
#     We denote this as "LUAR-protocol" (nearest-centroid on frozen encoder).
#     A second variant uses N-shot fine-tuning (adapts encoder to few-shot support).
#   - "Few-shot fraction" mapping to LUAR's N-shot:
#       frac=0.01 → ~N_support = small (exact N depends on class size)
#       frac=0.05, 0.20 → larger N
#     We use stratified sampling identical to our other baselines.
#
#   Two modes implemented:
#   [A] LUAR-NN:   frozen encoder + prototype cosine NN (faithful to paper)
#   [B] LUAR-FT:   like LUAR-NN but with N-shot fine-tuning on support (ablation)
#
# WHAT CHANGES vs original:
#   - Encoder: LUAR CRUD → UniXcoder-base (no LUAR weights on Kaggle)
#   - Task: binary pAUC → K-class macro-F1
#   - Dataset: M4 text → CoDET-M4 / AICD-T2 code
#   - Metric: pAUC → macro-F1 (note in JSON)
# =============================================================================
from __future__ import annotations
import os, sys, time, json, random, subprocess, importlib.util
from dataclasses import dataclass, field
from typing import List

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
for _p in ("numpy", "torch", "datasets", "transformers", "scikit-learn", "tqdm"):
    _ensure(_p)

import numpy as np
import torch, torch.nn.functional as F
from torch.utils.data import Dataset as TD, DataLoader
from sklearn.metrics import f1_score

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"
PAPER_F1      = 0.6633

import logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger("ext_luar")

@dataclass
class Cfg:
    benchmark: str  = "codet_m4"
    task:      str  = "author_iid"
    frac:      float = 0.20
    n_cls:     int  = 6
    seed:      int  = 42
    seq:       int  = 512
    bs:        int  = 64       # for embedding extraction
    # LUAR few-shot adaptation (mode B only)
    ft_epochs: int  = 5        # --num_few_shot_epochs default
    ft_lr:     float = 2e-5    # --adaptation_lr default
    device:    str  = "cuda"

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

# ── Tokenisation (CLS <encoder_only> SEP format) ─────────────────────────────
def _tokenize(code, tokenizer, max_len):
    toks = tokenizer.tokenize(" ".join(code.split()))[:max_len-4]
    toks = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + toks + [tokenizer.sep_token]
    ids  = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_len - len(ids))
    return ids[:max_len]

class CodeDataset(TD):
    def __init__(self, recs, tok, seq):
        self.recs, self.tok, self.seq = recs, tok, seq
    def __len__(self): return len(self.recs)
    def __getitem__(self, i):
        r = self.recs[i]
        ids = _tokenize(r["code"], self.tok, self.seq)
        return torch.tensor(ids, dtype=torch.long), torch.tensor(r["label"], dtype=torch.long)

# ── Embedding extraction (faithful to extract_luar_embeddings) ────────────────
@torch.no_grad()
def extract_embeddings(encoder, loader, device, pad_id):
    """
    Faithful to LUAR forward_utils.extract_luar_embeddings:
    mean-pool non-padding token embeddings → L2-normalise.
    """
    encoder.eval()
    all_emb, all_lab = [], []
    for ids, labs in loader:
        ids = ids.to(device)
        mask = ids.ne(pad_id)
        attn = mask.unsqueeze(1) * mask.unsqueeze(2)
        out  = encoder(ids, attention_mask=attn, output_hidden_states=True)
        tok  = out[0]
        emb  = (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1).clamp(min=1)
        emb  = F.normalize(emb, dim=-1)   # L2-normalise (LUAR default)
        all_emb.append(emb.cpu())
        all_lab.extend(labs.tolist())
    return torch.cat(all_emb, 0).numpy(), np.array(all_lab)

# ── Prototype cosine-NN (faithful to evaluate.py calculate_nn_metrics) ───────
def prototype_cosine_nn(
    support_emb: np.ndarray, support_lab: np.ndarray,
    query_emb:   np.ndarray, query_lab:   np.ndarray,
    n_cls: int,
) -> float:
    """
    Compute per-class centroid from support, classify query by cosine NN.
    Faithful to LUAR fewshot_helper.py calculate_nn_metrics:
      prototype[c] = mean of support embeddings where label == c
      pred = argmax_c cosine_sim(query, prototype[c])
    """
    classes = sorted(np.unique(support_lab).tolist())
    prototypes = []
    for c in classes:
        mask = support_lab == c
        proto = support_emb[mask].mean(0)
        proto = proto / (np.linalg.norm(proto) + 1e-8)
        prototypes.append(proto)
    protos = np.stack(prototypes, 0)   # (K, H)
    # cosine similarity: (N_q, K)
    sim = query_emb @ protos.T
    preds = classes[np.array([sim[i].argmax() for i in range(len(query_emb))])]
    return f1_score(query_lab, preds, average="macro", zero_division=0)

# ── Mode A: Frozen encoder + prototype-NN ─────────────────────────────────────
def run_luar_nn(cfg, splits, encoder, tokenizer, pad_id):
    """Faithful to LUAR PURE_EMBEDDING_METHODS branch (no training)."""
    tr_ds = CodeDataset(splits["train"], tokenizer, cfg.seq)
    vl_ds = CodeDataset(splits["val"],   tokenizer, cfg.seq)
    ts_ds = CodeDataset(splits["test"],  tokenizer, cfg.seq)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.bs, shuffle=False, num_workers=2)
    vl_dl = DataLoader(vl_ds, batch_size=cfg.bs, shuffle=False, num_workers=2)
    ts_dl = DataLoader(ts_ds, batch_size=cfg.bs, shuffle=False, num_workers=2)

    t0 = time.time()
    support_emb, support_lab = extract_embeddings(encoder, tr_dl, cfg.device, pad_id)
    val_emb,     val_lab     = extract_embeddings(encoder, vl_dl, cfg.device, pad_id)
    test_emb,    test_lab    = extract_embeddings(encoder, ts_dl, cfg.device, pad_id)

    val_f1  = prototype_cosine_nn(support_emb, support_lab, val_emb,  val_lab,  cfg.n_cls)
    test_f1 = prototype_cosine_nn(support_emb, support_lab, test_emb, test_lab, cfg.n_cls)
    return val_f1, test_f1, round(time.time()-t0, 1)

# ── Mode B: N-shot fine-tuning on support (LUAR's MAML-style adaptation) ─────
def run_luar_ft(cfg, splits, encoder, tokenizer, pad_id):
    """
    LUAR FAST_ADAPTATION_METHODS branch: fine-tune encoder on support set for
    cfg.ft_epochs steps with lr=cfg.ft_lr, then prototype-NN.
    Faithful to evaluate.py calculate_adaptation_metrics.
    """
    from transformers import get_linear_schedule_with_warmup
    from torch.cuda.amp import GradScaler, autocast
    import torch.nn as nn

    tr_ds = CodeDataset(splits["train"], tokenizer, cfg.seq)
    vl_ds = CodeDataset(splits["val"],   tokenizer, cfg.seq)
    ts_ds = CodeDataset(splits["test"],  tokenizer, cfg.seq)
    tr_dl = DataLoader(tr_ds, batch_size=min(cfg.bs, len(tr_ds)), shuffle=True,  num_workers=2)
    vl_dl = DataLoader(vl_ds, batch_size=cfg.bs, shuffle=False, num_workers=2)
    ts_dl = DataLoader(ts_ds, batch_size=cfg.bs, shuffle=False, num_workers=2)

    # Linear head for adaptation
    hidden = encoder.config.hidden_size
    head   = nn.Linear(hidden, cfg.n_cls).to(cfg.device)
    params = list(encoder.parameters()) + list(head.parameters())
    optim  = torch.optim.AdamW(params, lr=cfg.ft_lr, weight_decay=0.0)
    total  = cfg.ft_epochs * len(tr_dl)
    sched  = get_linear_schedule_with_warmup(optim, 0, total)
    scaler = GradScaler()

    t0 = time.time()
    encoder.train(); head.train()
    for ep in range(cfg.ft_epochs):
        for ids, labs in tr_dl:
            ids, labs = ids.to(cfg.device), labs.to(cfg.device)
            optim.zero_grad()
            with autocast(dtype=torch.bfloat16):
                mask = ids.ne(pad_id)
                attn = mask.unsqueeze(1) * mask.unsqueeze(2)
                out  = encoder(ids, attention_mask=attn, output_hidden_states=True)
                tok  = out[0]
                emb  = (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1).clamp(min=1)
                loss = F.cross_entropy(head(emb), labs)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(optim); scaler.update(); sched.step()

    # After fine-tuning: prototype-NN
    support_emb, support_lab = extract_embeddings(encoder, tr_dl, cfg.device, pad_id)
    val_emb,     val_lab     = extract_embeddings(encoder, vl_dl, cfg.device, pad_id)
    test_emb,    test_lab    = extract_embeddings(encoder, ts_dl, cfg.device, pad_id)

    val_f1  = prototype_cosine_nn(support_emb, support_lab, val_emb,  val_lab,  cfg.n_cls)
    test_f1 = prototype_cosine_nn(support_emb, support_lab, test_emb, test_lab, cfg.n_cls)
    return val_f1, test_f1, round(time.time()-t0, 1)

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    from transformers import RobertaConfig, RobertaModel, RobertaTokenizer

    model_path = os.path.join(KAGGLE_MODELS, "unixcoder-base")
    tokenizer  = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    pad_id     = tokenizer.pad_token_id

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

            # Fresh encoder for each run
            encoder = RobertaModel.from_pretrained(model_path, local_files_only=True).to(cfg.device)

            # ── Mode A: Frozen NN (faithful to LUAR paper) ─────────────
            val_a, test_a, wall_a = run_luar_nn(cfg, splits, encoder, tokenizer, pad_id)
            rec_a = {
                "tag":          f"ext_luar_nn_unixcoder-base_{bench}_f{frac}",
                "method":       "LUAR-NN",
                "mode":         "frozen_prototype_cosine_nn",
                "upstream":     "ICLR 2024 (Rivera Soto et al.)",
                "note":         "Frozen encoder, cosine prototype-NN. No training.",
                "enc":          "unixcoder-base",
                "bench":        bench,
                "frac":         frac,
                "val_macro":    val_a,
                "macro":        test_a,
                "val_test_gap": val_a - test_a,
                "dpaper":       test_a - PAPER_F1,
                "wall":         wall_a,
            }
            results.append(rec_a)
            logger.info(f"  [NN]  val={val_a:.4f} test={test_a:.4f} gap={val_a-test_a:+.4f}")

            # ── Mode B: N-shot FT + NN (MAML-style adaptation ablation) ─
            encoder = RobertaModel.from_pretrained(model_path, local_files_only=True).to(cfg.device)
            val_b, test_b, wall_b = run_luar_ft(cfg, splits, encoder, tokenizer, pad_id)
            rec_b = {
                "tag":          f"ext_luar_ft_unixcoder-base_{bench}_f{frac}",
                "method":       "LUAR-FT",
                "mode":         "fewshot_finetune_then_prototype_nn",
                "upstream":     "ICLR 2024 (Rivera Soto et al.) — MAML-adaptation variant",
                "note":         (f"N-shot CE fine-tuning ({cfg.ft_epochs}ep, lr={cfg.ft_lr}) "
                                 f"on support set, then prototype cosine NN."),
                "enc":          "unixcoder-base",
                "bench":        bench,
                "frac":         frac,
                "ft_epochs":    cfg.ft_epochs,
                "ft_lr":        cfg.ft_lr,
                "val_macro":    val_b,
                "macro":        test_b,
                "val_test_gap": val_b - test_b,
                "dpaper":       test_b - PAPER_F1,
                "wall":         wall_b,
            }
            results.append(rec_b)
            logger.info(f"  [FT]  val={val_b:.4f} test={test_b:.4f} gap={val_b-test_b:+.4f}")

    out_path = "results/ext_luar_results.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved → {out_path}")

    print(f"\n{'Method':<12} {'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9}")
    for r in results:
        print(f"{r['method']:<12} {r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f}")

if __name__ == "__main__":
    main()
