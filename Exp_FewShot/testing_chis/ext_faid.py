# ext_faid — Faithful K-class reproduction of FAID (EACL 2026)
# =============================================================================
# NAME       : FAID-K  (K-class author attribution adaptation)
# UPSTREAM   : "FAID: Fine-grained AI-generated Text Detection using Multi-task
#              Auxiliary and Multi-level Contrastive Learning", EACL 2026
#              https://github.com/mbzuai-nlp/FAID
# FAITHFULNESS:
#   Architecture (algorithm/src/simclr.py, SimCLR_Classifier, one_loss=False, AA=True):
#     - Encoder: ZurichNLP/unsup-simcse-xlm-roberta-base (we swap to UniXcoder)
#     - ClassificationHead: Linear(H,H//4) → tanh → Linear(H//4,H//16) → tanh
#                           → Linear(H//16, K)  [same as DeTeCtive head]
#     - Multi-level contrastive loss (SimCLR_Classifier._compute_logits):
#         5 contrastive objectives:
#           L_human:     same-label SupCon on human-written samples
#           L_mixed:     same-"mixed" (human-AI hybrid) SupCon on hybrid samples
#           L_mixed_set: same-mixed AND same-model SupCon on hybrid samples
#           L_set:       same-model SupCon on AI-generated samples
#           L_label:     same-author SupCon on AI-generated samples
#         Combined (from simclr.py line 263-270):
#           loss = a*L_set + (4b-a)*L_label + b*L_human + b*L_mixed + 2b*L_mixed_set + c*L_ce
#         Defaults (train_classifier.py line 408-410): a=2, b=1, c=1
#     - Temperature: 0.07
#     - CE head on top (L_ce = F.cross_entropy(head(q), author_label))
#   ADAPTATION for CoDET-M4 / AICD-T2 (all AI-generated, no "human" or "mixed"):
#     - In our data ALL samples are AI-generated → is_human=False, is_mixed=False
#     - Thus L_human=0, L_mixed=0, L_mixed_set=0 (their masks return empty tensors)
#     - Effective loss: a*L_set + (4b-a)*L_label + c*L_ce = 2*L_set + 2*L_label + 1*L_ce
#       where L_set = L_label = same-author SupCon (same-model=same-author in our case)
#     - This collapses to: (a + 4b - a)*L_label + c*L_ce = 4b*L_label + c*L_ce
#                        = 4 * SupCon_author + 1 * CE
#     - We implement this faithfully: run the exact _compute_logits logic; empty
#       tensors give 0 loss automatically.
#   WHAT CHANGES vs original:
#     - Encoder: xlm-roberta → unixcoder-base (protocol parity)
#     - Task: fine-grained 3-class (human/AI/mixed) → K-class author attribution (AA=True)
#     - Dataset: FAIDSet → CoDET-M4 / AICD-T2
#     - Inference: classifier head only (no kNN FAISS, same deviation as ext_detective)
#     - Schedule: 100ep cosine LR → RAS linear warmup+decay
# =============================================================================
from __future__ import annotations
import os, sys, time, json, random, subprocess, importlib.util
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
for _p in ("numpy", "torch", "datasets", "transformers", "scikit-learn", "tqdm"):
    _ensure(_p)

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset as TD, DataLoader
from sklearn.metrics import f1_score

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"
PAPER_F1      = 0.6633

import logging
logging.basicConfig(format="%(asctime)s %(levelname)s %(message)s",
                    datefmt="%H:%M:%S", level=logging.INFO)
logger = logging.getLogger("ext_faid")

@dataclass
class Cfg:
    benchmark:   str   = "codet_m4"
    task:        str   = "author_iid"
    frac:        float = 0.20
    n_cls:       int   = 6
    seed:        int   = 42
    seq:         int   = 512
    bs:          int   = 64
    epochs:      int   = 6
    lr_enc:      float = 3e-5
    warmup:      float = 0.10
    # FAID-specific (from train_classifier.py defaults)
    temperature: float = 0.07   # --temperature default
    a:           float = 2.0    # --a default (L_set weight)
    b:           float = 1.0    # --b default
    c:           float = 1.0    # --c default (CE weight)
    device:      str   = "cuda"

    def __post_init__(self):
        if self.frac <= 0.02:
            self.epochs, self.lr_enc, self.warmup = 10, 3e-5, 0.20
        elif self.frac <= 0.10:
            self.epochs, self.lr_enc, self.warmup = 6, 3e-5, 0.15
        else:
            self.epochs, self.lr_enc, self.warmup = 6, 4e-5, 0.10
        try:
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if mem >= 40: self.bs = 128
            elif mem >= 20: self.bs = 64
            else: self.bs = 32
        except: pass

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

# ── Tokenisation: UniXcoder CLS <encoder_only> SEP format ────────────────────
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

# ── Model: FAID ClassificationHead + multi-level SupCon (simclr.py faithful) ─
class ClassificationHead(nn.Module):
    """Exact copy from algorithm/src/simclr.py lines 8-31."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.dense1   = nn.Linear(in_dim, in_dim // 4)
        self.dense2   = nn.Linear(in_dim // 4, in_dim // 16)
        self.out_proj = nn.Linear(in_dim // 16, out_dim)
        for layer in (self.dense1, self.dense2, self.out_proj):
            nn.init.xavier_uniform_(layer.weight)
            nn.init.normal_(layer.bias, std=1e-6)

    def forward(self, x):
        return self.out_proj(torch.tanh(self.dense2(torch.tanh(self.dense1(x)))))


def _compute_logits_faid(q, k, q_label, k_label, temperature, eps=1e-6):
    """
    Faithful to SimCLR_Classifier._compute_logits (simclr.py lines 156-224).
    For our data: ALL samples are AI-generated (label != 1 in FAID sense → is_machine=True).
    In FAID: label=0 → AI, label=1 → human, index1=0→human / 1→mixed.
    We map: all our samples → is_machine=True, is_human=False, is_mixed=False.
    So: L_human=0, L_mixed=0, L_mixed_set=0. Only L_set and L_label are active.
    In our K-class setting: same_mixed=same_label (no "mixed" samples), same_set=same_label.
    """
    q_norm = F.normalize(q, dim=-1)
    k_norm = F.normalize(k, dim=-1)
    sim    = (q_norm @ k_norm.T) / temperature   # (N, M)

    q_lbl  = q_label.view(-1, 1)
    k_lbl  = k_label.view(1, -1)
    same_label = (q_lbl == k_lbl)

    # L_set: same-model SupCon on AI samples (= same_label for us since no "set" distinction)
    pos_set = (sim * same_label).sum(1) / same_label.float().sum(1).clamp(min=eps)
    neg_set = sim * (~same_label).float()
    logits_set = torch.cat([pos_set.unsqueeze(1), neg_set], dim=1)   # (N, M+1)

    # L_label: same-author SupCon on AI samples (= same logits_set in our case)
    logits_label = logits_set.clone()

    # L_human, L_mixed, L_mixed_set: empty for all-AI data → loss=0
    return logits_set, logits_label


class FAIDK(nn.Module):
    """
    Faithful adaptation of FAID SimCLR_Classifier (AA=True, one_loss=False).
    Loss = a*L_set + (4b-a)*L_label + b*L_human + b*L_mixed + 2b*L_mixed_set + c*L_ce
    For all-AI data: = a*L_set + (4b-a)*L_label + c*L_ce
    With defaults a=2, b=1, c=1: = 2*L_set + 2*L_label + 1*L_ce
    """
    def __init__(self, encoder, hidden, n_cls, pad_id, temperature, a, b, c):
        super().__init__()
        self.encoder     = encoder
        self.pad_id      = pad_id
        self.temperature = temperature
        self.a = a; self.b = b; self.c = c
        self.head = ClassificationHead(hidden, n_cls)
        self.eps  = 1e-6

    def _encode(self, input_ids):
        mask = input_ids.ne(self.pad_id)
        attn = mask.unsqueeze(1) * mask.unsqueeze(2)
        out  = self.encoder(input_ids, attention_mask=attn, output_hidden_states=True)
        tok  = out[0]
        return (tok * mask.unsqueeze(-1)).sum(1) / mask.sum(-1).unsqueeze(-1).clamp(min=1)

    def forward(self, input_ids, labels):
        q = self._encode(input_ids)
        k = q.detach()

        logits_set, logits_label = _compute_logits_faid(
            q, k, labels, labels, self.temperature, self.eps)

        gt = torch.zeros(q.size(0), dtype=torch.long, device=q.device)
        loss_set   = F.cross_entropy(logits_set,   gt)
        loss_label = F.cross_entropy(logits_label, gt)
        # L_human = L_mixed = L_mixed_set = 0 (all AI, empty tensors)

        out = self.head(q)
        loss_ce = F.cross_entropy(out, labels)

        # Full FAID loss (simclr.py line 263-270) with empty terms zeroed:
        loss = self.a * loss_set + (4*self.b - self.a) * loss_label + self.c * loss_ce
        return loss, out

# ── Train / evaluate ──────────────────────────────────────────────────────────
def evaluate(model, loader, device):
    model.eval()
    preds, labs = [], []
    with torch.no_grad():
        for ids, y in loader:
            ids, y = ids.to(device), y.to(device)
            _, logits = model(ids, y)
            preds.extend(logits.argmax(-1).cpu().tolist())
            labs.extend(y.cpu().tolist())
    return f1_score(labs, preds, average="macro", zero_division=0)

def train_one(cfg, splits):
    from transformers import (RobertaConfig, RobertaModel, RobertaTokenizer,
                               get_linear_schedule_with_warmup)
    from torch.cuda.amp import GradScaler, autocast

    model_path = os.path.join(KAGGLE_MODELS, "unixcoder-base")
    tokenizer  = RobertaTokenizer.from_pretrained(model_path, local_files_only=True)
    config     = RobertaConfig.from_pretrained(model_path, local_files_only=True)
    encoder    = RobertaModel.from_pretrained(model_path, local_files_only=True)
    pad_id     = tokenizer.pad_token_id

    model = FAIDK(encoder, config.hidden_size, cfg.n_cls, pad_id,
                  cfg.temperature, cfg.a, cfg.b, cfg.c).to(cfg.device)

    tr_ds = CodeDataset(splits["train"], tokenizer, cfg.seq)
    vl_ds = CodeDataset(splits["val"],   tokenizer, cfg.seq)
    ts_ds = CodeDataset(splits["test"],  tokenizer, cfg.seq)
    tr_dl = DataLoader(tr_ds, batch_size=cfg.bs,    shuffle=True,  num_workers=2, pin_memory=True)
    vl_dl = DataLoader(vl_ds, batch_size=cfg.bs*2,  shuffle=False, num_workers=2)
    ts_dl = DataLoader(ts_ds, batch_size=cfg.bs*2,  shuffle=False, num_workers=2)

    total_steps  = cfg.epochs * len(tr_dl)
    warmup_steps = int(total_steps * cfg.warmup)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_enc, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler    = GradScaler()

    best_val, best_test = 0.0, 0.0
    t0 = time.time()
    for ep in range(cfg.epochs):
        model.train()
        for ids, labs in tr_dl:
            ids, labs = ids.to(cfg.device), labs.to(cfg.device)
            optimizer.zero_grad()
            with autocast(dtype=torch.bfloat16):
                loss, _ = model(ids, labs)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer); scaler.update(); scheduler.step()

        val_f1 = evaluate(model, vl_dl, cfg.device)
        if val_f1 >= best_val:
            best_val  = val_f1
            best_test = evaluate(model, ts_dl, cfg.device)
        logger.info(f"[ep{ep+1}] val={val_f1:.4f} best_test={best_test:.4f}")

    return {
        "tag":          f"ext_faid_unixcoder-base_{cfg.benchmark}_f{cfg.frac}",
        "method":       "FAID-K",
        "upstream":     "EACL 2026 (mbzuai-nlp)",
        "note":         ("Multi-level SupCon: L = 2*L_set + 2*L_label + 1*L_ce. "
                         "L_human/L_mixed/L_mixed_set=0 (all-AI data). "
                         "Classifier head replaces kNN for protocol parity."),
        "enc":          "unixcoder-base",
        "bench":        cfg.benchmark,
        "frac":         cfg.frac,
        "epochs":       cfg.epochs,
        "lr_enc":       cfg.lr_enc,
        "a":            cfg.a, "b": cfg.b, "c": cfg.c,
        "val_macro":    best_val,
        "macro":        best_test,
        "val_test_gap": best_val - best_test,
        "dpaper":       best_test - PAPER_F1,
        "wall":         round(time.time() - t0, 1),
    }

def main():
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
            rec = train_one(cfg, splits)
            results.append(rec)
            logger.info(f"  val={rec['val_macro']:.4f}  test={rec['macro']:.4f}  "
                        f"gap={rec['val_test_gap']:+.4f}  Δpaper={rec['dpaper']:+.4f}")

    out_path = "results/ext_faid_results.json"
    os.makedirs("results", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nSaved → {out_path}")
    print(f"\n{'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9}")
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} "
              f"{r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f}")

if __name__ == "__main__":
    main()
