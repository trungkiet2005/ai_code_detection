# =============================================================================
# Theory-Track exp -- RDE (Representation Disentanglement Effect):
#
# ARXIV_ID      : THIS IS NEW - disentangling authorship factors into independent
#                 representation subspaces
# NAME          : RDE (Representation Disentanglement Effect)
# ONE-LINE CLAIM: Authorship representation is decomposed into independent factors
#                 (syntax, semantics, style) via VAE-like disentanglement.
# EQUATION      : h(x) = [z_syntax, z_semantic, z_style]
#                 where each z_i is independent: I(z_i, z_j) = 0 for i ≠ j
# PROPERTY      : Disentangled representations are more interpretable and robust
#                 because each factor can be independently manipulated.
# WHY NOT BEFORE: Standard representations are entangled.
#                 RDE provides interpretable, controllable authorship factors.
# FALSIFIER     : If disentangled representations improve OOD robustness,
#                 then factorization is key to understanding attribution.
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("sklearn"); _ensure("tqdm")

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
logger = logging.getLogger("exp54")

PAPER_BASELINE = 0.6633

# =============================================================================
# NEW MATHEMATICAL OBJECT: Representation Disentanglement Effect (RDE)
# =============================================================================
"""
RDE decomposes authorship representation into independent factors.

h(x) = [z_syntax, z_semantic, z_style]

Key properties:
- Each factor captures a distinct aspect of authorship
- Factors are statistically independent: I(z_i, z_j) = 0 for i ≠ j
- Manipulating one factor changes only that aspect

The total correlation loss enforces:
TC(z) = KL(p(z) || Π_i p(z_i))

Minimizing TC encourages factorization of the representation.
"""

# =============================================================================
# AST Feature Extraction
# =============================================================================

def extract_ast_features(code: str, max_len: int = 128) -> List[float]:
    """Extract AST structural features without tree-sitter dependency.

    Features capture hierarchical code structure:
    - Function/class definitions
    - Control flow patterns
    - Nesting depth
    - Loop structures
    """
    import re

    features = []

    # Count patterns that indicate structure
    n_func = len(re.findall(r'\b(def|function|func|fn)\s+\w+', code))
    n_class = len(re.findall(r'\b(class|struct|interface|enum)\s+\w+', code))
    n_if = len(re.findall(r'\bif\s*[\(\{]', code))
    n_for = len(re.findall(r'\b(for|foreach)\s*[\(\{]', code))
    n_while = len(re.findall(r'\bwhile\s*[\(\{]', code))
    n_return = len(re.findall(r'\breturn\b', code))
    n_import = len(re.findall(r'\b(import|from|include|require)\b', code))
    n_comment = len(re.findall(r'(//|#|/\*|\'\'\'|""")', code))

    # Nesting depth estimation
    max_depth = 0
    depth = 0
    for c in code:
        if c in '{([':
            depth += 1
            max_depth = max(max_depth, depth)
        elif c in '})]':
            depth = max(0, depth - 1)

    # Line statistics
    lines = code.split('\n')
    avg_indent = np.mean([len(l) - len(l.lstrip()) for l in lines if l.strip()]) if lines else 0

    features = [
        n_func / 10.0,
        n_class / 5.0,
        n_if / 20.0,
        n_for / 10.0,
        n_while / 10.0,
        n_return / 20.0,
        n_import / 10.0,
        n_comment / 50.0,
        max_depth / 15.0,
        avg_indent / 10.0,
        len(code) / 10000.0,
        len(lines) / 500.0,
    ]

    # Pad to fixed length
    while len(features) < max_len:
        features.append(0.0)

    return features[:max_len]


# =============================================================================
# RDE Model with Disentanglement
# =============================================================================

class DisentangledEncoder(nn.Module):
    """VAE-style encoder with factorized latent space."""
    def __init__(self, in_dim: int, z_syntax: int = 32, z_semantic: int = 64, z_style: int = 32):
        super().__init__()
        self.z_syntax = z_syntax
        self.z_semantic = z_semantic
        self.z_style = z_style

        # Shared encoder
        self.shared = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.GELU(),
        )

        # Factorized priors
        self.syntax_head = nn.Linear(256, z_syntax * 2)  # mu, logvar
        self.semantic_head = nn.Linear(256, z_semantic * 2)
        self.style_head = nn.Linear(256, z_style * 2)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.shared(x)

        # Syntax factor
        syntax_out = self.syntax_head(h)
        mu_s, lv_s = syntax_out[:, :self.z_syntax], syntax_out[:, self.z_syntax:]
        z_syntax = self.reparameterize(mu_s, lv_s)

        # Semantic factor
        sem_out = self.semantic_head(h)
        mu_sem, lv_sem = sem_out[:, :self.z_semantic], sem_out[:, self.z_semantic:]
        z_semantic = self.reparameterize(mu_sem, lv_sem)

        # Style factor
        style_out = self.style_head(h)
        mu_st, lv_st = style_out[:, :self.z_style], style_out[:, self.z_style:]
        z_style = self.reparameterize(mu_st, lv_st)

        # Total correlation (simplified: sum of KLs)
        kl_syntax = -0.5 * (1 + lv_s - mu_s.pow(2) - lv_s.exp()).sum(-1).mean()
        kl_semantic = -0.5 * (1 + lv_sem - mu_sem.pow(2) - lv_sem.exp()).sum(-1).mean()
        kl_style = -0.5 * (1 + lv_st - mu_st.pow(2) - lv_st.exp()).sum(-1).mean()

        kl_total = kl_syntax + kl_semantic + kl_style

        return z_syntax, z_semantic, z_style, kl_total

    def extract_mean(self, x):
        """Extract mean without sampling (for inference)."""
        h = self.shared(x)

        syntax_out = self.syntax_head(h)
        z_syntax = syntax_out[:, :self.z_syntax]

        sem_out = self.semantic_head(h)
        z_semantic = sem_out[:, :self.z_semantic]

        style_out = self.style_head(h)
        z_style = style_out[:, :self.z_style]

        return z_syntax, z_semantic, z_style


class RDEModel(nn.Module):
    """Representation Disentanglement Effect model."""
    def __init__(self, enc_name: str, n_cls: int, z_syntax: int = 32,
                 z_semantic: int = 64, z_style: int = 32):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size

        # Disentangled encoders
        self.ast_disentangler = DisentangledEncoder(64, z_syntax, z_semantic, z_style)

        # Semantic encoder
        self.sem_encoder = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
        )

        # Classifier on disentangled representation
        total_z = z_syntax + z_semantic + z_style
        self.proj = nn.Sequential(
            nn.Linear(total_z + 256, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.clf = nn.Linear(256, n_cls)

    def forward(self, ids, mask, ast_feat, sample=True):
        # Semantic encoding
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem_emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        sem = self.sem_encoder(sem_emb)

        # Disentangled AST factors
        if sample:
            z_syntax, z_semantic, z_style, kl = self.ast_disentangler(ast_feat)
        else:
            z_syntax, z_semantic, z_style = self.ast_disentangler.extract_mean(ast_feat)
            kl = torch.tensor(0.0, device=ast_feat.device)

        # Combine
        z = torch.cat([z_syntax, z_semantic, z_style], dim=-1)
        fused = torch.cat([z, sem], dim=-1)
        h = self.proj(fused)
        logits = self.clf(h)

        return logits, kl


# =============================================================================
# RDE Loss
# =============================================================================

def rde_loss(logits, kl, labels, lambda_rde=0.1):
    """RDE loss with disentanglement regularization."""
    ce = F.cross_entropy(logits, labels)

    # Disentanglement: minimize total correlation
    # Higher KL = better separation of factors
    return ce + lambda_rde * kl, ce.item(), kl.item()


# =============================================================================
# Config and Data Loading
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    enc: str = "unixcoder-base"
    frac: float = 0.05
    n_cls: int = 6
    seed: int = 42
    bs: int = 256
    seq: int = 512
    epochs: int = 3
    lr_enc: float = 2e-5
    lr_proj: float = 1e-4
    lr_head: float = 1e-4
    wd: float = 0.01
    lambda_rde: float = 0.1
    warmup: float = 0.1
    device: str = "cuda"


def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    return cfg


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


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
        return {"code": code, "label": label}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _conv_aicd(split):
    def row(r):
        return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1))}
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


def _load_aicd(task):
    task_map = {"t1": "T1", "t2": "T2", "t3": "T3"}
    task_name = task_map.get(task.lower(), None)
    if task_name is None:
        raise ValueError(f"[aicd] Unknown task '{task}'")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found")
    parquet_files = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not parquet_files:
        raise FileNotFoundError(f"[aicd] STRICT: No parquet files")
    ds = load_dataset("parquet", data_files=parquet_files, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0:
            return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1 / 9, seed=42)
    return s2["train"], s2["test"], s["test"]


class FSDS(TD):
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42):
        self.data = data
        self.tok = tok
        self.seq_len = seq_len
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            per_cls = {}
            for lbl in labels:
                cls_idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                n_select = max(1, int(len(cls_idx) * frac))
                per_cls[lbl] = rng.sample(cls_idx, min(n_select, len(cls_idx)))
            keep_idx = [i for idxs in per_cls.values() for i in idxs]
            self.data = self.data.select(keep_idx)
            logger.info(f"[FSDS] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        enc = self.tok(code, max_length=self.seq_len, padding="max_length",
                      truncation=True, return_tensors="pt")
        ids = enc["input_ids"].squeeze(0)
        mask = enc["attention_mask"].squeeze(0)
        ast_feat = extract_ast_features(code, 128)
        return {
            "ids": ids, "mask": mask,
            "ast_feat": torch.tensor(ast_feat, dtype=torch.float32),
            "label": r["label"]
        }


def train_epoch(model, loader, opt, sch, scaler, cfg, sample=True):
    model.train()
    total_loss, total_ce, total_kl = 0, 0, 0

    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"].to(cfg.device)

        with torch.autocast(device_type='cuda', enabled=(cfg.device == "cuda")):
            logits, kl = model(ids, mask, ast_feat, sample=sample)
            loss, ce, kl_val = rde_loss(logits, kl, labs, cfg.lambda_rde)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sch.step()

        total_loss += loss.item()
        total_ce += ce
        total_kl += kl_val

    n = len(loader)
    return total_loss / n, total_ce / n, total_kl / n


@torch.no_grad()
def eval_model(model, loader, cfg):
    model.eval()
    preds, labels = [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"]

        logits, _ = model(ids, mask, ast_feat, sample=False)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist())

    preds, labels = np.array(preds), np.array(labels)
    return {
        "acc": accuracy_score(labels, preds),
        "macro": f1_score(labels, preds, average="macro"),
        "weighted": f1_score(labels, preds, average="weighted"),
        "per_class": f1_score(labels, preds, average=None).tolist()
    }


def run_exp(cfg: Cfg, tag: str):
    set_seed(cfg.seed)
    cfg = _hw(cfg)
    logger.info(f"[exp54] RDE: {tag} | frac={cfg.frac}")

    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, "author", vocab)
        vl_data = _conv_codet(vl_raw, "author", vocab)
        ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw)
        vl_data = _conv_aicd(vl_raw)
        ts_data = _conv_aicd(ts_raw)

    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)

    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed + 1)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed + 2)

    logger.info(f"  Train: {len(tr_ds)} | Val: {len(vl_ds)} | Test: {len(ts_ds)}")

    loader_cfg = dict(batch_size=cfg.bs, num_workers=2, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    model = RDEModel(cfg.enc, cfg.n_cls).to(cfg.device)

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr_enc},
        {"params": model.ast_disentangler.parameters(), "lr": cfg.lr_proj},
        {"params": model.sem_encoder.parameters(), "lr": cfg.lr_proj},
        {"params": model.proj.parameters(), "lr": cfg.lr_proj},
        {"params": model.clf.parameters(), "lr": cfg.lr_head}
    ], weight_decay=cfg.wd)

    total_steps = len(tr_dl) * cfg.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg.lr_enc, cfg.lr_proj, cfg.lr_proj, cfg.lr_proj, cfg.lr_head],
        total_steps=total_steps, pct_start=cfg.warmup
    )
    scaler = GradScaler()

    best_val, best_state = 0, None
    for epoch in range(cfg.epochs):
        loss, loss_ce, loss_kl = train_epoch(model, tr_dl, opt, sch, scaler, cfg, sample=True)
        val_met = eval_model(model, vl_dl, cfg)
        logger.info(f"  E{epoch+1}: loss={loss:.4f} ce={loss_ce:.4f} kl={loss_kl:.4f} | val={val_met['macro']:.4f}")
        if val_met["macro"] > best_val:
            best_val = val_met["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met = eval_model(model, ts_dl, cfg)

    logger.info(f"  Test: macro={ts_met['macro']:.4f} | Δ={ts_met['macro']-PAPER_BASELINE:+.4f}")

    result = {
        "tag": tag,
        "method": "RDE",
        "enc": cfg.enc,
        "bench": cfg.benchmark,
        "frac": cfg.frac,
        "macro": ts_met["macro"],
        "weighted": ts_met["weighted"],
        "acc": ts_met["acc"],
        "dpaper": ts_met["macro"] - PAPER_BASELINE,
        "per_class_f1": ts_met["per_class"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    out_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, f"{tag}_results.json"), "w") as f:
        json.dump(result, f, indent=2)

    return result


def main():
    enc = "unixcoder-base"
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]

    for bench, task, n_cls in benchmarks:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac, n_cls=n_cls)
            tag = f"exp54_rde_{enc}_{bench}_f{frac:.2f}"
            try:
                r = run_exp(cfg, tag)
                logger.info(f"  RESULT: {tag} | macro={r['macro']:.4f} Δ={r['dpaper']:+.4f}")
            except Exception as e:
                logger.error(f"  FAILED: {tag} | {e}")


if __name__ == "__main__":
    main()
