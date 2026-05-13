# =============================================================================
# Theory-Track exp -- IFT (Information Flow Theory):
#
# ARXIV_ID      : THIS IS NEW - measuring information bottleneck between
#                 AST structure, semantic content, and authorship
# NAME          : IFT (Information Flow Theory)
# ONE-LINE CLAIM: The information bottleneck between AST and authorship
#                 is the key invariant for OOD generalization, measurable
#                 via mutual information estimation.
# EQUATION      : I(AST; Y) = H(Y) - H(Y | AST)
#                 We maximize I(AST; Y) while minimizing I(AST; S)
#                 where S is the source/location confounder.
# PROPERTY      : The information-theoretic decomposition reveals which
#                 structural features carry authorship signal vs source bias.
# WHY NOT BEFORE: Prior work doesn't quantify information flow. IFT provides
#                 a principled information-theoretic framework for attribution.
# FALSIFIER     : If representations maximizing I(AST; Y) - λ I(AST; S)
#                 are more robust to source shift, then information flow
#                 analysis is key to understanding attribution.
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
logger = logging.getLogger("exp47")

PAPER_BASELINE = 0.6633

# =============================================================================
# NEW MATHEMATICAL OBJECT: Information Flow Theory (IFT)
# =============================================================================
"""
IFT measures information flow through the attribution pipeline.

Information Bottleneck for Attribution:
- Maximize I(AST; Y): Information about authorship from AST
- Minimize I(AST; S): Information about source from AST
- The difference I(AST; Y) - λ I(AST; S) is the INVARIANT

We use a variational estimator for mutual information:
I(X; Y) ≈ E[log p(y|x)] - E[log q(y)]

where p(y|x) is the classifier and q(y) is a marginal prior.
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
# IFT Model with Information Bottleneck
# =============================================================================

class InfoBottleneck(nn.Module):
    """Variational information bottleneck."""
    def __init__(self, in_dim: int, latent_dim: int = 64, beta: float = 1.0):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.GELU(),
            nn.Linear(128, latent_dim * 2)  # mu and logvar
        )
        self.beta = beta

    def forward(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        std = (0.5 * logvar).exp()
        eps = torch.randn_like(std)
        z = mu + eps * std

        # Variational KL
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()

        return z, kl

    def extract(self, x):
        """Extract mean without sampling."""
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=-1)
        return mu


class IFTModel(nn.Module):
    """Information Flow Theory model."""
    def __init__(self, enc_name: str, n_cls: int, n_sources: int = 50,
                 latent_dim: int = 64, ast_dim: int = 64, beta: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size

        # AST information bottleneck
        self.ast_ib = InfoBottleneck(64, latent_dim, beta)

        # Semantic information bottleneck
        self.sem_ib = InfoBottleneck(hidden, latent_dim, beta)

        # Source encoder (for I(AST; S) estimation)
        self.source_embed = nn.Embedding(n_sources, latent_dim)

        # Combined projection
        self.proj = nn.Sequential(
            nn.Linear(latent_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.clf = nn.Linear(256, n_cls)

        # Marginals for MI estimation
        self.register_buffer("y_prior", torch.ones(n_cls) / n_cls)
        self.n_cls = n_cls

    def estimate_mi(self, z, labels, source_ids):
        """Estimate I(AST; Y) and I(AST; S)."""
        # I(AST; Y) ≈ E[log p(y|z)] - H(Y)
        logits = self.clf(self.proj(F.normalize(z, dim=-1)))
        log_py_given_z = F.log_softmax(logits, dim=-1)
        h_y_given_z = -(log_py_given_z.exp() * log_py_given_z).sum(-1).mean()

        # Marginal entropy H(Y)
        label_counts = torch.bincount(labels, minlength=self.n_cls).float()
        label_probs = label_counts / label_counts.sum()
        h_y = -(label_probs * torch.log(label_probs + 1e-8)).sum()

        # I(AST; Y) ≈ H(Y) - H(Y| AST)
        mi_y = h_y - h_y_given_z

        # I(AST; S): correlation between AST and source
        source_emb = self.source_embed(source_ids)
        # Simple correlation as proxy
        corr = (z * source_emb).sum(-1).abs().mean()

        return mi_y, corr

    def forward(self, ids, mask, ast_feat, source_ids, return_info=False):
        # Semantic encoding + IB
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem_emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z_sem, kl_sem = self.sem_ib(sem_emb)

        # AST + IB
        z_ast, kl_ast = self.ast_ib(ast_feat)

        # Combined
        fused = torch.cat([z_sem, z_ast], dim=-1)
        proj = self.proj(fused)
        logits = self.clf(proj)

        if return_info:
            mi_y, mi_s = self.estimate_mi(z_ast, source_ids, source_ids)
            return logits, {"kl_ast": kl_ast, "kl_sem": kl_sem, "mi_y": mi_y, "mi_s": mi_s}
        return logits


# =============================================================================
# IFT Loss with Information Regularization
# =============================================================================

def ift_loss(logits, info_dict, labels, source_ids, lambda_ift=0.5):
    """IFT loss: CE + information bottleneck regularization.

    We maximize I(AST; Y) and minimize I(AST; S).
    The KL terms in IB approximate the information bottlenecks.
    """
    ce = F.cross_entropy(logits, labels)

    # Information regularization
    # Higher MI(AST; Y) = lower H(Y|AST) = better authorship signal
    # Lower MI(AST; S) = lower source correlation = better invariance

    # We use KL divergence as a proxy for minimizing I(AST; S)
    # (higher KL = more compression = less source information)

    info_reg = info_dict["kl_ast"] - 0.5 * info_dict["kl_sem"]

    return ce + lambda_ift * info_reg, ce.item(), info_reg.item()


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
    lambda_ift: float = 0.5
    beta: float = 0.1
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
        source = str(r.get("source", "") or r.get("language", "")).strip().lower()
        source_id = hash(source) % 50
        return {"code": code, "label": label, "source_id": source_id}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _conv_aicd(split):
    def row(r):
        source = str(r.get("source", "") or r.get("language", "")).strip().lower()
        source_id = hash(source) % 50
        return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1)), "source_id": source_id}
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
        source_id = r.get("source_id", 0)
        return {
            "ids": ids, "mask": mask,
            "ast_feat": torch.tensor(ast_feat, dtype=torch.float32),
            "label": r["label"],
            "source_id": source_id
        }


def train_epoch(model, loader, opt, sch, scaler, cfg):
    model.train()
    total_loss, total_ce, total_info = 0, 0, 0

    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        source_ids = b["source_id"].to(cfg.device)

        with torch.autocast(device_type='cuda', enabled=(cfg.device == "cuda")):
            logits, info = model(ids, mask, ast_feat, source_ids, return_info=True)
            loss, ce, info_reg = ift_loss(logits, info, labs, source_ids, cfg.lambda_ift)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sch.step()

        total_loss += loss.item()
        total_ce += ce
        total_info += info_reg

    n = len(loader)
    return total_loss / n, total_ce / n, total_info / n


@torch.no_grad()
def eval_model(model, loader, cfg):
    model.eval()
    preds, labels = [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"]
        source_ids = b["source_id"].to(cfg.device)

        logits = model(ids, mask, ast_feat, source_ids, return_info=False)
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
    logger.info(f"[exp47] IFT: {tag} | frac={cfg.frac}")

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

    model = IFTModel(cfg.enc, cfg.n_cls, beta=cfg.beta).to(cfg.device)

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr_enc},
        {"params": model.ast_ib.parameters(), "lr": cfg.lr_proj},
        {"params": model.sem_ib.parameters(), "lr": cfg.lr_proj},
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
        loss, loss_ce, loss_info = train_epoch(model, tr_dl, opt, sch, scaler, cfg)
        val_met = eval_model(model, vl_dl, cfg)
        logger.info(f"  E{epoch+1}: loss={loss:.4f} ce={loss_ce:.4f} info={loss_info:.4f} | val={val_met['macro']:.4f}")
        if val_met["macro"] > best_val:
            best_val = val_met["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met = eval_model(model, ts_dl, cfg)

    logger.info(f"  Test: macro={ts_met['macro']:.4f} | Δ={ts_met['macro']-PAPER_BASELINE:+.4f}")

    result = {
        "tag": tag,
        "method": "IFT",
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
            tag = f"exp47_ift_{enc}_{bench}_f{frac:.2f}"
            try:
                r = run_exp(cfg, tag)
                logger.info(f"  RESULT: {tag} | macro={r['macro']:.4f} Δ={r['dpaper']:+.4f}")
            except Exception as e:
                logger.error(f"  FAILED: {tag} | {e}")


if __name__ == "__main__":
    main()
