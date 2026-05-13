# =============================================================================
# Theory-Track exp -- GRA (Genealogical Residual Analysis):
#
# ARXIV_ID      : THIS IS NEW - no prior work defines residual analysis where
#                 the residual is defined as "what AST structure remains after
#                 accounting for genealogical influence"
# NAME          : GRA (Genealogical Residual Analysis)
# ONE-LINE CLAIM: The residual AST pattern after removing genealogical influence
#                 captures author-specific style that is independent of generator family.
# EQUATION      : R(x) = AST(x) - E[AST | gene(x)] = AST(x) - μ_{gene(x)}
#                 where μ_{gene} is the mean AST pattern for a generator family
# PROPERTY      : The residual is "pure style" - structure attributable to individual
#                 author habits, not family-level patterns. This is what should be
#                 used for attribution.
# WHY NOT BEFORE: Prior work uses raw AST features.
#                 GRA introduces a causal decomposition: AST = genealogical_effect + residual.
#                 The residual is a NEW mathematical object that isolates individual style.
# FALSIFIER     : If residual-based features improve attribution over raw AST,
#                 then genealogical influence can be isolated and individual style exists.
# =============================================================================
from __future__ import annotations

# === KAGGLE PATHS ===
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

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
logger = logging.getLogger("exp45")

PAPER_BASELINE = 0.6633

# =============================================================================
# NEW MATHEMATICAL OBJECT: Genealogical Residual (GRA)
# =============================================================================
"""
GRA defines a causal decomposition of AST features:

    AST(x) = genealogical_effect(x) + residual(x)

where:
- genealogical_effect(x) = E[AST | gene(x)] = mean AST for generator family
- residual(x) = AST(x) - genealogical_effect(x) = "pure style"

KEY INSIGHT: The residual captures what remains after removing family-level patterns.
This is what distinguishes individual authors within a generator family.
The residual is a NEW mathematical object that only makes sense when you have
BOTH AST structure AND genealogical structure.

Training with residual features:
1. Compute family means from training data
2. Subtract family mean from each sample's AST features
3. Train on residuals instead of raw AST

The residual representation should:
- Be invariant to family-level patterns
- Capture individual author style
- Improve attribution by focusing on discriminative residuals
"""

# =============================================================================
# Genealogy Structures
# =============================================================================

# CODET-M4: 6 classes, 5 families
# Family 0: class 0 (human)
# Family 1: classes 1, 3 (gpt, codellama - siblings)
# Family 2: class 2 (llama)
# Family 3: class 4 (nxcode)
# Family 4: class 5 (qwen)
GENE_TREE_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
CODET_CLASS_TO_FAMILY = {0: 0, 1: 1, 3: 1, 2: 2, 4: 3, 5: 4}
CODET_N_FAMILIES = 5

# AICD T2: 12 classes, 4 families (3 models per family)
# Family 0: classes 0, 1, 2
# Family 1: classes 3, 4, 5
# Family 2: classes 6, 7, 8
# Family 3: classes 9, 10, 11
GENE_TREE_AICD = {i: [(i//3)*3 + j for j in range(3) if (i//3)*3 + j != i] for i in range(12)}
AICD_CLASS_TO_FAMILY = {i: i // 3 for i in range(12)}
AICD_N_FAMILIES = 4


def same_family(u: int, v: int, adj: Dict) -> bool:
    """Check if two generators are in the same family."""
    if u == v:
        return True
    for parent_u in adj.get(u, []):
        if v in adj.get(parent_u, []):
            return True
    return False


def get_family(u: int, class_to_family: Dict, gene_tree: Dict) -> int:
    """Get family ID for a generator."""
    return class_to_family.get(u, u)  # Map class ID to family index


# =============================================================================
# AST Feature Extraction
# =============================================================================

def extract_ast_features(code: str, max_len: int = 128) -> List[float]:
    """Extract structural code features (legacy-aligned, offline-only).

    Mirrors legacy/Exp_DM_weak/exp06_ast_irm.py::extract_structural_features:
    richer 22-feature structural vector, normalized + padded to max_len.
    No tree-sitter dependency (offline-safe).
    """
    import re as _re

    lines = code.split("\n")
    num_lines = max(len(lines), 1)
    line_lens = [len(l) for l in lines]
    avg_line_len = float(np.mean(line_lens)) if line_lens else 0.0
    max_line_len = float(max(line_lens)) if line_lens else 0.0

    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    avg_indent = float(np.mean(indents)) if indents else 0.0
    max_indent = float(max(indents)) if indents else 0.0
    indent_var = float(np.var(indents)) if indents else 0.0

    n_func = len(_re.findall(r"\b(def|function|func|fn)\s+\w+", code))
    n_class = len(_re.findall(r"\b(class|struct|interface|enum)\s+\w+", code))
    n_for = len(_re.findall(r"\b(for|foreach)\s*[\(\{]", code))
    n_while = len(_re.findall(r"\bwhile\s*[\(\{]", code))
    n_loops = n_for + n_while
    n_if = len(_re.findall(r"\bif\s*[\(\{]", code))
    n_else = code.count("else ") + code.count("elif ")
    n_cond = n_if + n_else
    n_return = len(_re.findall(r"\breturn\b", code))
    n_comment = code.count("//") + code.count("#") + code.count("/*")
    n_import = len(_re.findall(r"\b(import|from|include|require|using)\b", code))
    n_try = code.count("try") + code.count("catch") + code.count("except")

    max_depth = 0
    depth = 0
    for c in code:
        if c in "{([":
            depth += 1
            if depth > max_depth:
                max_depth = depth
        elif c in "})]":
            depth = max(0, depth - 1)

    identifiers = _re.findall(r"\b[a-zA-Z_]\w*\b", code)
    n_ids = max(len(identifiers), 1)
    snake_ratio = sum(1 for i in identifiers if "_" in i and i.islower()) / n_ids
    camel_ratio = sum(1 for i in identifiers if any(c.isupper() for c in i[1:]) and "_" not in i) / n_ids
    short_ratio = sum(1 for i in identifiers if len(i) == 1) / n_ids
    avg_id_len = float(np.mean([len(i) for i in identifiers])) if identifiers else 0.0

    empty_ratio = sum(1 for l in lines if not l.strip()) / num_lines
    code_len = max(len(code), 1)
    alpha_ratio = sum(c.isalpha() for c in code) / code_len
    digit_ratio = sum(c.isdigit() for c in code) / code_len
    space_ratio = sum(c.isspace() for c in code) / code_len

    features = [
        num_lines / 500.0,
        avg_line_len / 80.0,
        max_line_len / 200.0,
        avg_indent / 10.0,
        max_indent / 20.0,
        indent_var / 50.0,
        n_func / 10.0,
        n_class / 5.0,
        n_loops / 10.0,
        n_cond / 20.0,
        n_return / 20.0,
        n_comment / 50.0,
        n_import / 10.0,
        n_try / 10.0,
        max_depth / 15.0,
        snake_ratio,
        camel_ratio,
        short_ratio,
        avg_id_len / 10.0,
        empty_ratio,
        alpha_ratio,
        digit_ratio,
    ]

    if len(features) < max_len:
        features = features + [0.0] * (max_len - len(features))
    return features[:max_len]


# =============================================================================
# GRA Model
# =============================================================================

class FamilyMeanTracker:
    """Track running mean of AST features per family."""
    def __init__(self, n_families: int, feat_dim: int, device: str = "cuda"):
        self.n_families = n_families
        self.feat_dim = feat_dim
        self.device = device
        self.counts = torch.zeros(n_families, device=device)
        self.means = torch.zeros(n_families, feat_dim, device=device)

    def update(self, family_ids: torch.Tensor, ast_feats: torch.Tensor):
        """Update family means with new data."""
        family_ids = family_ids.to(self.device)
        ast_feats = ast_feats.to(self.device)
        for f in range(self.n_families):
            mask = (family_ids == f)
            if mask.sum() > 0:
                feats = ast_feats[mask]
                n_new = feats.shape[0]
                old_mean = self.means[f]
                old_count = self.counts[f]
                new_mean = feats.mean(dim=0)
                self.means[f] = (old_mean * old_count + new_mean * n_new) / (old_count + n_new)
                self.counts[f] = old_count + n_new

    def get_residual(self, family_ids: torch.Tensor, ast_feats: torch.Tensor) -> torch.Tensor:
        """Compute residuals: AST - family_mean."""
        family_ids = family_ids.to(self.device)
        ast_feats = ast_feats.to(self.device)
        residuals = torch.zeros_like(ast_feats)
        for i in range(family_ids.shape[0]):
            f = family_ids[i].item()
            residuals[i] = ast_feats[i] - self.means[f]
        return residuals


class GRAModel(nn.Module):
    """Genealogical Residual Analysis model."""
    def __init__(self, enc_name: str, n_cls: int, n_families: int = 4, ast_dim: int = 64):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size

        # AST encoder for residuals
        self.ast_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, ast_dim)
        )

        # Projectors
        self.proj = nn.Sequential(
            nn.Linear(hidden + ast_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.clf = nn.Linear(256, n_cls)

    def forward(self, ids, mask, residual_feat):
        # Semantic encoding
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem_emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        # Residual AST encoding
        residual_emb = self.ast_encoder(residual_feat)

        # Project
        fused = torch.cat([sem_emb, residual_emb], dim=-1)
        proj = self.proj(fused)

        # Classify
        logits = self.clf(proj)
        return logits


# =============================================================================
# GRA Loss
# =============================================================================

def compute_gra_loss(residual_emb: torch.Tensor, labels: torch.Tensor,
                   same_family_weight: float = 0.3) -> tuple:
    """Compute GRA-specific loss.

    Residuals from same family should be different (pure style).
    Residuals from different families should also be different.
    """
    B = residual_emb.shape[0]
    device = residual_emb.device

    # Residual should encode individual style, not family
    # => Same family residuals should be distinguishable
    residual_sim = torch.cdist(residual_emb, residual_emb, p=2)

    # Positive mask: same class (excluding diagonal)
    pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
    pos_mask = pos_mask * (1 - torch.eye(B, device=device))  # Zero out diagonal

    # Negative mask: different class
    neg_mask = 1 - pos_mask - torch.eye(B, device=device)

    # Loss: residuals from same class should be distinguishable
    same_class_sim = (residual_sim * pos_mask).sum() / (pos_mask.sum() + 1e-8)
    diff_class_sim = (residual_sim * neg_mask).sum() / (neg_mask.sum() + 1e-8)

    # Residuals should have LOW within-class similarity (high variety)
    # and HIGH across-class similarity (distinctive)
    loss = same_class_sim - 0.5 * diff_class_sim

    return loss, same_class_sim.item(), diff_class_sim.item()


# =============================================================================
# Config and Data Loading
# =============================================================================

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
    lambda_gra: float = 0.3
    warmup: float = 0.1
    device: str = "cuda"

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.gene_tree = GENE_TREE_CODET
            self.class_to_family = CODET_CLASS_TO_FAMILY
            self.n_families = CODET_N_FAMILIES
        else:
            self.gene_tree = GENE_TREE_AICD
            self.class_to_family = AICD_CLASS_TO_FAMILY
            self.n_families = AICD_N_FAMILIES


def _hw(cfg: Cfg) -> Cfg:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40:
            cfg.bs, cfg.seq = 256, 512
        elif mem >= 10:
            cfg.bs, cfg.seq = 128, 384
        else:
            cfg.bs, cfg.seq = 64, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} seq={cfg.seq}")
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
    def __init__(self, data, tok, seq_len, ast_dim=64, frac=1.0, seed=42):
        self.data = data
        self.tok = tok
        self.seq_len = seq_len
        self.ast_dim = ast_dim
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
        ast_feat = extract_ast_features(code, self.ast_dim)
        return {
            "ids": ids, "mask": mask,
            "ast_feat": torch.tensor(ast_feat, dtype=torch.float32),
            "label": r["label"]
        }


def train_epoch(model, loader, opt, sch, scaler, cfg, tracker):
    model.train()
    total_loss, total_ce, total_gra = 0, 0, 0

    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"].to(cfg.device)

        # Compute family IDs
        family_ids = torch.tensor([get_family(l.item(), cfg.class_to_family, cfg.gene_tree) for l in labs], device=cfg.device)

        # Update tracker with batch data
        tracker.update(family_ids, ast_feat)

        # Get residuals
        residual = tracker.get_residual(family_ids, ast_feat)

        # Residual AST encoding
        residual_emb = model.ast_encoder(residual)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            # Semantic encoding
            out = model.encoder(input_ids=ids, attention_mask=mask)
            sem_emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

            # Project and classify
            fused = torch.cat([sem_emb, residual_emb], dim=-1)
            proj = model.proj(fused)
            logits = model.clf(proj)

            loss_ce = F.cross_entropy(logits, labs)
            loss_gra, _, _ = compute_gra_loss(residual_emb, labs)
            loss = loss_ce + cfg.lambda_gra * loss_gra

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sch.step()

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_gra += loss_gra.item()

    n = len(loader)
    return total_loss / n, total_ce / n, total_gra / n


@torch.no_grad()
def eval_model(model, loader, cfg, tracker):
    model.eval()
    preds, labels = [], []

    # Compute validation residuals using training means
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"]

        # Compute family IDs
        family_ids = torch.tensor([get_family(l.item(), cfg.class_to_family, cfg.gene_tree) for l in labs], device=cfg.device)

        # Get residuals
        residual = tracker.get_residual(family_ids, ast_feat)
        residual_emb = model.ast_encoder(residual)

        # Semantic encoding
        out = model.encoder(input_ids=ids, attention_mask=mask)
        sem_emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)

        # Project and classify
        fused = torch.cat([sem_emb, residual_emb], dim=-1)
        proj = model.proj(fused)
        logits = model.clf(proj)

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

    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    model = GRAModel(cfg.enc, cfg.n_cls, cfg.n_families).to(cfg.device)

    # Family mean tracker
    tracker = FamilyMeanTracker(cfg.n_families, 64, device=cfg.device)

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr_enc},
        {"params": model.ast_encoder.parameters(), "lr": cfg.lr_proj},
        {"params": model.proj.parameters(), "lr": cfg.lr_proj},
        {"params": model.clf.parameters(), "lr": cfg.lr_head}
    ], weight_decay=cfg.wd)

    total_steps = len(tr_dl) * cfg.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg.lr_enc, cfg.lr_proj, cfg.lr_proj, cfg.lr_head],
        total_steps=total_steps, pct_start=cfg.warmup
    )
    scaler = GradScaler()

    best_val, best_state = 0, None
    for epoch in range(cfg.epochs):
        loss, loss_ce, loss_gra = train_epoch(model, tr_dl, opt, sch, scaler, cfg, tracker)
        val_met = eval_model(model, vl_dl, cfg, tracker)
        logger.info(f"[epoch {epoch+1}] val={val_met['macro']:.4f}")
        if val_met["macro"] > best_val:
            best_val = val_met["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met = eval_model(model, ts_dl, cfg, tracker)
    result = {
        "tag": tag,
        "method": "GRA",
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
    return result


def main():
    encoders = ["unixcoder-base"]
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]

    results = []
    for enc in encoders:
        for bench, task, n_cls in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac, n_cls=n_cls)
                cfg = _hw(cfg)
                tag = f"exp45_gra_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    elapsed = time.time() - t0
                    res["wall"] = round(elapsed, 1)
                    results.append(res)
                    logger.info(f"[{tag}] MacroF1={res['macro']:.4f} ({res['macro']-PAPER_BASELINE:+.4f} vs paper) time={elapsed:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

    out_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp45_gra_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 100)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Macro-F1':>10} {'dPaper':>10} {'Weighted':>10} {'Wall':>8}")
    print("-" * 100)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['macro']:>10.4f} "
              f"{r['dpaper']:>+10.4f} {r['weighted']:>10.4f} {r['wall']:>8.0f}s")
    print("=" * 100)
    if results:
        best = max(results, key=lambda x: x["macro"])
        print(f"\nBest Macro-F1: {best['macro']:.4f} @ {best['tag']}")


if __name__ == "__main__":
    main()
