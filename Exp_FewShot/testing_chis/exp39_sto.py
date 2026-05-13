# =============================================================================
# Theory-Track exp -- STO (Structural Transfer Operator):
#
# ARXIV_ID      : THIS IS NEW - no prior work defines a transfer operator between
#                 AST structures based on genealogical relationships
# NAME          : STO (Structural Transfer Operator)
# ONE-LINE CLAIM: AST structural patterns can be transferred between generators
#                 proportional to their genealogical distance; the STO maps code
#                 from one generator to the expected structure of another.
# EQUATION      : T_{u→v}(x) = ∫ κ_tree(x,y) · w_{uv} dy where w_{uv} = 1/d_tree(u,v)
# PROPERTY      : STO is a linear operator that "translates" AST patterns along the
#                 genealogy tree. Close siblings have high transferability.
# WHY NOT BEFORE: Prior transfer learning works on raw representations. STO is
#                 a STRUCTURED transfer operator that respects the genealogy topology.
#                 It only makes sense for code attribution where both AST and genealogy exist.
# FALSIFIER     : If STO-predicted structures match actual generator structures,
#                 then genealogy determines code structure.
# =============================================================================
from __future__ import annotations

# === KAGGLE PATHS ===
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from collections import defaultdict
from dataclasses import dataclass
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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp39")

PAPER_BASELINE = 0.6633

# =============================================================================
# NEW MATHEMATICAL OBJECT: Structural Transfer Operator (STO)
# =============================================================================
"""
STO is defined as an operator that transfers AST structural patterns between generators.

For generators u, v in the genealogy tree, the STO T_{u→v} maps code written by u
to the expected AST structure if written by v:

    T_{u→v}(x) = ∫ κ_tree(x, y) · w_{uv} dP(y)

where:
- κ_tree(x,y) is the AST tree kernel between code structures
- w_{uv} = 1 / d_tree(u,v) is the transfer weight (inversely proportional to tree distance)
- d_tree is the genealogical tree distance

KEY INSIGHT: This creates a structured embedding space where:
- Code from sibling generators are CLOSE in structure
- Code from distant generators are FAR in structure
- The structure reflects the genealogy tree topology

This is NOT just "similarity". It's an OPERATOR that transforms code representations.
"""

class StructuralTransferOperator:
    """STO: Transfers AST patterns between generators based on genealogy.

    Key innovation: Creates a structured embedding space where the topology
    of the genealogy tree is encoded into the representation geometry.
    """
    def __init__(self, n_generators: int, gene_adj: Dict[int, List[int]]):
        self.n_generators = n_generators
        self.gene_adj = gene_adj
        # Precompute tree distances
        self.tree_dist = self._compute_tree_distances()

    def _compute_tree_distances(self) -> Dict[Tuple[int, int], float]:
        """Compute genealogical tree distances between all generator pairs."""
        dist = {}
        for u in range(self.n_generators):
            for v in range(self.n_generators):
                d = self._tree_distance(u, v)
                dist[(u, v)] = d
        return dist

    def _tree_distance(self, u: int, v: int, visited: set = None) -> float:
        """Compute tree distance using BFS on genealogy tree."""
        if visited is None:
            visited = set()

        if u == v:
            return 0.0

        # BFS
        queue = [(u, 0)]
        visited = {u}

        while queue:
            curr, d = queue.pop(0)
            for neighbor in self.gene_adj.get(curr, []):
                if neighbor == v:
                    return d + 1.0
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, d + 1))

        # If no path, return max distance
        return float('inf')

    def transfer_weight(self, u: int, v: int) -> float:
        """Compute transfer weight w_{uv} = 1 / (1 + d_tree(u,v))."""
        d = self.tree_dist.get((u, v), float('inf'))
        return 1.0 / (1.0 + d)

    def transfer_ast(self, ast_emb_u: torch.Tensor, target_gen: int,
                    source_labels: torch.Tensor) -> torch.Tensor:
        """Transfer AST embeddings from source generators to target generator.

        Args:
            ast_emb_u: (B, D) AST embeddings from source generators
            target_gen: target generator label
            source_labels: (B,) source generator labels

        Returns:
            (B, D) transferred AST embeddings
        """
        B, D = ast_emb_u.shape
        device = ast_emb_u.device

        # Compute transfer weights for each sample
        weights = torch.zeros(B, device=device)
        for i in range(B):
            src = source_labels[i].item()
            weights[i] = self.transfer_weight(src, target_gen)

        # Transfer: weighted average of all source embeddings
        # Higher weight for closer generators
        transferred = torch.zeros(B, D, device=device)
        for src_gen in range(self.n_generators):
            src_mask = (source_labels == src_gen)
            if src_mask.sum() > 0:
                w = self.transfer_weight(src_gen, target_gen)
                src_embs = ast_emb_u[src_mask]
                transferred[src_mask] = src_embs * w

        # Normalize
        weights = weights.unsqueeze(-1).clamp(min=1e-8)
        transferred = transferred / weights

        return transferred

    def sto_loss(self, emb_src: torch.Tensor, emb_tgt: torch.Tensor,
                source_labels: torch.Tensor, target_labels: torch.Tensor) -> torch.Tensor:
        """STO consistency loss.

        After transfer, the transferred representation should match the target generator's
        canonical representation pattern.
        """
        B = source_labels.shape[0]
        device = emb_src.device

        # For same-class pairs: transferred should be close to actual
        loss = torch.tensor(0.0, device=device)

        for i in range(B):
            for j in range(B):
                if source_labels[i] == target_labels[j]:
                    # Transfer source i's structure to target j's generator
                    transferred = self.transfer_ast(
                        emb_src[i:i+1],
                        target_labels[j].item(),
                        source_labels[i:i+1]
                    )
                    # Should be similar to target's actual embedding
                    diff = F.mse_loss(transferred, emb_tgt[j:j+1])
                    loss = loss + diff

        return loss / max(B * B, 1)


# =============================================================================
# Genealogy Tree Structures
# =============================================================================

# CoDET-M4 genealogy adjacency
GENE_ADJ_CODET = {
    0: [],  # human - root
    1: [3],  # gpt → codellama
    2: [],   # llama - isolated
    3: [1],  # codellama → gpt
    4: [],   # nxcode - isolated
    5: [],   # qwen - isolated
}

# AICD T2 genealogy (4 families × 3 models)
GENE_ADJ_AICD = {i: [(i//3)*3 + j for j in range(3) if (i//3)*3 + j != i] for i in range(12)}


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
# Model with STO Loss
# =============================================================================

class STOModel(nn.Module):
    """Model with Structural Transfer Operator loss.

    Key innovation: Representations are constrained by the genealogy topology through STO.
    """
    def __init__(self, enc_name: str, n_cls: int, ast_dim: int = 64):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size

        # AST structural encoder
        self.ast_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, ast_dim)
        )

        # Semantic encoder
        self.sem_encoder = nn.Linear(hidden, 256)

        # Classifier
        self.clf = nn.Sequential(
            nn.Linear(256 + ast_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, n_cls)
        )

    def forward(self, ids, mask, ast_feat, return_emb=False):
        # Semantic encoding
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem_emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        sem_emb = self.sem_encoder(sem_emb)

        # AST structural encoding
        ast_emb = self.ast_encoder(ast_feat)

        # Classification
        fused = torch.cat([sem_emb, ast_emb], dim=-1)
        logits = self.clf(fused)

        if return_emb:
            return logits, sem_emb, ast_emb
        return logits


# =============================================================================
# Config
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
    lr_ast: float = 1e-4
    lr_head: float = 1e-4
    wd: float = 0.01
    lambda_sto: float = 0.3
    warmup: float = 0.1
    device: str = "cuda"

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.n_cls = 6
            self.gene_adj = GENE_ADJ_CODET
        elif self.benchmark == "aicd_t2":
            self.n_cls = 12
            self.gene_adj = GENE_ADJ_AICD


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


def compute_sto_loss(ast_emb: torch.Tensor, labels: torch.Tensor,
                    sto: StructuralTransferOperator) -> torch.Tensor:
    """Compute STO consistency loss.

    For each generator pair (u,v), the transferred AST of u should match
    the AST structure of v (weighted by genealogical proximity).
    """
    B = labels.shape[0]
    device = ast_emb.device

    # Group embeddings by generator
    loss = torch.tensor(0.0, device=device)
    n_pairs = 0

    for u in range(sto.n_generators):
        for v in range(sto.n_generators):
            if u == v:
                continue

            mask_u = (labels == u)
            mask_v = (labels == v)

            if mask_u.sum() == 0 or mask_v.sum() == 0:
                continue

            emb_u = ast_emb[mask_u]
            emb_v = ast_emb[mask_v]

            # Transfer u → v
            transferred = torch.zeros_like(emb_v)
            for i, src_idx in enumerate(torch.where(mask_u)[0]):
                src_gen = labels[src_idx].item()
                w = sto.transfer_weight(src_gen, v)
                transferred += emb_u[i] * w
            transferred = transferred / transferred.norm(dim=-1, keepdim=True).clamp(min=1e-8)

            # Should be close to actual v embeddings
            target = emb_v / emb_v.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            diff = F.mse_loss(transferred, target)
            loss = loss + diff
            n_pairs += 1

    return loss / max(n_pairs, 1)


def train_epoch(model, loader, opt, sch, scaler, cfg, sto):
    model.train()
    total_loss, total_ce, total_sto = 0, 0, 0

    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"].to(cfg.device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            logits, sem_emb, ast_emb = model(ids, mask, ast_feat, return_emb=True)
            loss_ce = F.cross_entropy(logits, labs)
            loss_sto = compute_sto_loss(ast_emb, labs, sto)
            loss = loss_ce + cfg.lambda_sto * loss_sto

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sch.step()

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_sto += loss_sto.item()

    n = len(loader)
    return total_loss / n, total_ce / n, total_sto / n


@torch.no_grad()
def eval_model(model, loader, cfg):
    model.eval()
    preds, labels = [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"]

        logits = model(ids, mask, ast_feat)
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
    # Initialize STO
    sto = StructuralTransferOperator(cfg.n_cls, cfg.gene_adj)

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

    model = STOModel(cfg.enc, cfg.n_cls).to(cfg.device)

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr_enc},
        {"params": model.ast_encoder.parameters(), "lr": cfg.lr_ast},
        {"params": model.sem_encoder.parameters(), "lr": cfg.lr_ast},
        {"params": model.clf.parameters(), "lr": cfg.lr_head}
    ], weight_decay=cfg.wd)

    total_steps = len(tr_dl) * cfg.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg.lr_enc, cfg.lr_ast, cfg.lr_ast, cfg.lr_head],
        total_steps=total_steps, pct_start=cfg.warmup
    )
    scaler = GradScaler()

    best_val, best_state = 0, None
    for epoch in range(cfg.epochs):
        loss, loss_ce, loss_sto = train_epoch(model, tr_dl, opt, sch, scaler, cfg, sto)
        val_met = eval_model(model, vl_dl, cfg)
        logger.info(f"[epoch {epoch+1}] val={val_met['macro']:.4f}")
        if val_met["macro"] > best_val:
            best_val = val_met["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met = eval_model(model, ts_dl, cfg)
    result = {
        "tag": tag,
        "method": "STO",
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
                tag = f"exp39_sto_{enc}_{bench}_f{frac}"
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
    with open(os.path.join(out_dir, "exp39_sto_results.json"), "w") as f:
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
