# =============================================================================
# Theory-Track exp -- SSL (Sibling Structure Loss):
#
# ARXIL_ID      : THIS IS NEW - no prior work defines a loss that specifically
#                 penalizes confusion between SIBLING generators in the genealogy tree
# NAME          : SSL (Sibling Structure Loss)
# ONE-LINE CLAIM: Sibling generators (e.g., gpt and codellama) share AST patterns
#                 but differ in subtle ways; SSL specifically penalizes sibling confusion
#                 by weighting errors proportional to genealogical proximity.
# EQUATION      : L_ssl = Σ_{i,j} w_{ij} · ℒ(f(x_i), y_j)
#                 where w_{ij} = 1 if siblings, w_{ij} = α if distant, α < 1
# PROPERTY      : Sibling pairs have higher loss weight; model is penalized more
#                 for confusing siblings than for confusing distant generators.
# WHY NOT BEFORE: All errors are treated equally in standard CE.
#                 SSL is defined by the genealogy tree structure, weighting errors
#                 based on genealogical distance - only makes sense for attribution.
# FALSIFIER     : If SSL reduces sibling confusion rate, then genealogical weighting
#                 of errors is the key to improving attribution accuracy.
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
logger = logging.getLogger("exp42")

PAPER_BASELINE = 0.6633

# =============================================================================
# NEW MATHEMATICAL OBJECT: Sibling Structure Loss (SSL)
# =============================================================================
"""
SSL is a weighted CE loss where error weights are determined by genealogical distance.

Standard CE: L = -log(p(y|x)) treats all errors equally
SSL: L_ssl = Σ_{i,j} w_{ij} · CE(f(x_i), y_j)

Weight matrix W is derived from genealogy tree:
- W[i,j] = 1.0 if i and j are siblings (d_tree = 1)
- W[i,j] = 0.7 if i and j share grandparent (d_tree = 2)
- W[i,j] = 0.3 if i and j are human vs model
- W[i,j] = 0.1 otherwise (distant generators)

KEY INSIGHT: This creates a curriculum where the model is penalized MORE for
confusing siblings than for confusing distant generators. This forces the model
to learn the subtle differences between structurally similar generators.
"""

# =============================================================================
# Genealogy Distance and Weight Matrix
# =============================================================================

GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD = {i: [(i//3)*3 + j for j in range(3) if (i//3)*3 + j != i] for i in range(12)}


def build_sibling_weight_matrix(n_cls: int, adj: Dict[int, List[int]],
                                sibling_w: float = 1.0,
                                cousin_w: float = 0.5,
                                distant_w: float = 0.1) -> torch.Tensor:
    """Build weight matrix based on genealogical distance.

    W[i,j] encodes how "serious" an error is when predicting j for true i.
    """
    W = torch.zeros(n_cls, n_cls)

    for i in range(n_cls):
        for j in range(n_cls):
            if i == j:
                W[i, j] = 0.0  # No penalty for correct prediction
            else:
                d = _gene_distance(i, j, adj)
                if d == 1:
                    W[i, j] = sibling_w  # Sibling confusion - most serious
                elif d == 2:
                    W[i, j] = cousin_w  # Cousin confusion
                elif (i == 0) != (j == 0):
                    W[i, j] = 0.3  # Human vs model
                else:
                    W[i, j] = distant_w  # Distant generators

    return W


def _gene_distance(u: int, v: int, adj: Dict[int, List[int]]) -> float:
    """Compute genealogical distance via BFS."""
    if u == v:
        return 0.0
    queue = [(u, 0)]
    visited = {u}
    while queue:
        curr, d = queue.pop(0)
        for neighbor in adj.get(curr, []):
            if neighbor == v:
                return d + 1.0
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, d + 1))
    return float('inf')


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
# Model
# =============================================================================

class SSLModel(nn.Module):
    """Sibling Structure Loss model."""
    def __init__(self, enc_name: str, n_cls: int, ast_dim: int = 64):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size

        self.ast_encoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, ast_dim)
        )

        self.proj = nn.Sequential(
            nn.Linear(hidden + ast_dim, 256),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self.clf = nn.Linear(256, n_cls)

    def forward(self, ids, mask, ast_feat):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem_emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        ast_emb = self.ast_encoder(ast_feat)
        fused = torch.cat([sem_emb, ast_emb], dim=-1)
        proj = self.proj(fused)
        logits = self.clf(proj)
        return logits, proj


# =============================================================================
# SSL Loss
# =============================================================================

def ssl_loss(logits: torch.Tensor, labels: torch.Tensor,
            weight_matrix: torch.Tensor) -> torch.Tensor:
    """Sibling Structure Loss - weighted cross entropy.

    L_ssl = Σ_{i} w_{y_true, y_pred} · CE(f(x_i), y_i)
    """
    n_cls = weight_matrix.shape[0]
    device = logits.device

    # Standard CE
    ce = F.cross_entropy(logits, labels, reduction='none')  # (B,)

    # Get weight for each sample based on true label and prediction
    preds = logits.argmax(dim=-1)  # (B,)
    weights = torch.zeros_like(ce)

    for i in range(len(labels)):
        true_label = labels[i].item()
        pred_label = preds[i].item()
        if true_label != pred_label:  # Only penalize errors
            w = weight_matrix[true_label, pred_label].item()
            weights[i] = w * ce[i]
        else:
            weights[i] = 0.0  # No penalty for correct

    return weights.mean()


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
    lambda_ssl: float = 0.3
    sibling_w: float = 1.0
    cousin_w: float = 0.5
    distant_w: float = 0.1
    warmup: float = 0.1
    device: str = "cuda"

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.gene_adj = GENE_ADJ_CODET
        else:
            self.gene_adj = GENE_ADJ_AICD


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


def train_epoch(model, loader, opt, sch, scaler, cfg, weight_matrix):
    model.train()
    total_loss, total_ce, total_ssl = 0, 0, 0

    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"].to(cfg.device)

        with torch.autocast(device_type='cuda', enabled=(cfg.device == "cuda")):
            logits, _ = model(ids, mask, ast_feat)
            loss_ce = F.cross_entropy(logits, labs)
            loss_ssl = ssl_loss(logits, labs, weight_matrix)
            loss = loss_ce + cfg.lambda_ssl * loss_ssl

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sch.step()

        total_loss += loss.item()
        total_ce += loss_ce.item()
        total_ssl += loss_ssl.item()

    n = len(loader)
    return total_loss / n, total_ce / n, total_ssl / n


@torch.no_grad()
def eval_model(model, loader, cfg):
    model.eval()
    preds, labels = [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"]

        logits, _ = model(ids, mask, ast_feat)
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
    logger.info(f"[exp42] SSL: {tag} | frac={cfg.frac}")

    # Build weight matrix
    weight_matrix = build_sibling_weight_matrix(
        cfg.n_cls, cfg.gene_adj,
        sibling_w=cfg.sibling_w,
        cousin_w=cfg.cousin_w,
        distant_w=cfg.distant_w
    ).to(cfg.device)

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

    model = SSLModel(cfg.enc, cfg.n_cls).to(cfg.device)

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
        loss, loss_ce, loss_ssl = train_epoch(model, tr_dl, opt, sch, scaler, cfg, weight_matrix)
        val_met = eval_model(model, vl_dl, cfg)
        logger.info(f"  E{epoch+1}: loss={loss:.4f} ce={loss_ce:.4f} ssl={loss_ssl:.4f} | val={val_met['macro']:.4f}")
        if val_met["macro"] > best_val:
            best_val = val_met["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met = eval_model(model, ts_dl, cfg)

    logger.info(f"  Test: macro={ts_met['macro']:.4f} | Δ={ts_met['macro']-PAPER_BASELINE:+.4f}")

    result = {
        "tag": tag,
        "method": "SSL",
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
            tag = f"exp42_ssl_{enc}_{bench}_f{frac:.2f}"
            try:
                r = run_exp(cfg, tag)
                logger.info(f"  RESULT: {tag} | macro={r['macro']:.4f} Δ={r['dpaper']:+.4f}")
            except Exception as e:
                logger.error(f"  FAILED: {tag} | {e}")


if __name__ == "__main__":
    main()
