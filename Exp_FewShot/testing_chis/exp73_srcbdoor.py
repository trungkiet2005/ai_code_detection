# =============================================================================
# Theory-Track exp -- SRCBDOOR (Source-Stratified Backdoor Adjustment)
#
# EXPLOITS       : S8 (source domain cf/lc/gh is observed AND confounding).
#                  CoDET-M4 ships source labels but standard detectors marginalise
#                  them away.  Legacy Exp18 documents a catastrophic OOD-gh gap
#                  (Macro-F1 28.34 vs 79.63 on IID -- a 51-pt collapse on the
#                  GitHub split).  Source is a textbook confound for code style.
# NAME           : SRCBDOOR  (Source-Stratified Backdoor Adjustment)
# ONE-LINE CLAIM : Conditioning on source domain S and marginalising via the
#                  empirical P(S) yields a causal estimate P(Y | do(X)),
#                  decoupling author attribution from platform style.
# EQUATION       : Standard CE classifier:  P(Y | X)
#                  Backdoor adjusted:        P(Y | do(X)) = sum_s P(Y | X, S=s) * P(S=s)
#                  Implementation:           ONE shared trunk; THREE source-specific
#                                            classifier heads (cf, lc, gh).
#                                            Train: use ground-truth S to pick head.
#                                            Eval:  average head logits with weights
#                                                   = empirical P(S=s) in training.
# WHY S8-SPECIFIC: Source labels are an OBSERVED context variable for CoDET-M4
#                  ({codeforces, leetcode, github}).  Backdoor adjustment is
#                  defined only when the confounder is observed.  This object
#                  is impossible for AICD-T2 (no source field) -- the model
#                  degenerates gracefully to a single shared head there, and
#                  we log that as part of the falsifier.
# WHY HIGH-IMPACT: Cross-source gap is the largest documented source of error
#                  in legacy Exp18 (Macro-F1 79.63 IID -> 28.34 OOD-gh).
#                  Even a modest IID improvement is plausible if the backdoor
#                  adjustment regularises out platform-style confounding.
# FALSIFIER      : If per-source IID macro-F1 collapses to the marginal (i.e.,
#                  the three heads predict identically), the model has not
#                  used the source signal.  We log per_source.macro_f1 in
#                  eval_pack and the test-time average of head divergence.
# REPORTS        : Full eval pack; per_source breakdown is the key reading.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from dataclasses import dataclass, field
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
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
)
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp73_srcbdoor")

PAPER_BASELINE = 0.6633

# Source vocab for CoDET-M4
SRC_VOCAB = {"cf": 0, "lc": 1, "gh": 2}
SRC_UNK = 0  # fallback to cf if unknown

# =============================================================================
# Genealogy
# =============================================================================
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}


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
        for j in adj.get(i, []):
            M[i, j] = 1.0
    return M


def extract_ast_features(code: str, max_len: int = 64) -> List[float]:
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
    max_depth, depth = 0, 0
    for c in code:
        if c in "{([":
            depth += 1
            if depth > max_depth: max_depth = depth
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
    features = [
        num_lines / 500.0, avg_line_len / 80.0, max_line_len / 200.0,
        avg_indent / 10.0, max_indent / 20.0, indent_var / 50.0,
        n_func / 10.0, n_class / 5.0, n_loops / 10.0, n_cond / 20.0,
        n_return / 20.0, n_comment / 50.0, n_import / 10.0, n_try / 10.0,
        max_depth / 15.0, snake_ratio, camel_ratio, short_ratio,
        avg_id_len / 10.0, empty_ratio, alpha_ratio, digit_ratio,
    ]
    if len(features) < max_len:
        features = features + [0.0] * (max_len - len(features))
    return features[:max_len]


# =============================================================================
# Backdoor-adjusted model
# =============================================================================

class SrcBdoorModel(nn.Module):
    """Shared trunk + per-source classifier heads.

    Forward modes:
      train:  use the SAMPLE'S source S to route to head h_S
      eval:   average over heads weighted by empirical P(S=s) -- the backdoor
              adjustment P(Y | do(X)).
    """
    def __init__(self, enc_name, n_cls, n_sources, ast_dim=64):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True
        )
        hidden = self.encoder.config.hidden_size
        self.n_sources = n_sources
        self.ast_encoder = nn.Sequential(
            nn.Linear(64, 128), nn.GELU(), nn.Linear(128, ast_dim)
        )
        self.trunk = nn.Sequential(
            nn.Linear(hidden + ast_dim, 256), nn.GELU(), nn.Dropout(0.1)
        )
        # one classifier head per source
        self.heads = nn.ModuleList([nn.Linear(256, n_cls) for _ in range(n_sources)])
        # empirical P(S=s) -- set after data load
        self.register_buffer("source_marginal", torch.ones(n_sources) / n_sources)

    def trunk_feat(self, ids, mask, ast_feat):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        ast_emb = self.ast_encoder(ast_feat)
        return self.trunk(torch.cat([sem, ast_emb], dim=-1))

    def forward(self, ids, mask, ast_feat, source_id=None):
        feat = self.trunk_feat(ids, mask, ast_feat)
        if source_id is None:
            # Backdoor-adjusted: sum_s P(S=s) * softmax(head_s(feat))
            logits_all = torch.stack([h(feat) for h in self.heads], dim=1)  # (B, n_src, n_cls)
            probs = F.softmax(logits_all, dim=-1)
            w = self.source_marginal.view(1, -1, 1)
            mix = (w * probs).sum(dim=1).clamp_min(1e-8)                    # (B, n_cls)
            # Return log to allow argmax / KL downstream consistently
            return torch.log(mix), feat
        else:
            # Training-time: gather the head for each sample
            logits_all = torch.stack([h(feat) for h in self.heads], dim=1)  # (B, n_src, n_cls)
            idx = source_id.view(-1, 1, 1).expand(-1, 1, logits_all.size(-1))
            logits = logits_all.gather(1, idx).squeeze(1)                   # (B, n_cls)
            return logits, feat


# =============================================================================
# Config
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    enc: str = "unixcoder-base"
    frac: float = 0.20
    n_cls: int = 6
    n_sources: int = 3
    seed: int = 42
    bs: int = 256
    seq: int = 512
    epochs: int = 3
    lr_enc: float = 2e-5
    lr_proj: float = 1e-4
    lr_head: float = 1e-4
    warmup: float = 0.1
    wd: float = 0.01
    device: str = "cuda"
    gene_adj: dict = field(default_factory=dict)


def adaptive_schedule(cfg: Cfg) -> Cfg:
    f = cfg.frac
    if f <= 0.02:
        cfg.epochs, cfg.lr_enc, cfg.warmup = 10, 3e-5, 0.20
    elif f <= 0.10:
        cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 3e-5, 0.15
    else:
        cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 4e-5, 0.10
    return cfg


def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40: cfg.bs, cfg.seq = 256, 512
        elif mem >= 10: cfg.bs, cfg.seq = 128, 384
        else: cfg.bs, cfg.seq = 64, 256
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


def _src_to_id(s):
    s = str(s or "").strip().lower()
    return SRC_VOCAB.get(s, SRC_UNK)


def _conv_codet(split, task, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code", "code"):
            v = r.get(f, "")
            if isinstance(v, str) and v.strip():
                code = v; break
        if task == "binary":
            label = 0 if _is_human(r.get("target", "")) else 1
        else:
            if _is_human(r.get("target", "")):
                label = 0
            else:
                label = vocab.get(str(r.get("model", "") or "").strip(), -1)
        return {
            "code": code, "label": label,
            "language": str(r.get("language", "")).strip().lower(),
            "source": str(r.get("source", "")).strip().lower(),
        }
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _conv_aicd(split):
    def row(r):
        return {
            "code": str(r.get("code", "")).strip(),
            "label": int(r.get("label", -1)),
            "language": str(r.get("language", "")).strip().lower(),
            "source": "",
        }
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
    if task_name is None: raise ValueError(f"[aicd] Unknown task '{task}'")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path):
        raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found")
    parquet_files = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not parquet_files: raise FileNotFoundError(f"[aicd] STRICT: No parquet files")
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
        self.data = data; self.tok = tok; self.seq_len = seq_len; self.ast_dim = ast_dim
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

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        enc = self.tok(code, max_length=self.seq_len, padding="max_length",
                       truncation=True, return_tensors="pt")
        return {
            "ids": enc["input_ids"].squeeze(0),
            "mask": enc["attention_mask"].squeeze(0),
            "ast_feat": torch.tensor(extract_ast_features(code, self.ast_dim), dtype=torch.float32),
            "label": r["label"],
            "language": r.get("language", "") or "",
            "source": r.get("source", "") or "",
            "source_id": _src_to_id(r.get("source", "")),
        }


# =============================================================================
# Eval pack
# =============================================================================

@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np):
    """Eval-time predictions use backdoor-adjusted mixture (source_id=None)."""
    model.eval()
    preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device); labs = b["label"]
        log_probs, _ = model(ids, mask, ast_feat, source_id=None)
        preds.extend(log_probs.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        lang_batch = b.get("language", [""] * len(labs))
        src_batch = b.get("source", [""] * len(labs))
        langs.extend(list(lang_batch) if not isinstance(lang_batch, list) else lang_batch)
        sources.extend(list(src_batch) if not isinstance(src_batch, list) else src_batch)
    preds = np.array(preds); labels = np.array(labels)
    n_cls = cfg.n_cls
    overall = {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0)),
    }
    per_class = {
        "f1": f1_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
        "precision": precision_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
        "recall": recall_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
    }
    cm = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    off_diag_total = int(cm.sum() - cm.trace())
    sib_conf = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls)
                       if i != j and sib_mask_np[i, j] > 0))
    sib_rate = sib_conf / max(off_diag_total, 1)
    D = build_distance_matrix(n_cls, cfg.gene_adj).numpy()
    cross_fam = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls)
                        if i != j and D[i, j] >= 3.0))
    cross_rate = cross_fam / max(off_diag_total, 1)
    per_lang, per_src = {}, {}
    if any(l for l in langs):
        langs_arr = np.array(langs)
        for L in sorted(set(langs)):
            if not L: continue
            sel = (langs_arr == L)
            if sel.sum() < 2: continue
            per_lang[L] = {
                "n": int(sel.sum()),
                "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                "accuracy": float(accuracy_score(labels[sel], preds[sel])),
            }
    if any(s for s in sources):
        src_arr = np.array(sources)
        for S in sorted(set(sources)):
            if not S: continue
            sel = (src_arr == S)
            if sel.sum() < 2: continue
            per_src[S] = {
                "n": int(sel.sum()),
                "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                "accuracy": float(accuracy_score(labels[sel], preds[sel])),
            }
    return {
        "overall": overall, "per_class": per_class,
        "per_language": per_lang, "per_source": per_src,
        "confusion_matrix": cm.tolist(),
        "sibling_confusion_rate": float(sib_rate),
        "cross_family_confusion_rate": float(cross_rate),
        "off_diag_total": off_diag_total, "n_samples": int(len(labels)),
    }


def train_epoch(model, loader, opt, sch, scaler, cfg):
    model.train()
    tot = 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device); labs = b["label"].to(cfg.device)
        src_id = b["source_id"].to(cfg.device) if torch.is_tensor(b["source_id"]) \
                 else torch.tensor(b["source_id"], dtype=torch.long, device=cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda")):
            logits, _ = model(ids, mask, ast_feat, source_id=src_id)
            loss = F.cross_entropy(logits, labs)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item()
    return tot / len(loader)


def estimate_source_marginal(data, n_sources=3):
    """Empirical P(S=s) from training set."""
    counts = np.zeros(n_sources, dtype=np.float64)
    for s in data["source"]:
        counts[_src_to_id(s)] += 1.0
    if counts.sum() == 0:
        return np.ones(n_sources) / n_sources
    return counts / counts.sum()


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    sib_mask_np = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()

    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, "author", vocab)
        vl_data = _conv_codet(vl_raw, "author", vocab)
        ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)

    tok = AutoTokenizer.from_pretrained(
        os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True
    )
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed + 1)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed + 2)

    # Empirical P(S) from training set (after subsampling)
    src_marginal = estimate_source_marginal(tr_ds.data, cfg.n_sources)
    logger.info(f"[src] empirical P(S) = {src_marginal.tolist()}")

    n_steps_per_ep = max(1, len(tr_ds) // cfg.bs)
    total_steps = n_steps_per_ep * cfg.epochs
    logger.info(
        f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} "
        f"warmup={cfg.warmup} n_sources={cfg.n_sources} total_steps={total_steps}"
    )

    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    model = SrcBdoorModel(cfg.enc, cfg.n_cls, cfg.n_sources).to(cfg.device)
    model.source_marginal = torch.tensor(src_marginal, dtype=torch.float32, device=cfg.device)
    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr_enc},
        {"params": model.ast_encoder.parameters(), "lr": cfg.lr_proj},
        {"params": model.trunk.parameters(), "lr": cfg.lr_proj},
        {"params": model.heads.parameters(), "lr": cfg.lr_head},
    ], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()

    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss = train_epoch(model, tr_dl, opt, sch, scaler, cfg)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        per_src_str = " ".join(
            f"{s}={d['macro_f1']:.3f}(n={d['n']})"
            for s, d in val_met["per_source"].items()
        )
        logger.info(
            f"[epoch {epoch+1}] loss={loss:.4f} val={v:.4f} "
            f"per_src=[{per_src_str}] sib_conf={val_met['sibling_confusion_rate']:.4f}"
        )
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np)
    test_macro = ts_met["overall"]["macro_f1"]
    val_test_gap = best_val - test_macro
    logger.info(
        f"[final] val={best_val:.4f} test={test_macro:.4f} gap={val_test_gap:+.4f} "
        f"sib_conf={ts_met['sibling_confusion_rate']:.4f}"
    )
    return {
        "tag": tag, "method": "SRCBDOOR", "enc": cfg.enc, "bench": cfg.benchmark,
        "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
        "n_sources": cfg.n_sources, "source_marginal": src_marginal.tolist(),
        "val_macro": best_val, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": val_test_gap, "dpaper": test_macro - PAPER_BASELINE,
        "test_metrics": ts_met, "val_history": val_hist,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    encoders = ["unixcoder-base"]
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for enc in encoders:
        for bench, task, n_cls in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac, n_cls=n_cls)
                tag = f"exp73_srcbdoor_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(
                        f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                        f"gap={res['val_test_gap']:+.4f} "
                        f"sib_conf={res['test_metrics']['sibling_confusion_rate']:.4f} "
                        f"time={res['wall']:.0f}s"
                    )
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                import gc; gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    try:
        _here = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        _here = os.getcwd()
    out_dir = os.path.join(_here, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp73_srcbdoor_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 140)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'PSrc(cf,lc,gh)':>22} {'Wall':>8}")
    print("-" * 140)
    for r in results:
        ps = r.get("source_marginal", [0, 0, 0])
        print(
            f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
            f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
            f"{r['dpaper']:>+9.4f} "
            f"({ps[0]:.2f},{ps[1]:.2f},{ps[2]:.2f})".rjust(22)
            + f" {r['wall']:>8.0f}s"
        )
    print("=" * 140)


if __name__ == "__main__":
    main()
