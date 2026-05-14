# =============================================================================
# Theory-Track exp -- TKL (Tree-Kernel Learning)
#
# ROLE           : Drops the hand-coded SSL weights {1.0, 0.5, 0.1, 0.3} and
#                  lets the model LEARN the per-distance penalty.  This both
#                  tests whether the hand-coded ratios were optimal AND lets
#                  the prior shift with regime: at low n the model can damp
#                  the prior implicitly by lowering all log_w.
# NAME           : TKL  (Tree-Kernel Learning)
# ONE-LINE CLAIM : The optimal sibling/cousin/human-vs-model/distant weight
#                  ratio is a function of the data regime, not a constant.
# EQUATION       : w_d = softplus(gamma_d)  for d in {1, 2, 3, 4+}
#                  Per-sample CE re-weight (same shape as SSL):
#                    sample_w(i) = w_{ d_tree(y_i, pred_i) }
#                  L_tkl = mean_i sample_w(i) * CE_i
#                  Total: L = CE + lambda * L_tkl
#                  gamma_d initialised so softplus(gamma_d) matches SSL's
#                  hand-coded values; the model is free to move them.
# PROPERTY       : SSL is the BOUNDARY case where the gradient on log_w is
#                  zero.  TKL >= SSL on training loss by construction; the
#                  test is whether TKL >= SSL on test macro-F1.
# WHY NOT BEFORE : Existing hierarchical losses fix the structure-to-weight
#                  mapping by hand.  TKL exposes 4 scalar parameters that
#                  isolate the SHAPE of the genealogy prior from its STRENGTH.
# FALSIFIER      : If the four learned w_d collapse to a single number across
#                  all distances, the tree structure is being ignored.  If
#                  they collapse to the SSL initial values, hand-coding was
#                  already optimal.
# REPORTS        : Full eval pack + learned w_d in JSON.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
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
logger = logging.getLogger("exp63_tkl")

PAPER_BASELINE = 0.6633

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


def build_distance_bucket_matrix(n_cls, adj):
    """Returns LongTensor B[y, pred] in {0, 1, 2, 3} indexing distance buckets.

    Buckets:
      0 -> correct prediction (y == pred), penalty = 0
      1 -> sibling      (d_tree == 1)
      2 -> cousin       (d_tree == 2)
      3 -> human<->model crossing (when applicable)
      4 -> distant      (everything else)
    """
    D = build_distance_matrix(n_cls, adj).numpy()
    B = np.zeros((n_cls, n_cls), dtype=np.int64)
    for i in range(n_cls):
        for j in range(n_cls):
            if i == j:
                B[i, j] = 0
            elif D[i, j] == 1.0:
                B[i, j] = 1
            elif D[i, j] == 2.0:
                B[i, j] = 2
            elif (i == 0) != (j == 0):
                B[i, j] = 3
            else:
                B[i, j] = 4
    return torch.from_numpy(B)


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
# Model with learnable distance weights
# =============================================================================

def _inv_softplus(y):
    """Inverse softplus so that softplus(_inv_softplus(y)) == y."""
    return math.log(math.expm1(y))


class TKLModel(nn.Module):
    def __init__(self, enc_name, n_cls, ast_dim=64):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True
        )
        hidden = self.encoder.config.hidden_size
        self.ast_encoder = nn.Sequential(
            nn.Linear(64, 128), nn.GELU(), nn.Linear(128, ast_dim)
        )
        self.proj = nn.Sequential(
            nn.Linear(hidden + ast_dim, 256), nn.GELU(), nn.Dropout(0.1)
        )
        self.clf = nn.Linear(256, n_cls)

        # Learnable distance weights initialised at SSL hand-coded ratios:
        # buckets [correct, sibling, cousin, human<->model, distant]
        # SSL had [0.0, 1.0, 0.5, 0.3, 0.1].  We learn buckets 1..4.
        init_vals = torch.tensor([1.0, 0.5, 0.3, 0.1])
        gammas = torch.tensor([_inv_softplus(v.item()) for v in init_vals])
        self.gamma = nn.Parameter(gammas)  # shape (4,)

    def get_weights(self):
        # bucket 0 (correct) has zero penalty; buckets 1..4 use softplus(gamma)
        w14 = F.softplus(self.gamma)
        w_full = torch.cat([torch.zeros(1, device=w14.device), w14], dim=0)  # (5,)
        return w_full

    def forward(self, ids, mask, ast_feat):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem_emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        ast_emb = self.ast_encoder(ast_feat)
        fused = torch.cat([sem_emb, ast_emb], dim=-1)
        return self.clf(self.proj(fused)), fused


def tkl_loss(logits, labels, bucket_mat, w_full):
    """Per-sample CE re-weighted by learned w_{d_bucket(y, pred)}."""
    ce = F.cross_entropy(logits, labels, reduction="none")  # (B,)
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        buckets = bucket_mat[labels, preds]  # (B,)
    sample_w = w_full[buckets]  # (B,)  -- carries grad through w_full
    return (sample_w * ce).mean()


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
    seed: int = 42
    bs: int = 256
    seq: int = 512
    epochs: int = 3
    lr_enc: float = 2e-5
    lr_proj: float = 1e-4
    lr_head: float = 1e-4
    lr_gamma: float = 1e-2
    warmup: float = 0.1
    wd: float = 0.01
    lambda_tkl: float = 0.5
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

    def __len__(self):
        return len(self.data)

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
        }


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np):
    model.eval()
    preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device); labs = b["label"]
        logits, _ = model(ids, mask, ast_feat)
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
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
    per_lang = {}
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
    per_src = {}
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


def train_epoch(model, loader, opt, sch, scaler, cfg, bucket_mat):
    model.train()
    tot, ce_sum, tkl_sum = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device); mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device); labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda")):
            logits, _ = model(ids, mask, ast_feat)
            w_full = model.get_weights()
            loss_ce = F.cross_entropy(logits, labs)
            loss_tkl = tkl_loss(logits, labs, bucket_mat, w_full)
            loss = loss_ce + cfg.lambda_tkl * loss_tkl
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_sum += loss_ce.item(); tkl_sum += loss_tkl.item()
    n = len(loader)
    return tot / n, ce_sum / n, tkl_sum / n


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    bucket_mat = build_distance_bucket_matrix(cfg.n_cls, cfg.gene_adj).to(cfg.device)
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

    n_steps_per_ep = max(1, len(tr_ds) // cfg.bs)
    total_steps = n_steps_per_ep * cfg.epochs
    logger.info(
        f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} "
        f"warmup={cfg.warmup} lambda_tkl={cfg.lambda_tkl} total_steps={total_steps}"
    )

    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    model = TKLModel(cfg.enc, cfg.n_cls).to(cfg.device)
    other_params = [p for n, p in model.named_parameters() if n != "gamma"]
    enc_param_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in other_params if id(p) not in enc_param_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head},
        {"params": [model.gamma], "lr": cfg.lr_gamma, "weight_decay": 0.0},
    ], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()

    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss, ce, tk = train_epoch(model, tr_dl, opt, sch, scaler, cfg, bucket_mat)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        w_now = model.get_weights().detach().cpu().tolist()
        logger.info(
            f"[epoch {epoch+1}] loss={loss:.4f} ce={ce:.4f} tkl={tk:.4f} "
            f"val={v:.4f} w=[corr=0.000, sib={w_now[1]:.3f}, cous={w_now[2]:.3f}, "
            f"hxm={w_now[3]:.3f}, dist={w_now[4]:.3f}]"
        )
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np)
    test_macro = ts_met["overall"]["macro_f1"]
    val_test_gap = best_val - test_macro
    final_w = model.get_weights().detach().cpu().tolist()
    logger.info(
        f"[final] val={best_val:.4f} test={test_macro:.4f} gap={val_test_gap:+.4f} "
        f"w=[sib={final_w[1]:.3f}, cous={final_w[2]:.3f}, "
        f"hxm={final_w[3]:.3f}, dist={final_w[4]:.3f}]"
    )
    return {
        "tag": tag, "method": "TKL", "enc": cfg.enc, "bench": cfg.benchmark,
        "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
        "lambda_tkl": cfg.lambda_tkl,
        "learned_weights": {
            "sibling": final_w[1], "cousin": final_w[2],
            "human_vs_model": final_w[3], "distant": final_w[4],
        },
        "ssl_init_weights": {"sibling": 1.0, "cousin": 0.5, "human_vs_model": 0.3, "distant": 0.1},
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
                tag = f"exp63_tkl_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    lw = res["learned_weights"]
                    logger.info(
                        f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                        f"gap={res['val_test_gap']:+.4f} "
                        f"sib={lw['sibling']:.2f} cous={lw['cousin']:.2f} "
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
    with open(os.path.join(out_dir, "exp63_tkl_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 150)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'w_sib':>6} {'w_cous':>7} {'w_hxm':>6} {'w_dist':>7} {'Wall':>8}")
    print("-" * 150)
    for r in results:
        lw = r["learned_weights"]
        print(
            f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
            f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
            f"{r['dpaper']:>+9.4f} "
            f"{lw['sibling']:>6.3f} {lw['cousin']:>7.3f} "
            f"{lw['human_vs_model']:>6.3f} {lw['distant']:>7.3f} {r['wall']:>8.0f}s"
        )
    print("=" * 150)


if __name__ == "__main__":
    main()
