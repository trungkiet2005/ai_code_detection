# =============================================================================
# Theory-Track exp -- SSL-RAS (Sibling Structure Loss + Regime-Adaptive Schedule)
#
# PARENT METHOD  : SSL (exp42_ssl.py) -- current leaderboard best
#                  (20% CoDET-M4 Macro-F1 = 0.6796, +0.0163 vs UniXcoder paper).
# NAME           : SSL-RAS  (Sibling Structure Loss, Regime-Adaptive Schedule)
# ONE-LINE CLAIM : The genealogy-weighted CE remains the loss; what changes is the
#                  training schedule. We treat fraction f as the regime variable and
#                  scale (epochs, peak LR, warmup) so that effective optimisation
#                  pressure is roughly constant across f.
# WHY            : At bs=256, 3 epochs on 20% data = ~937 steps -- 5x fewer updates
#                  than the legacy bs=64 / 3ep / 100K protocol (~4,687 steps) that
#                  reached 0.7055 on the same encoder family.  SSL was undertrained.
# NEW MATH OBJECT: epochs(f) = round(C * sqrt(N_full / (f * N_full)))  bounded;
#                  lr_enc(f)  = lr_base * sqrt(bs / bs_ref)  (sqrt-scaling rule);
#                  warmup(f)  = larger when total_steps small.
# FALSIFIER      : If SSL-RAS underperforms SSL at f=0.20, the bottleneck is not
#                  the schedule -- it is the SSL loss form itself.
# REPORTS        : val_macro, test_macro, val_test_gap (per repo-rules hook).
# =============================================================================
from __future__ import annotations

# === KAGGLE PATHS ===
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from collections import defaultdict
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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp56_sslras")

PAPER_BASELINE = 0.6633

# =============================================================================
# GENEALOGY (identical to exp42)
# =============================================================================
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}


def _gene_distance(u: int, v: int, adj: Dict[int, List[int]]) -> float:
    if u == v:
        return 0.0
    queue = [(u, 0)]
    visited = {u}
    while queue:
        curr, d = queue.pop(0)
        for nb in adj.get(curr, []):
            if nb == v:
                return d + 1.0
            if nb not in visited:
                visited.add(nb)
                queue.append((nb, d + 1))
    return float("inf")


def build_sibling_weight_matrix(n_cls, adj, sibling_w=1.0, cousin_w=0.5, distant_w=0.1):
    W = torch.zeros(n_cls, n_cls)
    for i in range(n_cls):
        for j in range(n_cls):
            if i == j:
                W[i, j] = 0.0
                continue
            d = _gene_distance(i, j, adj)
            if d == 1:
                W[i, j] = sibling_w
            elif d == 2:
                W[i, j] = cousin_w
            elif (i == 0) != (j == 0):
                W[i, j] = 0.3
            else:
                W[i, j] = distant_w
    return W


# =============================================================================
# AST features (legacy-aligned 22-feature vector, padded to ast_dim)
# =============================================================================

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
# Model (identical capacity to exp42; only schedule changes)
# =============================================================================

class SSLModel(nn.Module):
    def __init__(self, enc_name: str, n_cls: int, ast_dim: int = 64):
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

    def forward(self, ids, mask, ast_feat):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem_emb = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        ast_emb = self.ast_encoder(ast_feat)
        fused = torch.cat([sem_emb, ast_emb], dim=-1)
        proj = self.proj(fused)
        logits = self.clf(proj)
        return logits, proj


# =============================================================================
# SSL Loss (kept identical to exp42 -- this experiment isolates schedule effects)
# =============================================================================

def ssl_loss(logits, labels, weight_matrix):
    device = logits.device
    ce = F.cross_entropy(logits, labels, reduction="none")
    with torch.no_grad():
        preds = logits.argmax(dim=-1)
        wm = weight_matrix.to(device)
        sample_w = wm[labels, preds]
        sample_w = sample_w * (labels != preds).float()
    return (sample_w * ce).mean()


# =============================================================================
# Config + Regime-Adaptive Schedule
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
    # placeholders -- set by adaptive_schedule()
    epochs: int = 3
    lr_enc: float = 2e-5
    lr_proj: float = 1e-4
    lr_head: float = 1e-4
    warmup: float = 0.1
    wd: float = 0.01
    lambda_ssl: float = 0.3
    sibling_w: float = 1.0
    cousin_w: float = 0.5
    distant_w: float = 0.1
    device: str = "cuda"
    gene_adj: dict = field(default_factory=dict)


def adaptive_schedule(cfg: Cfg) -> Cfg:
    """Regime-adaptive training schedule keyed on cfg.frac.

    Goal: keep the number of *gradient updates* roughly comparable across regimes,
    and roughly match the legacy bs=64 / 3ep / 100K protocol that reached ~0.705.

    Reference (legacy): bs_ref=64, ~4,687 updates at 20% equivalent.
    Current:            bs=256.

    Rule of thumb chosen here:
        f=0.01 ->  epochs=10,  lr_enc=3e-5, warmup=0.20
        f=0.05 ->  epochs=6,   lr_enc=3e-5, warmup=0.15
        f=0.20 ->  epochs=6,   lr_enc=4e-5, warmup=0.10

    Why these numbers (not heroic tuning):
      - epochs decreases with f because each epoch is N_train/bs steps, so larger
        f already buys more updates per epoch.
      - lr_enc uses sqrt-scaling from the legacy 2e-5 @ bs=64 reference:
        2e-5 * sqrt(256/64) = 4e-5 at 20%; we soften to 3e-5 at low f where
        few updates make a higher LR riskier.
      - warmup is *larger* at small f because total_steps is small; we want
        warmup to occupy enough fraction to avoid an early-stage exploding LR.
    """
    f = cfg.frac
    if f <= 0.02:
        cfg.epochs = 10
        cfg.lr_enc = 3e-5
        cfg.warmup = 0.20
    elif f <= 0.10:
        cfg.epochs = 6
        cfg.lr_enc = 3e-5
        cfg.warmup = 0.15
    else:
        cfg.epochs = 6
        cfg.lr_enc = 4e-5
        cfg.warmup = 0.10
    # head/proj LRs are not bottlenecks; keep at 1e-4
    return cfg


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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


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
            "label": r["label"],
        }


def train_epoch(model, loader, opt, sch, scaler, cfg, weight_matrix):
    model.train()
    total, n_ce, n_ssl = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        ast_feat = b["ast_feat"].to(cfg.device)
        labs = b["label"].to(cfg.device)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=(cfg.device == "cuda")):
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

        total += loss.item()
        n_ce += loss_ce.item()
        n_ssl += loss_ssl.item()

    n = len(loader)
    return total / n, n_ce / n, n_ssl / n


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
        "per_class": f1_score(labels, preds, average=None).tolist(),
    }


def run_exp(cfg: Cfg, tag: str):
    set_seed(cfg.seed)
    cfg = _hw(cfg)
    cfg = adaptive_schedule(cfg)

    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    weight_matrix = build_sibling_weight_matrix(
        cfg.n_cls, cfg.gene_adj,
        sibling_w=cfg.sibling_w, cousin_w=cfg.cousin_w, distant_w=cfg.distant_w
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
        f"warmup={cfg.warmup} steps/ep={n_steps_per_ep} total_steps={total_steps}"
    )
    logger.info(f"  Train: {len(tr_ds)} | Val: {len(vl_ds)} | Test: {len(ts_ds)}")

    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    model = SSLModel(cfg.enc, cfg.n_cls).to(cfg.device)

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr_enc},
        {"params": model.ast_encoder.parameters(), "lr": cfg.lr_proj},
        {"params": model.proj.parameters(), "lr": cfg.lr_proj},
        {"params": model.clf.parameters(), "lr": cfg.lr_head},
    ], weight_decay=cfg.wd)

    warmup_steps = max(1, int(total_steps * cfg.warmup))
    sch = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)
    scaler = GradScaler()

    best_val, best_state = 0.0, None
    val_hist = []
    for epoch in range(cfg.epochs):
        loss, loss_ce, loss_ssl = train_epoch(model, tr_dl, opt, sch, scaler, cfg, weight_matrix)
        val_met = eval_model(model, vl_dl, cfg)
        val_hist.append(val_met["macro"])
        logger.info(
            f"[epoch {epoch+1}] loss={loss:.4f} ce={loss_ce:.4f} ssl={loss_ssl:.4f} "
            f"val_macro={val_met['macro']:.4f}"
        )
        if val_met["macro"] > best_val:
            best_val = val_met["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met = eval_model(model, ts_dl, cfg)
    val_test_gap = best_val - ts_met["macro"]
    logger.info(
        f"[final] val_macro={best_val:.4f}  test_macro={ts_met['macro']:.4f}  "
        f"val_test_gap={val_test_gap:+.4f}"
    )

    return {
        "tag": tag,
        "method": "SSL-RAS",
        "parent": "SSL (exp42)",
        "enc": cfg.enc,
        "bench": cfg.benchmark,
        "frac": cfg.frac,
        "epochs": cfg.epochs,
        "lr_enc": cfg.lr_enc,
        "warmup": cfg.warmup,
        "total_steps": total_steps,
        "val_macro": best_val,
        "macro": ts_met["macro"],
        "weighted": ts_met["weighted"],
        "acc": ts_met["acc"],
        "val_test_gap": val_test_gap,
        "dpaper": ts_met["macro"] - PAPER_BASELINE,
        "per_class_f1": ts_met["per_class"],
        "val_history": val_hist,
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
                tag = f"exp56_sslras_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(
                        f"[{tag}] test_macro={res['macro']:.4f} "
                        f"({res['dpaper']:+.4f} vs paper) "
                        f"val_test_gap={res['val_test_gap']:+.4f} "
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
    with open(os.path.join(out_dir, "exp56_sslras_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 116)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} {'Wt-F1':>8} {'Wall':>8}")
    print("-" * 116)
    for r in results:
        print(
            f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
            f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
            f"{r['dpaper']:>+9.4f} {r['weighted']:>8.4f} {r['wall']:>8.0f}s"
        )
    print("=" * 116)
    if results:
        best = max(results, key=lambda x: x["macro"])
        print(f"\nBest test Macro-F1: {best['macro']:.4f} @ {best['tag']} "
              f"(val={best['val_macro']:.4f}, gap={best['val_test_gap']:+.4f})")


if __name__ == "__main__":
    main()
