# =============================================================================
# Theory-Track exp -- GIBA (Genealogy InfoNCE Bottleneck for Attribution):
#
# ARXIV_ID      : NEW. Combines InfoNCE-MI lower bound (van den Oord 2018,
#                 Poole 2019) with variational IB (Tishby 1999, Alemi 2017)
#                 and adds a genealogy-aware *family* MI objective.
# NAME          : GIBA (Genealogy InfoNCE Bottleneck for Attribution)
# ONE-LINE CLAIM: A tight InfoNCE lower bound on I(z; family(y)) maximised
#                 against an explicit upper-bound surrogate on I(z; source),
#                 with a variational KL bottleneck on the encoder posterior,
#                 yields a representation that retains genealogy-relevant
#                 information and discards source bias.
# EQUATION      : L_giba = CE
#                          - mu  * I_NCE(z; family(y))
#                          + nu  * I_CLUB(z; source)
#                          + beta * KL(q(z|x) || N(0, I))
#                 where I_NCE(z; c) = E[log( exp(f(z, c+)) / sum_{c'} exp(f(z, c')) )]
#                 with f(z, c) = z^T W e_c / tau (learned critic),
#                 and  I_CLUB(z; s) = E[log p(s|z)] - E_{z, s indep.}[log p(s|z)]
#                 is the CLUB upper bound on MI (Cheng 2020).
# PROPERTY      : (a) I_NCE bound is tight to log(K) (Poole 2019 Proposition 2);
#                 (b) Family supervision uses HIER_FAM mapping (CoDET 6->4 families,
#                     AICD 12->4 families) so the bound is over a coarsened tree;
#                 (c) CLUB upper bound is correct gradient direction for *minimising*
#                     I(z; source), unlike the InfoNCE lower bound.
# WHY NOT BEFORE: Information-bottleneck attribution methods use Gaussian-KL
#                 only; they do not estimate I(z; family) directly. GIBA is
#                 the first to plug a tree-coarsened InfoNCE bound and a CLUB
#                 source-suppression bound into the same IB objective.
# FALSIFIER     : If I_NCE(z; family) saturates without classification gain,
#                 family-MI is not the right axis to maximise.
# =============================================================================
from __future__ import annotations

# === KAGGLE PATHS ===
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Tuple

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
logger = logging.getLogger("exp47_giba")

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
# GIBA: Variational latent + InfoNCE critic + CLUB upper bound
# =============================================================================

# Family mapping: HIER_FAM[class_idx] = coarsened family id.
HIER_FAM_CODET = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 3}        # 6 classes -> 4 families
HIER_FAM_AICD = {i: i // 3 for i in range(12)}                # 12 classes -> 4 families


class GaussianEncoder(nn.Module):
    """Variational encoder q(z|x) = N(mu(x), sigma^2(x))."""
    def __init__(self, in_dim: int, latent_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.GELU(),
            nn.Linear(256, latent_dim * 2),
        )

    def forward(self, x, sample: bool = True):
        h = self.net(x)
        mu, logvar = h.chunk(2, dim=-1)
        logvar = logvar.clamp(min=-8.0, max=8.0)
        if sample:
            std = (0.5 * logvar).exp()
            z = mu + torch.randn_like(std) * std
        else:
            z = mu
        # KL(q || N(0, I))
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(-1).mean()
        return z, kl, mu, logvar


class InfoNCEMICritic(nn.Module):
    """Critic for InfoNCE lower bound on I(z; c) where c is a discrete label.

    f(z, c) = z^T W e_c / tau, evaluated for all categories simultaneously.
    """
    def __init__(self, z_dim: int, n_categories: int, tau: float = 0.1):
        super().__init__()
        self.W = nn.Parameter(torch.randn(n_categories, z_dim) * 0.02)
        self.tau = tau
        self.n_categories = n_categories

    def info_nce(self, z: torch.Tensor, cats: torch.Tensor) -> torch.Tensor:
        """I_NCE(z; c) >= log(K) - L_xe(z, c) where L_xe is CE of critic logits."""
        logits = (z @ self.W.T) / self.tau                      # (B, K)
        ce_critic = F.cross_entropy(logits, cats, reduction="mean")
        return math.log(self.n_categories) - ce_critic           # tighter when ce small


class CLUBSourceUpperBound(nn.Module):
    """CLUB upper bound on I(z; s) (Cheng et al. 2020).

    Estimator:
        I_CLUB = E_{p(z, s)}[log p(s|z)] - E_{p(z) p(s)}[log p(s|z)]
    p(s|z) parametrized as a small classifier over discrete s.
    """
    def __init__(self, z_dim: int, n_sources: int):
        super().__init__()
        self.n_sources = n_sources
        self.q = nn.Sequential(
            nn.Linear(z_dim, 128), nn.GELU(),
            nn.Linear(128, n_sources),
        )

    def forward(self, z: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        log_q = F.log_softmax(self.q(z), dim=-1)                 # (B, n_sources)
        pos = log_q.gather(1, s.view(-1, 1)).squeeze(1)          # (B,)
        # Marginal: random pairing
        perm = torch.randperm(z.size(0), device=z.device)
        neg = log_q.gather(1, s[perm].view(-1, 1)).squeeze(1)    # (B,)
        return (pos - neg).mean()                                # ≥ 0 in expectation


class GIBAModel(nn.Module):
    """Genealogy InfoNCE Bottleneck for Attribution."""
    def __init__(self, enc_name: str, n_cls: int, hier_fam: Dict[int, int],
                 n_sources: int = 50, latent_dim: int = 128,
                 tau: float = 0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.n_cls = n_cls
        self.hier_fam = hier_fam
        self.n_fam = max(hier_fam.values()) + 1

        self.var_enc = GaussianEncoder(hidden, latent_dim)
        self.clf = nn.Linear(latent_dim, n_cls)

        self.family_critic = InfoNCEMICritic(latent_dim, self.n_fam, tau)
        self.source_club  = CLUBSourceUpperBound(latent_dim, n_sources)

        # Lookup class -> family on device.
        fam_map = torch.tensor([hier_fam[c] for c in range(n_cls)], dtype=torch.long)
        self.register_buffer("class_to_fam", fam_map)

    def forward(self, ids, mask, source_ids, labels=None, sample: bool = True):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z, kl, mu, _ = self.var_enc(sem, sample=sample)
        logits = self.clf(z)
        info = {"kl": kl}
        if labels is not None:
            fam = self.class_to_fam[labels]                                  # (B,)
            info["mi_fam_lb"] = self.family_critic.info_nce(z, fam)          # lower bound, MAX
            info["mi_src_ub"] = self.source_club(z, source_ids)              # upper bound, MIN
        return logits, info


def giba_loss(logits, info, labels, cfg) -> Tuple[torch.Tensor, float, float, float]:
    """L = CE - mu * I_NCE(z;fam) + nu * I_CLUB(z;s) + beta * KL."""
    ce = F.cross_entropy(logits, labels)
    mi_fam = info.get("mi_fam_lb", torch.zeros((), device=logits.device))
    mi_src = info.get("mi_src_ub", torch.zeros((), device=logits.device))
    kl = info["kl"]
    total = ce - cfg.mu_fam * mi_fam + cfg.nu_src * mi_src + cfg.beta_kl * kl
    return total, ce.item(), float(mi_fam.item()), float(mi_src.item())


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
    mu_fam: float = 0.5      # weight on -I_NCE(z; family)
    nu_src: float = 0.2      # weight on +I_CLUB(z; source)
    beta_kl: float = 1e-3    # weight on KL(q(z|x) || N(0, I))
    tau: float = 0.1
    n_sources: int = 50
    warmup: float = 0.1
    device: str = "cuda"

    def __post_init__(self):
        self.hier_fam = HIER_FAM_AICD if self.benchmark == "aicd_t2" else HIER_FAM_CODET


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
        source_id = r.get("source_id", 0)
        return {
            "ids": ids, "mask": mask,
            "ast_feat": torch.tensor(ast_feat, dtype=torch.float32),
            "label": r["label"],
            "source_id": source_id
        }


def train_epoch(model, loader, opt, sch, scaler, cfg):
    model.train()
    total_loss, total_ce, total_mi_fam, total_mi_src = 0.0, 0.0, 0.0, 0.0

    for b in tqdm(loader, desc="Train"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        source_ids = b["source_id"].to(cfg.device)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            logits, info = model(ids, mask, source_ids, labels=labs, sample=True)
            loss, ce, mi_fam, mi_src = giba_loss(logits, info, labs, cfg)

        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        opt.zero_grad()
        sch.step()

        total_loss += loss.item()
        total_ce += ce
        total_mi_fam += mi_fam
        total_mi_src += mi_src

    n = len(loader)
    return total_loss / n, total_ce / n, total_mi_fam / n, total_mi_src / n


@torch.no_grad()
def eval_model(model, loader, cfg):
    model.eval()
    preds, labels = [], []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids"].to(cfg.device)
        mask = b["mask"].to(cfg.device)
        labs = b["label"]
        source_ids = b["source_id"].to(cfg.device)

        logits, _ = model(ids, mask, source_ids, labels=None, sample=False)
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

    model = GIBAModel(
        enc_name=cfg.enc, n_cls=cfg.n_cls, hier_fam=cfg.hier_fam,
        n_sources=cfg.n_sources, latent_dim=128, tau=cfg.tau,
    ).to(cfg.device)

    opt = torch.optim.AdamW([
        {"params": model.encoder.parameters(), "lr": cfg.lr_enc},
        {"params": model.var_enc.parameters(), "lr": cfg.lr_proj},
        {"params": model.clf.parameters(), "lr": cfg.lr_head},
        {"params": list(model.family_critic.parameters()) + list(model.source_club.parameters()),
         "lr": cfg.lr_proj},
    ], weight_decay=cfg.wd)

    total_steps = len(tr_dl) * cfg.epochs
    sch = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=[cfg.lr_enc, cfg.lr_proj, cfg.lr_head, cfg.lr_proj],
        total_steps=total_steps, pct_start=cfg.warmup
    )
    scaler = GradScaler()

    best_val, best_state = 0, None
    for epoch in range(cfg.epochs):
        loss, loss_ce, mi_fam, mi_src = train_epoch(model, tr_dl, opt, sch, scaler, cfg)
        val_met = eval_model(model, vl_dl, cfg)
        logger.info(f"[epoch {epoch+1}] val={val_met['macro']:.4f}")
        if val_met["macro"] > best_val:
            best_val = val_met["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    ts_met = eval_model(model, ts_dl, cfg)
    result = {
        "tag": tag,
        "method": "GIBA",
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
                tag = f"exp47_giba_{enc}_{bench}_f{frac}"
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

    try:
        _here = os.path.dirname(os.path.realpath(__file__))
    except NameError:
        _here = os.getcwd()
    out_dir = os.path.join(_here, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp47_giba_results.json"), "w") as f:
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
