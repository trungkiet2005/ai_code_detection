# exp147 — JEPACODE
# =============================================================================
# NAME       : JEPACODE (Joint-Embedding Predictive Architecture for Code Authorship)
# REFERENCE  : LeCun (2022) "A Path Towards Autonomous Machine Intelligence";
#              Assran et al. 2023 "Self-Supervised Learning from Images with
#              a Joint-Embedding Predictive Architecture" (I-JEPA,
#              arXiv:2301.08243); V-JEPA (Bardes et al. 2024,
#              arXiv:2404.08471). NEVER applied to authorship.
# CLAIM      : An author's identity is the linear operator that PREDICTS
#              the bottom half of their code from the top half in a fixed
#              feature space. Attribution = argmin per-author prediction error.
# EQUATION   : z_A = phi(top(x)), z_B = phi(bottom(x));
#              P_c = argmin_P sum_{x in train_c} ||P z_A - z_B||^2 + alpha ||P||_F^2;
#              y_hat = argmin_c ||z_B_q - P_c z_A_q||.
# WHY NEW    : JEPA in deep learning is well-established (I-JEPA, V-JEPA);
#              the predictive-in-latent-space paradigm has NEVER been
#              applied to authorship attribution, and NEVER instantiated
#              with a classical Ridge regressor per author. No prior work
#              has framed authorship as a per-author predictive operator.
# WOW HOOK   : "Authorship is predictability. The top half of your code
#              PREDICTS the bottom — and the predictor is unique to you.
#              We learn one Ridge regressor per author in latent feature
#              space; attribution is argmin prediction error. No neural net,
#              no negatives, no decoder. JEPA collapses into ridge regression."
# FALSIFIER  : (F1) Per-author Ridge regressors must differ: mean off-
#              diagonal residual >= 1.5x diagonal residual on train.
#              Report falsifier_F1_other_over_own_ratio.
#              (F2) JEPA beats nearest-centroid on [z_A | z_B] by >= 0.005
#              (the predictor matters, not just the concatenated features).
#              Report falsifier_F2_jepa_minus_nc.
#              (F3) Ridge alpha sweep over {0.1, 1.0, 10.0}: best alpha is
#              NOT the largest (otherwise the predictor collapses to a
#              constant). Report falsifier_F3_alpha_sweep_val_macros and
#              falsifier_F3_best_alpha.
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
import multiprocessing as mp
from dataclasses import dataclass, field
from typing import List, Dict, Tuple

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("datasets"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp147_jepacode")

PAPER_BASELINE = 0.6633
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD  = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}


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
    D = np.full((n_cls, n_cls), default_dist)
    for i in range(n_cls):
        for j in range(n_cls):
            d = _gene_distance(i, j, adj)
            if d < float("inf"): D[i, j] = d
            elif (i == 0) != (j == 0): D[i, j] = 3.0
    return D


def build_sibling_mask(n_cls, adj):
    M = np.zeros((n_cls, n_cls))
    for i in range(n_cls):
        for j in adj.get(i, []): M[i, j] = 1.0
    return M


# =============================================================================
# Data loading
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD  = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


def _is_human(t):
    return str(t or "").strip().lower() in {"human", "human_written", "human-generated"}


def _vocab(train):
    names = {str(r.get("model", "") or "").strip() for r in train
             if not _is_human(r.get("target", "")) and r.get("model", "")}
    return {n: i + 1 for i, n in enumerate(sorted(names))}


def _conv_codet(split, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code", "code"):
            v = r.get(f, "")
            if isinstance(v, str) and v.strip(): code = v; break
        label = 0 if _is_human(r.get("target", "")) else vocab.get(
            str(r.get("model", "") or "").strip(), -1)
        return {"code": code, "label": label,
                "language": str(r.get("language", "")).strip().lower(),
                "source":   str(r.get("source",   "")).strip().lower()}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _conv_aicd(split):
    def row(r):
        return {"code":     str(r.get("code",     "")).strip(),
                "label":    int(r.get("label",    -1)),
                "language": str(r.get("language", "")).strip().lower(),
                "source":   ""}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _load_codet():
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        return tr, vl, ts
    s  = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


def _load_aicd(task):
    task_name = {"t1": "T1", "t2": "T2", "t3": "T3"}.get(task.lower())
    if task_name is None: raise ValueError(f"[aicd] Unknown task '{task}'")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path): raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found")
    pf = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: No parquet files")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0: return tr, vl, ts
    s  = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


def stratified_subsample(data, frac_or_n_per_class, seed=42):
    """If frac_or_n_per_class < 1.0: treat as fraction per class.
    If >= 1.0: treat as max count per class."""
    rng = random.Random(seed)
    labels = list(range(max(data["label"]) + 1))
    keep = []
    for lbl in labels:
        idx = [i for i, x in enumerate(data["label"]) if x == lbl]
        if frac_or_n_per_class < 1.0:
            n = min(max(1, int(len(idx) * frac_or_n_per_class)), len(idx))
        else:
            n = min(int(frac_or_n_per_class), len(idx))
        keep.extend(rng.sample(idx, n))
    return data.select(keep)


# =============================================================================
# JEPA core
# =============================================================================

D_RAW = 512   # HashingVectorizer dim
D_LAT = 32    # PCA latent dim
MAX_CHARS = 4000


def _split_halves(code: str, max_chars: int = MAX_CHARS) -> Tuple[str, str]:
    """Split code into top half and bottom half at the line midpoint.
    Fallback: char midpoint if too few lines."""
    code = code[:max_chars]
    lines = code.split("\n")
    if len(lines) >= 4:
        mid = len(lines) // 2
        top = "\n".join(lines[:mid])
        bot = "\n".join(lines[mid:])
    else:
        m = len(code) // 2
        top = code[:m]
        bot = code[m:]
    if not top.strip(): top = code[:1]
    if not bot.strip(): bot = code[-1:] if code else " "
    return top, bot


def _build_views(codes: List[str]) -> Tuple[List[str], List[str]]:
    tops, bots = [], []
    for c in codes:
        t, b = _split_halves(c)
        tops.append(t); bots.append(b)
    return tops, bots


def _fit_pca_extractor(train_tops: List[str], train_bots: List[str], seed=42):
    """Fit HashingVectorizer (stateless) + PCA on concatenated train top+bot raw vectors."""
    hv = HashingVectorizer(
        n_features=D_RAW,
        ngram_range=(1, 3),
        analyzer="char",
        norm="l2",
        alternate_sign=False,
    )
    # Use train tops + bots together to fit PCA (so latent space sees both halves)
    raw_all = hv.transform(train_tops + train_bots)  # sparse, dim D_RAW
    pca = PCA(n_components=D_LAT, random_state=seed)
    pca.fit(raw_all.toarray())
    return hv, pca


def _phi(hv, pca, texts: List[str]) -> np.ndarray:
    """Embed list of texts to R^{D_LAT}."""
    raw = hv.transform(texts).toarray()
    z = pca.transform(raw)
    # L2 normalize
    norms = np.linalg.norm(z, axis=1, keepdims=True) + 1e-12
    z = z / norms
    return z.astype(np.float32)


def _fit_per_author_ridges(zA_train: np.ndarray, zB_train: np.ndarray,
                           labels: List[int], n_cls: int,
                           alpha: float) -> Dict[int, Ridge]:
    """Fit one Ridge regressor per author: zA -> zB."""
    regs: Dict[int, Ridge] = {}
    for c in range(n_cls):
        idx = [i for i, l in enumerate(labels) if l == c]
        if len(idx) < 2:
            regs[c] = None
            continue
        Xc = zA_train[idx]
        Yc = zB_train[idx]
        reg = Ridge(alpha=alpha, fit_intercept=True)
        reg.fit(Xc, Yc)
        regs[c] = reg
    return regs


def _jepa_predict(regs: Dict[int, Ridge], zA: np.ndarray, zB: np.ndarray,
                  n_cls: int) -> np.ndarray:
    """For each query (zA[i], zB[i]), pick c minimising ||zB[i] - P_c zA[i]||."""
    N = zA.shape[0]
    errs = np.full((N, n_cls), np.inf, dtype=np.float32)
    for c in range(n_cls):
        reg = regs.get(c)
        if reg is None: continue
        pred = reg.predict(zA)  # (N, D_LAT)
        diff = pred - zB
        errs[:, c] = np.linalg.norm(diff, axis=1)
    return np.argmin(errs, axis=1)


def _nearest_centroid_predict(zAB_train: np.ndarray, labels: List[int],
                              zAB_test: np.ndarray, n_cls: int) -> np.ndarray:
    """Nearest-centroid on [zA | zB] for falsifier F2."""
    centroids = np.zeros((n_cls, zAB_train.shape[1]), dtype=np.float32)
    have = np.zeros(n_cls, dtype=bool)
    for c in range(n_cls):
        idx = [i for i, l in enumerate(labels) if l == c]
        if not idx: continue
        centroids[c] = zAB_train[idx].mean(axis=0)
        have[c] = True
    # distances
    d = np.linalg.norm(zAB_test[:, None, :] - centroids[None, :, :], axis=2)
    d[:, ~have] = np.inf
    return np.argmin(d, axis=1)


def _residual_matrix(regs: Dict[int, Ridge], zA: np.ndarray, zB: np.ndarray,
                     labels: List[int], n_cls: int) -> np.ndarray:
    """R[c, c'] = mean residual when predicting class-c samples with regressor c'.
    Diagonal = own residual; off-diagonal = other-author residual."""
    R = np.zeros((n_cls, n_cls), dtype=np.float64)
    for c in range(n_cls):
        idx = [i for i, l in enumerate(labels) if l == c]
        if not idx:
            R[c, :] = np.nan; continue
        za = zA[idx]; zb = zB[idx]
        for cp in range(n_cls):
            reg = regs.get(cp)
            if reg is None:
                R[c, cp] = np.nan; continue
            pred = reg.predict(za)
            R[c, cp] = float(np.mean(np.linalg.norm(pred - zb, axis=1)))
    return R


# =============================================================================
# Cfg
# =============================================================================

@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    frac: float = 0.20
    n_cls: int = 6
    seed: int = 42
    train_cap_per_class: int = 300
    test_cap_per_class:  int = 400
    val_cap_per_class:   int = 200
    alpha: float = 1.0
    alpha_sweep: tuple = (0.1, 1.0, 10.0)
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = MAX_CHARS


def set_seed(s):
    random.seed(s); np.random.seed(s)


# =============================================================================
# Eval pack
# =============================================================================

def eval_pack(preds, labels, langs, sources, n_cls, sib_mask, dist_mat):
    preds = np.asarray(preds); labels = np.asarray(labels)
    langs = np.asarray(langs); sources = np.asarray(sources)
    overall = {
        "accuracy":        float(accuracy_score(labels, preds)),
        "macro_f1":        float(f1_score(labels, preds, average="macro",    zero_division=0)),
        "weighted_f1":     float(f1_score(labels, preds, average="weighted", zero_division=0)),
        "micro_f1":        float(f1_score(labels, preds, average="micro",    zero_division=0)),
        "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
        "macro_recall":    float(recall_score(labels, preds, average="macro",    zero_division=0)),
    }
    per_class = {
        "f1":        f1_score(labels, preds, average=None, zero_division=0,
                              labels=list(range(n_cls))).tolist(),
        "precision": precision_score(labels, preds, average=None, zero_division=0,
                                     labels=list(range(n_cls))).tolist(),
        "recall":    recall_score(labels, preds, average=None, zero_division=0,
                                  labels=list(range(n_cls))).tolist(),
    }
    cm       = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    off_diag = int(cm.sum() - cm.trace())
    sib_conf = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls)
                       if i != j and sib_mask[i, j] > 0))
    sib_rate = sib_conf / max(off_diag, 1)
    cross = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls)
                    if i != j and dist_mat[i, j] >= 3.0))
    cross_rate = cross / max(off_diag, 1)
    per_lang, per_src = {}, {}
    if langs.size > 0 and any(l for l in langs.tolist()):
        for L in sorted(set(langs.tolist())):
            if not L: continue
            sel = (langs == L)
            if sel.sum() < 2: continue
            per_lang[L] = {
                "n":           int(sel.sum()),
                "macro_f1":    float(f1_score(labels[sel], preds[sel], average="macro",    zero_division=0)),
                "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                "accuracy":    float(accuracy_score(labels[sel], preds[sel])),
            }
    if sources.size > 0 and any(s for s in sources.tolist()):
        for S in sorted(set(sources.tolist())):
            if not S: continue
            sel = (sources == S)
            if sel.sum() < 2: continue
            per_src[S] = {
                "n":           int(sel.sum()),
                "macro_f1":    float(f1_score(labels[sel], preds[sel], average="macro",    zero_division=0)),
                "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                "accuracy":    float(accuracy_score(labels[sel], preds[sel])),
            }
    return {
        "overall":                     overall,
        "per_class":                   per_class,
        "per_language":                per_lang,
        "per_source":                  per_src,
        "confusion_matrix":            cm.tolist(),
        "sibling_confusion_rate":      float(sib_rate),
        "cross_family_confusion_rate": float(cross_rate),
        "off_diag_total":              off_diag,
        "n_samples":                   int(len(labels)),
    }


# =============================================================================
# run_exp
# =============================================================================

def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD

    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, vocab)
        vl_data = _conv_codet(vl_raw, vocab)
        ts_data = _conv_codet(ts_raw, vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)

    cfg.n_cls = max(tr_data["label"]) + 1
    dist_mat = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    sib_mask = build_sibling_mask(cfg.n_cls, cfg.gene_adj)

    tr_data_frac   = stratified_subsample(tr_data, cfg.frac,                       seed=cfg.seed)
    tr_data_capped = stratified_subsample(tr_data_frac, cfg.train_cap_per_class,   seed=cfg.seed + 1)
    ts_data_capped = stratified_subsample(ts_data, cfg.test_cap_per_class,         seed=cfg.seed + 2)
    vl_data_capped = stratified_subsample(vl_data, cfg.val_cap_per_class,          seed=cfg.seed + 3)

    train_codes  = [r["code"][:cfg.max_chars] for r in tr_data_capped]
    train_labels = list(tr_data_capped["label"])
    val_codes    = [r["code"][:cfg.max_chars] for r in vl_data_capped]
    val_labels   = list(vl_data_capped["label"])
    val_langs    = [r.get("language", "") or "" for r in vl_data_capped]
    val_sources  = [r.get("source", "") or ""   for r in vl_data_capped]
    test_codes   = [r["code"][:cfg.max_chars] for r in ts_data_capped]
    test_labels  = list(ts_data_capped["label"])
    test_langs   = [r.get("language", "") or "" for r in ts_data_capped]
    test_sources = [r.get("source", "") or ""   for r in ts_data_capped]

    logger.info(f"[setup] frac={cfg.frac} train={len(train_codes)} "
                f"val={len(val_codes)} test={len(test_codes)} n_cls={cfg.n_cls}")

    # Build views
    t0 = time.time()
    tr_tops, tr_bots = _build_views(train_codes)
    vl_tops, vl_bots = _build_views(val_codes)
    ts_tops, ts_bots = _build_views(test_codes)

    # Fit feature extractor on TRAIN ONLY
    logger.info("[jepa] fitting HashingVectorizer + PCA on train...")
    hv, pca = _fit_pca_extractor(tr_tops, tr_bots, seed=cfg.seed)

    # Embed
    zA_tr = _phi(hv, pca, tr_tops); zB_tr = _phi(hv, pca, tr_bots)
    zA_vl = _phi(hv, pca, vl_tops); zB_vl = _phi(hv, pca, vl_bots)
    zA_ts = _phi(hv, pca, ts_tops); zB_ts = _phi(hv, pca, ts_bots)
    logger.info(f"[jepa] phi done: zA_tr={zA_tr.shape} feat_time={time.time()-t0:.0f}s")

    # ---- F3 alpha sweep on VAL ----
    sweep_macros = {}
    for a in cfg.alpha_sweep:
        regs_a = _fit_per_author_ridges(zA_tr, zB_tr, train_labels, cfg.n_cls, alpha=a)
        preds_v = _jepa_predict(regs_a, zA_vl, zB_vl, cfg.n_cls)
        m = float(f1_score(val_labels, preds_v, average="macro", zero_division=0))
        sweep_macros[float(a)] = m
        logger.info(f"[jepa][sweep] alpha={a:>6.2f} val_macro={m:.4f}")
    best_alpha = max(sweep_macros, key=sweep_macros.get)
    sweep_alphas_sorted = sorted(sweep_macros.keys())
    largest_alpha = max(sweep_alphas_sorted)
    F3_pass = bool(best_alpha < largest_alpha)
    cfg.alpha = best_alpha

    # ---- Final fit at best alpha ----
    regs = _fit_per_author_ridges(zA_tr, zB_tr, train_labels, cfg.n_cls, alpha=cfg.alpha)

    # Val pass
    val_preds = _jepa_predict(regs, zA_vl, zB_vl, cfg.n_cls)
    val_met = eval_pack(val_preds, val_labels, val_langs, val_sources,
                        cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met["overall"]["macro_f1"]
    logger.info(f"[val] macro={val_macro:.4f} alpha={cfg.alpha}")

    # Test pass
    t1 = time.time()
    test_preds = _jepa_predict(regs, zA_ts, zB_ts, cfg.n_cls)
    test_time = time.time() - t1
    ts_met = eval_pack(test_preds, test_labels, test_langs, test_sources,
                       cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = val_macro - test_macro
    logger.info(f"[test] macro={test_macro:.4f} gap={gap:+.4f} time={test_time:.0f}s")

    # ---- F1: residual diagonal vs off-diagonal on TRAIN ----
    R = _residual_matrix(regs, zA_tr, zB_tr, train_labels, cfg.n_cls)
    diag = np.array([R[c, c] for c in range(cfg.n_cls)
                     if not np.isnan(R[c, c])])
    off = []
    for c in range(cfg.n_cls):
        for cp in range(cfg.n_cls):
            if c != cp and not np.isnan(R[c, cp]):
                off.append(R[c, cp])
    off = np.array(off) if off else np.array([np.nan])
    own_mean = float(diag.mean()) if diag.size else float("nan")
    other_mean = float(off.mean()) if off.size else float("nan")
    F1_ratio = other_mean / (own_mean + 1e-12) if own_mean > 0 else float("nan")
    F1_pass = bool(F1_ratio >= 1.5)

    # ---- F2: JEPA vs nearest-centroid on [zA | zB] ----
    zAB_tr = np.concatenate([zA_tr, zB_tr], axis=1)
    zAB_ts = np.concatenate([zA_ts, zB_ts], axis=1)
    nc_preds = _nearest_centroid_predict(zAB_tr, train_labels, zAB_ts, cfg.n_cls)
    nc_macro = float(f1_score(test_labels, nc_preds, average="macro", zero_division=0))
    F2_delta = test_macro - nc_macro
    F2_pass = bool(F2_delta >= 0.005)
    logger.info(f"[F2] jepa={test_macro:.4f} nc={nc_macro:.4f} delta={F2_delta:+.4f}")

    ts_met["falsifier_F1_other_over_own_ratio"] = F1_ratio
    ts_met["falsifier_F1_own_residual"]         = own_mean
    ts_met["falsifier_F1_other_residual"]       = other_mean
    ts_met["falsifier_F1_pass"]                 = F1_pass
    ts_met["falsifier_F2_jepa_minus_nc"]        = F2_delta
    ts_met["falsifier_F2_nc_macro"]             = nc_macro
    ts_met["falsifier_F2_pass"]                 = F2_pass
    ts_met["falsifier_F3_alpha_sweep_val_macros"] = sweep_macros
    ts_met["falsifier_F3_best_alpha"]             = float(best_alpha)
    ts_met["falsifier_F3_pass"]                   = F3_pass

    return {
        "tag": tag, "method": "JEPACODE",
        "note": ("Per-author Ridge regressor zA -> zB in PCA-32 latent space. "
                 "Argmin prediction error over per-author predictors. "
                 "No neural net, no negatives, no decoder. CPU-only."),
        "enc": "hashing-1to3char-512 -> PCA-32",
        "bench": cfg.benchmark, "frac": cfg.frac,
        "alpha": float(cfg.alpha),
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "falsifier_F1_other_over_own_ratio": F1_ratio,
        "falsifier_F2_jepa_minus_nc":        F2_delta,
        "falsifier_F3_best_alpha":           float(best_alpha),
        "falsifier_F3_alpha_sweep_val_macros": sweep_macros,
        "test_time_sec": test_time,
        "test_metrics": ts_met,
        "val_history": [sweep_macros[a] for a in sweep_alphas_sorted],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in benchmarks:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp147_jepacode_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} time={res['wall']:.0f}s "
                            f"F1ratio={res['falsifier_F1_other_over_own_ratio']:.3f} "
                            f"F2={res['falsifier_F2_jepa_minus_nc']:+.4f} "
                            f"best_alpha={res['falsifier_F3_best_alpha']}")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()

    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp147_jepacode_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 140)
    print(f"{'Benchmark':<12} {'Frac':>6} {'Train':>7} {'Test':>7} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Gap':>8} {'dPaper':>9} "
          f"{'F1ratio':>9} {'F2delta':>9} {'alpha*':>8} {'Wall':>8}")
    print("-" * 140)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} "
              f"{r['train_size_after_cap']:>7d} {r['test_size_after_cap']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} "
              f"{r['falsifier_F1_other_over_own_ratio']:>9.3f} "
              f"{r['falsifier_F2_jepa_minus_nc']:>+9.4f} "
              f"{r['falsifier_F3_best_alpha']:>8.2f} {r['wall']:>8.0f}s")
    print("=" * 140)


if __name__ == "__main__":
    main()
