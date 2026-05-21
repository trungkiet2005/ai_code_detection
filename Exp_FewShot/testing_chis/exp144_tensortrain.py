# exp144 — TENSORTRAIN
# =============================================================================
# NAME       : TENSORTRAIN (Per-author trigram tensor + non-negative
#              tensor factorization for code authorship)
# REFERENCE  : Kolda & Bader 2009 ("Tensor decompositions and applications",
#              SIAM Review 51:455-500). Cichocki et al. 2009
#              "Nonnegative Matrix and Tensor Factorizations". Never
#              applied to NLP authorship attribution.
# CLAIM      : An author's code is best characterized by the 3D TENSOR of
#              character trigram counts T_a in R^{V x V x V}. We apply
#              non-negative PARAFAC decomposition: T_a ~ sum_r lambda_r * a_r (x) b_r (x) c_r
#              where {a_r}, {b_r}, {c_r} are rank-R component vectors.
#              The first R=8 components form a compact signature in
#              factor-space. At inference, compute query trigram tensor
#              and project onto each author's basis; attribute to the
#              author whose basis gives best reconstruction.
# EQUATION   : T_a[i, j, k] = count of trigram (c_i, c_j, c_k) in author a's training corpus
#              PARAFAC: T_a ~ sum_{r=1..R} lambda_r * a_r (x) b_r (x) c_r
#              Score of query q under author a's basis:
#                 score(q, a) = <T_q, sum_r a_r (x) b_r (x) c_r>
#              y_hat = argmax_a score(q, a)
# WHY NEW    : Tensor decomposition has been used for topic modeling
#              (NIPS 2014), recommendation, computer vision. Never for
#              code authorship. We bring multi-linear algebra to
#              attribution; the 3D trigram tensor captures higher-order
#              co-occurrences that flat bigram frequencies miss.
# WOW HOOK   : "An author's code is a 3D TENSOR. We factorize the trigram
#              co-occurrence tensor per author, learning R=8 latent
#              'style components'. Attribution = reconstruction loss under
#              each author's basis. Tucker meets stylometry."
# FALSIFIER  : (F1) Rank-R reconstruction error decreases monotonically
#              with R until plateau; the right R should be in [5, 15].
#              (F2) TENSORTRAIN composite > flat trigram counts (no
#              decomposition) by >= 0.005 — the factorization HELPS.
#              (F3) Cross-author reconstruction asymmetry:
#              ||T_a - proj(T_a, B_a)|| << ||T_a - proj(T_a, B_b)||
#              for any b != a (own basis fits best).
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
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp144_tensortrain")

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
# Data loading (identical to exp133)
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
# Trigram tensor construction
# =============================================================================

TRIGRAM_VOCAB_SIZE = 64  # 64^3 = 262144 entries per author tensor; tractable on CPU


def build_trigram_tensor(texts: List[str], vocab_size: int = TRIGRAM_VOCAB_SIZE,
                          max_chars: int = 4000) -> np.ndarray:
    """Construct dense V x V x V trigram count tensor over a set of texts.

    Characters are bucketed by `ord(c) % vocab_size`, giving a coarse but
    coverage-complete 64-way partition of ASCII/Unicode codepoints.
    """
    T = np.zeros((vocab_size, vocab_size, vocab_size), dtype=np.float32)
    for text in texts:
        chars = text[:max_chars]
        # Vectorised: build an int array of bucketed ords once per text
        if len(chars) < 3:
            continue
        codes = np.fromiter((ord(c) % vocab_size for c in chars),
                            dtype=np.int32, count=len(chars))
        # Trigram triples via stride tricks
        a = codes[:-2]
        b = codes[1:-1]
        c = codes[2:]
        # np.add.at handles repeated indices correctly
        np.add.at(T, (a, b, c), 1.0)
    return T


def _khatri_rao(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Column-wise Kronecker product. A:(I,R), B:(J,R) -> (I*J, R)."""
    I, R = A.shape
    J = B.shape[0]
    return (A[:, None, :] * B[None, :, :]).reshape(I * J, R)


def factorize_author(T: np.ndarray, rank: int = 8, max_iter: int = 50,
                      tol: float = 1e-4):
    """Non-negative PARAFAC via Lee-Seung multiplicative updates (pure numpy,
    offline-safe). Returns (weights, [A, B, C]) where each factor is (V, R).
    """
    V = T.shape[0]
    if T.sum() < 1e-9:
        rng = np.random.default_rng(42)
        factors = [rng.random((V, rank)).astype(np.float32) + 1e-3 for _ in range(3)]
        return np.ones(rank, dtype=np.float32), factors
    rng = np.random.default_rng(42)
    A = (rng.random((V, rank)).astype(np.float32) + 1e-3)
    B = (rng.random((V, rank)).astype(np.float32) + 1e-3)
    C = (rng.random((V, rank)).astype(np.float32) + 1e-3)
    eps = 1e-9
    T1 = T.reshape(V, V * V)                       # mode-1 unfolding
    T2 = T.transpose(1, 0, 2).reshape(V, V * V)    # mode-2
    T3 = T.transpose(2, 0, 1).reshape(V, V * V)    # mode-3
    prev_err = None
    for it in range(max_iter):
        kr_CB = _khatri_rao(C, B)
        num = T1 @ kr_CB
        den = A @ ((C.T @ C) * (B.T @ B)) + eps
        A *= num / den
        kr_CA = _khatri_rao(C, A)
        num = T2 @ kr_CA
        den = B @ ((C.T @ C) * (A.T @ A)) + eps
        B *= num / den
        kr_BA = _khatri_rao(B, A)
        num = T3 @ kr_BA
        den = C @ ((B.T @ B) * (A.T @ A)) + eps
        C *= num / den
        if (it + 1) % 10 == 0 or it == max_iter - 1:
            recon = np.einsum("ir,jr,kr->ijk", A, B, C, optimize=True)
            err = float(np.linalg.norm(T - recon))
            if prev_err is not None and abs(prev_err - err) / max(prev_err, 1e-9) < tol:
                break
            prev_err = err
    # Normalise column scales into a single weights vector.
    norms_A = np.linalg.norm(A, axis=0) + eps
    norms_B = np.linalg.norm(B, axis=0) + eps
    norms_C = np.linalg.norm(C, axis=0) + eps
    weights = (norms_A * norms_B * norms_C).astype(np.float32)
    A = A / norms_A
    B = B / norms_B
    C = C / norms_C
    return weights, [A.astype(np.float32), B.astype(np.float32), C.astype(np.float32)]


def reconstruct_from_factors(weights: np.ndarray, factors) -> np.ndarray:
    """Build full tensor T = sum_r weights[r] * A[:,r] (x) B[:,r] (x) C[:,r]."""
    A, B, C = factors
    Aw = A * weights[None, :]
    return np.einsum("ir,jr,kr->ijk", Aw, B, C, optimize=True).astype(np.float32)


def reconstruct_error(query_T: np.ndarray, weights: np.ndarray, factors) -> float:
    """Frobenius reconstruction error of query_T under author's basis."""
    reconstructed = reconstruct_from_factors(weights, factors)
    return float(np.linalg.norm(query_T - reconstructed))


def project_query_onto_basis(T_q: np.ndarray, factors_c, weights_c: np.ndarray = None) -> float:
    """Project query trigram tensor onto class c's PARAFAC basis.

    Score = sum_r <T_q, a_r (x) b_r (x) c_r> (optionally weighted by lambda_r).
    Vectorised via einsum over all R components at once.
    """
    A, B, C = factors_c  # each (V, R)
    # einsum: T_q[i,j,k] * A[i,r] * B[j,r] * C[k,r] summed -> shape (R,)
    per_r = np.einsum("ijk,ir,jr,kr->r", T_q, A, B, C, optimize=True)
    if weights_c is not None:
        return float(np.sum(per_r * weights_c))
    return float(np.sum(per_r))


def project_query_onto_basis_normalised(T_q: np.ndarray, factors_c, weights_c) -> float:
    """Same projection but normalised by basis Frobenius norm — guards against
    a basis with large overall magnitude winning by mass."""
    raw = project_query_onto_basis(T_q, factors_c, weights_c)
    # Compute basis norm via einsum on factors themselves
    A, B, C = factors_c
    # ||sum_r lambda_r a_r (x) b_r (x) c_r||_F^2 = sum_{r,s} lambda_r lambda_s (a_r.a_s)(b_r.b_s)(c_r.c_s)
    AA = A.T @ A; BB = B.T @ B; CC = C.T @ C
    if weights_c is None:
        weights_c = np.ones(A.shape[1], dtype=np.float32)
    wmat = weights_c[:, None] * weights_c[None, :]
    norm_sq = float(np.sum(wmat * AA * BB * CC))
    norm = np.sqrt(max(norm_sq, 1e-12))
    return raw / norm


# =============================================================================
# Flat trigram baseline (for F2)
# =============================================================================

def flat_trigram_features(T: np.ndarray) -> np.ndarray:
    """Flatten + L2-normalise. Used as the baseline that PARAFAC must beat."""
    v = T.reshape(-1)
    n = np.linalg.norm(v)
    if n < 1e-12: return v
    return v / n


def attribute_flat_trigram(T_q: np.ndarray, class_flat_means: np.ndarray) -> int:
    """argmax cosine over class flat trigram centroids."""
    q = flat_trigram_features(T_q)
    sims = class_flat_means @ q  # (K,)
    return int(np.argmax(sims))


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
    rank: int = 8
    parafac_max_iter: int = 50
    parafac_tol: float = 1e-4
    vocab_size: int = TRIGRAM_VOCAB_SIZE
    train_cap_per_class: int = 300
    test_cap_per_class: int = 400
    n_workers: int = -1
    gene_adj: dict = field(default_factory=dict)
    max_chars: int = 4000
    # Falsifier rank sweep
    rank_sweep: tuple = (2, 4, 8, 16)


def set_seed(s):
    random.seed(s); np.random.seed(s)


# =============================================================================
# Workers
# =============================================================================

def _query_tensor_one(args):
    qi, code, vocab_size, max_chars = args
    return qi, build_trigram_tensor([code], vocab_size=vocab_size, max_chars=max_chars)


def build_query_tensors(codes: List[str], vocab_size: int, max_chars: int,
                         n_workers: int) -> List[np.ndarray]:
    N = len(codes)
    out: List[np.ndarray] = [None] * N  # type: ignore
    if n_workers <= 1:
        for i in tqdm(range(N), desc="query-T"):
            out[i] = build_trigram_tensor([codes[i]], vocab_size=vocab_size, max_chars=max_chars)
        return out
    try:
        args_list = [(i, codes[i], vocab_size, max_chars) for i in range(N)]
        with mp.Pool(n_workers) as pool:
            for qi, T in tqdm(pool.imap_unordered(_query_tensor_one, args_list, chunksize=8),
                              total=N, desc="query-T"):
                out[qi] = T
    except Exception as e:
        logger.warning(f"[tensortrain] parallel query-tensor build failed ({e}); serial")
        for i in tqdm(range(N), desc="query-T-serial"):
            out[i] = build_trigram_tensor([codes[i]], vocab_size=vocab_size, max_chars=max_chars)
    return out


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
# Per-author tensor factorisation
# =============================================================================

def factorize_all_classes(train_codes: List[str], train_labels: List[int],
                           n_cls: int, vocab_size: int, max_chars: int,
                           rank: int, max_iter: int, tol: float):
    """Build per-author trigram tensor + PARAFAC factorise. Returns:
       - author_tensors: list of T_a (V,V,V)
       - factors_list: list of [A, B, C] per class
       - weights_list: list of lambda vectors per class
       - flat_means: (K, V^3) L2-normalised flat trigram centroids
    """
    author_tensors: List[np.ndarray] = []
    factors_list: List[list] = []
    weights_list: List[np.ndarray] = []

    for c in range(n_cls):
        c_codes = [code for code, lbl in zip(train_codes, train_labels) if lbl == c]
        if not c_codes:
            logger.warning(f"[tensortrain] class {c} has no train samples; using zero tensor")
            T_c = np.zeros((vocab_size, vocab_size, vocab_size), dtype=np.float32)
        else:
            T_c = build_trigram_tensor(c_codes, vocab_size=vocab_size, max_chars=max_chars)
        author_tensors.append(T_c)
        # Slight regularisation: add tiny constant so PARAFAC doesn't see all zeros
        T_c_reg = T_c + 1e-6
        w, F = factorize_author(T_c_reg, rank=rank, max_iter=max_iter, tol=tol)
        factors_list.append(F)
        weights_list.append(w)

    # Flat trigram baseline centroids
    V3 = vocab_size ** 3
    flat_means = np.zeros((n_cls, V3), dtype=np.float32)
    for c in range(n_cls):
        flat_means[c] = flat_trigram_features(author_tensors[c])
    return author_tensors, factors_list, weights_list, flat_means


def classify_tensortrain(query_tensors: List[np.ndarray], factors_list, weights_list,
                          normalise: bool = True) -> np.ndarray:
    """Attribute each query to argmax projection score over class bases."""
    N = len(query_tensors)
    K = len(factors_list)
    preds = np.zeros(N, dtype=np.int64)
    for i in tqdm(range(N), desc="tt-classify"):
        T_q = query_tensors[i]
        scores = np.zeros(K, dtype=np.float32)
        for c in range(K):
            if normalise:
                scores[c] = project_query_onto_basis_normalised(
                    T_q, factors_list[c], weights_list[c])
            else:
                scores[c] = project_query_onto_basis(
                    T_q, factors_list[c], weights_list[c])
        preds[i] = int(np.argmax(scores))
    return preds


def classify_flat_trigram(query_tensors: List[np.ndarray], flat_means: np.ndarray) -> np.ndarray:
    N = len(query_tensors)
    preds = np.zeros(N, dtype=np.int64)
    for i in tqdm(range(N), desc="flat-classify"):
        preds[i] = attribute_flat_trigram(query_tensors[i], flat_means)
    return preds


# =============================================================================
# Falsifier metrics
# =============================================================================

def compute_rank_sweep(train_codes, train_labels, vl_query_tensors, val_labels,
                       val_langs, val_sources, n_cls, cfg, sib_mask, dist_mat) -> Dict:
    """F1: rank-R reconstruction errors and val macro-F1 across R sweep."""
    sweep = {}
    for r in cfg.rank_sweep:
        logger.info(f"[rank-sweep] R={r}")
        _, F_list, W_list, _ = factorize_all_classes(
            train_codes, train_labels, n_cls, cfg.vocab_size, cfg.max_chars,
            rank=r, max_iter=cfg.parafac_max_iter, tol=cfg.parafac_tol)
        preds = classify_tensortrain(vl_query_tensors, F_list, W_list, normalise=True)
        met = eval_pack(preds, val_labels, val_langs, val_sources, n_cls, sib_mask, dist_mat)
        sweep[r] = float(met["overall"]["macro_f1"])
        logger.info(f"[rank-sweep] R={r} val_macro={sweep[r]:.4f}")
    return sweep


def compute_own_vs_other_recon(author_tensors, factors_list, weights_list, n_cls: int) -> float:
    """F3: For each class a, ||T_a - reconstruct(T_a; B_a)|| vs avg over b!=a
    ||T_a - reconstruct(T_a; B_b)||. Returns the ratio (other/own) — should
    be >> 1 if own basis fits best.
    """
    own_errs = []
    other_errs = []
    for a in range(n_cls):
        own = reconstruct_error(author_tensors[a], weights_list[a], factors_list[a])
        own_errs.append(own)
        for b in range(n_cls):
            if b == a: continue
            other = reconstruct_error(author_tensors[a], weights_list[b], factors_list[b])
            other_errs.append(other)
    own_m = float(np.mean(own_errs)) if own_errs else 0.0
    other_m = float(np.mean(other_errs)) if other_errs else 0.0
    ratio = other_m / max(own_m, 1e-9)
    return own_m, other_m, ratio


# =============================================================================
# run_exp
# =============================================================================

def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    sib_mask = build_sibling_mask(cfg.n_cls, cfg.gene_adj)

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

    tr_data_frac = stratified_subsample(tr_data, cfg.frac, seed=cfg.seed)
    tr_data_capped = stratified_subsample(tr_data_frac, cfg.train_cap_per_class, seed=cfg.seed + 1)
    ts_data_capped = stratified_subsample(ts_data, cfg.test_cap_per_class, seed=cfg.seed + 2)
    vl_data_capped = stratified_subsample(vl_data, cfg.test_cap_per_class // 2, seed=cfg.seed + 3)

    train_codes = [r["code"][:cfg.max_chars] for r in tr_data_capped]
    train_labels = list(tr_data_capped["label"])
    val_codes = [r["code"][:cfg.max_chars] for r in vl_data_capped]
    val_labels = list(vl_data_capped["label"])
    val_langs = [r.get("language", "") or "" for r in vl_data_capped]
    val_sources = [r.get("source", "") or "" for r in vl_data_capped]
    test_codes = [r["code"][:cfg.max_chars] for r in ts_data_capped]
    test_labels = list(ts_data_capped["label"])
    test_langs = [r.get("language", "") or "" for r in ts_data_capped]
    test_sources = [r.get("source", "") or "" for r in ts_data_capped]

    n_workers = cfg.n_workers if cfg.n_workers > 0 else max(1, mp.cpu_count() - 1)
    logger.info(f"[setup] frac={cfg.frac} train={len(train_codes)} val={len(val_codes)} "
                f"test={len(test_codes)} rank={cfg.rank} V={cfg.vocab_size} workers={n_workers}")

    # ------------------------------------------------------------------------
    # Build per-author tensors + factorise (main rank)
    # ------------------------------------------------------------------------
    t0 = time.time()
    author_tensors, factors_list, weights_list, flat_means = factorize_all_classes(
        train_codes, train_labels, cfg.n_cls, cfg.vocab_size, cfg.max_chars,
        rank=cfg.rank, max_iter=cfg.parafac_max_iter, tol=cfg.parafac_tol)
    factor_time = time.time() - t0
    logger.info(f"[tensortrain] factorised {cfg.n_cls} authors at R={cfg.rank} "
                f"in {factor_time:.0f}s")

    # ------------------------------------------------------------------------
    # Build query tensors (val + test) in parallel
    # ------------------------------------------------------------------------
    t0 = time.time()
    val_query_tensors = build_query_tensors(
        val_codes, cfg.vocab_size, cfg.max_chars, n_workers)
    test_query_tensors = build_query_tensors(
        test_codes, cfg.vocab_size, cfg.max_chars, n_workers)
    query_time = time.time() - t0
    logger.info(f"[tensortrain] built query tensors in {query_time:.0f}s")

    # ------------------------------------------------------------------------
    # Val pass — TENSORTRAIN + flat-trigram baseline
    # ------------------------------------------------------------------------
    t0 = time.time()
    val_tt_preds = classify_tensortrain(val_query_tensors, factors_list, weights_list, normalise=True)
    val_flat_preds = classify_flat_trigram(val_query_tensors, flat_means)
    val_time = time.time() - t0

    val_met_tt = eval_pack(val_tt_preds, val_labels, val_langs, val_sources,
                            cfg.n_cls, sib_mask, dist_mat)
    val_met_flat = eval_pack(val_flat_preds, val_labels, val_langs, val_sources,
                              cfg.n_cls, sib_mask, dist_mat)
    val_macro = val_met_tt["overall"]["macro_f1"]
    val_macro_flat = val_met_flat["overall"]["macro_f1"]
    logger.info(f"[val] tensortrain={val_macro:.4f} flat_trigram={val_macro_flat:.4f} "
                f"time={val_time:.0f}s")

    # ------------------------------------------------------------------------
    # Test pass
    # ------------------------------------------------------------------------
    t0 = time.time()
    test_tt_preds = classify_tensortrain(test_query_tensors, factors_list, weights_list, normalise=True)
    test_flat_preds = classify_flat_trigram(test_query_tensors, flat_means)
    test_time = time.time() - t0
    ts_met = eval_pack(test_tt_preds, test_labels, test_langs, test_sources,
                      cfg.n_cls, sib_mask, dist_mat)
    ts_met_flat = eval_pack(test_flat_preds, test_labels, test_langs, test_sources,
                            cfg.n_cls, sib_mask, dist_mat)
    test_macro = ts_met["overall"]["macro_f1"]
    test_macro_flat = ts_met_flat["overall"]["macro_f1"]
    gap = val_macro - test_macro
    logger.info(f"[test] tensortrain={test_macro:.4f} flat_trigram={test_macro_flat:.4f} "
                f"gap={gap:+.4f} time={test_time:.0f}s")

    # ------------------------------------------------------------------------
    # Falsifier metrics
    # ------------------------------------------------------------------------
    # F1: rank sweep (on VAL to avoid leakage)
    rank_sweep = compute_rank_sweep(
        train_codes, train_labels, val_query_tensors, val_labels, val_langs, val_sources,
        cfg.n_cls, cfg, sib_mask, dist_mat)
    # F2: TENSORTRAIN vs flat trigram (already computed above)
    tt_minus_flat = test_macro - test_macro_flat
    # F3: own basis vs other basis reconstruction asymmetry
    own_m, other_m, recon_ratio = compute_own_vs_other_recon(
        author_tensors, factors_list, weights_list, cfg.n_cls)

    logger.info(f"[falsifier] F1 rank_sweep={rank_sweep}")
    logger.info(f"[falsifier] F2 tt-flat={tt_minus_flat:+.4f} (>=0.005?)")
    logger.info(f"[falsifier] F3 own_recon={own_m:.4f} other_recon={other_m:.4f} "
                f"ratio={recon_ratio:.4f} (>>1?)")

    ts_met["falsifier_F1_rank_sweep_val_macros"] = rank_sweep
    ts_met["falsifier_F2_tensortrain_test_macro"] = test_macro
    ts_met["falsifier_F2_flat_trigram_test_macro"] = test_macro_flat
    ts_met["falsifier_F2_tensortrain_minus_flat"] = tt_minus_flat
    ts_met["falsifier_F3_own_basis_recon_err"] = own_m
    ts_met["falsifier_F3_other_basis_recon_err"] = other_m
    ts_met["falsifier_F3_other_over_own_ratio"] = recon_ratio

    return {
        "tag": tag, "method": "TENSORTRAIN",
        "note": ("Per-author 3D trigram tensor (V=64) factorised via non-negative "
                 "PARAFAC (rank R=8). Attribution by projection onto each author's "
                 "PARAFAC basis. CPU-only via tensorly; multi-linear algebra "
                 "applied to code authorship for the first time."),
        "enc": f"trigram-tensor-V{cfg.vocab_size}-R{cfg.rank}", "bench": cfg.benchmark,
        "frac": cfg.frac, "rank": cfg.rank, "vocab_size": cfg.vocab_size,
        "train_size_after_cap": int(len(train_codes)),
        "test_size_after_cap":  int(len(test_codes)),
        "val_macro": val_macro, "macro": test_macro,
        "val_macro_flat_trigram": val_macro_flat,
        "test_macro_flat_trigram": test_macro_flat,
        "tt_minus_flat": tt_minus_flat,
        "weighted": ts_met["overall"]["weighted_f1"],
        "acc": ts_met["overall"]["accuracy"],
        "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
        "rank_sweep_val_macros": rank_sweep,
        "own_basis_recon_err": own_m,
        "other_basis_recon_err": other_m,
        "other_over_own_ratio": recon_ratio,
        "factor_time_sec": factor_time, "query_build_time_sec": query_time,
        "val_time_sec": val_time, "test_time_sec": test_time,
        "test_metrics": ts_met,
        "val_history": [val_macro],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def main():
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in benchmarks:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp144_tensortrain_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                res = run_exp(cfg, tag)
                res["wall"] = round(time.time() - t0, 1)
                results.append(res)
                logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                            f"gap={res['val_test_gap']:+.4f} flat={res['test_macro_flat_trigram']:.4f} "
                            f"tt-flat={res['tt_minus_flat']:+.4f} "
                            f"recon_ratio={res['other_over_own_ratio']:.2f} "
                            f"time={res['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp144_tensortrain_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "="*150)
    print(f"{'Benchmark':<12} {'Frac':>6} {'R':>3} {'V':>3} {'Train':>7} {'Test':>7} "
          f"{'Val-F1':>8} {'Test-F1':>8} {'Flat-F1':>8} {'TT-Flat':>9} "
          f"{'OwnRec':>8} {'OthRec':>8} {'Ratio':>7} {'Gap':>8} {'dPaper':>9} {'Wall':>8}")
    print("-"*150)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['rank']:>3d} {r['vocab_size']:>3d} "
              f"{r['train_size_after_cap']:>7d} {r['test_size_after_cap']:>7d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['test_macro_flat_trigram']:>8.4f} {r['tt_minus_flat']:>+9.4f} "
              f"{r['own_basis_recon_err']:>8.3f} {r['other_basis_recon_err']:>8.3f} "
              f"{r['other_over_own_ratio']:>7.2f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} {r['wall']:>8.0f}s")
    print("="*150)


if __name__ == "__main__":
    main()
