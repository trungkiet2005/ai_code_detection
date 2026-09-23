# exp162 — CHERNOFF  (E3: the distinguishability estimator)
# =============================================================================
# NAME       : CHERNOFF (Pairwise generator distinguishability / C_min estimator).
# REFERENCE  : new (D1 proposal). Grounds the attribution lower bound of
#              Chakraborty (arXiv:2304.04736, binary Chernoff sample complexity)
#              generalised to the m-ary worst-pair coupling C_min. Embeddings
#              from frozen UniXcoder (same encoder family as ext_gptsniffer).
# CLAIM      : The collapse curve of E2 is not a curve fit — it is set by the
#              MINIMUM PAIRWISE distinguishability C_min between generator output
#              distributions. We measure C_min directly from frozen-encoder
#              embeddings: a Gaussian-Chernoff/Bhattacharyya distance, a
#              symmetric-KL, and a model-free 2-sample-classifier AUROC proxy.
# EQUATION   : Gaussian Bhattacharyya (Chernoff at s=1/2):
#              D_B(i,j)=1/8 (mu_i-mu_j)^T Sig^-1 (mu_i-mu_j)
#                       +1/2 ln( det Sig / sqrt(det Sig_i det Sig_j) ),
#              Sig=(Sig_i+Sig_j)/2 ;  C_min = min_{i!=j} D_B(i,j).
# WHY NEW    : No code-attribution paper reports a pairwise generator-distance
#              matrix or a measured C_min; distinguishability is assumed, never
#              quantified from data, so no prior work can tie error to geometry.
# WOW HOOK   : "Every AI-content detector's failure has a number: the Chernoff
#              distance of its two most similar generators. Measure that one
#              worst pair and you have predicted the ceiling."
# FALSIFIER  : If, across benchmarks, pairwise distinguishability (any of the
#              three estimators) does NOT rank-correlate (|Spearman| < 0.3) with
#              the observed per-pair confusion of the exp160/exp161 attributor,
#              the estimator is not measuring what the theory needs -> E3 fails.
# =============================================================================
from __future__ import annotations

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
from dataclasses import dataclass
from typing import List, Dict

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])
_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
import torch
from datasets import load_dataset
from sklearn.decomposition import PCA
from sklearn.covariance import LedoitWolf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp162")

KAGGLE_MODELS = os.environ.get("MODELS_DIR", "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models")
KAGGLE_CODET  = os.environ.get("CODET_PARQUET", "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet")
KAGGLE_AICD   = os.environ.get("AICD_DIR", "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench")

# --- data loading (identical to ext_gptsniffer.py) ---------------------------
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
        label = 0 if _is_human(r.get("target", "")) else vocab.get(str(r.get("model", "") or "").strip(), -1)
        return {"code": code, "label": label,
                "language": str(r.get("language", "")).strip().lower(),
                "source": str(r.get("source", "")).strip().lower()}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _conv_aicd(split):
    def row(r):
        return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1)),
                "language": str(r.get("language", "")).strip().lower(), "source": ""}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _load_codet():
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        return tr
    s = ds.train_test_split(test_size=0.1, seed=42)
    return s["train"]

def _load_aicd(task="t2"):
    task_name = {"t1": "T1", "t2": "T2", "t3": "T3"}.get(task.lower())
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path): raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found")
    pf = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError("[aicd] STRICT: No parquet files")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        if len(tr) > 0: return tr
    s = ds.train_test_split(test_size=0.1, seed=42)
    return s["train"]

# --- frozen mean-pooled embeddings -------------------------------------------
class _CodeDS(TD):
    def __init__(self, codes, tok, seq_len):
        self.codes = codes; self.tok = tok; self.seq_len = seq_len
    def __len__(self): return len(self.codes)
    def __getitem__(self, i):
        enc = self.tok(self.codes[i][:5000], padding="max_length", max_length=self.seq_len, truncation=True)
        return {"input_ids": torch.tensor(enc["input_ids"], dtype=torch.long),
                "attention_mask": torch.tensor(enc["attention_mask"], dtype=torch.long)}

@torch.no_grad()
def embed_codes(codes, tok, model, seq_len, bs, device):
    ds = _CodeDS(codes, tok, seq_len)
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    out = []
    for b in tqdm(dl, desc="embed", leave=False):
        ids = b["input_ids"].to(device); mask = b["attention_mask"].to(device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(device == "cuda")):
            h = model(input_ids=ids, attention_mask=mask).last_hidden_state  # (B,L,d)
        m = mask.unsqueeze(-1).float()
        pooled = (h.float() * m).sum(1) / m.sum(1).clamp(min=1e-6)  # masked mean-pool
        out.append(pooled.cpu().numpy())
    return np.concatenate(out, 0)

# --- distinguishability estimators -------------------------------------------
def gaussian_bhattacharyya(Xi, Xj):
    """Gaussian Bhattacharyya distance = Chernoff info at s=1/2. Shrinkage cov."""
    mu_i = Xi.mean(0); mu_j = Xj.mean(0)
    Si = LedoitWolf().fit(Xi).covariance_
    Sj = LedoitWolf().fit(Xj).covariance_
    S = 0.5 * (Si + Sj)
    d = mu_i - mu_j
    Sinv = np.linalg.pinv(S)
    term1 = 0.125 * float(d @ Sinv @ d)
    _, ld_S = np.linalg.slogdet(S)
    _, ld_i = np.linalg.slogdet(Si)
    _, ld_j = np.linalg.slogdet(Sj)
    term2 = 0.5 * (ld_S - 0.5 * (ld_i + ld_j))
    return term1 + max(term2, 0.0)

def gaussian_sym_kl(Xi, Xj):
    mu_i = Xi.mean(0); mu_j = Xj.mean(0)
    Si = LedoitWolf().fit(Xi).covariance_
    Sj = LedoitWolf().fit(Xj).covariance_
    Sii = np.linalg.pinv(Si); Sji = np.linalg.pinv(Sj)
    d = mu_i - mu_j; k = Xi.shape[1]
    kl_ij = 0.5 * (np.trace(Sji @ Si) + float(d @ Sji @ d) - k
                   + (np.linalg.slogdet(Sj)[1] - np.linalg.slogdet(Si)[1]))
    kl_ji = 0.5 * (np.trace(Sii @ Sj) + float(d @ Sii @ d) - k
                   + (np.linalg.slogdet(Si)[1] - np.linalg.slogdet(Sj)[1]))
    return 0.5 * (kl_ij + kl_ji)

def two_sample_auroc(Xi, Xj, seed=0):
    """Model-free proxy: linear classifier i-vs-j, held-out AUROC. 0.5=indistinguishable."""
    X = np.concatenate([Xi, Xj], 0)
    y = np.concatenate([np.zeros(len(Xi)), np.ones(len(Xj))])
    if len(np.unique(y)) < 2 or len(y) < 6:
        return 0.5
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y)
    clf = LogisticRegression(max_iter=1000, C=1.0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xte)[:, 1]
    try:
        return float(roc_auc_score(yte, p))
    except Exception:
        return 0.5

def run_bench(bench, cfg):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    bs = 64 if device == "cuda" else 8
    if bench == "codet_m4":
        raw = _load_codet(); vocab = _vocab(raw); data = _conv_codet(raw, vocab); human_class = 0
    else:
        raw = _load_aicd("t2"); data = _conv_aicd(raw); human_class = None
    labels = np.array(data["label"]); codes = list(data["code"])
    all_labels = sorted(set(labels.tolist()))
    generators = [c for c in all_labels if c != human_class]
    if cfg["include_human"] and human_class is not None:
        classes_used = [human_class] + generators
    else:
        classes_used = generators
    if cfg["smoke"]:
        classes_used = classes_used[:3]
    logger.info(f"[{bench}] classes_used={classes_used} human={human_class}")

    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg["enc"]), local_files_only=True)
    model = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, cfg["enc"]), local_files_only=True).to(device).eval()

    # gather capped per-class codes, embed
    rng = random.Random(cfg["seed"]); emb_by_cls = {}
    for c in classes_used:
        idx = [i for i in range(len(codes)) if labels[i] == c]
        if len(idx) > cfg["max_per_class"]:
            idx = rng.sample(idx, cfg["max_per_class"])
        cls_codes = [codes[i] for i in idx]
        if len(cls_codes) < 4:
            logger.warning(f"  class {c}: only {len(cls_codes)} samples, skipping"); continue
        emb_by_cls[c] = embed_codes(cls_codes, tok, model, cfg["seq"], bs, device)
        logger.info(f"  class {c}: embedded {len(cls_codes)} -> {emb_by_cls[c].shape}")
    used = [c for c in classes_used if c in emb_by_cls]

    # PCA to stabilise covariance in 768-d
    stacked = np.concatenate([emb_by_cls[c] for c in used], 0)
    pca_dim = min(cfg["pca_dim"], stacked.shape[1], stacked.shape[0] - 1)
    pca = PCA(n_components=pca_dim, random_state=cfg["seed"]).fit(stacked)
    red = {c: pca.transform(emb_by_cls[c]) for c in used}

    n = len(used)
    bh = np.zeros((n, n)); skl = np.zeros((n, n)); au = np.full((n, n), 0.5)
    for a in range(n):
        for b2 in range(a + 1, n):
            Xi, Xj = red[used[a]], red[used[b2]]
            v_bh = gaussian_bhattacharyya(Xi, Xj)
            v_skl = gaussian_sym_kl(Xi, Xj)
            v_au = two_sample_auroc(Xi, Xj, seed=cfg["seed"])
            bh[a, b2] = bh[b2, a] = v_bh
            skl[a, b2] = skl[b2, a] = v_skl
            au[a, b2] = au[b2, a] = v_au

    def _offdiag_pairs(M):
        return [(used[i], used[j], float(M[i, j])) for i in range(n) for j in range(i + 1, n)]

    # generator-only worst pair (attribution C_min excludes human)
    gen_idx = [k for k, c in enumerate(used) if c != human_class]
    def _min_over(M, idxs, invert=False):
        vals = [(used[i], used[j], (M[i, j])) for a_, i in enumerate(idxs) for j in idxs[a_ + 1:]]
        if not vals: return None
        key = (lambda t: -t[2]) if invert else (lambda t: t[2])
        return min(vals, key=key)

    cmin_bh = _min_over(bh, gen_idx)
    cmin_skl = _min_over(skl, gen_idx)
    # AUROC: distinguishability high -> more separable; worst pair = min AUROC (closest to 0.5)
    cmin_au = _min_over(au, gen_idx)

    rec = {"tag": f"exp162_{bench}", "bench": bench, "embed_source": "frozen_unixcoder",
           "enc": cfg["enc"], "pca_dim": pca_dim, "max_per_class": cfg["max_per_class"],
           "classes_used": used, "human_class": (int(human_class) if human_class is not None else None),
           "n_classes": n,
           "bhattacharyya_matrix": bh.tolist(),
           "sym_kl_matrix": skl.tolist(),
           "auroc_matrix": au.tolist(),
           "pairs_bhattacharyya": _offdiag_pairs(bh),
           "pairs_auroc": _offdiag_pairs(au),
           "C_min_bhattacharyya": ({"class_i": cmin_bh[0], "class_j": cmin_bh[1], "value": cmin_bh[2]} if cmin_bh else None),
           "C_min_sym_kl": ({"class_i": cmin_skl[0], "class_j": cmin_skl[1], "value": cmin_skl[2]} if cmin_skl else None),
           "worst_pair_auroc": ({"class_i": cmin_au[0], "class_j": cmin_au[1], "value": cmin_au[2]} if cmin_au else None),
           "mean_bhattacharyya": float(np.mean([v for *_, v in _offdiag_pairs(bh)])) if n > 1 else None,
           "mean_auroc": float(np.mean([v for *_, v in _offdiag_pairs(au)])) if n > 1 else None,
           "note": "C_min_* computed over GENERATOR pairs only (attribution regime, human excluded).",
           "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}
    if cmin_bh:
        logger.info(f"[{bench}] C_min (Bhattacharyya) = {cmin_bh[2]:.4f} at pair {cmin_bh[0]}~{cmin_bh[1]}; "
                    f"worst AUROC={cmin_au[2]:.3f}")
    del model
    import gc; gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return rec

def main():
    bench = os.environ.get("BENCH", "codet_m4")
    cfg = {"enc": os.environ.get("ENC", "unixcoder-base"),
           "seq": int(os.environ.get("SEQ", "512")),
           "seed": int(os.environ.get("SEED", "42")),
           "pca_dim": int(os.environ.get("PCA_DIM", "64")),
           "max_per_class": int(os.environ.get("MAX_PER_CLASS", "500")),
           "include_human": os.environ.get("INCLUDE_HUMAN", "1") == "1",
           "smoke": os.environ.get("SMOKE", "0") == "1"}
    if cfg["smoke"]:
        cfg["max_per_class"] = 30; cfg["pca_dim"] = 8
    benches = [bench] if bench != "all" else ["codet_m4", "aicd_t2"]
    all_records = []
    for b in benches:
        try:
            all_records.append(run_bench(b, cfg))
        except Exception as e:
            logger.error(f"[{b}] FAILED: {e}")
            import traceback; traceback.print_exc()
    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp162_chernoff_results.json"), "w") as f:
        json.dump(all_records, f, indent=2)
    logger.info(f"[done] wrote {len(all_records)} records")

if __name__ == "__main__":
    main()
