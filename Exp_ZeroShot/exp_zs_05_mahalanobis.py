"""
[exp_zs_05] MahalanobisOnManifold -- zero-shot via Sigma-inverse-weighted
            distance to human-code centroid in ModernBERT embedding space.

THEORY HOOK:
  Singh et al. "On-Manifold Likelihood: A Riemannian Test for Machine Text"
  (TMLR 2025, arXiv:2504.07734).
  Theorem: under local Gaussianity of the pretraining manifold, the
  Mahalanobis distance
      d_M(x) = (phi(x) - mu)^T Sigma^-1 (phi(x) - mu)
  is the MINIMAX-optimal test for "phi(x) belongs to the reference
  distribution" vs "phi(x) drawn from an alternate distribution". Here
  mu, Sigma are fit on a *human-only* reference set. AI-code concentrates
  tighter around the pretraining mode than natural human code, so AI
  samples have LOWER d_M while human samples spread further along the
  manifold.

DIFFERENCE FROM exp_zs_04 SpectralSignature:
  - Spectral: 1-D PC-1 projection (scalar axis).
  - Mahalanobis: full-rank Sigma^-1 metric (uses all 768 dims, not just PC-1).
  Orthogonal signal; pairs well with spectral for ensembling.

IMPLEMENTATION:
  1. First call (assumed = dev split) -> fit mu, Sigma_inv on the embedded
     dev samples with label==0 (human only).
  2. All subsequent calls -> compute d_M(x) for every sample.
  3. We use a Ledoit-Wolf shrinkage estimator for Sigma to stay stable at
     d=768 with only 5K dev samples (otherwise Sigma is near-singular).

NOTE on sign: Mahalanobis distance is non-negative; AI code has LOWER
distance. We return +d_M and let the tau-calibration flip the sign if
dev human-recall is maximised on the OTHER side (we return +d_M; the
calibration step decides).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap -- Kaggle-compatible: clones the repo if needed, adds
# Exp_ZeroShot/ to sys.path. Works inside .py files, .ipynb cells, and
# notebook %run magic (no dependence on __file__).
# ---------------------------------------------------------------------------
import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"
REQUIRED_FILE = "_zs_runner.py"


def _bootstrap_zs_path() -> str:
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, "Exp_ZeroShot"),
        os.path.join(cwd, "ai_code_detection", "Exp_ZeroShot"),
    ]
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        candidates.insert(0, here)
    except NameError:
        pass
    for c in candidates:
        if os.path.exists(os.path.join(c, REQUIRED_FILE)):
            return c
    repo_dir = os.path.join(cwd, "ai_code_detection")
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir, ignore_errors=True)
    print(f"[bootstrap] Cloning {REPO_URL} -> {repo_dir}")
    subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, repo_dir])
    return os.path.join(repo_dir, "Exp_ZeroShot")


_zs_dir = _bootstrap_zs_path()
if _zs_dir not in sys.path:
    sys.path.insert(0, _zs_dir)
for _mod in list(sys.modules):
    if _mod in ("_common", "_zs_loaders", "_zs_runner"):
        del sys.modules[_mod]
print(f"[bootstrap] Exp_ZeroShot path: {_zs_dir}")

from typing import List

import numpy as np

from _common import ZSConfig, logger
from _zs_runner import run_zs_oral


_encoder = None
_tokenizer = None
_mu = None
_sigma_inv = None


def _get_encoder(cfg: ZSConfig):
    global _encoder, _tokenizer
    if _encoder is not None:
        return _encoder, _tokenizer
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"[ZS-05] Loading backbone {cfg.backbone} ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    _encoder = AutoModel.from_pretrained(cfg.backbone)
    _encoder.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _encoder = _encoder.to("cuda")
        if cfg.precision == "bf16":
            _encoder = _encoder.to(torch.bfloat16)
    return _encoder, _tokenizer


def _embed_batch(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    import torch

    encoder, tokenizer = _get_encoder(cfg)
    out: List[np.ndarray] = []
    with torch.no_grad():
        for i in range(0, len(codes), cfg.batch_size):
            chunk = codes[i : i + cfg.batch_size]
            enc = tokenizer(
                chunk,
                max_length=cfg.scorer_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            if cfg.device == "cuda":
                enc = {k: v.to("cuda") for k, v in enc.items()}
            res = encoder(**enc)
            cls = res.last_hidden_state[:, 0, :]
            out.append(cls.float().cpu().numpy())
    return np.concatenate(out, axis=0)


def _fit_gaussian(embs: np.ndarray) -> (np.ndarray, np.ndarray):
    """Fit mu and Sigma^-1 via Ledoit-Wolf shrinkage. embs: (N, D)."""
    from sklearn.covariance import LedoitWolf

    mu = embs.mean(axis=0)
    lw = LedoitWolf(assume_centered=False).fit(embs)
    sigma = lw.covariance_
    # Regularise diagonal to avoid numerical issues before inversion
    d = sigma.shape[0]
    sigma = sigma + 1e-4 * np.eye(d)
    sigma_inv = np.linalg.inv(sigma)
    logger.info(f"[ZS-05] Fit Ledoit-Wolf Gaussian on {embs.shape[0]} samples, d={d}, shrinkage={lw.shrinkage_:.4f}")
    return mu, sigma_inv


def _mahalanobis_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """First call fits mu/Sigma^-1 on the embedding space of these `codes`; later
    calls reuse them. Returns +Mahalanobis distance (higher = further from
    the fitted mode). Calibration step in _zs_runner will flip sign if AI
    samples ended up closer to mu than humans on dev.
    """
    global _mu, _sigma_inv

    embs = _embed_batch(codes, cfg)
    if _mu is None:
        # First call = dev split. Fit on ALL dev samples (label-free).
        _mu, _sigma_inv = _fit_gaussian(embs)

    diff = embs - _mu
    # d_M(x) = diff @ Sigma^-1 @ diff.T, per-row. Compute via batched matmul.
    intermediate = diff @ _sigma_inv
    dists = np.einsum("ij,ij->i", intermediate, diff)
    return dists.astype(np.float64)


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="MahalanobisOnManifold",
        exp_id="exp_zs_05",
        score_fn=_mahalanobis_score,
        cfg=cfg,
    )
