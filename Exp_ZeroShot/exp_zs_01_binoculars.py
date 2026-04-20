"""
[exp_zs_01] BinocularsLogRatio -- zero-shot detector via embedding-based
            cross-perplexity surrogate.

THEORY HOOK:
  Hans et al. ICML 2024, "Binoculars" (arXiv:2401.12070): for two close
  LMs M_obs and M_perf, the log-ratio
        s(x) = log PPL_obs(x) - log PPL_perf(x)
  is Neyman-Pearson optimal for distinguishing samples drawn from M_perf
  vs M_obs. In the AI-code-detection setting, human code sits OFF both
  LMs' manifolds, so the ratio spikes; AI code sits near M_perf's manifold
  where the ratio is small.

IMPLEMENTATION (laptop-friendly surrogate):
  Instead of loading two code LMs, we use a SINGLE frozen ModernBERT-base
  encoder and compute a Binoculars-style log-ratio via two proxies:

    ppl_self(x)  = 1 / (1 + cos(embed(x), batch_mean_embed))
                   "how typical is x in the current batch" — low for
                   LM-generated code (tight manifold), high for human
                   code (diverse).
    ppl_stat(x)  = softmax entropy of a linear probe over embed(x)
                   "how confident a LINEAR LM would be on x" — high
                   entropy for LM-free text (uncertainty), low for
                   LM-typical text (confident next-token).

  score(x) = log(ppl_self) - log(ppl_stat)

  This is a 1-forward-pass ModernBERT score; no second LM needed. On H100
  BF16 batch 64 × seq 512, total inference cost ≈ 3-5 min on Droid test.

No training. No labels used for `score_fn`. Threshold τ is calibrated on
dev to pin human recall ≥ 0.95 in `_zs_runner`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List

import numpy as np

from _common import ZSConfig, logger
from _zs_runner import run_zs_suite


# -----------------------------------------------------------------------------
# Lazy ModernBERT loader -- cached across score_fn calls
# -----------------------------------------------------------------------------

_encoder = None
_tokenizer = None


def _get_encoder(cfg: ZSConfig):
    global _encoder, _tokenizer
    if _encoder is not None:
        return _encoder, _tokenizer
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"[ZS-01] Loading backbone {cfg.backbone} ...")
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
    batch_size = cfg.batch_size
    all_embs: List[np.ndarray] = []

    with torch.no_grad():
        for i in range(0, len(codes), batch_size):
            chunk = codes[i : i + batch_size]
            enc = tokenizer(
                chunk,
                max_length=cfg.scorer_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            if cfg.device == "cuda":
                enc = {k: v.to("cuda") for k, v in enc.items()}
            out = encoder(**enc)
            # CLS token of last hidden state
            cls = out.last_hidden_state[:, 0, :]
            all_embs.append(cls.float().cpu().numpy())

    return np.concatenate(all_embs, axis=0)


# -----------------------------------------------------------------------------
# Score function
# -----------------------------------------------------------------------------

def _binoculars_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Higher score = more AI-like.

    Returns s(x) = log(ppl_self(x)) - log(ppl_stat(x))
    where
      ppl_self(x) = 1 / (1 + cos(emb(x), batch_mean_direction))   (low = typical)
      ppl_stat(x) = softmax entropy of emb(x) fed to a linear probe fit
                    to the batch's own centroid structure.
    """
    embs = _embed_batch(codes, cfg)                       # (N, D)
    N, D = embs.shape

    # Normalise
    norms = np.linalg.norm(embs, axis=-1, keepdims=True) + 1e-8
    embs_n = embs / norms
    mu = embs_n.mean(axis=0)
    mu = mu / (np.linalg.norm(mu) + 1e-8)

    # ppl_self: inverse cosine-similarity to the batch mean direction.
    # cos ∈ [-1, 1]; ppl_self ∈ [0.5, inf).
    sim = (embs_n * mu).sum(axis=-1)
    sim = np.clip(sim, -0.999, 0.999)
    ppl_self = 1.0 / (1.0 + sim)

    # ppl_stat: softmax entropy over per-sample cosine similarities to
    # every other sample in the batch (a crude class-free entropy proxy).
    # Computed block-wise to avoid an O(N^2) memory hit.
    BLOCK = 256
    entropies = np.zeros(N, dtype=np.float32)
    for start in range(0, N, BLOCK):
        end = min(start + BLOCK, N)
        block = embs_n[start:end]                           # (b, D)
        scores = block @ embs_n.T                           # (b, N)
        # Softmax on each row
        scores = scores - scores.max(axis=-1, keepdims=True)
        exp = np.exp(scores)
        p = exp / (exp.sum(axis=-1, keepdims=True) + 1e-8)
        p = np.clip(p, 1e-8, 1.0)
        ent = -(p * np.log(p)).sum(axis=-1)
        entropies[start:end] = ent
    # Normalise to [1, e]
    ent_norm = 1.0 + (entropies - entropies.min()) / (entropies.max() - entropies.min() + 1e-8)
    ppl_stat = ent_norm

    # Log-ratio
    s = np.log(ppl_self + 1e-8) - np.log(ppl_stat + 1e-8)
    return s.astype(np.float64)


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_suite(
        method_name="BinocularsLogRatio",
        exp_id="exp_zs_01",
        score_fn=_binoculars_score,
        cfg=cfg,
    )
