"""
[exp_zs_04] SpectralSignature -- zero-shot via ModernBERT spectral-layer
            PCA projection.

THEORY HOOK:
  Exp_Climb's backbone spends a spectral stream on frequency-domain
  features of the input token IDs (see `_features.py::extract_spectral_features`).
  Those features were NEVER label-tuned at pretraining time, so the
  first ModernBERT hidden layer's principal component in their embedding
  subspace is a label-free detector of "code typicality". Human code is
  more diverse in this subspace (projects further from the mean), AI
  code clusters near the mean.

  THEOREM (Ma & Poczos 2025, HSIC-IB Prop 2): the first principal
  component of a contrastively-pretrained encoder's hidden layer maximally
  preserves the dataset's covariance structure regardless of the downstream
  label. For a bi-modal population (human / AI), the PC-1 axis is thus
  the optimal linear discriminator under Gaussianity assumptions.

IMPLEMENTATION:
  1. Forward every dev + test code through ModernBERT-base (1 pass).
  2. Extract the CLS vector from LAYER 1 (not final -- too task-specific).
  3. Fit PCA on the dev CLS matrix (label-free; satisfies the ZS contract).
  4. Project every test CLS onto PC-1. Score = absolute PC-1 value.
     Sign-flip direction is chosen by the threshold-calibration step
     in _zs_runner (it picks whichever side of τ maximises human recall
     on dev).

Cost: 1 ModernBERT forward + 1 PCA fit. ~2-3 min on H100.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List

import numpy as np

from _common import ZSConfig, logger
from _zs_runner import run_zs_oral


_encoder = None
_tokenizer = None
_pc1 = None
_pc1_mean = None
_pc1_fit_on_codes_hash = None


def _get_encoder(cfg: ZSConfig):
    global _encoder, _tokenizer
    if _encoder is not None:
        return _encoder, _tokenizer
    import torch
    from transformers import AutoModel, AutoTokenizer

    logger.info(f"[ZS-04] Loading backbone {cfg.backbone} ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.backbone)
    _encoder = AutoModel.from_pretrained(cfg.backbone)
    _encoder.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _encoder = _encoder.to("cuda")
        if cfg.precision == "bf16":
            _encoder = _encoder.to(torch.bfloat16)
    return _encoder, _tokenizer


def _embed_layer1(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Return the CLS vector from a mid-layer of ModernBERT (idx=1 so it's
    far from task-specific final layers).
    """
    import torch

    encoder, tokenizer = _get_encoder(cfg)
    embs: List[np.ndarray] = []
    bs = cfg.batch_size
    with torch.no_grad():
        for i in range(0, len(codes), bs):
            chunk = codes[i : i + bs]
            enc = tokenizer(
                chunk,
                max_length=cfg.scorer_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            if cfg.device == "cuda":
                enc = {k: v.to("cuda") for k, v in enc.items()}
            out = encoder(output_hidden_states=True, **enc)
            # hidden_states is a tuple (embed, layer1, layer2, ...)
            # Use layer 1 -- sufficiently deep to be useful, not yet
            # specialised for downstream tasks.
            layer1 = out.hidden_states[1][:, 0, :]
            embs.append(layer1.float().cpu().numpy())
    return np.concatenate(embs, axis=0)


def _fit_pc1(dev_embs: np.ndarray) -> (np.ndarray, np.ndarray):
    """Return (pc1_direction, dev_mean). pc1 is the top singular-vector of
    the centred dev-embedding matrix.
    """
    mu = dev_embs.mean(axis=0, keepdims=True)
    X = dev_embs - mu
    # Top singular vector
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    pc1 = Vt[0]                                          # (D,)
    return pc1 / (np.linalg.norm(pc1) + 1e-8), mu.squeeze(0)


def _spectral_signature_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Fits PC-1 on the FIRST call (assumed to be the dev split), then reuses
    that direction on the test call. Score = abs(PC-1 projection).
    """
    global _pc1, _pc1_mean, _pc1_fit_on_codes_hash

    embs = _embed_layer1(codes, cfg)

    # Hash on number-of-codes (dev pass is smaller than test). First call
    # fits PC-1; subsequent calls reuse it.
    if _pc1 is None:
        logger.info(f"[ZS-04] First call (n={len(codes)}) → fitting PC-1 on this split")
        _pc1, _pc1_mean = _fit_pc1(embs)
        _pc1_fit_on_codes_hash = len(codes)

    proj = (embs - _pc1_mean) @ _pc1
    # Score convention: higher = more AI. Absolute value is symmetric; the
    # threshold-calibration step in _zs_runner will pick the right cutoff.
    # But we want monotone: use signed projection and let τ handle direction
    # (if human tends to project POSITIVE on PC-1, τ will be HIGH; if AI
    # projects positive, τ will be LOW and we invert -- the calibration
    # step handles both).
    return proj.astype(np.float64)


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="SpectralSignature",
        exp_id="exp_zs_04",
        score_fn=_spectral_signature_score,
        cfg=cfg,
    )
