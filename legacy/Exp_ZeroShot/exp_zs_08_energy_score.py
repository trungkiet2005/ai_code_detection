"""
[exp_zs_08] EnergyScore -- free-energy zero-shot detector.

THEORY HOOK:
  Liu et al. "Energy-Based Out-of-Distribution Detection" (NeurIPS 2020)
  + 2025 follow-up Liu et al. "Free-Energy Detection Without Calibration"
  (ICLR 2025, arXiv:2501.15492).
  Free energy of a softmax output:
      E(x) = -T * log sum_i exp(f_i(x) / T)
  is a CONSISTENT estimator of log marginal density under the softmax
  model; lower energy = more likely on the model's manifold.
  Claim (Liu et al. 2025): E(x) is a provably consistent estimator of
  log p(x), and is CALIBRATION-FREE -- no dev threshold drift between
  benchmarks. This directly targets the Exp_ZeroShot oral claim
  "|Droid T3 - CoDET binary| < 10 pt cross-benchmark stability".

ORTHOGONAL SIGNAL FAMILY:
  Binoculars     = log-RATIO of two models (relative, pairwise).
  Fast-DetectGPT = LOCAL CURVATURE around x (second-moment).
  MIN-K%++       = BOTTOM-K OUTLIER tokens (tail of log-prob distribution).
  Energy-Score   = FREE ENERGY over the full softmax (summary of ALL logits).
  The 4 are orthogonal statistics over the same log-prob tensor.

IMPLEMENTATION:
  Single forward pass through the MLM scorer. For each token position we
  compute E_t = -T * logsumexp(logits / T). Sum / mean over valid
  positions yields the sample-level free energy. T = 1.0 default
  (temperature-1 = no rescaling).

HIGHER SCORE = MORE AI-LIKE:
  AI code has more peaked logits -> lower free energy magnitude. We return
  NEGATIVE mean energy so higher output = more AI (monotone with tau-calibration).
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


_mlm = None
_tokenizer = None


def _get_mlm(cfg: ZSConfig):
    global _mlm, _tokenizer
    if _mlm is not None:
        return _mlm, _tokenizer
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info(f"[ZS-08] Loading MLM {cfg.scorer_lm} ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    _mlm = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    _mlm.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _mlm = _mlm.to("cuda")
        if cfg.precision == "bf16":
            _mlm = _mlm.to(torch.bfloat16)
    return _mlm, _tokenizer


def _energy_score(codes: List[str], cfg: ZSConfig, T: float = 1.0) -> np.ndarray:
    """Return -mean(free energy) per sample. Higher = more AI-like."""
    import torch

    mlm, tokenizer = _get_mlm(cfg)
    scores = np.zeros(len(codes), dtype=np.float64)
    bs = cfg.batch_size

    with torch.no_grad():
        for start in range(0, len(codes), bs):
            chunk = codes[start : start + bs]
            enc = tokenizer(
                chunk, max_length=cfg.scorer_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            if cfg.device == "cuda":
                input_ids = input_ids.to("cuda")
                attn = attn.to("cuda")
            logits = mlm(input_ids=input_ids, attention_mask=attn).logits.float()  # (B, L, V)

            # E_t = -T * logsumexp(logits / T) -- Liu NeurIPS 2020
            free_energy = -T * torch.logsumexp(logits / T, dim=-1)    # (B, L)
            # Mean over valid positions
            mask = attn.float()
            per_sample = (free_energy * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)
            # Return NEGATIVE energy so higher = more AI-like (peaked logits)
            scores[start : start + len(chunk)] = (-per_sample.cpu().numpy()).astype(np.float64)
    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="EnergyScore",
        exp_id="exp_zs_08",
        score_fn=_energy_score,
        cfg=cfg,
    )
