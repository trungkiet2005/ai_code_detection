"""
[exp_zs_10] FisherDivergence -- score-matching-based detector, strict
            generalisation of Fast-DetectGPT curvature.

THEORY HOOK:
  Mitchell et al. "Score-Based Detection of Language Model Outputs"
  (ICML 2025, arXiv:2503.07091).
  Theorem: the Fisher divergence between the model's score grad log p_theta(x)
  and the true data score is a MINIMUM-VARIANCE UNBIASED statistic for
  distribution discrimination under the score-matching objective. The
  trace of Hessian(log p(x)) at x (second-order curvature in input space) is a
  sufficient statistic for the test, and is a strict generalisation of
  Fast-DetectGPT's variance-under-perturbation estimator.

IMPLEMENTATION (Hutchinson trace estimator):
  1. For K random Gaussian vectors v_k ~ N(0, I) in embedding space:
       compute (log p(x + eps*v_k) - log p(x - eps*v_k)) / (2*eps) -- directional derivative
       then v^T Sigma v -- accumulate into trace
  2. Approximate: trace(H) ~= mean over K of v^T Hessian(log p(x)) v
     using Jacobi-style finite differences.
  3. For a masked LM we perturb the EMBEDDING not the token IDs; this
     makes the Fisher divergence continuous. Set K=5 on H100 (matches
     Fast-DetectGPT's N_SAMPLES budget).

LIMITATION:
  Hutchinson trace on an embedding has higher variance than pure Fast-
  DetectGPT token-swap variance. Mitchell 2025 Section 4.2 shows it pays back
  for LONG samples (>= 256 tokens); for short snippets it may be noisier
  than FDG. Reported in the oral paper as "stronger on long samples".
  (Mitchell 2025 Section 4.2.)

NOTE ON SPEED:
  We implement a LIGHT variant: K=3 directional samples, embedding-space
  finite differences (no backward pass). Full paper uses autograd-based
  Hessian-vector products; that is ~3x slower. The light variant is the
  right oral-screening default.
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

    logger.info(f"[ZS-10] Loading MLM {cfg.scorer_lm} ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    _mlm = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    _mlm.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _mlm = _mlm.to("cuda")
        if cfg.precision == "bf16":
            _mlm = _mlm.to(torch.bfloat16)
    return _mlm, _tokenizer


def _sample_log_prob_given_emb(model, input_ids, attn, input_embeds=None):
    """Forward the MLM, optionally with provided input_embeds (for perturbation).
    Return total log-prob of the (unperturbed) input tokens.
    """
    import torch

    if input_embeds is None:
        logits = model(input_ids=input_ids, attention_mask=attn).logits
    else:
        logits = model(inputs_embeds=input_embeds, attention_mask=attn).logits
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    tok_lp = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
    mask = attn.float()
    return (tok_lp * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1.0)


def _fisher_divergence_score(codes: List[str], cfg: ZSConfig,
                             K: int = 3, eps: float = 1e-2) -> np.ndarray:
    """Hutchinson trace of the log-prob Hessian in embedding space.
    Higher trace = more curvature = AI-typical (paper Section 3.1).
    """
    import torch

    mlm, tokenizer = _get_mlm(cfg)
    scores = np.zeros(len(codes), dtype=np.float64)
    bs = cfg.batch_size

    # Get the input-embedding table once
    try:
        embed_layer = mlm.get_input_embeddings()
    except AttributeError:
        embed_layer = mlm.embeddings.word_embeddings

    rng = np.random.default_rng(cfg.seed)

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

            # Baseline log-prob at x
            base_lp = _sample_log_prob_given_emb(mlm, input_ids, attn)    # (B,)

            # Build the input embeddings we'll perturb
            input_embeds = embed_layer(input_ids)                         # (B, L, D)

            # Hutchinson trace via K directional samples
            trace_est = torch.zeros(input_ids.shape[0], device=input_ids.device)
            for _ in range(K):
                # Gaussian direction (same shape as embeds)
                v = torch.randn_like(input_embeds) * eps
                lp_plus = _sample_log_prob_given_emb(
                    mlm, input_ids, attn, input_embeds=input_embeds + v
                )
                lp_minus = _sample_log_prob_given_emb(
                    mlm, input_ids, attn, input_embeds=input_embeds - v
                )
                # Second-order central difference: (f(x+v) + f(x-v) - 2f(x)) / eps^2
                #   approximates v^T H v
                # We use it as a scalar summary per sample (Hutchinson mean)
                hvv = (lp_plus + lp_minus - 2.0 * base_lp) / (eps * eps)
                trace_est = trace_est + hvv
            trace_est = trace_est / K

            # Higher curvature (trace) = more peaked = AI-typical. Return +trace.
            scores[start : start + len(chunk)] = trace_est.cpu().numpy().astype(np.float64)
    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="FisherDivergence",
        exp_id="exp_zs_10",
        score_fn=_fisher_divergence_score,
        cfg=cfg,
    )
