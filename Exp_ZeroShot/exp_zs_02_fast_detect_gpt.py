"""
[exp_zs_02] Fast-DetectGPT (code variant) -- zero-shot via masked-LM
            conditional curvature.

THEORY HOOK:
  Bao et al. ICLR 2024, "Fast-DetectGPT" (arXiv:2310.05130):
  samples drawn from an LM sit at LOCAL MAXIMA of their own conditional
  log-prob surface, so perturbing them lowers log-prob rapidly. Human
  text sits off-manifold — perturbations can go either way. The signed
  curvature κ(x) = E_{x' ~ perturb(x)} [log p(x) - log p(x')] is thus
  POSITIVE and large for LM-generated samples, near-zero or negative
  for human samples.

  For code, we use a CodeBERT masked-LM (smaller, faster than GPT-like
  models) as the scorer. Perturbation = independent 15% token masking,
  5 samples, variance used as curvature estimator (Bao et al. §4.2).

IMPLEMENTATION:
  score(x) = mean over 5 independent 15%-masked passes of:
      log p(x_original_tokens | masked_context) - log p(x_random_token | masked_context)
  Higher score ⇒ LM-generated (peaked conditional log-prob).

  Cost: 5 forward passes of CodeBERT per sample, batched to 64 × seq 512.
  On H100 BF16 this is ~8-10 min on Droid test.

No training. Threshold calibrated in `_zs_runner`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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

    logger.info(f"[ZS-02] Loading MLM scorer {cfg.scorer_lm} ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    _mlm = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    _mlm.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _mlm = _mlm.to("cuda")
        if cfg.precision == "bf16":
            _mlm = _mlm.to(torch.bfloat16)
    return _mlm, _tokenizer


def _fast_detect_gpt_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Curvature estimator: for each sample, mask 15% of tokens, compute
    log p(original_token | masked_context). Repeat 3 times and compare to
    log p(random_other_token | masked_context). The gap (higher = more
    LM-like) is the curvature score.
    """
    import torch

    mlm, tokenizer = _get_mlm(cfg)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError(f"Scorer LM {cfg.scorer_lm} has no mask_token")
    vocab_size = mlm.config.vocab_size

    N_SAMPLES = 3          # Monte-Carlo samples per example (paper uses 100; we use 3 for budget)
    MASK_RATE = 0.15
    scores = np.zeros(len(codes), dtype=np.float64)
    bs = cfg.batch_size

    rng = np.random.default_rng(cfg.seed)

    with torch.no_grad():
        for start in range(0, len(codes), bs):
            chunk = codes[start : start + bs]
            enc = tokenizer(
                chunk,
                max_length=cfg.scorer_max_length,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            if cfg.device == "cuda":
                input_ids = input_ids.to("cuda")
                attention_mask = attention_mask.to("cuda")

            B, L = input_ids.shape
            batch_scores = np.zeros(B, dtype=np.float64)

            for _ in range(N_SAMPLES):
                # Build mask: 15% of attended positions
                rand = torch.rand(B, L, device=input_ids.device)
                mask = (rand < MASK_RATE) & (attention_mask.bool())
                # Don't mask special tokens
                mask = mask & (input_ids != tokenizer.cls_token_id) & (input_ids != tokenizer.sep_token_id)
                if not mask.any():
                    continue

                masked_ids = input_ids.clone()
                masked_ids[mask] = mask_id
                logits = mlm(input_ids=masked_ids, attention_mask=attention_mask).logits
                log_probs = torch.log_softmax(logits.float(), dim=-1)

                # Log-prob at the original token (peakness indicator)
                orig_lp = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)
                # Log-prob at a random other token (baseline)
                random_tokens = torch.randint(
                    0, vocab_size, size=input_ids.shape, device=input_ids.device
                )
                rand_lp = log_probs.gather(-1, random_tokens.unsqueeze(-1)).squeeze(-1)

                # Gap on masked positions only
                gap = (orig_lp - rand_lp) * mask.float()
                per_sample_gap = gap.sum(dim=-1) / (mask.sum(dim=-1).clamp(min=1).float())
                batch_scores += per_sample_gap.cpu().numpy()

            batch_scores /= N_SAMPLES
            scores[start : start + len(chunk)] = batch_scores

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="FastDetectGPT",
        exp_id="exp_zs_02",
        score_fn=_fast_detect_gpt_score,
        cfg=cfg,
    )
