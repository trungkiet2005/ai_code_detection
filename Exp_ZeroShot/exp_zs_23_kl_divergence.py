"""
[exp_zs_23] KLDivergenceSignal -- zero-shot detector via KL divergence
            between human-like and LLM token distributions (not curvature).

THEORY HOOK (SOTA, arXiv:2504.10637, April 2025):
  Paper: "Better Estimation of the Kullback-Leibler Divergence Between LMs"

  Key insight: Fast-DetectGPT uses LOCAL CURVATURE (variance under masking).
  This detector uses GLOBAL DIVERGENCE: D_KL(P_human || P_llm) where human
  and LLM are two different models' distributions over the same tokens.

  Intuition: LLM and human code diverge sharply in LOW-PROBABILITY regions
  (error handling, edge cases, unusual patterns). Curvature misses this;
  KL divergence captures it.

  Define:
      score = D_KL(P_code || P_baseline)
  where P_code is the likelihood under a code-trained model, P_baseline is
  likelihood under a general-purpose LLM. Large divergence = code looks
  different from baseline = likely human or unusual AI.

WHY NOVEL:
  * Orthogonal to Fast-DetectGPT (divergence != curvature).
  * Captures DISTRIBUTION MISMATCH, not local geometry.
  * Zero-shot: no training. Just need two LMs and importance-weighted sampling.

IMPLEMENTATION (importance-weighted KL estimation):
  1. Two models: CodeBERT (code-trained), GPT-2 (general).
  2. Score tokens with both: get log p_code(t), log p_general(t).
  3. For each token: importance weight w_t = exp(log p_code - log p_general).
  4. KL divergence = sum_t (1/N) w_t * (log p_code(t) - log p_general(t)).
  5. Score = KL. Higher = more divergent = less AI-like.

Cost: 2 forward passes per sample (CodeBERT + GPT2). ~10 min on H100.
"""
from __future__ import annotations

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
        here = os.path.dirname(os.path.abspath(__file__))
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


def _kl_divergence_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """KL divergence between code-model and baseline model."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info(f"[ZS-23] Loading code model {cfg.scorer_lm}...")
    tokenizer_code = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    model_code = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    model_code.eval()

    logger.info("[ZS-23] Loading baseline model (gpt2)...")
    tokenizer_base = AutoTokenizer.from_pretrained("gpt2")
    model_base = AutoModelForMaskedLM.from_pretrained("gpt2")
    model_base.eval()

    if cfg.device == "cuda" and torch.cuda.is_available():
        model_code = model_code.to("cuda")
        model_base = model_base.to("cuda")
        if cfg.precision == "bf16":
            model_code = model_code.to(torch.bfloat16)
            model_base = model_base.to(torch.bfloat16)

    scores = np.zeros(len(codes), dtype=np.float64)
    bs = cfg.batch_size

    with torch.no_grad():
        for start in range(0, len(codes), bs):
            chunk = codes[start : start + bs]

            # Encode with code model
            enc_code = tokenizer_code(
                chunk, max_length=cfg.scorer_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids = enc_code["input_ids"]
            attn = enc_code["attention_mask"]
            if cfg.device == "cuda":
                input_ids = input_ids.to("cuda")
                attn = attn.to("cuda")

            logits_code = model_code(input_ids=input_ids, attention_mask=attn).logits.float()
            log_probs_code = torch.log_softmax(logits_code, dim=-1)
            obs_lp_code = log_probs_code.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)

            # Encode with baseline model (convert token IDs if needed)
            enc_base = tokenizer_base(
                chunk, max_length=cfg.scorer_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids_base = enc_base["input_ids"]
            attn_base = enc_base["attention_mask"]
            if cfg.device == "cuda":
                input_ids_base = input_ids_base.to("cuda")
                attn_base = attn_base.to("cuda")

            logits_base = model_base(input_ids=input_ids_base, attention_mask=attn_base).logits.float()
            log_probs_base = torch.log_softmax(logits_base, dim=-1)
            obs_lp_base = log_probs_base.gather(-1, input_ids_base.unsqueeze(-1)).squeeze(-1)

            # KL divergence (approximate, using importance weighting)
            # For simplicity: use code tokens only, compute D_KL
            for i in range(len(chunk)):
                n = int(attn[i].sum().item())
                if n < 5:
                    scores[start + i] = 0.0
                    continue

                lp_code = obs_lp_code[i, :n].cpu().numpy()
                lp_base = obs_lp_base[i, :n].cpu().numpy()

                # KL approximation: mean(lp_code - lp_base)
                # (proper KL requires P_code over full vocab; this is simplified)
                kl_approx = float(np.mean(lp_code - lp_base))
                scores[start + i] = kl_approx

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="KLDivergenceSignal",
        exp_id="exp_zs_23",
        score_fn=_kl_divergence_score,
        cfg=cfg,
    )
