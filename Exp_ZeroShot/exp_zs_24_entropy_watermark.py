"""
[exp_zs_24] EntropyWatermarkDetection -- zero-shot detector via cumulative
            entropy thresholding (generative-time signal).

THEORY HOOK (SOTA, arXiv:2504.12108, April 2025):
  Paper: "Entropy-Guided Watermarking for LLMs"

  Key insight: Rather than post-hoc detection, this measures CUMULATIVE ENTROPY
  of token generation trajectory. LLMs generate with lower entropy (optimized
  path); humans generate with higher entropy (exploration).

  Define cumulative entropy:
      E_cum(t) = sum_{i=1}^{t} H(token_i | prefix)
  where H is Shannon entropy of next-token distribution.

  AI code: LOW E_cum (narrow distribution at each step).
  Human code: HIGH E_cum (explores many branches, changes direction often).

WHY NOVEL:
  * PROCESS-LEVEL signal (not just final token sequence).
  * Captures GENERATION TRAJECTORY, not static features.
  * Orthogonal to all existing detectors (which analyze final code).

IMPLEMENTATION (cumulative entropy of token likelihood):
  1. For each code sample, tokenize.
  2. Use CodeBERT to compute P(next_token | prefix) at each position.
  3. Entropy H_t = -sum_v p_v log p_v.
  4. Cumulative entropy E_cum = sum_t H_t.
  5. Normalize by sequence length.
  6. Score = E_cum / len(tokens). Higher = human-like.

Cost: 1 forward pass per sample, compute entropy at each position. ~9 min H100.
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


def _entropy_watermark_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Cumulative entropy along generation trajectory."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info(f"[ZS-24] Loading encoder {cfg.scorer_lm}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    model = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    model.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        if cfg.precision == "bf16":
            model = model.to(torch.bfloat16)

    scores = np.zeros(len(codes), dtype=np.float64)
    bs = cfg.batch_size

    with torch.no_grad():
        for start in range(0, len(codes), bs):
            chunk = codes[start : start + bs]

            # Tokenize
            enc = tokenizer(
                chunk, max_length=cfg.scorer_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            if cfg.device == "cuda":
                input_ids = input_ids.to("cuda")
                attn = attn.to("cuda")

            # Get logits
            logits = model(input_ids=input_ids, attention_mask=attn).logits.float()
            probs = torch.softmax(logits, dim=-1)

            # Compute entropy at each position
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)  # (B, L)

            # Cumulative entropy per sample
            for i in range(len(chunk)):
                n = int(attn[i].sum().item())
                if n < 5:
                    scores[start + i] = 0.0
                    continue

                cum_entropy = entropy[i, :n].sum().item()
                norm_cum_entropy = cum_entropy / n
                scores[start + i] = float(norm_cum_entropy)

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="EntropyWatermarkDetection",
        exp_id="exp_zs_24",
        score_fn=_entropy_watermark_score,
        cfg=cfg,
    )
