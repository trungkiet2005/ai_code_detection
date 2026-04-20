"""
[exp_zs_07] MIN-K%++ -- vocabulary-normalised minimum log-prob test,
            transplanted from pretraining-data detection.

THEORY HOOK:
  Zhang et al. "Min-K%++: Improved Baselines for Detecting Pre-training
  Data from Large Language Models" (ICLR 2025, arXiv:2404.02936).
  Core improvement over MIN-K%: normalise each token's log-prob by the
  vocab-wide (mean, std) at that position, THEN average the bottom 20%.
  Formally:
      z_t = (log p(x_t | x_<t) - mu_vocab_t) / sigma_vocab_t
      score(x) = mean over bottom 20% of {z_t}
  Claim: a token is training-data-like iff its log-prob is a LOCAL MAXIMUM
  along the vocabulary axis at position t, not just a high absolute value.

WHY IT TRANSPLANTS TO AI-CODE DETECTION:
  "Training-data-like" is approximately "produced by the same model that trained on that
  data". For a code LM (e.g. CodeBERT MLM), MGT code is effectively its
  own training data -- so MIN-K%++ gives a direct signal of "how much
  does this look like CodeBERT was trained to emit it".

ORTHOGONAL TO Fast-DetectGPT:
  Fast-DetectGPT measures CURVATURE (local variance under perturbation).
  MIN-K%++ measures SIGNATURE-AT-SPECIFIC-TOKENS (bottom-k outliers).
  Same underlying LM (CodeBERT) but different sufficient statistic.

COST:
  One MLM forward pass. No perturbation loop. ~2x faster than Fast-DetectGPT.
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

    logger.info(f"[ZS-07] Loading MLM {cfg.scorer_lm} ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    _mlm = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    _mlm.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _mlm = _mlm.to("cuda")
        if cfg.precision == "bf16":
            _mlm = _mlm.to(torch.bfloat16)
    return _mlm, _tokenizer


def _min_k_pp_score(codes: List[str], cfg: ZSConfig, k_percent: float = 20.0) -> np.ndarray:
    """Return per-sample Min-K%++ score. Higher = more AI-typical.

    For each sample:
      1. Forward once; logits[t] is the distribution over vocab at position t.
      2. Compute z_t = (log p(x_t) - mu_vocab_t) / sigma_vocab_t.
      3. Take bottom 20% of z_t (smallest z scores = tokens the model
         finds LESS peaked than expected). Mean.
      4. Return NEGATION (higher => less anomalous => more AI-typical).
    """
    import torch

    mlm, tokenizer = _get_mlm(cfg)
    pad_id = tokenizer.pad_token_id or 0
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
            logits = mlm(input_ids=input_ids, attention_mask=attn).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)              # (B, L, V)

            # Actual log-prob at each token position
            token_lp = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1)  # (B, L)
            # Vocab-level stats per (B, L) -- need mean and std over vocab dim
            # logits already softmaxed -> log_probs. For MIN-K%++ we need
            # (mu, sigma) over the SOFTMAX DISTRIBUTION at each position.
            mu = log_probs.mean(dim=-1)                # (B, L)
            sd = log_probs.std(dim=-1).clamp(min=1e-6)  # (B, L)
            z = (token_lp - mu) / sd                   # (B, L)

            # Mask padding
            mask = attn.bool()

            # Per-sample: take bottom k% of z_t among valid positions
            for i in range(len(chunk)):
                z_i = z[i][mask[i]].cpu().numpy()
                if len(z_i) == 0:
                    scores[start + i] = 0.0
                    continue
                # bottom k%
                n = len(z_i)
                k = max(1, int(n * k_percent / 100.0))
                partition = np.partition(z_i, k - 1)[:k]
                # Higher z_i = more peaked at the observed token; AI code
                # has HIGHER overall z, so its BOTTOM-20% is less negative
                # than human. Return mean directly -> higher = more AI.
                scores[start + i] = float(partition.mean())
    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="MinKPlusPlus",
        exp_id="exp_zs_07",
        score_fn=_min_k_pp_score,
        cfg=cfg,
    )
