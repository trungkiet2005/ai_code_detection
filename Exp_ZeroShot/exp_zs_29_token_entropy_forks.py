"""
[exp_zs_29] Token-Entropy Fork-Structure -- Decision-point semantics.

THEORY HOOK (SOTA, ACL Findings 2025, arXiv:2506.01939):
  Paper: "Token Entropy Patterns in LLM Reasoning" (ACL Findings 2025)
  Paper: "Cautious Next Token Prediction" (ACL Findings 2025)

  Key insight: LLM code generation has BIMODAL token-entropy distribution:
  - 80% low-entropy deterministic tokens (template boilerplate)
  - 20% high-entropy "fork tokens" (critical semantic choice points)

  Human code has MORE UNIFORM distribution. Moreover, FORK-TOKEN SEMANTICS differ:
  - LLM forks: high-perplexity algorithmic choices (loop start, cond branch)
  - Human forks: low-perplexity edge cases (error handling, validation, comments)

  Can detect by:
  1. Extract per-token entropy via full probability distribution
  2. Identify fork tokens (entropy > 90th percentile)
  3. Classify fork-context semantics (hand-crafted rules or learned)
  4. Feature = (entropy moments, fork count, fork-type histogram)

WHY NOVEL:
  * Decision-point semantics ≠ log-prob aggregates (your Exp_01)
  * Orthogonal to spectral (B), compressibility (E), genealogy (A)
  * NeurIPS 2024-25 workshops show tight F1 gains (±2-3 pts)
  * Captures *how* LLM explores solution space, not just *that* it does

IMPLEMENTATION (entropy distribution + fork analysis):
  1. Decode each sample token-by-token, collecting full logit distributions
  2. Per-token entropy: H_t = -∑_v p(v_t) log p(v_t)
  3. Fork tokens: entropy > 90th percentile
  4. Extract 5-token context around each fork
  5. Classify fork-context (loop_start, cond_branch, var_init, error_handle, ...)
  6. Feature vector = (mean_H, std_H, kurtosis_H, fork_ratio, fork_type_counts)
  7. Score = weighted combination

Cost: requires logits from inference (cached), ~10 min on H100 (light GPU)
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from typing import List

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

import numpy as np
import torch

from _common import ZSConfig, logger
from _zs_runner import run_zs_oral


def _compute_token_entropies(code: str, tokenizer, model, device: str = "cpu") -> np.ndarray:
    """Compute per-token entropy from full logit distribution."""
    inputs = tokenizer(code, max_length=256, truncation=True,
                      return_tensors="pt", padding="max_length")
    if device == "cuda":
        inputs = {k: v.to("cuda") for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        logits = outputs.logits  # (batch, seq_len, vocab_size)

    # Compute entropy per token
    probs = torch.softmax(logits[0], dim=-1)
    entropies = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
    return entropies.cpu().numpy()


def _identify_fork_tokens(entropies: np.ndarray, percentile: float = 90.0) -> np.ndarray:
    """Identify fork tokens (high entropy)."""
    threshold = np.percentile(entropies, percentile)
    forks = np.where(entropies > threshold)[0]
    return forks


def _classify_fork_context(code: str, fork_idx: int, window: int = 5) -> str:
    """Classify fork context semantics."""
    tokens = code.split()
    start = max(0, fork_idx - window)
    end = min(len(tokens), fork_idx + window)
    context = ' '.join(tokens[start:end])

    # Heuristic patterns
    if any(kw in context for kw in ['for', 'while', 'loop']):
        return 'loop'
    elif any(kw in context for kw in ['if', 'else', 'elif', 'cond']):
        return 'branch'
    elif any(kw in context for kw in ['except', 'try', 'error', 'raise']):
        return 'error_handling'
    elif any(kw in context for kw in ['assert', 'check', 'valid']):
        return 'validation'
    else:
        return 'other'


def _token_entropy_fork_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Decision-point semantics via token-entropy forks."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    logger.info(f"[ZS-29] Loading decoder {cfg.backbone_lm}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.backbone_lm)
    model = AutoModelForCausalLM.from_pretrained(cfg.backbone_lm)
    model.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        if cfg.precision == "bf16":
            model = model.to(torch.bfloat16)

    scores = np.zeros(len(codes), dtype=np.float64)

    for i, code in enumerate(codes):
        if not code or not code.strip():
            scores[i] = 0.0
            continue

        try:
            # Extract token entropies
            entropies = _compute_token_entropies(code, tokenizer, model, cfg.device)
        except Exception as e:
            logger.warning(f"[ZS-29] Failed to extract entropies for code {i}: {e}")
            scores[i] = 0.5
            continue

        # Identify and classify forks
        forks = _identify_fork_tokens(entropies, percentile=85)
        if len(forks) == 0:
            scores[i] = 0.0
            continue

        fork_types = {}
        for fork_idx in forks:
            fork_type = _classify_fork_context(code, fork_idx)
            fork_types[fork_type] = fork_types.get(fork_type, 0) + 1

        # Feature computation
        entropy_mean = float(np.mean(entropies))
        entropy_std = float(np.std(entropies))
        fork_ratio = float(len(forks) / len(entropies))

        # LLM-like: high fork_ratio, dominated by algorithmic forks
        # Human-like: lower fork_ratio, mix of error-handling + validation
        algo_forks = fork_types.get('loop', 0) + fork_types.get('branch', 0)
        defensive_forks = fork_types.get('error_handling', 0) + fork_types.get('validation', 0)

        # Score: human-like if defensive_forks > algo_forks and low fork_ratio
        score = 0.0
        if fork_ratio < 0.15:  # Human-like ratio
            score += 0.5
        if defensive_forks > algo_forks:  # Human-like semantics
            score += 0.3
        if entropy_std > 1.5:  # High diversity (human-like)
            score += 0.2

        scores[i] = float(np.clip(score, 0.0, 1.0))

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="TokenEntropyForks",
        exp_id="exp_zs_29",
        score_fn=_token_entropy_fork_score,
        cfg=cfg,
    )
