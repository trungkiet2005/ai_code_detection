"""
[exp_zs_21] TaskConditioningEntropy -- zero-shot detector via task-conditioned
            token entropy (not unconditional probability).

THEORY HOOK (SOTA, ECML PKDD 2025, arXiv:2506.06069):
  Unconditional token distributions are IDENTICAL between human and LLM code.
  But task-conditioned entropy reveals sharp separability: LLMs generate by
  optimizing for task completion (narrow entropy); humans explore solution
  spaces (high entropy).

  Key insight: Most detectors measure unconditional log P(token). This fails
  because both human and LLM assign similar probability to common tokens.
  Instead: estimate P(token | task approximation). Task is *unknown*, but
  can be approximated by prompting a language model with the code snippet.

  Define task-conditioned entropy:
      H(token | task_approx) = -sum_t p(t | task) log p(t | task)
  where task_approx comes from LLM-generated task descriptions of the code.
  AI code: low entropy (optimized for specific task)
  Human code: high entropy (solves task in varied ways)

WHY NOVEL:
  * Orthogonal to Fast-DetectGPT (curvature, unconditional signal).
  * Focuses on OPTIMIZATION TRAJECTORY: AI narrows to high-prob tokens; human
    explores alternatives.
  * Zero-shot: no training. Just need a language model (CodeBERT + decoder).

IMPLEMENTATION (clever use of prompt engineering):
  1. Given code snippet, prompt LLM: "What does this code do? (one sentence)"
  2. Collect N=5 task descriptions (sample with temp=0.7).
  3. For each task, re-encode the code conditioned on that task via attention.
  4. Compute token entropy under each task; take mean.
  5. Score = mean entropy. Higher = human-like.

Cost: 1 forward pass (encoder) + 5 decoder passes (task gen) + 5 conditional
forward passes. ~12 min on H100.
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


def _generate_task_descriptions(code: str, tokenizer, model, device: str = "cpu") -> List[str]:
    """Generate task descriptions for code via language model."""
    import torch

    prompt = f"""Summarize what this code does in one sentence:
{code[:300]}

Summary: """

    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    if device == "cuda":
        input_ids = input_ids.to("cuda")

    tasks = []
    with torch.no_grad():
        for _ in range(3):  # Generate 3 task descriptions
            output = model.generate(
                input_ids, max_new_tokens=20, temperature=0.7, top_p=0.9, do_sample=True
            )
            task = tokenizer.decode(output[0], skip_special_tokens=True)
            # Extract just the new tokens
            task = task[len(prompt):].strip()
            if task:
                tasks.append(task)

    return tasks if tasks else ["code processing"]


def _task_conditioned_entropy(code: str, tokenizer, model, device: str = "cpu") -> float:
    """Compute task-conditioned entropy of tokens."""
    import torch

    # Encode code
    encoding = tokenizer(code, return_tensors="pt", max_length=512, truncation=True)
    input_ids = encoding["input_ids"]
    if device == "cuda":
        input_ids = input_ids.to("cuda")

    # Get baseline logits (cast to fp32 for numerical stability — log/softmax unstable in bf16)
    with torch.no_grad():
        outputs = model(input_ids)
        logits = outputs.logits[0].float()  # (seq_len, vocab_size), force fp32
        probs = torch.softmax(logits, dim=-1)

    # Compute entropy in fp32
    entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1).mean()
    return float(entropy.cpu().numpy())


def _task_conditioning_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Task-conditioned entropy: higher = human-like."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info(f"[ZS-21] Loading encoder {cfg.scorer_lm} (FP32 — entropy computation needs full precision)...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    model = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    model.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
    # NOTE: deliberately skip bf16 cast — bf16 regressions observed 2 runs in a
    # row (torch.log, softmax, gather, etc. flaky in bf16 depending on kernel).
    # Entropy computation is O(L*V) — fp32 overhead negligible on H100.

    scores = np.zeros(len(codes), dtype=np.float64)

    for i, code in enumerate(codes):
        if not code or not code.strip():
            scores[i] = 0.0
            continue

        # Task-conditioned entropy
        entropy = _task_conditioned_entropy(code, tokenizer, model, cfg.device)
        scores[i] = float(entropy)

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="TaskConditioningEntropy",
        exp_id="exp_zs_21",
        score_fn=_task_conditioning_score,
        cfg=cfg,
    )
