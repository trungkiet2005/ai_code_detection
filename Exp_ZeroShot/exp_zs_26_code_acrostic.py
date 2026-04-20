"""
[exp_zs_26] CodeAcrosticStructure -- zero-shot detector via comment-structure
            entropy and inter-comment semantic coherence.

THEORY HOOK (SOTA, arXiv:2512.14753, December 2025):
  Paper: "Code Acrostic: Robust Watermarking for Code Generation"

  Key insight: Comments are typically written by HUMANS. Even when code is
  LLM-generated, humans typically add/edit comments. This creates a META-
  LINGUISTIC signal: comment structure + semantic flow.

  Define metrics:
  1. Comment density: ratio of comment lines to total lines.
  2. Inter-comment semantic coherence: cosine similarity between consecutive
     comments' embeddings (low coherence = human context-switching).
  3. Comment entropy: diversity of comment lengths and vocabulary.

  AI code: comments are sparse, generic ("fix bug", "helper function").
  Human code: comments are dense, contextual, show intent evolution.

WHY NOVEL:
  * META-LINGUISTIC SIGNAL (comments as author fingerprint).
  * Orthogonal to all code-content detectors (focuses on documentation).
  * Opens Axis G (Data distribution side): comments as external data layer.

IMPLEMENTATION (comment extraction + semantic analysis):
  1. Extract all comments via regex.
  2. Compute comment density: #comment_lines / #total_lines.
  3. Encode comments with CodeBERT; compute pairwise cosine similarities.
  4. Coherence = mean(similarities) of consecutive comments.
  5. Comment entropy = Shannon entropy of comment lengths + vocabulary.
  6. Score = (density + coherence + entropy) / 3. Higher = human-like.

Cost: pure regex + one forward pass (CodeBERT on concatenated comments).
~7 min on H100.
"""
from __future__ import annotations

import os
import re
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

from _common import ZSConfig, logger
from _zs_runner import run_zs_oral


def _extract_comments(code: str) -> List[str]:
    """Extract all comments from code."""
    comments = []
    lines = code.split('\n')
    for line in lines:
        # Python-style comments
        if '#' in line:
            comment = line.split('#', 1)[1].strip()
            if comment:
                comments.append(comment)
    return comments


def _comment_density(code: str) -> float:
    """Ratio of comment lines to total lines."""
    lines = code.split('\n')
    comment_lines = sum(1 for line in lines if '#' in line and line.strip().startswith('#'))
    total_lines = len([l for l in lines if l.strip()])
    if total_lines == 0:
        return 0.0
    return float(comment_lines / total_lines)


def _comment_entropy(comments: List[str]) -> float:
    """Entropy of comment lengths + word diversity."""
    if len(comments) < 2:
        return 0.0

    lengths = np.array([len(c.split()) for c in comments])
    if len(lengths) < 2:
        return 0.0

    # Discretize lengths into bins
    bins = np.histogram_bin_edges(lengths, bins=5)
    counts, _ = np.histogram(lengths, bins=bins)
    probs = counts / counts.sum()
    entropy = float(-np.sum(probs[probs > 0] * np.log(probs[probs > 0] + 1e-10)))

    return entropy / np.log(len(bins))


def _comment_coherence(comments: List[str], tokenizer, model, device: str = "cpu") -> float:
    """Semantic coherence between consecutive comments."""
    if len(comments) < 2:
        return 0.0

    import torch

    # Encode comments
    encoded = tokenizer(
        comments, max_length=128, padding="max_length", truncation=True, return_tensors="pt"
    )
    if device == "cuda":
        encoded = {k: v.to("cuda") for k, v in encoded.items()}

    with torch.no_grad():
        outputs = model(**encoded)
        embeddings = outputs.logits[:, 0, :].cpu()  # Use CLS token

    # Compute cosine similarity between consecutive comments
    sims = []
    for i in range(len(embeddings) - 1):
        e1 = embeddings[i]
        e2 = embeddings[i+1]
        sim = float(torch.nn.functional.cosine_similarity(e1.unsqueeze(0), e2.unsqueeze(0)).item())
        sims.append(sim)

    return float(np.mean(sims)) if sims else 0.0


def _code_acrostic_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Comment-structure + semantic coherence."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info(f"[ZS-26] Loading encoder {cfg.scorer_lm}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    model = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
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

        # Extract features
        comments = _extract_comments(code)
        density = _comment_density(code)
        entropy = _comment_entropy(comments)

        # Coherence (requires embedding)
        if len(comments) > 1:
            coherence = _comment_coherence(comments, tokenizer, model, cfg.device)
        else:
            coherence = 0.0

        # Combined score (normalized to [0, 1])
        score = (density + entropy + coherence) / 3.0
        scores[i] = float(np.clip(score, 0.0, 1.0))

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="CodeAcrosticStructure",
        exp_id="exp_zs_26",
        score_fn=_code_acrostic_score,
        cfg=cfg,
    )
