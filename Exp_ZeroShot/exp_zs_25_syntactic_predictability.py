"""
[exp_zs_25] SyntacticPredictability -- zero-shot detector via POS/token-type
            n-gram entropy (STELA transfer to code).

THEORY HOOK (SOTA, arXiv:2510.13829, October 2025):
  Paper: "A Linguistics-Aware LLM Watermarking via Syntactic Predictability"

  Key insight: Instead of measuring token probability directly, measure
  SYNTACTIC PREDICTABILITY: entropy of next-token-type given previous
  syntax structure.

  For code: token types = {keyword, identifier, operator, literal, punctuation}.
  Define syntactic entropy:
      H_syn(t) = -sum_types p(type_t | type_{t-1}, type_{t-2}) log p(...)
  where p is the empirical distribution of next type given previous 2 types.

  AI code: LOW H_syn (follows predictable patterns: keyword -> identifier ->
  operator -> keyword).
  Human code: HIGH H_syn (irregular syntax: identifiers cluster, exceptions
  introduce unusual type sequences).

WHY NOVEL:
  * Extends Axis B (Spectral) to SYNTACTIC SCALES (not just frequencies).
  * Captures LINGUISTIC CONSTRAINTS on code structure.
  * Orthogonal to all existing (token/embedding/CFG/type-system detectors).

IMPLEMENTATION (token-type n-grams):
  1. Tokenize code with tokenizer.
  2. Map each token to a type: keyword, identifier, operator, literal, punct.
  3. Build n-gram distribution of token types (n=3).
  4. Compute entropy H of this distribution.
  5. Score = H. Higher = more irregular (human-like).

Cost: pure regex/tokenization, O(L). ~6 min on H100 (no GPU).
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


# Keywords for code
KEYWORDS = set(['if', 'else', 'elif', 'for', 'while', 'def', 'class', 'return',
                'import', 'from', 'try', 'except', 'with', 'as', 'and', 'or',
                'not', 'in', 'is', 'None', 'True', 'False', 'pass', 'break',
                'continue', 'yield', 'lambda', 'assert', 'del', 'raise', 'finally'])
OPERATORS = set(['+', '-', '*', '/', '//', '%', '**', '==', '!=', '<', '>', '<=',
                 '>=', '=', '+=', '-=', '*=', '/=', '&', '|', '^', '~', '<<', '>>'])
PUNCTUATION = set(['(', ')', '[', ']', '{', '}', ',', '.', ':', ';', '@', '`'])


def _token_type(token: str) -> str:
    """Classify token into type."""
    token_lower = token.lower()
    if token_lower in KEYWORDS:
        return "keyword"
    elif token in OPERATORS:
        return "operator"
    elif token in PUNCTUATION:
        return "punctuation"
    elif token[0].isdigit():
        return "literal"
    elif token[0] in ('"', "'"):
        return "literal"
    else:
        return "identifier"


def _syntactic_predictability_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Syntactic entropy: higher = more irregular (human-like)."""
    scores = np.zeros(len(codes), dtype=np.float64)

    for i, code in enumerate(codes):
        if not code or not code.strip():
            scores[i] = 0.0
            continue

        # Tokenize (simple regex split)
        tokens = re.findall(r'\w+|[+\-*/%=<>!&|^~()[\]{},.;:@`]', code)
        if len(tokens) < 5:
            scores[i] = 0.0
            continue

        # Get token types
        types = [_token_type(t) for t in tokens]

        # Build 3-gram distribution
        trigrams = {}
        for j in range(len(types) - 2):
            trigram = (types[j], types[j+1], types[j+2])
            trigrams[trigram] = trigrams.get(trigram, 0) + 1

        # Entropy of trigram distribution
        total = sum(trigrams.values())
        probs = np.array(list(trigrams.values())) / total
        entropy = float(-np.sum(probs * np.log(probs + 1e-10)))

        # Normalize by max entropy (log of number of possible trigrams)
        max_entropy = np.log(len(trigrams) + 1)
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0.0

        scores[i] = float(norm_entropy)

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="SyntacticPredictability",
        exp_id="exp_zs_25",
        score_fn=_syntactic_predictability_score,
        cfg=cfg,
    )
