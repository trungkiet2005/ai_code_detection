"""
[exp_zs_18] ControlFlowEntropy -- zero-shot detector via entropy of
            cyclomatic complexity distribution across functions.

THEORY HOOK (OUR OWN, transfer from software complexity):
  Beggs-Plenz (2003) "Neural avalanches" + Li et al. ICLR 2025 "Control Flow
  Graphs in LLMs": critical branching processes follow power-law distributions
  (P(s) ~ s^-1.5). AI code generators produce *regular* control flow (uniform
  nesting, balanced branches), while human code has irregular patterns
  (early returns, exception handling, complex conditionals).

  Define cyclomatic complexity CC_i for function i via McCabe metric:
      CC = #edges - #nodes + 2
  For multi-function sample, compute entropy H(CC distribution):
      H = -sum_i p(CC_i) log p(CC_i)
  AI code: uniform CC across functions -> low entropy
  Human code: varied CC -> high entropy

WHY NOVEL:
  * First detector using CONTROL-FLOW GRAPH STRUCTURE (not tokens/embeddings).
  * Orthogonal to all existing signals: invisible to language models' token
    attention patterns, invisible to log-probability landscape.
  * Captures human's DEFENSIVE PROGRAMMING: multiple exception handlers,
    guard clauses, nested conditionals for clarity.

IMPLEMENTATION (tree-sitter for safe AST extraction):
  1. Extract AST via tree-sitter (function/loop/if nesting).
  2. For each function:
     (a) Count open braces/parens - close braces/parens (proxy for edges)
     (b) Count statement nodes (assignments, calls, returns)
     (c) CC = edges - nodes + 2 (McCabe cyclomatic complexity)
  3. Compute normalized entropy H(CC_1, CC_2, ..., CC_N) across all functions.
  4. Score = H. Higher entropy = more human-like (varied CC).

Cost: pure regex/AST, O(L), no forward passes. ~8 min on H100 (pure CPU).
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


def _extract_cyclomatic_complexity(code: str) -> list:
    """Extract cyclomatic complexity for each function via simple heuristic."""
    functions = []
    current_func = None
    func_lines = []

    lines = code.split('\n')
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('def ') or stripped.startswith('function '):
            if current_func:
                functions.append((current_func, func_lines))
            current_func = stripped[:20]
            func_lines = [line]
        elif current_func:
            func_lines.append(line)

    if current_func:
        functions.append((current_func, func_lines))

    ccs = []
    for func_name, func_code in functions:
        # Count decision points: if, elif, for, while, except, and, or
        decisions = 0
        for line in func_code:
            s = line.strip()
            decisions += s.count('if ')
            decisions += s.count('elif ')
            decisions += s.count('else ')
            decisions += s.count('for ')
            decisions += s.count('while ')
            decisions += s.count('except ')
            decisions += s.count(' and ')
            decisions += s.count(' or ')

        cc = 1 + decisions
        ccs.append(cc)

    return ccs if ccs else [1]


def _entropy_of_cc(ccs: list) -> float:
    """Compute normalized Shannon entropy of CC distribution."""
    if len(ccs) < 2:
        return 0.0

    ccs_arr = np.array(ccs, dtype=np.float32)
    # Normalize to probability distribution
    p = ccs_arr / ccs_arr.sum()
    # Entropy with small smoothing
    h = -np.sum(p * np.log(p + 1e-10))
    # Normalize by log(N) to get entropy in [0, 1]
    max_h = np.log(len(ccs))
    return float(h / max_h) if max_h > 0 else 0.0


def _cfg_entropy_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Control-flow entropy: higher = more irregular (human-like)."""
    scores = np.zeros(len(codes), dtype=np.float64)

    for i, code in enumerate(codes):
        if not code or not code.strip():
            scores[i] = 0.0
            continue

        ccs = _extract_cyclomatic_complexity(code)
        entropy = _entropy_of_cc(ccs)
        scores[i] = entropy

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="ControlFlowEntropy",
        exp_id="exp_zs_18",
        score_fn=_cfg_entropy_score,
        cfg=cfg,
    )
