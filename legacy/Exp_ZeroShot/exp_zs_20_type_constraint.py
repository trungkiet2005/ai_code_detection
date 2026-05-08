"""
[exp_zs_20] TypeConstraintDeviation -- zero-shot detector via type-system
            slack: defensive programming indicators (Any, cast, guards).

THEORY HOOK (OUR OWN, transfer from type-system semantics):
  Type systems (Python typing, TypeScript strict mode) enforce structural
  constraints. Novel insight: AI code *over-satisfies* type constraints
  (minimal casting, strict annotations); human code uses Any, casts,
  isinstance() guards indicating *defensive programming* (trading strictness
  for robustness).

  Define type-slack ratio:
      ratio = (count_Any + count_cast + count_guard) / (total_typed_exprs + 1)
  where count_Any = explicit `Any` annotations, count_cast = `cast()` calls,
  count_guard = `isinstance()` / `hasattr()` / `try/except` defense.
  AI code: low ratio (strict, LLM follows type checker rules).
  Human code: high ratio (defensive, exploits type system escape hatches).

WHY NOVEL:
  * First detector using TYPE-SYSTEM SEMANTICS (not tokens/embeddings/CFG).
  * Captures DEFENSIVE PROGRAMMING: humans guard against edge cases; AI
    assumes happy path.
  * Orthogonal to all existing signals: invisible to probability, embeddings,
    control flow, structure.

IMPLEMENTATION (pure AST regex, no external type checker):
  1. Count annotations:
     (a) `Any` in type hints (e.g., `: Any` or `-> Any`).
     (b) `cast()` calls (e.g., `cast(int, x)`).
     (c) Guard clauses: `isinstance()`, `hasattr()`, `try/except`.
  2. Compute ratio: type_slack = (Any + cast + guard) / (total_annotations + 1).
  3. Score = type_slack. Higher = human-like.
  4. Optional: run `mypy --lenient` and count warnings as secondary signal.

Cost: pure regex + optional mypy subprocess (~1s per sample). ~9 min on H100.
"""
from __future__ import annotations

import os
import re
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
import tempfile

import numpy as np

from _common import ZSConfig, logger
from _zs_runner import run_zs_oral


def _count_type_slack(code: str) -> float:
    """Extract type-slack ratio from code (multi-language robust)."""
    # Python-ish explicit type slack
    count_any = len(re.findall(r':\s*Any\b|->.*Any\b|:\s*object\b', code))
    count_cast = len(re.findall(r'\bcast\s*\(|\bType\s*\[', code))

    # Defensive guards (common across languages: py / js / ts / java / go)
    count_isinstance = len(re.findall(r'\bisinstance\s*\(|\binstanceof\b|\btypeof\b', code))
    count_hasattr = len(re.findall(r'\bhasattr\s*\(|\bhasOwnProperty\b', code))
    count_try = len(re.findall(r'\btry\s*[:{\n]|\bcatch\s*[({]|\brescue\b', code))
    count_null_guard = len(re.findall(r'==\s*(None|null|nil|undefined)\b|!=\s*(None|null|nil|undefined)\b|\bis\s+None\b|\bis\s+not\s+None\b', code))

    count_guard = count_isinstance + count_hasattr + count_try + count_null_guard

    # Total annotation-like pattern (language-agnostic proxy)
    count_typed = len(re.findall(r':\s*[A-Za-z_][\w\[\],\s]*', code))
    # Also proxy via function signatures across languages
    count_typed += len(re.findall(r'\w+\s*:\s*\w+|\bfunction\s+\w+|\bdef\s+\w+|\bfunc\s+\w+', code))

    n_lines = max(1, len(code.split('\n')))

    if count_typed == 0:
        # No type annotations — fall back to guard density per line
        ratio = float(count_guard) / n_lines
    else:
        ratio = (count_any + count_cast + count_guard) / (count_typed + 1)

    return float(np.clip(ratio, 0.0, 1.0))


def _type_constraint_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Type-slack metric: higher = more defensive (human-like).

    Robust to multi-language benchmarks (Droid spans 9 langs). Adds a tiny
    length-based tiebreaker so that all-zero slack distributions don't collapse
    threshold calibration to τ=0 (the degenerate dev-scoring observed 2026-04-20).
    """
    scores = np.zeros(len(codes), dtype=np.float64)

    for i, code in enumerate(codes):
        if not code or not code.strip():
            scores[i] = 0.0
            continue

        slack = _count_type_slack(code)
        # Length-based tiebreaker in [0, 0.05] prevents degenerate τ=0 when
        # a large fraction of samples has slack=0 (common for JS/C/Go/Rust in Droid).
        tiebreak = 0.05 * np.tanh(len(code) / 1000.0)
        scores[i] = float(slack + tiebreak)

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="TypeConstraintDeviation",
        exp_id="exp_zs_20",
        score_fn=_type_constraint_score,
        cfg=cfg,
    )
