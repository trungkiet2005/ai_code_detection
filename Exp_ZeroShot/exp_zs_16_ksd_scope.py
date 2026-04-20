"""
[exp_zs_16] KSDScope -- zero-shot detector via Kernelised Stein
            Discrepancy over variable-scoping edges in the code.

THEORY HOOK (OUR OWN, Stein + variable-scoping):
  Gretton et al. "Kernelized Stein Discrepancy" + Liu-Lee-Jordan 2016:
      KSD(p, q) = sup_{f in RKHS} E_q[ (score_p - score_q) f ]
  where score_p = grad log p. KSD is a provably consistent goodness-of-
  fit test and equals zero iff the sample comes from the reference p.

OUR NOVEL CONSTRUCTION:
  For AI-code detection, we define a SCOPE KERNEL over (definition,
  first-use) edges in the code:
      k((d, u), (d', u')) = exp( -((d - d')^2 + (u - u')^2) / (2 * sigma^2) )
  The null distribution p is the LM's implicit distribution over scope
  graphs (human-written code deviates from it).
  KSD(p, empirical) approximates the expected Stein operator evaluated
  on the observed scope edges; large KSD => sample does NOT match the
  LM's scope-graph distribution => human.
  Small KSD => sample matches the LM's scope preferences => AI.

WHY NOVEL:
  * Existing ZS detectors treat code as a 1-D token sequence. NOBODY has
    constructed a Stein operator over SCOPE GRAPHS of code.
  * Structural mismatch is invisible to log-prob-only detectors:
    Fast-DetectGPT, DC-PDD, EnergyScore all read token logits; they
    cannot see that a human-written sample's FIRST-USE-AFTER-DEF pattern
    is different from the LM's training distribution.

IMPLEMENTATION (regex-based scope extraction, no external parser):
  1. Extract definition-first-use edges via regex on the raw code:
        def_pattern  = r'\\b([a-zA-Z_][a-zA-Z0-9_]*)\\s*='     (assignment)
        use_pattern  = r'\\b([a-zA-Z_][a-zA-Z0-9_]*)'
     For each variable name, record (def_line, first_use_line).
  2. Represent the scope graph as a point cloud E = {(d_i, u_i)} in R^2.
  3. KSD over this point cloud using a GAUSSIAN KERNEL. Since we don't
     have a true LM score over scope graphs, we approximate via the
     sample-mean kernel statistic:
        stat = (1 / |E|^2) * sum_{i, j} k(e_i, e_j)
     High stat = edges cluster = regular, LM-like
     Low stat = edges spread = irregular, human-like
  4. score = -stat  (higher = more AI; stat is higher for AI).

Cost: regex parsing + O(|E|^2) Gaussian kernel per sample. ~10 min on
H100 (GPU unused; CPU-parallel if Kaggle has spare cores).
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


_ASSIGN_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]{1,})\s*(?:=|:=|->)")
_USE_RE = re.compile(r"\b([a-zA-Z_][a-zA-Z0-9_]{1,})\b")


def _extract_scope_edges(code: str):
    """Return list of (def_line, first_use_line) edges over variable names.
    Lines are 1-indexed integers; definition = first assignment line,
    first_use = first line where the name appears OTHER than the def line.
    """
    defs = {}      # name -> def_line
    first_uses = {}   # name -> first_use_line
    lines = code.split("\n")
    for lineno, line in enumerate(lines, start=1):
        # Skip lines that are mostly comments (quick heuristic)
        stripped = line.strip()
        if stripped.startswith(("#", "//", "/*", "*")):
            continue
        # Find assignment targets
        for m in _ASSIGN_RE.finditer(line):
            name = m.group(1)
            if name not in defs:
                defs[name] = lineno
        # Find uses
        for m in _USE_RE.finditer(line):
            name = m.group(1)
            if name in defs and lineno != defs[name] and name not in first_uses:
                first_uses[name] = lineno

    edges = []
    for name, def_line in defs.items():
        if name in first_uses:
            edges.append((def_line, first_uses[name]))
    return edges


def _kernel_stat(edges: list, sigma: float = 5.0) -> float:
    """Mean Gaussian-kernel over edge pairs.
    stat = (1/|E|^2) sum_{i,j} exp(-((d_i-d_j)^2 + (u_i-u_j)^2) / (2 sigma^2))
    Higher = edges cluster = more regular = more AI-like.
    """
    if len(edges) < 2:
        return 0.0
    E = np.asarray(edges, dtype=np.float64)       # (|E|, 2)
    d = np.linalg.norm(E[:, None, :] - E[None, :, :], axis=-1) ** 2
    K = np.exp(-d / (2.0 * sigma ** 2))
    # Exclude diagonal to avoid self-similarity dominating
    n = K.shape[0]
    stat = (K.sum() - np.trace(K)) / max(n * (n - 1), 1)
    return float(stat)


def _ksd_scope_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Per-sample score = kernel-statistic over scope-graph edges.
    Higher = more clustered edges = more AI-typical.
    """
    scores = np.zeros(len(codes), dtype=np.float64)
    for i, code in enumerate(codes):
        if not code or not code.strip():
            scores[i] = 0.0
            continue
        edges = _extract_scope_edges(code)
        stat = _kernel_stat(edges)
        scores[i] = stat
    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3", device="cpu")  # CPU-only; no GPU work
    run_zs_oral(
        method_name="KSDScope",
        exp_id="exp_zs_16",
        score_fn=_ksd_scope_score,
        cfg=cfg,
    )
