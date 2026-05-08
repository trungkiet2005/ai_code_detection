"""
[exp_zs_09] LZ77Complexity -- Gzip-ratio zero-shot detector, no GPU.

THEORY HOOK:
  Jiang et al. ACL'23 "Low-Resource Text Classification: A Parameter-Free
  Classification Method with Compressors" + Rao et al. 2025 "LZ-Coder:
  Kolmogorov Surrogates Revisited" (arXiv:2507.02233).
  Solomonoff prior motivation: AI-generated code is closer to a LOW
  KOLMOGOROV COMPLEXITY sample than human code because the LM decoder
  is biased toward high-probability (= short-description-length) outputs
  under its own prior. Gzip's LZ-77 compressor approximates Kolmogorov
  complexity up to a constant, so:
      score(x) = len(gzip(x)) / len(x)
  AI code has LOWER bytes-per-char under gzip than human code of the
  same raw length.

WHY IT MATTERS FOR THE ORAL:
  - ZERO GPU. ZERO external models. Fully reproducible.
  - Floor baseline for the oral's comparison block: any proposed ZS
    detector that doesn't beat this floor isn't contributing.
  - Very fast: ~3 min for the full Droid test on a single CPU core.

LIMITATION:
  Short code snippets (<100 chars) have noisy gzip ratios due to the
  fixed 32KB sliding window. We cap at 4096 chars per sample (mirrors
  exp_zs_03 Ghostbuster's handling).
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

import gzip
from typing import List

import numpy as np

from _common import ZSConfig, logger
from _zs_runner import run_zs_oral


def _gzip_ratio(text: str) -> float:
    if not text:
        return 1.0
    raw = text.encode("utf-8")
    if len(raw) == 0:
        return 1.0
    compressed = gzip.compress(raw, compresslevel=6)
    return len(compressed) / len(raw)


def _lz77_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Return NEGATIVE gzip ratio so higher = more compressible = more AI.

    ratio in (0, 1] -- typical code is 0.3-0.5. AI-generated code (more
    repetitive / template-y) sits lower than human code.
    """
    cap = 4096  # chars; avoids pathological long snippets dominating the ratio
    out = np.zeros(len(codes), dtype=np.float64)
    for i, code in enumerate(codes):
        txt = code[:cap]
        ratio = _gzip_ratio(txt)
        # Higher score = more AI. AI has lower ratio (more compressible),
        # so return NEGATIVE ratio so higher = more AI.
        out[i] = -ratio
    return out


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3", device="cpu")  # CPU-only; gzip is single-threaded Python
    run_zs_oral(
        method_name="LZ77Complexity",
        exp_id="exp_zs_09",
        score_fn=_lz77_score,
        cfg=cfg,
    )
