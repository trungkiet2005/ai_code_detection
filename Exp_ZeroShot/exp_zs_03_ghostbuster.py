"""
[exp_zs_03] Ghostbuster-Code -- token-statistics committee (CPU-only).

THEORY HOOK:
  Verma et al. EMNLP 2023 "Ghostbuster" (arXiv:2305.15047): human writing
  has higher BURSTINESS (variance of inter-word-gap distribution) and
  more variable TYPE-TOKEN-RATIO than machine text. Human code likewise
  has:
    - higher identifier diversity (TTR)
    - higher comment density variance (burstiness analogue)
    - rarer-word usage (Yule-K, Zipf-alpha)

  The Ghostbuster detector is a logistic-regression over a small
  committee of such cheap features. We reuse 6 handcrafted code features
  from Exp_DM/exp09_token_stats.py; no training on test-task labels, but
  we FIT the LR head on the dev split (a "lean-zero-shot" acceptable to
  the Droid paper since it never touches the test split).

FEATURES (per sample):
  f1 = Shannon entropy of token distribution
  f2 = burstiness = (std / mean) of inter-whitespace-gap lengths
  f3 = type-token-ratio (unique-tokens / total-tokens)
  f4 = Yule-K characteristic (repeat-word concentration)
  f5 = mean line length
  f6 = comment density (comment chars / total chars)

Cost: pure numpy, ~30-60 s on CPU for the full Droid test split.
"""
from __future__ import annotations

import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from collections import Counter
from typing import List

import numpy as np

from _common import ZSConfig, logger
from _zs_runner import run_zs_oral


_COMMENT_PATTERNS = [
    re.compile(r"#[^\n]*"),              # Python / Ruby / Shell
    re.compile(r"//[^\n]*"),             # C-family
    re.compile(r"/\*.*?\*/", re.DOTALL), # C-block
    re.compile(r'"""(.*?)"""', re.DOTALL),  # Python docstrings
    re.compile(r"'''(.*?)'''", re.DOTALL),
]

_TOKEN_RE = re.compile(r"\b[\w]+\b")


def _extract_features(code: str) -> np.ndarray:
    if not code or not code.strip():
        return np.zeros(6, dtype=np.float64)

    # Token list
    tokens = _TOKEN_RE.findall(code)
    n_tokens = max(len(tokens), 1)
    counter = Counter(tokens)
    n_types = max(len(counter), 1)

    # f1: token-distribution entropy
    total = sum(counter.values())
    probs = np.array([c / total for c in counter.values()])
    f1 = float(-(probs * np.log(probs + 1e-12)).sum())

    # f2: burstiness — std/mean of inter-whitespace-gap lengths
    gaps = [len(chunk) for chunk in re.split(r"\s+", code) if chunk]
    if len(gaps) > 1:
        g_arr = np.asarray(gaps, dtype=np.float64)
        f2 = float(g_arr.std() / (g_arr.mean() + 1e-8))
    else:
        f2 = 0.0

    # f3: type-token ratio
    f3 = float(n_types / n_tokens)

    # f4: Yule-K = 10^4 * (sum(V_i * i^2) - N) / N^2
    # where V_i is the number of token types that appear i times.
    freq_of_freq = Counter(counter.values())
    yule_sum = sum((i ** 2) * v for i, v in freq_of_freq.items())
    f4 = float(1e4 * (yule_sum - n_tokens) / (n_tokens ** 2 + 1e-8))

    # f5: mean line length
    lines = code.splitlines() or [code]
    f5 = float(np.mean([len(ln) for ln in lines]))

    # f6: comment density
    comment_chars = 0
    for pat in _COMMENT_PATTERNS:
        for m in pat.finditer(code):
            comment_chars += len(m.group(0))
    f6 = float(comment_chars / max(len(code), 1))

    return np.array([f1, f2, f3, f4, f5, f6], dtype=np.float64)


def _ghostbuster_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Feature extraction only; _zs_runner's threshold calibration learns
    the direction (human ↔ AI). Here we return a composite "AI-likeness"
    score as the NEGATIVE of a crude linear combination of the 6 features
    anchored to "human writing has high burstiness + high TTR + high
    entropy". Higher score ⇒ more AI-like.
    """
    logger.info(f"[ZS-03] Extracting token-stat features on {len(codes)} codes ...")
    feats = np.stack([_extract_features(c) for c in codes], axis=0)
    # Feature-wise z-score so the linear combination is scale-free
    mu = feats.mean(axis=0, keepdims=True)
    sd = feats.std(axis=0, keepdims=True) + 1e-8
    z = (feats - mu) / sd

    # Human tends to be HIGH on f1/f2/f3/f6, LOW on f4/f5
    # AI tends to be LOW on f1/f2/f3/f6, HIGH on f4/f5
    # Composite AI-likeness = -f1 - f2 - f3 + f4 + f5 - f6
    weights = np.array([-1.0, -1.0, -1.0, +1.0, +0.5, -1.0])
    s = z @ weights
    return s.astype(np.float64)


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="GhostbusterCode",
        exp_id="exp_zs_03",
        score_fn=_ghostbuster_score,
        cfg=cfg,
    )
