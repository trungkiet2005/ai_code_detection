"""
[exp_zs_00] BaselineStats -- the absolute floor for zero-shot detection.

Two trivial scorers stacked for the oral paper's "method-vs-random" panel:

  (A) random scorer   -- score(x) = U(0, 1) noise. Calibrating human recall
      at 0.95 then binarising gives us a trivial ceiling on what NOT knowing
      anything yields. In theory macro-F1 ≈ 0.5 on balanced tasks.

  (B) length-only scorer -- score(x) = log(1 + len(code)).  Tests whether
      machine-generated code is systematically LONGER or SHORTER than human
      code. If this baseline already beats Fast-DetectGPT on Droid T3, the
      whole zero-shot detection field has a length-shortcut problem. If it
      doesn't (our expectation), we establish a tight floor for the oral
      comparison block.

Oral-paper use:
  - Table "Dual-benchmark Zero-Shot, Droid T3 + CoDET binary"
  - Expected: random ≈ 0.50; length-only ≈ 0.55-0.60 on both benches.
  - Any proposed ZS detector must clear BOTH floors AND the Fast-DetectGPT
    headline (64.54 on Droid T3) to contribute an oral-level claim.

Cost: pure numpy + char-level length, ~5 s total on CPU for both benches.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import List

import numpy as np

from _common import ZSConfig, logger
from _zs_runner import run_zs_oral


def _random_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Uniform noise. Seeded for reproducibility."""
    rng = np.random.default_rng(cfg.seed)
    return rng.random(len(codes)).astype(np.float64)


def _length_only_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Score = log(1 + char-length). Higher-length tends to correlate with
    LM-generated (longer, more verbose) vs human (terse) code -- this is a
    known but weak signal the paper's detectors SHOULD outperform.

    We return the raw log-length; the _zs_runner calibration step picks
    whichever side of τ maximises human recall on dev.
    """
    return np.log1p(np.asarray([len(c) for c in codes], dtype=np.float64))


if __name__ == "__main__":
    logger.info("\n" + "#" * 78)
    logger.info("# exp_zs_00 -- FLOOR (A): random scorer")
    logger.info("#" * 78)
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="RandomScorer",
        exp_id="exp_zs_00_random",
        score_fn=_random_score,
        cfg=cfg,
    )

    logger.info("\n" + "#" * 78)
    logger.info("# exp_zs_00 -- FLOOR (B): length-only scorer")
    logger.info("#" * 78)
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="LengthOnlyScorer",
        exp_id="exp_zs_00_length",
        score_fn=_length_only_score,
        cfg=cfg,
    )
