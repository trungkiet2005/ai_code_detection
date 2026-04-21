"""[exp_zs_31] Fast-DetectGPT OFFICIAL reproduction (paper-exact scoring kernel).

THEORY HOOK (Bao et al., ICLR 2024):
  Paper: "Fast-DetectGPT: Efficient Zero-Shot Detection of Machine-Generated
          Text via Conditional Probability Curvature" (arXiv:2310.05130)
  Repo:  https://github.com/baoguangsheng/fast-detect-gpt (MIT licence)

  vs Exp_zs_02: that exp uses our CodeBERT-MLM SURROGATE of the FDG signal —
  mask a token, score its predicted logprob variance. This approximates the
  paper's conditional-probability-curvature but is NOT paper-exact because:
    (1) FDG needs a CAUSAL LM (next-token logprobs), CodeBERT is MLM.
    (2) FDG uses TWO models (sampling ref + scoring) — we used one.
    (3) FDG's analytic discrepancy has closed-form var estimator vs ours MC.

  This exp uses Bao et al.'s EXACT scoring kernel
  (`get_sampling_discrepancy_analytic`) on the reference model pair, vendored
  into `Exp_ZeroShot/fast_detect_gpt/` under MIT licence.

  We expect Exp_31 to be closer to the paper's reported 64.54 Droid T3 W-F1
  number than Exp_02 (which scored 32.07 W-F1). Any remaining gap
  quantifies "code-distribution shift in paper-original FDG" — the paper
  evaluated on xsum/squad/writing text, not on a multi-language code corpus.

MODEL PAIR (selectable via `--pair` flag or ZSConfig subclass):
  - `gpt-neo-2.7B_gpt-neo-2.7B` (self-scoring, ~5 GB)  ← default, fits T4
  - `gpt-j-6B_gpt-neo-2.7B` (paper baseline)           ← needs H100
  - `falcon-7b_falcon-7b-instruct` (paper best BB)     ← needs H100, ~28 GB
  - `llama3-8b_llama3-8b-instruct` (paper Jan'26 best) ← needs H100, ~32 GB

HOW IT SCORES:
  For each code sample, compute `crit = FastDetectGPT.compute_crit(text)` —
  a scalar conditional-probability-curvature score. HIGHER = more AI-like.
  Our _zs_runner calibrates τ on dev to hit Human R ≥ 0.95, then reports
  Droid T3 W-F1 / CoDET binary Macro-F1 on test.

Cost: 1-2 causal-LM forwards per sample (depends on pair). ~30-60 min for
  Droid T3 + CoDET binary on H100 with gpt-neo-2.7B pair.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from types import SimpleNamespace
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
from fast_detect_gpt import FastDetectGPT


# -----------------------------------------------------------------------------
# Config — pick model pair via env var to keep file standalone
# -----------------------------------------------------------------------------

# Kaggle H100 can fit falcon-7b pair (28 GB fp16) comfortably alongside the
# Droid dev/test iterators. Override via env: FDG_PAIR=gpt-neo-2.7B for T4.
FDG_PAIR = os.environ.get("FDG_PAIR", "gpt-neo-2.7B_gpt-neo-2.7B")
# Parse "sampling_scoring" from pair key.
_parts = FDG_PAIR.split("_", 1)
if len(_parts) != 2:
    raise ValueError(
        f"FDG_PAIR must be '<sampling>_<scoring>', got {FDG_PAIR!r}. "
        f"Valid: gpt-neo-2.7B_gpt-neo-2.7B | gpt-j-6B_gpt-neo-2.7B | "
        f"falcon-7b_falcon-7b-instruct | llama3-8b_llama3-8b-instruct"
    )
SAMPLING_MODEL, SCORING_MODEL = _parts


_fdg_detector = None  # Lazy-init: build once, share across benches.


def _get_detector(cfg: ZSConfig) -> FastDetectGPT:
    global _fdg_detector
    if _fdg_detector is not None:
        return _fdg_detector

    cache_dir = os.environ.get("HF_HOME") or os.path.join(os.getcwd(), "fdg_cache")
    os.makedirs(cache_dir, exist_ok=True)
    args = SimpleNamespace(
        sampling_model_name=SAMPLING_MODEL,
        scoring_model_name=SCORING_MODEL,
        device=cfg.device if cfg.device == "cuda" else "cpu",
        cache_dir=cache_dir,
    )
    logger.info(f"[ZS-31] Loading Fast-DetectGPT pair: sampling={SAMPLING_MODEL}, "
                f"scoring={SCORING_MODEL}, device={args.device}, cache={cache_dir}")
    _fdg_detector = FastDetectGPT(args)
    return _fdg_detector


def _fast_detect_gpt_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Paper-exact Fast-DetectGPT scoring.

    Returns per-sample discrepancy (higher = more AI-like). τ-calibration
    in _zs_runner then pins Human Recall >= 0.95 on dev and evaluates test.
    """
    detector = _get_detector(cfg)
    scores = np.zeros(len(codes), dtype=np.float64)

    logger.info(f"[ZS-31] Scoring {len(codes)} samples with paper-exact "
                f"get_sampling_discrepancy_analytic...")
    for i, code in enumerate(codes):
        if not code or not code.strip():
            scores[i] = 0.0
            continue
        # Cap extremely long samples to avoid OOM on causal-LM forward.
        if len(code) > 8000:
            code = code[:8000]
        try:
            crit, _ntoken = detector.compute_crit(code)
            scores[i] = float(crit)
        except Exception as e:
            # Tokenizer-mismatch (e.g. tokenization assertion) or OOM → skip.
            # Keep at 0 so dev calibration can still find a threshold.
            if i < 5:
                logger.warning(f"[ZS-31] Sample {i} failed: {type(e).__name__}: {str(e)[:120]}")
            scores[i] = 0.0
        if (i + 1) % 2000 == 0:
            logger.info(f"[ZS-31] Scored {i + 1}/{len(codes)} "
                        f"(recent crit={scores[i]:.4f})")
    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    method_name = f"FastDetectGPT-Official-{FDG_PAIR}"
    run_zs_oral(
        method_name=method_name,
        exp_id="exp_zs_31",
        score_fn=_fast_detect_gpt_score,
        cfg=cfg,
    )
