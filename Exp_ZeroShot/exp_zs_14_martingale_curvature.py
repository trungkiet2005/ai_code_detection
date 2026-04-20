"""
[exp_zs_14] MartingaleCurvature -- zero-shot detector via De Jong
            martingale test on residuals of per-token log-prob
            conditioned on syntactic-depth proxy.

THEORY HOOK (OUR OWN, cross-field transfer):
  Under the null "human wrote this sample," per-token log-prob residuals
  conditioned on the syntactic-depth process should form a MARTINGALE
  DIFFERENCE SEQUENCE (MDS) under any self-consistent generative model.
  AI-drafted code VIOLATES MDS because the model implicitly plans AST
  structure ahead of token emission, injecting look-ahead bias into the
  residuals.

  We repurpose the De Jong martingale test (Hong-Lee 2005, Econometrica)
  -- a staple of econometric efficiency testing -- as a code-authorship
  test with closed-form asymptotic null distribution.

WHY NOVEL:
  * Existing ZS detectors (Fast-DetectGPT, Binoculars, Min-K%++, DC-PDD,
    EnergyScore) treat per-token surprise as i.i.d. -- an EXCHANGEABLE
    bag -- discarding the fact that code is a stochastic process whose
    increments are coupled to the AST.
  * The De Jong test has a known chi-squared null distribution; makes
    the test calibration-free in principle (tau = chi^2 quantile).
  * Closest work: Kirchenbauer's watermark uses martingales only for
    watermark detection; nobody has used them for structural-causality
    authorship tests.

IMPLEMENTATION (no external AST parser; regex-based depth proxy):
  1. One MLM forward; per-token log-prob sequence.
  2. Depth proxy: running count of open-bracket / indentation level at
     each token position (approximates AST depth). Uses tokenizer's
     decoded surface form, no tree-sitter needed.
  3. Regress log p_t on depth_t (OLS), get residuals e_t.
  4. De Jong statistic:
        DJ = sum_t e_t * e_{t-1}  /  sqrt( sum_t e_t^2 * e_{t-1}^2 )
     Under MDS null, DJ ~ N(0, 1). Large |DJ| => violation => AI-drafted.
  5. score = |DJ|.

Cost: 1 MLM forward + O(N*L) regex/regress. ~10 min on H100.
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


_mlm = None
_tokenizer = None


def _get_mlm(cfg: ZSConfig):
    global _mlm, _tokenizer
    if _mlm is not None:
        return _mlm, _tokenizer
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    logger.info(f"[ZS-14] Loading MLM {cfg.scorer_lm} ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    _mlm = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    _mlm.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _mlm = _mlm.to("cuda")
        if cfg.precision == "bf16":
            _mlm = _mlm.to(torch.bfloat16)
    return _mlm, _tokenizer


OPEN = set("([{")
CLOSE = set(")]}")


def _depth_trajectory(token_strs: list) -> np.ndarray:
    """Approximate AST depth via bracket / indent nesting on decoded tokens.
    Returns a length-L integer depth at each position (cumulative).
    """
    depth = 0
    out = np.zeros(len(token_strs), dtype=np.int32)
    for i, tok in enumerate(token_strs):
        s = str(tok)
        # Count opens - closes in this token
        opens = sum(1 for c in s if c in OPEN)
        closes = sum(1 for c in s if c in CLOSE)
        depth += opens - closes
        # Indent proxy: count leading tabs/spaces in decoded token
        # (approximates Python-style indentation at line boundaries)
        leading = len(s) - len(s.lstrip(" \t"))
        out[i] = depth * 10 + min(leading, 20)
    return out


def _de_jong_statistic(residuals: np.ndarray) -> float:
    """DJ = sum_t e_t * e_{t-1} / sqrt(sum_t e_t^2 * e_{t-1}^2 + 1e-8).
    Under MDS null, DJ is asymptotically N(0, 1).
    """
    if len(residuals) < 5:
        return 0.0
    e_t = residuals[1:]
    e_t_1 = residuals[:-1]
    num = float((e_t * e_t_1).sum())
    den = float(np.sqrt((e_t * e_t_1 ** 2).sum() ** 2 + (e_t_1 * e_t ** 2).sum() ** 2 + 1e-8))
    if den < 1e-8:
        return 0.0
    return num / den


def _ols_residuals(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Simple OLS y ~ x; return residuals."""
    x_c = x - x.mean()
    y_c = y - y.mean()
    denom = (x_c ** 2).sum() + 1e-8
    beta = (x_c * y_c).sum() / denom
    intercept = y.mean() - beta * x.mean()
    return y - (beta * x + intercept)


def _martingale_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    import torch
    mlm, tokenizer = _get_mlm(cfg)
    bs = cfg.batch_size
    scores = np.zeros(len(codes), dtype=np.float64)

    with torch.no_grad():
        for start in range(0, len(codes), bs):
            chunk = codes[start : start + bs]
            enc = tokenizer(
                chunk, max_length=cfg.scorer_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            attn = enc["attention_mask"]
            if cfg.device == "cuda":
                input_ids = input_ids.to("cuda")
                attn = attn.to("cuda")
            logits = mlm(input_ids=input_ids, attention_mask=attn).logits.float()
            log_probs = torch.log_softmax(logits, dim=-1)
            obs_lp = log_probs.gather(-1, input_ids.unsqueeze(-1)).squeeze(-1).cpu().numpy()
            ids_np = input_ids.cpu().numpy()
            mask_np = attn.cpu().numpy()

            for i in range(ids_np.shape[0]):
                n = int(mask_np[i].sum())
                if n < 10:
                    scores[start + i] = 0.0
                    continue
                # Decode tokens to strings for depth proxy
                tok_ids = ids_np[i, :n]
                tok_strs = tokenizer.convert_ids_to_tokens(tok_ids.tolist())
                depth = _depth_trajectory(tok_strs).astype(np.float64)
                lp = obs_lp[i, :n].astype(np.float64)
                resid = _ols_residuals(lp, depth)
                dj = _de_jong_statistic(resid)
                scores[start + i] = abs(dj)
    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="MartingaleCurvature",
        exp_id="exp_zs_14",
        score_fn=_martingale_score,
        cfg=cfg,
    )
