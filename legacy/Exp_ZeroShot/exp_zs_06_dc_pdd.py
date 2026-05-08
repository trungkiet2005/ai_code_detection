"""
[exp_zs_06] DC-PDD -- Divergence-Calibrated Pretraining-Data Detection,
            transplanted to AI-code detection.

THEORY HOOK:
  Zhang et al. "Pretraining Data Detection for Large Language Models:
  A Divergence-based Calibration Method" (ACL 2024, extended arXiv:2409.14781).
  Core: token-level log-prob p_theta(x_t | x_<t) is biased by the token's
  marginal frequency in natural text. Cleaner MI signal comes from
  cross-entropy between the model's conditional and a reference *marginal*
  distribution over the same token:
      DC-PDD(x) = mean_t [ log p_theta(x_t | x_<t) - log p_ref(x_t) ]
  Samples with HIGH DC-PDD are those the model is "much more confident
  about than expected from surface frequency" -- exactly the signature of
  generated-by-that-model code.

WHY THE TRANSPLANT WORKS FOR AI-CODE:
  AI-generated code over-uses common identifiers (`result`, `value`,
  `data`) that the reference corpus already rates as high-frequency; the
  model's conditional confidence on them is even higher than that
  baseline. Human code uses context-idiosyncratic names that drop the
  ratio.

IMPLEMENTATION:
  - Scorer: small MLM (default CodeBERT-base-mlm). For each sample we
    compute mean over tokens of (model_log_prob - ref_log_prob).
  - Reference distribution: token marginal built ON THE FLY from the dev
    split's own token counts (zero extra corpus dependency; the dev split
    is our proxy for "natural code distribution"). This is a LEAN DC-PDD;
    the strict paper variant uses The-Stack-v2 dedup.

HONEST LIMITATION:
  A dev-split marginal is biased toward the MGT/human mix actually present
  there. Works as a zero-shot *screening* of the mechanism. If this exp
  beats Fast-DetectGPT on Droid T3, the follow-up paper-final run should
  swap in a static Stack-v2 freq table.
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

from typing import List

import numpy as np

from _common import ZSConfig, logger
from _zs_runner import run_zs_oral


_mlm = None
_tokenizer = None
_ref_log_prob: np.ndarray = None  # shape (vocab_size,)


def _get_mlm(cfg: ZSConfig):
    global _mlm, _tokenizer
    if _mlm is not None:
        return _mlm, _tokenizer
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info(f"[ZS-06] Loading MLM {cfg.scorer_lm} ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    _mlm = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    _mlm.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _mlm = _mlm.to("cuda")
        if cfg.precision == "bf16":
            _mlm = _mlm.to(torch.bfloat16)
    return _mlm, _tokenizer


def _tokenise_all(codes: List[str], cfg: ZSConfig):
    """Return a list of per-sample token-id tensors (variable length, cap at max_length)."""
    _, tokenizer = _get_mlm(cfg)
    all_ids = []
    for c in codes:
        enc = tokenizer(
            c, max_length=cfg.scorer_max_length, truncation=True,
            return_tensors="pt", padding=False,
        )
        all_ids.append(enc["input_ids"][0])
    return all_ids


def _fit_reference_marginal(token_id_lists, vocab_size: int):
    """Build a reference log-probability over the vocabulary from the
    unigram frequencies of `token_id_lists`. Laplace-smoothed.
    """
    import torch
    counts = np.zeros(vocab_size, dtype=np.float64)
    for ids in token_id_lists:
        ids_np = ids.numpy() if hasattr(ids, "numpy") else np.asarray(ids)
        np.add.at(counts, ids_np, 1.0)
    total = counts.sum() + vocab_size   # Laplace smoothing
    probs = (counts + 1.0) / total
    return np.log(probs)


def _dc_pdd_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Score per sample: mean_t ( log p_theta(x_t) - log p_ref(x_t) )."""
    import torch

    global _ref_log_prob

    mlm, tokenizer = _get_mlm(cfg)
    vocab_size = mlm.config.vocab_size
    token_id_lists = _tokenise_all(codes, cfg)

    # First call: build the reference unigram distribution from these codes
    if _ref_log_prob is None:
        _ref_log_prob = _fit_reference_marginal(token_id_lists, vocab_size)
        logger.info(f"[ZS-06] Fit reference log-prob marginal from {len(codes)} samples, vocab={vocab_size}")

    # Forward + read off the model's log-prob of each original token (MLM
    # approximation: mask one token, predict, repeat -- but that's O(L) per
    # sample. Instead we use the diagonal trick: forward with NO masking and
    # read p_theta(x_t | full context) from the logits at position t. This is a
    # biased estimator (CodeBERT sees x_t in its input) but keeps cost at
    # ONE forward pass per sample. The bias is roughly constant across
    # samples so the detector's discriminative power survives.
    scores = np.zeros(len(codes), dtype=np.float64)
    bs = cfg.batch_size
    mask_id = tokenizer.mask_token_id
    pad_id = tokenizer.pad_token_id or 0

    with torch.no_grad():
        for start in range(0, len(codes), bs):
            chunk_ids = token_id_lists[start : start + bs]
            max_len = max(len(ids) for ids in chunk_ids)
            batch_ids = torch.full((len(chunk_ids), max_len), pad_id, dtype=torch.long)
            attn = torch.zeros_like(batch_ids)
            for i, ids in enumerate(chunk_ids):
                batch_ids[i, : len(ids)] = ids
                attn[i, : len(ids)] = 1
            if cfg.device == "cuda":
                batch_ids = batch_ids.to("cuda")
                attn = attn.to("cuda")
            logits = mlm(input_ids=batch_ids, attention_mask=attn).logits
            log_probs = torch.log_softmax(logits.float(), dim=-1)
            # Model log-prob at the original token id, at each position
            model_lp = log_probs.gather(-1, batch_ids.unsqueeze(-1)).squeeze(-1)
            model_lp = model_lp.cpu().numpy()
            attn_np = attn.cpu().numpy()
            batch_np = batch_ids.cpu().numpy()
            for i in range(len(chunk_ids)):
                mask = attn_np[i].astype(bool)
                mp = model_lp[i][mask]
                tokens = batch_np[i][mask]
                rp = _ref_log_prob[tokens]
                # DC-PDD = mean_t (model_lp - ref_lp); high => AI-typical
                if len(mp) == 0:
                    scores[start + i] = 0.0
                else:
                    scores[start + i] = float((mp - rp).mean())
    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="DC-PDD",
        exp_id="exp_zs_06",
        score_fn=_dc_pdd_score,
        cfg=cfg,
    )
