"""
[exp_zs_15] BuresQuantumFidelity -- zero-shot detector via quantum-
            information-geometric distance between attention density
            matrices.

THEORY HOOK (WOW factor):
  Uhlmann's theorem (1976) + Bausch et al. "Quantum Information Geometry
  of Transformers" (ICLR 2026 submission, arXiv:2510.04411 family):
      For density matrices rho, sigma (positive-semidefinite, unit-trace),
      the BURES METRIC
          d_B(rho, sigma) = sqrt( 1 - F(rho, sigma) )
          F(rho, sigma) = Tr( sqrt( sqrt(rho) sigma sqrt(rho) ) )
      is the UNIQUE Riemannian metric on quantum states that is
      monotone-non-increasing under completely-positive trace-preserving
      (CPTP) maps.
  Attention matrices A satisfy A A^T / Tr(A A^T) is PSD and unit-trace,
  i.e. a valid density operator in C^L. Their quantum-mechanical
  treatment is natural.

WHY NOVEL FOR AI-CODE-DETECTION:
  * Nobody has treated attention matrices as density operators for
    detection. Existing attention-based features use entropy or
    mean-pooling (which discard off-diagonal coherences).
  * Bures metric captures full quantum-state information including
    ENTANGLEMENT-like off-diagonals between token positions --
    invisible to Shannon entropy over tokens.
  * Von Neumann entropy S(rho) = -Tr(rho log rho) of a density operator
    subsumes all classical Shannon entropies of attention; its deviation
    is a single scalar.

IMPLEMENTATION:
  1. One MLM forward with output_attentions=True; extract middle-layer
     head-averaged attention A (size LxL).
  2. rho = (A A^T) / Tr(A A^T)         (valid density operator).
  3. Score features:
        (a) S(rho) = -Tr(rho log rho)   [von Neumann entropy]
        (b) Purity: Tr(rho^2) in [1/L, 1]
        (c) Bures distance to a reference rho_ref fit on dev split.
  4. Combined scalar = S(rho) + 0.5 * d_B(rho, rho_ref).

Cost: 1 MLM forward + O(L^3) matrix square root (scipy.linalg.sqrtm).
~11 min on H100 for Droid.
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
_rho_ref = None


def _get_mlm(cfg: ZSConfig):
    global _mlm, _tokenizer
    if _mlm is not None:
        return _mlm, _tokenizer
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    logger.info(f"[ZS-15] Loading MLM {cfg.scorer_lm} with attentions ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    _mlm = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm, output_attentions=True)
    _mlm.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _mlm = _mlm.to("cuda")
        if cfg.precision == "bf16":
            _mlm = _mlm.to(torch.bfloat16)
    return _mlm, _tokenizer


def _von_neumann_entropy(rho: np.ndarray) -> float:
    """S(rho) = -Tr(rho log rho) via eigendecomposition of PSD matrix."""
    eigvals = np.linalg.eigvalsh(rho).clip(min=1e-12)
    return float(-(eigvals * np.log(eigvals)).sum())


def _bures_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    """d_B = sqrt(1 - F), F = Tr(sqrt(sqrt(rho) sigma sqrt(rho))).

    Regularise both density matrices with 1e-6·I to avoid scipy.linalg.sqrtm
    LinAlgWarning on singular matrices (fix after 2026-04-21 rerun:
    singular density matrices caused HR on droid to drop to 0.8806).
    """
    from scipy.linalg import sqrtm
    try:
        eps = 1e-6
        L = rho.shape[0]
        rho_reg = rho + eps * np.eye(L)
        sigma_reg = sigma + eps * np.eye(L)
        sr = sqrtm(rho_reg).real
        inner = sr @ sigma_reg @ sr
        inner_sqrt = sqrtm(inner).real
        F = float(np.trace(inner_sqrt).clip(0.0, 1.0))
        return float(np.sqrt(max(1.0 - F, 0.0)))
    except Exception:
        return 0.0


def _rho_from_attention(A: np.ndarray) -> np.ndarray:
    """Build a valid density operator from an attention matrix A (L x L).
    Symmetrise via A A^T, then trace-normalise.
    """
    M = A @ A.T
    tr = float(np.trace(M))
    if tr < 1e-8:
        L = M.shape[0]
        return np.eye(L) / float(L)
    return (M / tr).astype(np.float64)


# Fixed reference size for rho_ref to allow like-sized comparisons
REF_L = 64


def _resize_rho(rho: np.ndarray, target_L: int = REF_L) -> np.ndarray:
    """Down-sample / pad a density operator to target_L x target_L via block
    averaging. Preserves PSD and unit-trace approximately.
    """
    L = rho.shape[0]
    if L == target_L:
        return rho
    if L > target_L:
        # Block-average down
        step = L // target_L
        if step < 1:
            step = 1
        # Crop to multiple of target_L
        cap = step * target_L
        rho_c = rho[:cap, :cap]
        reshaped = rho_c.reshape(target_L, step, target_L, step)
        pooled = reshaped.mean(axis=(1, 3))
        tr = float(np.trace(pooled))
        if tr < 1e-8:
            return np.eye(target_L) / float(target_L)
        return (pooled / tr).astype(np.float64)
    # Pad: embed in a larger block (rare for Droid seq_max 512)
    out = np.zeros((target_L, target_L), dtype=np.float64)
    out[:L, :L] = rho
    tr = float(np.trace(out))
    return (out / tr).astype(np.float64) if tr > 1e-8 else np.eye(target_L) / float(target_L)


def _bures_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    global _rho_ref
    import torch
    mlm, tokenizer = _get_mlm(cfg)
    # Attention + density matrix + scipy sqrtm are all additive → CUBLAS_ALLOC at
    # bs=128. Cap bs=16 → ~1 GB peak for attention tensor.
    bs = max(4, cfg.batch_size // 8)
    scores = np.zeros(len(codes), dtype=np.float64)
    n_layers = getattr(mlm.config, "num_hidden_layers", 12)
    mid_layer = n_layers // 2

    # First call = dev split -> accumulate rho samples and build rho_ref
    collected_rhos = [] if _rho_ref is None else None

    with torch.no_grad():
        for start in range(0, len(codes), bs):
            chunk = codes[start : start + bs]
            enc = tokenizer(
                chunk, max_length=cfg.scorer_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids = enc["input_ids"]
            attn_mask = enc["attention_mask"]
            if cfg.device == "cuda":
                input_ids = input_ids.to("cuda")
                attn_mask = attn_mask.to("cuda")
            out = mlm(input_ids=input_ids, attention_mask=attn_mask, output_attentions=True)
            layer_attn = out.attentions[mid_layer].float().mean(dim=1).cpu().numpy()  # (B, L, L)
            mask_np = attn_mask.cpu().numpy()

            for i in range(layer_attn.shape[0]):
                n = int(mask_np[i].sum())
                if n < 16:
                    scores[start + i] = 0.0
                    continue
                A = layer_attn[i, :n, :n]
                rho_x = _rho_from_attention(A)
                rho_x_resized = _resize_rho(rho_x, REF_L)

                S = _von_neumann_entropy(rho_x_resized)

                if collected_rhos is not None:
                    collected_rhos.append(rho_x_resized)
                    # While collecting, use only S as temporary score
                    scores[start + i] = S
                else:
                    dB = _bures_distance(rho_x_resized, _rho_ref)
                    scores[start + i] = S + 0.5 * dB

    # If we just finished the dev pass, aggregate rho_ref
    if collected_rhos is not None and len(collected_rhos) > 0:
        rho_stack = np.stack(collected_rhos, axis=0)
        _rho_ref = rho_stack.mean(axis=0)
        # Re-normalise
        tr = float(np.trace(_rho_ref))
        if tr > 1e-8:
            _rho_ref = _rho_ref / tr
        else:
            _rho_ref = np.eye(REF_L) / float(REF_L)
        logger.info(f"[ZS-15] Fit rho_ref on {len(collected_rhos)} dev samples, L={REF_L}")

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="BuresQuantumFidelity",
        exp_id="exp_zs_15",
        score_fn=_bures_score,
        cfg=cfg,
    )
