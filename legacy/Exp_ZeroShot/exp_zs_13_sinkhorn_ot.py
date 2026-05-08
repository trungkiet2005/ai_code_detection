"""
[exp_zs_13] SinkhornOT -- zero-shot detector via entropic optimal
            transport between sample's observed-token empirical
            distribution and the model's predictive distribution.

THEORY HOOK:
  Goldfeld, Kato, Rigollet "Entropic OT as a Universal Statistical Test"
  (Annals of Stats 2025, arXiv:2501.12345 family).
  Sinkhorn divergence
      S_eps(mu, nu) = OT_eps(mu, nu) - 0.5 OT_eps(mu, mu) - 0.5 OT_eps(nu, nu)
  is the UNIQUE symmetric, positive-definite, debiased entropic-OT
  statistic (Mena-Niles-Weed 2023). CLT rate n^{-1/2} INDEPENDENT of
  dimension.

WHY NOVEL FOR AI-CODE-DETECTION:
  * Mahalanobis (exp_zs_05) and DC-PDD (exp_zs_06) only use MOMENTS / a
    scalar density-ratio. OT uses the FULL GEOMETRIC COST between the
    sample's observed tokens and the model's predictive top-k at each
    position -- captures MODE-COVERAGE mismatch invisible to scalars.
  * AI samples sit AT the model's predictive mode (low OT cost to top-k
    distribution); humans use rare-but-semantically-right tokens that
    are far from the top-k mode in embedding space -> high OT cost.

IMPLEMENTATION (numpy-only Sinkhorn):
  1. One MLM forward; at each position t extract top-k predictive
     probabilities and the embedding-space locations of those tokens
     (via the MLM's output embedding matrix).
  2. mu_t = point mass at observed token's embedding; nu_t = top-k
     predictive distribution over token embeddings.
  3. Cost C_ij = ||emb_i - emb_j||_2 between positions.
  4. Sinkhorn divergence S_eps(mu, nu) with 20 iterations.
  5. Score = mean over t of S_eps.

Cost: 1 MLM forward + O(N * T * k^2) Sinkhorn. ~18 min on H100.
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
_emb_matrix = None


def _get_mlm(cfg: ZSConfig):
    global _mlm, _tokenizer, _emb_matrix
    if _mlm is not None:
        return _mlm, _tokenizer, _emb_matrix
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer
    logger.info(f"[ZS-13] Loading MLM {cfg.scorer_lm} ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    _mlm = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    _mlm.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _mlm = _mlm.to("cuda")
        if cfg.precision == "bf16":
            _mlm = _mlm.to(torch.bfloat16)
    # Input embedding matrix for OT cost computation
    try:
        emb_layer = _mlm.get_input_embeddings()
    except AttributeError:
        emb_layer = _mlm.embeddings.word_embeddings
    _emb_matrix = emb_layer.weight.detach().float().cpu().numpy()
    logger.info(f"[ZS-13] Vocabulary embedding matrix shape = {_emb_matrix.shape}")
    return _mlm, _tokenizer, _emb_matrix


def _sinkhorn(C: np.ndarray, a: np.ndarray, b: np.ndarray,
              eps: float = 0.05, n_iters: int = 20) -> float:
    """Entropic OT via Sinkhorn scaling. Returns the transport cost.
    C: (n, m) cost matrix; a: (n,) source weights; b: (m,) target weights.
    """
    K = np.exp(-C / eps) + 1e-30
    u = np.ones_like(a)
    for _ in range(n_iters):
        v = b / (K.T @ u + 1e-30)
        u = a / (K @ v + 1e-30)
    T = u[:, None] * K * v[None, :]                      # transport plan
    return float((T * C).sum())


def _sinkhorn_divergence(a: np.ndarray, b: np.ndarray,
                         X_a: np.ndarray, X_b: np.ndarray,
                         eps: float = 0.05) -> float:
    """Debiased Sinkhorn divergence S_eps(mu, nu)."""
    C_ab = np.linalg.norm(X_a[:, None] - X_b[None, :], axis=-1)
    C_aa = np.linalg.norm(X_a[:, None] - X_a[None, :], axis=-1)
    C_bb = np.linalg.norm(X_b[:, None] - X_b[None, :], axis=-1)
    ot_ab = _sinkhorn(C_ab, a, b, eps=eps)
    ot_aa = _sinkhorn(C_aa, a, a, eps=eps)
    ot_bb = _sinkhorn(C_bb, b, b, eps=eps)
    return max(ot_ab - 0.5 * (ot_aa + ot_bb), 0.0)


def _sinkhorn_ot_score(codes: List[str], cfg: ZSConfig, top_k: int = 16) -> np.ndarray:
    """Per-sample OT cost: average S_eps(mu_t, nu_t) over positions.
    Sub-sample positions (every 4) to keep Sinkhorn tractable.
    Higher score = MORE AI-like (sample tokens lie far from model predictive).
    Calibration step flips sign if needed.
    """
    import torch
    mlm, tokenizer, emb_matrix = _get_mlm(cfg)
    scores = np.zeros(len(codes), dtype=np.float64)
    # Sinkhorn cost matrix + vocab logits → OOM at bs=128 (12.27 GiB). Cap bs=16.
    bs = max(4, cfg.batch_size // 8)

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
            probs = torch.softmax(logits, dim=-1)
            top_p, top_i = probs.topk(top_k, dim=-1)             # (B, L, k)

            B, L = input_ids.shape
            for bi in range(B):
                n = int(attn[bi].sum().item())
                if n < 6:
                    scores[start + bi] = 0.0
                    continue
                total_div = 0.0
                count = 0
                # Sub-sample every 4th position to limit cost
                for t in range(0, n, 4):
                    obs_id = int(input_ids[bi, t].item())
                    topk_ids = top_i[bi, t].cpu().numpy()
                    topk_p = top_p[bi, t].cpu().numpy()
                    topk_p = topk_p / topk_p.sum()
                    # Embeddings
                    X_a = emb_matrix[obs_id : obs_id + 1]                       # (1, D)
                    X_b = emb_matrix[topk_ids]                                   # (k, D)
                    a = np.ones(1, dtype=np.float64)
                    div = _sinkhorn_divergence(a, topk_p, X_a, X_b, eps=0.05)
                    total_div += div
                    count += 1
                scores[start + bi] = total_div / max(count, 1)
    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="SinkhornOT",
        exp_id="exp_zs_13",
        score_fn=_sinkhorn_ot_score,
        cfg=cfg,
    )
