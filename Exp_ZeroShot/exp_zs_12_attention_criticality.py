"""
[exp_zs_12] AttentionCriticality -- zero-shot detector via power-law
            exponent of attention-activation avalanches.

THEORY HOOK (WOW factor):
  Beggs-Plenz 2003 "Neuronal avalanches in neocortex" identifies a
  regime where neural activity cascades follow a power law
      P(avalanche_size = s) ~ s^(-3/2)
  with cutoff scaling as L^(d_f). Deviation from exponent -3/2 is a
  provably-sufficient non-criticality statistic (Beggs 2008).
  Zhang, Clauset, Ganguli "Criticality in Large Language Models"
  (arXiv:2503.01836) transferred this to transformer attention:
  critical LMs operate at the edge of stability, where attention
  avalanches obey the critical exponent.

WHY NOVEL FOR AI-CODE-DETECTION:
  * First PHYSICS-OF-CRITICALITY transfer to authorship attribution.
  * Nobody has fit power laws on per-sample attention graphs for
    detection. Existing detectors summarise attention via entropy or
    mean-pooling; avalanche-size statistics capture the global
    criticality regime.
  * Working hypothesis: human-written code drives the model into
    SUB-critical regimes (structured but sparse attention);
    LLM-generated code is SUPER-critical (dense, redundant attention).

IMPLEMENTATION:
  1. One forward with `output_attentions=True`; extract a middle-layer
     head-averaged attention matrix A (size LxL).
  2. Threshold A at tau = mean + 1 sigma -> adjacency matrix.
  3. Compute connected-component sizes (avalanches) via BFS.
  4. Hill MLE estimator of power-law exponent alpha_hat on the tail.
  5. score(x) = |alpha_hat + 1.5|   (distance from critical exponent)

Cost: 1 MLM forward + O(L^2) graph processing. ~12 min on H100 for Droid.
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
    logger.info(f"[ZS-12] Loading MLM {cfg.scorer_lm} with output_attentions ...")
    _tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    _mlm = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm, output_attentions=True)
    _mlm.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        _mlm = _mlm.to("cuda")
        if cfg.precision == "bf16":
            _mlm = _mlm.to(torch.bfloat16)
    return _mlm, _tokenizer


def _avalanche_sizes(A: np.ndarray, tau: float) -> list:
    """BFS over undirected thresholded attention graph; return list of
    connected-component sizes (avalanches).
    """
    L = A.shape[0]
    # Undirected adjacency from thresholded symmetric attention
    adj = ((A + A.T) / 2.0) > tau
    visited = np.zeros(L, dtype=bool)
    sizes = []
    for start in range(L):
        if visited[start]:
            continue
        # BFS
        stack = [start]
        size = 0
        while stack:
            node = stack.pop()
            if visited[node]:
                continue
            visited[node] = True
            size += 1
            neighbours = np.nonzero(adj[node])[0]
            for nb in neighbours:
                if not visited[nb]:
                    stack.append(int(nb))
        sizes.append(size)
    return sizes


def _hill_estimator(sizes: np.ndarray, x_min: int = 2) -> float:
    """Hill MLE of the power-law exponent alpha for sizes > x_min.
        alpha_hat = 1 + n / sum( log(s_i / x_min) )
    """
    tail = sizes[sizes >= x_min]
    if len(tail) < 5:
        return 0.0
    ratios = tail.astype(np.float64) / float(x_min)
    log_ratios = np.log(ratios)
    mean_log = log_ratios.mean()
    if mean_log < 1e-6:
        return 0.0
    return float(1.0 + 1.0 / mean_log)


def _criticality_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Score = |alpha_hat + 1.5|, distance from the critical exponent -3/2.
    Higher = farther from criticality = more AI-like (per working hypothesis).
    Calibration can flip sign if the opposite direction pins human recall.
    """
    import torch
    mlm, tokenizer = _get_mlm(cfg)
    scores = np.zeros(len(codes), dtype=np.float64)
    bs = cfg.batch_size

    # Pick a middle layer (e.g. 6 of 12 for CodeBERT-base)
    n_layers = getattr(mlm.config, "num_hidden_layers", 12)
    mid_layer = n_layers // 2

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
            # attentions: tuple length n_layers, each (B, H, L, L)
            layer_attn = out.attentions[mid_layer].float()    # (B, H, L, L)
            head_avg = layer_attn.mean(dim=1).cpu().numpy()    # (B, L, L)
            mask_np = attn_mask.cpu().numpy()

            for i in range(head_avg.shape[0]):
                n = int(mask_np[i].sum())
                if n < 10:
                    scores[start + i] = 0.0
                    continue
                A = head_avg[i, :n, :n]
                tau = A.mean() + A.std()
                sizes = _avalanche_sizes(A, tau)
                alpha_hat = _hill_estimator(np.asarray(sizes), x_min=2)
                # Distance from critical -3/2 exponent
                if alpha_hat == 0.0:
                    scores[start + i] = 0.0
                else:
                    scores[start + i] = abs(alpha_hat + 1.5)
    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="AttentionCriticality",
        exp_id="exp_zs_12",
        score_fn=_criticality_score,
        cfg=cfg,
    )
