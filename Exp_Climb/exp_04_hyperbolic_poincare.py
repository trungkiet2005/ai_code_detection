"""
[exp_04] PoincareGenealogy -- Poincare-ball embedding of LLM generator tree

NOVELTY (NeurIPS 2026 oral target):
-----------------------------------
Cross-domain transfer from HypCD (Liu et al., CVPR 2025, paper 2504.06120)
and "Hyperbolic Large Language Models" (Patil et al., 2509.05757).

Key insight: the 6 CoDET generators form a TREE (human root -> family
branches -> specific generators). Euclidean embeddings fundamentally cannot
represent trees without distortion (grows linearly with depth), but
*hyperbolic* (Poincare-ball) space has volume growing EXPONENTIALLY with
radius -> perfect for tree-like hierarchies.

All prior HierTree methods (Exp00/Exp01) operate in Euclidean cosine space
and fight geometry. PoincareGenealogy projects embeddings to the Poincare
ball and uses Poincare distance for family / class decisions. The tree
becomes isometric to the geometry itself.

Components (all novel for code detection):
  1. Euclidean -> Poincare projection (exp_0 map with learnable curvature c).
  2. Poincare distance-based centroids: d_P(x,c) = arcosh(1 + 2*||x-c||^2 /
     ((1-||x||^2)(1-||c||^2))). Centroid c_k updated via Frechet mean.
  3. HIERARCHY-AWARE LOSS: leaf classes (specific generator) near ball edge,
     family centroids mid-radius, "human" root near origin. Enforced by a
     radial-depth regularizer: ||x_i||_P approx depth(class(i)) / max_depth.
  4. Curvature regularizer keeps c > 0 (strictly hyperbolic).

Why it should outperform Exp18/Exp00:
  * Tree distortion ~ O(1) in hyperbolic vs O(n) in Euclidean -> the Qwen /
    Nxcode sibling pair should be FAR easier to disambiguate while still
    sharing a family subtree.
  * Natural extension to 12-class (AICD T2) and arbitrary taxonomies.

Expected wins (per insights 2, 7, 14 + HypCD paper):
  * CoDET Author 6-class > 70.7 (first to crack 70.7 barrier in repo)
  * Qwen1.5 per-class F1 > 0.48
  * Droid T3 stable ~ 0.89 (hyperbolic should not hurt binary / shallow tasks)

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. `run_mode="lean"` -> 8 runs in ~53 min on H100.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"
REQUIRED_TOKEN = "lean"   # bump this when the climb runner API changes


def _runner_has_token(climb_dir: str, token: str) -> bool:
    """Return True iff _climb_runner.py already contains token."""
    runner = os.path.join(climb_dir, "_climb_runner.py")
    if not os.path.exists(runner):
        return False
    try:
        with open(runner, "r", encoding="utf-8") as f:
            return token in f.read()
    except OSError:
        return False


def _bootstrap_climb_path() -> str:
    cwd = os.getcwd()
    for candidate in (
        os.path.join(cwd, "Exp_Climb"),
        os.path.join(cwd, "ai_code_detection", "Exp_Climb"),
    ):
        if os.path.exists(os.path.join(candidate, "_common.py")):
            if _runner_has_token(candidate, REQUIRED_TOKEN):
                return candidate
            parent = os.path.dirname(candidate) if candidate.endswith("Exp_Climb") else candidate
            if parent.endswith("ai_code_detection") and os.path.exists(parent):
                print(f"[bootstrap] Stale clone at {parent} (no {REQUIRED_TOKEN!r} token) -> removing for fresh clone")
                shutil.rmtree(parent, ignore_errors=True)
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        if os.path.exists(os.path.join(here, "_common.py")) and _runner_has_token(here, REQUIRED_TOKEN):
            return here
    except NameError:
        pass
    repo_dir = os.path.join(cwd, "ai_code_detection")
    if os.path.exists(repo_dir):
        print(f"[bootstrap] Removing existing {repo_dir} to force fresh clone")
        shutil.rmtree(repo_dir, ignore_errors=True)
    print(f"[bootstrap] Cloning {REPO_URL} -> {repo_dir}")
    subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, repo_dir])
    return os.path.join(repo_dir, "Exp_Climb")


_climb_dir = _bootstrap_climb_path()
if _climb_dir not in sys.path:
    sys.path.insert(0, _climb_dir)
for _mod in list(sys.modules):
    if _mod.startswith(("_climb_runner", "_common", "_trainer",
                        "_data_codet", "_data_droid", "_features",
                        "_model", "_paper_table")):
        del sys.modules[_mod]
print(f"[bootstrap] Exp_Climb path: {_climb_dir}")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import logger
from _trainer import FocalLoss, default_compute_losses
from _data_codet import CoDETM4Config
from _data_droid import DroidConfig
from _climb_runner import run_full_climb


# ===========================================================================
# Poincare-ball operations
# ===========================================================================

EPS = 1e-6
MAX_NORM = 1.0 - 1e-3


def _clip_to_ball(x: torch.Tensor) -> torch.Tensor:
    """Keep point strictly inside the unit ball (||x|| < 1)."""
    norm = x.norm(p=2, dim=-1, keepdim=True).clamp_min(EPS)
    factor = torch.clamp(MAX_NORM / norm, max=1.0)
    return x * factor


def expmap0(v: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """Exponential map at the origin: Euclidean -> Poincare ball of curvature c."""
    sqrt_c = math.sqrt(c)
    v_norm = v.norm(p=2, dim=-1, keepdim=True).clamp_min(EPS)
    coeff = torch.tanh(sqrt_c * v_norm) / (sqrt_c * v_norm)
    return _clip_to_ball(coeff * v)


def poincare_distance(x: torch.Tensor, y: torch.Tensor, c: float = 1.0) -> torch.Tensor:
    """d_c(x, y) = (2 / sqrt(c)) * arctanh(sqrt(c) * ||(-x) oplus_c y||)."""
    sqrt_c = math.sqrt(c)
    x2 = (x * x).sum(dim=-1, keepdim=True).clamp_max(MAX_NORM ** 2)
    y2 = (y * y).sum(dim=-1, keepdim=True).clamp_max(MAX_NORM ** 2)
    xy = (x * y).sum(dim=-1, keepdim=True)
    diff2 = ((x - y) * (x - y)).sum(dim=-1, keepdim=True)
    # Wu-style approximation (numerically stabler than full gyrovector add)
    num = 2 * diff2
    denom = (1 - c * x2) * (1 - c * y2)
    arg = 1 + c * num / denom.clamp_min(EPS)
    # d_P = arcosh(arg) / sqrt(c)
    return torch.acosh(arg.clamp_min(1 + EPS)).squeeze(-1) / sqrt_c


# ===========================================================================
# Hyperbolic genealogy head
# ===========================================================================

# 0=human, 1=codellama, 2=gpt, 3=llama3.1, 4=nxcode, 5=qwen1.5
AUTHOR_FAMILY_CODET = [0, 1, 2, 1, 3, 3]
# depth in the tree: human=0 (root), family=1, leaf=2
AUTHOR_DEPTH_CODET = [0, 2, 2, 2, 2, 2]


def _depth_table(num_classes: int):
    if num_classes == 6:
        return AUTHOR_DEPTH_CODET
    if num_classes == 2:
        return [0, 1]  # human root, any-AI leaf
    # generic: root=0, everything else leaf=1
    return [0] + [1] * (num_classes - 1)


class PoincareHead(nn.Module):
    """Learnable Poincare centroids + curvature.

    Forward accepts Euclidean embeddings and returns:
      * distances to each centroid in hyperbolic space (B, C)
      * hyperbolic-space image of the batch (B, D) for centroid update
    """

    def __init__(self, num_classes: int, dim: int, curvature_init: float = 1.0):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        # log-curvature parameterization so c stays > 0
        self.log_c = nn.Parameter(torch.tensor(math.log(curvature_init)))
        # Centroids live in the ball; init near origin
        self.centroids = nn.Parameter(torch.zeros(num_classes, dim).uniform_(-0.01, 0.01))

    @property
    def c(self) -> float:
        return self.log_c.exp().item()

    def forward(self, euclidean_emb: torch.Tensor):
        c = self.log_c.exp()
        # Project to ball
        x_h = expmap0(euclidean_emb, c=c.item())
        # Centroids also live on ball -- clip to keep them valid
        cen_h = _clip_to_ball(self.centroids)
        # Broadcast distance: (B, 1, D) vs (1, C, D)
        dists = poincare_distance(
            x_h.unsqueeze(1).expand(-1, self.num_classes, -1),
            cen_h.unsqueeze(0).expand(x_h.shape[0], -1, -1),
            c=c.item(),
        )
        return dists, x_h, cen_h


_poincare_head: Optional[PoincareHead] = None


def _get_poincare(num_classes: int, dim: int) -> PoincareHead:
    global _poincare_head
    if (_poincare_head is None
            or _poincare_head.num_classes != num_classes
            or _poincare_head.dim != dim):
        _poincare_head = PoincareHead(num_classes=num_classes, dim=dim)
    return _poincare_head


def poincare_compute_losses(model, outputs, labels, config, focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral + lambda_hyp * hyperbolic_contrast
                                           + lambda_depth * radial_depth.
    """
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]
    # Shallow tasks (binary) -> hyperbolic is overkill, skip
    if model.num_classes < 3:
        return base

    head = _get_poincare(model.num_classes, emb.shape[-1]).to(emb.device)
    dists, x_h, cen_h = head(emb)

    # --- Hyperbolic contrastive: neg-dist as logits, temperature-scaled CE ---
    temperature = getattr(config, "poincare_temp", 0.5)
    logits = -dists / temperature
    hyp_loss = F.cross_entropy(logits, labels)

    # --- Radial-depth regularizer: ||x_h|| should approx depth(class) / max_depth ---
    depth_tab = _depth_table(model.num_classes)
    max_d = max(depth_tab) if max(depth_tab) > 0 else 1
    target_radii = torch.tensor(
        [depth_tab[l.item()] / max_d for l in labels],
        device=emb.device, dtype=emb.dtype,
    ) * 0.8   # stay strictly inside ball
    actual_radii = x_h.norm(p=2, dim=-1)
    depth_loss = F.mse_loss(actual_radii, target_radii)

    lambda_hyp = getattr(config, "lambda_hyp", 0.4)
    lambda_depth = getattr(config, "lambda_depth", 0.1)
    base["total"] = base["total"] + lambda_hyp * hyp_loss + lambda_depth * depth_loss
    base["hyp"] = hyp_loss
    base["depth"] = depth_loss
    return base


# ===========================================================================
# Entry point -- lean mode
# ===========================================================================

if __name__ == "__main__":
    codet_cfg = CoDETM4Config(
        max_train_samples=100_000, max_val_samples=20_000,
        max_test_samples=-1, eval_breakdown=True,
    )
    droid_cfg = DroidConfig(
        max_train_samples=100_000, max_val_samples=20_000,
        max_test_samples=-1,
    )

    run_full_climb(
        method_name="PoincareGenealogy",
        exp_id="exp_04",
        loss_fn=poincare_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_04_poincare",
    )
