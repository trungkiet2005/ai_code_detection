"""
[exp_06] FlowCodeDet -- flow-matching auxiliary head for discriminative regularization

NOVELTY (NeurIPS 2026 oral target):
-----------------------------------
Cross-domain transfer from "Flow Matching in the Low-Noise Regime"
(Zeng & Yan, arXiv 2509.20952, Sept 2025) and "High-Performance SSL by
Joint Training of Flow Matching" (Ukita & Okita, arXiv 2512.19729,
Dec 2025). Both show that flow-matching training produces STRONGER
discriminative representations than pure contrastive SSL, because the
model must learn *per-class velocity fields* that are locally contractive.

No prior AI-code-detection work uses flow matching. We adapt the SSL
recipe to a DISCRIMINATIVE setting:

  1. A small velocity-field MLP v_theta(x, t, y): given a corrupted
     embedding x_t = (1-t)*eps + t*embedding(code), class-conditioned
     on y, predict the target velocity embedding - eps.
  2. Training adds the flow-matching loss:
        L_fm = E_{t, eps} || v_theta(x_t, t, y) - (embedding - eps) ||^2
     This forces embeddings to sit on CLASS-CONDITIONED trajectories --
     a much stronger inductive bias than "same class pulled together"
     (classical SupCon).
  3. CRITICAL: the velocity head is CLASS-conditioned via a small
     embedding table (num_classes x 64). This makes the auxiliary task
     LABEL-AWARE, so it DIRECTLY regularizes the discriminative space
     (unlike unconditional FM which is purely generative).

Why this should outperform Exp18 HierTree + all SupCon variants:
  * Per-class velocity field = implicit class manifold -> stronger than
    pairwise contrastive signal.
  * Works for ANY num_classes (binary, 3, 6, 12) with zero code change.
  * Natural stochastic regularization via the `t` sampling -- acts as
    a mild adversarial augmentation without needing input-space attacks.

Components:
  * CondVelocityMLP(dim=256, class_dim=64): 3-layer MLP with class cond.
  * Linear interpolant x_t = (1 - t) * noise + t * embedding.
  * Loss term lambda_fm * ||v_theta(x_t, t, y) - target||^2.
  * Time sampling: t ~ U(0, 1) (linear) or Beta(1.5, 1) (late-skewed, per
    low-noise paper recommendation for fine-grained tasks).

Expected wins (per insights 6, 7 + FM papers):
  * CoDET Author > 70.6 (flow regularization > SupCon per FM papers)
  * Droid T4 adversarial > 0.85 (velocity field should be robust to pert.)
  * OOD-GEN-qwen1.5 > 0.51 (per-class manifold helps OOD, per paper claims)

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
REQUIRED_TOKEN = "_PAPER_BASELINES"  # bump when _climb_runner or _paper_table APIs change   # bump this when the climb runner API changes


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
# FlowCodeDet -- class-conditioned flow-matching auxiliary head
# ===========================================================================

AUTHOR_FAMILY_CODET = [0, 1, 2, 1, 3, 3]


def _build_family_table(num_classes: int):
    if num_classes == 6:
        return AUTHOR_FAMILY_CODET
    if num_classes == 3:
        return [0, 1, 1]
    if num_classes == 4:
        return [0, 1, 1, 1]
    return None


class HierarchicalAffinityLoss(nn.Module):
    def __init__(self, margin: float = 0.3, num_classes: int = 6):
        super().__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.family_table = _build_family_table(num_classes)
        self.active = self.family_table is not None

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.active or embeddings.shape[0] < 4:
            return embeddings.new_zeros(1).squeeze()
        fam = torch.tensor(
            [self.family_table[l.item()] if l.item() < len(self.family_table) else -1 for l in labels],
            device=labels.device,
        )
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        dist = 1.0 - torch.mm(emb_norm, emb_norm.t())
        loss = embeddings.new_zeros(1).squeeze()
        count = 0
        for i in range(embeddings.shape[0]):
            fi = fam[i].item()
            if fi == -1:
                continue
            same = (fam == fi); same[i] = False
            diff = (fam != fi) & (fam != -1)
            if same.sum() == 0 or diff.sum() == 0:
                continue
            d_pos = dist[i][same].max()
            d_neg = dist[i][diff].min()
            loss = loss + F.relu(d_pos - d_neg + self.margin)
            count += 1
        return loss / max(count, 1)


def _sinusoidal_time_embed(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    """Sinusoidal positional embedding for time t in [0, 1]."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class CondVelocityMLP(nn.Module):
    """v_theta(x, t, y): class-conditioned velocity field."""

    def __init__(self, dim: int = 256, num_classes: int = 6,
                 class_dim: int = 64, time_dim: int = 32, hidden: int = 512):
        super().__init__()
        self.dim = dim
        self.num_classes = num_classes
        self.class_embed = nn.Embedding(num_classes, class_dim)
        self.time_dim = time_dim
        in_dim = dim + class_dim + time_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.class_embed(y)
        t_embed = _sinusoidal_time_embed(t, dim=self.time_dim)
        h = torch.cat([x_t, c, t_embed], dim=-1)
        return self.net(h)


_vel_mlp: Optional[CondVelocityMLP] = None
_hier_fn: Optional[HierarchicalAffinityLoss] = None


def _get_vel(dim: int, num_classes: int) -> CondVelocityMLP:
    global _vel_mlp
    if (_vel_mlp is None
            or _vel_mlp.dim != dim
            or _vel_mlp.num_classes != num_classes):
        _vel_mlp = CondVelocityMLP(dim=dim, num_classes=num_classes)
    return _vel_mlp


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


def _flow_matching_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                        vel: CondVelocityMLP,
                        time_skew_beta: float = 1.0) -> torch.Tensor:
    """Class-conditioned flow matching loss between Gaussian noise and embeddings."""
    B, D = embeddings.shape
    # Target velocity under linear interpolant is simply (x_1 - x_0) = emb - noise
    noise = torch.randn_like(embeddings)
    if time_skew_beta != 1.0:
        # Beta-skewed t toward late schedule (low-noise regime, Zeng&Yan 2509.20952)
        t = torch.distributions.Beta(time_skew_beta, 1.0).sample((B,)).to(embeddings.device)
    else:
        t = torch.rand(B, device=embeddings.device)
    t_col = t.unsqueeze(-1)                            # (B, 1)
    x_t = (1 - t_col) * noise + t_col * embeddings      # (B, D)
    v_target = embeddings - noise                       # (B, D)
    v_pred = vel(x_t, t, labels)
    return F.mse_loss(v_pred, v_target)


def flow_compute_losses(model, outputs, labels, config, focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral + 0.4*hier + lambda_fm * flow_match."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # --- HierTree (preserved) ---
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # --- Flow matching auxiliary ---
    vel = _get_vel(emb.shape[-1], model.num_classes).to(emb.device)
    fm_loss = _flow_matching_loss(
        emb, labels, vel,
        time_skew_beta=getattr(config, "fm_time_beta", 1.5),
    )
    lambda_fm = getattr(config, "lambda_fm", 0.3)
    base["total"] = base["total"] + lambda_fm * fm_loss
    base["flow"] = fm_loss
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
        method_name="FlowCodeDet",
        exp_id="exp_06",
        loss_fn=flow_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_06_flow",
    )
