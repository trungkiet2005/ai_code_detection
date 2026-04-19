"""
[exp_08] POEMPolarizedCode -- polarized embeddings for source-invariant features

NOVELTY (NeurIPS 2026 oral target):
-----------------------------------
Cross-domain transfer from POEM (Jo & Yoon, arXiv 2305.13046). POEM shows
that splitting representations into INVARIANT + SPECIFIC halves via
orthogonal regularization beats adversarial domain-invariance methods on
DomainBed.

Insight #5 of Exp_Climb/tracker.md: DANN / GRL approaches (Exp19 EAGLECode)
CATASTROPHICALLY fail on author task -- generator-invariant features erase
the author signal. Exp02 GHSourceInvariantCode tries to fix this by grad-
reversing only on STYLE subspace, but adversarial training is still
fragile (IRM penalty explosions, etc.).

POEMPolarizedCode solves this via ORTHOGONAL polarization instead of
adversarial training:

  1. Split 512-dim embedding into TWO halves via learned projection:
        z_inv  = P_inv  @ emb   (256-d: source-invariant, author-relevant)
        z_spec = P_spec @ emb   (256-d: source-specific, author-irrelevant)
  2. ORTHOGONAL CONSTRAINT:
        L_ortho = || P_inv^T P_spec ||_F^2    (force decorrelation)
  3. SOURCE-INFO CONSTRAINT (via cosine, NOT adversarial):
        On z_spec: maximize categorical posterior p(source | z_spec)
            (keep the source info here where it is harmless)
        On z_inv:  minimize H(p(source | z_inv)) via entropy maximization
            (hide source here -- but via entropy-reg, not GRL)
  4. Classifier sees ONLY z_inv -> author decision is source-invariant
     WITHOUT adversarial gradient fragility.

Difference from Exp02:
  * Exp02 uses gradient reversal (fragile, Exp19 lesson).
  * Exp08 uses orthogonal polarization (deterministic, converges cleanly).
  * Both target GH-OOD but via opposite mathematical mechanisms.

Expected wins (per POEM paper + insight #16):
  * OOD-SRC-gh > 0.32 (POEM DomainBed gains generalize to code)
  * CoDET Author IID stable ~ 70.4 (half-rank split, small cost)
  * Droid T3 stable ~ 0.88 (invariant-only head on known task)

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
# POEM Polarized Code
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


class PolarizedProjector(nn.Module):
    """Splits embedding into orthogonal invariant / specific halves."""

    def __init__(self, dim: int, half_dim: int = 256, num_sources: int = 3):
        super().__init__()
        self.P_inv = nn.Linear(dim, half_dim, bias=False)
        self.P_spec = nn.Linear(dim, half_dim, bias=False)
        self.source_head_spec = nn.Linear(half_dim, num_sources)
        self.source_head_inv = nn.Linear(half_dim, num_sources)

    def forward(self, emb: torch.Tensor):
        z_inv = self.P_inv(emb)
        z_spec = self.P_spec(emb)
        return z_inv, z_spec

    def ortho_loss(self) -> torch.Tensor:
        """Frobenius norm of P_inv^T @ P_spec, normalized by max possible."""
        prod = self.P_inv.weight @ self.P_spec.weight.t()  # (half, half)
        return (prod * prod).sum() / (prod.numel() + 1e-8)


_hier_fn: Optional[HierarchicalAffinityLoss] = None
_polar: Optional[PolarizedProjector] = None


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


def _get_polar(dim: int, half_dim: int = 256, num_sources: int = 3) -> PolarizedProjector:
    global _polar
    if (_polar is None
            or _polar.P_inv.in_features != dim
            or _polar.P_inv.out_features != half_dim
            or _polar.source_head_spec.out_features != num_sources):
        _polar = PolarizedProjector(dim=dim, half_dim=half_dim, num_sources=num_sources)
    return _polar


def polarized_compute_losses(model, outputs, labels, config, focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral + 0.4*hier
       + lambda_ortho * P_inv^T P_spec
       + lambda_src_spec * CE(source | z_spec)   (keep source info here)
       + lambda_src_inv  * -H(source | z_inv)    (remove source info here).
    """
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # HierTree always
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # Polarized split only active if source info exists
    sources = outputs.get("sources", None)
    if sources is None or sources.dim() != 1 or sources.numel() != labels.numel():
        return base

    num_sources = int(sources.max().item()) + 1 if sources.numel() > 0 else 3
    num_sources = max(num_sources, 3)
    polar = _get_polar(emb.shape[-1], half_dim=min(256, emb.shape[-1] // 2),
                       num_sources=num_sources).to(emb.device)

    z_inv, z_spec = polar(emb)

    # 1) Orthogonality of projections
    ortho = polar.ortho_loss()

    # 2) z_spec should be PREDICTIVE of source (keep source here)
    src_logits_spec = polar.source_head_spec(z_spec)
    src_ce_spec = F.cross_entropy(src_logits_spec, sources)

    # 3) z_inv should be UNPREDICTIVE of source: maximize entropy of
    #    posterior p(source | z_inv). Note we use entropy reg, NOT GRL.
    src_logits_inv = polar.source_head_inv(z_inv.detach())   # stop-grad into polar
    p_inv = F.softmax(src_logits_inv, dim=-1)
    logp_inv = F.log_softmax(src_logits_inv, dim=-1)
    # Entropy: -sum p * log p. We want this HIGH on z_inv -> minimize -H.
    ent = -(p_inv * logp_inv).sum(dim=-1).mean()
    src_entropy_inv = -ent   # minimize this = maximize entropy

    base["total"] = (
        base["total"]
        + getattr(config, "lambda_ortho", 0.1) * ortho
        + getattr(config, "lambda_src_spec", 0.1) * src_ce_spec
        + getattr(config, "lambda_src_inv", 0.1) * src_entropy_inv
    )
    base["ortho"] = ortho
    base["src_spec"] = src_ce_spec
    base["src_inv"] = src_entropy_inv
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
        method_name="POEMPolarizedCode",
        exp_id="exp_08",
        loss_fn=polarized_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_08_polar",
    )
