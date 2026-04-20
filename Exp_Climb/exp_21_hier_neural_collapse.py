"""
[exp_21] HierNCoE -- Hierarchical Neural Collapse with tangent-space
          orthogonality to crack the Qwen1.5 <-> Nxcode sibling pair.

THEOREM HOOK (EMNLP 2026 Oral):
-------------------------------
Galanti, Poggio et al. "Hierarchical Neural Collapse" (arXiv:2501.09211,
2025): under the hierarchical-label regime, the optimal features form an
equiangular tight frame (ETF) *within each parent simplex*; sibling
children are orthogonal in the TANGENT SPACE of their shared parent.

Our target. The Qwen1.5 (class 5) / Nxcode (class 4) pair accounts for
~38% of all non-human errors across 14+ climb methods (both were
fine-tuned under the codellama-family in HierTree). Current methods
only *pull* siblings together via family affinity; none enforce
*orthogonality* inside the parent subspace.

Components:
  (A) lambda_etf: replace the final linear classifier's weight matrix
      with the fixed ETF simplex (vertices of a regular (C-1)-simplex in
      R^C). Trains only features; classifier is frozen. (Implementation
      note: we add a secondary cosine-head with ETF weights, not swap
      the mutable head, so we can ablate cleanly.)
  (B) lambda_tangent: for each batch, compute the family-mean
      mu_f = mean_{y in family f}(z_y) and the tangent residuals
      z^perp_y = z_y - mu_f. Penalise the inner product
      |<z^perp_qwen, z^perp_nxcode>|^2 averaged over in-batch pairs.
  (C) lambda_hier: standard HierTree family-affinity control (kept for
      direct comparison to Exp_00).

Success gate (lean):
  * Qwen1.5 per-class F1 > 0.50 (break the 0.44 ceiling)
  * CoDET Author IID >= 70.0
  * OOD-SRC-gh >= 0.32 (regression guard)

Kaggle: upload only this file; run_mode="lean" = 8 runs ~3h.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap (identical across climb files)
# ---------------------------------------------------------------------------

import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"
REQUIRED_TOKEN = "_PAPER_BASELINES"


def _runner_has_token(climb_dir: str, token: str) -> bool:
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
                shutil.rmtree(parent, ignore_errors=True)
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        if os.path.exists(os.path.join(here, "_common.py")) and _runner_has_token(here, REQUIRED_TOKEN):
            return here
    except NameError:
        pass
    repo_dir = os.path.join(cwd, "ai_code_detection")
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir, ignore_errors=True)
    subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, repo_dir])
    return os.path.join(repo_dir, "Exp_Climb")


_climb_dir = _bootstrap_climb_path()
if _climb_dir not in sys.path:
    sys.path.insert(0, _climb_dir)
for _mod in list(sys.modules):
    if _mod.startswith(("_climb_runner", "_common", "_trainer",
                        "_data_codet", "_data_droid", "_features",
                        "_model", "_paper_table", "_ablation")):
        del sys.modules[_mod]
print(f"[bootstrap] Exp_Climb path: {_climb_dir}")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import logger
from _trainer import FocalLoss, default_compute_losses
from _data_codet import CoDETM4Config
from _data_droid import DroidConfig
from _climb_runner import run_full_climb
from _ablation import emit_ablation_suite


# ---------------------------------------------------------------------------
# ETF simplex (Papyan, Han, Donoho 2020 + Galanti 2025 hierarchical extension)
# ---------------------------------------------------------------------------

def _simplex_etf(num_classes: int, feat_dim: int, device=None) -> torch.Tensor:
    """Return the (num_classes, feat_dim) matrix whose rows are the vertices
    of the regular (C-1)-simplex scaled to unit norm, placed in the first C-1
    coordinates.  Formula: M = sqrt(C/(C-1)) * (I - 1/C * 11^T) projected to
    the first C-1 dims, then zero-padded to feat_dim.  Rows are unit-norm
    and pairwise inner product = -1/(C-1).
    """
    C = num_classes
    assert feat_dim >= C - 1, f"feat_dim={feat_dim} must be >= C-1={C-1}"
    ones = torch.ones(C, C)
    I = torch.eye(C)
    M = math.sqrt(C / (C - 1.0)) * (I - ones / C)
    # Take the first C-1 eigenvectors of M (SVD) to get a (C, C-1) matrix.
    U, S, _ = torch.linalg.svd(M, full_matrices=False)
    # Keep the top C-1 directions (the last singular value is 0)
    etf = U[:, :C - 1] * S[:C - 1].unsqueeze(0).sqrt()
    # Normalise rows to unit norm
    etf = F.normalize(etf, p=2, dim=-1)
    # Pad with zeros to reach feat_dim
    if feat_dim > C - 1:
        pad = torch.zeros(C, feat_dim - (C - 1))
        etf = torch.cat([etf, pad], dim=-1)
    if device is not None:
        etf = etf.to(device)
    return etf


_etf_cache: dict = {}


def _get_etf(num_classes: int, feat_dim: int, device) -> torch.Tensor:
    key = (num_classes, feat_dim, str(device))
    if key not in _etf_cache:
        _etf_cache[key] = _simplex_etf(num_classes, feat_dim, device=device)
    return _etf_cache[key]


# ---------------------------------------------------------------------------
# HierTree
# ---------------------------------------------------------------------------

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


_hier_fn: Optional[HierarchicalAffinityLoss] = None


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


# ---------------------------------------------------------------------------
# Sibling tangent orthogonality penalty
# ---------------------------------------------------------------------------

def _tangent_orthogonality_loss(emb: torch.Tensor, labels: torch.Tensor,
                                family_table) -> torch.Tensor:
    """For every pair (i,j) of batch samples that share a family but have
    different class labels, compute the family-mean-subtracted vectors and
    penalise their squared inner product.  Implements Galanti 2025 Theorem 2:
    sibling children's tangent-space residuals should be orthogonal.
    """
    B = emb.shape[0]
    if B < 4 or family_table is None:
        return emb.new_zeros(1).squeeze()
    fam = torch.tensor(
        [family_table[l.item()] if l.item() < len(family_table) else -1 for l in labels],
        device=emb.device,
    )
    # Family means (only for samples with valid family)
    unique_fams = fam.unique()
    fam_mu = {}
    for f in unique_fams:
        if f.item() == -1:
            continue
        mask = (fam == f)
        if mask.sum() >= 2:
            fam_mu[f.item()] = emb[mask].mean(dim=0)
    if not fam_mu:
        return emb.new_zeros(1).squeeze()
    z_perp = emb.clone()
    for i in range(B):
        fi = fam[i].item()
        if fi in fam_mu:
            z_perp[i] = emb[i] - fam_mu[fi]
    # Accumulate sibling inner-product penalties
    loss = emb.new_zeros(1).squeeze()
    count = 0
    for i in range(B):
        fi = fam[i].item()
        if fi not in fam_mu:
            continue
        # Find siblings of i (same family, different class)
        siblings = (fam == fi) & (labels != labels[i])
        siblings[i] = False
        if siblings.sum() == 0:
            continue
        zi = F.normalize(z_perp[i], p=2, dim=-1)
        zj = F.normalize(z_perp[siblings], p=2, dim=-1)
        # inner prods (num_siblings,)
        ip = (zj @ zi).pow(2)
        loss = loss + ip.mean()
        count += 1
    return loss / max(count, 1)


def _etf_cosine_ce(emb: torch.Tensor, labels: torch.Tensor, num_classes: int,
                   temperature: float = 0.1) -> torch.Tensor:
    """Cosine classifier with a FROZEN ETF weight matrix.  Equivalent to a
    classifier whose row-geometry is fixed at the simplex ETF -- trains only
    the encoder to place features at the ETF vertices.
    """
    W = _get_etf(num_classes, emb.shape[1], emb.device)   # (C, feat_dim)
    emb_n = F.normalize(emb, p=2, dim=-1)
    logits = emb_n @ W.t() / temperature
    return F.cross_entropy(logits, labels)


ABLATION_TABLE = {
    "hier":    ("lambda_hier",    True),
    "etf":     ("lambda_etf",     True),
    "tangent": ("lambda_tangent", True),
}


def hnc_compute_losses(model, outputs, labels, config,
                      focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier   * HierTree affinity
       + lambda_etf    * ETF cosine CE (fixed classifier geometry)
       + lambda_tangent * Galanti-2025 sibling-tangent orthogonality.
    """
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]
    family_table = _build_family_table(model.num_classes)

    # HierTree
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # ETF classifier on the pooled embedding
    lambda_etf = getattr(config, "lambda_etf", 0.3)
    if lambda_etf > 0 and labels.unique().numel() >= 2:
        etf_loss = _etf_cosine_ce(emb, labels, model.num_classes,
                                  temperature=getattr(config, "etf_temperature", 0.1))
        base["total"] = base["total"] + lambda_etf * etf_loss
        base["etf"] = etf_loss

    # Tangent-space orthogonality
    lambda_tangent = getattr(config, "lambda_tangent", 0.3)
    if lambda_tangent > 0 and family_table is not None:
        tan_loss = _tangent_orthogonality_loss(emb, labels, family_table)
        base["total"] = base["total"] + lambda_tangent * tan_loss
        base["tangent"] = tan_loss

    return base


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
        method_name="HierNCoE",
        exp_id="exp_21",
        loss_fn=hnc_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_21_hnc",
    )

    emit_ablation_suite(
        method_name="HierNCoE",
        exp_id="exp_21",
        loss_fn=hnc_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
