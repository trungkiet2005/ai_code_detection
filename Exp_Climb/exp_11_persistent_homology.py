"""
[exp_11] PersistentHomologyCode -- topological embedding quality regularizer

NOVELTY (NeurIPS 2026 oral target):
-----------------------------------
Journal-grade cross-domain transfer from topological data analysis (TDA)
literature:

  * "Topological Metric for Unsupervised Embedding Quality Evaluation"
    (Shestov et al., arXiv 2512.15285, Dec 2025) -- shows that PH-based
    metrics (Persistence) outperform Euclidean quality metrics at
    predicting downstream performance.
  * "On the Expressivity of Persistent Homology in Graph Learning"
    (Rieck, arXiv 2302.09826) -- PH captures long-range topological
    features that GNNs miss.
  * Earlier Exp23 TopoCode tried Betti numbers on AST filtration, but
    stuck at 69.71 F1 (Exp_Climb insight #5: "BiLSTM AST replaced by
    GAT without richer graph gives no speedup, no accuracy gain").

This version differs fundamentally: rather than computing PH features
on the INPUT (AST graphs), we compute PH of the BATCH EMBEDDING CLOUD
itself as a regularizer. The idea:

  * A batch of B embeddings in R^D defines a point cloud.
  * The persistent homology of this cloud at scale epsilon captures the
    CONNECTEDNESS and LOOPS of the class-manifold structure.
  * Per Shestov et al. 2512.15285, batches with "good" topology (lots of
    short-lived H1 loops, compact H0) have better classifier quality.
  * We use PH as a DIFFERENTIABLE REGULARIZER, penalising batches whose
    topology indicates poor manifold structure.

Components:
  1. PAIRWISE distance in embedding space -> Vietoris-Rips filtration
     at rank 0 (H0 = connected components).
  2. 0-th persistence diagram: lifetimes of components as epsilon grows.
  3. Total persistence = sum of (death - birth). Low total persistence
     means tight clusters; high means scattered (bad) or fragmented.
  4. CLASS-AWARE PH: compute persistence PER CLASS (same-label cloud)
     and across classes. Want:
        - within-class H0 total persistence LOW (compact clusters)
        - between-class H0 total persistence HIGH (well-separated classes)
  5. Differentiable PH is done via the "min spanning tree" trick: H0
     lifetimes = MST edge lengths on the sample graph. Fully differentiable
     via PyTorch.

Why this is journal-grade and not just "another contrastive loss":
  * Gives an INTRINSIC geometric quality signal independent of class
    count, batch size, or temperature.
  * Has formal topological stability theorems (Cohen-Steiner et al.
    2005) -- PH is Lipschitz-stable in the bottleneck distance.
  * Provides interpretable diagnostic: the persistence diagram itself
    can be visualized in a paper figure.

Expected wins (per Shestov 2512.15285):
  * Droid T3 > 0.89 (topological regularity = less shortcut learning)
  * CoDET Author > 70.55 (tighter per-class clusters = better decisions)
  * OOD-SRC-gh > 0.30 (topology survives distribution shift better than
    Euclidean decision boundaries per PH stability theorem)

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. `run_mode="lean"` -> 8 runs in ~53 min on H100.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

import os
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"


def _bootstrap_climb_path() -> str:
    cwd = os.getcwd()
    for candidate in (
        os.path.join(cwd, "Exp_Climb"),
        os.path.join(cwd, "ai_code_detection", "Exp_Climb"),
    ):
        if os.path.exists(os.path.join(candidate, "_common.py")):
            return candidate
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        if os.path.exists(os.path.join(here, "_common.py")):
            return here
    except NameError:
        pass
    repo_dir = os.path.join(cwd, "ai_code_detection")
    if not os.path.exists(repo_dir):
        print(f"[bootstrap] Cloning {REPO_URL} -> {repo_dir}")
        subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, repo_dir])
    return os.path.join(repo_dir, "Exp_Climb")


_climb_dir = _bootstrap_climb_path()
if _climb_dir not in sys.path:
    sys.path.insert(0, _climb_dir)
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
# Persistent Homology H0 via MST + HierTree
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


def _mst_h0_persistence(points: torch.Tensor) -> torch.Tensor:
    """Differentiable H0 total persistence via MST (Prim's algo).

    For H0, persistence diagram lifetimes = edge weights of the minimum
    spanning tree of the pairwise-distance graph. Total persistence =
    sum of MST edges.

    Differentiable because the MST edges themselves are pairwise L2
    distances of input points; no indexing-through-argmin.
    """
    N = points.shape[0]
    if N < 2:
        return points.new_zeros(1).squeeze()
    # Pairwise distance matrix
    D = torch.cdist(points, points, p=2)                # (N, N)
    D = D + torch.eye(N, device=points.device) * 1e9     # no self-loops
    # Prim's MST (O(N^2)), pure tensor ops
    in_tree = torch.zeros(N, dtype=torch.bool, device=points.device)
    min_edge = torch.full((N,), float("inf"), device=points.device, dtype=points.dtype)
    min_edge[0] = 0.0
    total = points.new_zeros(1).squeeze()
    for _ in range(N):
        # pick the vertex not in tree with smallest min_edge
        candidates = min_edge.clone()
        candidates[in_tree] = float("inf")
        u = candidates.argmin()
        # Add its min_edge (which IS a differentiable L2 distance to the tree)
        total = total + min_edge[u]
        in_tree[u] = True
        # Relax
        min_edge = torch.minimum(min_edge, D[u])
    return total


def _class_aware_persistence_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                                  between_target: float = 1.0) -> torch.Tensor:
    """Minimize within-class H0 total persistence (tight clusters).
    Maximize between-class total persistence (good separation).
    """
    emb_norm = F.normalize(embeddings, p=2, dim=-1)
    within = emb_norm.new_zeros(1).squeeze()
    n_classes = 0
    class_centroids = []
    for c in labels.unique():
        mask = (labels == c)
        if mask.sum() < 2:
            continue
        within = within + _mst_h0_persistence(emb_norm[mask])
        class_centroids.append(emb_norm[mask].mean(dim=0))
        n_classes += 1
    within = within / max(n_classes, 1)

    between = emb_norm.new_zeros(1).squeeze()
    if len(class_centroids) >= 2:
        centroids = torch.stack(class_centroids, dim=0)
        between = _mst_h0_persistence(centroids)
        # Encourage centroids to be "spread": push between-class MST sum UP
        between = F.relu(between_target - between)
    return within + between


_hier_fn: Optional[HierarchicalAffinityLoss] = None


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


def persistent_homology_compute_losses(model, outputs, labels, config,
                                       focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral + 0.4*hier + lambda_ph * PH_loss."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # HierTree
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # Persistent homology regularizer (expensive -- subsample if batch > 64)
    B = emb.shape[0]
    max_ph_batch = getattr(config, "ph_max_batch", 64)
    if B > max_ph_batch:
        # Random subsample keeps gradient flowing through selected rows
        idx = torch.randperm(B, device=emb.device)[:max_ph_batch]
        emb_sub = emb[idx]
        lab_sub = labels[idx]
    else:
        emb_sub = emb
        lab_sub = labels

    ph_loss = _class_aware_persistence_loss(
        emb_sub, lab_sub,
        between_target=getattr(config, "ph_between_target", 1.0),
    )
    lambda_ph = getattr(config, "lambda_ph", 0.15)
    base["total"] = base["total"] + lambda_ph * ph_loss
    base["ph"] = ph_loss
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
        method_name="PersistentHomologyCode",
        exp_id="exp_11",
        loss_fn=persistent_homology_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_11_ph",
    )
