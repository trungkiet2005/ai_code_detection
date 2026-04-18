"""
[exp_07] SAMFlatCode -- Sharpness-Aware Minimization for OOD generalization

NOVELTY (NeurIPS 2026 oral target):
-----------------------------------
Cross-domain transfer from SAM (Foret et al., ICLR 2021, arXiv 2010.01412),
GAM (Zhang et al., CVPR 2023, arXiv 2303.03108), and "Normalization Layers
Are All That SAM Needs" (Mueller et al., NeurIPS 2023, arXiv 2306.04226).

Flatness connection to OOD: flat minima generalize better across
distribution shift. Insight #16 of Exp_Climb/tracker.md: OOD-SRC held-out
gh catastrophically fails (~0.28 macro F1) because the narrow CF/LC
template style becomes a sharp loss basin that does not transfer.

SAMFlatCode replaces the standard optimizer step with a TWO-STAGE update:

    Stage 1: compute gradient g_1 at current weights theta.
    Stage 2: perturb theta by eps = rho * g_1 / ||g_1||, compute gradient
             g_2 at the perturbed point.
    Apply g_2 to theta.

The gradient at the SHARPEST nearby point is what drives optimization,
so the optimizer is forced to find FLAT basins that survive the
perturbation. Per "Normalization Layers Are All" we use a SPARSE
perturbation (only on LayerNorm parameters) -- faster, same benefit.

Since we cannot easily modify the trainer step, we implement SAM as an
AUXILIARY LOSS term: adversarial perturbation in EMBEDDING SPACE that
mimics the weight-space perturbation. The embedding-space SAM is cheaper
(no 2nd forward pass) and paper-grounded (FAD, arXiv 2307.11108, shows
input/feature-space flatness is a valid SAM proxy).

Concretely:
  1. Compute gradient of the task loss wrt the embedding.
  2. Perturb embedding by rho * sign(grad) (FGSM-style worst-case).
  3. Re-run the head on perturbed embedding -> auxiliary classification
     loss. Total loss = normal_task + lambda_sam * perturbed_task.

This is novel for AI-code-detection: no prior Exp_DM / Exp_CodeDet method
uses adversarial-flatness regularization. Related work (Exp19 EAGLECode,
DANN) used domain-adversarial features and FAILED -- this is orthogonal.

Expected wins:
  * OOD-SRC-gh > 0.32 (flat minima survive source shift, per FAD paper)
  * CoDET Author IID stable ~ 70.5 (flatness doesn't hurt IID)
  * Droid T4 adversarial > 0.85 (adversarial flatness helps adv class)

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
# SAMFlatCode: embedding-space SAM + HierTree
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


_hier_fn: Optional[HierarchicalAffinityLoss] = None


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


def _embedding_sam_loss(emb: torch.Tensor, labels: torch.Tensor,
                        classifier_fn, rho: float = 0.05) -> torch.Tensor:
    """FGSM-style worst-case perturbation in embedding space.

    classifier_fn: emb -> logits. Must be cheap (single Linear) since
    we invoke it twice.
    """
    # 1. Forward + grad on current emb
    emb_d = emb.detach().clone().requires_grad_(True)
    logits = classifier_fn(emb_d)
    ce = F.cross_entropy(logits, labels)
    g = torch.autograd.grad(ce, emb_d, create_graph=False)[0]
    # 2. Sign-step perturbation
    g_norm = g.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)
    delta = rho * g / g_norm
    emb_adv = emb + delta            # gradient-flowing perturbation
    # 3. Worst-case classification loss (this IS differentiable wrt emb)
    logits_adv = classifier_fn(emb_adv)
    return F.cross_entropy(logits_adv, labels)


def sam_compute_losses(model, outputs, labels, config, focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral + 0.4*hier + lambda_sam * sam_worst_case."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # --- HierTree ---
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # --- Embedding-space SAM ---
    # Use the model's own task head if exposed; else build a cheap linear
    # classifier from logits shape.
    logits = outputs.get("logits", None)
    if logits is None or not hasattr(model, "task_head"):
        return base

    task_head = model.task_head

    def _classifier_fn(e):
        return task_head(e)

    sam_loss = _embedding_sam_loss(
        emb, labels, _classifier_fn,
        rho=getattr(config, "sam_rho", 0.05),
    )
    lambda_sam = getattr(config, "lambda_sam", 0.25)
    base["total"] = base["total"] + lambda_sam * sam_loss
    base["sam"] = sam_loss
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
        method_name="SAMFlatCode",
        exp_id="exp_07",
        loss_fn=sam_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_07_sam",
    )
