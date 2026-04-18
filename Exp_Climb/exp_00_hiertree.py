"""
[exp_00] HierTreeCode -- baseline climb method

Core novelty (method-specific code kept in this file only):
  Hierarchical Affinity Tree Loss models LLM genealogy as a family tree.
  For each anchor, enforce:
      dist(anchor, same_family) + margin < dist(anchor, diff_family)
  Families: Human / GPT / Llama-family (CodeLlama, Llama3.1) / Qwen-family (Nxcode, Qwen1.5).
  Auto-disables on binary tasks (num_classes == 2).

Data / model / trainer / suite runners are all imported from sibling _*.py
modules in Exp_Climb/ (no code duplication across exp files).

Kaggle workflow:
  1. Upload ONLY this file to Kaggle.
  2. Run. The file auto-clones github.com/trungkiet2005/ai_code_detection and
     imports Exp_Climb/_*.py helpers.
  3. At end of run, search log for `BEGIN_PAPER_TABLE` ... `END_PAPER_TABLE`
     and copy-paste into Exp_Climb/tracker.md.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap: clone repo if we're running outside Exp_Climb/ (e.g. fresh Kaggle)
# ---------------------------------------------------------------------------

import os
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"


def _bootstrap_climb_path() -> str:
    """Find or clone the Exp_Climb folder so sibling _*.py modules are importable.

    Works in three contexts:
      1. Local dev -- file lives inside Exp_Climb/ (uses __file__).
      2. Kaggle notebook upload -- __file__ undefined; clone repo to cwd.
      3. Repo already cloned in cwd -- reuse without re-clone.
    """
    cwd = os.getcwd()

    # Case A: cwd already contains a sibling Exp_Climb/
    for candidate in (
        os.path.join(cwd, "Exp_Climb"),
        os.path.join(cwd, "ai_code_detection", "Exp_Climb"),
    ):
        if os.path.exists(os.path.join(candidate, "_common.py")):
            return candidate

    # Case B: this script lives inside Exp_Climb/ (only when __file__ exists)
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        if os.path.exists(os.path.join(here, "_common.py")):
            return here
    except NameError:
        pass  # running inside a notebook -- __file__ not defined

    # Case C: nothing available -- clone repo
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
# Standard imports from the shared Exp_Climb modules
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
# Method-specific: HierarchicalAffinityLoss
# ===========================================================================

# Author family indices (CoDET-M4 6-class labels: 0=human, 1=codellama, 2=gpt,
#   3=llama3.1, 4=nxcode, 5=qwen1.5)
#   family 0 = human
#   family 1 = llama-family
#   family 2 = gpt
#   family 3 = qwen-family (Nxcode is fine-tuned Qwen -> same family)
AUTHOR_FAMILY_CODET = [0, 1, 2, 1, 3, 3]


def _build_family_table(num_classes: int) -> Optional[list]:
    """Return family-index list for the active task. None disables hier loss."""
    if num_classes == 6:
        return AUTHOR_FAMILY_CODET
    # For Droid T3/T4 (3/4-class human/generated/refined[/adv]) the hierarchy
    # is shallow enough that a "human vs machine-like" split is the only
    # meaningful family structure. Treat label 0 as its own family, rest
    # pooled together. On T1 (2-class) we return None to disable.
    if num_classes == 3:
        return [0, 1, 1]
    if num_classes == 4:
        return [0, 1, 1, 1]
    # AICD T2 (12-class family attribution) would need its own table; skip here.
    return None


class HierarchicalAffinityLoss(nn.Module):
    """Batch-hard triplet loss over family labels (cosine distance)."""

    def __init__(self, margin: float = 0.3, num_classes: int = 6):
        super().__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.family_table = _build_family_table(num_classes)
        self.active = self.family_table is not None

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.active or embeddings.shape[0] < 4:
            return embeddings.new_zeros(1).squeeze()
        family_labels = torch.tensor(
            [self.family_table[l.item()] if l.item() < len(self.family_table) else -1 for l in labels],
            device=labels.device,
        )
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        cos_sim = torch.mm(emb_norm, emb_norm.t())
        dist = 1.0 - cos_sim  # cosine distance in [0, 2]

        loss = embeddings.new_zeros(1).squeeze()
        count = 0
        B = embeddings.shape[0]
        for i in range(B):
            fi = family_labels[i].item()
            if fi == -1:
                continue
            same_mask = (family_labels == fi)
            same_mask[i] = False
            diff_mask = (family_labels != fi) & (family_labels != -1)
            if same_mask.sum() == 0 or diff_mask.sum() == 0:
                continue
            d_pos = dist[i][same_mask].max()      # hardest positive
            d_neg = dist[i][diff_mask].min()      # hardest negative
            triplet = F.relu(d_pos - d_neg + self.margin)
            loss = loss + triplet
            count += 1
        return loss / max(count, 1)


_hier_fn: Optional[HierarchicalAffinityLoss] = None


def _get_hier_loss(num_classes: int, margin: float) -> HierarchicalAffinityLoss:
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


def hiertree_compute_losses(model, outputs, labels, config, focal_loss_fn: Optional[FocalLoss] = None):
    """Focal + 0.3 neural + 0.3 spectral + lambda_hier * hier (HierTreeCode loss)."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    hier_fn = _get_hier_loss(model.num_classes, getattr(config, "hier_margin", 0.3))
    hier_fn = hier_fn.to(outputs["embeddings"].device)
    hier_loss = hier_fn(outputs["embeddings"], labels)
    lambda_hier = getattr(config, "lambda_hier", 0.4)
    base["total"] = base["total"] + lambda_hier * hier_loss
    base["hier"] = hier_loss
    return base


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    # CLIMB protocol: train on ~20% of each dataset, test on FULL test split.
    codet_cfg = CoDETM4Config(
        max_train_samples=100_000,   # ~20% of ~500K CoDET-M4 train
        max_val_samples=20_000,
        max_test_samples=-1,          # FULL test (paper-comparable)
        eval_breakdown=True,
    )
    droid_cfg = DroidConfig(
        max_train_samples=100_000,   # subsample train for data-efficiency claim
        max_val_samples=20_000,
        max_test_samples=-1,          # FULL test
    )

    # run_mode: "full" | "codet_only" | "droid_only" | "codet_iid" | "single"
    run_full_climb(
        method_name="HierTreeCode",
        exp_id="exp_00",
        loss_fn=hiertree_compute_losses,
        codet_cfg=codet_cfg,
        droid_cfg=droid_cfg,
        run_mode="full",
        run_preflight=True,
        checkpoint_tag_prefix="exp_00_hiertree",
    )
