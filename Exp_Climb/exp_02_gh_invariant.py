"""
[exp_02] GHSourceInvariantCode -- source-as-environment IRM + style-invariant projection

NOVELTY (NeurIPS 2026 oral target):
-----------------------------------
Insight #16 of Exp_Climb/tracker.md: GitHub source OOD is the biggest
unsolved lever in CoDET-M4. Every method so far reports OOD-SRC held-out=gh
at macro F1 ~0.28 (human recall 5.71%). The root cause: CF + LC are narrow
competitive-programming style, GH is real-world diverse -- the model
memorises the narrow-style shortcut.

Standard DANN (Exp19 EAGLECode) catastrophically fails for author tasks:
generator-invariant features erase the author signal. We instead apply
**SOURCE-invariant** features (cf / gh / lc) while keeping generator info
sharp. Two coupled pieces:

  1. SOURCE-IRM: treat `source` as an environment variable. Annealed
     IRMv1 penalty per source, so the classifier's risk-optimal weights
     are source-agnostic. Unlike IRM on languages (Exp06 failed, too few
     envs), source has 3 natural envs with >20K samples each -> well-posed.

  2. STYLE-ADVERSARIAL PROJECTION: a light adversary predicts source from
     the [style-only] 256-dim subspace. Gradient reversal makes the STYLE
     branch source-invariant. The CONTENT branch (256-dim) is NOT
     adversarially trained -- author signal preserved.

Crucially, the static family tree (HierTree) is kept intact: we add source
invariance ON TOP of proven genealogy modelling, rather than replacing it.

Expected wins (per insight 16 + 3):
  * OOD-SRC held-out=gh > 0.30 (break the 0.28 ceiling)
  * CoDET Author IID stays >= 70.3 (don't regress)
  * Droid T3 >= 0.88 (source invariance shouldn't hurt adversarial)

If OOD-SRC-gh > 0.35, this is paper-headline material: first method to
materially improve the hardest GH subgroup.

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. `run_mode="lean"` -> 8 runs in ~53 min on H100.
  3. Copy BEGIN_PAPER_TABLE ... END_PAPER_TABLE into Exp_Climb/tracker.md.
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
from torch.autograd import Function

from _common import logger
from _trainer import FocalLoss, default_compute_losses
from _data_codet import CoDETM4Config
from _data_droid import DroidConfig
from _climb_runner import run_full_climb


# ===========================================================================
# Method-specific: HierTree (kept) + SourceIRM + StyleAdversary
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
    """Same batch-hard triplet over families as Exp18. Kept to preserve genealogy wins."""

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


class GradReverse(Function):
    @staticmethod
    def forward(ctx, x, lambda_):
        ctx.lambda_ = lambda_
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -ctx.lambda_ * grad_output, None


def grad_reverse(x, lambda_=1.0):
    return GradReverse.apply(x, lambda_)


class StyleSourceAdversary(nn.Module):
    """Light MLP that predicts source (cf/gh/lc) from [style-only] subspace.

    Gradient-reversed -> style branch becomes source-invariant.
    """

    def __init__(self, style_dim: int = 256, num_sources: int = 3, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(style_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden, num_sources),
        )

    def forward(self, style_feats: torch.Tensor, lambda_: float = 1.0) -> torch.Tensor:
        return self.net(grad_reverse(style_feats, lambda_))


_hier_fn: Optional[HierarchicalAffinityLoss] = None
_adversary: Optional[StyleSourceAdversary] = None


def _get_hier(num_classes: int, margin: float) -> HierarchicalAffinityLoss:
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


def _get_adversary(style_dim: int, num_sources: int = 3) -> StyleSourceAdversary:
    global _adversary
    if _adversary is None or _adversary.net[0].in_features != style_dim:
        _adversary = StyleSourceAdversary(style_dim=style_dim, num_sources=num_sources)
    return _adversary


def _compute_irm_penalty(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """IRMv1 penalty: ||grad_w CE(w * logits, y)||^2 evaluated at w=1."""
    scale = torch.tensor(1.0, device=logits.device, requires_grad=True)
    loss = F.cross_entropy(logits * scale, labels)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return (grad ** 2).sum()


def gh_invariant_compute_losses(model, outputs, labels, config, focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral + 0.4*hier + 0.3*irm_source + 0.1*style_adv."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # --- HierTree (unchanged genealogy signal) ---
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    lambda_hier = getattr(config, "lambda_hier", 0.4)
    base["total"] = base["total"] + lambda_hier * hier_loss
    base["hier"] = hier_loss

    # --- Source-IRM + Style-Adversary: only active if batch has source metadata ---
    sources = outputs.get("sources", None)   # (B,) long tensor in {0,1,2} if available
    if sources is not None and sources.dim() == 1 and sources.numel() == labels.numel():
        # SOURCE-IRM: partition batch by source, compute IRM penalty per env.
        task_logits = outputs.get("logits", None)
        if task_logits is not None:
            irm_loss = task_logits.new_zeros(1).squeeze()
            n_envs = 0
            for src in torch.unique(sources):
                mask = (sources == src)
                if mask.sum() < 4:
                    continue
                irm_loss = irm_loss + _compute_irm_penalty(task_logits[mask], labels[mask])
                n_envs += 1
            irm_loss = irm_loss / max(n_envs, 1)
            lambda_irm = getattr(config, "lambda_irm", 0.3)
            # Annealed: scale by min(epoch / warm, 1.0) -- smooth ramp-up
            epoch = getattr(config, "_current_epoch", 1)
            warm = getattr(config, "irm_warmup_epochs", 1)
            scale = min(epoch / max(warm, 1), 1.0)
            base["total"] = base["total"] + lambda_irm * scale * irm_loss
            base["irm_source"] = irm_loss

        # STYLE-ADVERSARY: grad-reverse on style subspace (first 256 dims of emb)
        style_dim = min(256, emb.shape[-1])
        style = emb[:, :style_dim]
        adv = _get_adversary(style_dim=style_dim, num_sources=3).to(emb.device)
        src_logits = adv(style, lambda_=getattr(config, "adv_lambda", 1.0))
        adv_loss = F.cross_entropy(src_logits, sources)
        lambda_adv = getattr(config, "lambda_adv", 0.1)
        base["total"] = base["total"] + lambda_adv * adv_loss
        base["style_adv"] = adv_loss

    return base


# ===========================================================================
# Entry point -- lean mode
# ===========================================================================

if __name__ == "__main__":
    codet_cfg = CoDETM4Config(
        max_train_samples=100_000,
        max_val_samples=20_000,
        max_test_samples=-1,
        eval_breakdown=True,
    )
    droid_cfg = DroidConfig(
        max_train_samples=100_000,
        max_val_samples=20_000,
        max_test_samples=-1,
    )

    run_full_climb(
        method_name="GHSourceInvariantCode",
        exp_id="exp_02",
        loss_fn=gh_invariant_compute_losses,
        codet_cfg=codet_cfg,
        droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_02_ghinv",
    )
