"""
[exp_05] SinkhornOTCode -- optimal transport label assignment for imbalanced author

NOVELTY (NeurIPS 2026 oral target):
-----------------------------------
Cross-domain transfer from "Sparsity-Constrained Optimal Transport"
(Liu et al., arXiv 2209.15466) and "Selective Sinkhorn Routing"
(arXiv 2511.08972, NeurIPS 2025).

Problem: the 6 CoDET authors are MASSIVELY imbalanced at train time:
  human=51% / codellama=11% / gpt=4% / llama3.1=11% / nxcode=12% / qwen1.5=11%

Standard cross-entropy (even with focal) over-predicts human and
under-predicts gpt. Exp18 HierTree has Qwen1.5 at F1 0.44, CodeLlama at
0.41, gpt at ~0.28 -- classic imbalance signature.

SinkhornOTCode replaces the per-sample softmax assignment with a BATCH-
LEVEL Sinkhorn-Knopp optimal transport plan P* that jointly matches the
batch embeddings to the CLASS PROTOTYPES under the constraint that every
class receives an EQUAL share of the batch mass:

    min_P <P, D>_F - epsilon * H(P)
    s.t.  P 1 = 1/B * 1_B   (each sample = 1 row-mass)
          P^T 1 = 1/C * 1_C (each class equal column-mass)

Here D_ij = dist(x_i, prototype_j). Sinkhorn gives a soft target
distribution P*_i per sample that is already balanced -> we minimize
KL(P*_i || softmax(logits_i)).

Why this is different from focal / class-weighted CE:
  * Focal / class-weight scale the LOSS per sample -- still allows mass
    to concentrate on "easy" classes at head time.
  * OT reshapes the TARGET distribution itself so the head MUST allocate
    probability mass uniformly across classes -> minority classes get
    gradient signal proportional to 1/C, not proportional to frequency.

Components:
  * Sinkhorn iterations (3 rounds, differentiable) to compute P* each batch.
  * Class prototypes (EMA-updated, 256-d Euclidean).
  * KL(P* || softmax(logits)) as the new "task" loss.
  * HierTree kept as auxiliary genealogy loss (family structure is
    orthogonal to imbalance -- they stack).

Expected wins (per insight 2 + 7 + OT papers):
  * CoDET Author > 70.6 (balanced head should lift gpt / minority)
  * gpt per-class F1 > 0.45 (current ~0.27-0.30)
  * Droid T4 adversarial stable (OT doesn't touch binary geometry)

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
# Sinkhorn OT + HierTree
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


class PrototypeBank(nn.Module):
    """EMA-updated class prototypes used as Sinkhorn targets."""

    def __init__(self, num_classes: int, dim: int, ema: float = 0.99):
        super().__init__()
        self.num_classes = num_classes
        self.dim = dim
        self.ema = ema
        self.register_buffer("prototypes", torch.zeros(num_classes, dim))
        self.register_buffer("initialized", torch.zeros(num_classes, dtype=torch.bool))

    @torch.no_grad()
    def update(self, embeddings: torch.Tensor, labels: torch.Tensor):
        emb_norm = F.normalize(embeddings.detach(), p=2, dim=-1)
        for c in range(self.num_classes):
            mask = (labels == c)
            if mask.sum() == 0:
                continue
            m = F.normalize(emb_norm[mask].mean(dim=0), p=2, dim=-1)
            if not self.initialized[c]:
                self.prototypes[c] = m
                self.initialized[c] = True
            else:
                self.prototypes[c] = F.normalize(
                    self.ema * self.prototypes[c] + (1 - self.ema) * m, p=2, dim=-1
                )


def sinkhorn(cost: torch.Tensor, n_iter: int = 3, epsilon: float = 0.05) -> torch.Tensor:
    """Differentiable Sinkhorn-Knopp with equal-marginal constraint.

    cost: (B, C) non-negative cost matrix (e.g. 1 - cosine similarity).
    Returns P (B, C) with rows summing to 1/B, cols summing to 1/C.
    """
    B, C = cost.shape
    # Log-domain Sinkhorn for numerical stability
    log_K = -cost / epsilon                              # (B, C)
    # Targets in log-space
    log_u = torch.full((B,), -torch.log(torch.tensor(float(B))), device=cost.device)
    log_v = torch.full((C,), -torch.log(torch.tensor(float(C))), device=cost.device)

    log_a = torch.zeros(B, device=cost.device)
    log_b = torch.zeros(C, device=cost.device)
    for _ in range(n_iter):
        # Row-normalize
        log_a = log_u - torch.logsumexp(log_K + log_b.unsqueeze(0), dim=1)
        # Col-normalize
        log_b = log_v - torch.logsumexp(log_K + log_a.unsqueeze(1), dim=0)

    log_P = log_a.unsqueeze(1) + log_K + log_b.unsqueeze(0)
    return log_P.exp()   # (B, C)


_proto_bank: Optional[PrototypeBank] = None
_hier_fn: Optional[HierarchicalAffinityLoss] = None


def _get_bank(num_classes: int, dim: int) -> PrototypeBank:
    global _proto_bank
    if _proto_bank is None or _proto_bank.num_classes != num_classes or _proto_bank.dim != dim:
        _proto_bank = PrototypeBank(num_classes=num_classes, dim=dim)
    return _proto_bank


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


def sinkhorn_ot_compute_losses(model, outputs, labels, config, focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral + 0.4*hier + 0.5*sinkhorn_kl."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # --- HierTree genealogy (preserved) ---
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # --- Sinkhorn-OT balanced target ---
    if model.num_classes < 3 or emb.shape[0] < model.num_classes * 2:
        return base   # not enough batch for OT to make sense

    bank = _get_bank(model.num_classes, emb.shape[-1]).to(emb.device)
    bank.update(emb, labels)

    if not bank.initialized.all():
        return base   # need at least one sample of each class in the bank

    emb_norm = F.normalize(emb, p=2, dim=-1)
    cost = 1.0 - torch.mm(emb_norm, bank.prototypes.t())   # (B, C) in [0, 2]
    with torch.no_grad():
        P = sinkhorn(
            cost.detach(), n_iter=getattr(config, "sinkhorn_iters", 3),
            epsilon=getattr(config, "sinkhorn_epsilon", 0.05),
        )
        # Normalize rows to sum to 1 (they sum to 1/B after Sinkhorn) for KL target
        P = P * P.shape[0]
        P = P.clamp_min(1e-8)
        P = P / P.sum(dim=-1, keepdim=True)

    # Get logits from model output, fall back to dot-product if needed
    logits = outputs.get("logits", None)
    if logits is None:
        logits = torch.mm(emb_norm, bank.prototypes.t())

    log_pred = F.log_softmax(logits, dim=-1)
    sinkhorn_loss = F.kl_div(log_pred, P, reduction="batchmean")

    lambda_sinkhorn = getattr(config, "lambda_sinkhorn", 0.5)
    base["total"] = base["total"] + lambda_sinkhorn * sinkhorn_loss
    base["sinkhorn"] = sinkhorn_loss
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
        method_name="SinkhornOTCode",
        exp_id="exp_05",
        loss_fn=sinkhorn_ot_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_05_sinkhorn",
    )
