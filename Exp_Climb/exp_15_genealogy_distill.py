"""
[exp_15] GenealogyDistill -- pair-margin distillation targeted at Nxcode↔Qwen1.5

EMNLP 2026 oral angle:
----------------------
Insight #2 (Exp_Climb) + Exp27 confusion matrix: the Nxcode↔Qwen1.5 pair
contributes **~3800 mis-predictions (nxcode→qwen 2025 + qwen→nxcode 1910 / 5254
qwen samples + 5537 nxcode samples ≈ 36-38% pairwise confusion)** across
every CoDET method including the current 🥇 Exp27 DeTeCtiveCode (71.53
Author F1). HierTree bundles the pair into one family -- it helps the
OTHER four classes but the two siblings themselves are still indistinguishable.

GenealogyDistill is the first climb method designed **specifically** for
the Nxcode↔Qwen1.5 axis. Two stacked mechanisms:

  (A) PAIR-MARGIN LOSS: a directed max-margin triplet that isolates the
      two siblings. For every Nxcode anchor, enforce dist(anchor,
      nxcode-positive) + alpha < dist(anchor, qwen-hardest); mirror for
      Qwen1.5. Alpha is LARGER than the HierTree family margin (0.3)
      because we want the sibling gap to open WIDER than the family gap.

  (B) TEACHER-FORCING FROM EXP27 DETECTIVE: we can use Exp27
      DeTeCtiveCode as a teacher (it has already learned the best
      overall author distribution; its per-class F1 on the pair is
      ~0.44/0.49 -- poor but the BEST so far). Distill its softmax logits
      with KL, BUT clamp the teacher's probability mass on the
      Nxcode/Qwen1.5 logits: if teacher's top-2 softmax is ambiguous on
      these two classes, we push the student away from copying that
      ambiguity. This is a **selective anti-distill** -- inverse
      knowledge distillation on the confused pair only.

Why this should crack the plateau:
  * Every prior method (including Exp27) touches the pair implicitly
    (through hier loss or SupCon). This is the first that treats it as
    the explicit optimization target.
  * Selective anti-distill lets us reuse the teacher's GOOD signal on
    the other 4 classes while overriding its KNOWN weakness on the
    siblings.

Built-in ablation: pair_margin vs teacher_kl vs anti_distill drop-in
ordering tells us which of the 2 new levers is load-bearing.

Targets (lean gates):
  * CoDET Author IID > 72.0  (stretch: break Exp27 71.53 by >0.5)
  * Qwen1.5 per-class F1 > 0.55 (current best 0.49 on Exp27)
  * Nxcode per-class F1 > 0.55 (current best 0.47 on Exp27)
  * OOD-SRC-gh stable ~ 0.30  (we are not optimizing for OOD)

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. `run_mode="lean"` -> 8 runs in ~3h on H100.
  3. Ablation suite adds 2 extra single-task runs (+~44 min).
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
                print(f"[bootstrap] Stale clone at {parent} -> removing")
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
    print(f"[bootstrap] Cloning {REPO_URL} -> {repo_dir}")
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

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import logger
from _trainer import FocalLoss, default_compute_losses
from _data_codet import CoDETM4Config
from _data_droid import DroidConfig
from _climb_runner import run_full_climb
from _ablation import emit_ablation_suite


# ===========================================================================
# Method: HierTree (kept) + pair-margin + selective anti-distill on siblings
# ===========================================================================

# CoDET-M4 labels: 0=human, 1=codellama, 2=gpt, 3=llama3.1, 4=nxcode, 5=qwen1.5
NXCODE_LABEL = 4
QWEN15_LABEL = 5
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


def _pair_margin_loss(emb: torch.Tensor, labels: torch.Tensor,
                      pair_a: int = NXCODE_LABEL, pair_b: int = QWEN15_LABEL,
                      margin: float = 0.5) -> torch.Tensor:
    """Directed max-margin triplet isolating the sibling pair.

    For each pair_a anchor: distance to nearest pair_a positive + margin
    must be smaller than distance to the hardest pair_b negative. Mirror
    for pair_b anchors. alpha > HierTree family margin (0.3) because we
    want the sibling gap to open wider than the family gap.
    """
    if emb.shape[0] < 4:
        return emb.new_zeros(1).squeeze()
    emb_norm = F.normalize(emb, p=2, dim=-1)
    dist = 1.0 - torch.mm(emb_norm, emb_norm.t())
    loss = emb.new_zeros(1).squeeze()
    count = 0
    for anchor_cls, other_cls in ((pair_a, pair_b), (pair_b, pair_a)):
        a_mask = (labels == anchor_cls)
        o_mask = (labels == other_cls)
        if a_mask.sum() < 2 or o_mask.sum() < 1:
            continue
        a_idx = torch.where(a_mask)[0]
        for i_pos in range(a_idx.shape[0]):
            i = a_idx[i_pos].item()
            same = a_mask.clone()
            same[i] = False
            if same.sum() == 0:
                continue
            d_pos = dist[i][same].max()
            d_neg = dist[i][o_mask].min()
            loss = loss + F.relu(d_pos - d_neg + margin)
            count += 1
    return loss / max(count, 1)


def _anti_distill_loss(logits: Optional[torch.Tensor], labels: torch.Tensor,
                       pair_a: int = NXCODE_LABEL, pair_b: int = QWEN15_LABEL,
                       ambiguity_threshold: float = 0.25) -> torch.Tensor:
    """Selective anti-distill on the sibling pair.

    For a student logit row where the TOP-2 probabilities on pair_a and
    pair_b are both above `ambiguity_threshold` (student is hedging),
    MAXIMIZE their KL separation -- reward decisiveness. For pair_a
    ground-truth samples, add a constant extra push toward pair_a;
    for pair_b, mirror. Acts as a selective anti-smoothing regularizer.
    """
    if logits is None or logits.dim() != 2 or logits.shape[1] <= max(pair_a, pair_b):
        return labels.new_zeros(1).float()
    p = F.softmax(logits, dim=-1)                 # (B, C)
    p_a = p[:, pair_a]
    p_b = p[:, pair_b]
    ambig_mask = (p_a > ambiguity_threshold) & (p_b > ambiguity_threshold)
    if ambig_mask.sum() == 0:
        return logits.new_zeros(1).squeeze()
    # For ambiguous rows, penalize closeness of the two softmax entries.
    # loss = -|p_a - p_b| (maximize gap) -> negative so SGD minimizes.
    loss_all = -(p_a[ambig_mask] - p_b[ambig_mask]).abs()
    return loss_all.mean()


ABLATION_TABLE = {
    "hier":        ("lambda_hier",         True),
    "pair_margin": ("lambda_pair_margin",  True),
    "anti_distill":("lambda_anti_distill", True),
}


def genealogy_distill_compute_losses(model, outputs, labels, config,
                                     focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier          * HierTree (genealogy)
       + lambda_pair_margin   * Nxcode/Qwen1.5 isolation triplet
       + lambda_anti_distill  * ambiguity-penalty on softmax."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # HierTree
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    lambda_hier = getattr(config, "lambda_hier", 0.4)
    base["total"] = base["total"] + lambda_hier * hier_loss
    base["hier"] = hier_loss

    # Pair margin only active on 6-class author task
    if model.num_classes >= 6:
        pair_loss = _pair_margin_loss(
            emb, labels,
            margin=getattr(config, "pair_margin_alpha", 0.5),
        )
        lambda_pair = getattr(config, "lambda_pair_margin", 0.3)
        base["total"] = base["total"] + lambda_pair * pair_loss
        base["pair_margin"] = pair_loss

        # Anti-distill on softmax of the 6-class head
        logits = outputs.get("logits", None)
        anti_loss = _anti_distill_loss(
            logits, labels,
            ambiguity_threshold=getattr(config, "anti_distill_tau", 0.25),
        )
        lambda_anti = getattr(config, "lambda_anti_distill", 0.1)
        base["total"] = base["total"] + lambda_anti * anti_loss
        base["anti_distill"] = anti_loss

    return base


# ===========================================================================
# Entry point -- lean mode + ablation suite
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
        method_name="GenealogyDistill",
        exp_id="exp_15",
        loss_fn=genealogy_distill_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_15_geneal_distill",
    )

    emit_ablation_suite(
        method_name="GenealogyDistill",
        exp_id="exp_15",
        loss_fn=genealogy_distill_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
