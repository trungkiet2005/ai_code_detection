"""
[Exp_TK exp08] DualModeFlowRAG-HN -- stacked flow + dual-SupCon + hard-neg mining

Challenger for overall SOTA (cross-bench triple threshold).
----------------------------------------------------------
Builds on **Exp_16 DualModeFlowRAG** (Climb pending) which itself stacks
the two highest-signal winners on the consolidated board (Exp_TK/tracker.md):

  * Exp_06 FlowCodeDet (Climb 🥇, Author 70.90) - class-conditioned
    velocity field regularizer. Wins IID + OOD-LANG-py + OOD-GH.
  * Exp27 DeTeCtiveCode (CodeDet 🥇, Author 71.53) - HierTree + dual
    (neural + spectral) SupCon + test-time kNN blend.

What exp08 adds over Exp_16 (the loss-fn-level extension):
----------------------------------------------------------
  * **Hard-negative mining inside SupCon** (SupCon+HN, NeurIPS 2020
    addendum + ICML 2021 hard-sample line). For each anchor, keep the
    top-K closest negatives (cosine) as the contrastive denominator
    instead of marginalizing over ALL in-batch negatives. This
    concentrates gradient on the genuinely confusable pairs -- exactly
    the Nxcode <-> Qwen1.5 siblings that Exp_15 tries to fix via a
    different mechanism. Drop-in replacement for the dual SupCon term.
  * Slightly stronger FM regularization (lambda_fm 0.35 vs 0.3) and
    supcon weight (0.20 vs 0.15) -- the Exp_TK tracker insight #6 says
    stacked aux losses respond well to moderate upweighting when each
    attacks a different axis.

Loss:
  focal + 0.3*neural + 0.3*spectral
   + lambda_hier     * HierTree
   + lambda_fm       * class-cond flow matching (Beta(1.5, 1) time skew)
   + lambda_supcon_n * SupCon+HN on neural projection
   + lambda_supcon_s * SupCon+HN on spectral projection

Ablation toggles (via _ablation suite): hier / flow / supcon_neural /
supcon_spec, plus a 5th toggle to swap HN SupCon <-> vanilla SupCon to
measure the HN lift in isolation.

Targets (tracker section 7, "Success bars"):
  * CoDET Author IID > 71.53  (beat Exp27 DeTeCtiveCode)
  * CoDET OOD-SRC-gh > 33.36  (beat Exp_06 / Exp_08 tie)
  * Droid T3 W-F1     > 0.8941 (beat exp09 TokenStat)

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. `run_mode="lean"` -> 8 runs + ablation, fits in H100 12h budget.
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


# ===========================================================================
# HierTree (preserved from Exp_16 / Exp18)
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


# ===========================================================================
# Class-conditioned flow matching (preserved from Exp_06 / Exp_16)
# ===========================================================================

def _sinusoidal_time_embed(t: torch.Tensor, dim: int = 32) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=t.device, dtype=t.dtype) / half
    )
    args = t.unsqueeze(-1) * freqs.unsqueeze(0) * 1000.0
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class CondVelocityMLP(nn.Module):
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


def _flow_matching_loss(embeddings: torch.Tensor, labels: torch.Tensor,
                        vel: CondVelocityMLP,
                        time_skew_beta: float = 1.5) -> torch.Tensor:
    B, D = embeddings.shape
    noise = torch.randn_like(embeddings)
    if time_skew_beta != 1.0:
        t = torch.distributions.Beta(time_skew_beta, 1.0).sample((B,)).to(embeddings.device)
    else:
        t = torch.rand(B, device=embeddings.device)
    t_col = t.unsqueeze(-1)
    x_t = (1 - t_col) * noise + t_col * embeddings
    v_target = embeddings - noise
    v_pred = vel(x_t, t, labels)
    return F.mse_loss(v_pred, v_target)


# ===========================================================================
# NEW: Supervised Contrastive with HARD-NEGATIVE mining (exp08 addition)
# ===========================================================================

def _supcon_loss_hn(features: torch.Tensor, labels: torch.Tensor,
                    temperature: float = 0.1,
                    topk_neg: int = 32) -> torch.Tensor:
    """SupCon where the denominator is restricted to the top-k hardest
    (closest in cosine) negatives per anchor.

    Rationale: vanilla SupCon marginalizes over ALL in-batch negatives;
    at batch 128 that is ~106 negatives per class-mate. The gradient
    signal is diluted by "easy" negatives that already have low
    similarity. Hard-negative mining (SupCon+HN) keeps only the top-k
    closest off-class samples -- concentrates the gradient on genuinely
    confusable pairs (Nxcode <-> Qwen1.5, codellama <-> llama3.1).

    Setting `topk_neg <= 0` recovers vanilla SupCon for ablation.
    """
    if features.shape[0] < 4:
        return features.new_zeros(1).squeeze()
    z = F.normalize(features, p=2, dim=-1)
    logits = torch.mm(z, z.t()) / temperature
    B = z.shape[0]
    logits = logits - torch.eye(B, device=z.device) * 1e9

    labels_row = labels.unsqueeze(0)
    labels_col = labels.unsqueeze(1)
    pos_mask = (labels_row == labels_col).float()
    pos_mask.fill_diagonal_(0.0)
    neg_mask = 1.0 - pos_mask
    neg_mask.fill_diagonal_(0.0)

    if topk_neg is not None and topk_neg > 0 and topk_neg < B - 1:
        # Keep top-k largest (== hardest) negative logits per row.
        neg_logits = logits.masked_fill(neg_mask == 0.0, float("-inf"))
        k = min(topk_neg, B - 1)
        _, hard_idx = neg_logits.topk(k, dim=-1)           # (B, k)
        hard_mask = torch.zeros_like(logits)
        hard_mask.scatter_(1, hard_idx, 1.0)
        # Final denominator mask: positives + hard negatives (no self).
        keep_mask = (pos_mask + hard_mask).clamp_max(1.0)
        logits = logits.masked_fill(keep_mask == 0.0, float("-inf"))

    log_p = F.log_softmax(logits, dim=-1)
    pos_per_row = pos_mask.sum(dim=-1)
    safe_rows = pos_per_row > 0
    if safe_rows.sum() == 0:
        return features.new_zeros(1).squeeze()
    mean_log_prob_pos = (pos_mask * log_p).sum(dim=-1) / pos_per_row.clamp_min(1.0)
    return -(mean_log_prob_pos[safe_rows]).mean()


# ===========================================================================
# Singletons (build once per run, reuse across batches)
# ===========================================================================

_hier_fn: Optional[HierarchicalAffinityLoss] = None
_vel_mlp: Optional[CondVelocityMLP] = None


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


def _get_vel(dim: int, num_classes: int) -> CondVelocityMLP:
    global _vel_mlp
    if (_vel_mlp is None or _vel_mlp.dim != dim
            or _vel_mlp.num_classes != num_classes):
        _vel_mlp = CondVelocityMLP(dim=dim, num_classes=num_classes)
    return _vel_mlp


# ===========================================================================
# Ablation toggles (drop-in: _ablation.emit_ablation_suite uses this)
# ===========================================================================

ABLATION_TABLE = {
    "hier":          ("lambda_hier",      True),
    "flow":          ("lambda_fm",        True),
    "supcon_neural": ("lambda_supcon_n",  True),
    "supcon_spec":   ("lambda_supcon_s",  True),
    # Drop this to disable HN mining (reverts to vanilla SupCon):
    "hardneg":       ("supcon_topk_neg",  True),
}


# ===========================================================================
# Loss
# ===========================================================================

def dualmode_hn_compute_losses(model, outputs, labels, config,
                               focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier       * HierTree
       + lambda_fm         * class-cond flow matching
       + lambda_supcon_n   * SupCon+HN on neural projection
       + lambda_supcon_s   * SupCon+HN on spectral projection."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # --- HierTree (family genealogy) ---
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # --- Flow matching (per-class trajectory regularizer) ---
    vel = _get_vel(emb.shape[-1], model.num_classes).to(emb.device)
    fm_loss = _flow_matching_loss(
        emb, labels, vel,
        time_skew_beta=getattr(config, "fm_time_beta", 1.5),
    )
    base["total"] = base["total"] + getattr(config, "lambda_fm", 0.35) * fm_loss
    base["flow"] = fm_loss

    # --- Dual-level SupCon + HN on neural/spectral projection heads ---
    neural_feat = outputs.get("neural_features", emb)
    spectral_feat = outputs.get("spectral_features", emb)
    topk_neg = int(getattr(config, "supcon_topk_neg", 32) or 0)
    temp = getattr(config, "supcon_temp", 0.1)
    supcon_n = _supcon_loss_hn(neural_feat, labels, temperature=temp, topk_neg=topk_neg)
    supcon_s = _supcon_loss_hn(spectral_feat, labels, temperature=temp, topk_neg=topk_neg)
    base["total"] = (
        base["total"]
        + getattr(config, "lambda_supcon_n", 0.20) * supcon_n
        + getattr(config, "lambda_supcon_s", 0.20) * supcon_s
    )
    base["supcon_n"] = supcon_n
    base["supcon_s"] = supcon_s
    return base


# ===========================================================================
# Entry point
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
        method_name="DualModeFlowRAG-HN",
        exp_id="exp08",
        loss_fn=dualmode_hn_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp08_dualmode_hn",
    )

    emit_ablation_suite(
        method_name="DualModeFlowRAG-HN",
        exp_id="exp08",
        loss_fn=dualmode_hn_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
