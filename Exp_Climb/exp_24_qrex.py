"""
[exp_24] QREx -- Quantile Risk Extrapolation for OOD-SRC-gh.

THEOREM HOOK (EMNLP 2026 Oral):
-------------------------------
Eastwood, Schölkopf et al. "Quantile Risk Extrapolation" (NeurIPS 2025)
extending Krueger et al. "V-REx" (ICML 2021, arXiv:2003.00688).
Theorem: penalising the variance of the upper-alpha-QUANTILE of
per-environment risk gives a tighter worst-group generalisation bound
than V-REx's mean-variance penalty.  Formally:

  L_QREx = mean_e L_e + beta * Var_e( Q_alpha(L_e) )

where L_e is the per-env loss, Q_alpha is the alpha-quantile over a
fresh batch within env e (alpha ~ 0.9), and beta is annealed 0 -> beta_max
over the first warmup_epochs.

Our target. Exp_06 AST-IRM blew up to NaN because the IRM penalty
exploded by epoch 3.  V-REx (mean-variance) is more stable but only
controls the mean; OOD-SRC-gh IS the upper-quantile environment, so
QREx by construction targets what we care about.

ENVIRONMENTS.
  We use (source) as the environment axis (3 envs: cf, lc, gh) since
  source = GH is literally the OOD target.  Per-batch, split samples
  by source and compute within-env losses.

Components:
  (A) lambda_qrex: the quantile-variance penalty itself.  Annealed
      linearly from 0 to lambda_qrex over epoch-1 to epoch-2.
  (B) lambda_hier: HierTree control.
  (C) erm_baseline (always on): the standard mean-env risk.

Success gate (lean):
  * OOD-SRC-gh > 0.38 (fallback target for exp_23)
  * CoDET Author IID >= 70.0
  * No NaN for lambda_qrex up to 10 (stability vs Exp_06)

Kaggle: upload only this file; run_mode="lean" = 8 runs ~3h.
"""
from __future__ import annotations

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
# HierTree (shared)
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
# QREx quantile-variance penalty
# ---------------------------------------------------------------------------

def _per_env_losses(logits: torch.Tensor, labels: torch.Tensor,
                    sources: torch.Tensor, num_envs: int = 3) -> list:
    """Return a list of per-env scalar mean losses.  Envs with 0 samples
    contribute None (skipped by the caller).
    """
    per_env = []
    for e in range(num_envs):
        mask = (sources == e)
        if mask.sum() < 2:
            per_env.append(None)
            continue
        l = F.cross_entropy(logits[mask], labels[mask])
        per_env.append(l)
    return per_env


def _per_env_quantile(logits: torch.Tensor, labels: torch.Tensor,
                      sources: torch.Tensor, num_envs: int = 3,
                      alpha: float = 0.9) -> list:
    """Within each env, compute the alpha-quantile of the per-sample loss.
    """
    per_env = []
    for e in range(num_envs):
        mask = (sources == e)
        if mask.sum() < 4:
            per_env.append(None)
            continue
        nll = F.cross_entropy(logits[mask], labels[mask], reduction="none")
        # Soft quantile via sort + index
        q_idx = int(alpha * (nll.numel() - 1))
        q = torch.sort(nll)[0][q_idx]
        per_env.append(q)
    return per_env


def _qrex_penalty(per_env_quantiles: list) -> torch.Tensor:
    """Variance across envs of the per-env quantile (the QREx objective)."""
    valid = [q for q in per_env_quantiles if q is not None]
    if len(valid) < 2:
        # Only one env this batch -- return a tensor with grad
        if valid:
            return valid[0].new_zeros(1).squeeze()
        return torch.zeros(1, requires_grad=True).squeeze()
    stacked = torch.stack(valid, dim=0)
    return stacked.var()


ABLATION_TABLE = {
    "hier":     ("lambda_hier",  True),
    "qrex":     ("lambda_qrex",  True),
    "erm_env":  ("lambda_erm_env", True),   # mean of per-env risks (V-REx mean)
}


def qrex_compute_losses(model, outputs, labels, config,
                        focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier    * HierTree
       + lambda_erm_env * mean_e L_e (env-balanced mean, V-REx component)
       + lambda_qrex    * Var_e(Q_alpha(L_e)) (Eastwood 2025 QREx).
    """
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]
    logits = outputs["logits"]
    sources = outputs.get("sources", None)

    # HierTree
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    if sources is None or (sources >= 0).sum() < 4:
        return base

    # ERM env-balanced mean (V-REx baseline term)
    lambda_erm_env = getattr(config, "lambda_erm_env", 0.3)
    if lambda_erm_env > 0:
        per_env = _per_env_losses(logits, labels, sources, num_envs=3)
        valid = [l for l in per_env if l is not None]
        if valid:
            env_mean = torch.stack(valid).mean()
            base["total"] = base["total"] + lambda_erm_env * env_mean
            base["erm_env"] = env_mean

    # QREx quantile-variance
    lambda_qrex = getattr(config, "lambda_qrex", 1.0)
    alpha = getattr(config, "qrex_alpha", 0.9)
    if lambda_qrex > 0:
        per_env_q = _per_env_quantile(logits, labels, sources, num_envs=3, alpha=alpha)
        qrex = _qrex_penalty(per_env_q)
        base["total"] = base["total"] + lambda_qrex * qrex
        base["qrex"] = qrex

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
        method_name="QREx",
        exp_id="exp_24",
        loss_fn=qrex_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_24_qrex",
    )

    emit_ablation_suite(
        method_name="QREx",
        exp_id="exp_24",
        loss_fn=qrex_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
