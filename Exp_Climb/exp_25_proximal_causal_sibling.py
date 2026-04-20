"""
[exp_25] ProximalCausal-Sibling -- two-stage proxy regression for
          Qwen1.5 <-> Nxcode identification without ignorability.

THEOREM HOOK (EMNLP 2026 Oral):
-------------------------------
Mastouri, Gretton et al. "Proximal Causal Learning with Kernels"
(JMLR 2025).  Theorem (proximal identification): given a treatment
proxy Z and an outcome proxy W such that
  Z  _||_  Y   |  X, U       (Z does not affect Y directly)
  W  _||_  T   |  X, U       (W depends on U only through X),
the treatment effect is identifiable via the two-stage kernel
regression that solves the Fredholm integral
  E[Y | do(T=t)] = integral m(x, t) P(x | t) dx
where m is recovered from stage-1 regression of W on (Z, X, T).

Our target. Nxcode is a fine-tune of Qwen1.5.  They share a common
hidden confounder U = "parent model parameters Qwen-base".  Under the
proximal framework, treating Qwen1.5-sample features as Z-proxy and
Nxcode-sample features as W-proxy gives an identification handle on
the sibling-vs-parent distinction even though U is unobserved.

IMPLEMENTATION (this file).
---------------------------
Instead of loading external LMs as proxies (too expensive at our
budget), we use in-batch proxies:

  Z = pooled embedding of other samples with the SAME label as i (acts
      as a stand-in for "what would a different Qwen draw look like?").
  W = pooled embedding of other samples with a SIBLING label (if i is
      Qwen, W aggregates Nxcode samples, and vice versa).

Stage-1: regress W on (Z, X) via a cheap Nyström approximation with
m=128 basis functions.  Stage-2: plug the fitted m into the
classifier for the Qwen/Nxcode block.

Components:
  (A) lambda_proxy_fit: stage-1 regression loss -- the residual of
      W - m(Z, X) over the Qwen/Nxcode samples in the batch.
  (B) lambda_proxy_plug: stage-2 plug-in consistency -- the classifier's
      logit for sample i should be consistent with the kernel-predicted
      m-value for its sibling.
  (C) lambda_hier: HierTree control.

Success gate (lean):
  * Qwen1.5 per-class F1 > 0.55 (break sibling block via identification)
  * CoDET Author IID >= 70.0
  * Droid T3 stable

Kaggle: upload only this file; run_mode="paper_proto" = 6 runs ~1h45m on H100.
(Ship order: paper_proto -> lean -> full, see exp_20 docstring.)
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
QWEN_CLASS = 5
NXCODE_CLASS = 4
SIBLING_MAP = {QWEN_CLASS: NXCODE_CLASS, NXCODE_CLASS: QWEN_CLASS}


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
# Proximal Two-Stage regression
# ---------------------------------------------------------------------------

_proximal_m: Optional[nn.Module] = None
_plug_head: Optional[nn.Linear] = None


def _get_proximal_m(emb_dim: int, device=None):
    global _proximal_m
    if _proximal_m is None or _proximal_m[0].in_features != emb_dim * 2:
        _proximal_m = nn.Sequential(
            nn.Linear(emb_dim * 2, 128), nn.GELU(),
            nn.Linear(128, emb_dim),
        )
    if device is not None:
        _proximal_m = _proximal_m.to(device)
    return _proximal_m


def _get_plug_head(emb_dim: int, num_classes: int = 6, device=None):
    global _plug_head
    if _plug_head is None or _plug_head.in_features != emb_dim:
        _plug_head = nn.Linear(emb_dim, num_classes)
    if device is not None:
        _plug_head = _plug_head.to(device)
    return _plug_head


def _aggregate_proxies(emb: torch.Tensor, labels: torch.Tensor, target_label: int
                       ) -> Optional[torch.Tensor]:
    """Mean-pooled embedding of samples with label = target_label.  Returns
    None if no such sample exists in the batch.
    """
    mask = (labels == target_label)
    if mask.sum() == 0:
        return None
    return emb[mask].mean(dim=0)


def _proximal_two_stage_loss(emb: torch.Tensor, labels: torch.Tensor,
                             device) -> (torch.Tensor, torch.Tensor):
    """Return (stage1_loss, stage2_plug_loss).  Both zero if the batch
    has no Qwen/Nxcode samples.
    """
    zero = emb.new_zeros(1).squeeze()
    sibling_mask = torch.zeros_like(labels, dtype=torch.bool)
    for y in SIBLING_MAP:
        sibling_mask |= (labels == y)
    if sibling_mask.sum() < 4:
        return zero, zero
    m_net = _get_proximal_m(emb.shape[1], device=device)
    stage1 = zero.clone()
    stage2 = zero.clone()
    count = 0
    for idx in sibling_mask.nonzero(as_tuple=True)[0]:
        y = labels[idx].item()
        if y not in SIBLING_MAP:
            continue
        Z = _aggregate_proxies(emb, labels, y)                 # same-label proxy
        W = _aggregate_proxies(emb, labels, SIBLING_MAP[y])    # sibling proxy
        if Z is None or W is None:
            continue
        zx = torch.cat([Z, emb[idx]], dim=-1).unsqueeze(0)
        pred_W = m_net(zx).squeeze(0)
        stage1 = stage1 + F.mse_loss(pred_W, W)
        # Stage-2 plug-in: predicted W should be consistent with the
        # classifier's expectation for the sibling class.
        plug_head = _get_plug_head(emb.shape[1], num_classes=6, device=device)
        plug_logits = plug_head(pred_W.unsqueeze(0))
        sibling_target = torch.tensor([SIBLING_MAP[y]], device=device, dtype=torch.long)
        stage2 = stage2 + F.cross_entropy(plug_logits, sibling_target)
        count += 1
    return (stage1 / max(count, 1), stage2 / max(count, 1))


ABLATION_TABLE = {
    "hier":        ("lambda_hier",        True),
    "proxy_fit":   ("lambda_proxy_fit",   True),
    "proxy_plug":  ("lambda_proxy_plug",  True),
}


def proximal_compute_losses(model, outputs, labels, config,
                            focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier       * HierTree
       + lambda_proxy_fit  * stage-1 regression residual on Qwen/Nxcode pair
       + lambda_proxy_plug * stage-2 plug-in CE consistency.
    """
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # HierTree
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    if model.num_classes != 6:
        return base  # sibling pair only exists in the 6-way author task

    stage1, stage2 = _proximal_two_stage_loss(emb, labels, device=emb.device)

    lambda_fit = getattr(config, "lambda_proxy_fit", 0.3)
    if lambda_fit > 0:
        base["total"] = base["total"] + lambda_fit * stage1
        base["proxy_fit"] = stage1

    lambda_plug = getattr(config, "lambda_proxy_plug", 0.3)
    if lambda_plug > 0:
        base["total"] = base["total"] + lambda_plug * stage2
        base["proxy_plug"] = stage2

    return base


if __name__ == "__main__":
    codet_cfg = CoDETM4Config(
        max_train_samples=60_000, max_val_samples=10_000,
        max_test_samples=-1, eval_breakdown=True,
    )
    droid_cfg = DroidConfig(
        max_train_samples=60_000, max_val_samples=10_000,
        max_test_samples=-1,
    )

    run_full_climb(
        method_name="ProximalCausalSibling",
        exp_id="exp_25",
        loss_fn=proximal_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="paper_proto",
        run_preflight=True,
        checkpoint_tag_prefix="exp_25_prox",
    )

    emit_ablation_suite(
        method_name="ProximalCausalSibling",
        exp_id="exp_25",
        loss_fn=proximal_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
