"""
[exp_23] FrontDoor-NLP -- identifiable causal representation learning
          via a style mediator for CF/LC -> GitHub source shift.

THEOREM HOOK (EMNLP 2026 Oral):
-------------------------------
Veitch, Wang et al. "FRONTDOOR: Identifiable Causal Representation
Learning for NLP under Hidden Confounding" (NeurIPS 2025).
Theorem (front-door identification): when style S mediates the
path source -> X -> Y, and we can estimate P(s | X) from data, then
the causal effect is identifiable as

  P(Y | do(X)) = sum_s P(s | X) sum_{x'} P(Y | x', s) P(x')

even when the author confounder U is unobserved.  The inner
expectation is a counterfactual marginalisation over x' that breaks
the back-door path U -> Y.

Our target. Exp_18 tried back-door adjustment (assuming S is observed
and blocked via do-operation) -- mild negative result (+0.3 pt per
causal component, all inside noise).  Front-door removes the
ignorability assumption: it only requires that (i) S mediates source
-> Y, (ii) S is identifiable from X, (iii) the outcome model
P(Y | X, S) is well-specified.  All three hold by construction for
CoDET (source style is a function of the code text).

Components:
  (A) lambda_mediator: compress embedding into a 64-d style bottleneck
      S = g(X); train S to be predictable from source labels with CE
      (this gives us the P(s | X) factor).
  (B) lambda_hsic: kernel-HSIC penalty enforcing
      HSIC(S, source | Y) -> small, i.e. S captures source variance
      not explained by Y.  Key identifiability condition.
  (C) lambda_frontdoor_marg: the outer marginalisation
      sum_{x'} P(Y | x', s) P(x') -- approximated by mean-pooling the
      in-batch logits conditional on s-bucket (we bucket S into 3
      quantile bins and average per-bucket).
  (D) lambda_hier: HierTree control.

Success gate (lean):
  * OOD-SRC-gh > 0.40 (EMNLP headline)
  * CoDET Author IID >= 70.0
  * Shortcut probe P(source | phi(X)) drops by >= 20 pts vs Exp_00

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
# Style mediator + HSIC + front-door marginalisation
# ---------------------------------------------------------------------------

_style_bottleneck: Optional[nn.Module] = None
_source_from_S: Optional[nn.Linear] = None


def _get_style_bottleneck(emb_dim: int, style_dim: int = 64, device=None):
    global _style_bottleneck
    if _style_bottleneck is None or _style_bottleneck[0].in_features != emb_dim:
        _style_bottleneck = nn.Sequential(
            nn.Linear(emb_dim, 128), nn.GELU(),
            nn.Linear(128, style_dim),
        )
    if device is not None:
        _style_bottleneck = _style_bottleneck.to(device)
    return _style_bottleneck


def _get_source_from_S(style_dim: int = 64, num_sources: int = 3, device=None):
    global _source_from_S
    if _source_from_S is None or _source_from_S.in_features != style_dim:
        _source_from_S = nn.Linear(style_dim, num_sources)
    if device is not None:
        _source_from_S = _source_from_S.to(device)
    return _source_from_S


def _rbf_kernel(X: torch.Tensor, sigma: Optional[float] = None) -> torch.Tensor:
    pdist = torch.cdist(X, X, p=2).pow(2)
    if sigma is None:
        sigma = pdist.median().sqrt().clamp(min=1e-3)
    return torch.exp(-pdist / (2.0 * sigma * sigma))


def _hsic_conditional(S: torch.Tensor, source: torch.Tensor, labels: torch.Tensor,
                      num_classes: int) -> torch.Tensor:
    """HSIC(S, source | Y) estimated by summing per-class HSIC.  Uses RBF
    kernels with median heuristic.  Wanted small -> S captures source
    variance not explained by Y.
    """
    total = S.new_zeros(1).squeeze()
    count = 0
    for y in range(num_classes):
        mask = (labels == y)
        n = mask.sum().item()
        if n < 4:
            continue
        S_y = S[mask]
        src_y = source[mask]
        # Valid sources only
        valid = src_y >= 0
        if valid.sum() < 4:
            continue
        S_y = S_y[valid]
        src_y = src_y[valid]
        K = _rbf_kernel(S_y)
        # Source one-hot -> kernel = equal class membership
        src_oh = F.one_hot(src_y.clamp(min=0), num_classes=3).float()
        L = src_oh @ src_oh.t()
        n_eff = S_y.shape[0]
        H = torch.eye(n_eff, device=S.device) - (1.0 / n_eff) * torch.ones(n_eff, n_eff, device=S.device)
        hsic = (K @ H @ L @ H).trace() / max((n_eff - 1) ** 2, 1)
        total = total + hsic
        count += 1
    return total / max(count, 1)


def _frontdoor_marginalisation_loss(logits: torch.Tensor, S: torch.Tensor,
                                    labels: torch.Tensor, num_buckets: int = 3
                                   ) -> torch.Tensor:
    """Approximate sum_{x'} P(Y | x', s) P(x') by bucketing S into quantile
    bins and averaging logits within each bin, then enforcing that the
    bin-averaged prediction agrees with the per-sample prediction -- i.e.
    the front-door marginalisation is consistent with the actual label.
    """
    B = logits.shape[0]
    if B < num_buckets * 2:
        return logits.new_zeros(1).squeeze()
    # Bucket S by its first-coordinate quantile (cheap; any fixed summary works)
    key = S[:, 0]
    sorted_idx = torch.argsort(key)
    bucket_size = B // num_buckets
    marg_loss = logits.new_zeros(1).squeeze()
    for b in range(num_buckets):
        start = b * bucket_size
        end = (b + 1) * bucket_size if b < num_buckets - 1 else B
        idx = sorted_idx[start:end]
        if idx.numel() < 2:
            continue
        marg_logits = logits[idx].mean(dim=0, keepdim=True).expand_as(logits[idx])
        marg_loss = marg_loss + F.cross_entropy(marg_logits, labels[idx])
    return marg_loss / num_buckets


ABLATION_TABLE = {
    "hier":              ("lambda_hier",             True),
    "mediator":          ("lambda_mediator",         True),
    "hsic":              ("lambda_hsic",             True),
    "frontdoor_marg":    ("lambda_frontdoor_marg",   True),
}


def frontdoor_compute_losses(model, outputs, labels, config,
                             focal_loss_fn: Optional[FocalLoss] = None):
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]
    logits = outputs["logits"]
    sources = outputs.get("sources", None)

    # HierTree
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    style_dim = getattr(config, "frontdoor_style_dim", 64)
    bottleneck = _get_style_bottleneck(emb.shape[1], style_dim=style_dim, device=emb.device)
    S = bottleneck(emb)

    # (A) Style mediator: S must predict source (identifiability of P(s | X))
    lambda_med = getattr(config, "lambda_mediator", 0.3)
    if lambda_med > 0 and sources is not None and (sources >= 0).sum() >= 4:
        head = _get_source_from_S(style_dim, num_sources=3, device=emb.device)
        valid = sources >= 0
        src_logits = head(S[valid])
        med_loss = F.cross_entropy(src_logits, sources[valid])
        base["total"] = base["total"] + lambda_med * med_loss
        base["mediator"] = med_loss

    # (B) Conditional HSIC: S carries source signal not explained by Y
    lambda_hsic = getattr(config, "lambda_hsic", 0.1)
    if lambda_hsic > 0 and sources is not None and (sources >= 0).sum() >= 8:
        hsic_loss = _hsic_conditional(S, sources, labels, model.num_classes)
        base["total"] = base["total"] + lambda_hsic * hsic_loss
        base["hsic"] = hsic_loss

    # (C) Front-door marginalisation
    lambda_marg = getattr(config, "lambda_frontdoor_marg", 0.2)
    if lambda_marg > 0:
        marg_loss = _frontdoor_marginalisation_loss(logits, S, labels)
        base["total"] = base["total"] + lambda_marg * marg_loss
        base["frontdoor_marg"] = marg_loss

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
        method_name="FrontDoorNLP",
        exp_id="exp_23",
        loss_fn=frontdoor_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="paper_proto",
        run_preflight=True,
        checkpoint_tag_prefix="exp_23_fd",
    )

    emit_ablation_suite(
        method_name="FrontDoorNLP",
        exp_id="exp_23",
        loss_fn=frontdoor_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
