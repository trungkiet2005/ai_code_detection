"""
[Exp_TK exp09] CausalInterventionCode-W -- do-operations with warmup-scheduled lambdas

Challenger for OOD-SRC-gh breakthrough (NeurIPS 2026 headline target).
----------------------------------------------------------------------
Builds on **Exp_18 CausalInterventionCode** (Climb pending, the first
do-operations method in the repo) which adapts three causal-inference
mechanisms from vision/NLP to AI-code-detection:

  * (A) FEATURE-LEVEL COUNTERFACTUAL SWAP -- same-author-diff-source
    batch-mate; swap spectral half, enforce embedding invariance.
  * (B) BACKDOOR ADJUSTMENT over source confounder -- penalize variance
    of softmax predictions across source subgroups within same author
    (approximates P(author | do(code)) = sum_s P(author | code, s) P(s)).
  * (C) IV-STYLE ORTHOGONALITY -- penalize correlation between logit
    norm and spectral-feature norm (spectral head is the instrument).

Why fundamentally different from adversarial invariance (DANN, IRM,
VILW, POEM) all of which FAILED on this board (tracker section 4):
  * Causal do-operations BREAK the spurious correlation while
    provably preserving the causal signal (backdoor criterion).
  * Adversarial invariance ERASES the nuisance -- and in Exp19
    EAGLECode (-7.66% author F1) we confirmed this ALSO erases the
    causal signal when the target class is correlated with the nuisance.

What exp09 adds over Exp_18 (the loss-fn-level extension):
----------------------------------------------------------
  * **Epoch-tracked warmup for the three causal lambdas**. The Exp_TK
    tracker explicitly flags two causal/invariance cousins that
    destabilized training by ramping too fast:
      - Exp06 AST-IRM: penalty exploded to 1e4+ by epoch 3, NaN grads.
      - Exp01 CausAST: strict ortho penalty without warmup, crushed CE.
    Causal interventions have the same risk: at epoch 0 the embeddings
    are essentially random, so the counterfactual-swap + backdoor-var
    penalties fight the CE signal instead of refining it. We ramp
    lambda_cf / lambda_backdoor / lambda_iv linearly from 0 to target
    over the first `warmup_frac * total_steps` (default 0.5). HierTree
    and focal stay at full weight throughout.
  * Module-state step counter -- no trainer-side changes required.

Loss (post-warmup):
  focal + 0.3*neural + 0.3*spectral
   + lambda_hier         * HierTree
   + w(t) * lambda_cf    * counterfactual-swap consistency
   + w(t) * lambda_bd    * backdoor-adjustment variance penalty
   + w(t) * lambda_iv    * IV orthogonality penalty
  w(t) = min(1, step / warmup_steps)

Ablation toggles: hier / cf_swap / backdoor / iv_proj / warmup.

Targets (tracker section 7):
  * CoDET OOD-SRC-gh > 0.40  (current best Exp_06 / Exp_08: 0.33 --
    breaking 0.40 makes this the first method ever to clear the
    GitHub-source OOD threshold; NeurIPS 2026 headline)
  * CoDET Author IID >= 70.0  (causal do-ops are an OOD fix; IID is
    not expected to regress but should hold)
  * Droid T3 W-F1 ~ 0.88  (stable)

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. `run_mode="lean"` -> 8 runs + 4 ablation runs.
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
# HierTree (preserved)
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
# Causal interventions (preserved from Exp_18)
# ===========================================================================

def _counterfactual_swap_loss(emb: torch.Tensor, labels: torch.Tensor,
                              sources: torch.Tensor,
                              spectral_feat: Optional[torch.Tensor]) -> torch.Tensor:
    """(A) Same-author-diff-source batch-mate; swap spectral half; enforce
    L2 embedding consistency. Silently 0 if no qualifying batch-mate exists.
    """
    if spectral_feat is None or sources is None or spectral_feat.shape != emb.shape:
        return emb.new_zeros(1).squeeze()
    if sources.dim() != 1 or sources.numel() != emb.shape[0]:
        return emb.new_zeros(1).squeeze()

    B, D = emb.shape
    loss = emb.new_zeros(1).squeeze()
    count = 0
    half = D // 2
    for i in range(B):
        same_a = (labels == labels[i])
        diff_s = (sources != sources[i])
        cand = same_a & diff_s
        cand[i] = False
        if cand.sum() == 0:
            continue
        j = cand.nonzero(as_tuple=True)[0][0].item()
        cf = torch.cat([emb[i, :half], spectral_feat[j, :D - half]], dim=-1)
        a_n = F.normalize(emb[i], p=2, dim=-1)
        c_n = F.normalize(cf, p=2, dim=-1)
        loss = loss + (1.0 - (a_n * c_n).sum())
        count += 1
    return loss / max(count, 1)


def _backdoor_adjust_loss(logits: Optional[torch.Tensor], labels: torch.Tensor,
                          sources: torch.Tensor) -> torch.Tensor:
    """(B) Penalize variance of per-source softmax means: if the
    classifier is source-invariant for the marginal label distribution,
    this is 0 -- the backdoor path is closed.
    """
    if logits is None or logits.dim() != 2:
        return labels.new_zeros(1).float()
    if sources is None or sources.dim() != 1 or sources.numel() != labels.numel():
        return logits.new_zeros(1).squeeze()
    p = F.softmax(logits, dim=-1)
    src_set = sources.unique()
    if src_set.numel() < 2:
        return logits.new_zeros(1).squeeze()
    per_src_mean = []
    for s in src_set:
        mask = (sources == s)
        if mask.sum() == 0:
            continue
        per_src_mean.append(p[mask].mean(dim=0))
    if len(per_src_mean) < 2:
        return logits.new_zeros(1).squeeze()
    stacked = torch.stack(per_src_mean, dim=0)
    return stacked.var(dim=0).mean()


def _iv_projection_penalty(logits: Optional[torch.Tensor],
                           spectral_feat: Optional[torch.Tensor],
                           sources: torch.Tensor) -> torch.Tensor:
    """(C) Penalize |corr(||logits||, ||spectral_feat||)| -- IV
    orthogonality (Kim et al. CVPR 2023). 0 iff instrument (spectral
    head) is uncorrelated with outcome norm; the structural signal is
    then free of source confounding.
    """
    if logits is None or spectral_feat is None:
        return logits.new_zeros(1).squeeze() if logits is not None else sources.new_zeros(1).float()
    log_norm = logits.norm(p=2, dim=-1)
    spec_norm = spectral_feat.norm(p=2, dim=-1)
    log_c = log_norm - log_norm.mean()
    spec_c = spec_norm - spec_norm.mean()
    denom = (log_c.norm() * spec_c.norm()).clamp_min(1e-8)
    corr = (log_c * spec_c).sum() / denom
    return corr.abs()


# ===========================================================================
# NEW: warmup scheduler for causal lambdas (exp09 addition)
# ===========================================================================

_warmup_state = {"step": 0, "warmup_steps": None}


def _warmup_weight(config) -> float:
    """Linear ramp from 0 to 1 over `warmup_frac * total_steps`.

    total_steps is lazily read from config (set by _climb_runner via
    `config.num_training_steps`). Falls back to a sensible default
    (2000 steps ~= half of an 8-epoch run at batch 128 on 100k data)
    if unavailable.
    """
    if not getattr(config, "use_warmup", True):
        return 1.0
    if _warmup_state["warmup_steps"] is None:
        total = int(getattr(config, "num_training_steps", 0) or 0)
        if total <= 0:
            total = int(getattr(config, "max_steps", 0) or 4000)
        frac = float(getattr(config, "warmup_frac", 0.5))
        _warmup_state["warmup_steps"] = max(1, int(total * frac))
    _warmup_state["step"] += 1
    step = _warmup_state["step"]
    ws = _warmup_state["warmup_steps"]
    return min(1.0, step / ws)


_hier_fn: Optional[HierarchicalAffinityLoss] = None


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


# ===========================================================================
# Ablation toggles
# ===========================================================================

ABLATION_TABLE = {
    "hier":         ("lambda_hier",     True),
    "cf_swap":      ("lambda_cf",       True),
    "backdoor":     ("lambda_backdoor", True),
    "iv_proj":      ("lambda_iv",       True),
    "warmup":       ("use_warmup",      True),
}


# ===========================================================================
# Loss
# ===========================================================================

def causal_warmup_compute_losses(model, outputs, labels, config,
                                 focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier            * HierTree       (full weight)
       + w(t) * lambda_cf       * counterfactual-swap consistency
       + w(t) * lambda_backdoor * backdoor-adj variance penalty
       + w(t) * lambda_iv       * IV orthogonality penalty
       w(t) ramps linearly 0 -> 1 over warmup_frac * total steps.
    """
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # HierTree (full weight -- no warmup; it is well-behaved and helps IID)
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    sources = outputs.get("sources", None)
    spectral_feat = outputs.get("spectral_features", None)
    logits = outputs.get("logits", None)

    if sources is not None and spectral_feat is not None:
        w = _warmup_weight(config)

        # (A) Counterfactual feature swap
        cf_loss = _counterfactual_swap_loss(emb, labels, sources, spectral_feat)
        base["total"] = base["total"] + w * getattr(config, "lambda_cf", 0.25) * cf_loss
        base["cf_swap"] = cf_loss

        # (B) Backdoor adjustment
        bd_loss = _backdoor_adjust_loss(logits, labels, sources)
        base["total"] = base["total"] + w * getattr(config, "lambda_backdoor", 0.15) * bd_loss
        base["backdoor"] = bd_loss

        # (C) IV orthogonality
        iv_loss = _iv_projection_penalty(logits, spectral_feat, sources)
        base["total"] = base["total"] + w * getattr(config, "lambda_iv", 0.1) * iv_loss
        base["iv_proj"] = iv_loss
        base["warmup_w"] = torch.tensor(w, device=emb.device)

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
        method_name="CausalIntervention-W",
        exp_id="exp09",
        loss_fn=causal_warmup_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp09_causal_w",
    )

    emit_ablation_suite(
        method_name="CausalIntervention-W",
        exp_id="exp09",
        loss_fn=causal_warmup_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
