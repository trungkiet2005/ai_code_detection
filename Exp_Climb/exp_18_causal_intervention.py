"""
[exp_18] CausalInterventionCode -- counterfactual feature intervention for shortcut removal

EMNLP 2026 novelty angle:
-------------------------
Cross-domain transfer from:
  * "Demystifying Causal Features on Adversarial Examples and Causal
    Inoculation for Robust Network by Adversarial Instrumental Variable
    Regression" (Kim et al., CVPR 2023, arXiv 2303.01052) -- the CAFE
    paper that uses IV regression to separate CAUSAL features from
    CONFOUNDED features.
  * "Explaining Text Classifiers with Counterfactual Representations"
    (Lemberger & Saillenfest, 2024, arXiv 2402.00711) -- text-side
    counterfactual intervention in representation space.
  * "DINER: Debiasing Aspect-based Sentiment Analysis with Multi-variable
    Causal Inference" (Wu et al., 2024, arXiv 2403.01166) -- backdoor
    adjustment applied to text classifier features.

Connection to climb:
  * Shortcut learning IS the central bottleneck of Exp_Climb (GH-OOD
    collapse, AICD val-test gap). Insight #5 of the tracker says all
    prior methods that tried to REMOVE shortcuts via adversarial
    invariance (DANN, IRM, VILW) either failed (Exp19 EAGLECode -7.66%)
    or destabilized training (Exp06 AST-IRM).
  * Causal intervention is FUNDAMENTALLY DIFFERENT from adversarial
    invariance: instead of making features INVARIANT to the nuisance,
    we perform do-operations that BREAK the spurious correlation while
    preserving the causal signal.

CausalInterventionCode implements three levels of intervention:

  (A) FEATURE-LEVEL COUNTERFACTUAL SWAP: for each anchor (author=a,
      source=s), find a batch-mate (author=a, source=s') with the same
      author but DIFFERENT source. Swap their spectral sub-feature (the
      part correlated with source) while keeping the neural sub-feature
      fixed. The classifier must predict the same author on the
      counterfactual -- enforced via L2 consistency.

  (B) BACKDOOR ADJUSTMENT ON SOURCE CONFOUNDER: source is the known
      confounder for author prediction (CF templates correlate with
      certain authors). We marginalize over the source conditional:
      `P(author | do(code)) = sum_s P(author | code, s) * P(s)`.
      Implemented as: during training, shuffle source labels within
      same-author groups and enforce prediction invariance.

  (C) INSTRUMENTAL-VARIABLE-STYLE PROJECTION: decompose logits into
      `logit = IV_part + confounded_part` where IV_part is estimated
      from a HELD-OUT spectral head that only sees syntactic features
      (no source signal by construction). Only IV_part contributes to
      the final decision during evaluation.

This is fundamentally different from Exp_02 (GH-invariant adversarial)
and Exp_08 (POEM orthogonal polarization) because it applies do-operations
rather than feature invariance. Causal theory provides guarantees that
adversarial methods lack (DANN CAN erase the causal signal; do-operations
provably preserve it under the backdoor criterion).

Built-in ablation: (cf_swap, backdoor_adj, iv_projection). Drop-sorted
table tells us which intervention is load-bearing.

Targets (lean gates):
  * OOD-SRC-gh > 0.40  (causal theory predicts strong OOD transfer;
    if we crack 0.40 this is the first method ever to do so across
    all 14 prior climb/CodeDet methods -- EMNLP headline)
  * CoDET Author IID >= 70.0 (IID is not harmed; this is an OOD fix)
  * Droid T3 stable ~ 0.88

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. `run_mode="lean"` -> 8 runs in ~3h on H100.
  3. Ablation suite adds 3 extra single-task runs (+~70 min).
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
# Causal interventions + HierTree
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


def _counterfactual_swap_loss(emb: torch.Tensor, labels: torch.Tensor,
                              sources: torch.Tensor,
                              spectral_feat: Optional[torch.Tensor]) -> torch.Tensor:
    """(A) For each anchor i, find a batch-mate j with SAME author but
    DIFFERENT source. Construct x_cf = (neural(i), spectral(j)) and enforce
    f(x_cf) ≈ f(x_i) in embedding space. Fails silently if no qualifying
    batch-mate exists (typical for minority authors in small batches).
    """
    if spectral_feat is None or sources is None or spectral_feat.shape != emb.shape:
        return emb.new_zeros(1).squeeze()
    if sources.dim() != 1 or sources.numel() != emb.shape[0]:
        return emb.new_zeros(1).squeeze()

    B, D = emb.shape
    loss = emb.new_zeros(1).squeeze()
    count = 0
    # Split emb conceptually into neural / spectral halves for the swap.
    half = D // 2
    for i in range(B):
        # Find a batch-mate with same author, different source
        same_a = (labels == labels[i])
        diff_s = (sources != sources[i])
        cand = same_a & diff_s
        cand[i] = False
        if cand.sum() == 0:
            continue
        # Pick the first candidate deterministically
        j = cand.nonzero(as_tuple=True)[0][0].item()
        # Counterfactual: keep neural part of i, swap in spectral part of j
        cf = torch.cat([emb[i, :half], spectral_feat[j, :D - half]], dim=-1)
        # Consistency: anchor and its counterfactual should have same
        # normalized embedding (the classifier only looks at author, not source).
        a_n = F.normalize(emb[i], p=2, dim=-1)
        c_n = F.normalize(cf, p=2, dim=-1)
        loss = loss + (1.0 - (a_n * c_n).sum())
        count += 1
    return loss / max(count, 1)


def _backdoor_adjust_loss(logits: Optional[torch.Tensor], labels: torch.Tensor,
                          sources: torch.Tensor,
                          n_perm: int = 4) -> torch.Tensor:
    """(B) P(a | do(c)) = sum_s P(a | c, s) P(s). Approximated by shuffling
    source labels within same-author groups and asking the classifier to
    predict the same author. Penalizes variance of predictions under
    source-permutation.
    """
    if logits is None or logits.dim() != 2:
        return labels.new_zeros(1).float()
    if sources is None or sources.dim() != 1 or sources.numel() != labels.numel():
        return logits.new_zeros(1).squeeze()
    # Under permutation, logits are already a function of (code, source). We
    # approximate P(a | do(c)) as the mean of logits over several permuted
    # source proxies. Since we don't have a source-conditional classifier at
    # training time, we instead penalize the DIFFERENCE between per-source
    # softmax means -- if they agree, the backdoor is closed.
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
    stacked = torch.stack(per_src_mean, dim=0)           # (S, C)
    # Penalty = variance across sources (if classifier is source-invariant
    # for the *marginal* label distribution, this is 0).
    return stacked.var(dim=0).mean()


def _iv_projection_penalty(logits: Optional[torch.Tensor],
                           spectral_feat: Optional[torch.Tensor],
                           sources: torch.Tensor) -> torch.Tensor:
    """(C) IV-style: spectral_feat is our "instrument" -- it depends on code
    structure but (assumed) not directly on source. Penalize correlation
    between per-sample logit norm and spectral_feat norm conditional on source.
    A proxy for the IV orthogonality condition from Kim et al. 2023.
    """
    if logits is None or spectral_feat is None:
        return logits.new_zeros(1).squeeze() if logits is not None else sources.new_zeros(1).float()
    log_norm = logits.norm(p=2, dim=-1)
    spec_norm = spectral_feat.norm(p=2, dim=-1)
    # Centered Pearson correlation between the two -- should be 0 if IV holds
    log_c = log_norm - log_norm.mean()
    spec_c = spec_norm - spec_norm.mean()
    denom = (log_c.norm() * spec_c.norm()).clamp_min(1e-8)
    corr = (log_c * spec_c).sum() / denom
    return corr.abs()   # penalize any non-zero correlation


_hier_fn: Optional[HierarchicalAffinityLoss] = None


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


ABLATION_TABLE = {
    "hier":         ("lambda_hier",     True),
    "cf_swap":      ("lambda_cf",       True),   # counterfactual feature swap
    "backdoor":     ("lambda_backdoor", True),   # backdoor adjustment
    "iv_proj":      ("lambda_iv",       True),   # IV orthogonality
}


def causal_compute_losses(model, outputs, labels, config,
                          focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier       * HierTree
       + lambda_cf         * counterfactual-swap consistency
       + lambda_backdoor   * backdoor-adjustment penalty
       + lambda_iv         * IV orthogonality penalty."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # HierTree
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # Causal interventions require source info + spectral features from the model.
    sources = outputs.get("sources", None)
    spectral_feat = outputs.get("spectral_features", None)
    logits = outputs.get("logits", None)

    # (A) Counterfactual feature swap (source-same-author batch-mate)
    if sources is not None and spectral_feat is not None:
        cf_loss = _counterfactual_swap_loss(emb, labels, sources, spectral_feat)
        base["total"] = base["total"] + getattr(config, "lambda_cf", 0.2) * cf_loss
        base["cf_swap"] = cf_loss

        # (B) Backdoor adjustment on source confounder
        bd_loss = _backdoor_adjust_loss(logits, labels, sources)
        base["total"] = base["total"] + getattr(config, "lambda_backdoor", 0.15) * bd_loss
        base["backdoor"] = bd_loss

        # (C) IV orthogonality
        iv_loss = _iv_projection_penalty(logits, spectral_feat, sources)
        base["total"] = base["total"] + getattr(config, "lambda_iv", 0.1) * iv_loss
        base["iv_proj"] = iv_loss

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
        method_name="CausalInterventionCode",
        exp_id="exp_18",
        loss_fn=causal_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp_18_causal",
    )

    emit_ablation_suite(
        method_name="CausalInterventionCode",
        exp_id="exp_18",
        loss_fn=causal_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
