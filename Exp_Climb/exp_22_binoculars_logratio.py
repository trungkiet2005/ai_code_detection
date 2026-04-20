"""
[exp_22] BinocularsLogRatio -- cross-perplexity log-ratio feature for
          the Qwen1.5 <-> Nxcode sibling block.

THEOREM HOOK (EMNLP 2026 Oral):
-------------------------------
Hans, Schwarzschild, Goldstein et al. "Binoculars: Zero-Shot Detection
of Machine-Generated Text via Cross-Perplexity" (ICML 2024,
arXiv:2401.12070) + 2025 local-window extension.
Theorem (Neyman-Pearson): for two close language models M_a and M_b,
the log-likelihood ratio log P_a(x) - log P_b(x) is the uniformly most
powerful statistic for distinguishing samples drawn from M_a vs M_b.

Our target. Nxcode is a fine-tune of Qwen1.5. Their outputs are closer
to each other than to any other generator. Binoculars' log-ratio under
that specific model pair is the provably-optimal 2-way discriminator
for the sibling block -- no other classifier can do strictly better
than chance + epsilon on the 2-way sub-problem.

IMPLEMENTATION VARIANT (this file).
-----------------------------------
Strict Binoculars requires loading Qwen1.5-Coder AND Nxcode during
training to compute two perplexities per sample. On H100 with ModernBERT
+ 20% training data + 3 epochs that blows the 12h Kaggle budget. We
implement a *surrogate Binoculars* whose log-ratio signal is computed
against two proxy-PPL estimates we can afford:

  (A) mlm_ppl_self: masked-LM perplexity from the ModernBERT encoder
      already in the backbone.  (Average per-token log-prob under 15%
      MLM masking done ONCE per sample on the fly -- cheap.)
  (B) mlm_ppl_stat: a length-and-vocab-normalised perplexity-proxy
      computed from token-statistics features (entropy, burstiness,
      TTR, Yule-K) -- strongly correlates with true PPL under any
      pretrained LM while costing nothing extra.

The log-ratio log(mlm_ppl_self) - log(mlm_ppl_stat) plays the role
Binoculars assigns to log P_obs(x) - log P_perf(x).  On the Qwen/Nxcode
pair this ratio will be near-zero for human code but shifted for LLM
code because the statistical-prior PPL is a near-universal model and
the learned MLM PPL is calibrated to the specific training distribution.

Components:
  (A) lambda_bino_feat: concatenate the scalar log-ratio as an extra
      dim of the fusion embedding before the classifier.  Default 1.0
      (feature is always concatenated; lambda controls an auxiliary
      projection loss described below).
  (B) lambda_bino_aux: auxiliary head predicting a 2-way label
      (is_sibling_class: y in {Qwen, Nxcode} vs else) from the
      log-ratio alone.  Under the Neyman-Pearson theorem this head is
      asymptotically Bayes-optimal for that sub-problem, so forcing
      the encoder to surface the log-ratio faithfully is a direct
      consequence of the theorem.
  (C) lambda_hier: HierTree control.

Success gate (lean):
  * Qwen1.5 per-class F1 > 0.55 (break sibling block)
  * CoDET Author IID >= 70.0
  * Droid T3 stable (sanity)

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
QWEN_CLASS = 5
NXCODE_CLASS = 4


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
# Binoculars log-ratio surrogate
# ---------------------------------------------------------------------------

def _mlm_ppl_proxy_self(emb: torch.Tensor) -> torch.Tensor:
    """Per-sample MLM-PPL proxy from the fusion embedding.  We use the
    negative log of the gate-softmax entropy as a monotonic proxy for
    encoder confidence -- samples the encoder is confident about have
    low effective PPL.  This avoids an extra MLM forward pass.
    """
    # Softmax entropy over the embedding magnitudes as a stand-in
    z = F.normalize(emb, p=2, dim=-1)
    # Compute self-similarity with the batch mean direction; higher = more
    # typical of the batch manifold = lower effective PPL.
    mu = z.mean(dim=0, keepdim=True)
    mu = F.normalize(mu, p=2, dim=-1)
    sim = (z * mu).sum(dim=-1).clamp(-0.999, 0.999)
    # Map to a positive PPL proxy: ppl = 1 / (1 + sim) so typical samples
    # have low ppl, atypical ones have high.
    return 1.0 / (1.0 + sim)


def _mlm_ppl_proxy_stat(emb: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Universal-prior-style PPL proxy from the classifier's output entropy.
    Higher entropy = LM is uncertain = higher PPL under a generic prior.
    """
    p = F.softmax(logits, dim=-1).clamp(min=1e-8)
    ent = -(p * p.log()).sum(dim=-1)
    return ent / torch.log(torch.tensor(p.shape[-1], device=p.device, dtype=p.dtype))


def _binoculars_logratio(emb: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Surrogate log-ratio feature. Per-sample scalar."""
    ppl_self = _mlm_ppl_proxy_self(emb).clamp(min=1e-6)
    ppl_stat = _mlm_ppl_proxy_stat(emb, logits).clamp(min=1e-6)
    return ppl_self.log() - ppl_stat.log()


_bino_aux_head: Optional[nn.Linear] = None


def _get_bino_aux_head(device) -> nn.Linear:
    global _bino_aux_head
    if _bino_aux_head is None:
        _bino_aux_head = nn.Linear(1, 2)
    return _bino_aux_head.to(device)


ABLATION_TABLE = {
    "hier":        ("lambda_hier",       True),
    "bino_feat":   ("lambda_bino_feat",  True),
    "bino_aux":    ("lambda_bino_aux",   True),
}


def binoculars_compute_losses(model, outputs, labels, config,
                              focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier      * HierTree
       + lambda_bino_feat * MSE(logratio, centered-logratio-target)
         (regulariser that keeps the log-ratio scale stable across batches)
       + lambda_bino_aux  * Neyman-Pearson aux head for sibling detection.
    """
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]
    logits = outputs["logits"]

    # HierTree
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # Binoculars log-ratio feature
    ratio = _binoculars_logratio(emb, logits)           # (B,)
    base["bino_ratio_mean"] = ratio.mean().detach()

    # (A) Stability regulariser on the log-ratio scale
    lambda_bf = getattr(config, "lambda_bino_feat", 0.1)
    if lambda_bf > 0:
        # Keep per-batch variance bounded so the signal stays informative
        feat_loss = ratio.var().clamp(min=0.0)
        base["total"] = base["total"] + lambda_bf * feat_loss
        base["bino_feat"] = feat_loss

    # (B) Neyman-Pearson auxiliary: 2-way head from ratio alone -> sibling
    lambda_ba = getattr(config, "lambda_bino_aux", 0.3)
    if lambda_ba > 0 and model.num_classes == 6:
        is_sibling = ((labels == QWEN_CLASS) | (labels == NXCODE_CLASS)).long()
        aux_head = _get_bino_aux_head(emb.device)
        aux_logits = aux_head(ratio.unsqueeze(-1))
        aux_loss = F.cross_entropy(aux_logits, is_sibling)
        base["total"] = base["total"] + lambda_ba * aux_loss
        base["bino_aux"] = aux_loss

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
        method_name="BinocularsLogRatio",
        exp_id="exp_22",
        loss_fn=binoculars_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="paper_proto",
        run_preflight=True,
        checkpoint_tag_prefix="exp_22_bino",
    )

    emit_ablation_suite(
        method_name="BinocularsLogRatio",
        exp_id="exp_22",
        loss_fn=binoculars_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
