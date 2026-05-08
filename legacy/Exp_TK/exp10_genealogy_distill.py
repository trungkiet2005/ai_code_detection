"""
[Exp_TK exp10] GenealogyDistill-MP -- multi-pair sibling isolation + anti-distill

Challenger for CoDET Author IID peak (beat Exp27 DeTeCtiveCode 71.53).
---------------------------------------------------------------------
Builds on **Exp_15 GenealogyDistill** (Climb pending), the first climb
method designed explicitly for the Nxcode<->Qwen1.5 sibling confusion
(the tracker's cross-cutting insight #2: "Nxcode <-> Qwen is the single
biggest CoDET lever"). Exp_15 stacks:

  (A) Pair-margin triplet on (Nxcode=4, Qwen1.5=5) with margin > HierTree
      margin, to open the sibling gap WIDER than the family gap.
  (B) Selective anti-distill: when softmax on the pair is ambiguous
      (top-2 probs both > tau), penalize closeness -- an anti-smoothing
      regularizer that rewards decisiveness only on the confused pair.

What exp10 adds over Exp_15 (the loss-fn-level extension):
----------------------------------------------------------
  * **Multi-pair** sibling isolation. The CoDET author family table is
    `[0, 1, 2, 1, 3, 3]` -- family 1 has {codellama=1, llama3.1=3} and
    family 3 has {nxcode=4, qwen1.5=5}. Both families bundle two
    generators that share lineage. Exp_15 fixes only the family-3 pair.
    Exp27 DeTeCtiveCode confusion matrix (tracker footnote) shows
    family-1 has the SAME structural confusion: HierTree's hinge on
    same-family-max-distance makes them interchangeable by construction.
    Multi-pair margin enumerates EVERY sibling pair within each family
    and enforces per-pair isolation -- no mechanism change, just
    broader coverage. The pair list is built automatically from the
    family table (no hard-coded class IDs beyond the table).
  * **Family-aware anti-distill**: applied to every within-family
    pair, not just Nxcode<->Qwen. Same ambiguity threshold, same
    decisiveness reward. This matches the multi-pair margin so both
    mechanisms attack the full set of confusable pairs.

Loss:
  focal + 0.3*neural + 0.3*spectral
   + lambda_hier          * HierTree        (inter-family)
   + lambda_pair_margin   * sum over all within-family sibling pairs
                            of max-margin triplets (intra-family)
   + lambda_anti_distill  * sum over all within-family sibling pairs
                            of ambiguity-penalty (intra-family)

Ablation toggles: hier / pair_margin / anti_distill / multi_pair.
Drop multi_pair -> falls back to the Exp_15 single-pair formulation
(Nxcode<->Qwen only), which measures the multi-pair lift in isolation.

Targets (tracker section 7):
  * CoDET Author IID > 72.0  (beat Exp27 71.53 by >0.5 -- the climb
    plateau at 70.0-70.2 has been stuck; sibling isolation is the one
    lever the plateau hasn't touched)
  * Per-class F1 on {nxcode, qwen1.5, codellama, llama3.1} > 0.55
  * CoDET Binary stable ~99.1 (unaffected)

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

from typing import List, Optional, Tuple

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
# Family table + sibling-pair enumeration
# ===========================================================================

# CoDET-M4: 0=human, 1=codellama, 2=gpt, 3=llama3.1, 4=nxcode, 5=qwen1.5
# Family 0: {human}      -- singleton
# Family 1: {codellama, llama3.1}  -- Meta lineage
# Family 2: {gpt}        -- singleton
# Family 3: {nxcode, qwen1.5}      -- Alibaba/Qwen lineage (nxcode is a Qwen FT)
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


def _enumerate_sibling_pairs(family_table: Optional[List[int]]) -> List[Tuple[int, int]]:
    """Return [(a, b), ...] -- every unordered within-family pair (a<b).
    For the 6-class CoDET table this yields [(1,3), (4,5)].
    Singleton families contribute nothing.
    """
    if not family_table:
        return []
    by_fam: dict = {}
    for cls, fam in enumerate(family_table):
        by_fam.setdefault(fam, []).append(cls)
    pairs: List[Tuple[int, int]] = []
    for fam, members in by_fam.items():
        if len(members) < 2:
            continue
        members = sorted(members)
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                pairs.append((members[i], members[j]))
    return pairs


class HierarchicalAffinityLoss(nn.Module):
    """HierTree -- inter-family. Kept identical to Exp_15 / Exp18."""

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
# Multi-pair sibling isolation (exp10 addition)
# ===========================================================================

def _pair_margin_loss(emb: torch.Tensor, labels: torch.Tensor,
                      pair_a: int, pair_b: int,
                      margin: float = 0.55) -> torch.Tensor:
    """Directed max-margin triplet isolating one sibling pair.

    For each pair_a anchor: distance to nearest pair_a positive + margin
    must be smaller than distance to the hardest pair_b negative. Mirror
    for pair_b anchors. Default margin 0.55 > HierTree family margin
    (0.3) -- we want the sibling gap wider than the family gap.
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
            same = a_mask.clone(); same[i] = False
            if same.sum() == 0:
                continue
            d_pos = dist[i][same].max()
            d_neg = dist[i][o_mask].min()
            loss = loss + F.relu(d_pos - d_neg + margin)
            count += 1
    return loss / max(count, 1)


def _multi_pair_margin_loss(emb: torch.Tensor, labels: torch.Tensor,
                            pairs: List[Tuple[int, int]],
                            margin: float = 0.55) -> torch.Tensor:
    """Sum per-pair margin losses, normalized by number of active pairs."""
    if not pairs:
        return emb.new_zeros(1).squeeze()
    total = emb.new_zeros(1).squeeze()
    n_active = 0
    for a, b in pairs:
        if (labels == a).sum() < 2 or (labels == b).sum() < 1:
            continue
        total = total + _pair_margin_loss(emb, labels, a, b, margin=margin)
        n_active += 1
    return total / max(n_active, 1)


# ===========================================================================
# Family-aware selective anti-distill (exp10 extension of Exp_15)
# ===========================================================================

def _anti_distill_pair(logits: torch.Tensor, pair_a: int, pair_b: int,
                       ambiguity_threshold: float = 0.25) -> torch.Tensor:
    """For rows where softmax on (pair_a, pair_b) are BOTH above tau,
    penalize closeness |p_a - p_b|. Returns the mean over ambiguous rows,
    or 0 if there are none."""
    p = F.softmax(logits, dim=-1)
    p_a = p[:, pair_a]
    p_b = p[:, pair_b]
    ambig_mask = (p_a > ambiguity_threshold) & (p_b > ambiguity_threshold)
    if ambig_mask.sum() == 0:
        return logits.new_zeros(1).squeeze()
    return -(p_a[ambig_mask] - p_b[ambig_mask]).abs().mean()


def _multi_pair_anti_distill_loss(logits: Optional[torch.Tensor],
                                  pairs: List[Tuple[int, int]],
                                  ambiguity_threshold: float = 0.25) -> torch.Tensor:
    if logits is None or logits.dim() != 2 or not pairs:
        return (logits.new_zeros(1).squeeze()
                if logits is not None else torch.zeros(1))
    num_classes = logits.shape[1]
    total = logits.new_zeros(1).squeeze()
    n_active = 0
    for a, b in pairs:
        if a >= num_classes or b >= num_classes:
            continue
        total = total + _anti_distill_pair(logits, a, b, ambiguity_threshold)
        n_active += 1
    return total / max(n_active, 1)


# ===========================================================================
# Singletons + ablation toggles
# ===========================================================================

_hier_fn: Optional[HierarchicalAffinityLoss] = None
_pairs_cache: Optional[Tuple[int, List[Tuple[int, int]]]] = None


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


def _get_pairs(num_classes: int, multi_pair: bool) -> List[Tuple[int, int]]:
    """Return sibling-pair list.

    multi_pair=True  -> every within-family pair (Exp_TK exp10 default).
    multi_pair=False -> Exp_15 behavior: just (Nxcode=4, Qwen1.5=5), so
                        we can measure the multi-pair lift via ablation.
    """
    global _pairs_cache
    if _pairs_cache is not None and _pairs_cache[0] == (num_classes, multi_pair):
        return _pairs_cache[1]
    fam = _build_family_table(num_classes)
    if multi_pair:
        pairs = _enumerate_sibling_pairs(fam)
    else:
        pairs = ([(NXCODE_LABEL, QWEN15_LABEL)]
                 if num_classes == 6 else [])
    _pairs_cache = ((num_classes, multi_pair), pairs)
    return pairs


ABLATION_TABLE = {
    "hier":         ("lambda_hier",         True),
    "pair_margin":  ("lambda_pair_margin",  True),
    "anti_distill": ("lambda_anti_distill", True),
    # Drop this -> single (nxcode, qwen1.5) pair, i.e. Exp_15 formulation:
    "multi_pair":   ("use_multi_pair",      True),
}


# ===========================================================================
# Loss
# ===========================================================================

def genealogy_distill_mp_compute_losses(model, outputs, labels, config,
                                        focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier          * HierTree
       + lambda_pair_margin   * multi-pair sibling isolation (within-family)
       + lambda_anti_distill  * multi-pair ambiguity penalty (within-family)."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]

    # HierTree (inter-family)
    hier_fn = _get_hier(model.num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # Intra-family sibling isolation -- only active if family table yields pairs
    use_multi_pair = bool(getattr(config, "use_multi_pair", True))
    pairs = _get_pairs(model.num_classes, use_multi_pair)

    if pairs:
        pair_loss = _multi_pair_margin_loss(
            emb, labels, pairs,
            margin=getattr(config, "pair_margin_alpha", 0.55),
        )
        base["total"] = base["total"] + getattr(config, "lambda_pair_margin", 0.3) * pair_loss
        base["pair_margin"] = pair_loss

        logits = outputs.get("logits", None)
        anti_loss = _multi_pair_anti_distill_loss(
            logits, pairs,
            ambiguity_threshold=getattr(config, "anti_distill_tau", 0.25),
        )
        base["total"] = base["total"] + getattr(config, "lambda_anti_distill", 0.1) * anti_loss
        base["anti_distill"] = anti_loss

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
        method_name="GenealogyDistill-MP",
        exp_id="exp10",
        loss_fn=genealogy_distill_mp_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp10_geneal_mp",
    )

    emit_ablation_suite(
        method_name="GenealogyDistill-MP",
        exp_id="exp10",
        loss_fn=genealogy_distill_mp_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
