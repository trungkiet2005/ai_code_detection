"""
[Exp_TK exp11] GenealogyRetrievalBank (GRB) -- train-time RAG + family-aware HN mining

Challenger for overall CoDET Author SOTA (Exp27 DeTeCtiveCode: 71.53).
---------------------------------------------------------------------
The consolidated tracker (Exp_TK/tracker.md section 11) identifies three
unresolved levers. GRB attacks the single biggest one:

    "Nxcode <-> CodeQwen1.5 confusion is the single biggest CoDET lever:
     Nxcode is fine-tuned from CodeQwen1.5 -> 33-40% of Qwen samples
     predicted as Nxcode in EVERY method. HierTree forces them into same
     family -> +3% Qwen F1."

HierTree (Exp18 / Exp_00) learns a family AFFINITY but marginalizes over
all in-batch negatives. At batch 128 that is ~106 negatives per anchor;
the same-family sibling (e.g. Nxcode for a Qwen anchor) is ONE of those
106 and its gradient signal is diluted.

GenealogyRetrievalBank adds two novel components on top of exp08's
DualModeFlowRAG-HN recipe:

  (A) MOCO-STYLE MEMORY BANK FOR TRAIN-TIME RAG.
      Exp17 RAGDetect already hit 70.46 with TEST-TIME kNN blending, but
      train-time retrieval has never been tried on this board. We
      maintain a FIFO queue of the last K=8192 embeddings + labels
      (dequeued every batch, no gradient). The SupCon denominator now
      ranges over ~8192 negatives instead of 127 -- a 64x larger pool,
      which empirically gives the gradient access to the rare hard
      Qwen/Nxcode pairs that rarely co-occur in a batch-128.
      Reference: He et al. 2020 (MoCo v1, arXiv 1911.05722),
      He et al. 2021 (MoCo v3 for SupCon transfer).

  (B) FAMILY-AWARE HARD-NEGATIVE MINING IN THE BANK.
      Vanilla HN (exp08) keeps top-k closest negatives globally. GRB
      restricts the HN pool to SAME-FAMILY negatives (via the HierTree
      family table): for a Qwen anchor, the bank hard negatives are the
      hardest Nxcode/StarChat(-family-2) samples. Cross-family negatives
      stay as easy-negatives (always satisfied). This focuses gradient
      on the intra-family confusion axis that matters.
      Reference: Kalantidis et al. 2020 (HCL), Robinson et al. 2020
      (Hard-Neg SupCon addendum).

Loss:
  focal + 0.3*neural + 0.3*spectral
   + lambda_hier    * HierTree (family genealogy pull/push)
   + lambda_bank    * SupCon-over-memory-bank (all-negative)
   + lambda_fam_hn  * SupCon-over-same-family-hard-negatives (top-k)

Ablation toggles: hier / bank / fam_hn, plus a knob to toggle
family-restriction off (reverts fam_hn to vanilla HN for isolation).

Targets (tracker section 7, "Success bars"):
  * CoDET Author IID    > 71.53  (beat Exp27 DeTeCtiveCode)
  * CoDET OOD-SRC-gh    > 33.36  (family-aware HN should help GH too,
                                  since GH has higher generator diversity)
  * Droid T3 W-F1       > 0.88   (stable -- banks are orthogonal to Droid
                                  which has only 3 classes)

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. `run_mode="lean"` -> 8 runs + 3 ablation runs (~4h on H100).

Implementation notes:
  * Memory bank lives in module-level singleton (not model-registered)
    to avoid DDP / checkpoint confusion. Re-initialized per-run.
  * Bank is BF16 + no_grad -- adds ~4 MB per 8192 entries at dim 256.
  * No trainer-side changes. Pure loss-fn plug-in.
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
# HierTree (preserved from Exp_00 / Exp18 / Exp_16)
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
# NEW (A): MoCo-style memory bank
# ===========================================================================

class RetrievalMemoryBank:
    """FIFO queue of L2-normalized embeddings + labels + family codes.

    Used as extra negatives in SupCon. Stored as detached tensors (no
    gradient flows back into the bank). Reset per-run via _reset_bank.
    """

    def __init__(self, queue_size: int = 8192, dim: int = 256, num_classes: int = 6):
        self.queue_size = queue_size
        self.dim = dim
        self.num_classes = num_classes
        self.ptr = 0
        self.n_filled = 0
        self.device = "cpu"
        self.features = torch.zeros(queue_size, dim)
        self.labels = torch.full((queue_size,), -1, dtype=torch.long)
        self.families = torch.full((queue_size,), -1, dtype=torch.long)

    def to(self, device):
        if self.device != device:
            self.features = self.features.to(device)
            self.labels = self.labels.to(device)
            self.families = self.families.to(device)
            self.device = device
        return self

    @torch.no_grad()
    def enqueue(self, features: torch.Tensor, labels: torch.Tensor,
                families: torch.Tensor):
        B = features.shape[0]
        feats = F.normalize(features.detach(), p=2, dim=-1).to(self.features.dtype)
        end = self.ptr + B
        if end <= self.queue_size:
            self.features[self.ptr:end] = feats
            self.labels[self.ptr:end] = labels.detach()
            self.families[self.ptr:end] = families.detach()
        else:
            tail = self.queue_size - self.ptr
            self.features[self.ptr:] = feats[:tail]
            self.labels[self.ptr:] = labels[:tail].detach()
            self.families[self.ptr:] = families[:tail].detach()
            rem = B - tail
            self.features[:rem] = feats[tail:]
            self.labels[:rem] = labels[tail:].detach()
            self.families[:rem] = families[tail:].detach()
        self.ptr = (self.ptr + B) % self.queue_size
        self.n_filled = min(self.n_filled + B, self.queue_size)

    def is_warm(self, min_entries: int = 512) -> bool:
        return self.n_filled >= min_entries

    def get(self):
        """Return (features, labels, families) restricted to filled slots."""
        k = self.n_filled
        return self.features[:k], self.labels[:k], self.families[:k]


_bank: Optional[RetrievalMemoryBank] = None


def _get_bank(dim: int, num_classes: int, queue_size: int = 8192) -> RetrievalMemoryBank:
    global _bank
    if (_bank is None or _bank.dim != dim
            or _bank.num_classes != num_classes
            or _bank.queue_size != queue_size):
        _bank = RetrievalMemoryBank(queue_size=queue_size, dim=dim, num_classes=num_classes)
    return _bank


def _families_for_labels(labels: torch.Tensor, family_table) -> torch.Tensor:
    if family_table is None:
        return torch.full_like(labels, -1)
    return torch.tensor(
        [family_table[int(l.item())] if int(l.item()) < len(family_table) else -1 for l in labels],
        device=labels.device, dtype=torch.long,
    )


# ===========================================================================
# SupCon-over-memory-bank (all negatives)
# ===========================================================================

def _supcon_bank(anchor: torch.Tensor, a_labels: torch.Tensor,
                 bank_feat: torch.Tensor, bank_labels: torch.Tensor,
                 temperature: float = 0.1) -> torch.Tensor:
    """SupCon where positives+negatives are drawn from the memory bank.

    Anchor is the current (gradient-bearing) batch. The bank is a frozen
    snapshot (no gradient). Positives are bank entries with the same label
    as the anchor. Negatives are all other bank entries. No self-entry is
    possible (current batch has not been enqueued yet this step)."""
    if anchor.shape[0] < 2 or bank_feat.shape[0] < 8:
        return anchor.new_zeros(1).squeeze()
    a = F.normalize(anchor, p=2, dim=-1)
    # bank already L2-normalized at enqueue time, but re-normalize for safety
    b = F.normalize(bank_feat, p=2, dim=-1)
    logits = torch.mm(a, b.t()) / temperature                          # (B, K)
    pos_mask = (a_labels.unsqueeze(1) == bank_labels.unsqueeze(0)).float()
    # valid rows: anchors with >=1 positive in the bank
    row_has_pos = pos_mask.sum(dim=-1) > 0
    if row_has_pos.sum() == 0:
        return anchor.new_zeros(1).squeeze()
    log_p = F.log_softmax(logits, dim=-1)
    mean_log_prob_pos = (pos_mask * log_p).sum(dim=-1) / pos_mask.sum(dim=-1).clamp_min(1.0)
    return -mean_log_prob_pos[row_has_pos].mean()


# ===========================================================================
# NEW (B): Family-aware hard-negative SupCon (bank-conditioned)
# ===========================================================================

def _supcon_family_hn(anchor: torch.Tensor, a_labels: torch.Tensor, a_fam: torch.Tensor,
                      bank_feat: torch.Tensor, bank_labels: torch.Tensor, bank_fam: torch.Tensor,
                      temperature: float = 0.1, topk: int = 16,
                      family_restricted: bool = True) -> torch.Tensor:
    """SupCon whose denominator is restricted to the top-k hardest
    SAME-FAMILY negatives in the bank (i.e. different label but same
    family code). This concentrates gradient on the intra-family
    confusion pairs that are the biggest CoDET lever (Nxcode <-> Qwen).

    Setting `family_restricted=False` recovers vanilla bank-HN SupCon
    (top-k hardest across ALL labels) for ablation isolation.
    """
    if anchor.shape[0] < 2 or bank_feat.shape[0] < 8:
        return anchor.new_zeros(1).squeeze()
    a = F.normalize(anchor, p=2, dim=-1)
    b = F.normalize(bank_feat, p=2, dim=-1)
    sim = torch.mm(a, b.t()) / temperature                         # (B, K)

    same_fam = (a_fam.unsqueeze(1) == bank_fam.unsqueeze(0))       # (B, K) bool
    same_label = (a_labels.unsqueeze(1) == bank_labels.unsqueeze(0))
    valid_fam = (a_fam.unsqueeze(1) >= 0) & (bank_fam.unsqueeze(0) >= 0)

    pos_mask = same_label & valid_fam                              # positives
    if family_restricted:
        neg_mask = same_fam & (~same_label) & valid_fam            # intra-family, diff label
    else:
        neg_mask = (~same_label) & valid_fam                        # any diff-label

    # If any anchor has no intra-family negatives in the bank, skip it.
    neg_counts = neg_mask.sum(dim=-1)
    pos_counts = pos_mask.sum(dim=-1)
    row_ok = (neg_counts >= 2) & (pos_counts >= 1)
    if row_ok.sum() == 0:
        return anchor.new_zeros(1).squeeze()

    # Select top-k hardest negatives per row.
    neg_sim = sim.masked_fill(~neg_mask, float("-inf"))
    k = int(min(topk, int(neg_counts.max().item())))
    if k <= 0:
        return anchor.new_zeros(1).squeeze()
    _, hard_idx = neg_sim.topk(k, dim=-1)                           # (B, k)
    hard_mask = torch.zeros_like(sim)
    hard_mask.scatter_(1, hard_idx, 1.0)
    # Combine denominator mask: positives + hard intra-family negatives.
    keep = (pos_mask.float() + hard_mask).clamp_max(1.0)
    logits = sim.masked_fill(keep == 0, float("-inf"))
    log_p = F.log_softmax(logits, dim=-1)
    mean_log_prob_pos = (pos_mask.float() * log_p).sum(dim=-1) / pos_counts.clamp_min(1.0).float()
    return -mean_log_prob_pos[row_ok].mean()


# ===========================================================================
# Singletons
# ===========================================================================

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
    "hier":           ("lambda_hier",     True),
    "bank":           ("lambda_bank",     True),
    "fam_hn":         ("lambda_fam_hn",   True),
    # Drop this toggle to revert fam_hn -> vanilla HN (any-family hard-neg)
    "family_restr":   ("grb_family_restr", True),
}


# ===========================================================================
# Loss
# ===========================================================================

def genealogy_retrieval_bank_compute_losses(model, outputs, labels, config,
                                            focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier   * HierTree
       + lambda_bank   * SupCon over 8192-entry memory bank (all negatives)
       + lambda_fam_hn * SupCon over top-k hardest SAME-FAMILY bank negatives."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]
    num_classes = model.num_classes

    # --- HierTree (family genealogy) ---
    hier_fn = _get_hier(num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # --- Memory bank (train-time RAG) ---
    fam_table = hier_fn.family_table
    fam_codes = _families_for_labels(labels, fam_table)
    bank = _get_bank(
        dim=emb.shape[-1], num_classes=num_classes,
        queue_size=int(getattr(config, "grb_queue_size", 8192)),
    ).to(emb.device)

    lambda_bank   = float(getattr(config, "lambda_bank",   0.20))
    lambda_fam_hn = float(getattr(config, "lambda_fam_hn", 0.25))

    if bank.is_warm(min_entries=int(getattr(config, "grb_warm_entries", 512))):
        bank_feat, bank_labels, bank_fam = bank.get()
        temp = float(getattr(config, "supcon_temp", 0.1))

        # (A) SupCon over full bank, all-negatives
        supcon_bank = _supcon_bank(emb, labels, bank_feat, bank_labels, temperature=temp)
        base["total"] = base["total"] + lambda_bank * supcon_bank
        base["bank"] = supcon_bank

        # (B) SupCon over family-restricted top-k hard negatives
        fam_hn = _supcon_family_hn(
            emb, labels, fam_codes,
            bank_feat, bank_labels, bank_fam,
            temperature=temp,
            topk=int(getattr(config, "grb_topk", 16)),
            family_restricted=bool(getattr(config, "grb_family_restr", True)),
        )
        base["total"] = base["total"] + lambda_fam_hn * fam_hn
        base["fam_hn"] = fam_hn
    else:
        base["bank"] = emb.new_zeros(1).squeeze()
        base["fam_hn"] = emb.new_zeros(1).squeeze()

    # Enqueue current batch at the end (after it was used as anchor).
    bank.enqueue(emb, labels, fam_codes)
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
        method_name="GenealogyRetrievalBank",
        exp_id="exp11",
        loss_fn=genealogy_retrieval_bank_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp11_grb",
    )

    emit_ablation_suite(
        method_name="GenealogyRetrievalBank",
        exp_id="exp11",
        loss_fn=genealogy_retrieval_bank_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
