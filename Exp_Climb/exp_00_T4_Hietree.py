"""
[exp_00_T4] HierTreeCode -- Kaggle T4-tuned variant of exp_00_hiertree.py

Same method (Hierarchical Affinity Tree Loss) and same sibling _*.py helpers
as exp_00_hiertree.py. The ONLY differences live at the top of this file:

  1. A T4 hardware profile that replaces the H100 one via monkey-patch on
     `_common.apply_hardware_profile`. This keeps Exp_Climb/_common.py
     H100-first and untouched -- the T4 variant is a self-contained override.

  2. Smaller CLIMB data budgets so the full suite fits a single 12h Kaggle
     T4 session (T4 is ~4x slower than H100 and has 16 GB VRAM vs 80 GB).

T4 profile choices (why each one):
  * precision = fp16    -- T4 is Turing arch, no native bf16 support.
  * batch_size = 16     -- ModernBERT-base + seq 512 + AMP fp16 ~ 10-12 GB.
  * grad_accum = 2      -- effective batch 32 (matches H100 baseline LR).
  * seq_len = 512       -- paper baseline; seq 1024 OOMs on 16 GB.
  * workers = 2         -- Kaggle notebooks have 4 vCPU; 2 leaves headroom.
  * lr kept at 2e-5     -- effective batch 32 = base, no sqrt-scaling needed.

Note: running the FULL climb (IID + author + 11 OOD LOO = ~16 runs) on a
single T4 in 12 h is tight. Defaults below reduce train to 30 K samples and
epochs to 2. If you only need paper-comparable IID numbers, switch
`run_mode="codet_iid"` to drop to 2 runs (~2 h on T4).

Kaggle workflow:
  1. Upload ONLY this file to Kaggle.
  2. Attach the "GPU T4 x2" accelerator (this script uses 1 of the 2 T4s).
  3. Run. Logs clearly show the applied T4 profile.
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap: clone repo if we're running outside Exp_Climb/ (e.g. fresh Kaggle)
# ---------------------------------------------------------------------------

import os
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"


def _bootstrap_climb_path() -> str:
    """Find or clone the Exp_Climb folder so sibling _*.py modules are importable."""
    cwd = os.getcwd()

    for candidate in (
        os.path.join(cwd, "Exp_Climb"),
        os.path.join(cwd, "ai_code_detection", "Exp_Climb"),
    ):
        if os.path.exists(os.path.join(candidate, "_common.py")):
            return candidate

    try:
        here = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        if os.path.exists(os.path.join(here, "_common.py")):
            return here
    except NameError:
        pass  # Kaggle notebook upload -- __file__ not defined

    repo_dir = os.path.join(cwd, "ai_code_detection")
    if not os.path.exists(repo_dir):
        print(f"[bootstrap] Cloning {REPO_URL} -> {repo_dir}")
        subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, repo_dir])
    return os.path.join(repo_dir, "Exp_Climb")


_climb_dir = _bootstrap_climb_path()
if _climb_dir not in sys.path:
    sys.path.insert(0, _climb_dir)
print(f"[bootstrap] Exp_Climb path: {_climb_dir}")


# ---------------------------------------------------------------------------
# T4 hardware-profile monkey-patch (must run BEFORE _data_codet / _data_droid
# import `apply_hardware_profile` and hit the H100 branch)
# ---------------------------------------------------------------------------

import torch

import _common  # noqa: E402

_original_apply_hardware_profile = _common.apply_hardware_profile


def _apply_t4_profile(config):
    """Replacement for apply_hardware_profile with a T4 branch.

    Falls back to the original function for non-T4 GPUs (so H100 / A100
    machines still get the tuned profile from _common.py).
    """
    if config.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    gpu_name = _common.get_gpu_name().upper()
    if "T4" not in gpu_name:
        return _original_apply_hardware_profile(config)

    # T4 16 GB Turing profile
    config.precision = "fp16" if config.precision == "auto" else config.precision
    config.batch_size = 16
    config.grad_accum_steps = 2
    config.max_length = 512
    config.num_workers = 2
    config.prefetch_factor = 2
    config.log_every = 100
    config.eval_every = 1000
    # Keep base LR (effective batch 32 = paper baseline, no scaling needed)
    config.lr_encoder = 2e-5
    config.lr_heads = 1e-4
    _common.logger.info(
        "Applied Kaggle T4 profile | precision=%s | batch=%d (grad_accum=%d, eff=%d) | "
        "seq=%d | lr_enc=%.2e | lr_heads=%.2e | workers=%d",
        config.precision, config.batch_size, config.grad_accum_steps,
        config.batch_size * config.grad_accum_steps, config.max_length,
        config.lr_encoder, config.lr_heads, config.num_workers,
    )
    return config


_common.apply_hardware_profile = _apply_t4_profile


# ---------------------------------------------------------------------------
# Standard imports from the shared Exp_Climb modules (AFTER the patch above)
# ---------------------------------------------------------------------------

from typing import Optional  # noqa: E402

import torch.nn as nn  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from _common import logger  # noqa: E402
from _trainer import FocalLoss, default_compute_losses  # noqa: E402
from _data_codet import CoDETM4Config  # noqa: E402
from _data_droid import DroidConfig  # noqa: E402
from _climb_runner import run_full_climb  # noqa: E402


# ===========================================================================
# Method-specific: HierarchicalAffinityLoss (identical to exp_00_hiertree.py)
# ===========================================================================

# CoDET-M4 6-class labels: 0=human, 1=codellama, 2=gpt, 3=llama3.1, 4=nxcode, 5=qwen1.5
#   family 0 = human
#   family 1 = llama-family (CodeLlama, Llama3.1)
#   family 2 = gpt
#   family 3 = qwen-family (Nxcode is fine-tuned Qwen -> same family)
AUTHOR_FAMILY_CODET = [0, 1, 2, 1, 3, 3]


def _build_family_table(num_classes: int) -> Optional[list]:
    if num_classes == 6:
        return AUTHOR_FAMILY_CODET
    if num_classes == 3:
        return [0, 1, 1]
    if num_classes == 4:
        return [0, 1, 1, 1]
    return None


class HierarchicalAffinityLoss(nn.Module):
    """Batch-hard triplet loss over family labels (cosine distance)."""

    def __init__(self, margin: float = 0.3, num_classes: int = 6):
        super().__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.family_table = _build_family_table(num_classes)
        self.active = self.family_table is not None

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.active or embeddings.shape[0] < 4:
            return embeddings.new_zeros(1).squeeze()
        family_labels = torch.tensor(
            [self.family_table[l.item()] if l.item() < len(self.family_table) else -1 for l in labels],
            device=labels.device,
        )
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        cos_sim = torch.mm(emb_norm, emb_norm.t())
        dist = 1.0 - cos_sim

        loss = embeddings.new_zeros(1).squeeze()
        count = 0
        B = embeddings.shape[0]
        for i in range(B):
            fi = family_labels[i].item()
            if fi == -1:
                continue
            same_mask = (family_labels == fi)
            same_mask[i] = False
            diff_mask = (family_labels != fi) & (family_labels != -1)
            if same_mask.sum() == 0 or diff_mask.sum() == 0:
                continue
            d_pos = dist[i][same_mask].max()
            d_neg = dist[i][diff_mask].min()
            triplet = F.relu(d_pos - d_neg + self.margin)
            loss = loss + triplet
            count += 1
        return loss / max(count, 1)


_hier_fn: Optional[HierarchicalAffinityLoss] = None


def _get_hier_loss(num_classes: int, margin: float) -> HierarchicalAffinityLoss:
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


def hiertree_compute_losses(model, outputs, labels, config, focal_loss_fn: Optional[FocalLoss] = None):
    """Focal + 0.3 neural + 0.3 spectral + lambda_hier * hier."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    hier_fn = _get_hier_loss(model.num_classes, getattr(config, "hier_margin", 0.3))
    hier_fn = hier_fn.to(outputs["embeddings"].device)
    hier_loss = hier_fn(outputs["embeddings"], labels)
    lambda_hier = getattr(config, "lambda_hier", 0.4)
    base["total"] = base["total"] + lambda_hier * hier_loss
    base["hier"] = hier_loss
    return base


# ===========================================================================
# Entry point -- T4-sized data budgets
# ===========================================================================

if __name__ == "__main__":
    # CLIMB protocol on T4: reduce train to fit 12 h Kaggle session.
    # Test split stays FULL so numbers remain paper-comparable.
    codet_cfg = CoDETM4Config(
        max_train_samples=30_000,    # ~6% of ~500K (vs 20% on H100)
        max_val_samples=5_000,
        max_test_samples=-1,
        eval_breakdown=True,
    )
    droid_cfg = DroidConfig(
        max_train_samples=30_000,
        max_val_samples=5_000,
        max_test_samples=-1,
    )

    # On a single T4, the "full" suite (~16 train-eval cycles) may spill past
    # the 12 h Kaggle limit. For a faster first-pass run, switch to
    # run_mode="codet_iid" (2 runs, ~2 h on T4) to get Table-2/7 numbers.
    run_full_climb(
        method_name="HierTreeCode-T4",
        exp_id="exp_00_T4",
        loss_fn=hiertree_compute_losses,
        codet_cfg=codet_cfg,
        droid_cfg=droid_cfg,
        run_mode="full",
        run_preflight=True,
        checkpoint_tag_prefix="exp_00_T4_hiertree",
    )
