"""
Few-shot model: ModernBERT encoder + linear classifier head + optional NTK projector.

Lean-stack design:
  - No AST / tree-sitter (T4 may fail to install — skip entirely)
  - No spectral / FFT branch (negligible at K-shot, not worth the VRAM)
  - Encoder mean-pool + dropout + linear -> n_classes
  - Optional NTK alignment projector (Exp_FS_01)

Standalone: no Exp_Climb imports.
"""
from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from _common_fs import FSConfig, logger


def _mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_f = mask.unsqueeze(-1).float()
    summed = (hidden * mask_f).sum(dim=1)
    denom = mask_f.sum(dim=1).clamp(min=1.0)
    return summed / denom


class FSClassifier(nn.Module):
    """Bare-bones ModernBERT classifier for few-shot.

    forward returns dict:
      logits:    (B, n_classes)         <- main classifier
      embedding: (B, hidden)            <- mean-pooled features (for NTK / probes)
      ntk_proj:  (B, ntk_proj_dim)      <- L2-normalized projection (NTK loss)
    """

    def __init__(self, cfg: FSConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = AutoModel.from_pretrained(cfg.encoder_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, cfg.n_classes)
        self.ntk_proj = nn.Sequential(
            nn.Linear(hidden, cfg.ntk_proj_dim),
            nn.GELU(),
            nn.Linear(cfg.ntk_proj_dim, cfg.ntk_proj_dim),
        )
        logger.info(
            f"[model] {cfg.encoder_name} hidden={hidden} -> "
            f"classifier({cfg.n_classes}), ntk_proj({cfg.ntk_proj_dim})"
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = _mean_pool(out.last_hidden_state, attention_mask)
        logits = self.classifier(self.dropout(emb))
        ntk = F.normalize(self.ntk_proj(emb), dim=-1)
        return {"logits": logits, "embedding": emb, "ntk_proj": ntk}

    def param_groups(self):
        """Two-LR optimizer groups: encoder vs heads."""
        encoder_params = list(self.encoder.parameters())
        head_params = list(self.classifier.parameters()) + list(self.ntk_proj.parameters())
        return [
            {"params": encoder_params, "lr": self.cfg.lr_encoder, "weight_decay": self.cfg.weight_decay},
            {"params": head_params, "lr": self.cfg.lr_heads, "weight_decay": self.cfg.weight_decay},
        ]


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

def cross_entropy_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    return {"total": ce, "ce": ce}


def ntk_alignment_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_ntk: float = 0.4,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """CE + NTK target-kernel alignment.

    Target kernel: y_ij = 1 if labels[i]==labels[j] else 0  (normalized).
    Empirical kernel: K_ij = <ntk_proj_i, ntk_proj_j>  (cosine since L2-normalized).
    Loss term: ||K - Y||_F^2 / B^2  -- pull same-class projections together,
    push others apart, in a kernel-aligned sense (Cristianini 2001 / Jacot 2018).
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)

    z = outputs["ntk_proj"]                       # (B, D), L2-normalized
    K = z @ z.t()                                 # (B, B) cosine kernel
    Y = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()  # (B, B) target kernel
    # Center both kernels (HSIC-style) so the alignment is invariant to mean shift.
    B = z.size(0)
    H = torch.eye(B, device=z.device) - torch.full((B, B), 1.0 / B, device=z.device)
    K_c = H @ K @ H
    Y_c = H @ Y @ H
    # Scaled Frobenius distance: averaged per pair.
    align = ((K_c - Y_c) ** 2).mean()

    total = ce + lambda_ntk * align
    return {"total": total, "ce": ce, "ntk_align": align}


def supcon_loss(
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    lambda_supcon: float = 0.4,
    temperature: float = 0.07,
    class_weights: Optional[torch.Tensor] = None,
) -> Dict[str, torch.Tensor]:
    """CE + Supervised Contrastive (Khosla et al. NeurIPS 2020).

    Compared to NTK-alignment: SupCon uses a softmax over similarities, not a
    Frobenius distance. Empirically less sensitive to small batch sizes
    because it normalises per-anchor (sum over positives only), so a single
    same-class pair already provides usable signal.

    L_supcon = - 1/B * sum_i 1/|P(i)| * sum_{p in P(i)} log
                exp(z_i z_p / T) / sum_{a != i} exp(z_i z_a / T)

    Anchors with no in-batch positives contribute zero (skipped).
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)

    z = outputs["ntk_proj"]                       # reuse the same L2-normalised projector
    sim = z @ z.t() / temperature                 # (B, B) similarity matrix
    B = z.size(0)
    # Mask out self-similarity (i, i) entries.
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(eye, float("-inf"))

    # Positive mask: same-label, not self.
    pos_mask = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    has_pos = pos_mask.any(dim=1)

    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    # Mean log-prob over positives, per anchor.
    pos_log_prob = (log_prob * pos_mask.float()).sum(dim=1) / pos_mask.float().sum(dim=1).clamp(min=1.0)
    # Skip anchors with no positives (sum*0 / 1 = 0 anyway).
    if has_pos.any():
        sup = -(pos_log_prob[has_pos]).mean()
    else:
        sup = torch.zeros((), device=z.device)

    total = ce + lambda_supcon * sup
    return {"total": total, "ce": ce, "supcon": sup}
