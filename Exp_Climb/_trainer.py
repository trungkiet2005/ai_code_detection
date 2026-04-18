"""
Trainer + losses.

Trainer is **loss-function-agnostic**: pass your own `loss_fn(model, outputs,
labels, config, focal_loss_fn)` returning a dict with at least `"total"`.
Per-method losses (HierarchicalAffinity, SupCon, etc.) stay in the exp file.
"""
from __future__ import annotations

import logging
import os
from collections import defaultdict
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)
from transformers import get_cosine_schedule_with_warmup

from _common import SpectralConfig, autocast, logger, make_grad_scaler


# ---------------------------------------------------------------------------
# Losses
# ---------------------------------------------------------------------------

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


def default_compute_losses(model, outputs, labels, config, focal_loss_fn: Optional[FocalLoss] = None):
    """Baseline loss: focal on final + 0.3 on neural + 0.3 on spectral. No hier / contrastive.
    Exp files override by passing their own compute_losses to Trainer."""
    if focal_loss_fn is None:
        focal_loss_fn = FocalLoss(gamma=2.0)
    task_loss = focal_loss_fn(outputs["logits"], labels)
    neural_loss = focal_loss_fn(outputs["neural_logits"], labels)
    spectral_loss = focal_loss_fn(outputs["spectral_logits"], labels)
    total = task_loss + 0.3 * neural_loss + 0.3 * spectral_loss
    return {"total": total, "task": task_loss, "neural": neural_loss, "spectral": spectral_loss}


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

LossFn = Callable[..., Dict[str, torch.Tensor]]


class Trainer:
    def __init__(
        self,
        config: SpectralConfig,
        model: nn.Module,
        train_loader,
        val_loader,
        test_loader,
        class_weights: Optional[torch.Tensor] = None,
        loss_fn: Optional[LossFn] = None,
        checkpoint_tag_prefix: str = "model",
    ):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_weights = class_weights.to(config.device) if class_weights is not None else None
        self.loss_fn: LossFn = loss_fn or default_compute_losses
        self.checkpoint_tag_prefix = checkpoint_tag_prefix

        encoder_params = list(model.token_encoder.parameters())
        head_params = [p for n, p in model.named_parameters() if "token_encoder" not in n]
        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": config.lr_encoder, "weight_decay": config.weight_decay},
            {"params": head_params, "lr": config.lr_heads, "weight_decay": config.weight_decay},
        ])
        total_steps = len(train_loader) * config.epochs // max(config.grad_accum_steps, 1)
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

        precision = config.precision.lower()
        if precision == "auto":
            precision = "bf16" if config.device == "cuda" else "fp32"
        self.precision = precision
        self.use_amp = config.device == "cuda" and precision in ("bf16", "fp16")
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = make_grad_scaler(enabled=(self.use_amp and precision == "fp16"))
        self.focal_loss = FocalLoss(gamma=2.0, weight=self.class_weights)

        self.best_f1 = 0.0
        self.global_step = 0
        self.last_eval_metrics: Dict[str, Dict[str, object]] = {}

    # ---- Training loop --------------------------------------------------
    def train_epoch(self, epoch: int):
        self.model.train()
        total_losses: Dict[str, float] = defaultdict(float)
        num_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.config.device, non_blocking=self.config.non_blocking)
            attention_mask = batch["attention_mask"].to(self.config.device, non_blocking=self.config.non_blocking)
            ast_seq = batch["ast_seq"].to(self.config.device, non_blocking=self.config.non_blocking)
            struct_feat = batch["struct_feat"].to(self.config.device, non_blocking=self.config.non_blocking)
            labels = batch["label"].to(self.config.device, non_blocking=self.config.non_blocking)

            with autocast(device_type=self.config.device, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(input_ids, attention_mask, ast_seq, struct_feat, labels)
                losses = self.loss_fn(self.model, outputs, labels, self.config, self.focal_loss)
                loss = losses["total"] / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            for k, v in losses.items():
                total_losses[k] += float(v.item()) if torch.is_tensor(v) else float(v)
            num_batches += 1

            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_losses["total"] / num_batches
                lr = self.scheduler.get_last_lr()[0]
                extras = " | ".join(
                    f"{k}: {total_losses[k] / num_batches:.4f}"
                    for k in sorted(total_losses)
                    if k not in ("total",)
                )
                logger.info(
                    f"Epoch {epoch+1} | Step {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | {extras} | LR: {lr:.2e}"
                )

            if self.global_step > 0 and self.global_step % self.config.eval_every == 0:
                val_f1 = self.evaluate(self.val_loader, "Val")
                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.save_checkpoint("best")
                    logger.info(f"New best Val F1: {val_f1:.4f}")
                self.model.train()

        return {k: v / num_batches for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self, dataloader, split_name: str = "Val") -> float:
        self.model.eval()
        all_preds: List[int] = []
        all_labels: List[int] = []
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.config.device, non_blocking=self.config.non_blocking)
            attention_mask = batch["attention_mask"].to(self.config.device, non_blocking=self.config.non_blocking)
            ast_seq = batch["ast_seq"].to(self.config.device, non_blocking=self.config.non_blocking)
            struct_feat = batch["struct_feat"].to(self.config.device, non_blocking=self.config.non_blocking)
            labels = batch["label"].to(self.config.device, non_blocking=self.config.non_blocking)
            with autocast(device_type=self.config.device, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(input_ids, attention_mask, ast_seq, struct_feat, labels)
            preds = outputs["logits"].argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_loss += F.cross_entropy(outputs["logits"], labels).item()
            num_batches += 1

        macro_f1 = float(f1_score(all_labels, all_preds, average="macro"))
        weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted"))
        macro_recall = float(recall_score(all_labels, all_preds, average="macro", zero_division=0))
        weighted_recall = float(recall_score(all_labels, all_preds, average="weighted", zero_division=0))
        accuracy = float(accuracy_score(all_labels, all_preds))
        avg_loss = total_loss / max(num_batches, 1)

        logger.info(
            f"{split_name} | Loss: {avg_loss:.4f} | Macro-F1: {macro_f1:.4f} | "
            f"Weighted-F1: {weighted_f1:.4f}"
        )
        self.last_eval_metrics[split_name] = {
            "loss": avg_loss,
            "macro_f1": macro_f1,
            "weighted_f1": weighted_f1,
            "macro_recall": macro_recall,
            "weighted_recall": weighted_recall,
            "accuracy": accuracy,
        }
        if split_name == "Test":
            report = classification_report(all_labels, all_preds, digits=4, zero_division=0)
            logger.info(f"\n{split_name} Classification Report:\n{report}")
            report_dict = classification_report(
                all_labels, all_preds, digits=4, output_dict=True, zero_division=0,
            )
            self.last_eval_metrics[split_name]["classification_report"] = report_dict
        return macro_f1

    def _ckpt_path(self, tag: str) -> str:
        os.makedirs(self.config.save_dir, exist_ok=True)
        return os.path.join(
            self.config.save_dir, f"{self.checkpoint_tag_prefix}_{self.config.task}_{tag}.pt"
        )

    def save_checkpoint(self, tag: str = "latest"):
        path = self._ckpt_path(tag)
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_f1": self.best_f1,
            "global_step": self.global_step,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, tag: str = "best") -> bool:
        path = self._ckpt_path(tag)
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.config.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded checkpoint from {path}")
            return True
        return False

    # ---- Full train-then-test driver ------------------------------------
    def train(self) -> Dict[str, object]:
        logger.info("=" * 60)
        logger.info(f"Starting training | task={self.config.task} | classes={self.model.num_classes}")
        logger.info(
            f"Device={self.config.device} | Precision={self.precision} | "
            f"Batch={self.config.batch_size}x{self.config.grad_accum_steps}="
            f"{self.config.batch_size * self.config.grad_accum_steps}"
        )
        logger.info("=" * 60)

        for epoch in range(self.config.epochs):
            logger.info(f"\n{'='*40} Epoch {epoch+1}/{self.config.epochs} {'='*40}")
            train_losses = self.train_epoch(epoch)
            logger.info(
                "Epoch %d Train Summary: %s", epoch + 1,
                " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items()),
            )
            val_f1 = self.evaluate(self.val_loader, "Val")
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.save_checkpoint("best")
                logger.info(f"*** New Best Val Macro-F1: {val_f1:.4f} ***")
            self.save_checkpoint("latest")

        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST EVALUATION")
        logger.info("=" * 60)
        self.load_checkpoint("best")
        test_f1 = self.evaluate(self.test_loader, "Test")
        tm = self.last_eval_metrics.get("Test", {})
        test_weighted = tm.get("weighted_f1", test_f1)
        logger.info(f"*** Final Test Macro-F1: {test_f1:.4f} | Weighted-F1: {test_weighted:.4f} ***")

        return {
            "test_f1": float(test_f1),
            "test_macro_f1": float(test_f1),
            "test_weighted_f1": float(test_weighted),
            "test_macro_recall": float(tm.get("macro_recall", 0.0)),
            "test_weighted_recall": float(tm.get("weighted_recall", 0.0)),
            "test_accuracy": float(tm.get("accuracy", 0.0)),
            "best_val_f1": float(self.best_f1),
            "paper_primary_metric": "macro_f1",
            "num_classes": int(self.model.num_classes),
            "test_per_class": tm.get("classification_report", {}),
        }
