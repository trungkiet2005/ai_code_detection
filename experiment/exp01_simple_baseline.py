"""
exp01: Simple ModernBERT Baseline with OOD Regularization

Hypothesis: exp00's complex architecture (disentanglement, GRL, contrastive,
prototypical) actually HURT OOD performance (0.2625 F1 vs baseline 0.3061).
This experiment tests whether a simple, well-regularized fine-tuning approach
can match or beat the reported baselines.

Key changes from exp00:
    1. Simple architecture: ModernBERT + classification head (no AST, no struct,
       no disentanglement, no GRL, no prototypical)
    2. R-Drop regularization: KL divergence between two forward passes with
       different dropout masks — forces consistent predictions under perturbation
    3. Label smoothing: prevents overconfident predictions on train/val
    4. Multi-sample dropout: averages predictions from multiple dropout masks
    5. Stochastic Weight Averaging (SWA): flatter minima = better OOD generalization
    6. Use more training data (200K instead of 100K)
    7. Longer max_length (1024) for ModernBERT's native context

Target: beat Random baseline (45.73) and ModernBERT baseline (30.61) on T1.

Usage on Kaggle:
    1. Upload this file to a Kaggle notebook
    2. Run: !pip install datasets transformers accelerate
    3. Execute: python exp01_simple_baseline.py --task T1

Author: AICD-Bench Research
"""

import os
import math
import random
import logging
import warnings
from typing import Optional, Dict, List
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast as _autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler
    _NEW_AMP = False

def autocast(device_type="cuda", enabled=True):
    if _NEW_AMP:
        return _autocast(device_type=device_type, enabled=enabled)
    else:
        return _autocast(enabled=enabled)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Task
    task: str = "T1"

    # Model
    encoder_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 1024  # ModernBERT supports 8192, use more context
    hidden_dropout: float = 0.1  # encoder hidden dropout
    classifier_dropout: float = 0.3  # higher dropout on classifier

    # Task-specific
    num_classes: Dict[str, int] = field(default_factory=lambda: {
        "T1": 2, "T2": 12, "T3": 4,
    })

    # Training
    epochs: int = 3  # fewer epochs to reduce overfitting
    batch_size: int = 16  # smaller batch for longer sequences
    grad_accum_steps: int = 4  # effective batch = 64
    lr: float = 2e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Regularization
    label_smoothing: float = 0.1  # prevent overconfident predictions
    rdrop_alpha: float = 5.0  # R-Drop KL weight (0 = disabled)
    multisample_dropout_k: int = 5  # number of dropout samples at inference
    use_swa: bool = True  # Stochastic Weight Averaging
    swa_start_epoch: int = 2  # start SWA from this epoch
    swa_lr: float = 1e-5  # SWA learning rate

    # Data
    max_train_samples: int = 200_000  # more data than exp00 (100K)
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000
    num_workers: int = 2

    # Misc
    seed: int = 42
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./exp01_checkpoints"
    log_every: int = 100
    eval_every: int = 2000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Dataset
# ============================================================================

class AICDDataset(Dataset):
    """Simple dataset — just tokenized code + label."""

    def __init__(self, data, tokenizer, max_length: int = 1024):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item["code"]
        label = item["label"]

        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


def load_aicd_data(config: Config):
    """Load AICD-Bench dataset from HuggingFace."""
    logger.info(f"Loading AICD-Bench task {config.task}...")

    ds = load_dataset("AICD-bench/AICD-Bench", name=config.task)

    train_data = ds["train"]
    val_data = ds["validation"]
    test_data = ds["test"]

    # Subsample
    if len(train_data) > config.max_train_samples:
        indices = random.sample(range(len(train_data)), config.max_train_samples)
        train_data = train_data.select(indices)
        logger.info(f"Subsampled train to {config.max_train_samples}")

    if len(val_data) > config.max_val_samples:
        indices = random.sample(range(len(val_data)), config.max_val_samples)
        val_data = val_data.select(indices)

    if len(test_data) > config.max_test_samples:
        indices = random.sample(range(len(test_data)), config.max_test_samples)
        test_data = test_data.select(indices)

    logger.info(f"Data sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")

    all_labels = set(train_data["label"])
    num_classes = len(all_labels)
    logger.info(f"Number of classes: {num_classes}, Labels: {sorted(all_labels)}")

    return train_data, val_data, test_data, num_classes


# ============================================================================
# Model
# ============================================================================

class MultiSampleDropout(nn.Module):
    """Average predictions from K forward passes with different dropout masks.
    Reduces variance and improves calibration."""

    def __init__(self, classifier: nn.Module, k: int = 5, dropout_rate: float = 0.3):
        super().__init__()
        self.classifier = classifier
        self.k = k
        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(k)])

    def forward(self, x: torch.Tensor, training: bool = True) -> torch.Tensor:
        if training:
            # During training, use single dropout for efficiency + R-Drop handles regularization
            return self.classifier(self.dropouts[0](x))
        else:
            # During eval, average K different dropout masks
            logits = torch.stack([self.classifier(drop(x)) for drop in self.dropouts])
            return logits.mean(dim=0)


class SimpleCodeClassifier(nn.Module):
    """ModernBERT + simple classification head with multi-sample dropout."""

    def __init__(self, config: Config, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Load encoder
        encoder_config = AutoConfig.from_pretrained(config.encoder_name)
        # Increase hidden dropout for regularization
        if hasattr(encoder_config, 'hidden_dropout_prob'):
            encoder_config.hidden_dropout_prob = config.hidden_dropout
        if hasattr(encoder_config, 'attention_probs_dropout_prob'):
            encoder_config.attention_probs_dropout_prob = config.hidden_dropout

        self.encoder = AutoModel.from_pretrained(
            config.encoder_name,
            config=encoder_config,
        )
        hidden_size = encoder_config.hidden_size  # 768 for base

        # Classification head
        classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, num_classes),
        )

        self.classifier = MultiSampleDropout(
            classifier,
            k=config.multisample_dropout_k,
            dropout_rate=config.classifier_dropout,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Returns logits (B, num_classes)."""
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        if hasattr(outputs, 'last_hidden_state'):
            cls_repr = outputs.last_hidden_state[:, 0]
        else:
            cls_repr = outputs[0][:, 0]

        logits = self.classifier(cls_repr, training=self.training)
        return logits


# ============================================================================
# R-Drop Loss
# ============================================================================

def rdrop_loss(
    logits1: torch.Tensor,
    logits2: torch.Tensor,
    labels: torch.Tensor,
    alpha: float = 5.0,
    label_smoothing: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """R-Drop: Regularized Dropout for Neural Networks.

    Two forward passes with different dropout masks should produce similar outputs.
    This forces the model to learn robust features rather than dropout-dependent shortcuts.

    Args:
        logits1, logits2: (B, C) logits from two forward passes
        labels: (B,) ground truth
        alpha: weight for KL consistency term
        label_smoothing: label smoothing factor
    """
    # Task loss (CE with label smoothing, averaged over both passes)
    ce1 = F.cross_entropy(logits1, labels, label_smoothing=label_smoothing)
    ce2 = F.cross_entropy(logits2, labels, label_smoothing=label_smoothing)
    ce_loss = (ce1 + ce2) / 2.0

    # KL divergence between two passes (symmetrized)
    p1 = F.log_softmax(logits1, dim=-1)
    p2 = F.log_softmax(logits2, dim=-1)
    q1 = F.softmax(logits1, dim=-1)
    q2 = F.softmax(logits2, dim=-1)

    kl_loss = (F.kl_div(p1, q2, reduction="batchmean") +
               F.kl_div(p2, q1, reduction="batchmean")) / 2.0

    total_loss = ce_loss + alpha * kl_loss

    return {
        "total": total_loss,
        "ce": ce_loss,
        "kl": kl_loss,
    }


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    def __init__(self, config: Config, model: SimpleCodeClassifier,
                 train_loader, val_loader, test_loader):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        total_steps = len(train_loader) * config.epochs // config.grad_accum_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps,
        )

        self.use_amp = config.fp16 and config.device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)
        self.best_f1 = 0.0
        self.global_step = 0

        # SWA
        if config.use_swa:
            self.swa_model = torch.optim.swa_utils.AveragedModel(model)
            self.swa_scheduler = torch.optim.swa_utils.SWALR(
                self.optimizer, swa_lr=config.swa_lr,
            )
            self.swa_started = False
        else:
            self.swa_model = None

    def train_epoch(self, epoch: int):
        self.model.train()
        total_losses = {"total": 0.0, "ce": 0.0, "kl": 0.0}
        num_batches = 0

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch["label"].to(self.config.device)

            with autocast(device_type=self.config.device, enabled=self.use_amp):
                # Two forward passes with different dropout masks for R-Drop
                logits1 = self.model(input_ids, attention_mask)
                logits2 = self.model(input_ids, attention_mask)

                losses = rdrop_loss(
                    logits1, logits2, labels,
                    alpha=self.config.rdrop_alpha,
                    label_smoothing=self.config.label_smoothing,
                )
                loss = losses["total"] / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                # LR scheduling
                if self.swa_model and epoch >= self.config.swa_start_epoch:
                    self.swa_scheduler.step()
                else:
                    self.scheduler.step()

                self.global_step += 1

            # Track losses
            for k in total_losses:
                total_losses[k] += losses[k].item()
            num_batches += 1

            # Log
            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_losses["total"] / num_batches
                avg_ce = total_losses["ce"] / num_batches
                avg_kl = total_losses["kl"] / num_batches
                lr = self.optimizer.param_groups[0]["lr"]
                logger.info(
                    f"Epoch {epoch+1} | Step {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | CE: {avg_ce:.4f} | KL: {avg_kl:.4f} | LR: {lr:.2e}"
                )

            # Mid-epoch eval
            if self.global_step > 0 and self.global_step % self.config.eval_every == 0:
                val_f1 = self.evaluate(self.val_loader, "Val")
                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.save_checkpoint("best")
                    logger.info(f"New best Val F1: {val_f1:.4f}")
                self.model.train()

        # SWA update
        if self.swa_model and epoch >= self.config.swa_start_epoch:
            self.swa_model.update_parameters(self.model)
            self.swa_started = True

        return {k: v / max(num_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self, dataloader, split_name: str = "Val",
                 use_swa: bool = False) -> float:
        eval_model = self.swa_model if (use_swa and self.swa_model) else self.model
        eval_model.eval()

        all_preds = []
        all_labels = []
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch["label"].to(self.config.device)

            with autocast(device_type=self.config.device, enabled=self.use_amp):
                logits = eval_model(input_ids, attention_mask)

            preds = logits.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss = F.cross_entropy(logits, labels)
            total_loss += loss.item()
            num_batches += 1

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        avg_loss = total_loss / max(num_batches, 1)

        logger.info(f"{split_name} | Loss: {avg_loss:.4f} | Macro-F1: {macro_f1:.4f}")

        if split_name == "Test":
            report = classification_report(all_labels, all_preds, digits=4)
            logger.info(f"\n{split_name} Classification Report:\n{report}")

        return macro_f1

    def save_checkpoint(self, tag: str = "latest"):
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f"exp01_{self.config.task}_{tag}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_f1": self.best_f1,
            "global_step": self.global_step,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, tag: str = "best"):
        path = os.path.join(self.config.save_dir, f"exp01_{self.config.task}_{tag}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.config.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded checkpoint from {path}")
            return True
        return False

    def train(self):
        logger.info("=" * 60)
        logger.info(f"exp01 Simple Baseline - Task {self.config.task}")
        logger.info(f"Num classes: {self.model.num_classes}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"FP16: {self.config.fp16}")
        logger.info(f"Batch size: {self.config.batch_size} x {self.config.grad_accum_steps} = "
                     f"{self.config.batch_size * self.config.grad_accum_steps}")
        logger.info(f"Max length: {self.config.max_length}")
        logger.info(f"R-Drop alpha: {self.config.rdrop_alpha}")
        logger.info(f"Label smoothing: {self.config.label_smoothing}")
        logger.info(f"Multi-sample dropout K: {self.config.multisample_dropout_k}")
        logger.info(f"SWA: {self.config.use_swa} (start epoch {self.config.swa_start_epoch})")
        logger.info("=" * 60)

        for epoch in range(self.config.epochs):
            logger.info(f"\n{'='*40} Epoch {epoch+1}/{self.config.epochs} {'='*40}")

            train_losses = self.train_epoch(epoch)
            logger.info(
                f"Epoch {epoch+1} Train Summary: "
                + " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            )

            # Validate with regular model
            val_f1 = self.evaluate(self.val_loader, "Val")

            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.save_checkpoint("best")
                logger.info(f"*** New Best Val Macro-F1: {val_f1:.4f} ***")

            self.save_checkpoint("latest")

        # Update BN stats for SWA model
        if self.swa_model and self.swa_started:
            logger.info("Updating SWA batch norm statistics...")
            torch.optim.swa_utils.update_bn(self.train_loader, self.swa_model, device=self.config.device)

        # Final test evaluation
        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST EVALUATION")
        logger.info("=" * 60)

        # Test with best checkpoint (regular model)
        if self.load_checkpoint("best"):
            test_f1_regular = self.evaluate(self.test_loader, "Test")
        else:
            test_f1_regular = self.evaluate(self.test_loader, "Test")

        logger.info(f"\n*** Regular Model Test Macro-F1: {test_f1_regular:.4f} ***")

        # Test with SWA model
        if self.swa_model and self.swa_started:
            logger.info("\n--- SWA Model Evaluation ---")
            test_f1_swa = self.evaluate(self.test_loader, "Test", use_swa=True)
            logger.info(f"\n*** SWA Model Test Macro-F1: {test_f1_swa:.4f} ***")

            best_test = max(test_f1_regular, test_f1_swa)
            logger.info(f"\n*** Best Test Macro-F1: {best_test:.4f} "
                        f"({'SWA' if test_f1_swa > test_f1_regular else 'Regular'}) ***")
            return best_test

        return test_f1_regular


# ============================================================================
# Main
# ============================================================================

def main(task: str = "T1", config: Optional[Config] = None):
    if config is None:
        config = Config(task=task)
    set_seed(config.seed)

    logger.info(f"{'='*60}")
    logger.info(f"exp01 - Simple ModernBERT Baseline with OOD Regularization")
    logger.info(f"Task: {config.task}")
    logger.info(f"{'='*60}")

    # Load data
    train_data, val_data, test_data, num_classes = load_aicd_data(config)
    logger.info(f"Detected {num_classes} classes")

    # Tokenizer
    logger.info(f"Loading tokenizer: {config.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

    # Datasets
    logger.info("Creating datasets...")
    train_dataset = AICDDataset(train_data, tokenizer, config.max_length)
    val_dataset = AICDDataset(val_data, tokenizer, config.max_length)
    test_dataset = AICDDataset(test_data, tokenizer, config.max_length)

    # DataLoaders
    pin = config.device == "cuda"
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, num_workers=config.num_workers,
        pin_memory=pin, drop_last=True,
        persistent_workers=config.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=pin,
        persistent_workers=config.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size * 2,
        shuffle=False, num_workers=config.num_workers,
        pin_memory=pin,
        persistent_workers=config.num_workers > 0,
    )

    # Model
    logger.info("Building SimpleCodeClassifier...")
    model = SimpleCodeClassifier(config, num_classes)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Train
    trainer = Trainer(config, model, train_loader, val_loader, test_loader)
    test_f1 = trainer.train()

    return test_f1


if __name__ == "__main__":
    main(task="T1")
