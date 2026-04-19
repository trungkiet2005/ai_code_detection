"""
exp03: Frozen Encoder + Group DRO (Distributionally Robust Optimization)

Hypothesis: Fine-tuning the encoder causes it to overfit to training domain
artifacts (competitive programming templates, Python/Java/C++ idioms).
Freezing the encoder preserves its pre-trained knowledge while a lightweight
head learns domain-invariant classification.

Key insight from paper:
    - "Deep models exhibit a pronounced bias toward competitive programming
       artifacts: surface-level templates (#include, #define, class Solution,
       loop scaffolding) are overemphasized" (SHAP analysis, Section H)
    - "Domain shift emerges as the dominant source of error" (Table 5)
    - Train: Python, Java, C++ (algorithmic). Test: +6 languages + research/general

Strategy:
    1. FREEZE ModernBERT — don't let it adapt to training distribution
    2. Extract embeddings from multiple layers (not just [CLS]) — different
       layers capture different granularities
    3. Group DRO — optimize WORST-CASE loss across programming language groups
       This forces the head to work well on ALL languages, not just majority
    4. Temperature scaling for calibration
    5. Code augmentation: random variable renaming + comment removal to
       reduce reliance on surface patterns

Why Group DRO works here:
    - Standard ERM minimizes AVERAGE loss → model focuses on Python (largest group)
    - Group DRO minimizes WORST-GROUP loss → model must work on C++ too
    - This naturally improves OOD generalization to unseen languages

Target: Beat DeBERTa (34.13) and approach SVM TF-IDF (43.05) on T1.

Usage on Kaggle:
    1. Upload this file to a Kaggle notebook
    2. Run: !pip install datasets transformers accelerate
    3. Execute: python exp03_frozen_dro.py

Author: AICD-Bench Research
"""

import os
import re
import math
import random
import logging
import warnings
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

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
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    task: str = "T1"
    encoder_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 1024

    num_classes: Dict[str, int] = field(default_factory=lambda: {
        "T1": 2, "T2": 12, "T3": 4,
    })

    # Frozen encoder settings
    freeze_encoder: bool = True
    extract_layers: List[int] = field(default_factory=lambda: [-1, -4, -8])
    pool_strategy: str = "cls_mean"  # cls_mean: concat [CLS] + mean pooling

    # Head architecture
    head_hidden_dim: int = 512
    head_dropout: float = 0.3
    head_num_layers: int = 2

    # Group DRO
    dro_step_size: float = 0.01  # how fast to upweight worst group
    dro_alpha: float = 0.2  # DRO strength (0 = ERM, 1 = full DRO)

    # Code augmentation
    augment_prob: float = 0.5  # probability of augmenting each sample
    augment_rename_prob: float = 0.3  # probability of renaming variables
    augment_remove_comments_prob: float = 0.3  # probability of removing comments

    # Training
    epochs: int = 10  # more epochs since only head is trained
    batch_size: int = 32
    grad_accum_steps: int = 2
    lr: float = 1e-3  # higher LR since encoder is frozen
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.1

    # Data
    max_train_samples: int = 200_000
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000
    num_workers: int = 2

    # Misc
    seed: int = 42
    fp16: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir: str = "./exp03_checkpoints"
    log_every: int = 100
    eval_every: int = 2000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Code Augmentation (reduce surface-level shortcuts)
# ============================================================================

class CodeAugmenter:
    """Augment code to reduce reliance on surface patterns.

    Key augmentations:
    1. Variable renaming: replace identifiers with random names →
       prevents model from using naming conventions as shortcut
    2. Comment removal: strip all comments →
       prevents model from using comment style as shortcut
    3. Whitespace normalization: standardize indentation →
       prevents model from using whitespace patterns as shortcut
    """

    def __init__(self, rename_prob: float = 0.3, remove_comments_prob: float = 0.3):
        self.rename_prob = rename_prob
        self.remove_comments_prob = remove_comments_prob
        self._random_names = [
            "x", "y", "z", "a", "b", "c", "val", "tmp", "res", "item",
            "data", "obj", "elem", "node", "curr", "prev", "next_val",
            "count", "idx", "pos", "key", "value", "result", "output",
            "flag", "check", "temp", "var", "num", "arr", "lst",
        ]

    def augment(self, code: str) -> str:
        """Apply random augmentations."""
        if random.random() < self.rename_prob:
            code = self._rename_variables(code)
        if random.random() < self.remove_comments_prob:
            code = self._remove_comments(code)
        return code

    def _rename_variables(self, code: str) -> str:
        """Rename local variable-like identifiers randomly."""
        # Find short variable names (likely local variables)
        local_vars = set(re.findall(r'\b([a-z_]\w{0,5})\b', code))
        # Filter out keywords
        keywords = {
            'if', 'else', 'for', 'while', 'return', 'def', 'class', 'import',
            'from', 'in', 'is', 'not', 'and', 'or', 'try', 'except', 'with',
            'as', 'pass', 'break', 'continue', 'true', 'false', 'none', 'null',
            'int', 'str', 'float', 'bool', 'list', 'dict', 'set', 'tuple',
            'void', 'var', 'let', 'const', 'new', 'this', 'self',
        }
        local_vars -= keywords

        if not local_vars:
            return code

        # Rename a random subset
        to_rename = random.sample(list(local_vars), min(3, len(local_vars)))
        for old_name in to_rename:
            new_name = random.choice(self._random_names)
            # Only rename if it won't create conflicts
            if new_name not in local_vars:
                code = re.sub(r'\b' + re.escape(old_name) + r'\b', new_name, code)

        return code

    def _remove_comments(self, code: str) -> str:
        """Remove comments from code."""
        # Remove single-line comments
        code = re.sub(r'#.*$', '', code, flags=re.MULTILINE)
        code = re.sub(r'//.*$', '', code, flags=re.MULTILINE)
        # Remove multi-line comments (non-greedy)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)
        # Remove docstrings
        code = re.sub(r'""".*?"""', '""""""', code, flags=re.DOTALL)
        code = re.sub(r"'''.*?'''", "''''''", code, flags=re.DOTALL)
        return code


# ============================================================================
# Language Detection (for Group DRO)
# ============================================================================

def detect_language(code: str) -> int:
    """Heuristic language detection for group assignment.
    Returns group index (0-5) for DRO. Groups are coarser than exact languages.
    """
    code_lower = code[:500].lower()

    # C-family (C, C++, C#)
    if '#include' in code or 'namespace std' in code or 'printf(' in code:
        return 0
    if 'using System' in code or 'namespace ' in code_lower:
        return 0

    # Java
    if 'public class' in code or 'public static void main' in code:
        return 1

    # Python
    if 'def ' in code and ':' in code and ('import ' in code or 'self' in code):
        return 2

    # JavaScript/TypeScript
    if 'const ' in code and ('=>' in code or 'function' in code):
        return 3
    if 'var ' in code and 'function' in code:
        return 3

    # Go
    if 'func ' in code and 'package ' in code:
        return 4

    # PHP/Rust/Other
    if '<?php' in code or 'fn ' in code and '->' in code:
        return 5

    return 2  # default to Python group


# ============================================================================
# Dataset
# ============================================================================

class AICDDataset(Dataset):
    def __init__(self, data, tokenizer, max_length: int = 1024,
                 augmenter: CodeAugmenter = None, augment_prob: float = 0.0):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.augmenter = augmenter
        self.augment_prob = augment_prob

        # Pre-compute language groups for DRO
        self.groups = [detect_language(item["code"]) for item in data]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item["code"]
        label = item["label"]
        group = self.groups[idx]

        # Augment during training
        if self.augmenter and random.random() < self.augment_prob:
            code = self.augmenter.augment(code)

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
            "group": torch.tensor(group, dtype=torch.long),
        }


# ============================================================================
# Model
# ============================================================================

class MultiLayerPooler(nn.Module):
    """Extract and pool representations from multiple encoder layers.

    Different layers capture different granularities:
    - Early layers: syntax, token-level patterns
    - Middle layers: phrase-level, structural patterns
    - Late layers: semantic, task-relevant features

    Concatenating multiple layers gives a richer, more robust representation.
    """

    def __init__(self, hidden_size: int, extract_layers: List[int],
                 pool_strategy: str = "cls_mean"):
        super().__init__()
        self.extract_layers = extract_layers
        self.pool_strategy = pool_strategy

        # Output dimension depends on strategy
        if pool_strategy == "cls_mean":
            self.output_dim = hidden_size * len(extract_layers) * 2  # CLS + mean for each layer
        elif pool_strategy == "cls":
            self.output_dim = hidden_size * len(extract_layers)
        else:
            self.output_dim = hidden_size * len(extract_layers)

    def forward(self, hidden_states: Tuple[torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: tuple of (num_layers, B, seq_len, hidden)
            attention_mask: (B, seq_len)
        """
        representations = []

        for layer_idx in self.extract_layers:
            h = hidden_states[layer_idx]  # (B, seq_len, hidden)

            # CLS token
            cls_repr = h[:, 0]  # (B, hidden)
            representations.append(cls_repr)

            if self.pool_strategy == "cls_mean":
                # Mean pooling (masked)
                mask = attention_mask.unsqueeze(-1).float()  # (B, seq_len, 1)
                mean_repr = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
                representations.append(mean_repr)

        return torch.cat(representations, dim=-1)  # (B, output_dim)


class DROClassifier(nn.Module):
    """Frozen encoder + lightweight classification head."""

    def __init__(self, config: Config, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # Load encoder
        encoder_config = AutoConfig.from_pretrained(config.encoder_name)
        self.encoder = AutoModel.from_pretrained(
            config.encoder_name,
            config=encoder_config,
        )
        hidden_size = encoder_config.hidden_size

        # Freeze encoder
        if config.freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder frozen — only head will be trained")

        # Multi-layer pooler
        self.pooler = MultiLayerPooler(
            hidden_size, config.extract_layers, config.pool_strategy
        )

        # Classification head
        input_dim = self.pooler.output_dim
        layers = []
        for i in range(config.head_num_layers):
            out_dim = config.head_hidden_dim if i < config.head_num_layers - 1 else num_classes
            layers.extend([
                nn.Linear(input_dim, out_dim),
                *([] if i == config.head_num_layers - 1 else [
                    nn.GELU(),
                    nn.LayerNorm(out_dim),
                    nn.Dropout(config.head_dropout),
                ]),
            ])
            input_dim = out_dim

        self.head = nn.Sequential(*layers)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad() if self.config.freeze_encoder else torch.enable_grad():
            outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        pooled = self.pooler(outputs.hidden_states, attention_mask)
        logits = self.head(pooled)
        return logits


# ============================================================================
# Group DRO Loss
# ============================================================================

class GroupDROLoss(nn.Module):
    """Group Distributionally Robust Optimization.

    Instead of minimizing average loss, minimize worst-case group loss.
    Groups are programming language families detected heuristically.

    This forces the model to perform well on ALL languages, not just
    the majority (Python) in the training set.
    """

    def __init__(self, num_groups: int = 6, step_size: float = 0.01,
                 alpha: float = 0.2, label_smoothing: float = 0.1):
        super().__init__()
        self.num_groups = num_groups
        self.step_size = step_size
        self.alpha = alpha
        self.label_smoothing = label_smoothing

        # Group weights (updated during training)
        self.register_buffer('group_weights', torch.ones(num_groups) / num_groups)
        self.register_buffer('group_counts', torch.zeros(num_groups))

    def forward(self, logits: torch.Tensor, labels: torch.Tensor,
                groups: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            logits: (B, num_classes)
            labels: (B,)
            groups: (B,) group assignments
        """
        # Per-sample CE loss
        per_sample_loss = F.cross_entropy(
            logits, labels,
            reduction='none',
            label_smoothing=self.label_smoothing,
        )

        # Compute per-group average loss
        group_losses = torch.zeros(self.num_groups, device=logits.device)
        group_counts = torch.zeros(self.num_groups, device=logits.device)

        for g in range(self.num_groups):
            mask = (groups == g)
            if mask.any():
                group_losses[g] = per_sample_loss[mask].mean()
                group_counts[g] = mask.sum().float()

        # Update group weights (exponentiated gradient ascent)
        # Upweight groups with higher loss
        with torch.no_grad():
            valid_groups = group_counts > 0
            if valid_groups.any():
                self.group_weights[valid_groups] *= torch.exp(
                    self.step_size * group_losses[valid_groups]
                )
                self.group_weights /= self.group_weights.sum()  # normalize
                self.group_counts += group_counts

        # DRO loss: weighted combination of group losses
        # Interpolate between ERM (uniform weights) and DRO (adaptive weights)
        erm_loss = per_sample_loss.mean()
        dro_loss = (self.group_weights * group_losses).sum()
        total_loss = (1 - self.alpha) * erm_loss + self.alpha * dro_loss

        return {
            "total": total_loss,
            "erm": erm_loss,
            "dro": dro_loss,
            "worst_group_loss": group_losses[valid_groups].max() if valid_groups.any() else erm_loss,
            "group_weights": self.group_weights.clone(),
        }


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    def __init__(self, config: Config, model: DROClassifier,
                 train_loader, val_loader, test_loader):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        # Only optimize head parameters
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.lr,
            weight_decay=config.weight_decay,
        )

        total_steps = len(train_loader) * config.epochs // config.grad_accum_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        from transformers import get_cosine_schedule_with_warmup
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps,
        )

        self.use_amp = config.fp16 and config.device == "cuda"
        self.scaler = GradScaler(enabled=self.use_amp)

        self.dro_loss = GroupDROLoss(
            num_groups=6,
            step_size=config.dro_step_size,
            alpha=config.dro_alpha,
            label_smoothing=config.label_smoothing,
        ).to(config.device)

        self.best_f1 = 0.0
        self.global_step = 0

    def train_epoch(self, epoch: int):
        self.model.train()
        total_losses = defaultdict(float)
        num_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch["label"].to(self.config.device)
            groups = batch["group"].to(self.config.device)

            with autocast(device_type=self.config.device, enabled=self.use_amp):
                logits = self.model(input_ids, attention_mask)
                losses = self.dro_loss(logits, labels, groups)
                loss = losses["total"] / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad],
                    self.config.max_grad_norm,
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            for k in ["total", "erm", "dro", "worst_group_loss"]:
                total_losses[k] += losses[k].item()
            num_batches += 1

            if (batch_idx + 1) % self.config.log_every == 0:
                avg = {k: v / num_batches for k, v in total_losses.items()}
                lr = self.optimizer.param_groups[0]["lr"]
                gw = losses["group_weights"].cpu().numpy()
                logger.info(
                    f"Epoch {epoch+1} | Step {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg['total']:.4f} | ERM: {avg['erm']:.4f} | "
                    f"DRO: {avg['dro']:.4f} | Worst: {avg['worst_group_loss']:.4f} | "
                    f"LR: {lr:.2e} | GW: [{', '.join(f'{w:.2f}' for w in gw)}]"
                )

            if self.global_step > 0 and self.global_step % self.config.eval_every == 0:
                val_f1 = self.evaluate(self.val_loader, "Val")
                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.save_checkpoint("best")
                    logger.info(f"New best Val F1: {val_f1:.4f}")
                self.model.train()

        return {k: v / max(num_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self, dataloader, split_name: str = "Val") -> float:
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0
        num_batches = 0

        for batch in dataloader:
            input_ids = batch["input_ids"].to(self.config.device)
            attention_mask = batch["attention_mask"].to(self.config.device)
            labels = batch["label"].to(self.config.device)

            with autocast(device_type=self.config.device, enabled=self.use_amp):
                logits = self.model(input_ids, attention_mask)

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

    def save_checkpoint(self, tag: str):
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f"exp03_{self.config.task}_{tag}.pt")
        # Only save head (encoder is frozen/pretrained)
        head_state = {k: v for k, v in self.model.state_dict().items()
                      if 'encoder' not in k}
        torch.save({
            "head_state_dict": head_state,
            "best_f1": self.best_f1,
            "global_step": self.global_step,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, tag: str):
        path = os.path.join(self.config.save_dir, f"exp03_{self.config.task}_{tag}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.config.device, weights_only=True)
            self.model.load_state_dict(ckpt["head_state_dict"], strict=False)
            logger.info(f"Loaded checkpoint from {path}")
            return True
        return False

    def train(self):
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info("=" * 60)
        logger.info(f"exp03 Frozen Encoder + Group DRO - Task {self.config.task}")
        logger.info(f"Num classes: {self.model.num_classes}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"FP16: {self.config.fp16}")
        logger.info(f"Batch size: {self.config.batch_size} x {self.config.grad_accum_steps}")
        logger.info(f"DRO alpha: {self.config.dro_alpha}")
        logger.info(f"Augmentation prob: {self.config.augment_prob}")
        logger.info(f"Extract layers: {self.config.extract_layers}")
        logger.info("=" * 60)

        for epoch in range(self.config.epochs):
            logger.info(f"\n{'='*40} Epoch {epoch+1}/{self.config.epochs} {'='*40}")

            train_losses = self.train_epoch(epoch)
            logger.info(
                f"Epoch {epoch+1} Train Summary: "
                + " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            )

            val_f1 = self.evaluate(self.val_loader, "Val")
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.save_checkpoint("best")
                logger.info(f"*** New Best Val Macro-F1: {val_f1:.4f} ***")

            self.save_checkpoint("latest")

        # Final test
        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST EVALUATION")
        logger.info("=" * 60)

        if self.load_checkpoint("best"):
            test_f1 = self.evaluate(self.test_loader, "Test")
        else:
            test_f1 = self.evaluate(self.test_loader, "Test")

        logger.info(f"\n*** Final Test Macro-F1: {test_f1:.4f} ***")
        return test_f1


# ============================================================================
# Main
# ============================================================================

def load_aicd_data(config: Config):
    logger.info(f"Loading AICD-Bench task {config.task}...")
    ds = load_dataset("AICD-bench/AICD-Bench", name=config.task)

    train_data = ds["train"]
    val_data = ds["validation"]
    test_data = ds["test"]

    if len(train_data) > config.max_train_samples:
        indices = random.sample(range(len(train_data)), config.max_train_samples)
        train_data = train_data.select(indices)
    if len(val_data) > config.max_val_samples:
        indices = random.sample(range(len(val_data)), config.max_val_samples)
        val_data = val_data.select(indices)
    if len(test_data) > config.max_test_samples:
        indices = random.sample(range(len(test_data)), config.max_test_samples)
        test_data = test_data.select(indices)

    logger.info(f"Data sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
    num_classes = len(set(train_data["label"]))
    logger.info(f"Number of classes: {num_classes}")
    return train_data, val_data, test_data, num_classes


def main(task: str = "T1", config: Optional[Config] = None):
    if config is None:
        config = Config(task=task)
    set_seed(config.seed)

    logger.info("=" * 60)
    logger.info(f"exp03 - Frozen Encoder + Group DRO")
    logger.info(f"Task: {config.task}")
    logger.info("=" * 60)

    train_data, val_data, test_data, num_classes = load_aicd_data(config)

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

    augmenter = CodeAugmenter(
        rename_prob=config.augment_rename_prob,
        remove_comments_prob=config.augment_remove_comments_prob,
    )

    train_dataset = AICDDataset(
        train_data, tokenizer, config.max_length,
        augmenter=augmenter, augment_prob=config.augment_prob,
    )
    val_dataset = AICDDataset(val_data, tokenizer, config.max_length)
    test_dataset = AICDDataset(test_data, tokenizer, config.max_length)

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

    model = DROClassifier(config, num_classes)

    trainer = Trainer(config, model, train_loader, val_loader, test_loader)
    test_f1 = trainer.train()

    return test_f1


if __name__ == "__main__":
    main(task="T1")
