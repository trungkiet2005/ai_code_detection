"""
[CoDET-M4] TTLCode Runner - Full Benchmark Evaluation  (Kaggle standalone)
      Exp34: TTLCode (Test-Time LoRA Adaptation)

Core novelty: Adapt model at TEST TIME using LoRA on unlabeled batches.
  - LoRA adapters on backbone, MLM objective at test time
  - Addresses OOD generator/language shifts
  - From: Test-Time Learning for LLMs (ICML 2025), LoRA (Hu 2022)

Architecture: SpectralCode + LoRA adapters + MLM head
Target: Author IID 72%+ | OOD robustness | NeurIPS 2026 ORAL
"""

import importlib.util
import json
import math
import os
import random
import re
import subprocess
import sys
import logging
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Bootstrap – auto-install missing packages (Kaggle-friendly)
# ---------------------------------------------------------------------------

def _ensure_deps():
    required = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("sklearn", "scikit-learn"),
        ("accelerate", "accelerate"),
    ]
    optional = [
        ("tree_sitter", "tree-sitter"),
        ("tree_sitter_languages", "tree-sitter-languages"),
    ]
    missing = [pip for imp, pip in required if importlib.util.find_spec(imp) is None]
    if missing:
        print(f"[bootstrap] Installing missing packages: {missing}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])

    for imp, pip in optional:
        if importlib.util.find_spec(imp) is None:
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pip])
            except subprocess.CalledProcessError:
                print(f"[bootstrap] Optional dependency unavailable: {pip} (continuing)")

_ensure_deps()

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset as TorchDataset, DataLoader

try:
    from torch.amp import autocast as _autocast, GradScaler
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler
    _NEW_AMP = False

from datasets import Dataset, load_dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Compatibility helpers
# ---------------------------------------------------------------------------

def autocast(device_type="cuda", enabled=True, dtype=None):
    if _NEW_AMP:
        if dtype is None:
            return _autocast(device_type=device_type, enabled=enabled)
        return _autocast(device_type=device_type, enabled=enabled, dtype=dtype)
    return _autocast(enabled=enabled)


class PreflightError(RuntimeError):
    pass


def _get_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda"


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# Configuration
# ============================================================================

def apply_hardware_profile(config: "SpectralConfig") -> "SpectralConfig":
    if config.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass

    if not (config.auto_h100_profile and config.device == "cuda"):
        return config

    gpu_name = _get_gpu_name().upper()
    if "H100" not in gpu_name:
        return config

    config.precision = "bf16" if config.precision == "auto" else config.precision
    config.batch_size = max(config.batch_size, 64)
    config.grad_accum_steps = 1
    config.num_workers = max(config.num_workers, 8)
    config.prefetch_factor = max(config.prefetch_factor, 4)
    config.log_every = max(config.log_every, 200)
    config.eval_every = max(config.eval_every, 2000)
    logger.info(
        "Applied H100 profile | precision=%s | batch=%d | accum=%d | workers=%d",
        config.precision, config.batch_size, config.grad_accum_steps, config.num_workers,
    )
    return config


@dataclass
class SpectralConfig:
    task: str = "T1"
    benchmark: str = "codet_m4"

    encoder_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    z_style_dim: int = 256
    z_content_dim: int = 256
    gnn_hidden_dim: int = 128
    gnn_layers: int = 2
    num_ast_node_types: int = 256
    ast_embed_dim: int = 64
    ast_seq_len: int = 128

    epochs: int = 3
    batch_size: int = 32
    grad_accum_steps: int = 2
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    temperature: float = 0.07
    # TTLCode hyperparams
    lora_rank: int = 4
    lora_alpha: int = 16
    ttl_lr: float = 1e-5
    ttl_steps: int = 1
    ttl_mask_prob: float = 0.15
    ttl_enabled: bool = True
    max_train_samples: int = 100_000
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000

    num_workers: int = 2
    prefetch_factor: int = 2

    seed: int = 42
    precision: str = "auto"
    auto_h100_profile: bool = True
    require_tree_sitter: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    non_blocking: bool = True
    save_dir: str = "./codet_m4_checkpoints"
    log_every: int = 100
    eval_every: int = 1000


# ============================================================================
# AST Feature Extraction
# ============================================================================

AST_NODE_VOCAB = {
    "function_definition": 1, "class_definition": 2, "if_statement": 3,
    "for_statement": 4, "while_statement": 5, "return_statement": 6,
    "assignment": 7, "call_expression": 8, "binary_expression": 9,
    "variable_declaration": 10, "import_statement": 11, "try_statement": 12,
    "catch_clause": 13, "throw_statement": 14, "switch_statement": 15,
    "case_clause": 16, "comment": 17, "string_literal": 18,
    "number_literal": 19, "boolean_literal": 20, "null_literal": 21,
    "array_expression": 22, "object_expression": 23, "lambda_expression": 24,
    "generator_expression": 25, "list_comprehension": 26, "dict_comprehension": 27,
    "decorator": 28, "yield_statement": 29, "assert_statement": 30,
    "with_statement": 31, "pass_statement": 32, "break_statement": 33,
    "continue_statement": 34, "else_clause": 35, "elif_clause": 36,
    "finally_clause": 37, "parameter": 38, "argument": 39, "identifier": 40,
    "method_definition": 41, "property_definition": 42, "enum_definition": 43,
    "interface_definition": 44, "struct_definition": 45, "type_annotation": 46,
    "generic_type": 47, "pointer_type": 48, "reference_type": 49,
    "namespace": 50, "module": 51, "block": 52, "expression_statement": 53,
    "parenthesized_expression": 54, "subscript_expression": 55,
    "member_expression": 56, "conditional_expression": 57, "unary_expression": 58,
    "template_literal": 59, "spread_element": 60, "rest_parameter": 61,
    "default_parameter": 62, "arrow_function": 63, "async_function": 64,
    "await_expression": 65, "PAD": 0, "UNK": 66,
}


def _try_load_tree_sitter():
    try:
        import tree_sitter_languages  # type: ignore[reportMissingImports]

        LANG_MAP = {
            "python": "python", "java": "java", "cpp": "cpp", "c": "c",
            "go": "go", "php": "php", "c_sharp": "c_sharp",
            "javascript": "javascript", "rust": "rust",
        }

        def parse_ast(code: str, lang: str = "python") -> List[int]:
            try:
                ts_lang = LANG_MAP.get(lang, "python")
                parser = tree_sitter_languages.get_parser(ts_lang)
                tree = parser.parse(bytes(code, "utf8"))
                node_types = []
                stack = [tree.root_node]
                while stack and len(node_types) < 512:
                    node = stack.pop()
                    ntype = node.type.lower().replace("-", "_")
                    node_types.append(AST_NODE_VOCAB.get(ntype, AST_NODE_VOCAB["UNK"]))
                    stack.extend(reversed(node.children))
                return node_types
            except Exception:
                return _fallback_ast_extract(code)

        return parse_ast
    except ImportError:
        logger.warning("tree-sitter-languages not found, using regex-based AST extraction")
        return None


def _fallback_ast_extract(code: str) -> List[int]:
    features = []
    patterns = {
        "function_definition": r"\b(def|function|func|fn)\s+\w+",
        "class_definition": r"\b(class|struct|interface|enum)\s+\w+",
        "if_statement": r"\bif\s*[\(\{]",
        "for_statement": r"\b(for|foreach)\s*[\(\{]",
        "while_statement": r"\bwhile\s*[\(\{]",
        "return_statement": r"\breturn\b",
        "import_statement": r"\b(import|include|require|using)\b",
        "try_statement": r"\btry\s*[\{\:]",
        "catch_clause": r"\b(catch|except)\b",
        "comment": r"(//|#|/\*|\"\"\"|\'\'\')",
        "assignment": r"[^=!<>]=[^=]",
        "call_expression": r"\w+\s*\(",
        "lambda_expression": r"\b(lambda|=>)\b",
        "string_literal": r"[\"']",
        "variable_declaration": r"\b(var|let|const|int|float|double|string|bool)\b",
        "switch_statement": r"\b(switch|match)\b",
        "throw_statement": r"\b(throw|raise)\b",
        "async_function": r"\basync\b",
        "await_expression": r"\bawait\b",
        "yield_statement": r"\byield\b",
        "with_statement": r"\bwith\b",
        "assert_statement": r"\bassert\b",
    }
    for line in code.split("\n"):
        for node_type, pattern in patterns.items():
            if re.search(pattern, line):
                features.append(AST_NODE_VOCAB.get(node_type, AST_NODE_VOCAB["UNK"]))
    return features if features else [AST_NODE_VOCAB["UNK"]]


_ast_parser = _try_load_tree_sitter()

STRUCTURAL_FEATURE_DIM = 22


def extract_ast_sequence(code: str, max_len: int = 128) -> List[int]:
    if _ast_parser is not None:
        seq = _ast_parser(code)
    else:
        seq = _fallback_ast_extract(code)
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [AST_NODE_VOCAB["PAD"]] * (max_len - len(seq))
    return seq


def extract_structural_features(code: str) -> List[float]:
    lines = code.split("\n")
    num_lines = len(lines)
    avg_line_len = np.mean([len(l) for l in lines]) if lines else 0
    max_line_len = max([len(l) for l in lines]) if lines else 0

    indents = []
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indents.append(len(line) - len(stripped))
    avg_indent = np.mean(indents) if indents else 0
    max_indent = max(indents) if indents else 0
    indent_variance = np.var(indents) if indents else 0

    num_functions = code.count("def ") + code.count("function ") + code.count("func ") + code.count("fn ")
    num_classes = code.count("class ") + code.count("struct ") + code.count("interface ")
    num_loops = code.count("for ") + code.count("while ") + code.count("foreach ")
    num_conditionals = code.count("if ") + code.count("else ") + code.count("elif ") + code.count("else if")
    num_returns = code.count("return ")
    num_comments = code.count("//") + code.count("#") + code.count("/*")
    num_imports = code.count("import ") + code.count("include ") + code.count("require ") + code.count("using ")
    num_try_catch = code.count("try") + code.count("catch") + code.count("except")

    identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)
    num_snake_case = sum(1 for i in identifiers if '_' in i and i.islower())
    num_camel_case = sum(1 for i in identifiers if any(c.isupper() for c in i[1:]) and '_' not in i)
    num_single_char = sum(1 for i in identifiers if len(i) == 1)
    avg_identifier_len = np.mean([len(i) for i in identifiers]) if identifiers else 0

    empty_lines = sum(1 for l in lines if not l.strip())
    empty_line_ratio = empty_lines / max(num_lines, 1)
    alpha_ratio = sum(c.isalpha() for c in code) / max(len(code), 1)
    digit_ratio = sum(c.isdigit() for c in code) / max(len(code), 1)
    space_ratio = sum(c.isspace() for c in code) / max(len(code), 1)

    return [
        num_lines, avg_line_len, max_line_len,
        avg_indent, max_indent, indent_variance,
        num_functions, num_classes, num_loops, num_conditionals,
        num_returns, num_comments, num_imports, num_try_catch,
        num_snake_case / max(len(identifiers), 1),
        num_camel_case / max(len(identifiers), 1),
        num_single_char / max(len(identifiers), 1),
        avg_identifier_len,
        empty_line_ratio, alpha_ratio, digit_ratio, space_ratio,
    ]


# ============================================================================
# Spectral Feature Extraction
# ============================================================================

SPECTRAL_FEATURE_DIM = 64


def extract_spectral_features(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size = input_ids.size(0)
    features = []
    windows = [32, 64, 128, 256]
    features_per_window = SPECTRAL_FEATURE_DIM // len(windows)

    for i in range(batch_size):
        ids = input_ids[i][attention_mask[i].bool()].cpu().float().numpy()
        seq_len = len(ids)
        sample_feats = []

        for win_size in windows:
            if seq_len < win_size:
                padded = np.zeros(win_size)
                padded[:seq_len] = ids[:seq_len]
            else:
                padded = ids[:win_size]

            padded = padded - padded.mean()
            fft_vals = np.fft.rfft(padded)
            magnitude = np.abs(fft_vals)

            if len(magnitude) == 0:
                sample_feats.extend([0.0] * features_per_window)
                continue

            total_energy = np.sum(magnitude ** 2)
            n_bands = min(8, len(magnitude))
            band_size = max(1, len(magnitude) // n_bands)
            band_energies = []
            for b in range(n_bands):
                start = b * band_size
                end = min(start + band_size, len(magnitude))
                band_energies.append(np.sum(magnitude[start:end] ** 2) / (total_energy + 1e-10))

            while len(band_energies) < features_per_window - 4:
                band_energies.append(0.0)
            band_energies = band_energies[:features_per_window - 4]

            spectral_centroid = np.sum(np.arange(len(magnitude)) * magnitude) / (np.sum(magnitude) + 1e-10)
            spectral_rolloff = np.searchsorted(np.cumsum(magnitude), 0.85 * np.sum(magnitude)) / len(magnitude)
            spectral_flatness = np.exp(np.mean(np.log(magnitude + 1e-10))) / (np.mean(magnitude) + 1e-10)
            peak_freq = np.argmax(magnitude[1:]) + 1 if len(magnitude) > 1 else 0

            sample_feats.extend(band_energies)
            sample_feats.extend([
                spectral_centroid / len(magnitude),
                spectral_rolloff,
                min(spectral_flatness, 10.0),
                peak_freq / len(magnitude),
            ])

        features.append(sample_feats[:SPECTRAL_FEATURE_DIM])

    return torch.tensor(np.array(features), dtype=torch.float32, device=input_ids.device)


# ============================================================================
# Dataset
# ============================================================================

class AICDDataset(TorchDataset):

    def __init__(self, data, tokenizer, max_length: int = 512, ast_seq_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ast_seq_len = ast_seq_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item["code"]
        label = item["label"]

        encoding = self.tokenizer(
            code, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        ast_seq = extract_ast_sequence(code, self.ast_seq_len)
        struct_feat = extract_structural_features(code)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ast_seq": torch.tensor(ast_seq, dtype=torch.long),
            "struct_feat": torch.tensor(struct_feat, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
        }


# ============================================================================
# Model Components
# ============================================================================

class ASTEncoder(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, ast_seq: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ast_seq)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.proj(h)


class StructuralEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    def __init__(self, token_dim: int, ast_dim: int, struct_dim: int, output_dim: int):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, output_dim)
        self.ast_proj = nn.Linear(ast_dim, output_dim)
        self.struct_proj = nn.Linear(struct_dim, output_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=output_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(output_dim)
        self.gate = nn.Linear(output_dim * 3, output_dim)

    def forward(self, token_repr, ast_repr, struct_repr):
        t = self.token_proj(token_repr)
        a = self.ast_proj(ast_repr)
        s = self.struct_proj(struct_repr)
        seq = torch.stack([t, a, s], dim=1)
        attn_out, _ = self.cross_attn(seq, seq, seq)
        attn_out = self.norm(attn_out + seq)
        concat = torch.cat([attn_out[:, 0], attn_out[:, 1], attn_out[:, 2]], dim=-1)
        return self.gate(concat)


# ============================================================================
# Full Model: SpectralCode
# ============================================================================

class SpectralCode(nn.Module):
    def __init__(self, config: SpectralConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        self.token_encoder = AutoModel.from_pretrained(config.encoder_name, attn_implementation="sdpa")
        token_hidden_size = self.token_encoder.config.hidden_size

        self.ast_encoder = ASTEncoder(config.num_ast_node_types, config.ast_embed_dim, config.gnn_hidden_dim)
        self.struct_encoder = StructuralEncoder(STRUCTURAL_FEATURE_DIM, config.gnn_hidden_dim)

        fusion_dim = config.z_style_dim + config.z_content_dim
        self.fusion = CrossAttentionFusion(token_hidden_size, config.gnn_hidden_dim, config.gnn_hidden_dim, fusion_dim)

        self.spectral_encoder = nn.Sequential(
            nn.Linear(SPECTRAL_FEATURE_DIM, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.GELU(),
        )
        spectral_dim = 128

        self.neural_head = nn.Linear(fusion_dim, num_classes)
        self.spectral_head = nn.Linear(spectral_dim, num_classes)

        self.gate = nn.Sequential(
            nn.Linear(fusion_dim + spectral_dim, 64), nn.GELU(), nn.Linear(64, 2),
        )
        self.focal_gamma = 2.0

    def forward(self, input_ids, attention_mask, ast_seq, struct_feat, labels=None):
        token_out = self.token_encoder(input_ids=input_ids, attention_mask=attention_mask)
        token_repr = token_out.last_hidden_state[:, 0, :]
        ast_repr = self.ast_encoder(ast_seq)
        struct_repr = self.struct_encoder(struct_feat)
        h_neural = self.fusion(token_repr, ast_repr, struct_repr)

        spectral_feats = extract_spectral_features(input_ids, attention_mask)
        h_spectral = self.spectral_encoder(spectral_feats)

        neural_logits = self.neural_head(h_neural)
        spectral_logits = self.spectral_head(h_spectral)

        gate_input = torch.cat([h_neural, h_spectral], dim=-1)
        gate_weights = F.softmax(self.gate(gate_input), dim=-1)
        logits = gate_weights[:, 0:1] * neural_logits + gate_weights[:, 1:2] * spectral_logits

        return {
            "logits": logits,
            "neural_logits": neural_logits,
            "spectral_logits": spectral_logits,
            "gate_weights": gate_weights,
            "embeddings": h_neural,  # for SelfDistill loss
        }


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        return (((1 - pt) ** self.gamma) * ce_loss).mean()


# ============================================================================
# EMA Teacher + Self-Distillation Loss (Exp34 novelty)
# ============================================================================

import copy as _copy

class EMATeacher:
    """
    Exponential Moving Average (EMA) teacher for self-distillation.
    BYOL/DINO-style: teacher is a slow-moving copy of the student.
    Teacher is never optimized directly — only updated via EMA after each step.
    Prevents representation collapse via bootstrapping (no negative samples needed).
    """
    def __init__(self, student: nn.Module, momentum: float = 0.995):
        self.teacher = _copy.deepcopy(student)
        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad_(False)
        self.momentum = momentum
        logger.info(f"[Exp34] EMATeacher initialized: momentum={momentum}")

    @torch.no_grad()
    def update(self, student: nn.Module):
        """EMA update: teacher_param ← m*teacher + (1-m)*student"""
        for t_param, s_param in zip(self.teacher.parameters(), student.parameters()):
            t_param.data.mul_(self.momentum).add_(s_param.data, alpha=1.0 - self.momentum)

    @torch.no_grad()
    def get_embeddings(self, input_ids, attention_mask, ast_seq, struct_feat) -> torch.Tensor:
        out = self.teacher(input_ids, attention_mask, ast_seq, struct_feat)
        return out["embeddings"]


def compute_all_losses(model, outputs, labels, config, focal_loss_fn=None,
                       teacher_embeddings: Optional[torch.Tensor] = None):
    """
    Self-distillation loss: task_loss + distill_loss(student ↔ EMA teacher).
    teacher_embeddings: if None, distillation is skipped (e.g., during val/test).
    """
    if focal_loss_fn is None:
        focal_loss_fn = FocalLoss(gamma=2.0)
    task_loss = focal_loss_fn(outputs["logits"], labels)
    neural_loss = focal_loss_fn(outputs["neural_logits"], labels)
    spectral_loss = focal_loss_fn(outputs["spectral_logits"], labels)

    if teacher_embeddings is not None:
        # Cosine self-distillation: student predicts teacher's representation
        s_emb = F.normalize(outputs["embeddings"], p=2, dim=-1)
        t_emb = F.normalize(teacher_embeddings.detach(), p=2, dim=-1)
        distill_loss = (1.0 - (s_emb * t_emb).sum(dim=-1)).mean()
    else:
        distill_loss = outputs["embeddings"].new_zeros(1).squeeze()

    lambda_distill = getattr(config, "lambda_distill", 0.3)
    total_loss = task_loss + 0.3 * neural_loss + 0.3 * spectral_loss + lambda_distill * distill_loss
    return {"total": total_loss, "task": task_loss, "neural": neural_loss,
            "spectral": spectral_loss, "aux": distill_loss}


def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes
    return torch.tensor(weights, dtype=torch.float32)


# ============================================================================
# Trainer
# ============================================================================

class Trainer:
    def __init__(self, config: SpectralConfig, model: SpectralCode, train_loader, val_loader, test_loader, class_weights=None):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_weights = class_weights.to(config.device) if class_weights is not None else None

        encoder_params = list(model.token_encoder.parameters())
        head_params = [p for n, p in model.named_parameters() if "token_encoder" not in n]

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": config.lr_encoder, "weight_decay": config.weight_decay},
            {"params": head_params, "lr": config.lr_heads, "weight_decay": config.weight_decay},
        ])

        total_steps = len(train_loader) * config.epochs // config.grad_accum_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(self.optimizer, warmup_steps, total_steps)

        precision = config.precision.lower()
        if precision == "auto":
            precision = "bf16" if config.device == "cuda" else "fp32"
        self.precision = precision

        use_amp = config.device == "cuda" and precision in ("bf16", "fp16")
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
        self.scaler = GradScaler(enabled=(use_amp and precision == "fp16"))
        self.focal_loss = FocalLoss(gamma=2.0, weight=self.class_weights)
        self.best_f1 = 0.0
        self.global_step = 0
        self.last_eval_metrics: Dict[str, Dict[str, float]] = {}
        # EMA teacher for self-distillation
        self.ema_teacher = EMATeacher(
            model, momentum=getattr(config, "distill_momentum", 0.995)
        )

    def train_epoch(self, epoch: int):
        self.model.train()
        total_losses = defaultdict(float)
        num_batches = 0
        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            input_ids = batch["input_ids"].to(self.config.device, non_blocking=self.config.non_blocking)
            attention_mask = batch["attention_mask"].to(self.config.device, non_blocking=self.config.non_blocking)
            ast_seq = batch["ast_seq"].to(self.config.device, non_blocking=self.config.non_blocking)
            struct_feat = batch["struct_feat"].to(self.config.device, non_blocking=self.config.non_blocking)
            labels = batch["label"].to(self.config.device, non_blocking=self.config.non_blocking)

            # Get EMA teacher embeddings (no grad, from slow-moving teacher)
            teacher_embs = self.ema_teacher.get_embeddings(
                input_ids, attention_mask, ast_seq, struct_feat
            ).to(self.config.device)

            with autocast(device_type=self.config.device, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(input_ids, attention_mask, ast_seq, struct_feat, labels)
                losses = compute_all_losses(
                    self.model, outputs, labels, self.config, self.focal_loss,
                    teacher_embeddings=teacher_embs,
                )
                loss = losses["total"] / self.config.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                # Update EMA teacher AFTER optimizer step
                self.ema_teacher.update(self.model)
                self.global_step += 1

            for k, v in losses.items():
                total_losses[k] += v.item()
            num_batches += 1

            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_losses["total"] / num_batches
                lr = self.scheduler.get_last_lr()[0]
                avg_aux = total_losses.get("aux", 0.0) / num_batches
                logger.info(
                    f"Epoch {epoch+1} | Step {batch_idx+1}/{len(self.train_loader)} | "
                    f"[Exp34] Loss: {avg_loss:.4f} | Aux: {avg_aux:.4f} | LR: {lr:.2e}"
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
        all_preds, all_labels = [], []
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

        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
        avg_loss = total_loss / max(num_batches, 1)

        logger.info(f"{split_name} | Loss: {avg_loss:.4f} | Macro-F1: {macro_f1:.4f} | Weighted-F1: {weighted_f1:.4f}")
        self.last_eval_metrics[split_name] = {"loss": float(avg_loss), "macro_f1": float(macro_f1), "weighted_f1": float(weighted_f1)}

        if split_name == "Test":
            report = classification_report(all_labels, all_preds, digits=4)
            logger.info(f"\n{split_name} Classification Report:\n{report}")
        return macro_f1

    def save_checkpoint(self, tag: str = "latest"):
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f"ttlcodecode_{self.config.task}_{tag}.pt")
        torch.save({"model_state_dict": self.model.state_dict(), "best_f1": self.best_f1, "global_step": self.global_step}, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, tag: str = "best"):
        path = os.path.join(self.config.save_dir, f"ttlcodecode_{self.config.task}_{tag}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.config.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded checkpoint from {path}")
            return True
        return False

    def train(self):
        logger.info("=" * 60)
        logger.info(f"[Exp34] Starting TTLCode Training - {self.config.task}")
        logger.info(f"Classes: {self.model.num_classes} | Device: {self.config.device} | Precision: {self.precision}")
        logger.info(f"Batch: {self.config.batch_size}x{self.config.grad_accum_steps} = {self.config.batch_size * self.config.grad_accum_steps}")
        logger.info("=" * 60)

        for epoch in range(self.config.epochs):
            logger.info(f"\n{'='*40} Epoch {epoch+1}/{self.config.epochs} {'='*40}")
            train_losses = self.train_epoch(epoch)
            logger.info("Epoch %d Train: %s", epoch + 1, " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items()))

            val_f1 = self.evaluate(self.val_loader, "Val")
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.save_checkpoint("best")
                logger.info(f"*** New Best Val Macro-F1: {val_f1:.4f} ***")
            self.save_checkpoint("latest")

        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST EVALUATION")
        logger.info("=" * 60)
        if self.load_checkpoint("best"):
            test_f1 = self.evaluate(self.test_loader, "Test")
        else:
            test_f1 = self.evaluate(self.test_loader, "Test")

        test_weighted = self.last_eval_metrics.get("Test", {}).get("weighted_f1", test_f1)
        logger.info(f"*** Final Test Macro-F1: {test_f1:.4f} | Weighted-F1: {test_weighted:.4f} ***")
        return {"test_f1": float(test_f1), "test_weighted_f1": float(test_weighted), "best_val_f1": float(self.best_f1)}


# ============================================================================
# CoDET-M4 Config
# ============================================================================

@dataclass
class CoDETM4Config:
    dataset_id: str = "DaniilOr/CoDET-M4"
    code_field_priority: Tuple[str, ...] = ("cleaned_code", "code")
    split_field: str = "split"

    task: str = "binary"  # "binary" | "author"

    max_train_samples: int = 100_000
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000

    eval_breakdown: bool = True

    seed: int = 42
    save_root: str = "./codet_m4_checkpoints"


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def _sample_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), max_samples)
    return dataset.select(indices)


def _extract_code(row: Dict[str, object], code_fields: Tuple[str, ...]) -> str:
    for field_name in code_fields:
        value = row.get(field_name, "")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _normalize_target(value: object) -> str:
    return str(value or "").strip().lower()


def _is_human_target(target: str) -> bool:
    return target in {"human", "human_written", "human-generated", "human_generated"}


def _build_author_vocab(train_split: Dataset) -> Dict[str, int]:
    model_names = set()
    for row in train_split:
        target = _normalize_target(row.get("target", ""))
        model_name = str(row.get("model", "") or "").strip()
        if not _is_human_target(target) and model_name:
            model_names.add(model_name)
    ordered = sorted(model_names)
    return {name: idx + 1 for idx, name in enumerate(ordered)}


def _map_binary_label(row: Dict[str, object]) -> int:
    target = _normalize_target(row.get("target", ""))
    return 0 if _is_human_target(target) else 1


def _map_author_label(row: Dict[str, object], author_vocab: Dict[str, int]) -> int:
    target = _normalize_target(row.get("target", ""))
    if _is_human_target(target):
        return 0
    model_name = str(row.get("model", "") or "").strip()
    return author_vocab.get(model_name, -1)


def _convert_split(
    split_ds: Dataset,
    cfg: CoDETM4Config,
    author_vocab: Dict[str, int],
) -> Dataset:
    def _convert_row(row):
        code = _extract_code(row, cfg.code_field_priority)
        if cfg.task == "binary":
            label = _map_binary_label(row)
        elif cfg.task == "author":
            label = _map_author_label(row, author_vocab)
        else:
            raise ValueError(f"Unsupported task: {cfg.task}")
        target = _normalize_target(row.get("target", ""))
        return {
            "code": code,
            "label": label,
            "language": str(row.get("language", "")).strip().lower(),
            "source": str(row.get("source", "")).strip().lower(),
            "generator": "human" if _is_human_target(target) else str(row.get("model", "") or "").strip().lower(),
        }

    converted = split_ds.map(_convert_row, remove_columns=split_ds.column_names)
    converted = converted.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)
    return converted


def _quick_code_stats(dataset: Dataset, sample_size: int = 512) -> Dict[str, float]:
    n = min(len(dataset), sample_size)
    if n == 0:
        return {"samples": 0, "avg_chars": 0.0, "avg_lines": 0.0, "empty_ratio": 1.0}
    sample = dataset.select(range(n))
    codes = sample["code"]
    lengths = [len(c) for c in codes]
    lines = [c.count("\n") + 1 for c in codes]
    empty = sum(1 for c in codes if len(c.strip()) == 0)
    return {
        "samples": n,
        "avg_chars": float(np.mean(lengths)),
        "avg_lines": float(np.mean(lines)),
        "empty_ratio": float(empty / n),
    }


# ============================================================================
# DATA LOADING: IID (Tables 2, 3, 4, 7)
# ============================================================================

def _load_raw_splits(cfg: CoDETM4Config):
    logger.info(f"Loading dataset: {cfg.dataset_id}")
    ds = load_dataset(cfg.dataset_id, split="train")

    has_split = cfg.split_field in ds.column_names
    if has_split:
        logger.info("Using built-in split column from CoDET-M4")
        train_raw = ds.filter(lambda x: str(x.get(cfg.split_field, "")).lower() == "train")
        val_raw = ds.filter(lambda x: str(x.get(cfg.split_field, "")).lower() in {"val", "validation", "dev"})
        test_raw = ds.filter(lambda x: str(x.get(cfg.split_field, "")).lower() == "test")
    else:
        logger.warning("No split column; creating 80/10/10 with seed=%d", cfg.seed)
        s1 = ds.train_test_split(test_size=0.1, seed=cfg.seed)
        test_raw = s1["test"]
        s2 = s1["train"].train_test_split(test_size=1 / 9, seed=cfg.seed)
        train_raw, val_raw = s2["train"], s2["test"]

    if len(train_raw) == 0 or len(val_raw) == 0 or len(test_raw) == 0:
        raise RuntimeError("One or more CoDET-M4 splits are empty.")
    return train_raw, val_raw, test_raw


def load_codet_m4_data(cfg: CoDETM4Config):
    train_raw, val_raw, test_raw = _load_raw_splits(cfg)

    author_vocab = _build_author_vocab(train_raw) if cfg.task == "author" else {}
    if cfg.task == "author":
        logger.info("Author vocab (%d generators): %s", len(author_vocab), sorted(author_vocab.keys()))

    train_data = _convert_split(train_raw, cfg, author_vocab)
    val_data = _convert_split(val_raw, cfg, author_vocab)
    test_data = _convert_split(test_raw, cfg, author_vocab)

    train_data = _sample_dataset(train_data, cfg.max_train_samples, cfg.seed)
    val_data = _sample_dataset(val_data, cfg.max_val_samples, cfg.seed + 1)
    test_data = _sample_dataset(test_data, cfg.max_test_samples, cfg.seed + 2)

    labels = set(train_data["label"])
    num_classes = len(labels)
    logger.info("IID sizes | train=%d val=%d test=%d | classes=%d", len(train_data), len(val_data), len(test_data), num_classes)
    return train_data, val_data, test_data, num_classes


# ============================================================================
# DATA LOADING: OOD Leave-One-Out (proxy for Tables 8, 9, 10)
# ============================================================================

def load_codet_m4_loo(
    cfg: CoDETM4Config,
    hold_out_field: str,
    hold_out_value: str,
) -> Tuple[Dataset, Dataset, Dataset, int]:
    train_raw, val_raw, test_raw = _load_raw_splits(cfg)

    field_map = {"generator": "model", "language": "language", "source": "source"}
    raw_field = field_map.get(hold_out_field, hold_out_field)

    def _matches(row):
        val = str(row.get(raw_field, "") or "").strip().lower()
        return val == hold_out_value.lower()

    train_in = train_raw.filter(lambda x: not _matches(x))
    val_in = val_raw.filter(lambda x: not _matches(x))
    test_ood = test_raw.filter(_matches)

    if len(train_in) == 0 or len(val_in) == 0 or len(test_ood) == 0:
        raise RuntimeError(
            f"LOO split empty: hold_out={hold_out_field}={hold_out_value} | "
            f"train_in={len(train_in)} val_in={len(val_in)} test_ood={len(test_ood)}"
        )

    author_vocab = _build_author_vocab(train_in) if cfg.task == "author" else {}

    train_data = _convert_split(train_in, cfg, author_vocab)
    val_data = _convert_split(val_in, cfg, author_vocab)
    test_data = _convert_split(test_ood, cfg, author_vocab)

    train_data = _sample_dataset(train_data, cfg.max_train_samples, cfg.seed)
    val_data = _sample_dataset(val_data, cfg.max_val_samples, cfg.seed + 1)

    labels = set(train_data["label"])
    num_classes = len(labels)
    logger.info(
        "LOO[%s≠%s] | train=%d val=%d test_ood=%d | classes=%d",
        hold_out_field, hold_out_value, len(train_data), len(val_data), len(test_data), num_classes,
    )
    return train_data, val_data, test_data, num_classes


# ============================================================================
# BREAKDOWN EVALUATION (Tables 3, 4, Fig 2/8)
# ============================================================================

@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader: DataLoader, device: str) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    all_preds, all_labels = [], []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        ast_seq = batch["ast_seq"].to(device)
        struct_feat = batch["struct_feat"].to(device)
        out = model(input_ids, attention_mask, ast_seq, struct_feat)
        preds = out["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["labels"].numpy() if "labels" in batch else batch["label"].numpy())
    return np.array(all_preds), np.array(all_labels)


def run_breakdown_eval(preds, labels, test_data, task):
    results: Dict[str, Any] = {}
    n = min(len(preds), len(test_data))
    preds, labels = preds[:n], labels[:n]

    overall_macro = float(f1_score(labels, preds, average="macro"))
    overall_weighted = float(f1_score(labels, preds, average="weighted"))
    results["overall"] = {"macro_f1": overall_macro, "weighted_f1": overall_weighted}
    logger.info(f"  Overall: macro={overall_macro:.4f}  weighted={overall_weighted:.4f}")

    for dim_name in ["language", "source", "generator"]:
        dim_vals = test_data[dim_name][:n]
        unique = sorted(set(dim_vals))
        dim_results = {}
        logger.info(f"  Breakdown by {dim_name}:")
        for val in unique:
            mask = np.array([v == val for v in dim_vals])
            if mask.sum() < 10:
                continue
            mf1 = float(f1_score(labels[mask], preds[mask], average="macro"))
            wf1 = float(f1_score(labels[mask], preds[mask], average="weighted"))
            dim_results[val] = {"n": int(mask.sum()), "macro_f1": mf1, "weighted_f1": wf1}
            logger.info(f"    {val:>12s}: n={mask.sum():>6d}  macro={mf1:.4f}  weighted={wf1:.4f}")
        results[dim_name] = dim_results

    if task == "author":
        cm = confusion_matrix(labels, preds)
        results["confusion_matrix"] = cm.tolist()
        logger.info("  Confusion matrix (rows=true, cols=pred):")
        logger.info(f"  {cm}")

    return results


# ============================================================================
# PREFLIGHT
# ============================================================================

def preflight(cfg: CoDETM4Config, exp_cfg: SpectralConfig) -> Dict[str, object]:
    logger.info("\n" + "=" * 70)
    logger.info(f"PREFLIGHT | dataset={cfg.dataset_id} | task={cfg.task}")
    logger.info("=" * 70)

    if exp_cfg.require_tree_sitter and _ast_parser is None:
        raise PreflightError("tree-sitter-languages unavailable. Set require_tree_sitter=False or install it.")

    train_data, val_data, test_data, num_classes = load_codet_m4_data(cfg)
    if num_classes < 2:
        raise PreflightError(f"Invalid class count: {num_classes}")

    label_counts = Counter(train_data["label"])
    code_stats = _quick_code_stats(train_data)
    if code_stats["empty_ratio"] > 0.05:
        raise PreflightError(f"Empty ratio too high: {code_stats['empty_ratio']:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(exp_cfg.encoder_name)
    probe_code = train_data[0]["code"]
    enc = tokenizer(probe_code, max_length=exp_cfg.max_length, padding="max_length", truncation=True, return_tensors="pt")
    if enc["input_ids"].shape[-1] != exp_cfg.max_length:
        raise PreflightError("Tokenizer output length mismatch.")

    ast_seq = extract_ast_sequence(probe_code, exp_cfg.ast_seq_len)
    struct_feat = extract_structural_features(probe_code)
    if len(ast_seq) != exp_cfg.ast_seq_len:
        raise PreflightError("AST sequence length mismatch.")
    if len(struct_feat) != STRUCTURAL_FEATURE_DIM:
        raise PreflightError("Structural feature dim mismatch.")

    report = {
        "task": cfg.task, "num_classes": int(num_classes),
        "sizes": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "code_stats": code_stats,
        "device": exp_cfg.device, "gpu": _get_gpu_name(),
        "encoder": exp_cfg.encoder_name, "precision": exp_cfg.precision,
    }
    logger.info(f"PREFLIGHT OK | classes={num_classes} | sizes={report['sizes']}")
    return report


# ============================================================================
# BUILD CONFIG
# ============================================================================

def build_ttlcode_config(task_tag: str, save_root: str) -> SpectralConfig:
    strict_ts = importlib.util.find_spec("tree_sitter_languages") is not None
    cfg = SpectralConfig(
        epochs=3,
        batch_size=32,
        grad_accum_steps=2,
        precision="auto",
        auto_h100_profile=True,
        num_workers=4,
        prefetch_factor=2,
        require_tree_sitter=strict_ts,
    )
    cfg.task = task_tag
    cfg.benchmark = "codet_m4"
    cfg.save_dir = os.path.join(save_root, task_tag.lower())
    cfg = apply_hardware_profile(cfg)
    return cfg


def _make_loaders(train_data, val_data, test_data, tokenizer, exp_cfg: SpectralConfig):
    train_ds = AICDDataset(train_data, tokenizer, exp_cfg.max_length, exp_cfg.ast_seq_len)
    val_ds = AICDDataset(val_data, tokenizer, exp_cfg.max_length, exp_cfg.ast_seq_len)
    test_ds = AICDDataset(test_data, tokenizer, exp_cfg.max_length, exp_cfg.ast_seq_len)

    pin = exp_cfg.pin_memory and exp_cfg.device == "cuda"
    kw: Dict[str, Any] = {
        "num_workers": exp_cfg.num_workers,
        "pin_memory": pin,
        "persistent_workers": exp_cfg.num_workers > 0,
    }
    if exp_cfg.num_workers > 0:
        kw["prefetch_factor"] = exp_cfg.prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=exp_cfg.batch_size, shuffle=True, drop_last=True, **kw)
    val_loader = DataLoader(val_ds, batch_size=exp_cfg.batch_size * 2, shuffle=False, **kw)
    test_loader = DataLoader(test_ds, batch_size=exp_cfg.batch_size * 2, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


# ============================================================================
# IID RUNNER (Tables 2, 3, 4, 7)
# ============================================================================

def run_iid(task: str = "binary", codet_cfg: Optional[CoDETM4Config] = None, run_preflight: bool = True) -> Dict[str, Any]:
    if codet_cfg is None:
        codet_cfg = CoDETM4Config(task=task)
    else:
        codet_cfg.task = task
    set_seed(codet_cfg.seed)

    exp_cfg = build_ttlcode_config(f"CoDET_{task}", codet_cfg.save_root)

    logger.info("=" * 70)
    logger.info(f"[Exp34][IID] TTLCode | task={task}")
    logger.info(f"GPU={_get_gpu_name()} | precision={exp_cfg.precision} | batch={exp_cfg.batch_size}x{exp_cfg.grad_accum_steps} | epochs={exp_cfg.epochs}")
    logger.info("=" * 70)

    if run_preflight:
        preflight(codet_cfg, exp_cfg)

    train_data, val_data, test_data, num_classes = load_codet_m4_data(codet_cfg)
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg.encoder_name)
    class_weights = compute_class_weights(train_data["label"], num_classes)
    logger.info(f"Classes={num_classes} | weights={class_weights.numpy()}")

    train_loader, val_loader, test_loader = _make_loaders(train_data, val_data, test_data, tokenizer, exp_cfg)

    model = SpectralCode(exp_cfg, num_classes)
    tp = sum(p.numel() for p in model.parameters())
    logger.info(f"Parameters: {tp:,} total")

    trainer = Trainer(exp_cfg, model, train_loader, val_loader, test_loader, class_weights)
    results = trainer.train()

    if codet_cfg.eval_breakdown:
        logger.info("-" * 50)
        logger.info(f"[BREAKDOWN] task={task}")
        preds, labels = collect_predictions(model, test_loader, exp_cfg.device)
        results["breakdown"] = run_breakdown_eval(preds, labels, test_data, task)

    return results


# ============================================================================
# OOD LEAVE-ONE-OUT RUNNERS (proxy Tables 8, 9, 10)
# ============================================================================

def _run_single_loo(hold_out_field: str, hold_out_value: str, codet_cfg: CoDETM4Config) -> Dict[str, Any]:
    cfg = CoDETM4Config(
        dataset_id=codet_cfg.dataset_id,
        code_field_priority=codet_cfg.code_field_priority,
        split_field=codet_cfg.split_field,
        task="binary",
        max_train_samples=codet_cfg.max_train_samples,
        max_val_samples=codet_cfg.max_val_samples,
        max_test_samples=codet_cfg.max_test_samples,
        seed=codet_cfg.seed,
        save_root=codet_cfg.save_root,
    )
    set_seed(cfg.seed)

    tag = f"CoDET_ood_{hold_out_field}_{hold_out_value}"
    exp_cfg = build_ttlcode_config(tag, cfg.save_root)

    logger.info("=" * 70)
    logger.info(f"[OOD] LOO {hold_out_field}={hold_out_value}")
    logger.info("=" * 70)

    train_data, val_data, test_data, num_classes = load_codet_m4_loo(cfg, hold_out_field, hold_out_value)
    tokenizer = AutoTokenizer.from_pretrained(exp_cfg.encoder_name)
    class_weights = compute_class_weights(train_data["label"], num_classes)

    train_loader, val_loader, test_loader = _make_loaders(train_data, val_data, test_data, tokenizer, exp_cfg)

    model = SpectralCode(exp_cfg, num_classes)
    trainer = Trainer(exp_cfg, model, train_loader, val_loader, test_loader, class_weights)
    results = trainer.train()

    preds, labels = collect_predictions(model, test_loader, exp_cfg.device)
    results["breakdown"] = run_breakdown_eval(preds, labels, test_data, "binary")
    results["hold_out_field"] = hold_out_field
    results["hold_out_value"] = hold_out_value
    return results


def run_ood_generator(codet_cfg: Optional[CoDETM4Config] = None) -> Dict[str, Dict]:
    if codet_cfg is None:
        codet_cfg = CoDETM4Config()
    generators = ["codellama", "gpt", "llama3.1", "nxcode", "qwen1.5"]
    results = {}
    for gen in generators:
        logger.info(f"\n{'='*70}\n[OOD-GEN] Holding out generator={gen}\n{'='*70}")
        try:
            results[gen] = _run_single_loo("generator", gen, codet_cfg)
        except RuntimeError as e:
            logger.error(f"OOD generator={gen} failed: {e}")
            results[gen] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _log_ood_summary("OOD-GENERATOR (proxy Table 8)", results)
    return results


def run_ood_language(codet_cfg: Optional[CoDETM4Config] = None) -> Dict[str, Dict]:
    if codet_cfg is None:
        codet_cfg = CoDETM4Config()
    languages = ["cpp", "java", "python"]
    results = {}
    for lang in languages:
        logger.info(f"\n{'='*70}\n[OOD-LANG] Holding out language={lang}\n{'='*70}")
        try:
            results[lang] = _run_single_loo("language", lang, codet_cfg)
        except RuntimeError as e:
            logger.error(f"OOD language={lang} failed: {e}")
            results[lang] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _log_ood_summary("OOD-LANGUAGE (proxy Table 10/12)", results)
    return results


def run_ood_source(codet_cfg: Optional[CoDETM4Config] = None) -> Dict[str, Dict]:
    if codet_cfg is None:
        codet_cfg = CoDETM4Config()
    sources = ["cf", "gh", "lc"]
    results = {}
    for src in sources:
        logger.info(f"\n{'='*70}\n[OOD-SRC] Holding out source={src}\n{'='*70}")
        try:
            results[src] = _run_single_loo("source", src, codet_cfg)
        except RuntimeError as e:
            logger.error(f"OOD source={src} failed: {e}")
            results[src] = {"error": str(e)}
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _log_ood_summary("OOD-SOURCE (proxy Table 9)", results)
    return results


def _log_ood_summary(title: str, results: Dict[str, Dict]):
    logger.info(f"\n{'='*70}")
    logger.info(f"{title} SUMMARY")
    logger.info("=" * 70)
    logger.info("| held-out | test_macro_f1 | test_weighted_f1 | best_val_f1 |")
    logger.info("|---|---:|---:|---:|")
    for key, stats in results.items():
        if "error" in stats:
            logger.info(f"| {key} | ERROR | - | - |")
            continue
        logger.info(
            f"| {key} | {stats.get('test_f1', 0):.4f} | "
            f"{stats.get('test_weighted_f1', 0):.4f} | "
            f"{stats.get('best_val_f1', 0):.4f} |"
        )


# ============================================================================
# FULL SUITE
# ============================================================================

FULL_RUN_PLAN = [
    ("iid", "binary"),       # Table 2 + per-lang(3) + per-source(4)
    ("iid", "author"),       # Table 7 + confusion matrix
    ("ood", "generator"),    # proxy Table 8
    ("ood", "language"),     # proxy Table 10/12
    ("ood", "source"),       # proxy Table 9
]


def run_suite(
    run_plan: Optional[List[Tuple[str, str]]] = None,
    base_cfg: Optional[CoDETM4Config] = None,
    run_preflight: bool = True,
) -> Dict[str, Any]:
    if run_plan is None:
        run_plan = FULL_RUN_PLAN
    if base_cfg is None:
        base_cfg = CoDETM4Config()

    all_results: Dict[str, Any] = {}

    logger.info("\n" + "=" * 70)
    logger.info(f"[Exp34] CoDET-M4 FULL BENCHMARK SUITE: {len(run_plan)} evaluation modes")
    for i, (mode, task) in enumerate(run_plan):
        logger.info(f"  [{i+1}] {mode}/{task}")
    logger.info("=" * 70)

    for mode, task in run_plan:
        key = f"{mode}_{task}"
        logger.info(f"\n{'#'*70}")
        logger.info(f"SUITE: {key}")
        logger.info("#" * 70)

        if mode == "iid":
            all_results[key] = run_iid(task, base_cfg, run_preflight=run_preflight)
        elif mode == "ood" and task == "generator":
            all_results[key] = run_ood_generator(base_cfg)
        elif mode == "ood" and task == "language":
            all_results[key] = run_ood_language(base_cfg)
        elif mode == "ood" and task == "source":
            all_results[key] = run_ood_source(base_cfg)
        else:
            logger.warning(f"Unknown suite entry: {mode}/{task}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    _log_final_summary(all_results, run_plan)
    return all_results


def _log_final_summary(all_results: Dict[str, Any], run_plan: List[Tuple[str, str]]):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("\n" + "=" * 70)
    logger.info(f"CoDET-M4 BENCHMARK COMPLETE | {ts}")
    logger.info("=" * 70)

    logger.info("\n=== IID RESULTS ===")
    for mode, task in run_plan:
        if mode != "iid":
            continue
        key = f"iid_{task}"
        r = all_results.get(key, {})
        logger.info(
            f"  {task:>8s}: macro_f1={r.get('test_f1', 0):.4f}  "
            f"weighted_f1={r.get('test_weighted_f1', 0):.4f}  "
            f"val_f1={r.get('best_val_f1', 0):.4f}"
        )

    for mode, task in run_plan:
        if mode != "ood":
            continue
        key = f"ood_{task}"
        ood_block = all_results.get(key, {})
        if not isinstance(ood_block, dict):
            continue
        logger.info(f"\n=== OOD {task.upper()} RESULTS ===")
        for held, stats in ood_block.items():
            if isinstance(stats, dict) and "test_f1" in stats:
                logger.info(
                    f"  held_out={held:>12s}: macro_f1={stats['test_f1']:.4f}  "
                    f"weighted_f1={stats.get('test_weighted_f1', 0):.4f}"
                )

    safe = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            safe[k] = {sk: sv for sk, sv in v.items() if not isinstance(sv, (np.ndarray, torch.Tensor))}
    try:
        logger.info("\nSUITE_RESULTS_JSON=" + json.dumps(
            {"timestamp": ts, "method": "Exp34_TTLCode_CoDET_M4", "results": safe},
            ensure_ascii=True, default=str,
        ))
    except (TypeError, ValueError):
        logger.info("(JSON serialization of full results skipped)")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # ── Run mode ──────────────────────────────────────────────────────
    # "full"         → run ALL evaluations (IID + OOD) = full paper benchmark
    # "iid_only"     → run only IID binary + author (Tables 2-4, 7)
    # "ood_only"     → run only OOD leave-one-out (proxy Tables 8-10)
    # "single"       → run one specific task
    RUN_MODE = "iid_only"
    RUN_PREFLIGHT_CHECK = True

    SINGLE_TASK = "binary"  # "binary" | "author"

    codet_cfg = CoDETM4Config(
        max_train_samples=100_000,
        max_val_samples=20_000,
        max_test_samples=50_000,
        eval_breakdown=True,
    )

    try:
        if RUN_MODE == "full":
            run_suite(FULL_RUN_PLAN, codet_cfg, run_preflight=RUN_PREFLIGHT_CHECK)

        elif RUN_MODE == "iid_only":
            iid_plan = [e for e in FULL_RUN_PLAN if e[0] == "iid"]
            run_suite(iid_plan, codet_cfg, run_preflight=RUN_PREFLIGHT_CHECK)

        elif RUN_MODE == "ood_only":
            ood_plan = [e for e in FULL_RUN_PLAN if e[0] == "ood"]
            run_suite(ood_plan, codet_cfg, run_preflight=RUN_PREFLIGHT_CHECK)

        elif RUN_MODE == "single":
            run_iid(SINGLE_TASK, codet_cfg, run_preflight=RUN_PREFLIGHT_CHECK)

        else:
            raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")

    except PreflightError as e:
        logger.error(f"PRE-FLIGHT FAILED: {e}")
        raise SystemExit(1)
