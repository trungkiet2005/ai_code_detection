"""
[CoDET-M4] GroupDRO Runner – Full Benchmark Evaluation  (Kaggle standalone)
      Exp15: GroupDRO-SpectralCode (Group Distributionally Robust Optimization)

Key novelty: DRO over language×source environments (9 groups) with exponentiated-
gradient group weight updates. Forces worst-group accuracy maximization, directly
attacking the GitHub/cross-source generalization gap.

Inspired by: Sagawa et al. "Distributionally Robust Neural Networks" (ICLR 2020),
             WILDS benchmark (NeurIPS 2021), ChillStep (ICLR 2025)

Target: NeurIPS/ICLR 2026 (A* venue)
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

def autocast(device_type: str = "cuda", enabled: bool = True, dtype=None):
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

@dataclass
class GroupDROConfig:
    # --- Dataset / task ---
    dataset_id: str = "DaniilOr/CoDET-M4"
    code_field_priority: Tuple[str, ...] = ("cleaned_code", "code")
    split_field: str = "split"
    task: str = "binary"           # "binary" | "author"

    # --- Architecture ---
    encoder_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    z_style_dim: int = 256
    z_content_dim: int = 256
    ast_embed_dim: int = 64
    ast_hidden_dim: int = 128
    ast_seq_len: int = 128
    num_ast_node_types: int = 256

    # --- Training ---
    epochs: int = 3
    batch_size: int = 32
    grad_accum_steps: int = 2
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # --- GroupDRO ---
    dro_eta: float = 0.01          # group weight learning rate (exponentiated gradient)
    lambda_neural: float = 0.3     # aux neural head weight
    lambda_spectral: float = 0.3   # aux spectral head weight
    focal_gamma: float = 2.0       # focal loss gamma

    # --- Sample limits ---
    max_train_samples: int = 100_000
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000

    # --- Infra ---
    num_workers: int = 2
    prefetch_factor: int = 2
    seed: int = 42
    precision: str = "auto"
    auto_h100_profile: bool = True
    require_tree_sitter: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    non_blocking: bool = True
    save_dir: str = "./codet_m4_groupdro_checkpoints"
    log_every: int = 200
    eval_every: int = 2000
    eval_breakdown: bool = True


def apply_hardware_profile(config: GroupDROConfig) -> GroupDROConfig:
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
    config.log_every = 200
    config.eval_every = 2000
    logger.info(
        "Applied H100 profile | precision=%s | batch=%d | accum=%d | workers=%d",
        config.precision, config.batch_size, config.grad_accum_steps, config.num_workers,
    )
    return config


def preflight_check(config: GroupDROConfig) -> Dict[str, Any]:
    """Sanity-check dataset, tokenizer, and feature shapes before full training."""
    logger.info("\n" + "=" * 70)
    logger.info("PREFLIGHT CHECK | Exp15: GroupDRO-SpectralCode")
    logger.info("=" * 70)

    if config.require_tree_sitter and _ast_parser is None:
        raise PreflightError("tree-sitter-languages unavailable. Set require_tree_sitter=False or install it.")

    train_data, val_data, test_data, num_classes, _ = _load_iid_splits(config)
    if num_classes < 2:
        raise PreflightError(f"Invalid class count: {num_classes}")

    label_counts = Counter(train_data["label"])
    n = min(len(train_data), 256)
    sample_codes = train_data.select(range(n))["code"]
    empty = sum(1 for c in sample_codes if not c.strip())
    if empty / n > 0.05:
        raise PreflightError(f"Empty code ratio too high: {empty/n:.3f}")

    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
    probe = train_data[0]["code"]
    enc = tokenizer(probe, max_length=config.max_length, padding="max_length", truncation=True, return_tensors="pt")
    if enc["input_ids"].shape[-1] != config.max_length:
        raise PreflightError("Tokenizer output length mismatch.")

    ast_seq = extract_ast_sequence(probe, config.ast_seq_len)
    struct_feat = extract_structural_features(probe)
    if len(ast_seq) != config.ast_seq_len:
        raise PreflightError(f"AST seq length mismatch: got {len(ast_seq)}, expected {config.ast_seq_len}")
    if len(struct_feat) != STRUCTURAL_FEATURE_DIM:
        raise PreflightError(f"Structural feat dim mismatch: got {len(struct_feat)}, expected {STRUCTURAL_FEATURE_DIM}")

    # Probe group extraction
    grp = _make_group_id(train_data[0]["language"], train_data[0]["source"])
    logger.info("Probe group_id: %d  (%s, %s)", grp, train_data[0]["language"], train_data[0]["source"])

    report = {
        "task": config.task,
        "num_classes": int(num_classes),
        "sizes": {"train": len(train_data), "val": len(val_data), "test": len(test_data)},
        "label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "device": config.device,
        "gpu": _get_gpu_name(),
        "encoder": config.encoder_name,
        "precision": config.precision,
    }
    logger.info("PREFLIGHT OK | classes=%d | sizes=%s", num_classes, report["sizes"])
    return report


# ============================================================================
# Group definitions
# ============================================================================

# Canonical 9 groups: {cpp, java, python} x {cf, gh, lc}
LANGUAGES = ["cpp", "java", "python"]
SOURCES = ["cf", "gh", "lc"]
NUM_GROUPS = len(LANGUAGES) * len(SOURCES)  # 9

_LANG_IDX: Dict[str, int] = {l: i for i, l in enumerate(LANGUAGES)}
_SRC_IDX: Dict[str, int] = {s: i for i, s in enumerate(SOURCES)}


def _make_group_id(language: str, source: str) -> int:
    """Map (language, source) -> group_id in [0, NUM_GROUPS-1]. Unknown -> -1."""
    lang = str(language or "").strip().lower()
    src = str(source or "").strip().lower()
    li = _LANG_IDX.get(lang, -1)
    si = _SRC_IDX.get(src, -1)
    if li < 0 or si < 0:
        return -1
    return li * len(SOURCES) + si


def _group_name(gid: int) -> str:
    if gid < 0 or gid >= NUM_GROUPS:
        return "unknown"
    lang = LANGUAGES[gid // len(SOURCES)]
    src = SOURCES[gid % len(SOURCES)]
    return f"{lang}_{src}"


# ============================================================================
# AST Feature Extraction (BiLSTM backbone, identical to Exp11)
# ============================================================================

AST_NODE_VOCAB: Dict[str, int] = {
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
                node_types: List[int] = []
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
    features: List[int] = []
    patterns = {
        "function_definition": r"\b(def|function|func|fn)\s+\w+",
        "class_definition":    r"\b(class|struct|interface|enum)\s+\w+",
        "if_statement":        r"\bif\s*[\(\{]",
        "for_statement":       r"\b(for|foreach)\s*[\(\{]",
        "while_statement":     r"\bwhile\s*[\(\{]",
        "return_statement":    r"\breturn\b",
        "import_statement":    r"\b(import|include|require|using)\b",
        "try_statement":       r"\btry\s*[\{\:]",
        "catch_clause":        r"\b(catch|except)\b",
        "comment":             r"(//|#|/\*|\"\"\"|\'\'\')",
        "assignment":          r"[^=!<>]=[^=]",
        "call_expression":     r"\w+\s*\(",
        "lambda_expression":   r"\b(lambda|=>)\b",
        "string_literal":      r"[\"']",
        "variable_declaration": r"\b(var|let|const|int|float|double|string|bool)\b",
        "switch_statement":    r"\b(switch|match)\b",
        "throw_statement":     r"\b(throw|raise)\b",
        "async_function":      r"\basync\b",
        "await_expression":    r"\bawait\b",
        "yield_statement":     r"\byield\b",
        "with_statement":      r"\bwith\b",
        "assert_statement":    r"\bassert\b",
    }
    for line in code.split("\n"):
        for node_type, pattern in patterns.items():
            if re.search(pattern, line):
                features.append(AST_NODE_VOCAB.get(node_type, AST_NODE_VOCAB["UNK"]))
    return features if features else [AST_NODE_VOCAB["UNK"]]


_ast_parser = _try_load_tree_sitter()
STRUCTURAL_FEATURE_DIM = 22
SPECTRAL_FEATURE_DIM = 64


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
    avg_line_len = float(np.mean([len(l) for l in lines])) if lines else 0.0
    max_line_len = float(max((len(l) for l in lines), default=0))

    indents: List[int] = []
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indents.append(len(line) - len(stripped))
    avg_indent = float(np.mean(indents)) if indents else 0.0
    max_indent = float(max(indents)) if indents else 0.0
    indent_variance = float(np.var(indents)) if indents else 0.0

    num_functions   = code.count("def ") + code.count("function ") + code.count("func ") + code.count("fn ")
    num_classes     = code.count("class ") + code.count("struct ") + code.count("interface ")
    num_loops       = code.count("for ") + code.count("while ") + code.count("foreach ")
    num_conds       = code.count("if ") + code.count("else ") + code.count("elif ") + code.count("else if")
    num_returns     = code.count("return ")
    num_comments    = code.count("//") + code.count("#") + code.count("/*")
    num_imports     = code.count("import ") + code.count("include ") + code.count("require ") + code.count("using ")
    num_try_catch   = code.count("try") + code.count("catch") + code.count("except")

    identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)
    num_snake   = sum(1 for i in identifiers if '_' in i and i.islower())
    num_camel   = sum(1 for i in identifiers if any(c.isupper() for c in i[1:]) and '_' not in i)
    num_single  = sum(1 for i in identifiers if len(i) == 1)
    avg_id_len  = float(np.mean([len(i) for i in identifiers])) if identifiers else 0.0

    empty_lines = sum(1 for l in lines if not l.strip())
    n_id = max(len(identifiers), 1)
    return [
        num_lines, avg_line_len, max_line_len,
        avg_indent, max_indent, indent_variance,
        float(num_functions), float(num_classes), float(num_loops), float(num_conds),
        float(num_returns), float(num_comments), float(num_imports), float(num_try_catch),
        num_snake / n_id, num_camel / n_id, num_single / n_id, avg_id_len,
        empty_lines / max(num_lines, 1),
        sum(c.isalpha() for c in code) / max(len(code), 1),
        sum(c.isdigit() for c in code) / max(len(code), 1),
        sum(c.isspace() for c in code) / max(len(code), 1),
    ]


# ============================================================================
# FFT Spectral Features (identical to Exp11)
# ============================================================================

def extract_spectral_features(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    batch_size = input_ids.size(0)
    features: List[List[float]] = []
    windows = [32, 64, 128, 256]
    feats_per_win = SPECTRAL_FEATURE_DIM // len(windows)

    for i in range(batch_size):
        ids = input_ids[i][attention_mask[i].bool()].cpu().float().numpy()
        seq_len = len(ids)
        sample_feats: List[float] = []

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
                sample_feats.extend([0.0] * feats_per_win)
                continue

            total_energy = float(np.sum(magnitude ** 2))
            n_bands = min(8, len(magnitude))
            band_size = max(1, len(magnitude) // n_bands)
            band_energies: List[float] = []
            for b in range(n_bands):
                start = b * band_size
                end = min(start + band_size, len(magnitude))
                band_energies.append(float(np.sum(magnitude[start:end] ** 2) / (total_energy + 1e-10)))

            while len(band_energies) < feats_per_win - 4:
                band_energies.append(0.0)
            band_energies = band_energies[:feats_per_win - 4]

            spectral_centroid = float(np.sum(np.arange(len(magnitude)) * magnitude) / (np.sum(magnitude) + 1e-10))
            spectral_rolloff  = float(np.searchsorted(np.cumsum(magnitude), 0.85 * np.sum(magnitude))) / len(magnitude)
            spectral_flatness = float(np.exp(np.mean(np.log(magnitude + 1e-10))) / (np.mean(magnitude) + 1e-10))
            peak_freq = int(np.argmax(magnitude[1:]) + 1) if len(magnitude) > 1 else 0

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

class CoDETDataset(TorchDataset):
    """CoDET-M4 dataset wrapper; exposes group_id alongside standard fields."""

    def __init__(self, data: Dataset, tokenizer, max_length: int = 512, ast_seq_len: int = 128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ast_seq_len = ast_seq_len

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        item = self.data[idx]
        code: str = item["code"]
        label: int = int(item["label"])
        group_id: int = int(item.get("group_id", -1))

        encoding = self.tokenizer(
            code, max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt",
        )
        ast_seq = extract_ast_sequence(code, self.ast_seq_len)
        struct_feat = extract_structural_features(code)

        return {
            "input_ids":      encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ast_seq":        torch.tensor(ast_seq, dtype=torch.long),
            "struct_feat":    torch.tensor(struct_feat, dtype=torch.float32),
            "label":          torch.tensor(label, dtype=torch.long),
            "group_id":       torch.tensor(group_id, dtype=torch.long),
        }


# ============================================================================
# Model Components (identical backbone to Exp11 SpectralCode)
# ============================================================================

class FFTSpectralFeatures(nn.Module):
    """Learnable projection of raw FFT energy features (64-dim input)."""

    def __init__(self, spectral_in: int = SPECTRAL_FEATURE_DIM, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(spectral_in, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(256, out_dim), nn.LayerNorm(out_dim), nn.GELU(),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ASTFeatures(nn.Module):
    """BiLSTM AST encoder – 128-dim output from AST token sequence."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.out_dim = hidden_dim

    def forward(self, ast_seq: torch.Tensor) -> torch.Tensor:
        x = self.embedding(ast_seq)
        _, (h_n, _) = self.lstm(x)
        h = torch.cat([h_n[0], h_n[1]], dim=-1)
        return self.proj(h)


class StructuralFeatures(nn.Module):
    """MLP encoder for the 22 hand-crafted structural features."""

    def __init__(self, input_dim: int = STRUCTURAL_FEATURE_DIM, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.GELU(),
        )
        self.out_dim = hidden_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class GatedFusion(nn.Module):
    """Soft-gate fusion over [cls, ast, struct, spectral] streams."""

    def __init__(self, cls_dim: int, ast_dim: int, struct_dim: int, spectral_dim: int, out_dim: int):
        super().__init__()
        self.cls_proj     = nn.Linear(cls_dim, out_dim)
        self.ast_proj     = nn.Linear(ast_dim, out_dim)
        self.struct_proj  = nn.Linear(struct_dim, out_dim)
        self.spectral_proj = nn.Linear(spectral_dim, out_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=out_dim, num_heads=4, dropout=0.1, batch_first=True)
        self.norm = nn.LayerNorm(out_dim)
        self.gate = nn.Sequential(
            nn.Linear(out_dim * 4, 64), nn.GELU(), nn.Linear(64, 4),
        )
        self.out_dim = out_dim

    def forward(
        self,
        cls_repr: torch.Tensor,
        ast_repr: torch.Tensor,
        struct_repr: torch.Tensor,
        spectral_repr: torch.Tensor,
    ) -> torch.Tensor:
        c = self.cls_proj(cls_repr)
        a = self.ast_proj(ast_repr)
        s = self.struct_proj(struct_repr)
        sp = self.spectral_proj(spectral_repr)
        seq = torch.stack([c, a, s, sp], dim=1)          # [B, 4, D]
        attn_out, _ = self.cross_attn(seq, seq, seq)
        attn_out = self.norm(attn_out + seq)              # residual
        concat = attn_out.reshape(attn_out.size(0), -1)  # [B, 4*D]
        gate_w = F.softmax(self.gate(concat), dim=-1)    # [B, 4]
        fused = (
            gate_w[:, 0:1] * attn_out[:, 0] +
            gate_w[:, 1:2] * attn_out[:, 1] +
            gate_w[:, 2:3] * attn_out[:, 2] +
            gate_w[:, 3:4] * attn_out[:, 3]
        )
        return fused


class ClassifierHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(in_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.drop(x))


# ============================================================================
# GroupDROSpectralModel
# ============================================================================

class GroupDROSpectralModel(nn.Module):
    """
    SpectralCode backbone (Exp11) with GroupDRO-compatible multi-head outputs.

    Outputs three sets of logits:
      - main_logits   (gated fusion of all streams)
      - neural_logits (CLS + AST + struct only)
      - spectral_logits (spectral stream only)
    """

    def __init__(self, config: GroupDROConfig, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # BERT backbone
        self.token_encoder = AutoModel.from_pretrained(
            config.encoder_name, attn_implementation="sdpa"
        )
        cls_dim = self.token_encoder.config.hidden_size

        # Feature encoders
        self.ast_enc    = ASTFeatures(config.num_ast_node_types, config.ast_embed_dim, config.ast_hidden_dim)
        self.struct_enc = StructuralFeatures(STRUCTURAL_FEATURE_DIM, config.ast_hidden_dim)
        self.spec_enc   = FFTSpectralFeatures(SPECTRAL_FEATURE_DIM, out_dim=128)

        # Fusion dim = style + content (512)
        fusion_dim = config.z_style_dim + config.z_content_dim

        # Gated fusion over all 4 streams
        self.fusion = GatedFusion(
            cls_dim, config.ast_hidden_dim, config.ast_hidden_dim, 128, fusion_dim
        )

        # Classifier heads
        self.main_head      = ClassifierHead(fusion_dim, num_classes)
        self.neural_head    = ClassifierHead(fusion_dim, num_classes)   # CLS-only aux
        self.spectral_head  = ClassifierHead(128, num_classes)          # spectral-only aux

        # Separate neural projection (no spectral)
        self.neural_fusion = nn.MultiheadAttention(
            embed_dim=fusion_dim, num_heads=4, dropout=0.1, batch_first=True
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ast_seq: torch.Tensor,
        struct_feat: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        # Token encoding
        tok_out  = self.token_encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_repr = tok_out.last_hidden_state[:, 0, :]  # [B, cls_dim]

        # Auxiliary encoders
        ast_repr    = self.ast_enc(ast_seq)          # [B, ast_hidden_dim]
        struct_repr = self.struct_enc(struct_feat)   # [B, ast_hidden_dim]

        # Spectral features (computed from raw token IDs – no grad needed for extraction)
        with torch.no_grad():
            spec_raw = extract_spectral_features(input_ids, attention_mask)  # [B, 64]
        spec_raw = spec_raw.to(cls_repr.dtype)
        spec_repr = self.spec_enc(spec_raw)           # [B, 128]

        # Gated fusion: cls + ast + struct + spectral
        h_main = self.fusion(cls_repr, ast_repr, struct_repr, spec_repr)   # [B, fusion_dim]

        # Neural-only path (for aux loss): simple projection of cls
        # We reuse h_main but projected without spectral for contrast
        h_neural = h_main  # simplification: same representation, separate head

        main_logits     = self.main_head(h_main)
        neural_logits   = self.neural_head(h_neural)
        spectral_logits = self.spectral_head(spec_repr)

        return {
            "logits":          main_logits,
            "neural_logits":   neural_logits,
            "spectral_logits": spectral_logits,
        }


# ============================================================================
# GroupDRO Loss
# ============================================================================

class GroupDROLoss:
    """
    Exponentiated-gradient GroupDRO loss (Sagawa et al., ICLR 2020).

    Group weights w_g are maintained on CPU and moved to device as needed.
    Per-step update:
        L_g    = average focal CE over samples with group_id == g
        w_g   <- w_g * exp(eta * L_g)
        w_g   /= sum(w_g)                   (renormalize)
        L_DRO  = sum_g w_g * L_g
    """

    def __init__(
        self,
        num_groups: int = NUM_GROUPS,
        eta: float = 0.01,
        gamma: float = 2.0,
        lambda_neural: float = 0.3,
        lambda_spectral: float = 0.3,
        device: str = "cpu",
    ):
        self.num_groups = num_groups
        self.eta = eta
        self.gamma = gamma
        self.lambda_neural = lambda_neural
        self.lambda_spectral = lambda_spectral
        self.device = device

        # Initialize group weights uniformly
        self.group_weights = torch.ones(num_groups, dtype=torch.float32) / num_groups

        # Running group loss stats for logging
        self.group_loss_sum   = torch.zeros(num_groups)
        self.group_loss_count = torch.zeros(num_groups)

    def _focal_ce(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = "none",
    ) -> torch.Tensor:
        ce = F.cross_entropy(logits, targets, reduction=reduction)
        if reduction == "none":
            pt = torch.exp(-ce)
            return ((1.0 - pt) ** self.gamma) * ce
        return ce

    def compute(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        group_ids: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Returns (total_loss, per_group_loss_dict).
        Handles edge cases: groups absent in batch get no update.
        """
        device = labels.device
        w = self.group_weights.to(device)

        # Per-sample focal CE on main logits
        per_sample_loss = self._focal_ce(outputs["logits"], labels, reduction="none")

        # Compute per-group mean losses
        group_losses = torch.zeros(self.num_groups, device=device)
        group_counts = torch.zeros(self.num_groups, device=device)

        valid_mask = group_ids >= 0
        if valid_mask.sum() == 0:
            # No valid groups in batch – fall back to simple mean
            dro_loss = per_sample_loss.mean()
            self._update_cpu_stats(group_losses.cpu(), group_counts.cpu())
            info = {_group_name(g): 0.0 for g in range(self.num_groups)}
            return dro_loss, info

        for g in range(self.num_groups):
            mask = valid_mask & (group_ids == g)
            if mask.sum() > 0:
                group_losses[g] = per_sample_loss[mask].mean()
                group_counts[g] = mask.sum().float()

        # Exponentiated gradient weight update (with group_losses on CPU)
        gl_cpu = group_losses.detach().cpu()
        self.group_weights = self.group_weights * torch.exp(self.eta * gl_cpu)
        self.group_weights = self.group_weights / (self.group_weights.sum() + 1e-10)

        # Update running stats
        self._update_cpu_stats(gl_cpu, group_counts.detach().cpu())

        # DRO loss = weighted sum of group losses
        w_updated = self.group_weights.to(device)
        dro_loss = (w_updated * group_losses).sum()

        # Auxiliary losses
        neural_loss   = self._focal_ce(outputs["neural_logits"],   labels, reduction="mean")
        spectral_loss = self._focal_ce(outputs["spectral_logits"], labels, reduction="mean")

        total_loss = dro_loss + self.lambda_neural * neural_loss + self.lambda_spectral * spectral_loss

        info = {
            _group_name(g): float(group_losses[g].item())
            for g in range(self.num_groups)
        }
        info["dro_loss"]      = float(dro_loss.item())
        info["neural_loss"]   = float(neural_loss.item())
        info["spectral_loss"] = float(spectral_loss.item())
        return total_loss, info

    def _update_cpu_stats(self, gl_cpu: torch.Tensor, gc_cpu: torch.Tensor):
        self.group_loss_sum   = self.group_loss_sum   + gl_cpu * gc_cpu
        self.group_loss_count = self.group_loss_count + gc_cpu

    def get_weight_snapshot(self) -> Dict[str, float]:
        snap = {}
        worst_g, worst_w = -1, -1.0
        for g in range(self.num_groups):
            w = float(self.group_weights[g].item())
            snap[_group_name(g)] = w
            if w > worst_w:
                worst_w = w
                worst_g = g
        snap["_worst_group"] = _group_name(worst_g)
        snap["_worst_weight"] = worst_w
        return snap

    def format_weight_log(self) -> str:
        parts = []
        snap = self.get_weight_snapshot()
        worst = snap["_worst_group"]
        for g in range(self.num_groups):
            name = _group_name(g)
            w = snap[name]
            marker = " ***" if name == worst else ""
            parts.append(f"{name}={w:.3f}{marker}")
        return "  ".join(parts)

    def reset_stats(self):
        self.group_loss_sum   = torch.zeros(self.num_groups)
        self.group_loss_count = torch.zeros(self.num_groups)


# ============================================================================
# Data loading helpers
# ============================================================================

def _sample_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = rng.sample(range(len(dataset)), max_samples)
    return dataset.select(indices)


def _normalize_target(value: object) -> str:
    return str(value or "").strip().lower()


def _is_human(target: str) -> bool:
    return target in {"human", "human_written", "human-generated", "human_generated"}


def _build_author_vocab(train_ds: Dataset) -> Dict[str, int]:
    model_names: set = set()
    for row in train_ds:
        target = _normalize_target(row.get("target", row.get("generator", "")))
        model_name = str(row.get("model", row.get("generator", "")) or "").strip()
        if not _is_human(target) and model_name and model_name != "human":
            model_names.add(model_name)
    return {name: idx + 1 for idx, name in enumerate(sorted(model_names))}


def _convert_row(row: Dict, task: str, author_vocab: Dict[str, int]) -> Dict:
    # Code field
    code = ""
    for f in ("cleaned_code", "code"):
        v = row.get(f, "")
        if isinstance(v, str) and v.strip():
            code = v
            break

    # Generator / target
    generator_raw = str(row.get("generator", row.get("model", "")) or "").strip().lower()
    target_raw = _normalize_target(row.get("target", generator_raw))

    is_human = _is_human(target_raw) or generator_raw == "human"

    if task == "binary":
        label = 0 if is_human else 1
    elif task == "author":
        if is_human:
            label = 0
        else:
            gname = str(row.get("model", generator_raw) or "").strip().lower()
            label = author_vocab.get(gname, -1)
    else:
        raise ValueError(f"Unknown task: {task}")

    language = str(row.get("language", "")).strip().lower()
    source   = str(row.get("source",   "")).strip().lower()

    return {
        "code":      code,
        "label":     label,
        "language":  language,
        "source":    source,
        "generator": "human" if is_human else generator_raw,
        "group_id":  _make_group_id(language, source),
    }


def _convert_split(ds: Dataset, task: str, author_vocab: Dict[str, int]) -> Dataset:
    converted = ds.map(
        lambda row: _convert_row(row, task, author_vocab),
        remove_columns=ds.column_names,
    )
    return converted.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _get_raw_splits(config: GroupDROConfig):
    logger.info("Loading dataset: %s", config.dataset_id)
    ds = load_dataset(config.dataset_id, split="train")

    if config.split_field in ds.column_names:
        logger.info("Using built-in split column")
        train_r = ds.filter(lambda x: str(x.get(config.split_field, "")).lower() == "train")
        val_r   = ds.filter(lambda x: str(x.get(config.split_field, "")).lower() in {"val", "validation", "dev"})
        test_r  = ds.filter(lambda x: str(x.get(config.split_field, "")).lower() == "test")
    else:
        logger.warning("No split column; creating 80/10/10")
        s1 = ds.train_test_split(test_size=0.1, seed=config.seed)
        test_r = s1["test"]
        s2 = s1["train"].train_test_split(test_size=1/9, seed=config.seed)
        train_r, val_r = s2["train"], s2["test"]

    if min(len(train_r), len(val_r), len(test_r)) == 0:
        raise RuntimeError("One or more CoDET-M4 splits are empty.")
    return train_r, val_r, test_r


def _load_iid_splits(config: GroupDROConfig):
    train_r, val_r, test_r = _get_raw_splits(config)
    author_vocab = _build_author_vocab(train_r) if config.task == "author" else {}
    if config.task == "author":
        logger.info("Author vocab (%d): %s", len(author_vocab), sorted(author_vocab.keys()))

    train_data = _convert_split(train_r, config.task, author_vocab)
    val_data   = _convert_split(val_r,   config.task, author_vocab)
    test_data  = _convert_split(test_r,  config.task, author_vocab)

    train_data = _sample_dataset(train_data, config.max_train_samples, config.seed)
    val_data   = _sample_dataset(val_data,   config.max_val_samples,   config.seed + 1)
    test_data  = _sample_dataset(test_data,  config.max_test_samples,  config.seed + 2)

    num_classes = len(set(train_data["label"]))
    logger.info("IID | train=%d val=%d test=%d | classes=%d", len(train_data), len(val_data), len(test_data), num_classes)
    return train_data, val_data, test_data, num_classes, author_vocab


def _load_ood_splits(config: GroupDROConfig, hold_out_field: str, hold_out_value: str):
    train_r, val_r, test_r = _get_raw_splits(config)

    # Map friendly field name to actual dataset column
    field_map = {"generator": "generator", "language": "language", "source": "source"}
    raw_field = field_map.get(hold_out_field, hold_out_field)

    def _matches(row):
        v = str(row.get(raw_field, "") or "").strip().lower()
        return v == hold_out_value.lower()

    train_in  = train_r.filter(lambda x: not _matches(x))
    val_in    = val_r.filter(lambda x: not _matches(x))
    test_ood  = test_r.filter(_matches)

    if min(len(train_in), len(val_in)) == 0:
        raise RuntimeError(f"LOO train/val empty: {hold_out_field}={hold_out_value}")
    if len(test_ood) == 0:
        raise RuntimeError(f"LOO test_ood empty: {hold_out_field}={hold_out_value}")

    author_vocab = _build_author_vocab(train_in) if config.task == "author" else {}

    train_data = _convert_split(train_in,  config.task, author_vocab)
    val_data   = _convert_split(val_in,    config.task, author_vocab)
    test_data  = _convert_split(test_ood,  config.task, author_vocab)

    train_data = _sample_dataset(train_data, config.max_train_samples, config.seed)
    val_data   = _sample_dataset(val_data,   config.max_val_samples,   config.seed + 1)

    num_classes = len(set(train_data["label"]))
    logger.info(
        "OOD LOO [%s != %s] | train=%d val=%d test_ood=%d | classes=%d",
        hold_out_field, hold_out_value, len(train_data), len(val_data), len(test_data), num_classes,
    )
    return train_data, val_data, test_data, num_classes


def _make_loaders(
    train_data: Dataset, val_data: Dataset, test_data: Dataset,
    tokenizer, config: GroupDROConfig,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train_ds = CoDETDataset(train_data, tokenizer, config.max_length, config.ast_seq_len)
    val_ds   = CoDETDataset(val_data,   tokenizer, config.max_length, config.ast_seq_len)
    test_ds  = CoDETDataset(test_data,  tokenizer, config.max_length, config.ast_seq_len)

    pin = config.pin_memory and config.device == "cuda"
    kw: Dict[str, Any] = {
        "num_workers":        config.num_workers,
        "pin_memory":         pin,
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        kw["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(train_ds, batch_size=config.batch_size,     shuffle=True,  drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=config.batch_size * 2, shuffle=False, **kw)
    test_loader  = DataLoader(test_ds,  batch_size=config.batch_size * 2, shuffle=False, **kw)
    return train_loader, val_loader, test_loader


def _compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)
    w = 1.0 / counts
    w = w / w.sum() * num_classes
    return torch.tensor(w, dtype=torch.float32)


# ============================================================================
# Training loop
# ============================================================================

def train_epoch_groupdro(
    model: GroupDROSpectralModel,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    dro_loss_fn: GroupDROLoss,
    config: GroupDROConfig,
    epoch: int,
    scaler,
    use_amp: bool,
    amp_dtype,
    global_step_ref: List[int],  # mutable [step] counter
) -> Dict[str, float]:
    """One training epoch with GroupDRO. Returns per-group stats dict."""
    model.train()
    dro_loss_fn.reset_stats()
    total_loss_acc = 0.0
    num_batches    = 0
    optimizer.zero_grad()

    for batch_idx, batch in enumerate(loader):
        input_ids     = batch["input_ids"].to(config.device,     non_blocking=config.non_blocking)
        attention_mask = batch["attention_mask"].to(config.device, non_blocking=config.non_blocking)
        ast_seq       = batch["ast_seq"].to(config.device,       non_blocking=config.non_blocking)
        struct_feat   = batch["struct_feat"].to(config.device,   non_blocking=config.non_blocking)
        labels        = batch["label"].to(config.device,         non_blocking=config.non_blocking)
        group_ids     = batch["group_id"].to(config.device,      non_blocking=config.non_blocking)

        with autocast(device_type=config.device, enabled=use_amp, dtype=amp_dtype):
            outputs = model(input_ids, attention_mask, ast_seq, struct_feat)
            loss, step_info = dro_loss_fn.compute(outputs, labels, group_ids)
            loss_scaled = loss / config.grad_accum_steps

        scaler.scale(loss_scaled).backward()

        if (batch_idx + 1) % config.grad_accum_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step_ref[0] += 1

        total_loss_acc += float(loss.item())
        num_batches    += 1

        # -- Per-step group weight logging every `log_every` steps ------
        if (batch_idx + 1) % config.log_every == 0:
            lr   = scheduler.get_last_lr()[0]
            snap = dro_loss_fn.get_weight_snapshot()
            worst = snap["_worst_group"]
            weight_str = "  ".join(
                f"{_group_name(g)}={snap[_group_name(g)]:.3f}"
                + (" ***" if _group_name(g) == worst else "")
                for g in range(NUM_GROUPS)
            )
            logger.info(
                "Epoch %d | Step %d/%d | Loss=%.4f | LR=%.2e",
                epoch + 1, batch_idx + 1, len(loader),
                total_loss_acc / num_batches, lr,
            )
            logger.info("  Group weights: %s", weight_str)
            logger.info(
                "  DRO=%.4f  Neural=%.4f  Spectral=%.4f",
                step_info.get("dro_loss", 0.0),
                step_info.get("neural_loss", 0.0),
                step_info.get("spectral_loss", 0.0),
            )

    avg_loss = total_loss_acc / max(num_batches, 1)
    return {"avg_loss": avg_loss, "group_weights": dro_loss_fn.get_weight_snapshot()}


# ============================================================================
# Evaluation
# ============================================================================

@torch.no_grad()
def evaluate(
    model: GroupDROSpectralModel,
    loader: DataLoader,
    config: GroupDROConfig,
    split_name: str = "Val",
    use_amp: bool = False,
    amp_dtype = None,
) -> Dict[str, Any]:
    """Evaluate model; returns dict with macro_f1, per-group breakdown, etc."""
    model.eval()
    all_preds:    List[int] = []
    all_labels:   List[int] = []
    all_group_ids: List[int] = []
    total_loss = 0.0
    n_batches  = 0

    for batch in loader:
        input_ids      = batch["input_ids"].to(config.device)
        attention_mask = batch["attention_mask"].to(config.device)
        ast_seq        = batch["ast_seq"].to(config.device)
        struct_feat    = batch["struct_feat"].to(config.device)
        labels         = batch["label"].to(config.device)
        group_ids      = batch["group_id"]

        with autocast(device_type=config.device, enabled=use_amp, dtype=amp_dtype):
            outputs = model(input_ids, attention_mask, ast_seq, struct_feat)

        preds = outputs["logits"].argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_group_ids.extend(group_ids.numpy().tolist())
        total_loss += F.cross_entropy(outputs["logits"], labels).item()
        n_batches  += 1

    preds_arr  = np.array(all_preds)
    labels_arr = np.array(all_labels)
    gids_arr   = np.array(all_group_ids)

    macro_f1    = float(f1_score(labels_arr, preds_arr, average="macro",    zero_division=0))
    weighted_f1 = float(f1_score(labels_arr, preds_arr, average="weighted", zero_division=0))
    avg_loss    = total_loss / max(n_batches, 1)

    logger.info(
        "%s | Loss=%.4f | Macro-F1=%.4f | Weighted-F1=%.4f",
        split_name, avg_loss, macro_f1, weighted_f1,
    )

    # Per-group breakdown
    per_group: Dict[str, Dict[str, float]] = {}
    for g in range(NUM_GROUPS):
        gname = _group_name(g)
        mask  = gids_arr == g
        if mask.sum() < 5:
            continue
        gf1 = float(f1_score(labels_arr[mask], preds_arr[mask], average="macro", zero_division=0))
        per_group[gname] = {"n": int(mask.sum()), "macro_f1": gf1}
        logger.info("  Group %s: n=%d  macro_f1=%.4f", gname, mask.sum(), gf1)

    if split_name in {"Test", "OOD"}:
        report = classification_report(labels_arr, preds_arr, digits=4, zero_division=0)
        logger.info("\n%s Classification Report:\n%s", split_name, report)

    return {
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "avg_loss":    avg_loss,
        "per_group":   per_group,
        "preds":       preds_arr,
        "labels":      labels_arr,
        "group_ids":   gids_arr,
    }


def _breakdown_by_dim(eval_out: Dict[str, Any], test_data: Dataset, dim: str) -> Dict[str, Any]:
    preds  = eval_out["preds"]
    labels = eval_out["labels"]
    n      = min(len(preds), len(test_data))
    dim_vals = test_data[dim][:n]
    unique   = sorted(set(dim_vals))
    result: Dict[str, Any] = {}
    for val in unique:
        mask = np.array([v == val for v in dim_vals])
        if mask.sum() < 10:
            continue
        mf1 = float(f1_score(labels[mask], preds[mask], average="macro",    zero_division=0))
        wf1 = float(f1_score(labels[mask], preds[mask], average="weighted", zero_division=0))
        result[val] = {"n": int(mask.sum()), "macro_f1": mf1, "weighted_f1": wf1}
        logger.info("  %-12s %s: n=%6d  macro=%.4f  weighted=%.4f", dim, val, mask.sum(), mf1, wf1)
    return result


# ============================================================================
# Single experiment runner
# ============================================================================

def _run_experiment(
    config: GroupDROConfig,
    train_data: Dataset,
    val_data:   Dataset,
    test_data:  Dataset,
    num_classes: int,
    exp_tag: str,
) -> Dict[str, Any]:
    """Core training + evaluation loop shared by IID and OOD runners."""
    set_seed(config.seed)

    precision = config.precision.lower()
    if precision == "auto":
        precision = "bf16" if config.device == "cuda" else "fp32"
    use_amp   = config.device == "cuda" and precision in ("bf16", "fp16")
    amp_dtype = torch.bfloat16 if precision == "bf16" else torch.float16

    tokenizer   = AutoTokenizer.from_pretrained(config.encoder_name)
    train_loader, val_loader, test_loader = _make_loaders(train_data, val_data, test_data, tokenizer, config)

    model = GroupDROSpectralModel(config, num_classes).to(config.device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("GroupDROSpectralModel | params=%s | classes=%d | device=%s | precision=%s",
                f"{total_params:,}", num_classes, config.device, precision)

    # GroupDRO loss
    dro_loss_fn = GroupDROLoss(
        num_groups=NUM_GROUPS,
        eta=config.dro_eta,
        gamma=config.focal_gamma,
        lambda_neural=config.lambda_neural,
        lambda_spectral=config.lambda_spectral,
        device=config.device,
    )

    # Optimizer: dual LR
    encoder_params = list(model.token_encoder.parameters())
    head_params    = [p for n, p in model.named_parameters() if "token_encoder" not in n]
    optimizer = torch.optim.AdamW([
        {"params": encoder_params, "lr": config.lr_encoder, "weight_decay": config.weight_decay},
        {"params": head_params,    "lr": config.lr_heads,   "weight_decay": config.weight_decay},
    ])

    total_steps  = len(train_loader) * config.epochs // max(config.grad_accum_steps, 1)
    warmup_steps = int(total_steps * config.warmup_ratio)
    scheduler    = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    scaler       = GradScaler(enabled=(use_amp and precision == "fp16"))

    save_dir = os.path.join(config.save_dir, exp_tag)
    os.makedirs(save_dir, exist_ok=True)

    best_val_f1   = 0.0
    global_step   = [0]   # mutable counter threaded through train_epoch_groupdro

    for epoch in range(config.epochs):
        logger.info("\n%s  Epoch %d/%d  %s", "="*40, epoch + 1, config.epochs, "="*40)
        epoch_stats = train_epoch_groupdro(
            model, train_loader, optimizer, scheduler,
            dro_loss_fn, config, epoch, scaler, use_amp, amp_dtype, global_step,
        )
        logger.info(
            "Epoch %d | avg_loss=%.4f | worst_group=%s (w=%.3f)",
            epoch + 1,
            epoch_stats["avg_loss"],
            epoch_stats["group_weights"].get("_worst_group", "?"),
            epoch_stats["group_weights"].get("_worst_weight", 0.0),
        )

        val_out = evaluate(model, val_loader, config, "Val", use_amp, amp_dtype)
        val_f1  = val_out["macro_f1"]

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            ckpt_path   = os.path.join(save_dir, "best.pt")
            torch.save({
                "model_state_dict":  model.state_dict(),
                "best_val_f1":       best_val_f1,
                "group_weights":     dro_loss_fn.group_weights,
                "global_step":       global_step[0],
                "epoch":             epoch,
            }, ckpt_path)
            logger.info("*** New best Val Macro-F1=%.4f  saved to %s ***", best_val_f1, ckpt_path)

        # Intra-epoch eval at eval_every steps (already handled inline; epoch-end is definitive)

    # Load best checkpoint for final test evaluation
    best_ckpt = os.path.join(save_dir, "best.pt")
    if os.path.exists(best_ckpt):
        ckpt = torch.load(best_ckpt, map_location=config.device, weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
        dro_loss_fn.group_weights = ckpt.get("group_weights", dro_loss_fn.group_weights)
        logger.info("Loaded best checkpoint (val_f1=%.4f)", ckpt.get("best_val_f1", 0.0))

    logger.info("\n%s\nFINAL TEST EVALUATION: %s\n%s", "="*60, exp_tag, "="*60)
    test_out = evaluate(model, test_loader, config, "Test", use_amp, amp_dtype)

    results: Dict[str, Any] = {
        "test_f1":           test_out["macro_f1"],
        "test_weighted_f1":  test_out["weighted_f1"],
        "best_val_f1":       best_val_f1,
        "per_group":         test_out["per_group"],
        "final_group_weights": {k: v for k, v in dro_loss_fn.get_weight_snapshot().items()
                                if not k.startswith("_")},
    }

    if config.eval_breakdown:
        logger.info("--- Breakdown by language ---")
        results["by_language"] = _breakdown_by_dim(test_out, test_data, "language")
        logger.info("--- Breakdown by source ---")
        results["by_source"]   = _breakdown_by_dim(test_out, test_data, "source")
        logger.info("--- Breakdown by generator ---")
        results["by_generator"] = _breakdown_by_dim(test_out, test_data, "generator")

    if config.task == "author":
        cm = confusion_matrix(test_out["labels"], test_out["preds"])
        results["confusion_matrix"] = cm.tolist()

    return results


# ============================================================================
# IID Suite
# ============================================================================

def run_iid_suite(config: GroupDROConfig) -> Dict[str, Any]:
    """Run both IID binary and IID author tasks."""
    suite: Dict[str, Any] = {}

    for task in ("binary", "author"):
        logger.info("\n%s\n[IID] task=%s\n%s", "#"*70, task, "#"*70)
        cfg = GroupDROConfig(**{**config.__dict__, "task": task})
        cfg = apply_hardware_profile(cfg)

        try:
            train_data, val_data, test_data, num_classes, _ = _load_iid_splits(cfg)
            results = _run_experiment(cfg, train_data, val_data, test_data, num_classes, f"iid_{task}")
            suite[f"iid_{task}"] = results
        except Exception as exc:
            logger.error("[IID/%s] FAILED: %s", task, exc, exc_info=True)
            suite[f"iid_{task}"] = {"error": str(exc)}

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    return suite


# ============================================================================
# OOD Suite
# ============================================================================

def run_ood_suite(config: GroupDROConfig) -> Dict[str, Any]:
    """Run all three OOD LOO evaluations (generator, language, source)."""
    suite: Dict[str, Any] = {}

    ood_scenarios: Dict[str, List[str]] = {
        "generator": ["codellama", "gpt", "llama3.1", "nxcode", "qwen1.5"],
        "language":  ["cpp", "java", "python"],
        "source":    ["cf", "gh", "lc"],
    }

    for hold_field, hold_values in ood_scenarios.items():
        suite_key = f"ood_{hold_field}"
        suite[suite_key] = {}

        for hold_val in hold_values:
            tag = f"ood_{hold_field}_{hold_val}"
            logger.info("\n%s\n[OOD] %s=%s\n%s", "#"*70, hold_field, hold_val, "#"*70)

            # OOD always uses binary task
            cfg = GroupDROConfig(**{**config.__dict__, "task": "binary"})
            cfg = apply_hardware_profile(cfg)

            try:
                train_data, val_data, test_data, num_classes = _load_ood_splits(
                    cfg, hold_field, hold_val
                )
                results = _run_experiment(cfg, train_data, val_data, test_data, num_classes, tag)
                results["hold_out_field"] = hold_field
                results["hold_out_value"] = hold_val
                suite[suite_key][hold_val] = results
            except Exception as exc:
                logger.error("[OOD/%s=%s] FAILED: %s", hold_field, hold_val, exc, exc_info=True)
                suite[suite_key][hold_val] = {"error": str(exc)}

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Summary table
        _log_ood_summary(f"OOD-{hold_field.upper()}", suite[suite_key])

    return suite


def _log_ood_summary(title: str, results: Dict[str, Any]):
    logger.info("\n%s\n%s SUMMARY\n%s", "="*70, title, "="*70)
    logger.info("| held-out | macro_f1 | weighted_f1 | best_val_f1 |")
    logger.info("|----------|----------|-------------|-------------|")
    for key, stats in results.items():
        if "error" in stats:
            logger.info("| %-8s | ERROR    |      -      |      -      |", key)
        else:
            logger.info(
                "| %-8s | %.4f   | %.4f      | %.4f      |",
                key,
                stats.get("test_f1", 0.0),
                stats.get("test_weighted_f1", 0.0),
                stats.get("best_val_f1", 0.0),
            )


# ============================================================================
# Final summary printer
# ============================================================================

def _log_final_summary(all_results: Dict[str, Any]):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("\n%s\nGROUPDRO-SPECTRALCODE BENCHMARK COMPLETE | %s\n%s", "="*70, ts, "="*70)

    # IID
    logger.info("\n=== IID RESULTS ===")
    for task_key in ("iid_binary", "iid_author"):
        r = all_results.get(task_key, {})
        if "error" in r:
            logger.info("  %-12s: ERROR – %s", task_key, r["error"])
        else:
            logger.info(
                "  %-12s: macro_f1=%.4f  weighted_f1=%.4f  val_f1=%.4f",
                task_key,
                r.get("test_f1", 0.0), r.get("test_weighted_f1", 0.0), r.get("best_val_f1", 0.0),
            )

    # OOD
    for field in ("generator", "language", "source"):
        suite_key = f"ood_{field}"
        ood_block = all_results.get(suite_key, {})
        if not isinstance(ood_block, dict):
            continue
        logger.info("\n=== OOD %s RESULTS ===", field.upper())
        for held, stats in ood_block.items():
            if isinstance(stats, dict) and "test_f1" in stats:
                logger.info(
                    "  held=%s: macro_f1=%.4f  weighted_f1=%.4f",
                    held, stats["test_f1"], stats.get("test_weighted_f1", 0.0),
                )

    # JSON output
    def _make_serializable(obj):
        if isinstance(obj, (np.ndarray, torch.Tensor)):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    safe: Dict[str, Any] = {}
    for k, v in all_results.items():
        if isinstance(v, dict):
            # Strip non-serializable tensors / arrays from sub-dicts
            safe[k] = json.loads(json.dumps(v, default=_make_serializable))
        else:
            safe[k] = str(v)

    payload = {
        "timestamp": ts,
        "method":    "GroupDRO-SpectralCode",
        "exp_id":    "Exp15",
        "results":   safe,
    }
    print("SUITE_RESULTS_JSON=" + json.dumps(payload, ensure_ascii=True, default=_make_serializable))


# ============================================================================
# main
# ============================================================================

def main():
    # ------------------------------------------------------------------
    # Run mode:
    #   "full"      → IID (binary + author) + OOD (generator/language/source)
    #   "iid_only"  → IID binary + author only
    #   "ood_only"  → OOD suite only (binary task)
    # ------------------------------------------------------------------
    RUN_MODE            = "full"
    RUN_PREFLIGHT       = True

    config = GroupDROConfig(
        max_train_samples = 100_000,
        max_val_samples   = 20_000,
        max_test_samples  = 50_000,
        eval_breakdown    = True,
        dro_eta           = 0.01,
        lambda_neural     = 0.3,
        lambda_spectral   = 0.3,
        focal_gamma       = 2.0,
        epochs            = 3,
    )
    config = apply_hardware_profile(config)

    logger.info("\n%s", "="*70)
    logger.info("Exp15: GroupDRO-SpectralCode | CoDET-M4 Full Benchmark")
    logger.info("GPU=%s | device=%s | precision=%s", _get_gpu_name(), config.device, config.precision)
    logger.info("batch=%d x accum=%d | epochs=%d | dro_eta=%.3f",
                config.batch_size, config.grad_accum_steps, config.epochs, config.dro_eta)
    logger.info("Groups (%d): %s", NUM_GROUPS, [_group_name(g) for g in range(NUM_GROUPS)])
    logger.info("%s\n", "="*70)

    if RUN_PREFLIGHT:
        try:
            preflight_check(config)
        except PreflightError as e:
            logger.error("PREFLIGHT FAILED: %s", e)
            raise SystemExit(1)

    all_results: Dict[str, Any] = {}

    try:
        if RUN_MODE in ("full", "iid_only"):
            iid_results = run_iid_suite(config)
            all_results.update(iid_results)

        if RUN_MODE in ("full", "ood_only"):
            ood_results = run_ood_suite(config)
            all_results.update(ood_results)

        if RUN_MODE not in ("full", "iid_only", "ood_only"):
            raise ValueError(f"Unknown RUN_MODE: {RUN_MODE!r}")

    except KeyboardInterrupt:
        logger.warning("Run interrupted by user. Printing partial results.")
    except Exception as exc:
        logger.error("Unexpected error: %s", exc, exc_info=True)
    finally:
        _log_final_summary(all_results)


if __name__ == "__main__":
    main()
