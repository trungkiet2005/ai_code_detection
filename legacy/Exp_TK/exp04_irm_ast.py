"""
AST-IRM: AST-driven Invariant Risk Minimization for
Cross-Language Robust AI-Generated Code Detection

Key Innovation:
- Treats programming languages as distinct environments for IRM
- Forces model to learn causal, language-agnostic detection features
- Penalizes gradients that vary across language environments
- Directly solves cross-language performance degradation
- Uses IRMv1 penalty with environment-aware batching

Loss: L = Sigma_e L_CE(e) + lambda_irm * Sigma_e ||grad_{w|w=1} L_CE(e) * w||^2

Reference: Tier B high-upside method from research portfolio
Target: NeurIPS 2026 ORAL
"""

import os
import json
import math
import random
import logging
import warnings
import importlib.util
import subprocess
import sys
from datetime import datetime
from typing import Optional, Dict, List, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter


def ensure_runtime_dependencies():
    """
    Install missing runtime dependencies on-the-fly (Kaggle-friendly).
    This runs before third-party imports so the script is standalone.
    """
    required = [
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("datasets", "datasets"),
        ("transformers", "transformers"),
        ("sklearn", "scikit-learn"),
        ("accelerate", "accelerate"),
        ("tree_sitter", "tree-sitter"),
        ("tree_sitter_languages", "tree-sitter-languages"),
    ]
    missing = [pip_name for import_name, pip_name in required if importlib.util.find_spec(import_name) is None]
    if not missing:
        return

    print(f"[bootstrap] Installing missing packages: {missing}")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing])
    except Exception as e:
        raise RuntimeError(
            f"Failed to auto-install dependencies {missing}. "
            f"Please install manually via pip before running."
        ) from e


ensure_runtime_dependencies()

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
try:
    from torch.amp import autocast as _autocast, GradScaler  # PyTorch >= 2.0
    _NEW_AMP = True
except ImportError:
    from torch.cuda.amp import autocast as _autocast, GradScaler  # PyTorch < 2.0
    _NEW_AMP = False

def autocast(device_type="cuda", enabled=True, dtype=None):
    """Compatible autocast wrapper for PyTorch < 2.0 and >= 2.0."""
    if _NEW_AMP:
        if dtype is None:
            return _autocast(device_type=device_type, enabled=enabled)
        return _autocast(device_type=device_type, enabled=enabled, dtype=dtype)
    else:
        # Old API: only supports CUDA, ignore device_type
        return _autocast(enabled=enabled)

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
from sklearn.metrics import f1_score, classification_report

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class PreflightError(RuntimeError):
    """Raised when preflight checks fail and training must stop."""


def _get_gpu_name() -> str:
    if not torch.cuda.is_available():
        return "cpu"
    try:
        return torch.cuda.get_device_name(0)
    except Exception:
        return "cuda"


def apply_hardware_profile(config: "Config") -> "Config":
    """
    Apply runtime tuning for available hardware.
    On Kaggle H100 80GB, increase throughput defaults safely.
    """
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

    # H100 profile: prioritize throughput while staying stable for seq_len=512.
    config.precision = "bf16" if config.precision == "auto" else config.precision
    config.batch_size = max(config.batch_size, 64)
    config.grad_accum_steps = 1
    config.num_workers = max(config.num_workers, 8)
    config.prefetch_factor = max(config.prefetch_factor, 4)
    config.log_every = max(config.log_every, 200)
    config.eval_every = max(config.eval_every, 2000)

    logger.info(
        "Applied H100 profile | precision=%s | batch_size=%d | accum=%d | workers=%d | prefetch=%d",
        config.precision, config.batch_size, config.grad_accum_steps, config.num_workers, config.prefetch_factor,
    )
    return config

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    # Task
    task: str = "T1"  # T1, T2, T3, T4 (T4 is Droid-only 4-class mode)
    benchmark: str = "aicd"  # aicd, droid

    # Model
    encoder_name: str = "answerdotai/ModernBERT-base"
    max_length: int = 512
    z_style_dim: int = 256
    z_content_dim: int = 256
    gnn_hidden_dim: int = 128
    gnn_layers: int = 2
    num_ast_node_types: int = 256  # vocabulary size for AST node types
    ast_embed_dim: int = 64
    ast_seq_len: int = 128  # max AST node sequence length

    # Task-specific
    num_classes: Dict[str, int] = field(default_factory=lambda: {
        "T1": 2,   # human vs machine
        "T2": 12,  # 11 families + human
        "T3": 4,   # human, machine, hybrid, adversarial
        "T4": 4,   # Droid-only: human, machine, refined, adversarial
    })
    num_languages: int = 9  # auxiliary task
    num_domains: int = 3    # algorithmic, research, general-purpose

    # Training
    epochs: int = 3  # aligned with paper-style baseline setting
    batch_size: int = 32  # auto-upscaled for H100 profile
    grad_accum_steps: int = 2  # effective batch = 64 (or 64x1 on H100)
    lr_encoder: float = 2e-5
    lr_heads: float = 1e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Loss weights
    lambda_disentangle: float = 0.1
    lambda_contrastive: float = 0.3
    lambda_reconstruct: float = 0.1

    # Contrastive learning
    temperature: float = 0.07
    prototype_momentum: float = 0.99

    # IRM
    lambda_irm: float = 1.0  # IRM penalty weight
    irm_anneal_epochs: int = 1  # epochs before IRM penalty kicks in
    irm_penalty_weight_max: float = 1e4  # max IRM penalty
    num_environments: int = 9  # number of language environments

    # Data
    max_train_samples: int = 100_000  # subsample for Kaggle feasibility
    max_val_samples: int = 20_000
    max_test_samples: int = 50_000
    droid_dataset_id: str = "project-droid/DroidCollection"
    droid_config_name: str = "default"
    droid_split_map: Dict[str, str] = field(default_factory=lambda: {
        "train": "train",
        "validation": "dev",
        "test": "test",
    })
    num_workers: int = 2
    prefetch_factor: int = 2

    # Misc
    seed: int = 42
    precision: str = "auto"  # auto, bf16, fp16, fp32
    auto_h100_profile: bool = True
    require_tree_sitter: bool = True  # strict preflight for paper-grade AST features
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    pin_memory: bool = True
    non_blocking: bool = True
    save_dir: str = "./irm_ast_checkpoints"
    log_every: int = 100
    eval_every: int = 1000


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# AST Feature Extraction (lightweight, no tree-sitter dependency fallback)
# ============================================================================

# AST node type vocabulary (language-agnostic structural tokens)
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


def try_load_tree_sitter():
    """Try to import tree-sitter. Returns parser function or None."""
    try:
        import tree_sitter_languages  # type: ignore[reportMissingImports]

        LANG_MAP = {
            "python": "python", "java": "java", "cpp": "cpp", "c": "c",
            "go": "go", "php": "php", "c_sharp": "c_sharp",
            "javascript": "javascript", "rust": "rust",
        }

        def parse_ast(code: str, lang: str = "python") -> List[int]:
            """Parse code into AST node type sequence."""
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
                return fallback_ast_extract(code)

        return parse_ast
    except ImportError:
        logger.warning("tree-sitter-languages not found, using regex-based AST extraction")
        return None


def fallback_ast_extract(code: str) -> List[int]:
    """Regex-based structural feature extraction as fallback."""
    import re
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


# Global AST parser
_ast_parser = try_load_tree_sitter()


def extract_ast_sequence(code: str, max_len: int = 128) -> List[int]:
    """Extract AST node type sequence from code."""
    if _ast_parser is not None:
        seq = _ast_parser(code)
    else:
        seq = fallback_ast_extract(code)

    # Pad or truncate
    if len(seq) > max_len:
        seq = seq[:max_len]
    else:
        seq = seq + [AST_NODE_VOCAB["PAD"]] * (max_len - len(seq))
    return seq


def extract_structural_features(code: str) -> List[float]:
    """Extract numeric structural features from code (graph-level proxy)."""
    lines = code.split("\n")
    num_lines = len(lines)
    avg_line_len = np.mean([len(l) for l in lines]) if lines else 0
    max_line_len = max([len(l) for l in lines]) if lines else 0

    # Indentation patterns (proxy for control flow depth)
    indents = []
    for line in lines:
        stripped = line.lstrip()
        if stripped:
            indents.append(len(line) - len(stripped))
    avg_indent = np.mean(indents) if indents else 0
    max_indent = max(indents) if indents else 0
    indent_variance = np.var(indents) if indents else 0

    # Code complexity proxies
    num_functions = code.count("def ") + code.count("function ") + code.count("func ") + code.count("fn ")
    num_classes = code.count("class ") + code.count("struct ") + code.count("interface ")
    num_loops = code.count("for ") + code.count("while ") + code.count("foreach ")
    num_conditionals = code.count("if ") + code.count("else ") + code.count("elif ") + code.count("else if")
    num_returns = code.count("return ")
    num_comments = code.count("//") + code.count("#") + code.count("/*")
    num_imports = code.count("import ") + code.count("include ") + code.count("require ") + code.count("using ")
    num_try_catch = code.count("try") + code.count("catch") + code.count("except")

    # Naming style features (key cross-language signal from paper)
    import re
    identifiers = re.findall(r'\b[a-zA-Z_]\w*\b', code)
    num_snake_case = sum(1 for i in identifiers if '_' in i and i.islower())
    num_camel_case = sum(1 for i in identifiers if any(c.isupper() for c in i[1:]) and '_' not in i)
    num_single_char = sum(1 for i in identifiers if len(i) == 1)
    avg_identifier_len = np.mean([len(i) for i in identifiers]) if identifiers else 0

    # Whitespace patterns
    empty_lines = sum(1 for l in lines if not l.strip())
    empty_line_ratio = empty_lines / max(num_lines, 1)

    # Character distribution
    alpha_ratio = sum(c.isalpha() for c in code) / max(len(code), 1)
    digit_ratio = sum(c.isdigit() for c in code) / max(len(code), 1)
    space_ratio = sum(c.isspace() for c in code) / max(len(code), 1)

    features = [
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
    return features


STRUCTURAL_FEATURE_DIM = 22  # number of features from extract_structural_features


# ============================================================================
# Dataset with Environment (Language) Labels for IRM
# ============================================================================

class IRMDataset(Dataset):
    """Dataset with environment (language) labels for IRM."""

    def __init__(self, data, tokenizer, max_length=512, ast_seq_len=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.ast_seq_len = ast_seq_len

        # Try to extract language info
        self.lang_map = {}
        self.lang_counter = 0

    def _get_language_id(self, item):
        """Extract or infer language environment from data item."""
        # Try common field names
        lang = None
        for field_name in ["language", "lang", "programming_language", "Language"]:
            if field_name in item:
                lang = item[field_name]
                break

        if lang is None:
            # Fallback: infer from code heuristics
            code = item.get("code", "")
            lang = self._infer_language(code)

        if lang not in self.lang_map:
            self.lang_map[lang] = self.lang_counter
            self.lang_counter += 1

        return self.lang_map[lang]

    def _infer_language(self, code: str) -> str:
        """Simple language inference heuristic."""
        if "def " in code and ":" in code and "import " in code:
            return "python"
        if "public static void main" in code or "System.out" in code:
            return "java"
        if "#include" in code and ("{" in code):
            return "cpp"
        if "func " in code and "package " in code:
            return "go"
        if "fn " in code and "let mut" in code:
            return "rust"
        if "function " in code and ("var " in code or "const " in code or "let " in code):
            return "javascript"
        if "<?php" in code or "$" in code:
            return "php"
        if "using System" in code or "namespace " in code:
            return "csharp"
        return "unknown"

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        code = item["code"]
        label = item["label"]
        env_id = self._get_language_id(item)

        encoding = self.tokenizer(
            code,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        ast_seq = extract_ast_sequence(code, self.ast_seq_len)
        struct_feat = extract_structural_features(code)

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "ast_seq": torch.tensor(ast_seq, dtype=torch.long),
            "struct_feat": torch.tensor(struct_feat, dtype=torch.float32),
            "label": torch.tensor(label, dtype=torch.long),
            "env_id": torch.tensor(env_id, dtype=torch.long),
        }


# ============================================================================
# Data Loading (unchanged from exp00)
# ============================================================================

def load_aicd_data(config: Config):
    """Load AICD-Bench dataset from HuggingFace."""
    logger.info(f"Loading AICD-Bench task {config.task}...")

    ds = load_dataset("AICD-bench/AICD-Bench", name=config.task)

    train_data = ds["train"]
    val_data = ds["validation"]
    test_data = ds["test"]

    # Subsample for Kaggle feasibility
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

    # Get number of unique labels
    all_labels = set(train_data["label"])
    num_classes = len(all_labels)
    logger.info(f"Number of classes: {num_classes}, Labels: {sorted(all_labels)}")

    return train_data, val_data, test_data, num_classes


def _sample_dataset(dataset, max_samples: int):
    """Randomly sample at most max_samples rows from a HF dataset."""
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    indices = random.sample(range(len(dataset)), max_samples)
    return dataset.select(indices)


def _supports_task_for_benchmark(benchmark: str, task: str) -> bool:
    """Return whether a (benchmark, task) pair is supported."""
    supported = {
        "aicd": {"T1", "T2", "T3"},
        "droid": {"T1", "T3", "T4"},
    }
    return task in supported.get(benchmark, set())


def _is_droid_adversarial_row(row: Dict[str, object]) -> bool:
    """
    Detect adversarial samples in Droid rows from explicit textual markers.
    This is intentionally strict to avoid inventing adversarial labels.
    """
    marker_fields = ["Label", "Generation_Mode", "Source", "Generator"]
    text_blob = " ".join(str(row.get(k, "")) for k in marker_fields).upper()
    markers = ("ADVERSARIAL", "EVASIVE", "JAILBREAK", "ATTACK")
    return any(m in text_blob for m in markers)


def _map_droid_label_to_task(row: Dict[str, object], task: str) -> int:
    """Map DroidCollection rows to task labels."""
    normalized = str(row.get("Label", "")).upper()
    if task == "T1":
        return 0 if normalized == "HUMAN_GENERATED" else 1
    if task == "T3":
        # Droid 3-class: human / machine-generated / machine-refined
        if normalized == "HUMAN_GENERATED":
            return 0
        if normalized == "MACHINE_GENERATED":
            return 1
        if normalized == "MACHINE_REFINED":
            return 2
        return -1
    if task == "T4":
        # Droid 4-class: human / machine-generated / machine-refined / adversarial
        if normalized == "HUMAN_GENERATED":
            return 0
        if normalized == "MACHINE_GENERATED":
            return 1
        if _is_droid_adversarial_row(row):
            return 3
        if normalized == "MACHINE_REFINED":
            return 2
        return -1
    # T2 family IDs are benchmark-specific, so we do not merge Droid by default.
    return -1


def load_droid_data(config: Config):
    """Load DroidCollection and convert to common {code, label} schema."""
    if not _supports_task_for_benchmark("droid", config.task):
        raise ValueError(
            f"Task {config.task} is unsupported for Droid benchmark. Use one of: T1 (2-class), T3 (3-class), T4 (4-class)."
        )

    def _convert_split(split_name: str, max_samples: int):
        source_split = config.droid_split_map[split_name]
        logger.info(f"Loading Droid split '{source_split}' for {split_name}...")
        ds = load_dataset(config.droid_dataset_id, name=config.droid_config_name, split=source_split)
        ds = _sample_dataset(ds, max_samples)

        def _convert_row(row):
            mapped_label = _map_droid_label_to_task(row, config.task)
            return {
                "code": row.get("Code", "") or "",
                "label": mapped_label,
            }

        converted = ds.map(_convert_row, remove_columns=ds.column_names)
        converted = converted.filter(lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)
        return converted

    train_data = _convert_split("train", config.max_train_samples)
    val_data = _convert_split("validation", config.max_val_samples)
    test_data = _convert_split("test", config.max_test_samples)

    labels = set(train_data["label"])
    num_classes = len(labels)
    if config.task == "T4" and 3 not in labels:
        raise ValueError(
            "Droid 4-class mode requires adversarial samples, but none were detected in the loaded train split. "
            "The public default split appears 3-class only unless an adversarial subset is provided."
        )
    logger.info(
        f"Droid sizes - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}"
    )
    logger.info(f"Droid classes: {num_classes}, Labels: {sorted(labels)}")
    return train_data, val_data, test_data, num_classes


def _quick_code_stats(dataset, code_key: str = "code", sample_size: int = 512) -> Dict[str, float]:
    """Compute lightweight code stats for sanity checks."""
    n = min(len(dataset), sample_size)
    if n == 0:
        return {"samples": 0, "avg_chars": 0.0, "avg_lines": 0.0, "empty_ratio": 1.0}
    sample = dataset.select(range(n))
    codes = sample[code_key]
    lengths = [len(c) for c in codes]
    lines = [c.count("\n") + 1 for c in codes]
    empty = sum(1 for c in codes if len(c.strip()) == 0)
    return {
        "samples": n,
        "avg_chars": float(np.mean(lengths)),
        "avg_lines": float(np.mean(lines)),
        "empty_ratio": float(empty / n),
    }


def preflight_single_benchmark(config: Config) -> Dict[str, object]:
    """
    Run strict preflight checks before training one benchmark.
    Raises PreflightError on failure.
    """
    logger.info("\n" + "=" * 70)
    logger.info(f"PREFLIGHT START | benchmark={config.benchmark} | task={config.task}")
    logger.info("=" * 70)

    if config.require_tree_sitter and _ast_parser is None:
        raise PreflightError(
            "tree-sitter-languages is unavailable. Install tree-sitter + tree-sitter-languages "
            "or disable strict mode by setting require_tree_sitter=False."
        )
    if not _supports_task_for_benchmark(config.benchmark, config.task):
        raise PreflightError(
            f"Unsupported task/benchmark pair: benchmark={config.benchmark}, task={config.task}."
        )

    if config.benchmark == "aicd":
        train_data, val_data, test_data, num_classes = load_aicd_data(config)
    elif config.benchmark == "droid":
        train_data, val_data, test_data, num_classes = load_droid_data(config)
    else:
        raise PreflightError(f"Unsupported benchmark: {config.benchmark}")

    # Basic split sanity
    if len(train_data) == 0 or len(val_data) == 0 or len(test_data) == 0:
        raise PreflightError("One or more splits are empty.")
    if num_classes < 2:
        raise PreflightError(f"Invalid class count: {num_classes}")

    # Label distribution sanity (training set)
    label_counts = Counter(train_data["label"])
    if any(v <= 0 for v in label_counts.values()):
        raise PreflightError("Detected invalid label counts.")

    # Code/text sanity
    code_stats = _quick_code_stats(train_data, code_key="code", sample_size=512)
    if code_stats["empty_ratio"] > 0.05:
        raise PreflightError(f"Too many empty code samples: {code_stats['empty_ratio']:.3f}")

    # Tokenizer sanity
    logger.info(f"Preflight tokenizer check: {config.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)
    probe_code = train_data[0]["code"]
    encoded = tokenizer(
        probe_code,
        max_length=config.max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    if encoded["input_ids"].shape[-1] != config.max_length:
        raise PreflightError("Tokenizer output length mismatch.")

    # Feature extraction sanity
    ast_seq = extract_ast_sequence(probe_code, config.ast_seq_len)
    struct_feat = extract_structural_features(probe_code)
    if len(ast_seq) != config.ast_seq_len:
        raise PreflightError("AST sequence length mismatch.")
    if len(struct_feat) != STRUCTURAL_FEATURE_DIM:
        raise PreflightError("Structural feature dimension mismatch.")

    report = {
        "benchmark": config.benchmark,
        "task": config.task,
        "num_classes": int(num_classes),
        "sizes": {
            "train": int(len(train_data)),
            "validation": int(len(val_data)),
            "test": int(len(test_data)),
        },
        "train_label_counts": {str(k): int(v) for k, v in sorted(label_counts.items())},
        "quick_code_stats": code_stats,
        "device": config.device,
        "gpu_name": _get_gpu_name(),
        "encoder_name": config.encoder_name,
        "precision": config.precision,
        "batch_size": config.batch_size,
        "grad_accum_steps": config.grad_accum_steps,
        "num_workers": config.num_workers,
    }

    logger.info(
        f"PREFLIGHT OK | {config.benchmark.upper()} | "
        f"sizes={report['sizes']} | classes={report['num_classes']}"
    )
    return report


def preflight_benchmark_suite(task: str, base_config: Config) -> Dict[str, object]:
    """Run preflight checks for all requested benchmarks and print copyable summary."""
    reports: Dict[str, object] = {}
    planned = [bench for bench in ["aicd", "droid"] if _supports_task_for_benchmark(bench, task)]

    for bench in planned:
        cfg = Config(**vars(base_config))
        cfg.task = task
        cfg.benchmark = bench
        cfg = apply_hardware_profile(cfg)
        reports[bench] = preflight_single_benchmark(cfg)

    # Copy/paste block for research logs
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "task": task,
        "preflight_reports": reports,
    }
    logger.info("\n=== PREFLIGHT_REPORT_START ===")
    logger.info(json.dumps(payload, ensure_ascii=True, indent=2))
    logger.info("=== PREFLIGHT_REPORT_END ===")
    return payload


# ============================================================================
# Model Components
# ============================================================================

class ASTEncoder(nn.Module):
    """Encodes AST node type sequences into a fixed-size representation."""

    def __init__(self, vocab_size: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim, hidden_dim, num_layers=1,
            batch_first=True, bidirectional=True,
        )
        self.proj = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, ast_seq: torch.Tensor) -> torch.Tensor:
        # ast_seq: (B, seq_len)
        x = self.embedding(ast_seq)  # (B, seq_len, embed_dim)
        output, (h_n, _) = self.lstm(x)  # h_n: (2, B, hidden)
        # Concatenate forward and backward final hidden states
        h = torch.cat([h_n[0], h_n[1]], dim=-1)  # (B, hidden*2)
        return self.proj(h)  # (B, hidden)


class StructuralEncoder(nn.Module):
    """Encodes structural/graph-level features."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CrossAttentionFusion(nn.Module):
    """Fuse multi-granularity representations via cross-attention."""

    def __init__(self, token_dim: int, ast_dim: int, struct_dim: int, output_dim: int):
        super().__init__()
        self.token_proj = nn.Linear(token_dim, output_dim)
        self.ast_proj = nn.Linear(ast_dim, output_dim)
        self.struct_proj = nn.Linear(struct_dim, output_dim)

        self.cross_attn = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=4,
            dropout=0.1,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(output_dim)
        self.gate = nn.Linear(output_dim * 3, output_dim)

    def forward(
        self,
        token_repr: torch.Tensor,   # (B, token_dim)
        ast_repr: torch.Tensor,     # (B, ast_dim)
        struct_repr: torch.Tensor,  # (B, struct_dim)
    ) -> torch.Tensor:
        # Project all to same dimension
        t = self.token_proj(token_repr)    # (B, output_dim)
        a = self.ast_proj(ast_repr)        # (B, output_dim)
        s = self.struct_proj(struct_repr)  # (B, output_dim)

        # Stack as sequence for cross-attention: (B, 3, output_dim)
        seq = torch.stack([t, a, s], dim=1)

        # Self-attention across views
        attn_out, _ = self.cross_attn(seq, seq, seq)  # (B, 3, output_dim)
        attn_out = self.norm(attn_out + seq)

        # Gated fusion
        concat = torch.cat([attn_out[:, 0], attn_out[:, 1], attn_out[:, 2]], dim=-1)
        fused = self.gate(concat)  # (B, output_dim)

        return fused


class StyleContentDisentangler(nn.Module):
    """
    Disentangles code representation into style (authorship) and content (semantics).
    Uses variational approach with mutual information minimization.
    """

    def __init__(self, input_dim: int, z_style_dim: int, z_content_dim: int):
        super().__init__()

        # Style encoder: captures HOW code is written
        self.style_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, z_style_dim * 2),  # mean + logvar
        )

        # Content encoder: captures WHAT code does
        self.content_encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim, z_content_dim * 2),  # mean + logvar
        )

        # Reconstruction decoder
        self.decoder = nn.Sequential(
            nn.Linear(z_style_dim + z_content_dim, input_dim),
            nn.LayerNorm(input_dim),
            nn.GELU(),
            nn.Linear(input_dim, input_dim),
        )

        # CLUB MI estimator (Cheng et al., 2020)
        self.mi_estimator = nn.Sequential(
            nn.Linear(z_style_dim, 256),
            nn.ReLU(),
            nn.Linear(256, z_content_dim),
        )

        self.z_style_dim = z_style_dim
        self.z_content_dim = z_content_dim

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu

    def forward(self, h_code: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Encode style
        style_params = self.style_encoder(h_code)
        style_mu, style_logvar = style_params.chunk(2, dim=-1)
        z_style = self.reparameterize(style_mu, style_logvar)

        # Encode content
        content_params = self.content_encoder(h_code)
        content_mu, content_logvar = content_params.chunk(2, dim=-1)
        z_content = self.reparameterize(content_mu, content_logvar)

        # Reconstruct
        h_recon = self.decoder(torch.cat([z_style, z_content], dim=-1))

        # MI estimation (CLUB upper bound)
        content_pred = self.mi_estimator(z_style.detach())

        return {
            "z_style": z_style,
            "z_content": z_content,
            "style_mu": style_mu,
            "style_logvar": style_logvar,
            "content_mu": content_mu,
            "content_logvar": content_logvar,
            "h_recon": h_recon,
            "content_pred": content_pred,
        }

    def compute_mi_loss(self, z_style: torch.Tensor, z_content: torch.Tensor) -> torch.Tensor:
        """CLUB MI upper bound estimation."""
        content_pred = self.mi_estimator(z_style)

        # Positive samples: matched pairs
        positive = -0.5 * ((z_content - content_pred) ** 2).sum(dim=-1)

        # Negative samples: random shuffle
        z_content_shuffle = z_content[torch.randperm(z_content.size(0), device=z_content.device)]
        negative = -0.5 * ((z_content_shuffle - content_pred) ** 2).sum(dim=-1)

        mi_upper = (positive - negative).mean()
        return F.relu(mi_upper)  # Ensure non-negative

    def compute_kl_loss(
        self,
        style_mu: torch.Tensor, style_logvar: torch.Tensor,
        content_mu: torch.Tensor, content_logvar: torch.Tensor,
    ) -> torch.Tensor:
        """KL divergence to regularize latent spaces."""
        kl_style = -0.5 * (1 + style_logvar - style_mu.pow(2) - style_logvar.exp()).sum(dim=-1).mean()
        kl_content = -0.5 * (1 + content_logvar - content_mu.pow(2) - content_logvar.exp()).sum(dim=-1).mean()
        return 0.5 * (kl_style + kl_content)


class PrototypicalContrastiveHead(nn.Module):
    """
    Hierarchical prototypical contrastive learning for family attribution.
    Maintains learnable prototypes for each class.
    """

    def __init__(self, input_dim: int, num_classes: int, temperature: float = 0.07):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, input_dim))
        nn.init.xavier_uniform_(self.prototypes)
        self.temperature = temperature
        self.num_classes = num_classes
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, z_style: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: classification logits
            proto_logits: prototype-based similarity logits
        """
        logits = self.classifier(z_style)

        # Prototype-based classification
        z_norm = F.normalize(z_style, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        proto_logits = torch.mm(z_norm, p_norm.t()) / self.temperature

        return logits, proto_logits

    def contrastive_loss(
        self, z_style: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        """Supervised prototypical contrastive loss."""
        z_norm = F.normalize(z_style, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)

        # Similarity to all prototypes
        sim = torch.mm(z_norm, p_norm.t()) / self.temperature  # (B, num_classes)

        # Cross-entropy with prototype targets
        loss = F.cross_entropy(sim, labels)

        # Pull toward own prototype, push from others (SupCon-style)
        batch_size = z_style.size(0)
        if batch_size < 2:
            return loss

        # Pairwise similarity in batch
        z_sim = torch.mm(z_norm, z_norm.t()) / self.temperature  # (B, B)

        # Mask: same class = positive (no grad, safe for in-place)
        label_mask = (labels.unsqueeze(0) == labels.unsqueeze(1))  # (B, B)
        self_mask = ~torch.eye(batch_size, dtype=torch.bool, device=z_style.device)
        label_mask = label_mask & self_mask  # exclude self

        if label_mask.sum() == 0:
            return loss

        # SupCon loss - avoid in-place ops on tensors in computation graph
        exp_sim = torch.exp(z_sim) * self_mask.float()  # zero out self-similarity

        log_prob = z_sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
        mean_log_prob = (label_mask * log_prob).sum(dim=1) / (label_mask.sum(dim=1) + 1e-8)
        supcon_loss = -mean_log_prob[label_mask.sum(dim=1) > 0].mean()

        return loss + 0.5 * supcon_loss


# ============================================================================
# Full Model: IRMAST
# ============================================================================

class IRMAST(nn.Module):
    """
    AST-IRM: AST-driven Invariant Risk Minimization for
    Cross-Language Robust AI-Generated Code Detection.

    Components:
    1. Multi-Granularity Encoder (token + AST + structural)
    2. Style-Content Disentanglement
    3. Hierarchical Prototypical Contrastive Learning
    4. IRM penalty replaces domain adversarial training

    IRM forces the model to learn causal features that are invariant
    across programming language environments, eliminating the need
    for explicit adversarial alignment.
    """

    def __init__(self, config: Config, num_classes: int):
        super().__init__()
        self.config = config
        self.num_classes = num_classes

        # === Component 1: Multi-Granularity Encoder ===
        # Token-level encoder (ModernBERT)
        self.token_encoder = AutoModel.from_pretrained(config.encoder_name)
        token_hidden_size = self.token_encoder.config.hidden_size

        # AST-level encoder
        self.ast_encoder = ASTEncoder(
            vocab_size=config.num_ast_node_types,
            embed_dim=config.ast_embed_dim,
            hidden_dim=config.gnn_hidden_dim,
        )

        # Structural encoder (graph-level proxy)
        self.struct_encoder = StructuralEncoder(
            input_dim=STRUCTURAL_FEATURE_DIM,
            hidden_dim=config.gnn_hidden_dim,
        )

        # Cross-attention fusion
        fusion_dim = config.z_style_dim + config.z_content_dim  # unified repr dim
        self.fusion = CrossAttentionFusion(
            token_dim=token_hidden_size,
            ast_dim=config.gnn_hidden_dim,
            struct_dim=config.gnn_hidden_dim,
            output_dim=fusion_dim,
        )

        # === Component 2: Style-Content Disentanglement ===
        self.disentangler = StyleContentDisentangler(
            input_dim=fusion_dim,
            z_style_dim=config.z_style_dim,
            z_content_dim=config.z_content_dim,
        )

        # === Component 3: Task Head (Prototypical) ===
        self.proto_head = PrototypicalContrastiveHead(
            input_dim=config.z_style_dim,
            num_classes=num_classes,
            temperature=config.temperature,
        )

        # Content auxiliary head (predict language from z_content for disentanglement)
        self.content_aux_head = nn.Linear(config.z_content_dim, config.num_languages)

        # Focal loss gamma
        self.focal_gamma = 2.0

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        ast_seq: torch.Tensor,
        struct_feat: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # === Multi-Granularity Encoding ===
        # Token-level
        token_out = self.token_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # Use [CLS] token representation
        token_repr = token_out.last_hidden_state[:, 0, :]  # (B, hidden)

        # AST-level
        ast_repr = self.ast_encoder(ast_seq)  # (B, gnn_hidden)

        # Structural-level
        struct_repr = self.struct_encoder(struct_feat)  # (B, gnn_hidden)

        # Fuse
        h_code = self.fusion(token_repr, ast_repr, struct_repr)  # (B, fusion_dim)

        # === Style-Content Disentanglement ===
        disent_out = self.disentangler(h_code)
        z_style = disent_out["z_style"]      # (B, z_style_dim)
        z_content = disent_out["z_content"]  # (B, z_content_dim)

        # === Task Prediction ===
        logits, proto_logits = self.proto_head(z_style)

        # === Content Auxiliary ===
        content_lang_logits = self.content_aux_head(z_content)

        output = {
            "logits": logits,
            "proto_logits": proto_logits,
            "z_style": z_style,
            "z_content": z_content,
            "h_code": h_code,
            "h_recon": disent_out["h_recon"],
            "content_lang_logits": content_lang_logits,
            "style_mu": disent_out["style_mu"],
            "style_logvar": disent_out["style_logvar"],
            "content_mu": disent_out["content_mu"],
            "content_logvar": disent_out["content_logvar"],
        }

        return output


# ============================================================================
# Loss Functions
# ============================================================================

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance."""

    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight, reduction="none")
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_irm_penalty(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    IRMv1 penalty: ||grad_{w|w=1} CE(w * logits, y)||^2
    Measures how much the optimal classifier varies across environments.
    """
    # Scale parameter (dummy scalar = 1.0)
    scale = torch.ones(1, device=logits.device, requires_grad=True)

    # Loss with scale
    loss = F.cross_entropy(logits * scale, labels)

    # Gradient w.r.t. scale
    grad = torch.autograd.grad(loss, scale, create_graph=True)[0]

    # Penalty = gradient norm squared
    return (grad ** 2).sum()


def compute_env_irm_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    env_ids: torch.Tensor,
    lambda_irm: float,
    focal_loss_fn=None,
) -> Dict[str, torch.Tensor]:
    """
    Compute IRM loss across environments.
    L = Sigma_e L_CE(e) + lambda * Sigma_e penalty(e)
    """
    if focal_loss_fn is None:
        focal_loss_fn = FocalLoss(gamma=2.0)

    unique_envs = torch.unique(env_ids)

    total_ce = torch.tensor(0.0, device=logits.device)
    total_penalty = torch.tensor(0.0, device=logits.device)
    num_envs = 0

    for env in unique_envs:
        mask = env_ids == env
        if mask.sum() < 2:  # need at least 2 samples
            continue

        env_logits = logits[mask]
        env_labels = labels[mask]

        # Per-environment CE loss
        env_loss = focal_loss_fn(env_logits, env_labels)
        total_ce = total_ce + env_loss

        # IRM penalty for this environment
        penalty = compute_irm_penalty(env_logits, env_labels)
        total_penalty = total_penalty + penalty

        num_envs += 1

    if num_envs > 0:
        total_ce = total_ce / num_envs
        total_penalty = total_penalty / num_envs

    total = total_ce + lambda_irm * total_penalty

    return {
        "total": total,
        "ce": total_ce,
        "irm_penalty": total_penalty,
    }


def compute_all_losses(
    model: IRMAST,
    outputs: Dict[str, torch.Tensor],
    labels: torch.Tensor,
    env_ids: torch.Tensor,
    config: Config,
    current_irm_lambda: float,
    focal_loss_fn: Optional[FocalLoss] = None,
) -> Dict[str, torch.Tensor]:
    """Compute all loss components including IRM penalty."""

    # 1. IRM-aware task loss (focal loss per environment + IRM penalty)
    if focal_loss_fn is None:
        focal_loss_fn = FocalLoss(gamma=2.0)

    irm_losses = compute_env_irm_loss(
        outputs["logits"], labels, env_ids, current_irm_lambda, focal_loss_fn,
    )
    task_loss = irm_losses["ce"]
    irm_penalty = irm_losses["irm_penalty"]

    # Prototype cross-entropy loss
    proto_loss = F.cross_entropy(outputs["proto_logits"], labels)

    # Contrastive loss
    contrastive_loss = model.proto_head.contrastive_loss(outputs["z_style"], labels)

    # 2. Disentanglement losses
    # MI minimization between z_style and z_content
    mi_loss = model.disentangler.compute_mi_loss(outputs["z_style"], outputs["z_content"])

    # KL regularization
    kl_loss = model.disentangler.compute_kl_loss(
        outputs["style_mu"], outputs["style_logvar"],
        outputs["content_mu"], outputs["content_logvar"],
    )

    # Reconstruction loss
    recon_loss = F.mse_loss(outputs["h_recon"], outputs["h_code"].detach())

    disentangle_loss = mi_loss + 0.01 * kl_loss

    # Total loss (IRM replaces adversarial)
    total_loss = (
        task_loss
        + current_irm_lambda * irm_penalty
        + 0.3 * proto_loss
        + config.lambda_contrastive * contrastive_loss
        + config.lambda_disentangle * disentangle_loss
        + config.lambda_reconstruct * recon_loss
    )

    return {
        "total": total_loss,
        "task": task_loss,
        "irm_penalty": irm_penalty,
        "proto": proto_loss,
        "contrastive": contrastive_loss,
        "disentangle": disentangle_loss,
        "recon": recon_loss,
        "mi": mi_loss,
        "kl": kl_loss,
    }


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    def __init__(self, config: Config, model: IRMAST, train_loader, val_loader, test_loader, class_weights=None):
        self.config = config
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_weights = class_weights.to(config.device) if class_weights is not None else None

        # Separate parameter groups for different learning rates
        encoder_params = list(model.token_encoder.parameters())
        head_params = [p for n, p in model.named_parameters() if "token_encoder" not in n]

        self.optimizer = torch.optim.AdamW([
            {"params": encoder_params, "lr": config.lr_encoder, "weight_decay": config.weight_decay},
            {"params": head_params, "lr": config.lr_heads, "weight_decay": config.weight_decay},
        ])

        total_steps = len(train_loader) * config.epochs // config.grad_accum_steps
        warmup_steps = int(total_steps * config.warmup_ratio)
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer, warmup_steps, total_steps,
        )

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

    def _get_irm_lambda(self, epoch: int) -> float:
        """IRM penalty annealing schedule."""
        if epoch < self.config.irm_anneal_epochs:
            return 0.0  # Pure ERM phase
        progress = (epoch - self.config.irm_anneal_epochs) / max(self.config.epochs - self.config.irm_anneal_epochs, 1)
        return self.config.lambda_irm * min(progress * self.config.irm_penalty_weight_max, self.config.irm_penalty_weight_max)

    def train_epoch(self, epoch: int):
        self.model.train()
        total_losses = defaultdict(float)
        num_batches = 0
        nan_count = 0

        # IRM annealing schedule
        current_irm_lambda = self._get_irm_lambda(epoch)

        self.optimizer.zero_grad()

        for batch_idx, batch in enumerate(self.train_loader):
            # Move to device
            input_ids = batch["input_ids"].to(self.config.device, non_blocking=self.config.non_blocking)
            attention_mask = batch["attention_mask"].to(self.config.device, non_blocking=self.config.non_blocking)
            ast_seq = batch["ast_seq"].to(self.config.device, non_blocking=self.config.non_blocking)
            struct_feat = batch["struct_feat"].to(self.config.device, non_blocking=self.config.non_blocking)
            labels = batch["label"].to(self.config.device, non_blocking=self.config.non_blocking)
            env_ids = batch["env_id"].to(self.config.device, non_blocking=self.config.non_blocking)

            # Forward with mixed precision
            with autocast(device_type=self.config.device, enabled=self.use_amp, dtype=self.amp_dtype):
                outputs = self.model(input_ids, attention_mask, ast_seq, struct_feat, labels)
                losses = compute_all_losses(
                    self.model, outputs, labels, env_ids,
                    self.config, current_irm_lambda, self.focal_loss,
                )
                loss = losses["total"] / self.config.grad_accum_steps

            # NaN detection (IRM can be unstable)
            if torch.isnan(loss) or torch.isinf(loss):
                nan_count += 1
                if nan_count > 10:
                    logger.warning(f"Too many NaN/Inf losses ({nan_count}), reducing IRM lambda")
                    current_irm_lambda *= 0.5
                    nan_count = 0
                self.optimizer.zero_grad()
                continue

            # Backward
            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.config.grad_accum_steps == 0:
                self.scaler.unscale_(self.optimizer)
                # Extra aggressive gradient clipping for IRM stability
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
                self.scheduler.step()
                self.global_step += 1

            # Track losses
            for k, v in losses.items():
                if not (torch.isnan(v) or torch.isinf(v)):
                    total_losses[k] += v.item()
            num_batches += 1

            # Log
            if (batch_idx + 1) % self.config.log_every == 0:
                avg_loss = total_losses["total"] / num_batches
                lr = self.scheduler.get_last_lr()[0]
                irm_pen = total_losses.get("irm_penalty", 0.0) / max(num_batches, 1)
                logger.info(
                    f"Epoch {epoch+1} | Step {batch_idx+1}/{len(self.train_loader)} | "
                    f"Loss: {avg_loss:.4f} | IRM-pen: {irm_pen:.4f} | "
                    f"IRM-lam: {current_irm_lambda:.2e} | LR: {lr:.2e}"
                )

            # Mid-epoch eval
            if self.global_step > 0 and self.global_step % self.config.eval_every == 0:
                val_f1 = self.evaluate(self.val_loader, "Val")
                if val_f1 > self.best_f1:
                    self.best_f1 = val_f1
                    self.save_checkpoint("best")
                    logger.info(f"New best Val F1: {val_f1:.4f}")
                self.model.train()

        # Return average losses
        return {k: v / max(num_batches, 1) for k, v in total_losses.items()}

    @torch.no_grad()
    def evaluate(self, dataloader, split_name: str = "Val") -> float:
        self.model.eval()
        all_preds = []
        all_labels = []
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

            # Normalize both to same scale before ensembling
            logits_norm = F.softmax(outputs["logits"], dim=-1)
            proto_norm = F.softmax(outputs["proto_logits"], dim=-1)
            combined_logits = 0.7 * logits_norm + 0.3 * proto_norm
            preds = combined_logits.argmax(dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            loss = F.cross_entropy(outputs["logits"], labels)
            total_loss += loss.item()
            num_batches += 1

        # Macro F1
        macro_f1 = f1_score(all_labels, all_preds, average="macro")
        weighted_f1 = f1_score(all_labels, all_preds, average="weighted")
        avg_loss = total_loss / max(num_batches, 1)

        logger.info(
            f"{split_name} | Loss: {avg_loss:.4f} | Macro-F1: {macro_f1:.4f} | "
            f"Weighted-F1: {weighted_f1:.4f}"
        )
        self.last_eval_metrics[split_name] = {
            "loss": float(avg_loss),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        }

        # Per-class report
        if split_name == "Test":
            report = classification_report(all_labels, all_preds, digits=4)
            logger.info(f"\n{split_name} Classification Report:\n{report}")

        return macro_f1

    def save_checkpoint(self, tag: str = "latest"):
        os.makedirs(self.config.save_dir, exist_ok=True)
        path = os.path.join(self.config.save_dir, f"irm_ast_{self.config.task}_{tag}.pt")
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "best_f1": self.best_f1,
            "global_step": self.global_step,
        }, path)
        logger.info(f"Saved checkpoint to {path}")

    def load_checkpoint(self, tag: str = "best"):
        path = os.path.join(self.config.save_dir, f"irm_ast_{self.config.task}_{tag}.pt")
        if os.path.exists(path):
            ckpt = torch.load(path, map_location=self.config.device, weights_only=True)
            self.model.load_state_dict(ckpt["model_state_dict"])
            logger.info(f"Loaded checkpoint from {path}")
            return True
        return False

    def train(self):
        logger.info("=" * 60)
        logger.info(f"Starting AST-IRM Training - Task {self.config.task}")
        logger.info(f"Num classes: {self.model.num_classes}")
        logger.info(f"Device: {self.config.device}")
        logger.info(f"Precision: {self.precision}")
        logger.info(f"Batch size: {self.config.batch_size} x {self.config.grad_accum_steps} = {self.config.batch_size * self.config.grad_accum_steps}")
        logger.info(f"IRM anneal epochs: {self.config.irm_anneal_epochs}")
        logger.info(f"IRM penalty max: {self.config.irm_penalty_weight_max}")
        logger.info("=" * 60)

        for epoch in range(self.config.epochs):
            logger.info(f"\n{'='*40} Epoch {epoch+1}/{self.config.epochs} {'='*40}")

            irm_lam = self._get_irm_lambda(epoch)
            logger.info(f"IRM lambda for epoch {epoch+1}: {irm_lam:.2e}")

            # Train
            train_losses = self.train_epoch(epoch)
            logger.info(
                f"Epoch {epoch+1} Train Summary: "
                + " | ".join(f"{k}: {v:.4f}" for k, v in train_losses.items())
            )

            # Validate
            val_f1 = self.evaluate(self.val_loader, "Val")

            # Save best
            if val_f1 > self.best_f1:
                self.best_f1 = val_f1
                self.save_checkpoint("best")
                logger.info(f"*** New Best Val Macro-F1: {val_f1:.4f} ***")

            # Save latest
            self.save_checkpoint("latest")

        # Final test evaluation
        logger.info("\n" + "=" * 60)
        logger.info("FINAL TEST EVALUATION")
        logger.info("=" * 60)

        if self.load_checkpoint("best"):
            test_f1 = self.evaluate(self.test_loader, "Test")
        else:
            logger.warning("No best checkpoint found, evaluating current model")
            test_f1 = self.evaluate(self.test_loader, "Test")

        test_weighted = self.last_eval_metrics.get("Test", {}).get("weighted_f1", test_f1)
        logger.info(f"\n*** Final Test Macro-F1: {test_f1:.4f} | Weighted-F1: {test_weighted:.4f} ***")
        return {
            "test_f1": float(test_f1),
            "test_weighted_f1": float(test_weighted),
            "best_val_f1": float(self.best_f1),
        }


# ============================================================================
# Main
# ============================================================================

def compute_class_weights(labels: List[int], num_classes: int) -> torch.Tensor:
    """Compute inverse frequency class weights."""
    counts = np.bincount(labels, minlength=num_classes).astype(float)
    counts = np.maximum(counts, 1.0)  # avoid division by zero
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize
    return torch.tensor(weights, dtype=torch.float32)


def main(task: str = "T1", config: Optional[Config] = None):
    # Config
    if config is None:
        config = Config(task=task)
    config = apply_hardware_profile(config)
    set_seed(config.seed)

    logger.info(f"{'='*60}")
    logger.info(f"AST-IRM - AI-Generated Code Detection")
    logger.info(f"Task: {config.task}")
    logger.info(f"Benchmark: {config.benchmark}")
    logger.info(f"GPU: {_get_gpu_name()}")
    logger.info(f"{'='*60}")
    if not _supports_task_for_benchmark(config.benchmark, config.task):
        raise ValueError(
            f"Unsupported task/benchmark pair: benchmark={config.benchmark}, task={config.task}"
        )

    # Load one benchmark only (separate model per benchmark).
    if config.benchmark == "aicd":
        train_data, val_data, test_data, num_classes = load_aicd_data(config)
    elif config.benchmark == "droid":
        train_data, val_data, test_data, num_classes = load_droid_data(config)
    else:
        raise ValueError(f"Unsupported benchmark: {config.benchmark}")

    # Update num_classes from actual data
    logger.info(f"Detected {num_classes} classes")

    # Tokenizer
    logger.info(f"Loading tokenizer: {config.encoder_name}")
    tokenizer = AutoTokenizer.from_pretrained(config.encoder_name)

    # Datasets (using IRMDataset with environment labels)
    logger.info("Creating IRM datasets with language environments...")
    train_dataset = IRMDataset(train_data, tokenizer, config.max_length, config.ast_seq_len)
    val_dataset = IRMDataset(val_data, tokenizer, config.max_length, config.ast_seq_len)
    test_dataset = IRMDataset(test_data, tokenizer, config.max_length, config.ast_seq_len)

    # Class weights for imbalanced data
    train_labels = train_data["label"]
    class_weights = compute_class_weights(train_labels, num_classes)
    logger.info(f"Class weights: {class_weights.numpy()}")

    # DataLoaders
    pin = config.pin_memory and config.device == "cuda"
    common_loader_kwargs = {
        "num_workers": config.num_workers,
        "pin_memory": pin,
        "persistent_workers": config.num_workers > 0,
    }
    if config.num_workers > 0:
        common_loader_kwargs["prefetch_factor"] = config.prefetch_factor

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, drop_last=True,
        **common_loader_kwargs,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size * 2,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=config.batch_size * 2,
        shuffle=False,
        **common_loader_kwargs,
    )

    # Model
    logger.info(f"Building IRMAST model...")
    model = IRMAST(config, num_classes)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")

    # Trainer
    trainer = Trainer(config, model, train_loader, val_loader, test_loader, class_weights)

    # Train
    run_stats = trainer.train()

    return run_stats


def run_benchmark_suite(task: str = "T1", base_config: Optional[Config] = None):
    """
    Train the same method on two independent benchmarks sequentially.
    Each benchmark run initializes a fresh model and optimizer state.
    """
    if base_config is None:
        base_config = Config(task=task)

    results: Dict[str, Dict[str, float]] = {}
    benchmarks = [bench for bench in ["aicd", "droid"] if _supports_task_for_benchmark(bench, task)]
    original_save_dir = base_config.save_dir

    for bench in benchmarks:
        cfg = Config(**vars(base_config))
        cfg.task = task
        cfg.benchmark = bench
        cfg = apply_hardware_profile(cfg)
        # Keep checkpoints fully separated between benchmarks.
        cfg.save_dir = os.path.join(original_save_dir, f"{bench}_{task}")

        logger.info("\n" + "=" * 70)
        logger.info(f"BENCHMARK SUITE RUN: {bench.upper()} | task={task}")
        logger.info("=" * 70)
        results[bench] = main(task=task, config=cfg)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("\nBenchmark suite summary:")
    for bench, stats in results.items():
        logger.info(
            f"{bench.upper()} -> best_val_f1={stats['best_val_f1']:.4f}, "
            f"test_f1={stats['test_f1']:.4f}"
        )

    # Copy-paste friendly block for tracker logging.
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("\n=== SUITE_RESULTS_START ===")
    logger.info(f"timestamp={ts} | task={task} | method=AST-IRM")
    logger.info("| benchmark | task | best_val_f1 | test_f1 |")
    logger.info("|---|---|---:|---:|")
    for bench in ["aicd", "droid"]:
        if bench in results:
            logger.info(
                f"| {bench.upper()} | {task} | "
                f"{results[bench]['best_val_f1']:.4f} | {results[bench]['test_f1']:.4f} |"
            )
            if "test_weighted_f1" in results[bench]:
                logger.info(
                    f"{bench.upper()} weighted_test_f1={results[bench]['test_weighted_f1']:.4f}"
                )
    logger.info("=== SUITE_RESULTS_END ===")

    machine_block = {
        "timestamp": ts,
        "task": task,
        "method": "AST-IRM",
        "results": results,
    }
    logger.info("SUITE_RESULTS_JSON=" + json.dumps(machine_block, ensure_ascii=True))
    return results


# ============================================================================
# Entry point
# ============================================================================

if __name__ == "__main__":
    # Kaggle standalone defaults:
    # - No CLI/terminal arguments required.
    # - Edit this block directly, then run the script.
    RUN_MODE = "both"  # "single" or "both"
    TASK = "T1"        # "T1", "T2", "T3", "T4" (T4 is Droid-only 4-class)
    BENCHMARK = "aicd" # used only when RUN_MODE == "single"
    RUN_PREFLIGHT_CHECK = True  # If True, stop training immediately when any check fails.

    cfg = Config(
        task=TASK,
        benchmark=BENCHMARK,
        epochs=3,
        batch_size=32,
        grad_accum_steps=2,
        precision="auto",
        auto_h100_profile=True,
        num_workers=4,
        prefetch_factor=2,
        max_train_samples=100_000,
        max_val_samples=20_000,
        max_test_samples=50_000,
        require_tree_sitter=True,
        lambda_irm=1.0,
        irm_anneal_epochs=1,
        irm_penalty_weight_max=1e4,
    )
    cfg = apply_hardware_profile(cfg)

    try:
        if RUN_PREFLIGHT_CHECK:
            if RUN_MODE == "both":
                preflight_benchmark_suite(task=cfg.task, base_config=cfg)
            else:
                preflight_single_benchmark(cfg)

        if RUN_MODE == "both":
            run_benchmark_suite(task=cfg.task, base_config=cfg)
        else:
            main(task=cfg.task, config=cfg)
    except PreflightError as e:
        logger.error(f"PRE-FLIGHT FAILED: {e}")
        raise SystemExit(1)
