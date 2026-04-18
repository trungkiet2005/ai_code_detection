"""
Feature extraction — AST + structural + spectral.

All feature pipelines used by the `SpectralCode` backbone live here so every
climb exp file imports the same, paper-reproducible preprocessing.
"""
from __future__ import annotations

import logging
import re
from typing import List

import numpy as np
import torch

logger = logging.getLogger("exp_climb")


# ---------------------------------------------------------------------------
# AST node vocabulary + parsers
# ---------------------------------------------------------------------------

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

STRUCTURAL_FEATURE_DIM = 22
SPECTRAL_FEATURE_DIM = 64


def _try_load_tree_sitter():
    try:
        import tree_sitter_languages  # type: ignore

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
        logger.warning("tree-sitter-languages not found; using regex fallback for AST extraction")
        return None


def _fallback_ast_extract(code: str) -> List[int]:
    """Regex-based heuristic when tree-sitter is unavailable."""
    features: List[int] = []
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


def ast_parser_available() -> bool:
    return _ast_parser is not None


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


# ---------------------------------------------------------------------------
# Structural (hand-crafted stylometric) features
# ---------------------------------------------------------------------------

def extract_structural_features(code: str) -> List[float]:
    lines = code.split("\n")
    num_lines = len(lines)
    avg_line_len = float(np.mean([len(l) for l in lines])) if lines else 0.0
    max_line_len = max((len(l) for l in lines), default=0)

    indents = [len(l) - len(l.lstrip()) for l in lines if l.strip()]
    avg_indent = float(np.mean(indents)) if indents else 0.0
    max_indent = max(indents, default=0)
    indent_variance = float(np.var(indents)) if indents else 0.0

    num_functions = code.count("def ") + code.count("function ") + code.count("func ") + code.count("fn ")
    num_classes = code.count("class ") + code.count("struct ") + code.count("interface ")
    num_loops = code.count("for ") + code.count("while ") + code.count("foreach ")
    num_conditionals = code.count("if ") + code.count("else ") + code.count("elif ") + code.count("else if")
    num_returns = code.count("return ")
    num_comments = code.count("//") + code.count("#") + code.count("/*")
    num_imports = code.count("import ") + code.count("include ") + code.count("require ") + code.count("using ")
    num_try_catch = code.count("try") + code.count("catch") + code.count("except")

    identifiers = re.findall(r"\b[a-zA-Z_]\w*\b", code)
    num_snake_case = sum(1 for i in identifiers if "_" in i and i.islower())
    num_camel_case = sum(1 for i in identifiers if any(c.isupper() for c in i[1:]) and "_" not in i)
    num_single_char = sum(1 for i in identifiers if len(i) == 1)
    avg_identifier_len = float(np.mean([len(i) for i in identifiers])) if identifiers else 0.0

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


# ---------------------------------------------------------------------------
# Spectral (FFT) features on token sequences
# ---------------------------------------------------------------------------

def extract_spectral_features(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Multi-scale FFT energy + shape features. Output dim = SPECTRAL_FEATURE_DIM (64)."""
    batch_size = input_ids.size(0)
    features: List[List[float]] = []
    windows = [32, 64, 128, 256]
    features_per_window = SPECTRAL_FEATURE_DIM // len(windows)

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
                sample_feats.extend([0.0] * features_per_window)
                continue

            total_energy = np.sum(magnitude ** 2)
            n_bands = min(8, len(magnitude))
            band_size = max(1, len(magnitude) // n_bands)
            band_energies: List[float] = []
            for b in range(n_bands):
                start = b * band_size
                end = min(start + band_size, len(magnitude))
                band_energies.append(float(np.sum(magnitude[start:end] ** 2) / (total_energy + 1e-10)))

            while len(band_energies) < features_per_window - 4:
                band_energies.append(0.0)
            band_energies = band_energies[:features_per_window - 4]

            spectral_centroid = float(np.sum(np.arange(len(magnitude)) * magnitude) / (np.sum(magnitude) + 1e-10))
            spectral_rolloff = float(np.searchsorted(np.cumsum(magnitude), 0.85 * np.sum(magnitude)) / len(magnitude))
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
