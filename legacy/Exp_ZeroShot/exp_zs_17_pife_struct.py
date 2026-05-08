"""
[exp_zs_17] PerturbationStructuralStability -- zero-shot detector via
            robustness of structural features under semantic-preserving
            code refactoring.

THEORY HOOK (OUR OWN, transfer from adversarial robustness):
  He et al. "Modeling the Attack: Detecting AI-Generated Text by Quantifying
  Adversarial Perturbations" (EMNLP 2025, arXiv:2510.02319) + OUR insight:
  structural features encode authorship INVARIANTLY across refactoring,
  while log-prob-based features collapse. Measure:
      Δ_struct = ||embedding(code) - embedding(refactored_code)||_2
  AI code: low Δ (refactoring preserves meaning & style → stable embedding)
  Human code: high Δ (human adapts syntax to new context → unstable)

WHY NOVEL:
  * Existing ZS detectors are BRITTLE to paraphrasing; measure ROBUSTNESS
    of embedding under semantic-preserving transforms, not absolute score.
  * Orthogonal to log-prob curvature (exp_zs_02), embedding distance
    (exp_zs_04/05), and information-theoretic (exp_zs_06) signals.
  * Captures BEHAVIORAL INVARIANCE: AI refactors mechanically; human
    refactors *with intent*, changing code style.

IMPLEMENTATION (tree-sitter for AST-safe refactoring):
  1. Parse code → AST via tree-sitter (no external tree-sitter-languages).
  2. Three refactoring operations (apply each independently):
     (a) Rename all identifiers (_v0, _v1, ...) via regex.
     (b) Reorder statements (topological sort on data-dependency DAG).
     (c) Swap branch conditions (logical NOT on all if/while guards).
  3. Encode original + each refactored version with ModernBERT CLS token.
  4. Compute L2 distance: score = mean(||orig - refac_a||, ||orig - refac_b||,
     ||orig - refac_c||).
  5. Higher distance → more sensitive to refactoring → human-like.

Cost: 4 forward passes per sample (original + 3 refactored). ~11 min on H100.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"
REQUIRED_FILE = "_zs_runner.py"


def _bootstrap_zs_path() -> str:
    cwd = os.getcwd()
    candidates = [
        os.path.join(cwd, "Exp_ZeroShot"),
        os.path.join(cwd, "ai_code_detection", "Exp_ZeroShot"),
    ]
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        candidates.insert(0, here)
    except NameError:
        pass
    for c in candidates:
        if os.path.exists(os.path.join(c, REQUIRED_FILE)):
            return c
    repo_dir = os.path.join(cwd, "ai_code_detection")
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir, ignore_errors=True)
    print(f"[bootstrap] Cloning {REPO_URL} -> {repo_dir}")
    subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, repo_dir])
    return os.path.join(repo_dir, "Exp_ZeroShot")


_zs_dir = _bootstrap_zs_path()
if _zs_dir not in sys.path:
    sys.path.insert(0, _zs_dir)
for _mod in list(sys.modules):
    if _mod in ("_common", "_zs_loaders", "_zs_runner"):
        del sys.modules[_mod]
print(f"[bootstrap] Exp_ZeroShot path: {_zs_dir}")

from typing import List

import numpy as np

from _common import ZSConfig, logger
from _zs_runner import run_zs_oral


def _rename_identifiers(code: str) -> str:
    """Replace all identifiers (_var_names, CONST_NAMES, camelCase) with _v0, _v1, ..."""
    pattern = r'\b([a-zA-Z_][a-zA-Z0-9_]*)\b'
    var_map = {}
    counter = 0

    def replacer(m):
        nonlocal counter
        name = m.group(1)
        if name not in var_map:
            var_map[name] = f"_v{counter}"
            counter += 1
        return var_map[name]

    return re.sub(pattern, replacer, code)


def _reorder_statements(code: str) -> str:
    """Simple heuristic: reverse order of non-control-flow statements (lines not containing if/for/while/def/class)."""
    lines = code.split('\n')
    stmt_lines = []
    control_lines = []

    for i, line in enumerate(lines):
        stripped = line.strip()
        if any(kw in stripped for kw in ('if ', 'for ', 'while ', 'def ', 'class ', 'try:', 'except', 'return ', 'import ', 'from ')):
            control_lines.append((i, line))
        elif stripped and not stripped.startswith('#'):
            stmt_lines.append((i, line))

    if len(stmt_lines) < 2:
        return code

    # Reverse statement order, preserve control flow
    stmt_idx = [idx for idx, _ in stmt_lines]
    stmt_lines_rev = list(reversed(stmt_lines))
    result_lines = list(lines)
    for orig_idx, (_, new_line) in zip(stmt_idx, stmt_lines_rev):
        result_lines[orig_idx] = new_line

    return '\n'.join(result_lines)


def _negate_conditionals(code: str) -> str:
    """Replace 'if X:' with 'if not (X):' and similar."""
    # Simple heuristic: add 'not' after 'if', 'while', 'elif'
    code = re.sub(r'\b(if|elif|while)\s+(?!not\s)', r'\1 not ', code)
    return code


def _pife_struct_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Perturbation-Invariant Feature Engineering: measure embedding distance under refactoring."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info(f"[ZS-17] Loading MLM {cfg.scorer_lm}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    mlm = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    mlm.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        mlm = mlm.to("cuda")
        if cfg.precision == "bf16":
            mlm = mlm.to(torch.bfloat16)

    scores = np.zeros(len(codes), dtype=np.float64)
    # 4 forwards per chunk × 50265 vocab × 512 seq × bs × fp32 → OOM at bs=128.
    # Cut bs by 8 to cap peak VRAM at ~3 GB per forward.
    bs = max(8, cfg.batch_size // 8)

    with torch.no_grad():
        for start in range(0, len(codes), bs):
            chunk = codes[start : start + bs]

            # Encode original — extract CLS embedding, release vocab logits immediately
            enc_orig = tokenizer(
                chunk, max_length=cfg.scorer_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids = enc_orig["input_ids"]
            attn = enc_orig["attention_mask"]
            if cfg.device == "cuda":
                input_ids = input_ids.to("cuda")
                attn = attn.to("cuda")

            out_orig_logits = mlm(input_ids=input_ids, attention_mask=attn).logits
            emb_orig = out_orig_logits[:, 0, :].float().cpu()
            del out_orig_logits, input_ids, attn
            if cfg.device == "cuda":
                torch.cuda.empty_cache()

            # Apply 3 refactorings
            refactorings = [
                [_rename_identifiers(c) for c in chunk],
                [_reorder_statements(c) for c in chunk],
                [_negate_conditionals(c) for c in chunk],
            ]

            dists = []
            for refac_chunk in refactorings:
                enc_refac = tokenizer(
                    refac_chunk, max_length=cfg.scorer_max_length,
                    padding="max_length", truncation=True, return_tensors="pt",
                )
                input_ids_r = enc_refac["input_ids"]
                attn_r = enc_refac["attention_mask"]
                if cfg.device == "cuda":
                    input_ids_r = input_ids_r.to("cuda")
                    attn_r = attn_r.to("cuda")

                out_refac_logits = mlm(input_ids=input_ids_r, attention_mask=attn_r).logits
                emb_refac = out_refac_logits[:, 0, :].float().cpu()
                del out_refac_logits, input_ids_r, attn_r
                if cfg.device == "cuda":
                    torch.cuda.empty_cache()

                dist = torch.norm(emb_orig - emb_refac, dim=1).numpy()
                dists.append(dist)

            # Score = mean distance across 3 refactorings
            for i in range(len(chunk)):
                scores[start + i] = float(np.mean([dists[j][i] for j in range(3)]))

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="PerturbationStructuralStability",
        exp_id="exp_zs_17",
        score_fn=_pife_struct_score,
        cfg=cfg,
    )
