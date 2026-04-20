"""
[exp_zs_19] SemanticDriftDetector -- zero-shot detector via semantic
            embedding shift under code-preserving refactoring.

THEORY HOOK (OUR OWN, transfer from paraphrasing robustness):
  Meng et al. "A Gradient-Based Evader Against AI-Generated Text Detectors"
  (USENIX Sec 2025) + TempParaphraser EMNLP 2025: paraphrasing is the
  strongest evasion against log-prob detectors. Novel insight: measure how
  *semantic meaning* shifts under code-preserving refactoring.

  AI code: Meaning STABLE (refactoring preserves intent; LLM mechanically
  remaps tokens while keeping semantics fixed).
  Human code: Meaning DRIFTS (human's refactoring trades code style for
  adaptation to new context; semantics shifts).

  Define semantic drift as:
      drift = ||embedding(code) - embedding(refactored_code)||_2
  where refactoring = (rename identifiers + reorder statements).
  Higher drift = human-like (adaptive); lower drift = AI-like (mechanical).

WHY NOVEL:
  * Orthogonal to structural stability (exp_zs_17): measure SEMANTIC shift,
    not structural robustness.
  * Captures the PARAPHRASING EVASION MECHANISM: human code rewrites freely;
    AI code rigidly preserves semantics.
  * Bridges adversarial robustness (evasion) and authorship detection.

IMPLEMENTATION (CodeBERT for semantic encoding):
  1. Encode original code with CodeBERT CLS token -> emb_orig.
  2. Refactor: rename identifiers (regex) + reorder statements (topological).
  3. Encode refactored code -> emb_refac.
  4. Score = ||emb_orig - emb_refac||_2.
  5. Higher score = human-like; calibrate on dev split.

Cost: 2 forward passes per sample (original + refactored). ~14 min on H100.
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
    """Replace all identifiers with _v0, _v1, ... to preserve semantics while changing syntax."""
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
    """Reorder non-control-flow statements while preserving execution."""
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

    stmt_idx = [idx for idx, _ in stmt_lines]
    stmt_lines_rev = list(reversed(stmt_lines))
    result_lines = list(lines)
    for orig_idx, (_, new_line) in zip(stmt_idx, stmt_lines_rev):
        result_lines[orig_idx] = new_line

    return '\n'.join(result_lines)


def _semantic_drift_score(codes: List[str], cfg: ZSConfig) -> np.ndarray:
    """Measure semantic drift (embedding distance) under refactoring."""
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    logger.info(f"[ZS-19] Loading encoder {cfg.scorer_lm}...")
    tokenizer = AutoTokenizer.from_pretrained(cfg.scorer_lm)
    model = AutoModelForMaskedLM.from_pretrained(cfg.scorer_lm)
    model.eval()
    if cfg.device == "cuda" and torch.cuda.is_available():
        model = model.to("cuda")
        if cfg.precision == "bf16":
            model = model.to(torch.bfloat16)

    scores = np.zeros(len(codes), dtype=np.float64)
    bs = cfg.batch_size

    with torch.no_grad():
        for start in range(0, len(codes), bs):
            chunk = codes[start : start + bs]

            # Encode originals
            enc_orig = tokenizer(
                chunk, max_length=cfg.scorer_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids = enc_orig["input_ids"]
            attn = enc_orig["attention_mask"]
            if cfg.device == "cuda":
                input_ids = input_ids.to("cuda")
                attn = attn.to("cuda")

            out_orig = model(input_ids=input_ids, attention_mask=attn).logits.float()
            emb_orig = out_orig[:, 0, :].cpu()

            # Refactor: rename + reorder
            refac_chunk = [_reorder_statements(_rename_identifiers(c)) for c in chunk]

            enc_refac = tokenizer(
                refac_chunk, max_length=cfg.scorer_max_length,
                padding="max_length", truncation=True, return_tensors="pt",
            )
            input_ids_r = enc_refac["input_ids"]
            attn_r = enc_refac["attention_mask"]
            if cfg.device == "cuda":
                input_ids_r = input_ids_r.to("cuda")
                attn_r = attn_r.to("cuda")

            out_refac = model(input_ids=input_ids_r, attention_mask=attn_r).logits.float()
            emb_refac = out_refac[:, 0, :].cpu()

            # Score = L2 distance
            dists = torch.norm(emb_orig - emb_refac, dim=1).numpy()
            for i in range(len(chunk)):
                scores[start + i] = float(dists[i])

    return scores


if __name__ == "__main__":
    cfg = ZSConfig(benchmark="droid_T3")
    run_zs_oral(
        method_name="SemanticDriftDetector",
        exp_id="exp_zs_19",
        score_fn=_semantic_drift_score,
        cfg=cfg,
    )
