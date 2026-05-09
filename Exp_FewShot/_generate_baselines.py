"""Generator: produce 4 self-contained baseline reimplementation inline files.

Each output file is a copy of exp_fs_inline_baseline.py with:
  - Different encoder_name (UniXcoder / CodeBERT / GraphCodeBERT / CodeT5)
  - Updated METHOD_NAME, EXP_ID
  - Docstring referencing the paper baseline being reimplemented

These files run the SAME few-shot/fraction sweep as our FS-Baseline-CE,
just with a different pretrained backbone -- giving us apples-to-apples
comparison against paper UniXcoder 66.33 / CodeBERT 64.80 / CodeT5 62.45 etc.

Run locally:
  python Exp_FewShot/_generate_baselines.py
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "exp_fs_inline_baseline.py")

# (file_id, exp_id, method_name, encoder_name, paper_score, blurb)
BASELINES = [
    ("unixcoder",     "exp_fs_baseline_unixcoder",
     "FS-Baseline-UniXcoder",
     "microsoft/unixcoder-base",       66.33,
     "UniXcoder (Guo et al. ACL'22) -- strongest paper baseline at 66.33 full-data Macro-F1"),
    ("codebert",      "exp_fs_baseline_codebert",
     "FS-Baseline-CodeBERT",
     "microsoft/codebert-base",         64.80,
     "CodeBERT (Feng et al. 2020) -- 64.80 full-data Macro-F1 in CoDET-M4 paper Table 7"),
    ("graphcodebert", "exp_fs_baseline_graphcodebert",
     "FS-Baseline-GraphCodeBERT",
     "microsoft/graphcodebert-base",    None,
     "GraphCodeBERT (Guo et al. ICLR'21) -- AST-aware extension of CodeBERT, no published number on CoDET-M4"),
    ("codet5",        "exp_fs_baseline_codet5",
     "FS-Baseline-CodeT5",
     "Salesforce/codet5-base",          62.45,
     "CodeT5 (Wang et al. EMNLP'21) -- encoder-decoder; we use the encoder side. 62.45 full-data in paper Table 7"),
]


def render(file_id, exp_id, method_name, encoder_name, paper_score, blurb, src_text):
    score_line = f"\nPaper full-data baseline: **{paper_score:.2f} Macro-F1** (Author IID, CoDET-M4)." if paper_score else ""
    new_doc = f'''"""\n{exp_id}.py -- ONE-FILE self-contained baseline reimplementation for Kaggle T4.

Method: {method_name}
Encoder: {encoder_name}
{blurb}{score_line}

Reimplements the paper baseline under our few-shot / %-fraction sweep
protocol so we can plot the data-efficiency curve apples-to-apples
against our NTKAlign + ModernBERT result.

Paste into a Kaggle cell, run. No `git clone`, no other files needed.

Default sweep: K=128 + fraction in {{0.01, 0.05}} = 3 configs (~50 min on T4).
Override via FS_SWEEP_KS / FS_SWEEP_FRACS env vars.

Output: /kaggle/working/results/{exp_id}_<label>_seed<S>.json
"""'''

    out = re.sub(r'^""".*?"""', new_doc, src_text, count=1, flags=re.DOTALL)

    # Replace the encoder_name in FSConfig dataclass default.
    out = out.replace(
        'encoder_name: str = "answerdotai/ModernBERT-base"',
        f'encoder_name: str = {encoder_name!r}',
    )

    # Replace METHODS dict -- since each file has hardcoded method via
    # _generate_split_inline, the dict is still there. We want to swap the
    # 'baseline' entry to use our exp_id.
    out = out.replace(
        '"baseline":       ("FS-Baseline-CE",       "exp_fs_inline_baseline",       "ce",     False),',
        f'"baseline":       ({method_name!r},   {exp_id!r},   "ce",     False),',
    )

    # The hardcoded method_key from split generator is 'baseline' -- keep that
    # since this file is still a "baseline" type method (CE only, free encoder).
    # Just need to make sure logger name matches the new exp_id for cleanliness.
    out = out.replace(
        f'logger = logging.getLogger("exp_fs_inline_baseline")',
        f'logger = logging.getLogger({exp_id!r})',
    )

    return out


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"missing source: {SRC}\nrun _generate_split_inline.py first")
    with open(SRC, "r", encoding="utf-8") as f:
        src_text = f.read()
    for file_id, exp_id, method_name, encoder, score, blurb in BASELINES:
        out_path = os.path.join(HERE, f"{exp_id}.py")
        new_text = render(file_id, exp_id, method_name, encoder, score, blurb, src_text)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_text)
        print(f"[gen] {out_path}  ({len(new_text)} bytes)  encoder={encoder}")


if __name__ == "__main__":
    main()
