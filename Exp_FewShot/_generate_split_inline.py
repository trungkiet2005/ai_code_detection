"""Generator: produce 6 self-contained inline files from exp_fs_inline.py.

Each output file has METHOD_KEY hardcoded so the user pastes the file into
a Kaggle cell, runs, and gets that one method's sweep without touching env vars.

Run locally:
  python Exp_FewShot/_generate_split_inline.py
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(HERE, "exp_fs_inline.py")

METHODS = [
    ("baseline",      "FS-Baseline-CE",        "exp_fs_inline_baseline",       "ce",     False, "ModernBERT + CE only (floor)"),
    ("ntkalign",      "FS-NTKAlign",           "exp_fs_inline_ntkalign",       "ntk",    False, "CE + NTK target-kernel alignment"),
    ("supcon",        "FS-SupCon",             "exp_fs_inline_supcon",         "supcon", False, "CE + Supervised Contrastive (Khosla 2020)"),
    ("frozen",        "FS-Frozen-LinearProbe", "exp_fs_inline_frozen",         "ce",     True,  "encoder frozen, CE only (linear probe)"),
    ("ntk_frozen",    "FS-NTKAlign-Frozen",    "exp_fs_inline_ntk_frozen",     "ntk",    True,  "encoder frozen + NTK loss"),
    ("supcon_frozen", "FS-SupCon-Frozen",      "exp_fs_inline_supcon_frozen",  "supcon", True,  "encoder frozen + SupCon loss"),
]


def render(method_key: str, method_name: str, exp_id: str, loss_kind: str,
           freeze: bool, blurb: str, src_text: str) -> str:
    """Take master inline text, hardcode METHOD_KEY, swap docstring, return new text."""
    new_doc = f'''"""\n{exp_id}.py -- ONE-FILE self-contained few-shot suite for Kaggle T4.

Method: {method_name} ({blurb})

Paste this entire file into a Kaggle notebook cell, run. No `git clone`,
no other files needed -- bootstraps pip-installs only.

Default sweep: K=128 + fraction in {{0.01, 0.05}} = 3 configs (~50 min on T4).
The K=32 cell is intentionally skipped (we already have those numbers).
Override via env vars:
    FS_SWEEP_KS     -- "8,16,32,64,128"   (empty -> skip K-shot regime)
    FS_SWEEP_FRACS  -- "0.01,0.05,0.1"    (empty -> skip fraction regime)
    FS_SEED         -- 42
    FS_LAMBDA_NTK   -- 0.4 (NTK variants)
    FS_LAMBDA_SUPCON -- 0.4 (SupCon variants)
    FS_TEMP         -- 0.07 (SupCon variants)
    FS_LR_HEADS     -- 1e-3 (frozen variants)

Output: /kaggle/working/results/{exp_id}_<label>_seed<S>.json
        (or ./results/... locally)
"""'''
    # Replace the original module docstring (everything from the first
    # triple-quote pair at the top of the file).
    new_text = re.sub(r'^""".*?"""', new_doc, src_text, count=1, flags=re.DOTALL)

    # Replace the FS_METHOD env-var read with a hardcoded constant.
    new_text = new_text.replace(
        'method_key = os.environ.get("FS_METHOD", "ntkalign").strip().lower()',
        f'method_key = {method_key!r}',
    )

    # Update the logger / banner so it shows the file's own exp_id
    new_text = new_text.replace(
        'logger = logging.getLogger("exp_fs_inline")',
        f'logger = logging.getLogger({exp_id!r})',
    )
    return new_text


def main():
    with open(SRC, "r", encoding="utf-8") as f:
        src_text = f.read()

    for method_key, method_name, exp_id, loss_kind, freeze, blurb in METHODS:
        out_path = os.path.join(HERE, f"{exp_id}.py")
        new_text = render(method_key, method_name, exp_id, loss_kind, freeze, blurb, src_text)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_text)
        print(f"[gen] {out_path}  ({len(new_text)} bytes)")


if __name__ == "__main__":
    main()
