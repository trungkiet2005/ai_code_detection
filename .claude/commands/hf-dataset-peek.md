---
description: Emit a Kaggle cell that prints schema + sample rows from one of the three HF benchmarks
argument-hint: <aicd | droid | codet_m4> [split] [N]
allowed-tools: Read
---

# HuggingFace dataset peek (Kaggle cell)

The user develops locally without Python compute. This command emits a **Kaggle-ready cell** that streams schema + sample rows from a project benchmark. Useful when writing a new data loader or debugging a column-shape issue.

Do NOT run `datasets.load_dataset` locally. Output is copy-paste-ready cell text only.

## Arguments
`$ARGUMENTS`:
- `<benchmark>` — `aicd`, `droid`, or `codet_m4`
- `[split]` — default `train`; options include `validation`, `test`
- `[N]` — number of sample rows to print, default 3

## Benchmark → HF ID mapping

- `aicd` → `AICD-bench/AICD-Bench`
- `droid` → `project-droid/DroidCollection`
- `codet_m4` → `DaniilOr/CoDET-M4`

## Steps

Emit a single fenced Python code block the user can paste into a Kaggle notebook cell. The block must:

- `from datasets import load_dataset`
- Load with `streaming=True` (no full download)
- Print the `features` dict
- Iterate `N` rows, print each row's keys + value types + truncated value preview (max 200 chars per field)
- Handle the gated-dataset case: `try/except` around load; if 401, print "run `huggingface-cli login` in the Kaggle cell above"

After the cell, add a short Claude-side note reminding the user to paste the output back so Claude can summarize:
- Label column name + class count
- Text/code column name
- Domain / language / generator columns if present
- Any column a naive loader would miss

## Rules
- Use `streaming=True`. The cell must never trigger a multi-GB download.
- Output text only. Do not `python -c` locally.
- Do not cache results to `docs/` or commit them.
