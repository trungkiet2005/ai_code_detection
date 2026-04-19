---
description: Append a result row to the correct tracker markdown after a finished run
argument-hint: <expNN> <benchmark> <path-to-log-or-pasted-metrics>
allowed-tools: Read, Edit, Grep, Glob, Bash(ls:*), Bash(tail:*)
---

# Update tracker

Append a new row to the appropriate tracker after an experiment finishes. Never rewrite existing rows.

## Arguments
`$ARGUMENTS`:
- `<expNN>` — experiment ID (e.g., `exp14`, `exp_17`, `run_codet_m4_exp21`)
- `<benchmark>` — `aicd`, `droid`, or `codet_m4`
- `<source>` — either a log file path or metrics pasted inline by the user

## Steps

1. **Pick the target tracker** based on benchmark:
   - `aicd` or `droid` → `Exp_DM/dm_tracker.md`
   - `codet_m4` → `Exp_CodeDet/tracker.md`
   - Climb suite (`exp_NN`) → `Exp_Climb/tracker.md`
2. **Read the last 50 rows of the target tracker** to learn the exact column schema — match it. Do NOT invent columns.
3. **Extract metrics from the source**:
   - Required: Best Val F1, Test Primary, Val-Test Gap, per-task breakdown if multi-task
   - Optional: per-class report, Δ-vs-paper
   - Primary metric: Macro-F1 (AICD, CoDET-M4) or Weighted-F1 (Droid)
4. **Check for duplicates**. If a row for this `expNN` on this benchmark/task already exists, ask the user: overwrite-in-place (discouraged) or append as a new run-suffix row (preferred).
5. **Append** the row with `Edit`. Do not reformat the rest of the file. Do not touch historical rows.
6. **Verify** by reading the last 10 lines and confirming the new row renders.
7. Report which row was added and its position.

## Rules
- Historical rows are immutable.
- If any required metric is missing from the source, stop and ask — never invent or estimate.
- val-test gap is non-negotiable. Compute it if only val and test are present: `gap = val - test`.
