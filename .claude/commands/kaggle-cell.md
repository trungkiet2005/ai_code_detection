---
description: Generate copy-paste-ready Kaggle notebook cells for an experiment script
argument-hint: <path/to/expNN_*.py> [--smoke]
allowed-tools: Read, Glob
---

# Kaggle cells for an experiment

The user develops locally but runs experiments on Kaggle H100. This command emits a block of Kaggle cells the user can paste straight into a new Kaggle notebook.

## Arguments
`$ARGUMENTS`:
- `<path/to/expNN_*.py>` — the experiment file (required)
- `--smoke` — optional; produce a tiny-sample variant for a ~2-min sanity check

## Steps

1. **Read the experiment file's header** (first ~80 lines) to extract:
   - `BENCHMARK` (aicd / droid / codet_m4)
   - `TASK` (T1 / T2 / T3 / binary / author)
   - Top-level config constants (sample caps, epochs, batch size, model name)
2. **Check the suite**:
   - `Exp_DM/` → AICD + Droid
   - `Exp_CodeDet/` → CoDET-M4
   - `Exp_Climb/` → EMNLP suite
   - `Exp_TK/` → baselines
3. **Emit the following cells** as fenced code blocks the user can paste. Do NOT execute anything locally.

## Output template

Print a compact "Kaggle-ready" block like this (fill in variables, keep the structure):

````markdown
## Kaggle cells for `<path>`

**Accelerator**: GPU P100 or T4×2 if H100 quota missing; BF16 falls back to FP16 on non-H100.
**Internet**: ON (required for HF dataset + tree-sitter).

### Cell 1 — install deps
```bash
!pip install -q datasets transformers accelerate tree-sitter tree-sitter-languages 2>&1 | tail -n 3
```

### Cell 2 — fetch the script
```bash
# Option A: upload the file as a Kaggle Dataset input, then:
!cp /kaggle/input/<your-dataset-slug>/<filename>.py /kaggle/working/

# Option B: git-clone (if repo is public):
# !git clone --depth 1 https://github.com/<user>/AICD_Bench.git /kaggle/working/repo
# %cd /kaggle/working/repo
```

### Cell 3 — run
```bash
!cd /kaggle/working && python <filename>.py 2>&1 | tee /kaggle/working/<expNN>_<benchmark>_<task>.log
```

### Cell 4 — tail the summary
```bash
!tail -n 80 /kaggle/working/<expNN>_<benchmark>_<task>.log
```

### After it finishes
Copy the full log (or just the final summary block) back into Claude and run `/analyze-log` — Claude will parse metrics, compute val-test gap, and emit a tracker row for `<dm_tracker.md | tracker.md | Exp_Climb/tracker.md>`.
````

## `--smoke` mode

If `--smoke` is passed, additionally emit a **Cell 2b — patch for smoke** cell that uses `sed` or a tiny Python one-liner to override the sample caps to small values (e.g. `MAX_TRAIN_SAMPLES=2000`, `MAX_VAL_SAMPLES=500`, `MAX_TEST_SAMPLES=500`, `EPOCHS=1`). Read the actual variable names from the file header first — do not guess.

## Rules

- This command emits text only. It must NOT call `python`, `!pip`, `git push`, or anything that mutates state.
- Never claim a run has happened. The user will paste the log back.
- The benchmark/metric reminder (AICD→Macro-F1, Droid→Weighted-F1, CoDET-M4→Macro-F1) should appear at the bottom of the output as a one-liner.
