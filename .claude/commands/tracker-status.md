---
description: Summarize current leaderboard state across all three trackers
allowed-tools: Read, Grep, Glob
---

# Tracker status

Give a one-screen overview of the repo's experimental standing.

## Steps

1. Read the headline tables from:
   - `Exp_DM/dm_tracker.md` (AICD + Droid)
   - `Exp_CodeDet/tracker.md` (CoDET-M4)
   - `Exp_Climb/tracker.md` (EMNLP 2026 suite)
2. For each benchmark+task, identify:
   - Current best method name + primary metric
   - Val-test gap (if reported)
   - Whether the best method beats the source paper's baseline
3. Count experiments by status (Done / Pending / Failed).

## Output format

```
## AICD-Bench
  T1 (Macro-F1): best = <method>  test=<x>  val-test gap=<y>  vs paper: <Δ>
  T2: ...
  T3: ...

## DroidCollection
  T1 (Weighted-F1): ...

## CoDET-M4
  binary: ...
  author: ...

## Pipeline
  Done: N  Pending: M  Failed: K
  Next most-promising pending experiment: <expNN> — <rationale>
```

Keep it tight. No prose beyond the structured summary.
