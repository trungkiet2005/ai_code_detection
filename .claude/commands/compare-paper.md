---
description: Compare a result against the source paper's baseline numbers
argument-hint: <benchmark> <task> <our-metric-value>
allowed-tools: Read, Grep, Glob
---

# Compare against paper baseline

Compute Δ-vs-paper for a single result and report whether it would be publishable-strong, marginal, or a regression.

## Arguments
`$ARGUMENTS`:
- `<benchmark>` — `aicd`, `droid`, or `codet_m4`
- `<task>` — e.g., `T1`, `T2`, `T3`, `binary`, `author`
- `<our-metric-value>` — float in [0, 1] or percentage

## Steps

1. **Load the source paper table** from `docs/references/`:
   - `aicd` → `paper_AICD.md`
   - `droid` → `paper_Droid.md`
   - `codet_m4` → `paper_CodeDet_M4.md`
2. **Extract the best baseline for `<task>`** as reported in the paper. If the paper reports multiple baselines, use the strongest one (this is the bar to beat).
3. **Normalize units** — convert both to the same scale (0–1 or 0–100).
4. **Report**:
   - Paper best baseline (name + value)
   - Our value
   - Δ = ours − baseline (absolute)
   - Verdict:
     - `Δ ≥ +2.0 pp` → **strong** (publishable signal)
     - `0.0 ≤ Δ < +2.0` → **marginal** (needs multi-seed confirmation)
     - `Δ < 0` → **regression** (flag in report)
5. If the task is **AICD T1**, add a reminder: all known methods suffer val-test collapse; "strong" here means closing the gap, not beating val.

## Rules
- Do not cherry-pick a weaker baseline. Use the strongest entry in the paper.
- If the paper does not report this exact task, say so — do not substitute.
