---
description: Scaffold a new experiment file in Exp_DM, Exp_CodeDet, or Exp_Climb
argument-hint: <suite> <expNN_name> "<one-line method description>"
allowed-tools: Read, Write, Glob, Grep, Bash(python:*), Bash(ls:*)
---

# New experiment scaffold

Create a new standalone experiment file for the AICD-Bench research project.

## Arguments
User invocation: `/new-experiment $ARGUMENTS`

Parse `$ARGUMENTS` as: `<suite> <expNN_name> "<description>"` where:
- `<suite>` ∈ {`Exp_DM`, `Exp_CodeDet`, `Exp_Climb`, `Exp_TK`}
- `<expNN_name>` = e.g. `exp31_mymethod` (DM/TK), `run_codet_m4_exp39_mymethod` (CodeDet), `exp_20_mymethod` (Climb)
- `<description>` = one-line core innovation claim

## Steps

1. **Verify the suite directory exists** and the expNN ID is not already taken (glob for `$SUITE/exp*<NN>*.py`). If taken, stop and tell the user which file collides.
2. **Identify the closest existing sibling** — pick the most recent matching experiment in the same suite (prefer one with similar methodology if clear from the description). Read its full contents as a template.
3. **Copy the skeleton** into the new file path. Keep training loop, preflight, metric export, tracker-row emission identical. Change only:
   - Class/function names
   - `EXP_NAME`, `EXP_DESC`, method-specific hyperparam constants at the top
   - Model class + loss formula
4. **Do NOT** wire it into any shared package. Standalone is the rule.
5. Report back: the created file path, the template it was based on, and a reminder that the user must append a row to the matching tracker (`Exp_DM/dm_tracker.md`, `Exp_CodeDet/tracker.md`, or `Exp_Climb/tracker.md`) only after a real run produces numbers.

## Guardrails
- Refuse to create under `experiment/` (legacy) or `Exp_TK/` unless explicitly asked.
- Never reuse an exp number. Never rewrite a sibling file.
- Do not add a tracker row with placeholder numbers.
