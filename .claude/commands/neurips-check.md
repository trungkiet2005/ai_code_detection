---
description: Validate a LaTeX file against NeurIPS 2026 formatting requirements
argument-hint: <path/to/paper.tex>
allowed-tools: Read, Grep, Glob, Bash(ls:*)
---

# NeurIPS 2026 formatting check

Run a static lint of a LaTeX file against the NeurIPS 2026 template rules.

## Arguments
`$ARGUMENTS` — path to the `.tex` file (default: any `.tex` under `Formatting_Instructions_For_NeurIPS_2026/` that isn't a template stub).

## Steps

1. Read the `.sty` file in `Formatting_Instructions_For_NeurIPS_2026/` to confirm the current year's rules (page limits, font size, margin commands, anonymity flags).
2. Read the target `.tex` and check:
   - Uses `\documentclass` from the provided template — not a generic article class
   - `\usepackage{neurips_2026}` (or the exact year's package) is present
   - Anonymization: for submission, `\usepackage[final]{neurips_2026}` must NOT be set
   - Page limit respected (main paper ≤ template limit; references + appendix unlimited per NeurIPS policy, but appendix is separate)
   - No hand-tweaked `\textheight`, `\textwidth`, `\vspace` hacks to squeeze content
   - Fonts are template defaults (no `\usepackage{times}` if template forbids)
   - Figures have captions; tables use `\toprule \midrule \bottomrule` (booktabs)
   - No `\cite{?}` or `\ref{?}` dangling references (grep for `?`)
   - No TODO/FIXME/XXX comments in the compiled body
3. Report a checklist:
   - ✔ / ✘ per rule
   - For each ✘, give the file:line (if greppable) and the suggested fix
4. Do NOT edit the file. This command is read-only.

## Rules
- Do not guess the rule set — read it from the `.sty` every time, since NeurIPS updates rules yearly.
- If `.sty` and this checklist disagree, trust the `.sty`.
