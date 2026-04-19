# CLAUDE.md

> Root entry point for Claude Code. Full repo context is in [docs/CLAUDE.md](docs/CLAUDE.md) — read it before doing anything non-trivial.

@docs/CLAUDE.md

---

## Operational rules for this repo

1. **Execution lives on Kaggle, not locally.** The user's local machine is for coding, scaffolding, and analysis only — no GPU, no training. The loop is: (1) Claude edits `expNN_*.py` locally, (2) user uploads/clones to a Kaggle H100 notebook and runs it there, (3) user pastes the log back into Claude for analysis. Do NOT run `python Exp_*/exp*.py`, `pytest` on training loops, or any compute-heavy script locally. Use `/kaggle-cell` to emit copy-paste cells for Kaggle; use `/analyze-log` on the pasted output.
2. **Never mix benchmarks in one training run.** AICD, Droid, CoDET-M4 each get their own model.
3. **Primary metric is benchmark-specific.** Macro-F1 for AICD and CoDET-M4, Weighted-F1 for Droid. Do not substitute.
4. **Always report val-test gap** alongside test metrics — the OOD collapse is the paper's headline signal.
5. **Experiment IDs are immutable.** Never reuse an `expNN` number. Never rewrite historical tracker rows; always append.
6. **Every experiment file is standalone.** No shared package imports across `Exp_DM/`, `Exp_CodeDet/`, `Exp_Climb/`.
7. **Kaggle H100 BF16 is the target.** Effective batch 64. Scripts auto-install `tree-sitter`, `tree-sitter-languages`.
8. **Don't commit** `logs/`, `results/`, `codet_m4_checkpoints/`, `*.pt`, `*.bin` — they are gitignored for a reason.
9. **NeurIPS template (`Formatting_Instructions_For_NeurIPS_2026/`) is read-only.** Do not edit `.sty` files.

## Source-of-truth files

- Results (AICD + Droid): [Exp_DM/dm_tracker.md](Exp_DM/dm_tracker.md)
- Results (CoDET-M4): [Exp_CodeDet/tracker.md](Exp_CodeDet/tracker.md)
- Climb / EMNLP 2026 experiments: [Exp_Climb/tracker.md](Exp_Climb/tracker.md)
- Baseline implementation: [Exp_TK/exp00_codeorigin.py](Exp_TK/exp00_codeorigin.py)
- Paper references: [docs/references/](docs/references/)

## Slash commands & agents

Custom tooling lives in [.claude/](.claude/). See [.claude/README.md](.claude/README.md) for what's available.
