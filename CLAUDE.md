# CLAUDE.md

> Root entry point for Claude Code. Full repo context is in [docs/CLAUDE.md](docs/CLAUDE.md) — read it before doing anything non-trivial.

@docs/CLAUDE.md

---

## ⏱️ TL;DR — current state (2026-05-08)

- **Submission target:** EMNLP 2026 Main, deadline ~2026-05-26 (~18 days). Pivoted from NeurIPS 2026 Oral on 2026-05-08.
- **Locked headline numbers:** Exp_27 DeTeCtive **71.53** (CoDET Author full data, +5.20 vs UniXcoder) · Exp_13 NTKAlign **71.03** (lean 20%, +4.70) · Exp_04 Poincare **89.76** (Droid T3, +0.98).
- **Paper draft:** [Paper/latex/main.tex](Paper/latex/main.tex), 5 pages, storytelling, committed `2268691`. See [Paper/outline.md](Paper/outline.md) for day-by-day plan.
- **Active pivot:** few-shot K-sweep on `Exp_FewShot/` (T4-native, Phase A done `a0a0798`). Phase B Day 4 = Kaggle T4 K∈{8,16,32,64,128}. Day 5 = decision gate.
- **Excluded from headline:** AICD-Bench (universal val→test collapse), Zero-Shot suite (in `legacy/`, reproduction gap −32 pt vs paper FDG).

## Operational rules for this repo

1. **Execution lives on Kaggle, not locally.** Local machine = code + analyze. Kaggle = run. Use `/kaggle-cell` to emit cells; `/analyze-log` on pasted output. Do NOT run `python Exp_*/exp*.py` locally.
2. **Never mix benchmarks in one training run.** AICD, Droid, CoDET-M4 each get their own model.
3. **Primary metric is benchmark-specific.** Macro-F1 for AICD and CoDET-M4, Weighted-F1 for Droid. Do not substitute.
4. **Always report val-test gap** alongside test metrics.
5. **Experiment IDs are immutable.** Never reuse an `expNN` number. Never rewrite historical tracker rows; always append.
6. **Every experiment file is standalone.** No shared package imports across `Exp_DM/`, `Exp_CodeDet/`, `Exp_Climb/`, `Exp_FewShot/`.
7. **Hardware tiers:** `Exp_Climb/` H100 BF16 batch 64 · `Exp_FewShot/` T4 fp16 batch 16 seq 384 · auto-detect on both.
8. **Don't commit** `logs/`, `results/`, `codet_m4_checkpoints/`, `*.pt`, `*.bin`, LaTeX `*.aux/.log/.pdf` — gitignored for a reason.
9. **Templates are read-only.** Do not edit `Paper/latex/acl.sty` or `Formatting_Instructions_For_NeurIPS_2026/*.sty`.
10. **Headline numbers are locked.** Don't pivot scope or rerun experiments that would change Table 1/2 of the paper without explicit user OK.

## Source-of-truth files

- **Paper draft (THIS IS THE GOAL):** [Paper/latex/main.tex](Paper/latex/main.tex)
- Section outline + 18-day plan: [Paper/outline.md](Paper/outline.md)
- 20%-data climb leaderboard: [Exp_Climb/tracker.md](Exp_Climb/tracker.md)
- CoDET-M4 full-data leaderboard: [Exp_CodeDet/tracker.md](Exp_CodeDet/tracker.md)
- Few-shot K-sweep + decision gate: [Exp_FewShot/tracker.md](Exp_FewShot/tracker.md)
- AICD + Droid champions: [Exp_DM/dm_tracker.md](Exp_DM/dm_tracker.md)
- Paper references: [docs/references/](docs/references/)
- Archived suites + rationale: [legacy/README.md](legacy/README.md)

## Slash commands & agents

Custom tooling lives in [.claude/](.claude/). See [.claude/README.md](.claude/README.md) for what's available.
