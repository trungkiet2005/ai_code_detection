# CLAUDE.md

> Root entry point for Claude Code. Full repo context is in [docs/CLAUDE.md](docs/CLAUDE.md) — read it before doing anything non-trivial.

@docs/CLAUDE.md

---

## ⏱️ TL;DR — current state (2026-05-09 evening)

- **Submission target:** EMNLP 2026 (Main + reach for Oral). Deadline ~2026-05-26 (~17 days).
- **Safe floor (anchor / fallback for Main):** Exp_27 DeTeCtive **71.53** (CoDET Author full data) · Exp_13 NTKAlign **71.03** (lean 20%) · **NEW** Testing FS-NTKAlign **0.665** (5% data ≈ paper UniXcoder full) · Exp_04 Poincare **89.76** (Droid T3). Paper draft v1 builds on these.
- **Paper draft:** [Paper/latex/main.tex](Paper/latex/main.tex), 5 pages storytelling.
- **Active two-track exploration in `Exp_FewShot/`:**
  - **`testing/`** — setup-variation track. 14+ inline files (method × encoder × K × fraction). Already produced the 0.665 5%-data result. Free of novelty gate; build cells liberally to fill the data-efficiency matrix.
  - **`novel/`** — theory-driven oral track. Each file = one new mathematical object + falsifiable claim + theorem hook. Files numbered `exp_nNN_*.py`. Must pass the 5-gate Novelty Filter ([novel/README.md](Exp_FewShot/novel/README.md)) before landing. Currently: **`exp_n01_sibling_residual.py`** — Sibling-Residual Discriminant targeting K=32 codellama↔nxcode collapse via Hierarchical Neural Collapse (Galanti-Poggio 2025).
- **Excluded from headline:** AICD-Bench (universal val→test collapse), Zero-Shot suite (in `legacy/`, reproduction gap −32 pt vs paper FDG).

## 📐 Two-track policy (testing/ vs novel/)

| Question | testing/ | novel/ |
|:--|:--|:--|
| When to use | hyperparameter point, encoder swap, loss combo of existing pieces | new mathematical object with theorem |
| Filename | `exp_fs_*.py` (free naming) | `exp_nNN_<concept>.py` (sequential) |
| Header docstring | regular | mandatory: NAME / ONE-LINE CLAIM / EQUATION / THEORY HOOK / WHY NOT BEFORE / FALSIFIER |
| Gate | none | 5-gate Novelty Filter (see novel/README.md) |
| Anti-patterns | — | "stack 3 losses", λ-tuning, augmentation cocktail, "apply to new domain" alone |
| Failure handling | overwrite / iterate | move to `legacy/novel_failed_NN_*.py`; never delete |

## 🧪 Exp_FewShot/ — portfolio policy

Patterned after the cross-modal ReID workspace ([Exp_FewShot/README.md](Exp_FewShot/README.md)):

- **Each `exp_fs_NN_*.py` is fully self-contained.** Bootstrap auto-clones from GitHub if support modules are missing — Kaggle just runs `!python exp_fs_01_ntkalign.py`.
- **Two regimes per file** via env vars:
  - `FS_K_SHOT=N` (N ≥ 1) → K-shot mode (1 epoch, no LR schedule)
  - `FS_TRAIN_FRACTION=0.05` (with FS_K_SHOT=0) → %-fraction mode (3 epochs, cosine LR + warmup)
- **One file = one method.** Hyperparameters via env vars (`FS_LAMBDA_NTK`, `FS_TEMP`, `FS_LR_HEADS`, ...).
- **Sweep via `run_fs_portfolio.py`** — fires every (method, regime, value) into `logs/<exp>_<label>.log`.
- **Tracker is a portfolio MATRIX**, not a single leaderboard. Cells get filled as runs complete.
- **NEVER lock the paper headline before the matrix is at least half-filled.** The 20%-data Exp_13 number is the fallback; we are searching for whether a better cell exists.

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
