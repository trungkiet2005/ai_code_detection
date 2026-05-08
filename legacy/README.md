# legacy/ — Archived Experiment Suites

> **Read-only.** Content here is preserved for git history + provenance only. **Do not run, edit, or import from these paths.** Active suites live in `Exp_Climb/`, `Exp_CodeDet/`, `Exp_DM/` (5 champions only).

Archived on 2026-05-08 as part of the EMNLP Main 2026 scope-tightening (CoDET-M4 Author IID + Droid T3 only).

---

## Contents

### `Exp_ZeroShot/` — 31 zero-shot detectors (DROPPED)
- 26 ZS methods spanning Bayesian curvature, signature, OT, quantum-info, criticality, etc.
- Dropped because **0/31 methods beat paper Fast-DetectGPT 64.54** on Droid T3.
- Reproduction gap: our FDG run = 32.07 W-F1 vs paper 64.54 → **−32.33 pt**. Within-suite Δ unciteable until that gap is closed (Exp_31 paper-exact attempt was the last, never finished).
- Not used in EMNLP Main paper. Could anchor a separate "reproduction gap as finding" workshop paper.

### `Exp_TK/` — Consolidated champion folder (REDUNDANT)
- Originally a sandbox to copy champion files from Exp_DM / Exp_Climb / Exp_CodeDet into one place.
- Redundant once each source tracker became the canonical record.
- Champions still live in their original suites — Exp_TK was just a mirror.

### `Exp_DM_weak/` — 21 underperformers from Exp_DM (DROPPED)
Files moved: `exp01–06`, `exp08`, `exp10`, `exp12`, `exp19–30`.
- Most never produced final numbers (status: `Pending` in `dm_tracker.md`).
- exp27/28/29/30 (DeTeCtive variants in DM) are duplicates of the canonical `Exp_CodeDet/run_codet_m4_exp27_detective.py`.
- 5 champions kept in active `Exp_DM/`: exp07 DomainMix, exp09 TokenStat, exp11 SpectralCode, exp13 SlotCode, exp18 HierTreeCode.

### `experiment/` — Pre-Exp_DM legacy baselines
- Early-phase exp00–04 baselines from the project's first month.
- exp00_codeorigin.py is still the conceptual root, but Exp_Climb's `_model.py` + `_trainer.py` is now canonical.

### `performance_tracker.md`
- Pre-split tracker, superseded by per-suite trackers (`Exp_Climb/tracker.md`, `Exp_DM/dm_tracker.md`, `Exp_CodeDet/tracker.md`).

---

## Why this archive exists

EMNLP Main 2026 scope (chốt 2026-05-08):
- **PRIMARY:** CoDET-M4 Author IID Macro-F1 — best **71.53** (Exp27 DeTeCtive full) / **71.03** (Exp_13 NTK lean 20% data) vs UniXcoder 66.33 → +5.20 / +4.70
- **SECONDARY:** Droid T3 Weighted-F1 — best **89.76** (Exp_04 Poincare) vs DroidDetectCLS-Large 88.78 → +0.98
- **DROP:** AICD-Bench (universal val→test collapse, no SOTA to beat → appendix discussion only)
- **DROP:** Zero-shot suite (reproduction gap)

Anything not contributing evidence to those four bullets was archived here.
