# Implementation Plan - CoDET-M4 Benchmark Tracker

Create a premium, visually rich `tracker.md` to document the performance of **SpectralCode** on the `CoDET-M4` dataset (~500K samples) and compare results directly against the ACL 2025 paper (Orel et al.).

## User Review Required

> [!IMPORTANT]
> The current logs represent an `iid_only` run. The OOD (Out-of-Domain) benchmarks for Source, Language, and Generator are identified as proxies in the paper. I will mark these as **Pending/Proxy Results** in the tracker.

## Comparison Baseline Mapping

| Eval Mode (CoDET-M4) | Paper Baseline (UniXcoder) | SpectralCode (Current) | Gain |
|----------------------|---------------------------|-------------------------|------|
| **iid_binary**       | 98.65 (Table 2)           | **99.06**               | +0.41% |
| **iid_author**       | 66.33 (Table 7)           | **69.82**               | +3.49% |

### Breakdown Highlights (Binary)
- **Language**: C++ (99.09), Java (99.54), Python (98.61) vs Paper (98.24, 99.02, 98.60).
- **Source**: CodeForces (98.39), GitHub (98.38), LeetCode (98.69) vs Paper (96.54, 98.46, 97.87).

## Proposed Changes

### [Exp_CodeDet](file:///d:/ai_code_detection/Exp_CodeDet)

#### [NEW] [tracker.md](file:///d:/ai_code_detection/Exp_CodeDet/tracker.md)
- **Aesthetic**: Modern BERT blue/cyan theme, sleek tables, and glassmorphism-inspired layout (simulated in Markdown).
- **Deep-Dive**: Including confusion matrix highlights: **Nxcode <=> Qwen1.5 confusion** noted in paper is confirmed (+ major improvement on individual model recall).

## Open Questions

1. Should I also update the main `Exp_DM/dm_tracker.md` with a summary of these CoDET-M4 results?
2. Do you have official OOD results, or should I just reference the Paper Table 8/9/10 as the baseline to beat?

## Verification Plan

### Manual Verification
- Review the generated `tracker.md` in the VS Code preview.
- Ensure all 47,744 test samples are accounted for in the report.
