# EMNLP 2026 Submission — Section Outline (1 page)

> **Title:** Data-Efficient AI-Code Authorship Detection via Hierarchical Family-Aware Learning
> **Deadline:** ~2026-05-26 (18 days from 2026-05-08)
> **Status:** Day 1 — skeleton + outline locked

---

## Locked headline numbers

| Bench / Task | Method | Score | Δ vs paper SOTA |
|:--|:--|:-:|:-:|
| **CoDET-M4 Author IID** (full) | Exp27 DeTeCtiveCode | **71.53** | +5.20 vs UniXcoder |
| **CoDET-M4 Author IID** (lean 20%) | Exp_13 NTKAlignCode | **71.03** | +4.70 vs UniXcoder |
| **DroidCollection T3** (lean) | Exp_04 PoincareGenealogy | **89.76** | +0.98 vs DroidDetectCLS-Large |

**Plus:** 20+ methods clustering above 70.0 on CoDET Author; 7+ methods beating DroidDetectCLS-Large on T3.

---

## Section structure (8 pages)

### §1 Introduction (1 page) — Day 7
- Hook: AI-code authorship is the hard task (binary saturated 99%, 6-class 66%).
- Setup: CoDET-M4 + DroidCollection define IID/OOD axes.
- Failure mode: 14+ methods cluster ~0.71 IID but collapse to ~0.36 on held-out GitHub source.
- Approach: hierarchical family prior + NTK alignment, 20% data.
- C1 method: +4.70 Macro-F1 with 20% data on CoDET-M4 Author IID.
- C2 cross-bench: +0.98 Weighted-F1 on DroidCollection T3.
- C3 diagnosis: source-confounding failure mode characterized via cross-method scatter.

### §2 Related Work (0.75 pg) — Day 8
- AI-code detection: GPTSniffer, CodeGPTSensor, CoDET-M4, DroidCollection, DeTeCtive, CodeMirage.
- Authorship as multi-class.
- NTK theory for OOD generalization (Jacot, Galanti-Poggio).

### §3 Method (1.5 pg) — Day 3
- 3.1 Problem formulation (6-class authorship, val/test/OOD splits).
- 3.2 Hierarchical family prior — genealogy graph over 11 model families.
- 3.3 NTK target-kernel alignment loss — Jacot+Galanti-Poggio motivation.
- 3.4 Multi-signal heads (neural + spectral + hier).
- 3.5 Lean curriculum: 20% stratified sampling, val/test full size.

### §4 Experiments (3 pg) — Day 4-5
- 4.1 Setup — H100 BF16, batch 64, seq 512, 3 epochs lean.
- 4.2 CoDET-M4 Authorship — Table 1 (top-5 ours + 3 paper baselines).
- 4.3 DroidCollection T3 — Table 2 (top-5 ours + DroidDetectCLS).
- 4.4 Data efficiency — Figure 1 (5/10/20/50/100% curve, NTKAlign).
- 4.5 Ablation — Table 3 (HierTree on/off, NTK on/off, Spectral on/off).

### §5 Analysis (1 pg) — Day 9
- 5.1 Source-confounding failure mode — train→test gap 0.71→0.36 on GH.
- 5.2 Sibling generator confusion — Qwen↔Nxcode bottleneck (44%↔42% F1).
- 5.3 Cross-method robustness — Figure 2 (Author F1 vs OOD-gh F1 scatter, 14+ methods).

### §6 Limitations (0.5 pg) — Day 10
- OOD-Source-gh remains hard (best 0.36).
- AICD-Bench out of scope (val→test universal collapse, separate dataset issue).
- 20% data choice not exhaustive — 5/10/50% curves in Figure 1 only.

### §7 Conclusion (0.25 pg) — Day 10

---

## Tables (4 + 2 figures)

| # | Content | Source data | Day |
|:-:|:--|:--|:-:|
| T1 | CoDET Author top-5 + 3 paper | Exp_Climb tracker + Exp_CodeDet tracker | Day 4 |
| T2 | Droid T3 top-5 + 3 paper | Exp_Climb + Exp_DM trackers | Day 5 |
| T3 | Ablation matrix (NTK / Hier / Spectral) | Exp_13 + Exp_15 + Exp_16 ablation logs | Day 5-6 |
| T4 | Per-class confusion (Qwen↔Nxcode) | Exp_27 detail logs | Day 9 |
| F1 | Data-efficiency curve | run NTKAlign at 5/10/50/100% — **OPTIONAL** if buffer day allows | Day 2 |
| F2 | Author F1 vs OOD-gh F1 scatter, 14+ methods | tracker — purely re-plot | Day 2 |

**F1 risk:** running 4 new lean ratios on Kaggle (~12h H100) — if Kaggle quota tight, fall back to extrapolation note.

---

## Critical-path risks

| Risk | Mitigation |
|:--|:--|
| Co-authors review chậm | Email Day 7 báo trước "drafts ready Day 12" |
| Kaggle quota | Avoid new experiments after Day 2; use existing logs |
| Reviewer "+5.20 too small" | Frame: 14+ methods cluster-break, paper cluster 62-66 |
| Reviewer "OOD not solved" | C3 frames it as diagnosis, not failed claim |
| OOD-paper-protocol loaders missing | Use LOO proxy framing, note in §4.1 |

---

## Action items today (after this commit)

- [x] Day 1: Verify numbers, set up paper/, outline locked
- [ ] Day 2: Plot Figure 2 (scatter) using existing logs — easy first
- [ ] Day 2: Decide if Figure 1 needs new runs (5/10/50/100%)
- [ ] Day 3: Method draft 1.5 pg
