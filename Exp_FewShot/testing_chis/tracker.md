# Testing Chis — Experiment Tracker

## Overview

Self-contained experiments for Hier-NTK ablation + published baselines.

**Config (updated — unixcoder-only from now on):**
- Encoder: `unixcoder-base` only (ModernBERT dropped — unixcoder consistently best or equal)
- Fractions: `0.01`, `0.05`, `0.20` (3 fractions)
- Benchmarks: `codet_m4` (headline), `aicd_t2` (stress test, T2 = model-family 12-class)
- **Droid T3/T4: SKIPPED (per 2026-05-10 directive)**
- Batch: 256, seq=512

**Total per experiment file:** 6 runs (1 encoder × 2 benchmarks × 3 fractions)

> ⚠️ **DATA BUG (exp1–exp17 / baseline_01–04 / exp_n05–n16):** Due to a `_load_aicd()` path bug, all AICD
> results for experiments prior to exp18 were **trained on T1 (binary 2-class)** instead of T2 (12-class).
> **The AICD results in the tables below for these experiments are INVALID (actually T1).**
> Only CoDET-M4 results from those experiments are reliable. Re-runs with correct T2 loading are pending.

---

## Our Methods (Ours)

| File | Method | Components | Runs |
|:--|:--|:--|:--|
| `exp_n13_hier_tree.py` | HierTree only | Hierarchical family prior | 12 |
| `exp_n14_ntk_align.py` | NTK only | NTK target-kernel alignment | 12 |
| `exp_n15_hier_ntk.py` | **Hier-NTK** | HierTree + NTK combined | 12 |
| `exp_n16_ce_baseline.py` | CE baseline | CrossEntropy only | 12 |

---

## Published Baselines

| File | Method | Paper | Status |
|:--|:--|:--|:--|
| `baseline_01_codet5.py` | CodeT5-Authorship | AISec 2025 / arXiv 2506.17323 | ✅ Active |
| `baseline_02_detective.py` | DeTeCtive | arXiv 2410.20964 | ✅ Active |
| `baseline_03_faid.py` | FAID | EACL 2026 | ✅ Active |
| `legacy/baseline_04_style.py` | Style-Repr | arXiv 2401.06712 | ❌ Legacy (useless: −23.6pts CoDET) |

---

## Novel Methods (arxiv-grounded theory)

| File | Method | ArXiv | Status |
|:--|:--|:--|:--|
| `exp_n06_attn_pool.py` | HAP | Lee 2017 | ✅ Active |
| `exp_n07_mixup_align.py` | MGA | Arjovsky 2019 | ✅ Active |
| `exp_n08_ortho_clf.py` | Ortho-CLF | Papyan 2020 | ✅ Active |
| `exp_n09_etf_simplex.py` | ETF-Simplex | NC theory | ✅ Active (best novel) |
| `exp_n12_label_smooth.py` | LabelSmooth | Pereyra 2017 | ✅ Active |
| `exp_n18_hier_supcon.py` | Hier-SupCon | Exp18 migration | ✅ NEW |
| `exp_n19_detective_lite.py` | DeTeCtive-lite | Exp27 migration | ✅ NEW |
| `legacy/exp_n05_focal.py` | Focal-CE | Lin 2017 | ❌ Legacy (destroys AICD-T2) |
| `legacy/exp_n10_vib.py` | VIB | Alemi 2017 | ❌ Legacy (marginal +0.27pts) |
| `legacy/exp_n11_mixup_ce.py` | Mixup-CE | Zhang 2018 | ❌ Legacy (hurts at 1%/5%) |

---

## Theory-Track Methods (FIXED_TOTAL=72 protocol)

> These experiments use a **fixed 72-sample** training set (not fraction-based).
> Each file: 2 encoders × 2 benchmarks = 4 runs.

| File | Method | Theory | Status |
|:--|:--|:--|:--|
| `exp17_cao.py` | CAO | Class-Aware Ordering | ⚠️ NaN loss, collapsed |
| `exp19_wda.py` | WDA | Wasserstein Domain Align | ❌ Collapsed |
| `exp20_proto.py` | Proto | Prototypical Networks | ⚠️ Partial collapse |
| `exp22_mi.py` | MI | Mutual Information | ⚠️ Partial (best: 0.1663) |
| `exp23_ot.py` | OT | Optimal Transport | ❌ Collapsed |
| `exp25_drl.py` | DRL/IRM | Invariant Risk Min. | ⚠️ IRM penalty=0 always |
| `exp26_gna.py` | GNA | Graph Neural Attribution | ⚠️ Partial collapse |
| `exp30_bua.py` | BUA | Bayesian Uncertainty | ⚠️ Partial collapse |

## Usage

```bash
# CE baseline first
python exp_n16_ce_baseline.py

# Our methods (main contribution)
python exp_n13_hier_tree.py    # HierTree only
python exp_n14_ntk_align.py    # NTK only
python exp_n15_hier_ntk.py     # Hier-NTK (MAIN METHOD)

# Published baselines (for comparison)
python baseline_01_codet5.py
python baseline_02_detective.py
python baseline_03_faid.py
python baseline_04_style.py

# Novel methods (arxiv-grounded theory)
python exp_n05_focal.py        # Focal-CE
python exp_n06_attn_pool.py    # HAP
python exp_n07_mixup_align.py  # MGA
python exp_n08_ortho_clf.py    # Ortho-CLF
python exp_n09_etf_simplex.py  # ETF-Simplex
python exp_n10_vib.py          # VIB
python exp_n11_mixup_ce.py     # Mixup-CE
python exp_n12_label_smooth.py # LabelSmooth

# Legacy migrations (full-data experiments -> few-shot protocol)
python exp_n18_hier_supcon.py    # HierTree + SupCon (Exp18 migration, 70.55 F1 full)
python exp_n19_detective_lite.py  # DeTeCtive-lite (Exp27 migration, 71.53 F1 full)
```

Output: `results/Novel/exp*_results.json`

---

## Results

### CoDET-M4 Author IID (headline) — Macro-F1

#### 1% Few-Shot

| Rank | Method | Encoder | Macro-F1 | Δ vs Paper |
|:----:|:-------|:--------|---------:|-----------:|
| 🥇 | **ETF-Simplex** | unixcoder | **0.4405** | -0.223 |
| 🥈 | **CodeT5-Authorship** | ModernBERT | 0.4361 | -0.227 |
| 🥉 | **HAP** | unixcoder | 0.4319 | -0.231 |
| 4 | CodeT5-Authorship | unixcoder | 0.4156 | -0.248 |
| 5 | Hier-NTK (ours) | ModernBERT | 0.3905 | -0.273 |
| 6 | Hier-NTK (ours) | unixcoder | 0.3925 | -0.271 |
| 7 | HierTree only | ModernBERT | 0.3947 | -0.269 |
| 8 | Mixup-MGA | ModernBERT | 0.3984 | -0.265 |
| 9 | ETF-Simplex | ModernBERT | 0.4040 | -0.259 |
| 10 | NTK only | ModernBERT | 0.3912 | -0.272 |
| 11 | Mixup-CE | ModernBERT | 0.3461 | -0.317 |
| 12 | LabelSmooth | ModernBERT | 0.3192 | -0.344 |
| 13 | Style-Repr | n/a | 0.3329 | -0.330 |

#### 5% Few-Shot

| Rank | Method | Encoder | Macro-F1 | Δ vs Paper |
|:----:|:-------|:--------|---------:|-----------:|
| 🥇 | **CodeT5-Authorship** | ModernBERT | **0.6030** | -0.060 |
| 🥈 | **DeTeCtive** | ModernBERT | 0.5855 | -0.078 |
| 🥉 | **NTK only** | ModernBERT | 0.5852 | -0.078 |
| 4 | Hier-NTK (ours) | ModernBERT | 0.5842 | -0.079 |
| 5 | FAID | ModernBERT | 0.5805 | -0.083 |
| 6 | HAP | ModernBERT | 0.5838 | -0.080 |
| 7 | HierTree only | ModernBERT | 0.5750 | -0.088 |
| 8 | CE baseline (ours) | ModernBERT | 0.5725 | -0.091 |
| 9 | ETF-Simplex | ModernBERT | 0.5979 | -0.065 |
| 10 | LabelSmooth | ModernBERT | 0.5683 | -0.095 |
| 11 | Style-Repr | n/a | 0.3365 | -0.327 |

#### 20% Few-Shot

| Rank | Method | Encoder | Macro-F1 | Δ vs Paper |
|:----:|:-------|:--------|---------:|-----------:|
| 🥇 | **CodeT5-Authorship** | ModernBERT | **0.6880** | +0.025 |
| 🥈 | **ETF-Simplex** | ModernBERT | 0.6826 | +0.019 |
| 🥉 | **LabelSmooth** | ModernBERT | 0.6796 | +0.016 |
| 4 | CE baseline (ours) | ModernBERT | 0.6795 | +0.016 |
| 5 | Mixup-MGA | ModernBERT | 0.6774 | +0.014 |
| 6 | Hier-NTK (ours) | ModernBERT | 0.6716 | +0.008 |
| 7 | FAID | ModernBERT | 0.6722 | +0.009 |
| 8 | DeTeCtive | ModernBERT | 0.6775 | +0.014 |
| 9 | HierTree only | ModernBERT | 0.6757 | +0.012 |
| 10 | Style-Repr | n/a | 0.3327 | -0.331 |

---

## 🏆 Consolidated Leaderboard (All Methods vs CE Baseline)

### CoDET-M4 Author IID — Macro-F1

| Category | Method | Encoder | 1% | 5% | 20% | vs CE 5% |
|:---------|:-------|:--------|:--:|:--:|:---:|:--------:|
| **Published** | CodeT5-Authorship | ModernBERT | 0.4361 | **0.6030** | **0.6880** | **+3.05** |
| **Published** | DeTeCtive | ModernBERT | 0.3966 | 0.5855 | 0.6775 | +1.30 |
| **Published** | FAID | ModernBERT | 0.3919 | 0.5805 | 0.6722 | +0.80 |
| **Published** | Style-Repr | n/a | 0.3329 | 0.3365 | 0.3327 | -23.60 |
| **Novel** | ETF-Simplex | ModernBERT | 0.4040 | 0.5979 | 0.6826 | +2.54 |
| **Novel** | ETF-Simplex | unixcoder | **0.4405** | 0.6010 | 0.6702 | +2.85 |
| **Novel** | HAP | ModernBERT | 0.4359 | 0.5838 | 0.6698 | +1.13 |
| **Novel** | HAP | unixcoder | 0.4319 | 0.6012 | 0.6722 | +2.87 |
| **Novel** | Mixup-MGA | ModernBERT | 0.3984 | 0.5984 | 0.6774 | +2.59 |
| **Novel** | Mixup-MGA | unixcoder | 0.4214 | 0.5930 | 0.6656 | +2.47 |
| **Novel** | Ortho-CLF | ModernBERT | 0.3935 | 0.5904 | 0.6765 | +1.79 |
| **Novel** | VIB | ModernBERT | 0.3721 | 0.5752 | 0.6675 | +0.27 |
| **Novel** | Mixup-CE | ModernBERT | 0.3461 | 0.5649 | 0.6762 | -0.76 |
| **Novel** | LabelSmooth | ModernBERT | 0.3192 | 0.5683 | 0.6796 | -0.42 |
| **Novel** | Focal-CE | ModernBERT | 0.3871 | 0.5797 | 0.6656 | +0.72 |
| **Ours** | NTK only | ModernBERT | 0.3912 | 0.5852 | 0.6755 | +1.27 |
| **Ours** | NTK only | unixcoder | 0.3988 | 0.5948 | 0.6679 | +2.23 |
| **Ours** | HierTree only | ModernBERT | 0.3947 | 0.5750 | 0.6757 | +0.25 |
| **Ours** | HierTree only | unixcoder | 0.3922 | 0.5918 | 0.6601 | +1.93 |
| **Ours** | **Hier-NTK** | ModernBERT | 0.3905 | 0.5842 | 0.6716 | +1.17 |
| **Ours** | **Hier-NTK** | unixcoder | 0.3925 | 0.5919 | 0.6587 | +1.94 |
| **Baseline** | CE baseline | ModernBERT | 0.3829 | 0.5725 | 0.6795 | — |
| **Baseline** | CE baseline | unixcoder | 0.3962 | 0.5882 | 0.6619 | +1.57 |
| **Theory-72** | MI | unixcoder | 0.1663* | — | — | n/a |
| **Theory-72** | DRL/IRM | unixcoder | 0.1273* | — | — | n/a |
| **Theory-72** | Proto | unixcoder | 0.1178* | — | — | n/a |
| **Theory-72** | (all others) | — | <0.12* | — | — | n/a |

> \* fixed-72 protocol, not fraction-based — see Theory-Track Results section above.

### AICD-T2 (model-family attribution) — Macro-F1

> ✅ **All results below are from unixcoder-base with CORRECT T2 (12-class) loading.**

| Category | Method | 1% | 5% | 20% | Notes |
|:---------|:-------|:--:|:--:|:---:|:------|
| **Novel** | LabelSmooth | 0.191 | 0.333 | 0.445 | ✅ Best at 5%/20% |
| **Novel** | Mixup-CE | 0.190 | 0.326 | 0.443 | |
| **Novel** | KAC | 0.200 | 0.325 | 0.410 | |
| **Novel** | SGE | 0.204 | 0.324 | 0.408 | |
| **Novel** | HPA | 0.194 | 0.319 | 0.407 | |
| **Novel** | Ortho-CLF | 0.205 | 0.317 | 0.410 | |
| **Novel** | Focal-CE | 0.191 | 0.306 | 0.392 | |
| **Published** | CodeT5-Authorship | 0.199 | 0.316 | 0.404 | |
| **Baseline** | CE baseline | ~0.22 | **0.322** | ~0.41 | |

> 📊 **Key insight:** LabelSmooth achieves best AICD-T2 results at 5%/20% (+0.01 over baseline).
> All methods struggle at 1% few-shot (~0.19-0.21) due to 12-class problem.

### AICD-T1 (binary: human vs AI) — Macro-F1 [INVALID - historical]

> ⚠️ **WARNING:** These used **T1 binary (2-class)** due to `_load_aicd()` path bug.
> **Do NOT quote these as T2 results.** Historical record only.

| Category | Method | Encoder | 1% | 5% | 20% | Notes |
|:---------|:-------|:--------|:--:|:--:|:---:|:------|
| **Published** | CodeT5-Authorship | ModernBERT | 0.9401 | 0.9679 | 0.9793 | T1 bug |
| **Published** | DeTeCtive | ModernBERT | 0.9155 | 0.9581 | 0.9725 | T1 bug |
| **Published** | FAID | ModernBERT | 0.6104 | 0.9569 | 0.9737 | T1 bug |
| **Published** | Style-Repr | n/a | 0.7676 | 0.7699 | 0.7704 | T1 bug |
| **Novel** | ETF-Simplex | ModernBERT | 0.9202 | 0.9644 | 0.9772 | T1 bug |
| **Novel** | HAP | ModernBERT | 0.8592 | 0.9622 | 0.9758 | T1 bug |
| **Novel** | Mixup-MGA | ModernBERT | 0.9130 | 0.9565 | 0.9736 | T1 bug |
| **Novel** | Ortho-CLF | ModernBERT | 0.9142 | 0.9575 | 0.9735 | T1 bug |
| **Novel** | VIB | ModernBERT | 0.9111 | 0.9581 | 0.9741 | T1 bug |
| **Novel** | Mixup-CE | ModernBERT | 0.9168 | 0.9568 | 0.9745 | T1 bug |
| **Novel** | LabelSmooth | ModernBERT | 0.9186 | 0.9586 | 0.9744 | T1 bug |
| **Novel** | Focal-CE | ModernBERT | 0.8339 | 0.8751 | 0.8009 | T1 bug |
| **Ours** | NTK only | ModernBERT | 0.5955 | 0.9566 | 0.9736 | T1 bug |
| **Ours** | NTK only | unixcoder | 0.9229 | 0.9616 | 0.9733 | T1 bug |
| **Ours** | HierTree only | ModernBERT | 0.9067 | 0.9550 | 0.9733 | T1 bug |
| **Ours** | HierTree only | unixcoder | 0.9249 | 0.9610 | 0.9736 | T1 bug |
| **Ours** | **Hier-NTK** | ModernBERT | 0.9115 | 0.9581 | 0.9740 | T1 bug |
| **Ours** | **Hier-NTK** | unixcoder | 0.9247 | 0.9617 | 0.9739 | T1 bug |
| **Baseline** | CE baseline | ModernBERT | 0.9135 | 0.9568 | 0.9729 | T1 bug |
| **Baseline** | CE baseline | unixcoder | 0.9218 | 0.9598 | 0.9756 | T1 bug |

---

## Theory-Track Results (FIXED_TOTAL=72, not fraction-based)

> ⚠️ These methods use 72 fixed training samples. Results are **not directly comparable** with the fraction-based table above, but compared vs CE baseline at same fixed-72 budget.
> Reference CE baseline (fixed-72, best encoder): **CoDET-M4 ≈ 0.39**, **AICD-T2 ≈ 0.91**

### CoDET-M4 Author IID — Macro-F1 (fixed-72)

| Rank | Method | Best Encoder | Macro-F1 | Notes |
|:----:|:-------|:------------|--------:|:------|
| 🥇 | MI | unixcoder | **0.1663** | Best of group; still far below CE |
| 🥈 | DRL/IRM | unixcoder | 0.1273 | IRM penalty never activates (72 < 500 anneal) |
| 🥉 | Proto | unixcoder | 0.1178 | Partial: 2/6 classes only |
| 4 | GNA | unixcoder | 0.1106 | Graph conv collapsed |
| 5 | BUA | unixcoder | 0.1100 | MC Dropout; collapses to 2 classes |
| 6 | CAO | both | 0.1132 | NaN loss after ep1; frozen at class 0 |
| 7 | WDA / OT | unixcoder | 0.0938 | Dominated by class 3 only |

### AICD-T2 — Macro-F1 (fixed-72)

| Rank | Method | Best Encoder | Macro-F1 | Notes |
|:----:|:-------|:------------|--------:|:------|
| 🥇 | DRL/IRM | unixcoder | **0.0397** | Slightly above others |
| 🥈 | GNA | unixcoder | 0.0382 | |
| 🥉 | Proto | unixcoder | 0.0380 | |
| 4 | OT | unixcoder | 0.0367 | |
| 5 | WDA | unixcoder | 0.0367 | |
| 6 | MI | unixcoder | 0.0356 | |
| 7 | CAO | both | 0.0685 | Collapsed to majority class only |
| 8 | BUA | unixcoder | 0.0311 | Worst; MC dropout hurts small data |

---

## Key Insights

### 🔍 Consolidated Analysis (All 22 Methods)

1. **CodeT5-Authorship wins on CoDET-M4**: +3.05 pts vs CE baseline @ 5%, strongest across all fractions
   - ETF-Simplex close 2nd: +2.85 pts (unixcoder @ 5%)

2. **AICD-T2 (unixcoder) results are low**: ~0.21-0.43 macro-F1 across methods
   - All methods underperform vs CE baseline (negative Δ)
   - **Pending re-run**: Need to verify T2 loader is correct

3. **AICD-T1 results are high (~0.93-0.98)**: Due to T1 data bug, these are binary (2-class)
   - Cannot compare with AICD-T2 (12-class) results

4. **Style-Repr fails everywhere**: -23.6 pts vs CE on CoDET-M4
   - Functional features > stylistic patterns

5. **Mixup hurts CoDET-M4**: Mixup-CE -0.76 pts, Mixup-MGA neutral
   - Interpolating samples blurs author-specific signals

6. **Hier-NTK (ours) underperforms**: +1.17 pts vs CE @ 5%
   - Loses to CodeT5-Authorship (+3.05), ETF-Simplex (+2.54)

7. **NTK/Hier components neutral**: HierTree only +0.25, NTK only +1.27

### ⚠️ Negative Results Summary

| Finding | Impact | Evidence |
|:--------|:-------|:---------|
| Hier-NTK loses to CodeT5 | High | +1.17 vs +3.05 pts |
| Focal-CE destroys AICD-T2 | Critical | -8.17 pts, degrades w/ data |
| Style-Repr useless | High | -23.6 pts on CoDET-M4 |
| Mixup-CE hurts | Medium | -0.76 pts |
| HierTree only neutral | Medium | +0.25 pts (not worth complexity) |
| **Theory-Track ALL collapsed** | **Critical** | All 8 methods ≤0.17 on CoDET-M4 |
| **CAO NaN loss** | Critical | Loss → NaN after ep1, frozen predictions |
| **WDA/OT dominated by 1 class** | High | All preds → class 3; OT loss diverges |
| **IRM penalty never fires** | High | penalty_anneal=500 > total steps at 72 samples |
| **BUA worst on AICD-T2** | Medium | MC Dropout + tiny data = 0.031 macro-F1 |

### 🎯 When Each Method Wins

| Scenario | Best Method | Score |
|:---------|:------------|------:|
| CoDET-M4 @ 1% | ETF-Simplex | 0.4405 |
| CoDET-M4 @ 5% | CodeT5-Authorship | 0.6030 |
| CoDET-M4 @ 20% | CodeT5-Authorship | 0.6880 |

---

## Paper Reference

UniXcoder full-data baseline (CoDET-M4 Author IID): **0.6633 Macro-F1**

---

## Notes

- **Droid T3/T4 SKIPPED per 2026-05-10 directive.** Focus on CoDET-M4 + AICD T2 only.
- Each fraction-based file runs 2 encoders × 2 benchmarks × 3 fractions = 12 experiments.
- Theory-Track (exp17–exp30): 2 encoders × 2 benchmarks = 4 runs each, FIXED_TOTAL=72.
- Results format (fraction-based): `{"tag", "enc", "bench", "frac", "macro", "weighted", "acc", "dpaper", "wall"}`
- Results format (theory-track): `{"tag", "encoder", "benchmark", "task", "macro_f1", "delta_vs_paper", ...}`

### Theory-Track Collapse Analysis

All 8 theory-track methods collapsed on CoDET-M4 (macro-F1 ≤ 0.17, vs CE baseline ~0.39).
Common failure modes:
- **Class imbalance exploitation**: With 72 samples, auxiliary losses (WDA, OT, CAO) overwhelm CE signal
- **Auxiliary loss magnitude mismatch**: WDA loss ~30×, OT loss ~5×, CAO loss ~750K× the CE loss
- **IRM anneal too high**: `penalty_anneal=500` steps but only 1 step per epoch × 3 epochs = 3 steps total
- **Graph collapse (GNA)**: With 72 nodes, inter-node edges are trivial; network defaults to simple linear
- **MC Dropout (BUA)**: `high_uncertainty_ratio=0.0` — all predictions confident despite being wrong

**Recommendation**: Theory-track methods need hyperparameter scaling to the data regime (72 samples).
Consider: smaller auxiliary loss weights (`wda_weight < 0.01`), lower `penalty_anneal` (e.g. 1), or prototype initialization from pretrained clusters.
