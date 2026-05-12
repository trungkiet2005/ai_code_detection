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
> **The AICD-T2 results in the per-benchmark tables below for these experiments are INVALID (actually T1).**
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
| 🥉 | **HAP** | ModernBERT | 0.4359 | -0.227 |
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

### AICD (⚠️ T1-binary data — INVALID for T2; re-run pending) — Macro-F1

> 🚨 **All results below used T1 binary (2-class) data due to `_load_aicd()` bug in exp1–17.**
> **Do NOT use these for comparison vs T2 (12-class) benchmarks.**
> These scores look high (~0.93) because binary classification is trivially easy.
> Re-run with corrected T2 loader is pending for active experiments.

#### 1% Few-Shot

| Rank | Method | Encoder | Macro-F1 | Δ vs Paper |
|:----:|:-------|:--------|---------:|-----------:|
| 🥇 | **CodeT5-Authorship** | ModernBERT | **0.9401** | +0.277 |
| 🥈 | **ETF-Simplex** | ModernBERT | 0.9202 | +0.257 |
| 🥉 | **HAP** | unixcoder | 0.9270 | +0.264 |
| 4 | Hier-NTK (ours) | unixcoder | 0.9247 | +0.261 |
| 5 | NTK only | unixcoder | 0.9229 | +0.260 |
| 6 | HierTree only | unixcoder | 0.9249 | +0.262 |
| 7 | Mixup-MGA | unixcoder | 0.9248 | +0.261 |
| 8 | CE baseline (ours) | ModernBERT | 0.9135 | +0.250 |
| 9 | Focal-CE | ModernBERT | 0.8339 | +0.171 |
| 10 | LabelSmooth | ModernBERT | 0.9186 | +0.255 |
| 11 | Style-Repr | n/a | 0.7676 | +0.104 |

#### 5% Few-Shot

| Rank | Method | Encoder | Macro-F1 | Δ vs Paper |
|:----:|:-------|:--------|---------:|-----------:|
| 🥇 | **ETF-Simplex** | ModernBERT | **0.9644** | +0.301 |
| 🥈 | **CodeT5-Authorship** | ModernBERT | 0.9679 | +0.305 |
| 🥉 | **Hier-NTK (ours)** | ModernBERT | 0.9581 | +0.295 |
| 4 | NTK only | ModernBERT | 0.9566 | +0.293 |
| 5 | HAP | ModernBERT | 0.9622 | +0.299 |
| 6 | HierTree only | ModernBERT | 0.9550 | +0.292 |
| 7 | Mixup-MGA | ModernBERT | 0.9565 | +0.293 |
| 8 | CE baseline (ours) | ModernBERT | 0.9568 | +0.293 |
| 9 | FAID | ModernBERT | 0.9569 | +0.294 |
| 10 | DeTeCtive | ModernBERT | 0.9581 | +0.295 |
| 11 | Style-Repr | n/a | 0.7699 | +0.107 |

#### 20% Few-Shot

| Rank | Method | Encoder | Macro-F1 | Δ vs Paper |
|:----:|:-------|:--------|---------:|-----------:|
| 🥇 | **ETF-Simplex** | ModernBERT | **0.9772** | +0.314 |
| 🥈 | **CodeT5-Authorship** | ModernBERT | 0.9793 | +0.316 |
| 🥉 | **HAP** | ModernBERT | 0.9758 | +0.312 |
| 4 | Hier-NTK (ours) | ModernBERT | 0.9740 | +0.311 |
| 5 | CE baseline (ours) | ModernBERT | 0.9729 | +0.310 |
| 6 | NTK only | ModernBERT | 0.9736 | +0.310 |
| 7 | HierTree only | ModernBERT | 0.9733 | +0.310 |
| 8 | FAID | ModernBERT | 0.9737 | +0.310 |
| 9 | DeTeCtive | ModernBERT | 0.9725 | +0.309 |
| 10 | Style-Repr | n/a | 0.7704 | +0.107 |

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

## 🏆 Consolidated Leaderboard (All Methods vs CE Baseline)

### CoDET-M4 Author IID — Macro-F1

| Category | Method | Encoder | 1% | 5% | 20% | vs CE 5% |
|:---------|:-------|:--------|:--:|:--:|:---:|:--------:|
| **Published** | CodeT5-Authorship | ModernBERT | 0.4361 | **0.6030** | **0.6880** | **+3.05** |
| **Published** | DeTeCtive | ModernBERT | 0.3966 | 0.5855 | 0.6775 | +1.30 |
| **Published** | FAID | ModernBERT | 0.3919 | 0.5805 | 0.6722 | +0.80 |
| **Published** | Style-Repr | n/a | 0.3329 | 0.3365 | 0.3327 | -23.60 |
| **Novel** | ETF-Simplex | ModernBERT | 0.4040 | 0.5979 | 0.6826 | +2.54 |
| **Novel** | ETF-Simplex | unixcoder | **0.4405** | 0.6010 | 0.6700 | +2.85 |
| **Novel** | HAP | ModernBERT | 0.4359 | 0.5838 | 0.6698 | +1.13 |
| **Novel** | HAP | unixcoder | 0.4320 | 0.6010 | 0.6716 | +2.85 |
| **Novel** | Mixup-MGA | ModernBERT | 0.3984 | 0.5984 | 0.6774 | +2.59 |
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

| Category | Method | Encoder | 1% | 5% | 20% | vs CE 5% |
|:---------|:-------|:--------|:--:|:--:|:---:|:--------:|
| **Published** | CodeT5-Authorship | ModernBERT | **0.9401** | **0.9679** | **0.9793** | **+1.11** |
| **Published** | DeTeCtive | ModernBERT | 0.9155 | 0.9581 | 0.9725 | +0.13 |
| **Published** | FAID | ModernBERT | 0.6104 | 0.9569 | 0.9737 | -0.02 |
| **Published** | Style-Repr | n/a | 0.7676 | 0.7699 | 0.7704 | -18.69 |
| **Novel** | ETF-Simplex | ModernBERT | 0.9202 | 0.9644 | **0.9772** | +0.76 |
| **Novel** | ETF-Simplex | unixcoder | 0.6164 | 0.9595 | 0.9730 | +0.27 |
| **Novel** | HAP | ModernBERT | 0.8592 | 0.9622 | 0.9758 | +0.54 |
| **Novel** | HAP | unixcoder | 0.9270 | 0.9614 | 0.9754 | +0.46 |
| **Novel** | Mixup-MGA | ModernBERT | 0.9130 | 0.9565 | 0.9736 | -0.03 |
| **Novel** | Ortho-CLF | ModernBERT | 0.9142 | 0.9575 | 0.9735 | +0.07 |
| **Novel** | VIB | ModernBERT | 0.9111 | 0.9581 | 0.9741 | +0.13 |
| **Novel** | Mixup-CE | ModernBERT | 0.9168 | 0.9568 | 0.9745 | -0.00 |
| **Novel** | LabelSmooth | ModernBERT | 0.9186 | 0.9586 | 0.9744 | +0.18 |
| **Novel** | Focal-CE | ModernBERT | 0.8339 | 0.8751 | 0.8009 | -8.17 |
| **Ours** | NTK only | ModernBERT | 0.5955 | 0.9566 | 0.9736 | -0.02 |
| **Ours** | NTK only | unixcoder | 0.9229 | 0.9616 | 0.9733 | +0.48 |
| **Ours** | HierTree only | ModernBERT | 0.9067 | 0.9550 | 0.9733 | -0.18 |
| **Ours** | HierTree only | unixcoder | 0.9249 | 0.9610 | 0.9736 | +0.42 |
| **Ours** | **Hier-NTK** | ModernBERT | 0.9115 | 0.9581 | 0.9740 | +0.13 |
| **Ours** | **Hier-NTK** | unixcoder | 0.9247 | 0.9617 | 0.9739 | +0.49 |
| **Baseline** | CE baseline | ModernBERT | 0.9135 | 0.9568 | 0.9729 | — |
| **Baseline** | CE baseline | unixcoder | 0.9218 | 0.9598 | 0.9756 | +0.30 |
| **Theory-72** | CAO (best) | both | 0.0685* | — | — | n/a |
| **Theory-72** | DRL/IRM | unixcoder | 0.0397* | — | — | n/a |
| **Theory-72** | (all others) | — | <0.04* | — | — | n/a |

> \* fixed-72 protocol, not fraction-based — see Theory-Track Results section above.

---

## Key Insights

### 🔍 Consolidated Analysis (All 22 Methods)

1. **CodeT5-Authorship wins on CoDET-M4**: +3.05 pts vs CE baseline @ 5%, strongest across all fractions
   - ETF-Simplex close 2nd: +2.85 pts (unixcoder @ 5%)

2. **ETF-Simplex wins on AICD-T2**: +0.76 pts vs CE baseline @ 5%
   - Orthogonal/equiangular classifier heads help model-family separation

3. **Focal-CE catastrophically fails on AICD-T2**: -8.17 pts vs CE @ 5%
   - Degrades with more data (0.83 → 0.80 @ 20%)

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
| AICD-T2 @ 1% | CodeT5-Authorship | 0.9401 |
| AICD-T2 @ 5% | ETF-Simplex | 0.9644 |
| AICD-T2 @ 20% | CodeT5-Authorship | 0.9793 |

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
