# Testing Chis — Experiment Tracker

## Overview

Self-contained experiments for Hier-NTK ablation + published baselines.

**Config for all experiments:**
- Encoders: `ModernBERT-base`, `unixcoder-base` (2 encoders)
- Fractions: `0.01`, `0.05`, `0.20` (3 fractions)
- Benchmarks: `codet_m4`, `aicd_t2`, `droid_t3`, `droid_t4` (4 benchmarks)
- Batch: 256, seq=512

**Total per experiment file:** 24 runs (2 encoders × 4 benchmarks × 3 fractions)

---

## Our Methods (Ours)

| File | Method | Components | Runs |
|:--|:--|:--|:--|
| `exp_01_hier_tree.py` | HierTree only | Hierarchical family prior | 24 |
| `exp_02_ntk_align.py` | NTK only | NTK target-kernel alignment | 24 |
| `exp_03_hier_ntk.py` | **Hier-NTK** | HierTree + NTK combined | 24 |
| `exp_04_ce_baseline.py` | CE baseline | CrossEntropy only | 24 |

---

## Published Baselines

| File | Method | Paper | Runs |
|:--|:--|:--|:--|
| `baseline_01_codet5.py` | CodeT5-Authorship | AISec 2025 / arXiv 2506.17323 | 24 |
| `baseline_02_detective.py` | DeTeCtive | arXiv 2410.20964 | 24 |
| `baseline_03_faid.py` | FAID | EACL 2026 | 24 |
| `baseline_04_style.py` | Style-Repr | arXiv 2401.06712 | 12 (no encoder) |

---

## Usage

```bash
# Our methods (run first - they are the main contribution)
python exp_04_ce_baseline.py  # CE baseline
python exp_01_hier_tree.py    # HierTree only
python exp_02_ntk_align.py    # NTK only
python exp_03_hier_ntk.py     # Hier-NTK (MAIN METHOD)

# Published baselines (for comparison)
python baseline_01_codet5.py
python baseline_02_detective.py
python baseline_03_faid.py
python baseline_04_style.py
```

Output: `results/baseline_XX_results.json` / `results/ours_XX_results.json`

---

## Results

### CoDET-M4 Author IID (headline)

| Method | Encoder | 1% | 5% | 20% |
|:--|:--|-:|-:|-:|
| **Hier-NTK (ours)** | ModernBERT | - | - | - |
| **Hier-NTK (ours)** | unixcoder | - | - | - |
| HierTree only | ModernBERT | - | - | - |
| NTK only | ModernBERT | - | - | - |
| CE baseline (ours) | ModernBERT | - | - | - |
| CodeT5-Authorship | ModernBERT | - | - | - |
| DeTeCtive | ModernBERT | - | - | - |
| FAID | ModernBERT | - | - | - |

### AICD-T2 (stress test)

| Method | Encoder | 1% | 5% | 20% |
|:--|:--|-:|-:|-:|
| **Hier-NTK (ours)** | ModernBERT | - | - | - |
| **Hier-NTK (ours)** | unixcoder | - | - | - |

### Droid-T3 (detection support)

| Method | Encoder | 1% | 5% | 20% |
|:--|:--|-:|-:|-:|
| **Hier-NTK (ours)** | ModernBERT | - | - | - |
| **Hier-NTK (ours)** | unixcoder | - | - | - |

### Droid-T4 (adversarial robustness)

| Method | Encoder | 1% | 5% | 20% |
|:--|:--|-:|-:|-:|
| **Hier-NTK (ours)** | ModernBERT | - | - | - |
| **Hier-NTK (ours)** | unixcoder | - | - | - |

---

## Paper Reference

UniXcoder full-data baseline (CoDET-M4 Author IID): **0.6633 Macro-F1**
