# Testing Chis — Experiment Tracker

## Overview

Self-contained experiments for Hier-NTK ablation + published baselines.

**Config for all experiments:**
- Encoders: `ModernBERT-base`, `unixcoder-base` (2 encoders)
- Fractions: `0.01`, `0.05`, `0.20` (3 fractions)
- Benchmarks: `codet_m4` (headline), `aicd_t2` (stress test)
- **Droid T3/T4: SKIPPED (per 2026-05-10 directive)**
- Batch: 256, seq=512

**Total per experiment file:** 12 runs (2 encoders × 2 benchmarks × 3 fractions)

---

## Our Methods (Ours)

| File | Method | Components | Runs |
|:--|:--|:--|:--|
| `exp_01_hier_tree.py` | HierTree only | Hierarchical family prior | 12 |
| `exp_02_ntk_align.py` | NTK only | NTK target-kernel alignment | 12 |
| `exp_03_hier_ntk.py` | **Hier-NTK** | HierTree + NTK combined | 12 |
| `exp_04_ce_baseline.py` | CE baseline | CrossEntropy only | 12 |

---

## Published Baselines

| File | Method | Paper | Runs |
|:--|:--|:--|:--|
| `baseline_01_codet5.py` | CodeT5-Authorship | AISec 2025 / arXiv 2506.17323 | 12 |
| `baseline_02_detective.py` | DeTeCtive | arXiv 2410.20964 | 12 |
| `baseline_03_faid.py` | FAID | EACL 2026 | 12 |
| `baseline_04_style.py` | Style-Repr | arXiv 2401.06712 | 6 (no encoder) |

---

## Novel Methods (arxiv-grounded theory)

| File | Method | ArXiv | Runs |
|:--|:--|:--|:--|
| `exp_n05_focal.py` | Focal-CE | Lin 2017 | 12 |
| `exp_n06_attn_pool.py` | HAP | Lee 2017 | 12 |
| `exp_n07_mixup_align.py` | MGA | Arjovsky 2019 | 12 |
| `exp_n08_ortho_clf.py` | Ortho-CLF | Papyan 2020 | 12 |
| `exp_n09_etf_simplex.py` | ETF-Simplex | NC theory | 12 |
| `exp_n10_vib.py` | VIB | Alemi 2017 | 12 |
| `exp_n11_mixup_ce.py` | Mixup-CE | Zhang 2018 | 12 |
| `exp_n12_label_smooth.py` | LabelSmooth | Pereyra 2017 | 12 |

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

### AICD-T2 (stress test — model-family attribution)

| Method | Encoder | 1% | 5% | 20% |
|:--|:--|-:|-:|-:|
| **Hier-NTK (ours)** | ModernBERT | - | - | - |
| **Hier-NTK (ours)** | unixcoder | - | - | - |
| HierTree only | ModernBERT | - | - | - |
| NTK only | ModernBERT | - | - | - |
| CE baseline (ours) | ModernBERT | - | - | - |

---

## Paper Reference

UniXcoder full-data baseline (CoDET-M4 Author IID): **0.6633 Macro-F1**

---

## Notes

- **Droid T3/T4 SKIPPED per 2026-05-10 directive.** Focus on CoDET-M4 + AICD T2 only.
- Each file runs 2 encoders × 2 benchmarks × 3 fractions = 12 experiments (vs 24 before).
- Results format: `{"tag", "enc", "bench", "frac", "macro", "weighted", "acc", "dpaper", "wall"}`
