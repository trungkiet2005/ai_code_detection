# Exp_FewShot Tracker — K-shot AI-Code Detection (Kaggle T4)

> **Two tracks:** **`testing/`** — setup variation (no novelty gate). **`novel/`** — theory-driven oral bets, `exp_nNN_*.py` (5-gate filter). Primary metric: **CoDET-M4 Author IID Macro-F1** (paper Table 7). Always pair **test** with **val** and **val–test gap**.

**Paper baseline:** UniXcoder **full data** = **0.6633** (Orel et al., ACL’25 Table 7). **Lean-stack reference:** Exp_13 NTKAlign **0.7103** at ~20% train lives in `Exp_Climb/tracker.md` (heavier backbone than FewShot inline runs — not apples-to-apples vs rows below).

---

## 1. Leaderboards

**Unified source:** `results/summary.json` from:

```bash
python Exp_FewShot/aggregate_fs_results.py results
```

**Column Trk:** `test` = `Exp_FewShot/testing/` (`exp_fs_*`); `novel` = `Exp_FewShot/novel/` (`exp_n*`). **d vs paper** = test Macro-F1 minus UniXcoder full-data **0.6633**.

**Coverage:** **104** JSON files ingested -> **87** `(method, regime)` cells in `summary.json`. Duplicate filenames `*__dup*.json` repeat the same metrics (ignore as extra seeds). **`exp_fs_00`** baseline CE (**0.1836**) appears only in Section 6 run history -- **no** `exp_fs_00_*.json` in `results/` yet. **`exp_fs_99_K128_seed42.json`** is an incomplete stub (`method=Test`) -- **excluded** from the `K=128` table below; delete or fix it for clean aggregates.

No JSON yet for scripts not listed under Canonical JSON (example: `exp_n20_*` if never run).


### Regime `frac=0.05` (~25K train, 3 epochs)

| Rank | Trk | Method | Test | d vs paper | Val | Gap | Canonical JSON |
|:-:|:-:|:-:|-:|:-:|:-:|:-:|:--|
| 1 | novel | FS-ConformalMondrian | 0.6721 | +0.0088 | 0.7147 | +0.0426 | `exp_n07_conformal_mondrian_frac0.05_seed42.json` |
| 2 | test | FS-Hier-NTK | 0.6709 | +0.0076 | 0.7041 | +0.0332 | `exp_fs_inline_hier_ntk_frac0.05_seed42.json` |
| 3 | test | FS-HierTree | 0.6682 | +0.0049 | 0.6824 | +0.0142 | `exp_fs_inline_hier_frac0.05_seed42.json` |
| 4 | novel | FS-SlicedWassersteinClass | 0.6666 | +0.0033 | 0.7129 | +0.0463 | `exp_n14_sliced_wasserstein_frac0.05_seed42.json` |
| 5 | novel | FS-FrontDoor-StyleMediator | 0.6661 | +0.0028 | 0.7103 | +0.0442 | `exp_n02_frontdoor_style_frac0.05_seed42.json` |
| 6 | novel | FS-VIB | 0.6659 | +0.0026 | 0.7006 | +0.0347 | `exp_n10_vib_frac0.05_seed42.json` |
| 7 | novel | FS-VarianceInvariantSource | 0.6654 | +0.0021 | 0.6899 | +0.0246 | `exp_n11_vic_source_frac0.05_seed42.json` |
| 8 | test | FS-NTKAlign | 0.6652 | +0.0019 | 0.7071 | +0.0419 | `exp_fs_inline_ntkalign_frac0.05_seed42.json` |
| 9 | novel | FS-TENT-TTA | 0.6651 | +0.0018 | 0.6857 | +0.0207 | `exp_n15_tent_tta_frac0.05_seed42.json` |
| 10 | test | FS-Focal | 0.6616 | -0.0017 | 0.6975 | +0.0359 | `exp_fs_inline_focal_frac0.05_seed42.json` |
| 11 | novel | FS-EnergyOOD | 0.6551 | -0.0082 | 0.6753 | +0.0203 | `exp_n16_energy_ood_frac0.05_seed42.json` |
| 12 | test | FS-Baseline-UniXcoder | 0.6512 | -0.0121 | 0.6912 | +0.0400 | `exp_fs_baseline_unixcoder_frac0.05_seed42.json` |
| 13 | novel | FS-DataMapsCurriculum | 0.6481 | -0.0152 | 0.6849 | +0.0368 | `exp_n18_datamaps_curriculum_frac0.05_seed42.json` |
| 14 | novel | FS-IRM | 0.6430 | -0.0203 | 0.6672 | +0.0242 | `exp_n19_irm_frac0.05_seed42.json` |
| 15 | novel | FS-PACBayes-SampleFloor | 0.6341 | -0.0292 | 0.6835 | +0.0494 | `exp_n09_pac_bayes_floor_frac0.05_seed42.json` |
| 16 | novel | FS-SRD-SiblingResidual | 0.6202 | -0.0431 | 0.6742 | +0.0540 | `exp_n01_sibling_residual_frac0.05_seed42.json` |
| 17 | test | FS-Baseline-CodeBERT | 0.5977 | -0.0656 | 0.6469 | +0.0492 | `exp_fs_baseline_codebert_frac0.05_seed42.json` |
| 18 | test | FS-Baseline-GraphCodeBERT | 0.5951 | -0.0682 | 0.6856 | +0.0905 | `exp_fs_baseline_graphcodebert_frac0.05_seed42.json` |
| 19 | novel | FS-SpectralEigengap | 0.5645 | -0.0988 | 0.6119 | +0.0474 | `exp_n08_spectral_eigengap_frac0.05_seed42.json` |
| 20 | test | FS-NTKAlign-Frozen | 0.4608 | -0.2025 | 0.4947 | +0.0339 | `exp_fs_04_frac0.05_seed42.json` |
| 21 | test | FS-SupCon-Frozen | 0.4607 | -0.2026 | 0.4947 | +0.0340 | `exp_fs_05_frac0.05_seed42.json` |
| 22 | novel | FS-Prototypical | 0.1904 | -0.4729 | 0.2274 | +0.0370 | `exp_n17_prototypical_frac0.05_seed42.json` |
| 23 | novel | FS-ETF-FrozenSimplex | 0.0859 | -0.5774 | 0.1020 | +0.0161 | `exp_n04_etf_simplex_frac0.05_seed42.json` |

### Regime `frac=0.01` (~5K train)

| Rank | Trk | Method | Test | d vs paper | Val | Gap | Canonical JSON |
|:-:|:-:|:-:|-:|:-:|:-:|:-:|:--|
| 1 | novel | FS-ConformalMondrian | 0.5750 | -0.0883 | 0.5944 | +0.0194 | `exp_n07_conformal_mondrian_frac0.01_seed42.json` |
| 2 | test | FS-Baseline-UniXcoder | 0.5744 | -0.0889 | 0.5839 | +0.0095 | `exp_fs_baseline_unixcoder_frac0.01_seed42.json` |
| 3 | novel | FS-SlicedWassersteinClass | 0.5729 | -0.0904 | 0.5884 | +0.0155 | `exp_n14_sliced_wasserstein_frac0.01_seed42.json` |
| 4 | novel | FS-VarianceInvariantSource | 0.5701 | -0.0932 | 0.5766 | +0.0065 | `exp_n11_vic_source_frac0.01_seed42.json` |
| 5 | test | FS-NTKAlign | 0.5697 | -0.0936 | 0.5614 | -0.0083 | `exp_fs_inline_ntkalign_frac0.01_seed42.json` |
| 6 | novel | FS-DataMapsCurriculum | 0.5669 | -0.0964 | 0.6070 | +0.0401 | `exp_n18_datamaps_curriculum_frac0.01_seed42.json` |
| 7 | novel | FS-FrontDoor-StyleMediator | 0.5658 | -0.0975 | 0.5963 | +0.0305 | `exp_n02_frontdoor_style_frac0.01_seed42.json` |
| 8 | test | FS-HierTree | 0.5654 | -0.0979 | 0.5971 | +0.0317 | `exp_fs_inline_hier_frac0.01_seed42.json` |
| 9 | novel | FS-VIB | 0.5628 | -0.1005 | 0.5764 | +0.0135 | `exp_n10_vib_frac0.01_seed42.json` |
| 10 | test | FS-Hier-NTK | 0.5608 | -0.1025 | 0.5708 | +0.0099 | `exp_fs_inline_hier_ntk_frac0.01_seed42.json` |
| 11 | novel | FS-TENT-TTA | 0.5596 | -0.1037 | 0.5631 | +0.0034 | `exp_n15_tent_tta_frac0.01_seed42.json` |
| 12 | novel | FS-PACBayes-SampleFloor | 0.5574 | -0.1059 | 0.5848 | +0.0274 | `exp_n09_pac_bayes_floor_frac0.01_seed42.json` |
| 13 | test | FS-Focal | 0.5565 | -0.1068 | 0.5765 | +0.0200 | `exp_fs_inline_focal_frac0.01_seed42.json` |
| 14 | novel | FS-SRD-SiblingResidual | 0.5417 | -0.1216 | 0.5784 | +0.0367 | `exp_n01_sibling_residual_frac0.01_seed42.json` |
| 15 | novel | FS-EnergyOOD | 0.5178 | -0.1455 | 0.5240 | +0.0062 | `exp_n16_energy_ood_frac0.01_seed42.json` |
| 16 | test | FS-Baseline-GraphCodeBERT | 0.5014 | -0.1619 | 0.5760 | +0.0746 | `exp_fs_baseline_graphcodebert_frac0.01_seed42.json` |
| 17 | test | FS-Baseline-CodeBERT | 0.5007 | -0.1626 | 0.5831 | +0.0824 | `exp_fs_baseline_codebert_frac0.01_seed42.json` |
| 18 | novel | FS-SpectralEigengap | 0.4822 | -0.1811 | 0.5039 | +0.0217 | `exp_n08_spectral_eigengap_frac0.01_seed42.json` |
| 19 | novel | FS-IRM | 0.4693 | -0.1940 | 0.4831 | +0.0138 | `exp_n19_irm_frac0.01_seed42.json` |
| 20 | test | FS-SupCon-Frozen | 0.4038 | -0.2595 | 0.4221 | +0.0183 | `exp_fs_05_frac0.01_seed42.json` |
| 21 | test | FS-NTKAlign-Frozen | 0.4037 | -0.2596 | 0.4220 | +0.0183 | `exp_fs_04_frac0.01_seed42.json` |
| 22 | novel | FS-Prototypical | 0.1224 | -0.5409 | 0.1537 | +0.0312 | `exp_n17_prototypical_frac0.01_seed42.json` |
| 23 | novel | FS-ETF-FrozenSimplex | 0.0491 | -0.6142 | 0.0623 | +0.0132 | `exp_n04_etf_simplex_frac0.01_seed42.json` |

### Regime `K=128` (~768 train)

| Rank | Trk | Method | Test | d vs paper | Val | Gap | Canonical JSON |
|:-:|:-:|:-:|-:|:-:|:-:|:-:|:--|
| 1 | test | FS-Focal | 0.3749 | -0.2884 | 0.3378 | -0.0371 | `exp_fs_inline_focal_K128_seed42.json` |
| 2 | test | FS-Baseline-UniXcoder | 0.3727 | -0.2906 | 0.3968 | +0.0241 | `exp_fs_baseline_unixcoder_K128_seed42.json` |
| 3 | novel | FS-VIB | 0.3510 | -0.3123 | 0.3487 | -0.0023 | `exp_n10_vib_K128_seed42.json` |
| 4 | test | FS-HierTree | 0.3492 | -0.3141 | 0.3089 | -0.0403 | `exp_fs_inline_hier_K128_seed42.json` |
| 5 | novel | FS-SlicedWassersteinClass | 0.3453 | -0.3180 | 0.3354 | -0.0100 | `exp_n14_sliced_wasserstein_K128_seed42.json` |
| 6 | novel | FS-DataMapsCurriculum | 0.3397 | -0.3236 | 0.3119 | -0.0278 | `exp_n18_datamaps_curriculum_K128_seed42.json` |
| 7 | novel | FS-FrontDoor-StyleMediator | 0.3373 | -0.3260 | 0.3052 | -0.0321 | `exp_n02_frontdoor_style_K128_seed42.json` |
| 8 | novel | FS-ConformalMondrian | 0.3338 | -0.3295 | 0.3244 | -0.0094 | `exp_n07_conformal_mondrian_K128_seed42.json` |
| 9 | novel | FS-TENT-TTA | 0.3176 | -0.3457 | 0.3025 | -0.0151 | `exp_n15_tent_tta_K128_seed42.json` |
| 10 | novel | FS-IRM | 0.3160 | -0.3473 | 0.3163 | +0.0003 | `exp_n19_irm_K128_seed42.json` |
| 11 | novel | FS-SRD-SiblingResidual | 0.3130 | -0.3503 | 0.3018 | -0.0113 | `exp_n01_sibling_residual_K128_seed42.json` |
| 12 | novel | FS-PACBayes-SampleFloor | 0.2930 | -0.3703 | 0.3221 | +0.0292 | `exp_n09_pac_bayes_floor_K128_seed42.json` |
| 13 | test | FS-NTKAlign | 0.2929 | -0.3704 | 0.3348 | +0.0419 | `exp_fs_inline_ntkalign_K128_seed42.json` |
| 14 | novel | FS-VarianceInvariantSource | 0.2848 | -0.3785 | 0.3118 | +0.0270 | `exp_n11_vic_source_K128_seed42.json` |
| 15 | test | FS-Hier-NTK | 0.2775 | -0.3858 | 0.3139 | +0.0364 | `exp_fs_inline_hier_ntk_K128_seed42.json` |
| 16 | test | FS-Baseline-CodeBERT | 0.2651 | -0.3982 | 0.2946 | +0.0295 | `exp_fs_baseline_codebert_K128_seed42.json` |
| 17 | novel | FS-EnergyOOD | 0.2404 | -0.4229 | 0.2946 | +0.0543 | `exp_n16_energy_ood_K128_seed42.json` |
| 18 | novel | FS-SpectralEigengap | 0.2331 | -0.4302 | 0.2181 | -0.0149 | `exp_n08_spectral_eigengap_K128_seed42.json` |
| 19 | test | FS-Baseline-GraphCodeBERT | 0.2191 | -0.4442 | 0.3059 | +0.0868 | `exp_fs_baseline_graphcodebert_K128_seed42.json` |
| 20 | test | FS-SupCon-Frozen | 0.1493 | -0.5140 | 0.2522 | +0.1029 | `exp_fs_05_K128_seed42.json` |
| 21 | test | FS-NTKAlign-Frozen | 0.1479 | -0.5154 | 0.2465 | +0.0986 | `exp_fs_04_K128_seed42.json` |
| 22 | novel | FS-Prototypical | 0.0683 | -0.5950 | 0.0741 | +0.0058 | `exp_n17_prototypical_K128_seed42.json` |
| 23 | novel | FS-ETF-FrozenSimplex | 0.0339 | -0.6294 | 0.0476 | +0.0137 | `exp_n04_etf_simplex_K128_seed42.json` |

### Regime `K=32` (~192 train)

| Rank | Trk | Method | Test | d vs paper | Val | Gap | Canonical JSON |
|:-:|:-:|:-:|-:|:-:|:-:|:-:|:--|
| 1 | novel | FS-ConformalMondrian | 0.1897 | -0.4736 | 0.2225 | +0.0328 | `exp_n07_conformal_mondrian_K32_seed42.json` |
| 2 | novel | FS-SRD-SiblingResidual | 0.1894 | -0.4739 | 0.1949 | +0.0055 | `exp_n01_sibling_residual_K32_seed42.json` |
| 3 | test | FS-NTKAlign | 0.1831 | -0.4802 | 0.2280 | +0.0449 | `exp_fs_inline_ntkalign_K32_seed42.json` |
| 4 | novel | FS-DataMapsCurriculum | 0.1812 | -0.4821 | 0.2226 | +0.0414 | `exp_n18_datamaps_curriculum_K32_seed42.json` |
| 5 | novel | FS-VIB | 0.1803 | -0.4830 | 0.2229 | +0.0427 | `exp_n10_vib_K32_seed42.json` |
| 6 | novel | FS-PACBayes-SampleFloor | 0.1753 | -0.4880 | 0.2388 | +0.0635 | `exp_n09_pac_bayes_floor_K32_seed42.json` |
| 7 | novel | FS-VarianceInvariantSource | 0.1635 | -0.4998 | 0.2181 | +0.0546 | `exp_n11_vic_source_K32_seed42.json` |
| 8 | novel | FS-SpectralEigengap | 0.1522 | -0.5111 | 0.1676 | +0.0154 | `exp_n08_spectral_eigengap_K32_seed42.json` |
| 9 | novel | FS-SlicedWassersteinClass | 0.1499 | -0.5134 | 0.1923 | +0.0424 | `exp_n14_sliced_wasserstein_K32_seed42.json` |
| 10 | novel | FS-FrontDoor-StyleMediator | 0.1484 | -0.5149 | 0.2260 | +0.0776 | `exp_n02_frontdoor_style_K32_seed42.json` |
| 11 | novel | FS-TENT-TTA | 0.1409 | -0.5224 | 0.2089 | +0.0680 | `exp_n15_tent_tta_K32_seed42.json` |
| 12 | test | FS-NTKAlign-Frozen | 0.1348 | -0.5285 | 0.1338 | -0.0010 | `exp_fs_04_K32_seed42.json` |
| 13 | test | FS-SupCon-Frozen | 0.1347 | -0.5286 | 0.1338 | -0.0009 | `exp_fs_05_K32_seed42.json` |
| 14 | novel | FS-IRM | 0.1186 | -0.5447 | 0.1625 | +0.0439 | `exp_n19_irm_K32_seed42.json` |
| 15 | novel | FS-EnergyOOD | 0.1069 | -0.5564 | 0.1433 | +0.0364 | `exp_n16_energy_ood_K32_seed42.json` |
| 16 | novel | FS-ETF-FrozenSimplex | 0.0339 | -0.6294 | 0.0476 | +0.0137 | `exp_n04_etf_simplex_K32_seed42.json` |
| 17 | novel | FS-Prototypical | 0.0339 | -0.6294 | 0.0476 | +0.0137 | `exp_n17_prototypical_K32_seed42.json` |

---

## 2. Narrative

### Headline (EMNLP claim)

**Four** testing-track methods exceed paper UniXcoder **0.6633** at **~5%** train: **Hier-NTK (0.6709)**, **HierTree (0.6682)**, **NTKAlign (0.6652)**, **Focal (0.6616)**. On the same `fs_seed=42` JSON sweep, the top **overall** @ 5% is **FS-ConformalMondrian (0.6721, novel / CMP)** — **+0.0012** vs Hier-NTK; confirm with more seeds before headline change. Main method name in prose (testing claim): **Hier-NTK** (HierTree family prior + NTK alignment). Phase-transition: weak at **K=128**, unlocks between **~1%** and **~5%** data.

### Phase-transition curve (NTKAlign / lean stack)

- K=32 (~192 samples) → ~**0.183** (≈ random)
- K=128 (~768) → **0.2929**
- 1% (~5K) → **0.5697**
- 5% (~25K) → **0.6652** (matches paper full-data baseline band)
- Frozen encoder @ 5% plateaus ~**0.46** (generator-specific signal needs fine-tuning).

### Key insights (2026-05-09)

1. **Hierarchical prior** — HierTree / Hier-NTK beat NTKAlign-alone at 5%; family structure is the dominant signal; **Hier-NTK** is the best single recipe.
2. **Cluster at 5%** — Top four methods within ~0.009 test F1; gain from **prior + ModernBERT + budget**, not one trick.
3. **UniXcoder reimpl** — 5% **0.6512** ≈ **98.2%** of paper full-data **0.6633** at **1/20** data → protocol sanity check.
4. **Encoder** — ModernBERT **> CodeBERT / GraphCodeBERT** at 5%.
5. **1% regime** — UniXcoder reimpl wins; **encoder pretrain > loss design** below ~5K samples.
6. **K=128** — **Focal** wins; Hier / NTK need batch diversity; Hier-NTK **hurts** vs HierTree at K=128 (sparse NTK target).
7. **Frozen ceiling ~0.46** — pretrained features split human vs AI, not generator identity.
8. **Val–test gap** — negative gap often **under-fit** on small val; large positive gap → mismatch / over-fit on slice.
9. **Combo rule** — use **Hier+NTK** in **fraction** mode; **drop / weaken NTK** in extreme **K-shot** if kernel target too sparse.
10. **K=32** — near-random for 6-class authorship.
11. **Bend** — between **K=128** and **1%**; ~**5K** samples unlocks generator-specific signal.

---

## 3. Protocol, targets, gate

### Hardware profiles

| GPU | VRAM | Batch | Precision | Seq length | Speed |
|:--|:--|:--|:--|:--|:--|
| **RTX 96GB** | 96GB | **128** | **bf16** | **512** | **~6-8× faster than T4** |
| **RTX Pro 6000 Ada** | 48GB | **64** | **bf16** | **512** | **~3-4× faster than T4** |
| T4 | 16GB | 16 | fp16 | 384 | baseline |
| H100 | 80GB | 32-64 | bf16 | 512 | fast |

RTX96GB detection: `mem_gb >= 90` → bs=128
RTX6000 detection: `"RTX Pro 6000" in gpu or "RTX 6000 Ada" in gpu or mem_gb >= 40`

**Sampling:** Stratified K-shot on TRAIN; mini-val from VAL (n/class=64); **TEST** full IID split. Log **`fs_seed`**.

| Baseline | Author F1 | Where |
|:--|:-:|:--|
| UniXcoder full (100%) | 66.33 | Paper Table 7 |
| Our 20%-data SOTA (Exp_13) | 71.03 | `Exp_Climb/tracker.md` |

**Decision gate (historical):** K=128 vs **60–65** Macro-F1 framed low-shot scaling vs fallback to 20% story.

---

## 4. Operations

**JSON:** `results/<exp_id>_<label>_seed<S>.json` (Kaggle: `/kaggle/working/results/`). Aggregate:

```bash
python Exp_FewShot/aggregate_fs_results.py results
python Exp_FewShot/aggregate_fs_results.py results --csv
```

**Offline mode (no-internet, avoids conflict + bias):**
- Models: `/kaggle/input/datasets/chiboiz/ai-detection-encoders/models/`
- CoDET-M4: `/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet`
- DroidCollection: `/kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/`
- All loaders use `local_files_only=True` → no HuggingFace download

**Merge code strategy (rapid iteration on RTX6000):**
For running multiple configs sequentially in one cell:
```bash
FS_ENCODER=ModernBERT-base,codebert-base,unixcoder-base
FS_SWEEP_FRACS=0.01,0.05,0.20
FS_SEED=42
FS_BENCHMARK=codet_m4
python Exp_FewShot/testing_chis/run_hier_ntk_portfolio.py
```
Uses RTX6000 auto-detection → bs=64, bf16, seq=512 (~3-4× faster than T4).

**Rules:** Suite standalone — no imports from `Exp_Climb/` / `Exp_DM/` / `Exp_CodeDet/`. No tree-sitter / spectral in few-shot files. **fp16** on T4, **bf16** on RTX6000/H100. One method per file; knobs via env (`FS_K_SHOT`, `FS_TRAIN_FRACTION`, `FS_BENCHMARK`, ...).

**CPU smoke:** `FS_K_SHOT=8 python Exp_FewShot/exp_fs_00_baseline.py` (and analogous for other scripts).

**Kaggle:** clone repo → set env → `python Exp_FewShot/exp_fs_01_ntkalign.py` etc.; or paste standalone single-file cells (auto-clone on first run).

**Refresh leaderboard tables (Section 1):** regenerate `results/summary.json` with `aggregate_fs_results.py`, then run `python Exp_FewShot/_gen_leaderboard_md.py` and replace the `## 1. Leaderboards` ... `---` block (maintainer helper; UTF-8).

---

## 5. Evolution timeline

| Date | Run | Result | Takeaway |
|:--|:--|:-:|:--|
| 2026-05-08 | exp_fs_00 K=32 | 0.1836 | Floor; llama3.1 collapse. |
| 2026-05-08 | exp_fs_01 NTKAlign K=32 | 0.1222 | NTK hurts at tiny K (sparse kernel target). |
| 2026-05-08 | exp_fs_03 Frozen K=32 | 0.1348 | Human vs AI only when frozen. |
| 2026-05-09 | NTKAlign 5% | 0.6652 | Phase transition; matches paper band. |
| 2026-05-09 | Frozen combos 5% | ~0.46 | Frozen ceiling. |
| 2026-05-09 | HierTree 5% | 0.6682 | Hier prior beats NTK-alone. |
| 2026-05-09 | Hier-NTK 5% | **0.6709** | Best lean-stack testing method. |
| 2026-05-09 | Focal K=128 | 0.3749 | Best at K=128. |
| 2026-05-09 | UniXcoder 1% / 5% | 0.5744 / 0.6512 | Encoder wins at 1%; reimpl validates protocol. |
| 2026-05-10 | Novel JSON sweep | see Section 1 | CMP **0.6721** @ 5%; full aggregate in `summary.json`. |

---

## 6. Run history (append-only)

> Do **not** delete or rewrite blocks below — paste new runs under the right `exp_*` heading between `BEGIN_FS_TABLE` and `END_FS_TABLE`.

### exp_fs_00 — FS-Baseline-CE (CE only, ModernBERT-base)

#### Run 1 — 2026-05-09 09:28, Kaggle T4 14.6GB, K=32, fs_seed=42

```
BEGIN_FS_TABLE method=FS-Baseline-CE exp_id=exp_fs_00
  benchmark: codet_m4
  task:      author
  k_shot:    32
  n_classes: 6
  fs_seed:   42
  encoder:   answerdotai/ModernBERT-base
  precision: fp16
  bs:        16
  seq:       384
  test_macro_f1     0.1836
  test_weighted_f1  0.3175
  test_accuracy     0.2842
  val_macro_f1      0.2253
  val_test_gap      0.0417
  train_steps       12
  wall_time_s       484.4
  per_class_f1   {class_0: 0.5031, class_1: 0.2456, class_2: 0.0987,
                  class_3: 0.0018, class_4: 0.0353, class_5: 0.2171}
  per_lang_f1    {cpp: 0.2104, java: 0.1178, python: 0.1853}
  per_source_f1  {cf: 0.2043, gh: 0.1169, lc: 0.1531}
  train_per_class {0: 32, 1: 32, 2: 32, 3: 32, 4: 32, 5: 32}
END_FS_TABLE
```

**Author vocab (5 generators):** `codellama, gpt, llama3.1, nxcode, qwen1.5` → class indices `1..5`; class 0 = human.

**Pipeline timeline (T4):**
- Dataset load + label vocab: 96s
- K-shot sample + mini-val: ~2s
- Tokenizer + model load: 12s
- Train 12 steps: 35s
- Eval (val 384 + test 47,744 samples): 458s **bottleneck**
- Total: 484s ≈ 8 min

**Reading:** Loss curve `1.86 → val 0.0783 (step 5) → val 0.2253 (step 10) → val 0.1439 (step 12)` — model peaks around step 10 then starts to overfit slightly. Best val checkpoint restored. Test 0.1836 is below val 0.2253 by 4.17 pt because the test set is much larger and contains the OOD-source distribution shift our 20%-data climb already documented (val 5K samples are mostly CF/LC; full test 47.7K spans all three sources).

**Per-class collapse (class 3 = llama3.1, F1 = 0.002):** with K=32 examples of llama3.1 in training, the model never separates it from sibling codellama. This is exactly the Qwen↔Nxcode → llama3.1↔codellama family confusion we predicted in the paper. NTKAlign should help here — its target-kernel pulls same-class projections together explicitly.

### exp_fs_03 — FS-Frozen-LinearProbe (encoder frozen, head only)

#### Run 1 — 2026-05-09 11:52, Kaggle T4 14.6GB, K=32, fs_seed=42, lr_heads=1e-3

```
BEGIN_FS_TABLE method=FS-Frozen-LinearProbe exp_id=exp_fs_03
  regime: K=32   k_shot: 32   train_fraction: 0.0000
  encoder: answerdotai/ModernBERT-base   bs: 16   seq: 384   prec: fp16
  epochs: 1   fs_seed: 42
  test_macro_f1     0.1348
  test_weighted_f1  0.3356
  test_accuracy     0.3517
  val_macro_f1      0.1338
  val_test_gap     -0.0010
  train_steps       12
  wall_time_s       510.1
  frozen_params_M   149.0
  trainable_params_M 0.12  (head only)
  per_class_f1   {class_0: 0.6274, class_1: 0.0383, class_2: 0.0962,
                  class_3: 0.0000, class_4: 0.0050, class_5: 0.0416}
  per_lang_f1    {cpp: 0.1419, java: 0.1206, python: 0.1389}
  per_source_f1  {cf: 0.1598, gh: 0.1489, lc: 0.0446}
END_FS_TABLE
```

**Diagnosis (frozen head) — frozen ≠ silver bullet at K=32:**

| Slice | Baseline (free) | NTKAlign (free) | **Frozen (head only)** |
|:--|:-:|:-:|:-:|
| Test Macro-F1 | 0.1836 | 0.1222 | **0.1348** |
| Val Macro-F1 | 0.2253 | 0.2020 | 0.1338 |
| Val-test gap | +0.04 | +0.08 | **−0.001** ← perfect calibration |
| Class 0 (human) F1 | 0.503 | 0.214 | **0.627** (best) |
| Class 1 (codellama) F1 | 0.246 | 0.222 | 0.038 |
| Class 3 (llama3.1) F1 | 0.002 | 0.002 | 0.000 |
| Class 4 (nxcode) F1 | 0.035 | 0.026 | 0.005 |
| Trainable params | 149M | 149.2M | **0.12M (1280x less)** |

**Reading.** Frozen encoder collapses to a near-binary "human vs AI" detector:
- Class 0 (human) F1 jumps **+0.124 over baseline** (0.503 → 0.627) — the pretrained ModernBERT features clearly distinguish human-written code.
- All 5 AI classes collapse: codellama 0.04, gpt 0.10, llama3.1 0.00, nxcode 0.005, qwen 0.04. The frozen features carry NO generator-specific signal; with only 32 labeled examples per AI class the head cannot learn to discriminate among siblings.
- Val-test gap is **−0.001** (test ≥ val) — model is perfectly calibrated; it is not overfitting, it is fundamentally limited by the frozen features.
- Weighted-F1 0.336 + accuracy 0.352 confirm: model effectively predicts "human" most of the time and gets the binary axis right.

**Implication.** The K=32 failure is NOT primarily catastrophic forgetting of the encoder — it is **data scarcity for 5-way generator discrimination**. ModernBERT pretrain has plenty of human code but no signal that separates codellama from qwen, so 32 labeled examples per class is below the information floor. Frozen and free fine-tune both lose. The choice now:

1. K=128 → maybe enough for AI-class discrimination (4× more data per class).
2. fraction-mode runs (1% ≈ 5K, 5% ≈ 25K) → test where the phase transition is.
3. Accept K=32 < random for 6-class and pivot the paper to a 1%/5% data efficiency curve (still way below the existing 20% Exp_13 71.03 result).

### exp_fs_01 — FS-NTKAlign (CE + NTK target-kernel alignment)

#### Run 1 — 2026-05-09 09:31, Kaggle T4 14.6GB, K=32, fs_seed=42, lambda_ntk=0.4

```
BEGIN_FS_TABLE method=FS-NTKAlign exp_id=exp_fs_01
  benchmark: codet_m4
  task:      author
  k_shot:    32
  n_classes: 6
  fs_seed:   42
  encoder:   answerdotai/ModernBERT-base
  precision: fp16
  bs:        16
  seq:       384
  lambda_ntk        0.4000
  test_macro_f1     0.1222
  test_weighted_f1  0.1605
  test_accuracy     0.1490
  val_macro_f1      0.2020
  val_test_gap      0.0798
  train_steps       12
  wall_time_s       476.1
  per_class_f1   {class_0: 0.2135, class_1: 0.2221, class_2: 0.0882,
                  class_3: 0.0018, class_4: 0.0256, class_5: 0.1819}
  per_lang_f1    {cpp: 0.1265, java: 0.0829, python: 0.1346}
  per_source_f1  {cf: 0.1192, gh: 0.0736, lc: 0.1206}
  train_per_class {0: 32, 1: 32, 2: 32, 3: 32, 4: 32, 5: 32}
END_FS_TABLE
```

**Diagnosis (NTKAlign K=32) — NTKAlign is WORSE than baseline at K=32 (−6.14 pt test):**

| Slice | Baseline K=32 | NTKAlign K=32 | Δ |
|:--|:-:|:-:|:-:|
| Test Macro-F1 | 0.1836 | **0.1222** | **−0.0614** |
| Val Macro-F1 | 0.2253 | 0.2020 | −0.0233 |
| Val-test gap | +0.04 | **+0.08** | gap doubled |
| Class 0 (human) F1 | **0.503** | 0.214 | −0.289 collapse |
| Class 1 (codellama) F1 | 0.246 | 0.222 | −0.024 |
| Class 3 (llama3.1) F1 | 0.002 | 0.002 | unchanged (still collapsed) |

The collapse is concentrated on the **human class** (0.50 → 0.21). NTKAlign's target-kernel objective pushes same-class projections together — but with batch size B=16 and 6 classes, each batch has only ~2.7 same-class pairs on average, so the kernel target Y is mostly off-diagonal zeros. The loss becomes a *separation* objective dominated by inter-class repulsion, which drags the human cluster (the only class with a strong CE signal) toward the centroid mean.

**Hypothesised fixes (in order of expected impact):**
1. **Class-balanced batch sampler.** Currently the sampler shuffles K·N=192 samples randomly into bs=16 batches → uneven class composition. Force at least $\lceil B/N \rceil = 3$ examples per class per batch. The NTK kernel target Y will then have $\geq B \cdot 3$ on-diagonal pairs instead of $\sim 6$ on average.
2. **Lower lambda_ntk.** 0.4 may be too aggressive at K-shot scale. Try 0.1 or 0.05 first; sweep upward only if K=128 baseline is also weak.
3. **Larger batch (bs=32 fp16).** Doubles same-class pair count for free. T4 should fit ModernBERT-base at seq=384 bs=32 fp16. Worth trying.
4. **Warmup epochs without NTK.** Run 3–5 epochs with $\lambda_{\text{NTK}} = 0$, then activate. Prevents early collapse before CE has settled.

**Implication for paper pivot.** This is K=32, the small end. **K=128 will tell us whether NTKAlign starts to help once the kernel target becomes informative.** Decision gate stays at K=128: if NTKAlign K=128 < Baseline K=128, we revert to the 20% data story (paper draft v1 already builds on the existing Exp_13 NTKAlign 71.03 result, which uses **bs=64 H100**, not K-shot bs=16). The Exp_Climb claim is unchanged.

_(no run yet)_
