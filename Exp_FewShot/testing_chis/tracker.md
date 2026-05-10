# Testing Chis Tracker - Hier-NTK Portfolio

Purpose: keep the run ledger for `run_hier_ntk_portfolio.py` separate from the global `Exp_FewShot/tracker.md`.

Primary metric: **CoDET-M4 Author IID Macro-F1**. Paper reference: UniXcoder full-data Table 7 = **0.6633** Macro-F1.

Current result source: `results/hier_ntk_p*.json` plus `results/hier_ntk_portfolio_seed42.json`. Older suite results are archived under `results/Legacy/` and are not mixed into this tracker.

Published-baseline runners: `Exp_FewShot/testing_chis/exp00_sota_codet5_authorship.py`, `exp01_sota_detective_supcon.py`, `exp02_sota_faid_multitask.py`, and `exp03_sota_style_repr_logreg.py`. Use them for paper-comparison baselines only; keep `run_hier_ntk_portfolio.py` for our Hier-NTK method runs.

---

## 1. Current Leaderboard

Run batch: **2026-05-10 13:39**, `FS-Hier-NTK`, seed 42, CoDET-M4 author attribution, 3 encoders x 3 train fractions.

Runtime profile from JSON: **bf16**, **seq=512**, **batch=128**, **epochs=3**, `lambda_hier=0.4`, `lambda_ntk=0.4`, `lr_encoder=2e-5`, `lr_heads=1e-4`.

| Rank | Exp | Encoder | Fraction | Test Macro-F1 | d vs paper | Val Macro-F1 | Gap | Weighted-F1 | Acc | Steps | Wall |
| :-: | :-- | :-- | -: | -: | -: | -: | -: | -: | -: | -: | --: |
| 1 | `hier_ntk_p3` | ModernBERT-base | 0.20 | **0.6895** | **+0.0262** | 0.7257 | +0.0362 | 0.8068 | 0.8101 | 1899 | 1009.2s |
| 2 | `hier_ntk_p9` | unixcoder-base | 0.20 | 0.6789 | +0.0156 | 0.7233 | +0.0445 | 0.8001 | 0.8036 | 1899 | 600.3s |
| 3 | `hier_ntk_p6` | codebert-base | 0.20 | 0.6373 | -0.0260 | 0.7096 | +0.0723 | 0.7714 | 0.7732 | 1899 | 599.6s |
| 4 | `hier_ntk_p2` | ModernBERT-base | 0.05 | 0.6307 | -0.0326 | 0.6722 | +0.0416 | 0.7674 | 0.7707 | 477 | 306.4s |
| 5 | `hier_ntk_p8` | unixcoder-base | 0.05 | 0.6213 | -0.0420 | 0.6723 | +0.0510 | 0.7616 | 0.7637 | 477 | 177.1s |
| 6 | `hier_ntk_p5` | codebert-base | 0.05 | 0.5109 | -0.1524 | 0.6107 | +0.0998 | 0.6806 | 0.6843 | 477 | 176.1s |
| 7 | `hier_ntk_p7` | unixcoder-base | 0.01 | 0.4600 | -0.2033 | 0.5042 | +0.0443 | 0.6297 | 0.6091 | 96 | 64.4s |
| 8 | `hier_ntk_p1` | ModernBERT-base | 0.01 | 0.4182 | -0.2451 | 0.4100 | -0.0082 | 0.6111 | 0.6017 | 96 | 126.3s |
| 9 | `hier_ntk_p4` | codebert-base | 0.01 | 0.3806 | -0.2827 | 0.4686 | +0.0880 | 0.5549 | 0.5317 | 96 | 63.6s |

---

## 2. Encoder x Fraction Matrix

| Encoder | 1% | 5% | 20% | Best | Reading |
| :-- | -: | -: | -: | -: | :-- |
| ModernBERT-base | 0.4182 | 0.6307 | **0.6895** | **0.6895 @ 20%** | Best overall; scales strongly with data. |
| unixcoder-base | **0.4600** | 0.6213 | 0.6789 | 0.6789 @ 20% | Strongest at 1%, second at 20%. |
| codebert-base | 0.3806 | 0.5109 | 0.6373 | 0.6373 @ 20% | Largest val-test gap; weaker encoder for this recipe. |

Key takeaways:

1. The 20% runs beat the published UniXcoder full-data reference for ModernBERT-base and unixcoder-base under the same CoDET-M4 author Macro-F1 metric.
2. The 5% batch here is **below** the older `results/Legacy/exp_fs_inline_hier_ntk_frac0.05_seed42.json` result (**0.6709**). Do not merge the two as identical protocols: this batch uses bs=128, bf16, seq512, offline encoder aliases, and the current portfolio script.
3. The 1% regime is still unstable. UnixCoder is best at 1%, but all 1% runs remain far below paper baseline.
4. CodeBERT has consistent positive val-test gaps, especially 5% and 20%, suggesting worse transfer from mini-val to full test.

---

## 3. Protocol Notes

Script: `Exp_FewShot/testing_chis/run_hier_ntk_portfolio.py`

Default command shape:

```bash
FS_ENCODER=ModernBERT-base,codebert-base,unixcoder-base
FS_BENCHMARK=codet_m4
FS_SWEEP_FRACS=0.01,0.05,0.20
FS_SEED=42
python Exp_FewShot/testing_chis/run_hier_ntk_portfolio.py
```

Locked defaults from the current run:

| Field | Value |
| :-- | :-- |
| Benchmark | `codet_m4` |
| Task | `author` |
| Classes | 6 |
| Seed | 42 |
| Loss | HierTree family prior + NTK target-kernel alignment |
| `lambda_hier` | 0.4 |
| `lambda_ntk` | 0.4 |
| Epochs | 3 |
| Precision | bf16 |
| Sequence length | 512 |
| Batch size | 128 |

Paper-facing caution: use **test Macro-F1** as the primary number. Weighted-F1 and accuracy are only supporting metrics.

---

## 4. Published Baseline Reproduction Queue

Shared implementation: `Exp_FewShot/testing_chis/run_published_sota_portfolio.py`

Run one published baseline at a time:

```bash
CODET5_ENCODERS=codet5-base
FS_BENCHMARK=codet_m4
FS_SWEEP_FRACS=0.01,0.05,0.20
FS_SEED=42
python Exp_FewShot/testing_chis/exp00_sota_codet5_authorship.py

FS_ENCODER=ModernBERT-base,codebert-base,unixcoder-base
FS_BENCHMARK=codet_m4
FS_SWEEP_FRACS=0.01,0.05,0.20
FS_SEED=42
python Exp_FewShot/testing_chis/exp01_sota_detective_supcon.py
python Exp_FewShot/testing_chis/exp02_sota_faid_multitask.py

FS_BENCHMARK=codet_m4
FS_SWEEP_FRACS=0.01,0.05,0.20
FS_SEED=42
python Exp_FewShot/testing_chis/exp03_sota_style_repr_logreg.py
```

| Script | Key | Paper family | What the runner reproduces/adapts | Status |
| :-- | :-- | :-- | :-- | :-- |
| `exp00_sota_codet5_authorship.py` | `codet5_authorship` | CodeT5-Authorship, LLM code stylometry/authorship attribution | Encoder-only CodeT5, first-token embedding, 2-layer GELU/dropout classifier | ready |
| `exp01_sota_detective_supcon.py` | `detective_supcon` | DeTeCtive multi-level contrastive AI-text detection | CE + supervised contrastive style separation on code-author labels | ready |
| `exp02_sota_faid_multitask.py` | `faid_multitask` | FAID multi-task auxiliary + multi-level contrastive detection | Author CE + family auxiliary CE + class/family SupCon | ready |
| `exp03_sota_style_repr_logreg.py` | `style_repr_logreg` | Few-shot style representations for MGT detection | Code stylometry features + balanced logistic regression | ready |
| - | `ATC` | Zero-shot LLM-generated code detection via task conditioning | Needs task prompt reconstruction; not safe to compare on CoDET author yet | blocked |
| - | `Binoculars/Ghostbuster` | Training-free/weak-LM text detectors | Binary human-vs-AI only unless adapted; not author attribution | optional |

Important: these are CoDET-M4 adaptations for fair comparison under our task. In paper text, call them "published-method adaptations" unless we exactly rerun the authors' released benchmark/code.

---

## 5. Extra Benchmark Task Check

From `docs/references`:

| Benchmark | Official authorship/attribution task? | Recommended run | Paper role |
| :-- | :-- | :-- | :-- |
| CoDET-M4 | Yes: 6-way author identification, human + 5 LLMs | already primary | headline |
| AICD-Bench | Yes: **Task 2 model-family attribution**, 12 classes | `FS_BENCHMARK=aicd_t2` | optional attribution stress test |
| DroidCollection | No official generator-authorship task; official tasks are 2/3/4-class detection/refined/adversarial | `FS_BENCHMARK=droid_t3` then `droid_t4` | cross-benchmark detection/adversarial |

Runner support:

```bash
# AICD authorship-like attribution
FS_BENCHMARK=aicd_t2
FS_ENCODER=ModernBERT-base,unixcoder-base
FS_SWEEP_FRACS=0.01,0.05
FS_SEED=42
python Exp_FewShot/testing_chis/run_hier_ntk_portfolio.py

# Droid detection/adversarial, not authorship
FS_BENCHMARK=droid_t3
FS_ENCODER=ModernBERT-base,unixcoder-base
FS_SWEEP_FRACS=0.05,0.20
FS_SEED=42
python Exp_FewShot/testing_chis/run_hier_ntk_portfolio.py
```

Note: `run_hier_ntk_portfolio.py` now supports `aicd_t1`, `aicd_t2`, `aicd_t3`, `droid_t3`, and `droid_t4`. For Droid, string labels are mapped explicitly instead of using modulo arithmetic.

---

## 6. Append-Only Run Blocks

### Batch 2026-05-10 - Portfolio Seed 42

Source files:

```text
results/hier_ntk_p1_ModernBERT-base_frac0.01_seed42.json
results/hier_ntk_p2_ModernBERT-base_frac0.05_seed42.json
results/hier_ntk_p3_ModernBERT-base_frac0.2_seed42.json
results/hier_ntk_p4_codebert-base_frac0.01_seed42.json
results/hier_ntk_p5_codebert-base_frac0.05_seed42.json
results/hier_ntk_p6_codebert-base_frac0.2_seed42.json
results/hier_ntk_p7_unixcoder-base_frac0.01_seed42.json
results/hier_ntk_p8_unixcoder-base_frac0.05_seed42.json
results/hier_ntk_p9_unixcoder-base_frac0.2_seed42.json
results/hier_ntk_portfolio_seed42.json
```

```text
BEGIN_CHIS_TABLE method=FS-Hier-NTK batch=2026-05-10_seed42
  benchmark: codet_m4
  task: author
  seed: 42
  encoders: ModernBERT-base, codebert-base, unixcoder-base
  fractions: 0.01, 0.05, 0.20
  precision: bf16
  batch_size: 128
  seq_len: 512
  epochs: 3
  lambda_hier: 0.4
  lambda_ntk: 0.4
  best: hier_ntk_p3 ModernBERT-base frac=0.20 test_macro_f1=0.6895
  paper_ref: UniXcoder full-data Table7 test_macro_f1=0.6633
END_CHIS_TABLE
```
