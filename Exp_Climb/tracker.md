# Exp_Climb Tracker — Data-Efficient Dual-Bench Leaderboard

> **Strategy:** Each `exp_NN_*.py` file trains ONE method on BOTH target benchmarks (**CoDET-M4** + **DroidCollection**) sequentially, using **~20% of the training data** while evaluating on the **FULL test set**.
>
> **Paper angle:** "With only 20% of training samples, our method matches or exceeds full-data paper baselines on two major AI-code-detection benchmarks (ACL 2025 + EMNLP 2025)."

---

## Folder structure (modular, no code duplication)

```
Exp_Climb/
├── _common.py          # bootstrap deps, SpectralConfig, autocast, H100 profile
├── _features.py        # AST + structural + spectral feature extractors
├── _model.py           # AICDDataset, SpectralCode backbone, sub-encoders
├── _trainer.py         # Trainer (loss-fn-agnostic), FocalLoss, default_compute_losses
├── _data_codet.py      # CoDETM4Config, IID + OOD LOO suite (all 5 paper tables)
├── _data_droid.py      # DroidConfig, T1/T3/T4 suite (paper Tables 3/4)
├── _climb_runner.py    # run_full_climb() -- orchestrator, combined paper table
├── _paper_table.py     # emit_paper_table() -- BEGIN/END_PAPER_TABLE markdown block
├── tracker.md          # this file
└── exp_NN_<method>.py  # ONE file per method (numbered), thin ~200-line wrappers
```

**Rule:** shared code is in `_*.py`. Method-specific loss + tree/constraint logic lives in `exp_NN_*.py` and is passed to `Trainer` via the `loss_fn=` parameter.

---

## Kaggle workflow (per exp file)

Each `exp_NN_*.py` is **fully standalone** — upload it alone to Kaggle, it auto-clones the repo and imports the shared `_*.py` helpers:

```python
# Top of every exp file
REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"
# Auto-clones to /kaggle/working/ai_code_detection if not already present,
# then adds Exp_Climb/ to sys.path and imports _common, _data_codet, etc.
```

Run it → at bottom of log there's a `BEGIN_PAPER_TABLE ... END_PAPER_TABLE` block → paste into this file's Leaderboard section below.

---

## Protocol (identical for every climb method)

| Setting | Value |
|---|---|
| Train subsample | **100 000** (≈ 20% of ~500K CoDET-M4, ≈ 10% of ~1M Droid) |
| Val subsample | 20 000 |
| **Test** | **FULL test set (no subsampling)** |
| Hardware | NVIDIA H100 80GB HBM3 |
| Precision | bf16 |
| Batch | 64 × 1 |
| Epochs | 3 |
| Seed | 42 |
| Flow per file | CoDET full suite (IID binary + IID author + OOD gen×5 + OOD lang×3 + OOD src×3 = **13 runs**) → cleanup (VRAM + gc) → Droid T1 + T3 + T4 = **3 runs** → combined paper table → **16 runs total** |

---

## 📊 Paper baselines to beat

### CoDET-M4 (Orel, Azizov & Nakov, **ACL Findings 2025**)

**Table 2 — Binary IID Macro-F1 (%):**

| Model | Binary F1 | Note |
|:------|:---------:|:-----|
| **UniXcoder** | **98.65** | paper best |
| CodeT5 | 98.35 | |
| CodeBERT | 95.70 | |
| CatBoost | 88.78 | classical ML |
| SVM | 72.19 | |
| Baseline (logistic) | 62.03 | |

**Table 3 — Binary per-language (UniXcoder reference):** C++ 98.24 / Java 99.02 / Python 98.60.

**Table 4 — Binary per-source (UniXcoder reference):** CodeForces 96.54 / LeetCode 97.87 / GitHub 98.46.

**Table 7 — Author 6-class IID Macro-F1:**

| Model | Author F1 |
|:------|:---------:|
| **UniXcoder** | **66.33** |
| CodeBERT | 64.80 |
| CodeT5 | 62.45 |
| CatBoost | 45.42 |
| SVM | 27.63 |

**Table 8 — OOD Generator LOO avg F1:** UniXcoder **93.22**, CatBoost 92.31, CodeT5 79.43.

**Table 9 — OOD Source LOO avg F1:** CodeT5 **58.22**, UniXcoder 55.01, CatBoost 50.62.

**Table 10/12 — OOD Language LOO avg F1:** UniXcoder **88.96**, CodeT5 71.47, CodeBERT 58.78.

### DroidCollection (Orel, Paul, Gurevych, Nakov, **EMNLP 2025**)

**Table 3/4 — 3-class (human / generated / refined) Weighted-F1 averaged across domains:**

| Model | Droid 3-class W-F1 |
|:------|:---------:|
| **DroidDetectCLS-Large** | **0.8878** |
| DroidDetectCLS-Base | 0.8676 |
| CoDet-M4FT | 0.8325 |
| GPT-SnifferFT | 0.8275 |
| M4FT | 0.7350 |

---

## 🎯 SOTA targets (goals for every climb method)

| Benchmark / Task | Paper best | **Our target** | Minimum to claim SOTA |
|:-----------------|:----------:|:--------------:|:---------------------:|
| CoDET-M4 Binary IID Macro-F1 | 98.65 (UniXcoder) | **≥ 99.00** | > 98.65 |
| CoDET-M4 Author (6-class) Macro-F1 | 66.33 (UniXcoder) | **≥ 70.00** | > 66.33 |
| CoDET-M4 OOD Generator avg | 93.22 (UniXcoder) | ≥ 94.00 | > 93.22 |
| CoDET-M4 OOD Source avg | 58.22 (CodeT5) | **≥ 58.50** | > 58.22 |
| CoDET-M4 OOD Language avg | 88.96 (UniXcoder) | ≥ 85.00 (stretch) | > 88.96 (hard) |
| Droid T3 Weighted-F1 | 0.8878 (DroidDetectCLS-Large) | **≥ 0.90** | > 0.8878 |
| Droid T4 Weighted-F1 (adversarial-aware) | — | ≥ 0.85 | first to report |

**Data-efficiency framing:** any of the targets above, achieved with ~20% training data, becomes a paper-worthy headline. Paper baselines all use 100% training data.

---

## 📈 Method leaderboard

Fill in after each `exp_NN_*.py` run by pasting the `BEGIN_PAPER_TABLE` block. Bold = beats corresponding paper baseline.

| # | Method | File | Train % | CoDET Bin F1 | CoDET Auth F1 | Droid T3 W-F1 | Droid T4 W-F1 | CoDET OOD-Src | Status |
|:-:|:-------|:-----|:-------:|:------------:|:-------------:|:-------------:|:-------------:|:-------------:|:------:|
| 0 | **HierTreeCode** | [exp_00_hiertree.py](exp_00_hiertree.py) | 20% | TBD | TBD | TBD | TBD | TBD | ⏳ run pending |

---

## How to add a new method

1. Copy `exp_00_hiertree.py` → `exp_NN_<method>.py` (bump `NN`).
2. Replace the three method-specific pieces:
   - Custom loss class (if any), e.g. `SupConLoss`, `TopologyLoss`...
   - A `<method>_compute_losses(model, outputs, labels, config, focal_loss_fn)` function returning `{"total": ..., ...}`.
   - Pass that function as `loss_fn=` to `run_full_climb`.
3. Keep `run_full_climb(method_name="...", exp_id="exp_NN", ...)` — this handles both benches + combined paper table automatically.
4. Run on Kaggle, copy the `BEGIN_PAPER_TABLE` block, update row in the leaderboard above.

**Candidate methods to climb next** (ranked by tracker performance on individual benches):

| Priority | Method | Source exp | Why |
|:--------:|:-------|:-----------|:----|
| 1 | TokenStat | [Exp_DM/exp09_token_stats.py](../Exp_DM/exp09_token_stats.py) | Best Droid T3/T4 in DM tracker (0.8556/0.8488), untested on CoDET |
| 2 | SpectralCode | [Exp_DM/exp11_spectral_code.py](../Exp_DM/exp11_spectral_code.py) | Strong on both benches (0.2983 AICD / 0.8473 Droid) |
| 3 | DeTeCtiveCode | [Exp_DM/exp27_detective_code.py](../Exp_DM/exp27_detective_code.py) | HierTree + multi-level SupCon + kNN blend — pending run |
| 4 | MoECode | [Exp_CodeDet/run_codet_m4_exp21_moe.py](../Exp_CodeDet/run_codet_m4_exp21_moe.py) | Best CoDET binary (99.09), second-best ensemble |

---

## Insights log (fill in post-run)

Keep it to one paragraph per method. Per-class detail lives in the `BEGIN_PAPER_TABLE` block in the raw log, not here.

### exp_00 HierTreeCode (pending)
- **Expected strengths:** CoDET Author 6-class (family tree forces Qwen/Nxcode to cluster, matching the paper's known confusion).
- **Expected weaknesses:** Droid adversarial class — hier loss may over-regularize when the "families" are shallow (3–4 class).
- **Data-efficiency claim:** prior Exp18 runs at 20% train already beat UniXcoder (+4.22 Author, +0.41 Binary, and +0.41 OOD-Source) on CoDET. This climb file is the FULL-TEST re-run + first Droid pass under the same backbone.
- **Result after run:** _(paste BEGIN_PAPER_TABLE block summary)_

---

## Paper narrative (Table 1 of the eventual draft)

When 2+ methods have climbed, the "Method leaderboard" above becomes Table 1. The left-most "Train %" column is the pitch:

> "Table 1. Data-efficient SOTA. Our methods trained on 20% of each dataset match or exceed full-data paper baselines on both CoDET-M4 (ACL 2025) and DroidCollection (EMNLP 2025) across binary, attribution, and OOD settings."

Then each subsequent section (binary, author, OOD generator/language/source, Droid adversarial) cites the paper's table and our Δ.

---

## Not in this folder

- **AICD-Bench** — excluded deliberately. 23+ methods in `Exp_DM/` all fail to close val-test gap (0.99 → 0.25 on T1). Paper treats AICD as "open challenge / negative result," not a main claim.
- **Ablations** — belong in `Exp_DM/` / `Exp_CodeDet/` where you freely tune. Climb files are for final dual-bench SOTA runs only.
- **Wrapper / variant methods** (exp28/29/30) — if they graduate, they become their own `exp_NN_*.py`.
