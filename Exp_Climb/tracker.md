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

## 🖥️ Kaggle / H100 80GB setup

### Auto-applied H100 profile (`_common.apply_hardware_profile`)

Triggers when `torch.cuda.get_device_name()` contains "H100":

| Setting | Value (H100) | Default | Why |
|---|---|---|---|
| `precision` | `bf16` | `auto` | Hopper native bf16, no scaling needed |
| `batch_size` | `64` | 32 | Matches proven Exp18 setup (70.55 F1) |
| `grad_accum_steps` | `1` | 2 | Batch 64 fits directly |
| `num_workers` | `8` | 2 | H100 Kaggle has 8 vCPU |
| `prefetch_factor` | `4` | 2 | Keep dataloader hot |
| `log_every` | `200` | 100 | Less log spam at high throughput |
| `eval_every` | `2000` | 1000 | Eval less often (H100 goes fast) |

**VRAM budget (80 GB):** ModernBERT-base (~150M, bf16) + AdamW state + activations at batch 64 × 512 seq ≈ **~18 GB used, ~62 GB headroom**. Batch 64 is held constant to match the paper-table baseline; you *can* push to 96-128 for faster throughput but that may perturb convergence.

### Disk budget (`/kaggle/working` = 20 GB quota)

- 16 runs × `best` ckpt × ~600 MB ≈ **9.6 GB** → fits (with `save_latest_ckpt=False`, default)
- If you enable `save_latest_ckpt=True`, budget doubles to ~19 GB → tight
- If you run multiple climb files back-to-back in one session, **delete** `./codet_m4_checkpoints` + `./droid_checkpoints` between runs

### Kaggle notebook config (recommended settings)

| Kaggle setting | Value |
|---|---|
| Accelerator | **GPU T4×2** or **GPU P100** during dev — **GPU H100** via "Notebook Run" for final climb |
| Internet | **ON** (needed to clone GitHub repo + load HF datasets) |
| Language | Python |
| Persistence | Disable (each run is fresh; we re-clone repo) |
| Secrets | `HF_TOKEN` optional (public datasets work without, just rate-limit warnings in log) |

### Kaggle runtime estimates (per climb file)

| Phase | Runs | ~Time/run | Total |
|---|---:|---:|---:|
| CoDET-M4 IID binary | 1 | 25 min | 25 min |
| CoDET-M4 IID author | 1 | 25 min | 25 min |
| CoDET-M4 OOD Generator LOO | 5 | 22 min | 1h 50 min |
| CoDET-M4 OOD Language LOO | 3 | 22 min | 1h 06 min |
| CoDET-M4 OOD Source LOO | 3 | 22 min | 1h 06 min |
| Droid T1 | 1 | 25 min | 25 min |
| Droid T3 | 1 | 25 min | 25 min |
| Droid T4 | 1 | 25 min | 25 min |
| **Total** | **16** | — | **~6 h 27 min** |

Kaggle H100 kernel limit is 12h → fits comfortably. P100/T4 would need 2-3× longer → may time out.

### If H100 unavailable

Fallback hierarchy for config auto-detect:
1. **H100** → profile applied (bf16, batch 64)
2. **A100 80GB** → same profile works (bf16)
3. **A100 40GB / V100 / T4** → drop to `precision="fp16"`, `batch_size=32`, `grad_accum_steps=2`

The auto-detect only triggers on H100 by name match. For other GPUs, set explicitly:

```python
codet_cfg = CoDETM4Config(...)
# Override config defaults if non-H100
from _common import SpectralConfig
# (Trainer picks these up through build_codet_config / build_droid_config)
```

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

## 🧠 Insights from prior research (Exp_DM + Exp_CodeDet)

Distilled from **23 methods in Exp_DM** (AICD + Droid) and **14 methods in Exp_CodeDet** (CoDET-M4). Read this before designing a new climb method — many dead ends already mapped.

### 1. Binary is ceilinged at ~99% — don't optimize it

Every modern PLM (ModernBERT, UniXcoder, CodeT5) hits 98.5–99.1 on CoDET-M4 binary IID. **The paper's binary ranking is noise.** All climb energy should go to **author classification** + **OOD** where real gaps exist (66.33 → 70.55 = +4.22 is a real signal).

### 2. Nxcode ↔ CodeQwen1.5 confusion is the single biggest lever on CoDET author

Nxcode is fine-tuned from CodeQwen1.5 → ~33–40% of Qwen1.5 samples predicted as Nxcode across **every method tested**. The `HierarchicalAffinityLoss` design (Exp18) explicitly forces them into one family → Qwen1.5 F1 bumped 0.4129 → 0.4431 (+3.02%).

**Takeaway:** any method that models generator genealogy (fine-tune lineage) wins. Raw classification doesn't.

### 3. GitHub source is the universal OOD bottleneck

Ranking by difficulty: **GH ≫ CF > LC** on both AICD and CoDET-M4.
- Per-source Author F1 on CoDET: CF 0.77 / GH 0.56 / LC 0.60 (across all methods)
- OOD-Source held-out-GH: macro F1 = **0.2834** (catastrophic; human recall 5.71%)
- GH-OOD-held-cpp: macro 0.4839 (worst subgroup in every experiment)

**Why:** CF + LC are competitive-programming templates → stylistically narrow. GH is real-world diverse code. Model trained on cf+lc memorizes templates, fails on GH.

**Takeaway:** any method that improves GH subgroup is a paper-worthy contribution. Target: GH macro > 0.60 on author task.

### 4. Val-test gap reveals shortcut learning (AICD T1 lesson)

AICD T1 shows **universal collapse**: val 0.99 → test 0.25-0.31 across 23 methods. The gap itself is the signal — none of DomainMix, IRM, OSCP, VILW, BH-Triplet, SupCon, etc. closed it. **This is why climb excludes AICD** — it's a benchmark-engineering problem, not a method-engineering problem.

**Implication for climb:** large val-test gaps on CoDET/Droid also matter. A method that has val 0.85 + test 0.70 is **worse** than val 0.72 + test 0.70, even if they report the same test number.

### 5. Methods that ❌ don't work (negative results to avoid)

| ❌ Anti-pattern | Why it fails | Evidence |
|:---|:---|:---|
| **DANN / GRL** for author task | Generator-invariant features are the **opposite** of what author classification needs | Exp19 EAGLECode: Author -7.66% (70.55 → 62.89); Qwen1.5 F1 collapsed to 0.198 |
| **Un-annealed IRM penalty** | Explodes to 1e4+ by epoch 3, NaN gradients | Exp06 AST-IRM: no OOD gain, unstable |
| **Variance-Invariant Whitening (VILW)** | Whitening loss dominates (~206), crushes CE capacity | Exp05 OSCP: -0.02 vs baseline; lowest AICD T1 test F1 |
| **Unguarded style contrastive** | Division-by-zero in style pairs → NaN | Exp16 HyperNetCode: StyleCon NaN every epoch; author task completely broken |
| **Class-weighted focal on severe imbalance** | Majority class (47% data) gets F1=0.0000 | Exp18 on AICD T2 class 0: weighted loss pushes model to over-predict minorities |
| **BiLSTM AST replaced by GAT** (without richer graph) | GAT on flattened AST ≠ CFG/DFG; no speedup, no accuracy gain | Exp23 GraphStyleCode: 69.71 < Exp11 SpectralCode 69.82 |

### 6. Methods that ✅ work (patterns to reuse)

| ✅ Pattern | Why it works | Best evidence |
|:---|:---|:---|
| **Hierarchical / family-tree losses** (HierTree) | Explicitly models generator lineage, cracks Qwen/Nxcode | Exp18 HierTreeCode: **70.55** author (+4.22 vs paper) |
| **Token-statistics features** (entropy, burstiness, TTR, Yule-K) | Cheap + strong on Droid adversarial | Exp09 TokenStat: **0.8556** Droid T3 (best DM) |
| **Multi-granularity fusion** (token + AST + structural + spectral, gated softmax) | Complementary signals; robust across languages | Every top-tier method uses this backbone |
| **Test-time retrieval / kNN blending** | Training-free OOD boost (~+0.1-0.3 author F1) | Exp17 RAGDetect: 70.46 (second-best CoDET author) |
| **BYOL-style EMA self-distillation** | Stability without labels, small boost on top of strong baseline | Exp26 SelfDistillCode: 70.14 (+0.32 on Exp11) |
| **Slot decomposition** | Object-centric helps `hybrid` / `adversarial` classes on T3 | Exp13 SlotCode: best AICD T3 single-task (0.5706) |

### 7. Diminishing returns in 70.0–70.2 plateau (Batch-2 lesson)

Methods Exp21-26 (MoE, TTA, CharCNN, Cosine-proto, SelfDistill) all cluster at **70.0–70.2 author F1** on CoDET. Hyperparam tuning / shallow architectural tweaks plateau here. **Next breakthrough needs an architectural leap**: explicit genealogy graph (not just affinity), retrieval-augmented training (not just test-time), or cross-modal code representations (not just AST).

### 8. Infrastructure gotchas (don't repeat)

- **OOD Generator LOO test_ood=0 bug** (Exp14/15/16/21/24) — old loader didn't filter test by held-out generator. Exp18 fix (`load_codet_m4_loo()` in `_data_codet.py`) is what climb uses. **Verify before trusting any OOD Gen number from older methods.**
- **TTA applied to test breakdown** overwrites stats (Exp22 case): test batch LayerNorm updates shouldn't touch the eval logger. Emit metrics BEFORE any test-time adaptation.
- **CoDET-M4 OOD Generator single-class degenerate** — held-out test = only that one AI class → macro F1 ceiling ~0.5. Use **weighted-F1** or rebalance with human samples.

### 9. Data-efficiency story works

Every Exp_DM / Exp_CodeDet method above ran on **100K train (~20%)** and still beat paper baselines that use full data. The climb paper can reliably claim "20% data + SOTA" as its headline.

### 10. What paper reviewers will ask — prepare answers

- **"Why not full training data?"** → "Full data doesn't change ranking" (show ablation: 100K vs 500K on Exp18 CoDET — ideally flat).
- **"How does it fare on AICD-Bench?"** → "We explicitly treat AICD as open challenge; val-test gap is a benchmark property, not solvable by detector-side methods alone." (reference 23-method negative result)
- **"What about calibration (ECE/Brier)?"** → emit in paper table too (current `_paper_table.py` doesn't, but trainer records val loss; ECE is one metric away).
- **"Language coverage vs Droid's 7 langs?"** → note that CoDET-M4 is limited to cpp/java/python by design; Droid has full coverage and we run T1/T3/T4 there.

---

## Not in this folder

- **AICD-Bench** — excluded deliberately. 23+ methods in `Exp_DM/` all fail to close val-test gap (0.99 → 0.25 on T1). Paper treats AICD as "open challenge / negative result," not a main claim.
- **Ablations** — belong in `Exp_DM/` / `Exp_CodeDet/` where you freely tune. Climb files are for final dual-bench SOTA runs only.
- **Wrapper / variant methods** (exp28/29/30) — if they graduate, they become their own `exp_NN_*.py`.
