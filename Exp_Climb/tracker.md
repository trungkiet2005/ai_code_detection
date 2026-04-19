# Exp_Climb Tracker — Data-Efficient Dual-Bench Leaderboard

> **Strategy:** Each `exp_NN_*.py` file trains ONE method on BOTH target benchmarks (**CoDET-M4** + **DroidCollection**) sequentially, using **~20% of the training data** while evaluating on the **FULL test set**.
>
> **Paper angle:** "With only 20% of training samples, our method matches or exceeds full-data paper baselines on two major AI-code-detection benchmarks (ACL 2025 + EMNLP 2025)."

---

## 🏆 Climb Leaderboard — sorted by CoDET Author Macro-F1 ↓

> **Primary metric** = **Author IID Macro-F1** (6-class, hardest task). Binary is ceiling-bound ~99%.
> Paper baselines: **UniXcoder** 98.65 binary / **66.33** author / 88.96 lang-OOD / **DroidDetectCLS-Large** 0.8878 T3.
> Δ-paper columns compare against the SAME paper baseline on that task.

All 8 lean-mode tasks are reported below (CoDET: 2 IID + 3 OOD representative · Droid: T1 + T3 + T4).

| Rank | Exp | Method | Mode | **Bin** | Δ | **Auth** | **Δ** | **Gen-qw** | **Lang-py** | **Src-gh** | **T1** | **T3** | Δ | **T4** | Status |
|:----:|:---|:-------|:----:|:-----:|:---:|:-----:|:-----:|:----------:|:-----------:|:----------:|:------:|:------:|:---:|:------:|:------:|
| 🥇 | **Exp_06** | **FlowCodeDet** | lean | 99.02 | `+0.37` | **70.90** | **`+4.57`** | 49.67 | **64.50** | **33.36** | 97.01 | **89.34** | `+0.56` | **88.30** | ✅ |
| 🥈 | **Exp_07** | SAMFlatCode | lean | 99.05 | `+0.40` | 70.22 | `+3.89` | 49.74 | 54.13 | 31.41 | 96.92 | 88.73 | `-0.05` | 87.51 | ✅ |
| 🥉 | **Exp_02** | GHSourceInvariantCode | lean | 98.99 | `+0.34` | 70.20 | `+3.87` | 49.76 | 50.70 | 30.44 | 97.03 | 88.05 | `-0.73` | 87.38 | ✅ |
| 4 | **Exp_00** | HierTreeCode (baseline) | full | 99.03 | `+0.38` | 69.93 | `+3.60` | 49.68 | 53.25 | 27.20 | 97.08 | 88.63 | `-0.15` | 87.60 | ✅ |
| 5 | **Exp_03** | TokenStatRAGCode | lean | 99.05 | `+0.40` | 69.90 | `+3.57` | 49.62 | 52.60 | 30.19 | 97.07 | 88.94 | `+0.16` | 87.40 | ✅ |
| 6 | **Exp_08** | POEMPolarizedCode | lean | **99.06** | `+0.41` | 69.68 | `+3.35` | 49.72 | 54.11 | 33.33 | 97.05 | 88.57 | `-0.21` | 87.48 | ✅ |
| 7 | **Exp_04** | PoincareGenealogy | lean | 99.01 | `+0.36` | 69.58 | `+3.25` | 49.65 | 47.61 | 26.47 | 96.97 | **89.76** | `+0.98` | 87.99 | ✅ |
| 8 | **Exp_05** | SinkhornOTCode | lean | 99.05 | `+0.40` | 68.40 | `+2.07` | 49.67 | 57.54 | 28.96 | 97.01 | 88.03 | `-0.75` | 87.52 | ✅ |
| — | Exp_01 | GenealogyGraphCode | lean | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_09 | EpiplexityCode | lean | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_10 | PredictiveCodingCode | lean | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_11 | PersistentHomologyCode | lean | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_12 | AvailabilityPredictivityCode | lean | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_13 | NTKAlignCode | lean | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| **REF** | Paper | **UniXcoder** | — | 98.65 | — | **66.33** | — | — | — | — | — | — | — | — | reference |
| REF | Paper | CodeT5 | — | 98.35 | `-0.30` | 62.45 | `-3.88` | — | — | — | — | — | — | — | reference |
| REF | Paper | CodeBERT | — | 95.70 | `-2.95` | 64.80 | `-1.53` | — | — | — | — | — | — | — | reference |
| **REF** | Paper | **DroidDetectCLS-Large** | — | — | — | — | — | — | — | — | — | **88.78** | — | — | reference (Droid) |
| REF | Paper | DroidDetectCLS-Base | — | — | — | — | — | — | — | — | — | 86.76 | `-2.02` | — | reference (Droid) |
| REF | Paper | CoDet-M4FT (Droid) | — | — | — | — | — | — | — | — | — | 83.25 | `-5.53` | — | reference (Droid) |

Columns:
- **Bin / Auth** = CoDET-M4 IID binary / author Macro-F1 (×100). Δ vs UniXcoder 98.65 / 66.33.
- **Gen-qw / Lang-py / Src-gh** = CoDET-M4 OOD LOO Macro-F1 (×100) for held-out generator=qwen1.5, language=python, source=gh. Note: OOD-GEN LOO test contains only the held-out AI class → macro-F1 is ceiling-bound ~0.5 (signal is in weighted-F1 and per-class delta, not in the raw macro number).
- **T1 / T3 / T4** = DroidCollection Weighted-F1 (×100) for 2/3/4-class tasks. Δ on T3 vs DroidDetectCLS-Large 88.78.

### Quick reads

- **Best CoDET Author:** `exp_06 FlowCodeDet` at **70.90** (+4.57 vs UniXcoder). First climb entry to clear **70.6**; class-conditioned flow-matching auxiliary beats plain HierTree on Author.
- **Best Droid T3:** `exp_04 PoincareGenealogy` at **89.76** (+0.98 vs DroidDetectCLS-Large). Hyperbolic geometry helps ID more than it helps author/OOD.
- **Best OOD-SRC-gh:** tie `exp_06 FlowCodeDet` **33.36** / `exp_08 POEMPolarizedCode` **33.33** — both break the 0.30 GH ceiling with different mechanisms.
- **Best OOD-LANG-python:** `exp_06 FlowCodeDet` **64.50** — +11 pts over the method pack (most cluster 47–57). Flow matching helps the hardest language LOO.
- **Best Droid T1 (binary):** `exp_00 HierTreeCode` **97.08** — all methods sit in 96.9–97.1 range (noise-level spread, as expected for binary ceiling).
- **Best Droid T4 (4-class adversarial):** `exp_06 FlowCodeDet` **88.30** — beats the pack by ~0.3–0.9 pts; consistent with its best-on-board ranking elsewhere.
- **OOD-GEN qwen1.5 flat at ~49.7 across all methods** — expected: LOO test contains only qwen1.5 samples → macro-F1 is degenerate (ceiling 0.5 on a 2-class head seeing only one class). No method-level signal here; we keep the column to detect regressions, not to rank.
- **Compare to Exp_CodeDet board:** Exp27 DeTeCtiveCode holds **71.53** Author on full CoDET (different codebase, HierTree + dual SupCon + kNN). The climb board's **70.90** is best-to-date for the `run_full_climb` harness.
- **Still pending:** Exp_01 + 09–13 (6 files). Journal-grade methods (Epiplexity / PredictiveCoding / PH / NTK / Availability) unclaimed.

> Δ-Bin vs UniXcoder 98.65. Δ-Auth vs UniXcoder 66.33. Δ-T3 vs DroidDetectCLS-Large 0.8878 (×100). No paper baseline for OOD-SRC-gh (per-source LOO subgroup), OOD-GEN LOO, Droid T1/T4, or OOD-LANG held-out-python — those stand as relative comparisons between our methods. Status ✅ = run finished; ⏳ = file exists, not yet run on Kaggle.

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

| Setting | H100 value | Base default | Why |
|---|---|---|---|
| `precision` | `bf16` | `auto` | Hopper native bf16, no loss scaling needed |
| `batch_size` | **`128`** | 32 | 2× activations → utilize 80 GB VRAM (~40 GB used) |
| `max_length` | **`512`** | 512 | Kept at paper-baseline for direct comparability. *Seq 1024 OOMs at batch 128 in bf16 — attention + RoPE buffers blow past 80 GB. Use seq 1024 only with batch 64.* |
| `lr_encoder` | **`2.8e-5`** | 2e-5 | sqrt(2) LR scaling for 2× batch |
| `lr_heads` | **`1.4e-4`** | 1e-4 | same |
| `grad_accum_steps` | `1` | 2 | Batch 128 fits directly |
| `num_workers` | `8` | 2 | Kaggle H100 kernel has 8 vCPU |
| `prefetch_factor` | `4` | 2 | Keep dataloader hot |
| `log_every` | `200` | 100 | Less log spam at high throughput |
| `eval_every` | `2000` | 1000 | Eval less often (H100 goes fast) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | — | Auto-set in `_common.py`; reduces fragmentation for long-lived runs |

**VRAM budget (80 GB) — empirically measured, batch 128 × seq 512:**

| Component | Size (bf16) |
|:----------|:-----------:|
| ModernBERT-base params + grads + Adam state | ~5 GB |
| Activations (forward, batch 128 × seq 512 × 768 × 22 layers) | **~20 GB** |
| Backward-pass activations + RoPE buffers | ~15 GB |
| PyTorch/CUDA overhead + fragmentation | ~5 GB |
| **Total used** | **~40-45 GB (50-55%)** |
| Headroom | ~35-40 GB |

Previously batch 64 × seq 512 used only ~18 GB (22%) — wasteful.
**Attempted:** batch 128 × seq 1024 (~79 GB forward alone → OOM on backward). Dropped to seq 512 for safety.

**Why not batch 192+ or ModernBERT-large?**
- Batch 192: ~60 GB forward + ~30 GB backward = 90 GB total → OOM
- Batch 256: same issue, worse
- ModernBERT-large (~395M) + batch 128: ~70 GB forward → OOM on backward, plus 2× runtime risks 12h timeout
- If user wants **seq 1024**: must explicitly drop batch to 64 (override `CoDETM4Config(max_train_samples=...)` isn't enough — need to set `SpectralConfig.max_length=1024` + `batch_size=64` manually)
- **Current config (batch 128 × seq 512) is the Pareto optimum for H100 80GB + 12h Kaggle limit.**

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

### Kaggle runtime estimates (per climb file, batch 128 × seq 512)

Based on measured Exp18 runs (~5-10 min/run at batch 64 × seq 512). Batch 128
processes ~2× tokens per step with minimal kernel overhead, so steps/epoch
halve → net ~10% faster per epoch.

| Phase | Runs | ~Time/run | Total |
|---|---:|---:|---:|
| CoDET-M4 IID binary | 1 | ~7 min | 7 min |
| CoDET-M4 IID author | 1 | ~7 min | 7 min |
| CoDET-M4 OOD Generator LOO | 5 | ~6 min | 30 min |
| CoDET-M4 OOD Language LOO | 3 | ~6 min | 18 min |
| CoDET-M4 OOD Source LOO | 3 | ~6 min | 18 min |
| Droid T1 | 1 | ~7 min | 7 min |
| Droid T3 | 1 | ~7 min | 7 min |
| Droid T4 | 1 | ~7 min | 7 min |
| **Total** | **16** | — | **~1 h 41 min** |

**Lean mode** (`run_mode="lean"`) — fast screening, 8 runs:

| Phase | Runs | ~Time/run | Total |
|---|---:|---:|---:|
| CoDET-M4 IID binary + author | 2 | ~22 min | 44 min |
| OOD-SRC held-out=gh (hardest) | 1 | ~22 min | 22 min |
| OOD-LANG held-out=python | 1 | ~22 min | 22 min |
| OOD-GEN held-out=qwen1.5 | 1 | ~22 min | 22 min |
| Droid T1 + T3 + T4 | 3 | ~22 min | 66 min |
| **Total** | **8** | — | **~2 h 56 min** |

> **Measured** on H100 BF16 batch=64 × seq=512 from exp_08 POEMPolarized run
> (2026-04-18, 10464 s total). Prior estimate "~53 min" was based on an
> optimistic extrapolation and proved too low by ~3×.
>
> `full` mode (16 runs) ≈ **~5 h 52 min** on the same hardware. Kaggle 12h
> kernel limit still safe, but plan only ONE full-mode run per session.

Use `lean` to screen ideas (1-2 ideas per session). Switch to `full` only
for paper-final winner after lean confirms novelty.

Kaggle H100 kernel limit: 12h → **massive buffer** (~7× runtime). P100/T4 fallback: ~3-4× longer (~6-8h) — drop batch to 64 manually on those GPUs.

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
| Flow per file (`full`) | CoDET full suite (IID binary + IID author + OOD gen×5 + OOD lang×3 + OOD src×3 = **13 runs**) → cleanup → Droid T1+T3+T4 = **3 runs** → paper table → **16 runs total** |
| Flow per file (`lean`) | CoDET IID binary + author + OOD-SRC-gh + OOD-LANG-python + OOD-GEN-qwen1.5 = **5 runs** → cleanup → Droid T1+T3+T4 = **3 runs** → paper table → **8 runs total, ~53 min** |

---

## 📊 Paper baselines to beat (full enumeration)

### CoDET-M4 (Orel, Azizov & Nakov, **ACL Findings 2025**)

All values from Tables 2-14 of `docs/references/paper_CodeDet_M4.md`. Metric: **Macro-F1 (%)** unless noted.

**Table 2 — Binary IID:**

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | 98.65 | 98.65 | **98.65** | 98.65 |
| CodeT5 | 98.36 | 98.35 | 98.35 | 98.35 |
| CodeBERT | 95.70 | 95.72 | 95.70 | 95.71 |
| CatBoost | 88.71 | 88.81 | 88.78 | 88.79 |
| SVM | 72.19 | 72.19 | 72.19 | 72.19 |
| Baseline | 62.03 | 65.17 | 62.03 | 65.17 |

**Table 3 — Binary per-language (F1):**

| Model | C++ | Python | Java |
|:------|:-:|:-:|:-:|
| **UniXcoder** | **98.24** | **98.60** | **99.02** |
| CodeT5 | 97.86 | 98.22 | 98.89 |
| CodeBERT | 95.73 | 94.84 | 96.54 |
| CatBoost | 91.94 | 86.04 | 88.81 |

**Table 4 — Binary per-source (F1):**

| Model | CodeForces | LeetCode | GitHub |
|:------|:-:|:-:|:-:|
| **UniXcoder** | 96.54 | **97.87** | 98.46 |
| CodeT5 | 97.24 | 66.23 | 98.54 |
| CodeBERT | 91.67 | 87.63 | 95.31 |
| CatBoost | 90.18 | 71.23 | 80.52 |

**Table 7 — Author 6-class IID Macro-F1 (primary author metric):**

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | 64.80 | 69.54 | **66.33** | 79.35 |
| CodeBERT | 63.14 | 68.10 | 64.80 | 77.65 |
| CodeT5 | 62.67 | 69.40 | 62.45 | 78.25 |
| CatBoost | 50.46 | 44.41 | 45.42 | 66.19 |
| SVM | 29.10 | 28.51 | 27.63 | 49.70 |

**Table 8 — OOD-Generator (unseen LLMs, binary). Precision excluded (only positive class in labels):**

| Model | R | **F** | A |
|:------|:-:|:-:|:-:|
| **UniXcoder** | 87.30 | **93.22** | 87.30 |
| CatBoost | 85.71 | 92.31 | 85.71 |
| CodeT5 | 65.87 | 79.43 | 65.87 |
| CodeBERT | 50.00 | 66.67 | 50.00 |
| SVM | 80.16 | 88.99 | 80.16 |
| Baseline | 29.37 | 59.65 | 64.68 |

**Table 9 — OOD-Source (unseen domains):**

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **CodeT5** | 78.43 | 59.18 | **58.22** | 74.11 |
| UniXcoder | 76.00 | 57.11 | 55.01 | 72.81 |
| CatBoost | 60.32 | 53.54 | 50.62 | 69.11 |
| Baseline | 67.31 | 50.34 | 49.84 | 50.30 |
| CodeBERT | 45.69 | 48.91 | 43.16 | 66.01 |
| SVM | 37.11 | 41.37 | 38.66 | 55.16 |

**Table 10 — OOD-Language per-language (UniXcoder is the clear winner):**

| Model | C# | Golang | JavaScript | PHP |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | **91.31** | **90.01** | **81.48** | **96.17** |
| CodeT5 | 78.55 | 88.78 | 34.81 | 93.66 |
| CodeBERT | 45.31 | 52.46 | 26.84 | 56.68 |
| CatBoost | 51.01 | 64.72 | 26.03 | 45.08 |
| SVM | 43.16 | 20.94 | 24.23 | 41.94 |
| Baseline | 39.60 | 46.15 | 56.27 | 28.33 |

**Table 12 — OOD-Language avg (unseen langs):**

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | 89.13 | 89.20 | **88.96** | 88.96 |
| CodeT5 | 76.87 | 73.29 | 71.47 | 72.17 |
| CodeBERT | 60.31 | 59.79 | 58.78 | 59.10 |
| CatBoost | 61.25 | 57.86 | 53.42 | 56.26 |
| Baseline | 70.53 | 57.36 | 51.53 | 59.64 |
| SVM | 26.42 | 38.27 | 28.68 | 36.29 |

**Table 13 — Hybrid Authorship (UniXcoder's ceiling, precision excluded):**

| Model | R | **F** | A |
|:------|:-:|:-:|:-:|
| **UniXcoder** | 33.22 | **39.36** | 64.71 |
| Baseline | 14.86 | 22.91 | 29.72 |

**Table 14 — Ternary (human / LLM / hybrid) Macro-F1:**

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | 86.48 | 85.93 | **86.10** | 86.16 |
| CodeBERT | 85.91 | 85.96 | 85.94 | 85.84 |
| CodeT5 | 79.72 | 78.78 | 78.99 | 79.43 |

### DroidCollection (Orel, Paul, Gurevych, Nakov, **EMNLP 2025**)

All values from Tables 3-6 of `docs/references/paper_Droid.md`. Metric: **Weighted-F1 (%)** unless noted. Best backbone = **DroidDetectCLS-Large**.

**Table 3 — 2-class / 3-class per-domain (Full Training split, Avg column is our canonical target):**

| Model (Full Training) | 2-Class Gen | 2-Class Algo | 2-Class RDS | **2-Class Avg** | 3-Class Gen | 3-Class Algo | 3-Class RDS | **3-Class Avg** |
|:---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **DroidDetectCLS-Large** | 99.38 | 98.39 | 93.24 | **97.00** | 93.08 | 92.86 | 80.42 | **88.78** |
| DroidDetectCLS-Base | 99.22 | 98.22 | 87.57 | 95.00 | 92.78 | 93.05 | 74.46 | 86.76 |
| CoDet-M4FT | 98.89 | 98.23 | 83.77 | 93.63 | 85.46 | 90.41 | 73.88 | 83.25 |
| GPT-SnifferFT | 97.72 | 96.52 | 80.46 | 91.56 | 89.42 | 88.12 | 70.72 | 82.75 |
| M4FT | 92.99 | 89.36 | 73.99 | 85.45 | 80.98 | 80.72 | 58.80 | 73.50 |
| **Zero-Shot: Fast-DetectGPT** | 75.07 | 63.05 | 65.43 | 67.85 | 66.43 | 62.90 | 64.30 | 64.54 |
| Zero-Shot: CoDet-M4 | 53.41 | 44.63 | 65.43 | 54.49 | 41.90 | 46.06 | 55.43 | 47.80 |

**Table 4 — 2-class / 3-class per-language (Full Training split, Avg column):**

| Model | 2-Class C/C++ | C# | Go | Java | Python | JS | **2-Class Avg** | 3-Class C/C++ | C# | Go | Java | Python | JS | **3-Class Avg** |
|:---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **DroidDetectCLS-Large** | 99.31 | 99.51 | 99.32 | 99.45 | 99.11 | 98.67 | **99.23** | 94.24 | 93.87 | 94.42 | 94.05 | 94.13 | 91.27 | **93.66** |
| DroidDetectCLS-Base | 99.29 | 99.33 | 99.32 | 99.45 | 98.87 | 98.38 | 99.11 | 94.43 | 94.06 | 93.98 | 93.93 | 93.95 | 90.99 | 93.56 |
| CoDet-M4FT | 99.36 | 99.22 | 99.31 | 99.04 | 98.28 | 99.24 | 99.08 | 89.98 | 88.94 | 89.73 | 91.70 | 85.80 | 91.46 | 89.60 |
| GPT-SnifferFT | 97.64 | 97.36 | 97.33 | 97.96 | 95.07 | 97.94 | 97.22 | 85.14 | 85.75 | 85.78 | 86.97 | 79.02 | 88.28 | 85.16 |
| M4FT | 94.18 | 89.98 | 92.19 | 92.63 | 87.19 | 93.61 | 91.63 | 79.56 | 75.55 | 77.63 | 79.55 | 69.63 | 77.53 | 76.57 |
| Fine-Tuned Baseline: CatBoost | 94.00 | 91.20 | 90.57 | 92.26 | 89.51 | 82.55 | 90.02 | 84.57 | 81.32 | 81.54 | 82.42 | 78.15 | 70.98 | 78.83 |
| Fine-Tuned Baseline: GCN | 79.06 | 78.33 | 84.33 | 80.04 | 72.49 | 69.69 | 77.32 | 65.97 | 58.03 | 65.20 | 60.13 | 55.22 | 54.72 | 59.88 |

**Table 5 — Adversarial robustness (Recall on human-written vs. adversarial samples):**

| Model | Human-written Recall | Adversarial Recall |
|:---|:-:|:-:|
| **DroidDetectCLS-Large** | **0.98** | **0.92** |
| DroidDetectCLS-Base | 0.93 | 0.92 |
| CoDet-M4FT | 0.96 | 0.51 |
| GPT-SnifferFT | 0.97 | 0.55 |
| M4FT | 0.91 | 0.67 |
| Fast-DetectGPT (zero-shot) | 0.84 | 0.48 |
| GPT-Sniffer (zero-shot) | 0.65 | 0.49 |
| M4 (zero-shot) | 0.40 | 0.73 |
| CoDet-M4 (zero-shot) | 0.38 | 0.63 |
| GPT-Zero | 0.53 | 0.10 |

**Table 6 — DroidDetect backbone ablation (Weighted-F1, avg across settings, our final method name in the paper):**

| Model | 2-Class Gen | 2-Class Algo | 2-Class RDS | 3-Class Gen | 3-Class Algo | 3-Class RDS |
|:---|:-:|:-:|:-:|:-:|:-:|:-:|
| **DroidDetect** (final) | **99.18** | **99.25** | **94.36** | **95.17** | **92.95** | **94.30** |
| DroidDetectSCL | see paper | see paper | see paper | see paper | see paper | see paper |
| DroidDetectCLS | 99.22 | 98.22 | 87.57 | 92.78 | 93.05 | 74.46 |

> Note: "RDS" = Research/DataScience. "Algo" = Algorithmic. "Gen" = General.

---

## 🎯 SOTA targets (goals for every climb method)

Target = beat the **strongest** published baseline for that task (not just beat the popular reference).
Paper columns anchor to the tables above.

| Benchmark / Task | Paper best | Source | **Our target** | Minimum to claim SOTA |
|:-----------------|:----------:|:-------|:--------------:|:---------------------:|
| CoDET Binary IID Macro-F1 | **98.65** | UniXcoder, Table 2 | **≥ 99.00** | > 98.65 |
| CoDET Binary per-lang (C++) | **98.24** | UniXcoder, Table 3 | — | > 98.24 |
| CoDET Binary per-lang (Python) | **98.60** | UniXcoder, Table 3 | — | > 98.60 |
| CoDET Binary per-lang (Java) | **99.02** | UniXcoder, Table 3 | — | > 99.02 |
| CoDET Binary per-src (CodeForces) | **96.54** | UniXcoder, Table 4 | — | > 96.54 |
| CoDET Binary per-src (LeetCode) | **97.87** | UniXcoder, Table 4 | — | > 97.87 |
| CoDET Binary per-src (GitHub) | **98.46** | UniXcoder, Table 4 | — | > 98.46 |
| **CoDET Author (6-class) Macro-F1** | **66.33** | UniXcoder, Table 7 | **≥ 70.00** | > 66.33 |
| CoDET OOD-Generator (unseen LLMs) F | **93.22** | UniXcoder, Table 8 | ≥ 94.00 | > 93.22 |
| CoDET OOD-Source (unseen domains) F | **58.22** | **CodeT5**, Table 9 | **≥ 58.50** | > 58.22 |
| CoDET OOD-Language avg F | **88.96** | UniXcoder, Table 12 | ≥ 85.00 (stretch) | > 88.96 (hard) |
| CoDET OOD-Language (C#) | **91.31** | UniXcoder, Table 10 | — | > 91.31 |
| CoDET OOD-Language (Golang) | **90.01** | UniXcoder, Table 10 | — | > 90.01 |
| CoDET OOD-Language (JavaScript) | **81.48** | UniXcoder, Table 10 | — | > 81.48 |
| CoDET OOD-Language (PHP) | **96.17** | UniXcoder, Table 10 | — | > 96.17 |
| CoDET Hybrid Authorship F | **39.36** | UniXcoder, Table 13 | ≥ 45.00 | > 39.36 (very hard) |
| CoDET Ternary (human/LLM/hybrid) F | **86.10** | UniXcoder, Table 14 | ≥ 87.00 | > 86.10 |
| **Droid 3-Class per-domain Avg W-F1** | **88.78** | DroidDetectCLS-Large, Table 3 | **≥ 90.00** | > 88.78 |
| Droid 2-Class per-domain Avg W-F1 | **97.00** | DroidDetectCLS-Large, Table 3 | ≥ 97.50 | > 97.00 |
| **Droid 3-Class per-language Avg W-F1** | **93.66** | DroidDetectCLS-Large, Table 4 | ≥ 94.50 | > 93.66 |
| Droid 2-Class per-language Avg W-F1 | **99.23** | DroidDetectCLS-Large, Table 4 | ≥ 99.30 | > 99.23 |
| Droid 3-Class (Research/DS) | **80.42** | DroidDetectCLS-Large, Table 3 | ≥ 82.00 | > 80.42 |
| Droid Adversarial Recall | **0.92** | DroidDetectCLS-L, Table 5 | ≥ 0.94 | > 0.92 |
| Droid 3-Class General (DroidDetect) | **95.17** | DroidDetect final, Table 6 | ≥ 95.50 | > 95.17 |

**Data-efficiency framing:** any target above, achieved with ~20% training data, becomes a paper-worthy headline. All paper baselines use 100% training data.

> **Our climb's primary metric mapping:** `Bin` in the top leaderboard maps to **CoDET Binary IID Macro-F1** (Table 2 row). `Auth` maps to **Table 7 Author Macro-F1**. `Src-gh` is our **held-out-GH subset of Table 4's GitHub column** — but we test the **author** head on it (6-class), so it is strictly harder than UniXcoder's Table 4 number (which is only binary). `T3` maps to **Table 3 3-Class Avg Weighted-F1**. Other climb columns (OOD-LANG-python LOO, OOD-GEN-qwen1.5 LOO, Droid T1, Droid T4) do **not** have a direct paper baseline — they are relative-to-other-methods comparisons.

---

## 📈 Method leaderboard

Fill in after each `exp_NN_*.py` run by pasting the `BEGIN_PAPER_TABLE` block.

> **See the [🏆 Climb Leaderboard](#-climb-leaderboard--sorted-by-codet-author-macro-f1-) at the top of this file** for the full 8-task ranked results. This section is reserved for per-method narrative (novelty, risks, lean-gate outcome) — kept below in the `Insights log` subsection.
>
> Retired duplicate summary table: the top leaderboard now covers every lean task (Bin / Auth / OOD-GEN-qw / OOD-LANG-py / OOD-SRC-gh / Droid T1 / T3 / T4) so there is no second table to sync.

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

### exp_01 GenealogyGraphCode (pending, lean mode)
- **Novelty:** replaces static family table (Exp18) with a LEARNABLE weighted graph over generator prototypes. EMA-updated prototypes + 1-layer GNN propagation + InfoNCE proto-contrast + graph-smoothness prior on the static tree.
- **Targets insight #2 + #14:** breaks the plateau around the Nxcode↔Qwen1.5 pair by making the genealogy itself a learned parameter rather than hand-crafted.
- **Success criteria (lean screening):** CoDET Author > 70.55 OR Qwen1.5 per-class F1 > 0.47 OR OOD-GEN held-out=qwen1.5 > 0.51.
- **Risk:** prototypes may collapse if EMA too strong; graph-smooth term is a warm-start prior that tapers off.
- **Result after run:** _(paste BEGIN_PAPER_TABLE block summary)_

### exp_02 GHSourceInvariantCode (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** first method to attack the GH-source OOD bottleneck (insight #16). Couples SOURCE-IRM (per-source IRMv1 penalty, 3 envs cf/gh/lc) with a gradient-reversed STYLE-only adversary predicting source. HierTree preserved for genealogy signal.
- **Targets insight #3 + #16:** surface-style shortcut on CF/LC templates is the root cause of GH-OOD catastrophic failure. Source-invariance on style subspace (not on content) should close gap without hurting author signal.
- **Success criteria (lean screening):** OOD-SRC held-out=gh > 0.30 AND CoDET Author IID >= 70.3 (no regression).
- **Risk:** IRM penalty may explode without annealing (Exp06 lesson) — mitigated by `irm_warmup_epochs=1`. Requires dataset to expose per-sample `source` field in outputs dict.
- **Result after run:** Full suite `2026-04-18 19:53:52` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9899** (best val 0.9894); IID author **0.7020** (val 0.7118); OOD source GH **0.3044** (val 0.9928); OOD language python **0.5070**; OOD generator qwen1.5 **0.4976** (per-class table shows class 0 support 0 — headline macro is still a degenerate-case signal). **Droid** — T1/T3/T4 weighted-F1 test **0.9703 / 0.8805 / 0.8738** (primary vs best val **0.9714 / 0.8420 / 0.8392**). OOD-SRC-gh clears the 0.30 bar; author lands at 70.20 (just under the 70.30 regression guard). Droid T3/T4 sit slightly below the DroidDetectCLS-Large paper line (0.8878) but remain strong ID.

### exp_03 TokenStatRAGCode (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** first method to use retrieval DURING TRAINING (not just test-time). In-batch kNN over token-statistics features (label-independent surface features → cannot trivially memorise). Same-label neighbours pull, different-label neighbours hinge-push.
- **Targets insight #12 + #14:** combines cheap token-stat booster (Droid) with training-time retrieval signal (OOD + Author). Prior work (Exp17 RAGDetect 70.46) used embedding-space retrieval = circular.
- **Success criteria (lean screening):** CoDET Author > 70.60 AND Droid T3 > 0.89 AND OOD-SRC-gh > 0.32.
- **Risk:** token-stat features need to be surfaced by the backbone (`outputs["tokenstat"]` or `spectral_features`); fallback uses last-16-dim embedding slice.
- **Result after run:** Full suite `2026-04-18 20:24:37` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9905** (best val 0.9902); IID author **0.6990** (val 0.7043); OOD source GH **0.3019** (val 0.9926); OOD language python **0.5260**; OOD generator qwen1.5 **0.4962** (per-class: class 0 support 0 — same degenerate macro caveat as other LOO gen runs). **Droid** — T1/T3/T4 weighted-F1 test **0.9707 / 0.8894 / 0.8740** (macro-F1 **0.9706 / 0.8493 / 0.8401**; best val primary **0.9709 / 0.8497 / 0.8391**). **Lean gates:** Droid T3 **0.8894** is just under the 0.89 bar; OOD-SRC-gh **0.3019** under 0.32; author **69.90** under 70.60 — screening criteria not all met, but Droid T3 still edges past DroidDetectCLS-Large paper (0.8878) and binary/ID Droid stay strong.

### exp_04 PoincareGenealogy (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** Poincaré-ball embeddings + hyperbolic distance for the CoDET generator tree (Euclidean → exp map, learnable curvature, hierarchy/radial regularizers) — geometry aimed at low tree distortion vs Euclidean HierTree-style losses.
- **Targets (from exp file):** CoDET author **> 70.7**, Qwen1.5 per-class F1 **> 0.48**, Droid T3 stable **~0.89**; OOD GH remains the stress test.
- **Risk:** hyperbolic optimization can be brittle; family depth targets may trade off against author discrimination.
- **Result after run:** Full suite `2026-04-18 19:55:51` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9901** (best val 0.9904); IID author **0.6958** (val 0.7111); OOD source GH **0.2647** (val 0.9928); OOD language python **0.4761**; OOD generator qwen1.5 **0.4965** (class 0 support 0 in per-class table — same LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9697 / 0.8976 / 0.8799** (macro-F1 **0.9696 / 0.8592 / 0.8465**; best val primary **0.9704 / 0.8623 / 0.8430**). **Takeaway:** Droid T3 clears DroidDetectCLS-Large (**0.8976** vs 0.8878) — best T3 in the climb board so far; CoDET author sits below the 70.7 stretch goal and OOD-GH (**0.2647**) is weaker than exp_02/exp_03, so hyperbolic genealogy helps Droid ID more than GH-OOD under this lean recipe.

### exp_05 SinkhornOTCode (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** batch-level Sinkhorn–Knopp optimal-transport targets (balanced class mass) + KL to softmax logits vs standard CE/focal — aims to fix minority-class starvation on imbalanced 6-way author.
- **Targets (from exp file):** CoDET author **> 70.6** (balanced OT head); stackable with HierTree auxiliary.
- **Risk:** OT iterations each batch add compute; equal column-mass may fight natural human majority if ε too small.
- **Result after run:** Full suite `2026-04-18 20:10:15` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9905** (best val 0.9897); IID author **0.6840** (val 0.6924); OOD source GH **0.2896** (val 0.9927); OOD language python **0.5754**; OOD generator qwen1.5 **0.4967** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9701 / 0.8803 / 0.8752** (macro-F1 **0.9700 / 0.8393 / 0.8415**; best val primary **0.9711 / 0.8421 / 0.8410**). **Takeaway:** beats UniXcoder on author (**+2.07** pt) but **below** the file’s 70.6 OT target and behind stronger climb authors (exp_02/03/04). **OOD-LANG python** (**0.5754**) is a bright spot vs recent climbs; OOD-GH **0.2896** sits under the 0.30 bar; Droid T3/T4 trail paper SOTA and exp_04’s T3 peak.

### exp_06 FlowCodeDet (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** class-conditioned flow-matching auxiliary head (`CondVelocityMLP`) — per-class velocity field regularizes embeddings (stronger than pairwise SupCon per cited FM work); linear noise interpolant + MSE on predicted velocity.
- **Targets (from exp file):** CoDET author **> 70.6**, Droid T4 **> 0.85**, OOD-GEN qwen1.5 **> 0.51** (degenerate macro caveat applies).
- **Risk:** extra FM loss weight vs CE; sampling `t` can dominate early training if `lambda_fm` too high.
- **Result after run:** Full suite `2026-04-18 20:04:43` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9902** (best val 0.9897); IID author **0.7090** (val 0.7158); OOD source GH **0.3336** (val 0.9918); OOD language python **0.6450**; OOD generator qwen1.5 **0.4967** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9701 / 0.8934 / 0.8830** (macro-F1 **0.9701 / 0.8541 / 0.8479**; best val primary **0.9706 / 0.8542 / 0.8488**). **Takeaway:** **strongest CoDET author in the climb board so far (70.90)** and **best OOD-GH (0.3336)** among completed lean runs; clears the file’s **70.6** author bar and Droid T4 **> 0.85**. Droid T3 **0.8934** beats DroidDetectCLS-Large (0.8878) but sits slightly under exp_04’s peak T3; OOD-LANG python **0.645** is best-in-climb. UniXcoder gap on author: **+4.57** pt.

### exp_07 SAMFlatCode (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** Sharpness-Aware–style flatness via **embedding-space** adversarial perturbation (FGSM on features + auxiliary head loss) as a cheap proxy for SAM — targets OOD transfer without DANN-style domain confusion.
- **Targets (from exp file):** OOD-SRC-gh **> 0.32**, CoDET author **~70.5**, Droid T4 **> 0.85**.
- **Risk:** feature-space SAM can over-smooth discriminative directions if `lambda_sam` too large.
- **Result after run:** Full suite `2026-04-18 20:18:54` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9905** (best val 0.9902); IID author **0.7022** (val 0.7124); OOD source GH **0.3141** (val 0.9927); OOD language python **0.5413**; OOD generator qwen1.5 **0.4974** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9692 / 0.8873 / 0.8751** (macro-F1 **0.9692 / 0.8468 / 0.8411**; best val primary **0.9702 / 0.8477 / 0.8411**). **Takeaway:** author **70.22** and Droid T4 **0.875** meet “stable IID / adv” expectations; OOD-GH **0.3141** lands **just under** the **0.32** screening bar from the exp file. Droid T3 **0.8873** sits slightly below DroidDetectCLS-Large (0.8878) and under exp_04/06 peaks; overall a solid mid-pack run vs **FlowCodeDet** on GH-OOD and author. UniXcoder author gap: **+3.89** pt.

### exp_08 POEMPolarizedCode (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** POEM-style **orthogonal polarization** — split embedding into invariant vs source-specific subspaces (`L_ortho` on projectors), put source prediction on `z_spec` and entropy-regularize source on `z_inv` (no GRL / DANN).
- **Targets (from exp file):** OOD-SRC-gh **> 0.32**, CoDET author **~70.4**, Droid T3 **~0.88**.
- **Risk:** rank split can under-feed the author head if `z_inv` is too narrow.
- **Result after run:** Full suite `2026-04-18 20:16:47` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9906** (best val 0.9902); IID author **0.6968** (val 0.7083); OOD source GH **0.3333** (val 0.9928); OOD language python **0.5411**; OOD generator qwen1.5 **0.4972** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9705 / 0.8857 / 0.8748** (macro-F1 **0.9704 / 0.8450 / 0.8409**; best val primary **0.9711 / 0.8466 / 0.8412**). **Takeaway:** OOD-GH **0.3333** clears the exp-file **0.32** bar and matches the **exp_06** GH cluster (0.3336) within noise; author **69.68** tracks the ~70.4 expectation but trails **FlowCodeDet**; Droid T3 **0.8857** sits just under DroidDetectCLS-Large (0.8878). UniXcoder author gap: **+3.35** pt.

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

### 11. Droid is stable across methods — don't use it to differentiate (NEW)

From **14 DM methods** all running Droid T3/T4: scores cluster in **0.84–0.90 W-F1**, gap between best (exp09 TokenStat 0.8941) and worst (exp07 DomainMix 0.7930) is mostly explained by backbone strength, not method innovation. **The claim to beat is DroidDetectCLS-Large (0.8878)** — only exp09 TokenStat and exp18 HierTreeCode come close on T3 (0.8941/0.8917). Use Droid T3 as a sanity check (did we regress?), not as a discriminator between ideas.

### 12. Token statistics are the best cheap Droid booster (NEW)

`exp09 TokenStat` (entropy, burstiness, TTR, Yule-K token distribution features) consistently gives **+0.003–0.005 Droid T3 W-F1** over pure spectral backbone. It's also cheap (no extra model, just numpy stats per sample). **Recommend: include token stats in every climb method as a free Droid booster.**

### 13. AICD T1 OOD collapse is structural — skip it entirely in climb (NEW)

**23 methods tested across Exp_DM** (DomainMix, IRM, OSCP, VILW, SupCon, TripletLoss, etc.) — all show val 0.99 → test 0.25-0.31 on AICD T1. The val-test gap is a dataset property (AICD train/test have different domain distributions by design). **No method-level fix works.** Climb excludes AICD deliberately. Report it as "open challenge" if reviewers ask.

### 14. HierTree + SupCon + kNN is the current best cocktail (NEW)

From Exp_CodeDet: **HierTree loss alone = 70.55** (Exp18). Adding **kNN test-time blend = 70.46** (Exp17, slight different impl). Adding **SupCon = 70.33** (Exp32 HyperCode). The pattern that works:
1. HierTree family loss (Qwen/Nxcode cluster) — non-negotiable, +1.5 F1 vs base
2. Token stats features — free +0.3-0.5 Droid
3. kNN blend at test time — +0.1-0.3 author F1, zero training cost

Methods that don't add above this baseline (KAN head, Hypernetwork, IB compression, DANN) all converge to 70.0-70.3 range.

### 15. Lean screening protocol — 3 ideas per H100 session (NEW)

Use `run_mode="lean"` (8 runs, ~53 min) to screen. Promotion criteria:
- **CoDET Author F1 > 70.55** (must beat current SOTA = Exp18) OR
- **Droid T3 W-F1 > 0.8941** (must beat exp09 TokenStat) OR  
- **OOD-SRC gh > 0.30** (any improvement on hardest subgroup is paper-worthy)

If none of the 3 OOD representative runs (gh/python/qwen1.5) shows improvement, abort and try next idea. Only promote to `run_mode="full"` (16 runs, ~1h41m) when at least one criterion is met.

### 16. GitHub source is the biggest untapped opportunity (NEW)

From Exp_Climb insights + Exp_CodeDet OOD data:
- **OOD-SRC held-out=gh Author macro F1 ≈ 0.2834** — catastrophic across all 14+ methods
- Human recall on GH-held-out = **5.71%** (model never predicts human for GH code)
- Root cause: CF+LC = competitive-programming templates (narrow style), GH = real-world diverse code. Model trained on cf+lc memorizes templates, fails completely on GH diversity.
- **Any method that breaks 0.40 on OOD-SRC-gh is a NeurIPS-worthy result.** No method has come close yet.
- Target for next architectural direction: explicitly train on source-diverse batches (GH-aware sampling) or use source as an environment variable in IRM-style training.

---

## Not in this folder

- **AICD-Bench** — excluded deliberately. 23+ methods in `Exp_DM/` all fail to close val-test gap (0.99 → 0.25 on T1). Paper treats AICD as "open challenge / negative result," not a main claim.
- **Ablations** — belong in `Exp_DM/` / `Exp_CodeDet/` where you freely tune. Climb files are for final dual-bench SOTA runs only.
- **Wrapper / variant methods** (exp28/29/30) — if they graduate, they become their own `exp_NN_*.py`.
