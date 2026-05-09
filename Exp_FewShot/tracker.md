# Exp_FewShot Tracker — K-shot AI-Code Detection on Kaggle T4

> **Pivot rationale (2026-05-08):** 20% data SOTA already done; we believe few-shot
> (K=8..128 examples per class) is the higher-impact claim for EMNLP Main 2026.
> 17 days to deadline. Phase B Day 5 = decision gate; if K=128 < 60 Author F1 we
> bail back to the 20% story.
>
> **Hardware:** Kaggle T4 (16 GB VRAM) — bs=16, fp16, seq=384, 1 epoch. H100/A100
> auto-upgrade if available, never required.
>
> **Protocol:**
> - K-shot stratified sampling on TRAIN per (k_shot, fs_seed). Sampler in
>   `_fewshot_sampler.py`. Reproducibility: every run logs `fs_seed`.
> - Mini-val held out from the original VAL pool (n_per_class=64) for early stop.
> - TEST = full paper-comparable split (no subsampling).
> - 1 epoch over K * n_classes samples; eval every floor(steps/8); early-stop on
>   plateau (patience=5 evals).
> - Always report **val-test gap** alongside test metrics (CLAUDE.md mandate).
> - Primary metric: CoDET-M4 Author IID Macro-F1 (matches paper Table 7).

---

## 🎯 Targets to beat

| Baseline | Author F1 | Reference |
|:--|:-:|:--|
| **UniXcoder (full data, 100%)** | **66.33** | Orel et al. ACL'25, Table 7 |
| CodeBERT (full data) | 64.80 | same |
| CodeT5 (full data) | 62.45 | same |
| **Our 20%-data SOTA (Exp_13 NTKAlign)** | **71.03** | Exp_Climb tracker, lean run |

**Decision gate (Day 5):**
- 🟢 **K=64 ≥ 65 Author F1** → continue Phase C (top-3 methods × all K × Droid)
- 🟡 **K=128 ≥ 65 but K=64 < 65** → "low-shot scaling" framing (still publishable)
- 🔴 **K=128 < 60** → abort pivot, fall back to the 20% data story

---

## 📊 K-Shot Leaderboard — CoDET-M4 6-class Author Macro-F1

> Sorted by `K=32` Author F1 (the practical sweet spot for the EMNLP claim).
> Each row = one method. Columns = test Macro-F1 at K ∈ {8, 16, 32, 64, 128}.

### K-shot regime (per-class examples)

| Method | Exp | K=8 | K=16 | K=32 | K=64 | K=128 | Notes |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:--|
| FS-Baseline-CE      | exp_fs_00 | — | — | **0.1836** (+0.04) | — | — | floor; CE only |
| FS-NTKAlign         | exp_fs_01 | — | — | **0.1222** (+0.08) | — | — | ⚠️ **−6.14 pt vs baseline** at K=32 |
| FS-SupCon           | exp_fs_02 | — | — | — | — | — | ⏳ per-anchor softmax (Khosla'20) |
| FS-Frozen           | exp_fs_03 | — | — | — | — | — | ⏳ encoder frozen, head only |
| FS-NTKAlign+Frozen  | exp_fs_04 | — | — | — | — | — | ⏳ NTK + frozen encoder |
| FS-SupCon+Frozen    | exp_fs_05 | — | — | — | — | — | ⏳ SupCon + frozen encoder |

### %-fraction regime (phase-transition curve)

| Method | Exp | 1% (~5K) | 5% (~25K) | 10% (~50K) | 20% (~100K) | 50% (~250K) |
|:--|:--|:-:|:-:|:-:|:-:|:-:|
| FS-Baseline-CE     | exp_fs_00 | — | — | — | — | — |
| FS-NTKAlign        | exp_fs_01 | — | — | — | (Exp_13 lean: **0.71**) | — |
| FS-SupCon          | exp_fs_02 | — | — | — | — | — |
| FS-Frozen          | exp_fs_03 | — | — | — | — | — |
| FS-NTKAlign+Frozen | exp_fs_04 | — | — | — | — | — |
| FS-SupCon+Frozen   | exp_fs_05 | — | — | — | — | — |

---

## 💾 JSON output — convention for downloading from Kaggle

Every `exp_fs_*.py` writes a JSON to `/kaggle/working/results/<exp_id>_<label>_seed<S>.json`
(falls back to `./results/...` locally). The label encodes the regime:

```
exp_fs_01_K128_seed42.json     ← K-shot regime, K=128
exp_fs_03_frac0.05_seed42.json ← %-fraction regime, fraction=0.05
```

After the Kaggle session ends, download the entire `/kaggle/working/results/`
folder (Kaggle: "Output" tab → ZIP). Locally:

```bash
# Aggregate everything into a leaderboard
python Exp_FewShot/aggregate_fs_results.py results
# Also dump CSV
python Exp_FewShot/aggregate_fs_results.py results --csv
```

This prints a table sorted by `test_macro_f1` and writes `results/summary.json`
+ optional `summary.csv`. **Append-only** — re-running with a different seed
adds a row, never overwrites.

> The 20% cell for NTKAlign references **Exp_13 NTKAlignCode 0.7103** in
> `Exp_Climb/tracker.md` — that uses the FULL Exp_Climb backbone (HierTree +
> spectral + neural heads), not the lean ModernBERT-only stack here. Direct
> comparison is informative but **not** apples-to-apples; we will run the
> lean stack at 20% when we have results at 5% / 1% to characterise the
> simple-recipe curve.

> **Cell format:** test Macro-F1 with val-test gap in parens, e.g. `0.43 (-0.05)`.
> Negative gap = test ≥ val (good); large positive gap = overfitting on K-shot train.

**🔍 First-run reading (2026-05-09, exp_fs_00 K=32):** Baseline CE on K=32 lands at
**0.1836** test Macro-F1 (val 0.2253, gap +0.04). With only 12 training steps
(192 samples / bs=16 / 1 epoch) the model barely converges — and that is the
point of the floor. Per-class F1 is very uneven: human (class 0) **0.50**,
codellama 0.25, qwen 0.22, but llama3.1 (class 3) collapses to **0.002**. The
val-test gap is tight (+0.04), so we are not overfitting; we are simply
under-fitting. The headroom for the NTK-aligned method is therefore the full
range from 0.18 toward our 20%-data ceiling 0.71.

---

## 📊 Run history

> Append-only. Never rewrite a row. Each run paste the BEGIN_FS_TABLE block here
> after Kaggle log finishes.

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

**Author vocab (5 generators):** `codellama, gpt, llama3.1, nxcode, qwen1.5` →
class indices `1..5`; class 0 = human.

**Pipeline timeline (T4):**
- Dataset load + label vocab: 96s
- K-shot sample + mini-val: ~2s
- Tokenizer + model load: 12s
- Train 12 steps: 35s
- Eval (val 384 + test 47,744 samples): 458s **bottleneck**
- Total: 484s ≈ 8 min

**Reading:** Loss curve `1.86 → val 0.0783 (step 5) → val 0.2253 (step 10) →
val 0.1439 (step 12)` — model peaks around step 10 then starts to overfit
slightly. Best val checkpoint restored. Test 0.1836 is below val 0.2253 by
4.17 pt because the test set is much larger and contains the OOD-source
distribution shift our 20%-data climb already documented (val 5K samples are
mostly CF/LC; full test 47.7K spans all three sources).

**Per-class collapse (class 3 = llama3.1, F1 = 0.002):** with K=32 examples
of llama3.1 in training, the model never separates it from sibling
codellama. This is exactly the Qwen↔Nxcode → llama3.1↔codellama family
confusion we predicted in the paper. NTKAlign should help here — its
target-kernel pulls same-class projections together explicitly.

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

**🚨 Diagnosis — NTKAlign is WORSE than baseline at K=32 (−6.14 pt test):**

| Slice | Baseline K=32 | NTKAlign K=32 | Δ |
|:--|:-:|:-:|:-:|
| Test Macro-F1 | 0.1836 | **0.1222** | **−0.0614** |
| Val Macro-F1 | 0.2253 | 0.2020 | −0.0233 |
| Val-test gap | +0.04 | **+0.08** | gap doubled |
| Class 0 (human) F1 | **0.503** | 0.214 | −0.289 collapse |
| Class 1 (codellama) F1 | 0.246 | 0.222 | −0.024 |
| Class 3 (llama3.1) F1 | 0.002 | 0.002 | unchanged (still collapsed) |

The collapse is concentrated on the **human class** (0.50 → 0.21). NTKAlign's
target-kernel objective pushes same-class projections together — but with
batch size B=16 and 6 classes, each batch has only ~2.7 same-class pairs on
average, so the kernel target Y is mostly off-diagonal zeros. The loss
becomes a *separation* objective dominated by inter-class repulsion, which
drags the human cluster (the only class with a strong CE signal) toward
the centroid mean.

**Hypothesised fixes (in order of expected impact):**
1. **Class-balanced batch sampler.** Currently the sampler shuffles K·N=192 samples randomly into bs=16 batches → uneven class composition. Force at least $\lceil B/N \rceil = 3$ examples per class per batch. The NTK kernel target Y will then have $\geq B \cdot 3$ on-diagonal pairs instead of $\sim 6$ on average.
2. **Lower lambda_ntk.** 0.4 may be too aggressive at K-shot scale. Try 0.1 or 0.05 first; sweep upward only if K=128 baseline is also weak.
3. **Larger batch (bs=32 fp16).** Doubles same-class pair count for free. T4 should fit ModernBERT-base at seq=384 bs=32 fp16. Worth trying.
4. **Warmup epochs without NTK.** Run 3-5 epochs with $\lambda_{\text{NTK}} = 0$, then activate. Prevents early collapse before CE has settled.

**Implication for paper pivot.** This is K=32, the small end. **K=128
will tell us whether NTKAlign starts to help once the kernel target
becomes informative.** Decision gate stays at K=128: if NTKAlign K=128
< Baseline K=128, we revert to the 20% data story (paper draft v1
already builds on the existing Exp_13 NTKAlign 71.03 result, which uses
**bs=64 H100**, not K-shot bs=16). The Exp_Climb claim is unchanged.

_(no run yet)_

---

## ⏳ Ship order

1. **Day 3 (today)** — CPU smoke-test `exp_fs_00` + `exp_fs_01` with K=8 on a
   tiny subset of the val split (sanity check that the pipeline runs end-to-end).
2. **Day 4** — Kaggle T4 K-sweep on `exp_fs_01` (NTKAlign) at K ∈ {8, 16, 32, 64, 128}.
   ~5 runs × 10–30 min each = 1–3h Kaggle session.
3. **Day 4** — Same K-sweep on `exp_fs_00` (baseline) for direct CE-vs-NTK Δ.
4. **Day 5** — DECISION GATE based on K=64 / K=128 numbers.
5. **Day 6+** — (if go) port FlowCodeDet (Exp_06) and Poincare (Exp_04) to few-shot.

---

## 🛠️ Engineering rules

- **Kaggle T4 first.** Every script must auto-detect and adapt. H100 is bonus only.
- **Standalone per suite.** No imports from `Exp_Climb/`, `Exp_DM/`, `Exp_CodeDet/`.
- **No tree-sitter / AST.** Drop entirely — install fragility on Kaggle isn't worth
  the marginal feature gain at K=8..128.
- **No spectral / FFT branch.** Same reason; few-shot has too little training signal
  to benefit from auxiliary heads. Add back later if K-sweep numbers warrant.
- **fp16 only.** T4 doesn't support bf16. Cast logits to fp32 in loss if any
  numerical instability.
- **One method per file.** `exp_fs_NN_<method>.py`. Method-specific hyperparams
  exposed via env vars (FS_K_SHOT, FS_SEED, FS_LAMBDA_NTK, ...) for sweep scripting.

---

## 🧪 Smoke test (CPU)

```bash
# K=8, mini-val tiny, full test → ~5 min CPU sanity
FS_K_SHOT=8 python Exp_FewShot/exp_fs_00_baseline.py
FS_K_SHOT=8 FS_LAMBDA_NTK=0.4 python Exp_FewShot/exp_fs_01_ntkalign.py
```

## 🚀 Kaggle T4 run — TWO equivalent ways

### Option A: full repo clone (recommended for sweeps)

```python
# Cell 1
!git clone --depth 1 https://github.com/trungkiet2005/ai_code_detection.git
%cd /kaggle/working/ai_code_detection

# Cell 2: K-sweep on NTKAlign + Baseline
import os, pathlib
pathlib.Path("logs").mkdir(exist_ok=True)
for k in [8, 16, 32, 64, 128]:
    os.environ["FS_K_SHOT"] = str(k)
    !python Exp_FewShot/exp_fs_01_ntkalign.py 2>&1 | tee logs/exp_fs_01_K{k}.log
    !python Exp_FewShot/exp_fs_00_baseline.py 2>&1 | tee logs/exp_fs_00_K{k}.log
```

### Option B: standalone single-file (paste into a cell)

The exp file auto-clones the repo on first run if support modules aren't
present, then imports from the cloned `Exp_FewShot/`. Useful when you only
have the script and don't want to manage `cd`.

```python
# Cell: just upload exp_fs_01_ntkalign.py to /kaggle/working and run
import os
os.environ["FS_K_SHOT"] = "32"
!python /kaggle/working/exp_fs_01_ntkalign.py
# First run prints: [fs-bootstrap] cloning ... -> /kaggle/working/ai_code_detection
# Subsequent runs reuse the clone.
```

Each run emits a `BEGIN_FS_TABLE ... END_FS_TABLE` block. Paste those blocks back
into this tracker under the "Run history" section.
