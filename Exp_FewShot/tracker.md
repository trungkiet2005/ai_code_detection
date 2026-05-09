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

| Method | Exp | K=8 | K=16 | K=32 | K=64 | K=128 | Notes |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:--|
| FS-Baseline-CE | exp_fs_00 | — | — | **0.1836** (+0.04) | — | — | floor; only 12 train steps at K=32 |
| FS-NTKAlign | exp_fs_01 | — | — | — | — | — | ⏳ pending |

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
