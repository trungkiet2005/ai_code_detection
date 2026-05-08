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
| FS-Baseline-CE | exp_fs_00 | — | — | — | — | — | ⏳ pending |
| FS-NTKAlign | exp_fs_01 | — | — | — | — | — | ⏳ pending |

> **Cell format:** test Macro-F1 with val-test gap in parens, e.g. `0.43 (-0.05)`.
> Negative gap = test ≥ val (good); large positive gap = overfitting on K-shot train.

---

## 📊 Run history

> Append-only. Never rewrite a row. Each run paste the BEGIN_FS_TABLE block here
> after Kaggle log finishes.

### exp_fs_00 — FS-Baseline-CE (CE only, ModernBERT-base)

_(no run yet)_

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

## 🚀 Kaggle T4 run

```python
# Cell 1: clone + cd
!git clone https://github.com/<your-fork>/ai_code_detection.git
%cd /kaggle/working/ai_code_detection

# Cell 2: K-sweep on NTKAlign (CoDET Author)
import os
for k in [8, 16, 32, 64, 128]:
    os.environ["FS_K_SHOT"] = str(k)
    !python Exp_FewShot/exp_fs_01_ntkalign.py 2>&1 | tee logs/exp_fs_01_K{k}.log
```

Each run emits a `BEGIN_FS_TABLE ... END_FS_TABLE` block. Paste those blocks back
into this tracker under the "Run history" section.
