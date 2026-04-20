# Exp_ZeroShot Tracker — No-Training Detectors for Droid / CoDET

> **Scope:** ZERO training. Each `exp_zs_NN_*.py` computes a per-sample score, calibrates a decision threshold τ on the dev split to pin human recall ≥ 0.95, reports Macro-F1 + Weighted-F1 + Human/AI/Adversarial recall on the FULL test split.
>
> **Paper angle (separate submission, or appendix of the Exp_Climb oral):** "Our fusion-embedding + spectral backbone, used *zero-shot* without any Droid training, matches Fast-DetectGPT on Droid T3 at +X pt. The full-trained version (Exp_Climb) beats DroidDetectCLS-Large by Y pt. Same backbone, two regimes, one paper."
>
> **Scope discipline:** Exp_ZeroShot is **separate from Exp_Climb**. Exp_Climb = training-only oral; Exp_ZeroShot = no-training contrast. Do NOT mix numbers across the two directories.

---

## 🎯 Baselines to beat (Droid paper Tables 3 / 4 / 5 zero-shot rows)

| Baseline | T1 (2-cls) | T3 (3-cls) | T1 per-lang Avg | T3 per-lang Avg | Human R | Adv R | Notes |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:--|
| **Fast-DetectGPT** | **67.85** | **64.54** | **76.58** | **70.98** | 0.84 | 0.48 | strongest ZS baseline in the paper |
| GPTZero | 56.91 | 49.10 | 54.81 | 51.48 | 0.53 | 0.10 | weakest human recall |
| M4 | 50.92 | 55.27 | 52.81 | 57.92 | **0.40** | **0.73** | ⚠️ adv-chaser: high adv R, crashed human R |
| CoDet-M4 | 54.49 | 47.80 | 47.97 | 41.28 | **0.38** | 0.63 | ⚠️ same failure mode as M4 |
| GPTSniffer | 41.07 | 38.95 | 52.40 | 49.96 | 0.65 | 0.49 | — |

**Oral claim surface for Exp_ZeroShot:**
1. **Headline**: beat Fast-DetectGPT 64.54 on Droid T3 (3-cls) with ≤ 5 min of inference per test run.
2. **Adversarial contrast**: hold human R ≥ 0.95 *while* matching Fast-DetectGPT's adv R of 0.48. That pins the M4 / CoDet-M4 failure mode ("adv-chaser") as avoidable.
3. **Per-domain breakdown**: zero-shot detector that transfers across `general` / `algorithmic` / `research_ds` equally (max-Δ across domains < 5 pt) is a separate paragraph claim.

---

## 🏆 ZS Leaderboard — dual-benchmark (Droid T3 Macro-F1 + CoDET binary Macro-F1)

Each ZS file now runs on BOTH benchmarks via `run_zs_oral` and emits a combined `BEGIN_ZS_ORAL_TABLE` block. Oral-level pass gate requires **all three** of:
1. Beat Fast-DetectGPT Droid T3 (**64.54**).
2. Hold Human-Recall ≥ 0.95 on both benches (paper Table 5 failure mode).
3. Cross-benchmark stability: |Droid T3 − CoDET binary| < 10 pt.

| Rank | Exp | Method | **Droid T3 (3-cls)** | **CoDET binary** | Human R (D/C) | Adv R (D only) | Wall | Status |
|:-:|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| — | exp_zs_00 | **RandomScorer** (floor A) | — | — | — / — | — | — | ⏳ pending |
| — | exp_zs_00 | **LengthOnlyScorer** (floor B) | — | — | — / — | — | — | ⏳ pending |
| — | exp_zs_03 | **Ghostbuster-Code** (token-stat committee) | — | — | — / — | — | — | ⏳ pending |
| — | exp_zs_04 | **SpectralSignature** (ModernBERT PC-1) | — | — | — / — | — | — | ⏳ pending |
| — | exp_zs_01 | **BinocularsLogRatio** (ModernBERT surrogate) | — | — | — / — | — | — | ⏳ pending |
| — | exp_zs_02 | **Fast-DetectGPT** (CodeBERT curvature) | — | — | — / — | — | — | ⏳ pending |
| **REF** | Paper | **Fast-DetectGPT** (Droid paper Table 3/5) | **64.54** | — | 0.84 / — | 0.48 | — | reference |
| REF | Paper | M4 (ZS) | 55.27 | — | 0.40 ⚠️ / — | 0.73 | — | reference |
| REF | Paper | GPTZero | 49.10 | — | 0.53 / — | 0.10 | — | reference |
| REF | Paper | CoDet-M4 (ZS) | 47.80 | — | 0.38 ⚠️ / — | 0.63 | — | reference |
| REF | Paper | GPTSniffer (ZS) | 38.95 | — | 0.65 / — | 0.49 | — | reference |

⚠️ M4 + CoDet-M4 have chased adversarial recall at the cost of human recall — the failure mode the paper explicitly flags in §4.5.4. Our oral claim avoids this by pinning Human R ≥ 0.95 in the τ-calibration step.

No direct paper baseline for CoDET binary zero-shot (CoDET-M4 paper's only ZS number is Fast-DetectGPT 62.03 in Table 2, reported as "baseline" — this cell is already in our CoDET mirror leaderboard).

---

## 🧭 ZS Methods — theory hooks

| Exp | Method | Theorem / theoretical hook | Signal | Cost |
|:--|:--|:--|:--|:-:|
| exp_zs_01 | **BinocularsLogRatio** | Hans et al. ICML'24 (arXiv:2401.12070) — log P_obs(x) - log P_perf(x) is Neyman-Pearson optimal for two close LMs | in-batch self-similarity of ModernBERT embeddings vs classifier entropy (surrogate for two-LM log-ratio) | 1 forward pass on ModernBERT-base |
| exp_zs_02 | **Fast-DetectGPT** | Bao et al. ICLR'24 — local curvature of conditional log-prob; higher curvature ⇒ on-manifold ⇒ likely LM-generated | masked-LM log-prob from CodeBERT as proxy; curvature approximated by variance under 15% mask sampling | 2 forward passes on CodeBERT |
| exp_zs_03 | **Ghostbuster-Code** | Verma et al. EMNLP'23 — committee of cheap handcrafted features (entropy / burstiness / TTR / Yule-K); human has higher burstiness, lower entropy ratio | feature vector + logistic-regression trained on dev (lean-ZS since only the LR head sees labels) | ~0 forward passes, pure numpy |
| exp_zs_04 | **SpectralSignature** | Our own — Exp_Climb's spectral encoder already learned code-frequency features at pretraining; project to a 1-D "human-likeness" axis via PCA on the first ModernBERT layer | 1 forward pass on ModernBERT, then 1-D projection | 1 forward pass + PCA |

All methods feed **scalar scores** (higher = more AI-like) into the shared `_zs_runner.run_zs_oral` → threshold calibration on dev (Droid + CoDET independently) → test metrics + per-dim breakdown + combined `BEGIN_ZS_ORAL_TABLE` block with oral-claim check.

**Floor baselines (exp_zs_00):** `RandomScorer` (U(0,1) noise, macro-F1 ≈ 0.50) + `LengthOnlyScorer` (log-char-length, expected 0.55–0.60). Any proposed ZS detector that doesn't clear both floors AND Fast-DetectGPT 64.54 on Droid T3 is *not* an oral-level contribution.

---

## 🖥️ Kaggle H100 80GB hardware profile (auto-applied by `apply_hardware_profile`)

Triggered when `torch.cuda.get_device_properties(0).total_memory >= 70 GB`:

| Setting | H100 80GB | A100 40GB / V100 | T4 / P100 | CPU |
|:--|:-:|:-:|:-:|:-:|
| `batch_size` | **128** | 64 | 32 | 8 |
| `precision` | bf16 | bf16 / fp16 | fp16 | fp32 |
| `num_workers` | **8** | 4 | 2 | 0 |
| `prefetch_factor` | **4** | 2 | 2 | 2 |
| `pin_memory` | **True** | True | False | False |
| `fast_detect_n_samples` (Fast-DetectGPT MC mask) | **10** | 5 | 3 | 3 |
| `binoculars_block_size` (reference pool size) | **1024** | 512 | 256 | 256 |

**VRAM budget (80 GB, seq 512 · batch 128):**
- ModernBERT-base + bf16 activations: ~6-7 GB
- CodeBERT-base MLM (Fast-DetectGPT only): +3 GB
- Binoculars (N × R cosine chunks, R=1024): < 1 GB
- Peak usage across all 5 ZS exps: **~12 GB / 80 GB**. Plenty of headroom for a second model alongside (e.g. Exp_Climb in same Kaggle session).

**Wall-time budget on H100 (both benches per exp):**

| Exp | Droid T3 | CoDET binary | **Both** |
|:--|:-:|:-:|:-:|
| exp_zs_00 (random + length) | < 5 s | < 5 s | **~10 s** |
| exp_zs_03 (Ghostbuster, pure numpy) | ~60 s | ~45 s | **~2 min** |
| exp_zs_04 (SpectralSignature, 1 MBERT fwd + PCA) | ~3 min | ~2 min | **~5 min** |
| exp_zs_01 (Binoculars, 1 MBERT fwd + O(N·R)) | ~5 min | ~3 min | **~8 min** |
| exp_zs_02 (Fast-DetectGPT, 10 MLM fwds) | ~20 min | ~15 min | **~35 min** |
| **All 5 sequentially** | — | — | **~50 min** |

Well under the 2h Exp_Climb paper_proto envelope. `exp_zs_02` is the heaviest — if Kaggle session budget is tight (<90 min total for ZS), drop `fast_detect_n_samples` back to 5 via `cfg.fast_detect_n_samples = 5` before `run_zs_oral`.

**Regression guard on the Binoculars O(N²) memory fix (2026-04-20):** the earlier implementation computed the cosine matrix against **all N test samples** (N² allocations), which OOMs on full Droid test (~106K × 106K floats ≈ 45 GB). The fix replaces the all-pairs softmax with a **reference-pool sketch** of size `binoculars_block_size` (1024 on H100). Memory drops from O(N²) to O(N·R). Theory unchanged — the reference pool plays the role of Binoculars' null "performer LM" population.

### Kaggle free-tier (T4 15GB / P100 16GB) — slower but fits

Same harness, slower forward pass (~6-8× H100). T4-preset bumps `num_workers` to 4 (Kaggle vCPU count) + keeps batch 32 fp16. T4 does NOT support bf16, so precision auto-drops to fp16.

**Wall-time estimate (T4 single-GPU, full Droid + CoDET tests per exp):**

| Exp | T4 single | H100 speedup | Notes |
|:--|:-:|:-:|:--|
| exp_zs_00 floor | ~15 s | 1.5× | CPU-bound, GPU idle |
| exp_zs_03 Ghostbuster (pure numpy) | **~2 min** | 1× | Identical — never touches GPU |
| exp_zs_04 SpectralSignature | ~25-30 min | 5-6× | 1 ModernBERT forward per bench |
| exp_zs_01 Binoculars | ~40-45 min | 5-6× | 1 ModernBERT forward + cosine sketch |
| exp_zs_02 Fast-DetectGPT | **~2 h 30 min** | 4-5× | 3 CodeBERT forwards × 2 benches, slowest by far |
| **Sequential total (all 5)** | **~3 h 45 min** | | Single 12 h Kaggle session handles it |

**Kaggle T4 recommendation.** If budget is tight, SKIP `exp_zs_02 Fast-DetectGPT` (the heaviest) and run only the first 4 in ~1h 10 min. Fast-DetectGPT is the theorem-carrying proof-of-curvature; having it is nice but not load-bearing for the oral story (Binoculars + SpectralSignature already cover the Neyman-Pearson / spectral angles). Run Fast-DetectGPT separately on an H100 kernel when available.

**Kaggle T4 × 2 (dual GPU).** Default Kaggle free tier. Our harness is single-device by design (no DataParallel). Best use: run Droid on GPU 0 and CoDET on GPU 1 *manually* by launching the script twice with different `CUDA_VISIBLE_DEVICES` and setting `benchmarks=("droid_T3",)` / `("codet_binary",)`. Halves the total sequential wall time to ~1 h 50 min for all 5 exps. We do NOT automate this (adds fragility for a small speedup); users who need it can edit `if __name__ == "__main__"` in each file.

| GPU preset | batch | prec | workers | FDG N | Bino R | Total wall (all 5, both benches) |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| H100 80GB | 128 | bf16 | 8 | 10 | 1024 | **~50 min** |
| A100 40GB / V100 32GB | 64 | bf16 / fp16 | 4 | 5 | 512 | ~1 h 30 min |
| **Kaggle T4 / P100 15-16GB** | **32** | **fp16** | **4** | **3** | **256** | **~3 h 45 min** |
| CPU | 8 | fp32 | 0 | 3 | 256 | day-scale, smoke only |

---

## 📐 Protocol (dual-benchmark oral default)

```python
from _common import ZSConfig
from _zs_runner import run_zs_oral

cfg = ZSConfig(
    benchmark="droid_T3",         # placeholder; overridden inside run_zs_oral
    max_dev_samples=5_000,        # calibration budget per bench
    max_test_samples=-1,          # FULL test
    human_recall_target=0.95,     # pin human R on dev calibration
)
run_zs_oral(
    method_name="MyZSMethod",
    exp_id="exp_zs_NN",
    score_fn=my_score_fn,          # (codes: list[str], cfg) -> np.ndarray
    cfg=cfg,
    # benchmarks=("droid_T3", "codet_binary") -- default; override if needed
)
```

- **No training.** `score_fn` must be stateless w.r.t. labels (except for Ghostbuster-Code which fits a tiny LR on dev — noted).
- **Threshold calibrated** on dev **per benchmark** (Droid and CoDET get their own τ); applied unchanged on that bench's test split.
- **Metrics per bench**: Macro-F1 + Weighted-F1 (binarised human vs AI) + Human R + AI R + Adversarial R + per-dim breakdown (domain / language / source_raw / generator / model_family).
- **Oral-level emit**: combined `BEGIN_ZS_ORAL_TABLE` block at end of the run with three automatic claim checks (Fast-DetectGPT beat · human recall floor · cross-benchmark stability).

---

## Ship order (recommended, dual-benchmark)

Each file runs BOTH Droid T3 + CoDET binary, so wall times below are per-file totals.

1. **exp_zs_00 Floor (random + length)** — ~10 s total. First gate: confirm the oral claim surface isn't a length-shortcut artefact.
2. **exp_zs_03 Ghostbuster-Code** — pure numpy, ~2 min both benches.
3. **exp_zs_04 SpectralSignature** — 1 ModernBERT fwd × 2, ~6 min.
4. **exp_zs_01 Binoculars** — 1 ModernBERT fwd × 2, ~10 min.
5. **exp_zs_02 Fast-DetectGPT** — 3 CodeBERT fwds × 2 (heaviest), ~20 min.

**Total Kaggle budget for all 5: ~40 min.** Well under the 2 h Exp_Climb paper_proto envelope, so can run in the same session.

---

## Oral-paper narrative (for the method section)

Single story the 5 runs together tell:

1. **Floor** (exp_zs_00) — random ≈ 0.50, length-only ≈ 0.55-0.60 on both benches. Gives the oral comparison block an anchor.
2. **Easy signal** (exp_zs_03 Ghostbuster) — token-statistics committee. Expected ≈ 0.60-0.65 on Droid T3, directly approaches Fast-DetectGPT. Weak baselines ARE this strong — establishes the ceiling for any "purely statistical" method.
3. **Spectral surrogate** (exp_zs_04 SpectralSignature) — ModernBERT layer-1 PC-1. Expected ≈ 0.60-0.68. Tests whether the Exp_Climb backbone's spectral stream is discriminative without any training — if yes, the axis-B spectral claim (paper §4) is already half-proven at inference time.
4. **Theoretical optimum for sibling pairs** (exp_zs_01 Binoculars + exp_zs_02 Fast-DetectGPT) — Neyman-Pearson-motivated scorers. These are the *theorem-carrying* methods that justify the oral framing.
5. **Cross-benchmark stability** — all 5 runs report on Droid T3 **and** CoDET binary. The oral claim surface is "our best ZS detector transfers within 10 pt across two distinct code-detection benchmarks" — a concrete, falsifiable statement.

Any one of these, reported alone, is a poster. Together they form the **ablation structure** a dedicated EMNLP zero-shot paper (or the appendix of the Exp_Climb oral) needs.

---

## What's NOT here (never will be)

- **Training runs** — those belong in `Exp_Climb/`. Don't blur the two.
- **AICD-Bench** — we excluded AICD from Exp_Climb as an open challenge; same exclusion applies to Exp_ZeroShot until an AICD zero-shot story emerges.
- **Text-only detectors** (RAID / M4-Text / MGT-Bench) — scope is code only. Text detection is a different SCM.
- **External LM scorers (GPT-4, Gemini)** — reproducibility blocker for EMNLP.
