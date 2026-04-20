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

## 🏆 ZS Leaderboard — sorted by Droid T3 Macro-F1 ↓

| Rank | Exp | Method | Bench | Macro-F1 | Weighted-F1 | Human R | AI R | Adv R | τ | Wall | Status |
|:-:|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| — | exp_zs_01 | BinocularsLogRatio (fusion-embedding surrogate) | droid_T3 | — | — | — | — | — | — | — | ⏳ pending |
| — | exp_zs_02 | Fast-DetectGPT (CodeBERT scorer) | droid_T3 | — | — | — | — | — | — | — | ⏳ pending |
| — | exp_zs_03 | Ghostbuster-Code (token-stat committee) | droid_T3 | — | — | — | — | — | — | — | ⏳ pending |
| — | exp_zs_04 | SpectralSignature (our backbone ZS) | droid_T3 | — | — | — | — | — | — | — | ⏳ pending |
| **REF** | Paper | **Fast-DetectGPT** | droid_T3 | **64.54** | — | 0.84 | — | 0.48 | — | — | reference |
| REF | Paper | M4 | droid_T3 | 55.27 | — | 0.40 | — | 0.73 | — | — | reference |
| REF | Paper | GPTZero | droid_T3 | 49.10 | — | 0.53 | — | 0.10 | — | — | reference |
| REF | Paper | CoDet-M4 (ZS) | droid_T3 | 47.80 | — | 0.38 | — | 0.63 | — | — | reference |
| REF | Paper | GPTSniffer (ZS) | droid_T3 | 38.95 | — | 0.65 | — | 0.49 | — | — | reference |

---

## 🧭 ZS Methods — theory hooks

| Exp | Method | Theorem / theoretical hook | Signal | Cost |
|:--|:--|:--|:--|:-:|
| exp_zs_01 | **BinocularsLogRatio** | Hans et al. ICML'24 (arXiv:2401.12070) — log P_obs(x) - log P_perf(x) is Neyman-Pearson optimal for two close LMs | in-batch self-similarity of ModernBERT embeddings vs classifier entropy (surrogate for two-LM log-ratio) | 1 forward pass on ModernBERT-base |
| exp_zs_02 | **Fast-DetectGPT** | Bao et al. ICLR'24 — local curvature of conditional log-prob; higher curvature ⇒ on-manifold ⇒ likely LM-generated | masked-LM log-prob from CodeBERT as proxy; curvature approximated by variance under 15% mask sampling | 2 forward passes on CodeBERT |
| exp_zs_03 | **Ghostbuster-Code** | Verma et al. EMNLP'23 — committee of cheap handcrafted features (entropy / burstiness / TTR / Yule-K); human has higher burstiness, lower entropy ratio | feature vector + logistic-regression trained on dev (lean-ZS since only the LR head sees labels) | ~0 forward passes, pure numpy |
| exp_zs_04 | **SpectralSignature** | Our own — Exp_Climb's spectral encoder already learned code-frequency features at pretraining; project to a 1-D "human-likeness" axis via PCA on the first ModernBERT layer | 1 forward pass on ModernBERT, then 1-D projection | 1 forward pass + PCA |

All four feed **scalar scores** (higher = more AI-like) into the shared `_zs_runner.run_zs_suite` → threshold calibration on dev → test metrics + breakdown.

---

## 📐 Protocol

```python
from _common import ZSConfig
from _zs_runner import run_zs_suite

cfg = ZSConfig(
    benchmark="droid_T3",         # droid_T3 | droid_T1 | codet_binary
    max_dev_samples=5_000,        # calibration budget
    max_test_samples=-1,          # FULL test
    human_recall_target=0.95,     # pin human R on dev calibration
)
result = run_zs_suite(
    method_name="MyZSMethod",
    exp_id="exp_zs_NN",
    score_fn=my_score_fn,          # (codes: list[str], cfg) -> np.ndarray
    cfg=cfg,
)
```

- **No training.** `score_fn` must be stateless w.r.t. labels (except for Ghostbuster-Code which fits a tiny LR on dev — noted).
- **Threshold calibrated** on dev to pin human recall ≥ target; applied unchanged on test.
- **Metrics**: Macro-F1 + Weighted-F1 (binarised human vs AI) + Human R + AI R + Adversarial R + per-dim breakdown (domain / language / source_raw / generator / model_family).
- **Paper-ready emit**: `BEGIN_ZS_PAPER_TABLE` block at end of each run with Δ vs every paper ZS baseline.

---

## Ship order (recommended)

1. **exp_zs_03 Ghostbuster-Code** first — no LM forward passes, only numpy. Proves the harness end-to-end in ~60 s on CPU.
2. **exp_zs_04 SpectralSignature** — uses the Exp_Climb backbone, ~3 min on H100.
3. **exp_zs_01 Binoculars** — one extra forward pass, ~5 min.
4. **exp_zs_02 Fast-DetectGPT** — heaviest (2 forward passes + variance), ~10 min.

Total Kaggle budget: ~20 min for all 4. Way under the 2 h Exp_Climb paper_proto envelope, so can be run in the same session.

---

## What's NOT here (never will be)

- **Training runs** — those belong in `Exp_Climb/`. Don't blur the two.
- **AICD-Bench** — we excluded AICD from Exp_Climb as an open challenge; same exclusion applies to Exp_ZeroShot until an AICD zero-shot story emerges.
- **Text-only detectors** (RAID / M4-Text / MGT-Bench) — scope is code only. Text detection is a different SCM.
- **External LM scorers (GPT-4, Gemini)** — reproducibility blocker for EMNLP.
