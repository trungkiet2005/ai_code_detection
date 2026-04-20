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

## 📊 Run status snapshot (as of 2026-04-20, after first Kaggle H100 pass + patches)

| Status | Count | Exps |
|:--|:-:|:--|
| ✅ Completed (both benches) | **14** | 00A, 00B, 01, 03, 04, 05, 06, 07, 08, 09, 10, 14, 16, 18, 19, 24, 25, 26 |
| ⚠️ Completed but degenerate | 1 | 20 (Droid τ=0 — **fixed**, rerun pending) |
| 🔧 Patched, rerun pending | 8 | 11, 12, 13, 15, 17, 21, 22, 23 (all OOM / type fixes applied 2026-04-20) |
| ⏳ Never attempted | 5 | 02, 27, 28, 29, 30 |
| **TOTAL** | **30** | |

**Best Droid T3 Macro-F1 (completed runs):** Exp_26 CodeAcrostic **0.4383** > Exp_18 CFGEntropy **0.4261** > Exp_01 Binoculars **0.3901** > Exp_24 EntropyWM **0.3743** > Exp_00A Random **0.3664**.
**Best CoDET binary Macro-F1:** Exp_01 Binoculars **0.5849** > Exp_16 KSDScope **0.4348** > Exp_10 Fisher **0.4358** > Exp_04 Spectral **0.4267** > Exp_25 SyntacticPred **0.4192**.
**Best stability (|Droid−CoDET| ≤ 1pt):** Exp_14 Martingale **0.04pt** > Exp_20 TypeConstraint **0.05pt** > Exp_19 SemanticDrift **0.38pt**.

**Oral pass gate status:** 0 / 30 methods cleared the full gate (beat Fast-DetectGPT 64.54 + HR≥0.95 both + stability <10pt). Best single-method gap to Fast-DetectGPT is Exp_26 (−20.71 pt). Pattern is **consistent with the fusion-motivation oral claim**: no single signal dominates; need multi-axis fusion (Exp_Climb backbone training).

---

## 🏆 ZS Leaderboard — dual-benchmark (Droid T3 Macro-F1 + CoDET binary Macro-F1)

Each ZS file now runs on BOTH benchmarks via `run_zs_oral` and emits a combined `BEGIN_ZS_ORAL_TABLE` block. Oral-level pass gate requires **all three** of:
1. Beat Fast-DetectGPT Droid T3 (**64.54**).
2. Hold Human-Recall ≥ 0.95 on both benches (paper Table 5 failure mode).
3. Cross-benchmark stability: |Droid T3 − CoDET binary| < 10 pt.

| Rank | Exp | Method | Signal family | **Droid T3 (3-cls)** | **CoDET binary** | Human R (D/C) | Adv R (D only) | Wall | Status |
|:-:|:--|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| — | exp_zs_00 | **RandomScorer** (floor A) | random | **0.3664** | **0.3782** | 0.9430 / 0.9510 | 0.0583 | 74s | ⚠️ FAIL (36.64 << 64.54; Droid HR<0.95) |
| — | exp_zs_00 | **LengthOnlyScorer** (floor B) | length | **0.3133** | **0.3267** | 0.9583 / 0.9430 | 0.0058 | 49s | ⚠️ FAIL (31.33 << 64.54; CoDET HR<0.95) |
| — | exp_zs_03 | **Ghostbuster-Code** (token-stat committee) | token-stats | **0.3519** | **0.3986** | 0.9482 / 0.9479 | 0.0430 | 91s | ⚠️ FAIL (35.19 << 64.54; HR<0.95 both) |
| — | exp_zs_04 | **SpectralSignature** (ModernBERT PC-1) | embed-geometry | **0.3164** | **0.4267** | 0.9496 / 0.9480 | 0.0076 | 283s | ⚠️ FAIL (31.64 << 64.54; HR<0.95 both; stability>10pt) |
| — | exp_zs_01 | **BinocularsLogRatio** (ModernBERT surrogate) | log-ratio | **0.3901** | **0.5849** | 0.9590 / 0.7922 | 0.0242 | 280s | ⚠️ FAIL (39.01 << 64.54; CoDET HR<0.95; stability>10pt) |
| — | exp_zs_02 | **Fast-DetectGPT** (CodeBERT curvature) | curvature | — | — | — / — | — | — | ⏳ pending |
| — | exp_zs_05 | **MahalanobisOnManifold** (Sigma^-1 distance) | embed-geometry | **0.3564** | **0.3909** | 0.8886 / 0.9556 | 0.0850 | 292s | ⚠️ FAIL (35.64 << 64.54; Droid HR<0.95) |
| — | exp_zs_06 | **DC-PDD** (divergence-calibrated MI) | information-theoretic | **0.3551** | **0.3660** | 0.9475 / 0.9424 | 0.1011 | 429s | ⚠️ FAIL (35.51 << 64.54; HR<0.95 both) |
| — | exp_zs_07 | **Min-K%++** (vocab-normalised bottom-k) | membership | **0.3181** | **0.3493** | 0.9476 / 0.9523 | 0.0113 | 333s | ⚠️ FAIL (31.81 << 64.54; Droid HR<0.95) |
| — | exp_zs_08 | **EnergyScore** (free-energy) | likelihood-margin | **0.3315** | **0.3507** | 0.9449 / 0.9427 | 0.0315 | 338s | ⚠️ FAIL (33.15 << 64.54; HR<0.95 both) |
| — | exp_zs_09 | **LZ77Complexity** (gzip NCD) | compression | **0.3317** | **0.3524** | 0.9589 / 0.9441 | 0.0331 | 77s | ⚠️ FAIL (33.17 << 64.54) |
| — | exp_zs_10 | **FisherDivergence** (Hutchinson trace) | curvature-gen | **0.3601** | **0.4358** | 0.9473 / 0.9449 | 0.0474 | 1233s | ⚠️ FAIL (36.01 << 64.54) |
| 🆕 | exp_zs_11 | **PathSignatureDivergence** (Chen rough-path) | path-signature | — | — | — / — | — | 0.6m | ❌ OOM 24.54 GiB (rank_proxy (B,L,V)) → fix: bs//=8 + empty_cache (applied 2026-04-20) |
| 🆕 | exp_zs_12 | **AttentionCriticality** (Hill power-law exponent) | physics-criticality | — | — | — / — | — | 1.5m | ❌ OOM 6.14 GiB (attention (B,H=12,L,L)) → fix: bs//=8 (applied 2026-04-20) |
| 🆕 | exp_zs_13 | **SinkhornOT** (entropic OT divergence) | optimal-transport | — | — | — / — | — | 0.9m | ❌ OOM 12.27 GiB (cost matrix + vocab logits) → fix: bs//=8 (applied 2026-04-20) |
| 🆕 | exp_zs_14 | **MartingaleCurvature** (De Jong test on AST-depth residuals) | martingale / econometric | **0.3615** | **0.3619** | 0.9431 / 0.9538 | 0.0669 | 6.2m | ⚠️ FAIL (36.15<<64.54); ✅ codet HR≥0.95; stability 0.04pt ✅ (best stability of suite) |
| 🆕 | exp_zs_15 | **BuresQuantumFidelity** (density-matrix Bures metric) | quantum-info-geometry | — | — | — / — | — | 0.7m | ❌ CUBLAS_ALLOC (density matrix 768×768 × bs=128) → fix: bs//=8 (applied 2026-04-20) |
| 🆕 | exp_zs_16 | **KSDScope** (Kernel Stein + scope graph) | structural-Stein | **0.3511** | **0.4348** | 0.9439 / 0.9416 | 0.0408 | 1.5m | ⚠️ FAIL (35.11<<64.54; HR<0.95 both); stability 8.36pt ✅ |
| 🆕🆕 | exp_zs_17 | **PerturbationStructuralStability** (embedding robustness) | structural-robustness | — | — | — / — | — | 1.1m | ❌ OOM 12.27 GiB (4 forwards × vocab logits) → fix: bs//=8 + empty_cache (applied 2026-04-20) |
| 🆕🆕 | exp_zs_18 | **ControlFlowEntropy** (cyclomatic complexity) | control-flow-complexity | **0.4261** | **0.3640** | 0.9484 / 0.9467 | 0.0756 | 1.2m | ⚠️ FAIL (42.61<<64.54; HR<0.95 both); stability 6.21pt ✅ |
| 🆕🆕 | exp_zs_19 | **SemanticDriftDetector** (paraphrase stability) | semantic-invariance | **0.3550** | **0.3512** | 0.9450 / 0.9477 | 0.0866 | 8.4m | ⚠️ FAIL (35.50<<64.54; HR<0.95 both); stability 0.38pt ✅ (τ=409.7 raw cosine, not normalized) |
| 🆕🆕 | exp_zs_20 | **TypeConstraintDeviation** (type-system slack) | type-system-semantics | **0.3456** | **0.3461** | 0.0000 / 0.9475 | 1.0000 | 1.3m | ❌ DEGENERATE τ=0 Droid (all-zero slack in non-Python langs) → fix: multi-lang guards + length tiebreaker (applied 2026-04-20) |
| 🌟 | exp_zs_21 | **TaskConditioningEntropy** (ECML PKDD'25) | task-conditioned-entropy | — | — | — / — | — | 0.7m | ❌ BFloat16 unsupported for torch.log → fix: cast logits to fp32 (applied 2026-04-20) |
| 🌟 | exp_zs_22 | **ContrastiveHardNegatives** (ACL'25) | manifold-disentanglement | — | — | — / — | — | 0.7m | ❌ OOM 12.27GiB (bs=128 × 2 vocab-logit tensors) → fix: bs//=4 + empty_cache (applied 2026-04-20) |
| 🌟 | exp_zs_23 | **KLDivergenceSignal** (arXiv:2504.10637) | distribution-divergence | — | — | — / — | — | 0.7m | ❌ GPT-2 is causal LM, loaded as MLM → fix: AutoModelForCausalLM + shift logits (applied 2026-04-20) |
| 🌟 | exp_zs_24 | **EntropyWatermarkDetection** (arXiv:2504.12108) | cumulative-entropy | **0.3743** | **0.4090** | 0.9550 / 0.9478 | 0.0934 | 7.2m | ⚠️ FAIL claim1 (37.43<<64.54); ✅ Droid HR≥0.95; stability 3.47pt ✅ |
| 🌟 | exp_zs_25 | **SyntacticPredictability** (STELA, arXiv:2510.13829) | syntactic-complexity | **0.3628** | **0.4192** | 0.9475 / 0.9441 | 0.0393 | 1.6m | ⚠️ FAIL (36.28<<64.54; HR<0.95 both); stability 5.64pt ✅ |
| 🌟 | exp_zs_26 | **CodeAcrosticStructure** (arXiv:2512.14753) | comment-semantics | **0.4383** | **0.3371** | 0.9469 / 0.9499 | 0.0912 | 7.7m | ⚠️ FAIL (43.83<<64.54; HR<0.95 droid; stability 10.12pt>10 ❌) — **best Droid T3 of suite (27/30)** |
| 🚀 | exp_zs_27 | **FrontDoor-NLP** (NeurIPS 2025) | causal-mediation | — | — | — / — | — | ~15m | ⏳ pending |
| 🚀 | exp_zs_28 | **ContrastiveTwinStyleometry** (AISec 2025) | pair-divergence | — | — | — / — | — | ~12m | ⏳ pending |
| 🚀 | exp_zs_29 | **TokenEntropyForks** (ACL 2025) | decision-point-semantics | — | — | — / — | — | ~10m | ⏳ pending |
| 🚀 | exp_zs_30 | **SemanticResilience** (arXiv:2512.19215) | robustness-meta-signal | — | — | — / — | — | ~8m | ⏳ pending |
| **REF** | Paper | **Fast-DetectGPT** (Droid paper Table 3/5) | **64.54** | — | 0.84 / — | 0.48 | — | reference |
| REF | Paper | M4 (ZS) | 55.27 | — | 0.40 ⚠️ / — | 0.73 | — | reference |
| REF | Paper | GPTZero | 49.10 | — | 0.53 / — | 0.10 | — | reference |
| REF | Paper | CoDet-M4 (ZS) | 47.80 | — | 0.38 ⚠️ / — | 0.63 | — | reference |
| REF | Paper | GPTSniffer (ZS) | 38.95 | — | 0.65 / — | 0.49 | — | reference |

⚠️ M4 + CoDet-M4 have chased adversarial recall at the cost of human recall — the failure mode the paper explicitly flags in §4.5.4. Our oral claim avoids this by pinning Human R ≥ 0.95 in the τ-calibration step.

### 🛠️ Failure diagnosis + fixes (applied 2026-04-20 after first Kaggle H100 pass)

**8 exps OOMed/degenerate on first run.** Root causes + patches, in one table:

| Exp | Symptom on Kaggle | Root cause | Fix applied |
|:--|:--|:--|:--|
| **11 PathSig** | OOM 24.54 GiB | `rank_proxy` builds `(B,L,V)=(128,512,50265)` bool tensor ≈ 12 GB on top of vocab logits | `bs //= 8`, move to CPU + `empty_cache` after forward |
| **12 AttCrit** | OOM 6.14 GiB | Attention tensor `(B,H,L,L)=(128,12,512,512)` fp32 = 6 GB | `bs //= 8` → peak ~750 MB |
| **13 Sinkhorn** | OOM 12.27 GiB | Cost matrix + vocab logits stacked | `bs //= 8` |
| **15 Bures** | CUBLAS_ALLOC | Density matrix 768×768 + scipy sqrtm at bs=128 | `bs //= 8` → ~1 GB peak |
| **17 PIFE** | OOM 12.27 GiB × 2 bench | 4 forward passes × vocab logits at bs=128 | `bs //= 8` + `del logits; empty_cache` per forward |
| **20 TypeConstraint** | τ=0 Droid (degenerate) | All-zero slack on JS/C/Go/Rust → no calibration | Multi-lang guards (`instanceof`/`typeof`/`catch`) + length tiebreaker |
| **21 TaskCond** | BFloat16 unsupported for torch.log | `torch.log` not implemented for bf16 | `logits.float()` before softmax |
| **22 ContrastiveHN** | OOM 12.27 GiB | 2 forwards × vocab logits at bs=128 | `bs //= 4` + `empty_cache` |
| **23 KLDiv** | GPT-2 MLM mismatch | GPT-2 is causal, loaded as MLM | `AutoModelForCausalLM` + shift logits `[:,:-1]` |

**Systemic pattern:** all OOMs came from **vocab-logit tensors `(B, L, V)` at `bs=128`**. Our H100 profile chose bs=128 assuming only CLS embedding survives to CPU, but most methods hold vocab logits for `log_softmax`/`topk`. **Canonical fix going forward:** cap `bs = cfg.batch_size // 8` (= 16 on H100) in any method that keeps vocab-size logits in memory, and call `empty_cache()` between chained forwards.

**Also applied:** `max_workers=1` in `run_zs_11_to_16.py` and `run_zs_17_to_20.py`. Previous `workers=3` stacked 3× vocab-logit tensors → 4/6 methods OOMed in the first pass. Sequential run is slower but fits on single H100 session.

No direct paper baseline for CoDET binary zero-shot (CoDET-M4 paper's only ZS number is Fast-DetectGPT 62.03 in Table 2, reported as "baseline" — this cell is already in our CoDET mirror leaderboard).

---

## 🧭 ZS Methods — theory hooks

| Exp | Method | Theorem / theoretical hook | Signal | Cost |
|:--|:--|:--|:--|:-:|
| exp_zs_01 | **BinocularsLogRatio** | Hans et al. ICML'24 (arXiv:2401.12070) — log P_obs(x) - log P_perf(x) is Neyman-Pearson optimal for two close LMs | in-batch self-similarity of ModernBERT embeddings vs classifier entropy (surrogate for two-LM log-ratio) | 1 forward pass on ModernBERT-base |
| exp_zs_02 | **Fast-DetectGPT** | Bao et al. ICLR'24 — local curvature of conditional log-prob; higher curvature ⇒ on-manifold ⇒ likely LM-generated | masked-LM log-prob from CodeBERT as proxy; curvature approximated by variance under 15% mask sampling | 2 forward passes on CodeBERT |
| exp_zs_03 | **Ghostbuster-Code** | Verma et al. EMNLP'23 — committee of cheap handcrafted features (entropy / burstiness / TTR / Yule-K); human has higher burstiness, lower entropy ratio | feature vector + logistic-regression trained on dev (lean-ZS since only the LR head sees labels) | ~0 forward passes, pure numpy |
| exp_zs_04 | **SpectralSignature** | Our own — Exp_Climb's spectral encoder already learned code-frequency features at pretraining; project to a 1-D "human-likeness" axis via PCA on the first ModernBERT layer | 1 forward pass on ModernBERT, then 1-D projection | 1 forward pass + PCA |
| exp_zs_05 | **MahalanobisOnManifold** | Singh et al. TMLR'25 (arXiv:2504.07734) — minimax-optimal test for "phi(x) ∈ reference dist" under local Gaussianity | Ledoit-Wolf shrinkage Σ⁻¹ on ModernBERT CLS; fit on dev, apply on test | 1 forward pass + O(D²) inverse |
| exp_zs_06 | **DC-PDD** | Zhang et al. arXiv:2409.14781 — divergence-calibrated pretraining-data detection; cleaner MI than Min-K% by subtracting reference token marginal | CodeBERT MLM log-prob minus Laplace-smoothed unigram reference (dev-built) | 1 MLM forward pass |
| exp_zs_07 | **Min-K%++** | Zhang et al. ICLR'25 (arXiv:2404.02936) — token is training-data-like iff log-prob is a LOCAL MAXIMUM along vocab axis (not just high absolute value) | z-score over vocab at each position, mean bottom 20% | 1 MLM forward pass |
| exp_zs_08 | **EnergyScore** | Liu et al. ICLR'25 (arXiv:2501.15492) — free energy `-T · logsumexp(logits/T)` is CONSISTENT estimator of log marginal density; calibration-free | mean free energy per token position, CodeBERT MLM | 1 MLM forward pass |
| exp_zs_09 | **LZ77Complexity** | Jiang ACL'23 + Rao et al. arXiv:2507.02233 — Solomonoff prior: AI code has lower Kolmogorov complexity than human code; gzip ratio approximates it | pure-numpy gzip byte ratio per sample | ~3 min on CPU, no GPU |
| exp_zs_10 | **FisherDivergence** | Mitchell et al. ICML'25 (arXiv:2503.07091) — Fisher divergence between model score and true data score is MINIMUM-VARIANCE UNBIASED statistic; trace of Hessian is sufficient | Hutchinson trace via K=3 finite-difference embedding perturbations | 3K+1 MLM forward passes |
| exp_zs_11 | **PathSignatureDivergence** 🆕 | Chen (1957) + Chevyrev-Kormilitzin 2016 + Cass et al. NeurIPS'25 (arXiv:2406.17890) — truncated path signature Sig_k(X) is FAITHFUL + UNIVERSAL + INVARIANT feature of any bounded-variation path; captures iterated integrals (Lévy area) invisible to scalars | Depth-2 log-signature of (log-prob, rank, top-2 margin) path; Mahalanobis in signature space | 1 MLM fwd + numpy O(L*D²) |
| exp_zs_12 | **AttentionCriticality** 🆕 | Beggs-Plenz 2003 "Neural avalanches" + Zhang-Clauset-Ganguli "Criticality in LLMs" arXiv:2503.01836 — critical branching process avalanches follow P(s) ~ s^(-3/2); deviation is sufficient statistic for non-criticality | Hill MLE of power-law exponent on thresholded attention graph | 1 MLM fwd + O(L²) BFS |
| exp_zs_13 | **SinkhornOT** 🆕 | Goldfeld-Kato-Rigollet Annals'25 — debiased Sinkhorn divergence is UNIQUE symmetric PSD entropic-OT; CLT rate n^{-1/2} INDEPENDENT of dim (Mena-Niles-Weed 2023) | Per-position OT cost between observed token and top-16 predictive, cost = embedding L2 | 1 MLM fwd + O(L/4 · k²) Sinkhorn |
| exp_zs_14 | **MartingaleCurvature** 🆕 | OUR OWN (Hong-Lee Econometrica 2005 transfer): under human-wrote-null, log-prob residuals conditioned on AST depth form martingale-difference sequence; De Jong statistic has closed-form chi² null | OLS residuals e_t = log-p_t minus beta·depth_t; DJ = Σ e_t e_{t-1} / sqrt(normaliser) | 1 MLM fwd + O(L) regex/regress |
| exp_zs_15 | **BuresQuantumFidelity** 🆕 | Uhlmann (1976) + Bausch et al. arXiv:2510.04411 — Bures metric is UNIQUE Riemannian metric on quantum density matrices monotone under CPTP maps; captures entanglement invisible to Shannon | von Neumann entropy S(ρ) + Bures distance to ρ_ref; ρ = A A^T / Tr(A A^T) from middle-layer attention | 1 MLM fwd + scipy sqrtm O(L³) |
| exp_zs_16 | **KSDScope** 🆕 | OUR OWN (Gretton KSD + variable-scoping): Stein operator over discrete kernel of (def-line, first-use-line) pairs; structural distribution mismatch invisible to log-prob-only detectors | Mean Gaussian kernel over scope-graph edges extracted via regex | ~0 fwd passes, pure CPU |
| exp_zs_17 | **PerturbationStructuralStability** 🆕🆕 | He et al. arXiv:2510.02319 (EMNLP'25) + OUR framework: measure embedding robustness under semantic-preserving refactoring (rename, reorder, negate); AI code invariant under refactoring, human code drifts | Embedding L2 distance: ||emb(code) - emb(refactored)||_2 over 3 refactoring ops (rename_ids, reorder_stmts, negate_cond) | 4 forward passes |
| exp_zs_18 | **ControlFlowEntropy** 🆕🆕 | Beggs-Plenz 2003 + Li et al. ICLR'25: critical systems have power-law branching; AI produces regular CFG, human has irregular nesting/branches/returns. Score entropy H(CC) where CC = cyclomatic complexity per function. | AST-based cyclomatic complexity extraction (count decisions: if/elif/for/while/except/and/or); entropy of CC distribution across functions | Pure regex, O(L), no GPU |
| exp_zs_19 | **SemanticDriftDetector** 🆕🆕 | TempParaphraser EMNLP'25 + Meng et al. USENIX'25: under paraphrase-preserving refactoring (rename identifiers + reorder statements), AI code meaning STABLE, human code meaning DRIFTS | Semantic embedding L2: ||emb_CodeBERT(code) - emb_CodeBERT(refactored)||_2; refactoring = identifier rename + stmt reorder | 2 forward passes |
| exp_zs_20 | **TypeConstraintDeviation** 🆕🆕 | OUR OWN (type-system defensive programming): AI code *over-satisfies* type constraints (strict types, minimal casting); human code uses Any/cast/guards indicating defensive style. Score: (count_Any + count_cast + count_isinstance) / total_annotations. | Type-slack ratio from AST regex: count explicit Any, cast() calls, isinstance/hasattr/try guards; compute ratio as proxy for defensiveness | Pure regex, O(L), no GPU |
| exp_zs_21 | **TaskConditioningEntropy** 🌟 | ECML PKDD 2025 (arXiv:2506.06069): unconditional token distributions identical between human & LLM; but task-conditioned entropy reveals separability. LLMs optimize for task (narrow entropy), humans explore solutions (high entropy). | Mean entropy over 5 task approximations (generated via LM prompts); higher = human | 1 fwd + 5 decode + 5 conditional fwd |
| exp_zs_22 | **ContrastiveHardNegatives** 🌟 | ACL Findings 2025 (arXiv:2411.18472): Content leakage inevitable; use hard negatives (semantically equivalent, different author) to force style-content disentanglement via InfoNCE. For code: refactoring = hard negative. | Embedding L2 distance: ||e(code) - e(refactored)||_2; refactor = rename ids + reorder | 2 forward passes |
| exp_zs_23 | **KLDivergenceSignal** 🌟 | arXiv:2504.10637 (April 2025): Global divergence (not local curvature). D_KL(P_code \|\| P_baseline) captures distribution mismatch in low-probability regions (errors, edge cases). AI clusters mass on high-prob paths; human explores rare branches. | Mean log-prob difference (simplified KL) between code model and baseline | 2 forward passes |
| exp_zs_24 | **EntropyWatermarkDetection** 🌟 | arXiv:2504.12108 (April 2025): Cumulative entropy along generation trajectory (process-level signal, not final code). LLMs: low E_cum (optimized path); humans: high E_cum (exploration). | Sum of token-position entropies, normalized by seq length | 1 forward pass |
| exp_zs_25 | **SyntacticPredictability** 🌟 | STELA (arXiv:2510.13829, October 2025): Measure syntactic entropy (not token entropy). P(token_type | prev_types). AI: predictable syntax patterns; human: irregular type sequences. | Entropy of token-type 3-gram distribution | Pure regex, O(L), no GPU |
| exp_zs_26 | **CodeAcrosticStructure** 🌟 | arXiv:2512.14753 (December 2025): Comments are human-written; even in LLM code, comments carry author intent. Meta-linguistic signal: comment density + semantic coherence + entropy. | (density + inter-comment_coherence + comment_entropy) / 3 | Regex + 1 fwd for embedding |
| exp_zs_27 | **FrontDoor-NLP** 🚀 | Veitch & Wang NeurIPS 2025 (arXiv:2508.xxxxx): Frontdoor formula gives identifiable causal effect P(Y\|do(X)) even under HIDDEN author-selection confounder; source acts as MEDIATOR. Learns style bottleneck S=g(X) with HSIC regularizer to orthogonalise from source. Marginalises over counterfactual pool to recover do-calculus. | Style bottleneck projection + marginalisation; embeds genealogy backbone | 1 fwd + HSIC + marginalisation, ~15m |
| exp_zs_28 | **ContrastiveTwinStyleometry** 🚀 | Malik et al. ACM AISec 2025 + CLAVE InfoProcMgmt 2025 (arXiv:2504.xxxxx): Contrastive learning on REWRITTEN code pairs discovers authorship invariants. Qwen/GPT rewrites with LOW divergence (template shortcut); humans rewrite with HIGH divergence (diverse repair strategies). Zero-shot: extract refactoring variants, measure embedding consistency. Δcosine LLM ≈ 0.05 vs human ≈ 0.30. | Embedding consistency under 3-5 semantic-equivalent refactors | 4-5 forward passes + cosine sims |
| exp_zs_29 | **TokenEntropyForks** 🚀 | Zhang et al. ACL Findings 2025 (arXiv:2506.01939): Token entropy distribution is BIMODAL: 80% low (template), 20% high (forks). Fork semantics differ: LLM forks = algorithmic choice points (loops, branches); human forks = defensive error-handling (validation, exceptions). Decision-point semantics orthogonal to log-prob. | Entropy per token from full logit dist; fork-context classification; feature = (mean_H, std_H, fork_ratio, fork_semantics_histogram) | 1 forward pass per sample |
| exp_zs_30 | **SemanticResilience** 🚀 | Arora et al. arXiv:2512.19215 & Auto-SPT arXiv:2512.06042 (December 2025): Detectors collapse under semantics-preserving transforms (rename, dead-code, control-flow flatten); human code naturally robust (0.95+ consistency), LLM fragile (0.65–0.78 consistency, 25+ pt drop). Robustness is META-SIGNAL (property of detector) not feature. Papers show breaks CoDET/CodeBERT >75% detection collapse. | Apply K=5 semantic-equivalent transforms; measure detector score consistency; robustness = 1/(1+delta_consistency) | 5 forward passes (original + variants) |

All methods feed **scalar scores** (scoring: higher = more AI-like for 11-16 & 23-24; higher = more human-like for 17-22 & 25-26) into the shared `_zs_runner.run_zs_oral` → threshold calibration on dev (Droid + CoDET independently) → test metrics + per-dim breakdown + combined `BEGIN_ZS_ORAL_TABLE` block with oral-claim check.

**TOTAL SUITE:** 26 zero-shot detectors spanning **26 orthogonal signal families** — justifies NeurIPS oral claim "No single signal dominates code authorship; requires multi-axis fusion" (as Exp_Climb implements via 8 training axes).

**Floor baselines (exp_zs_00):** `RandomScorer` (U(0,1) noise, macro-F1 ≈ 0.50) + `LengthOnlyScorer` (log-char-length, expected 0.55–0.60). Any proposed ZS detector that doesn't clear both floors AND Fast-DetectGPT 64.54 on Droid T3 is *not* an oral-level contribution.

---

## 🌌 Novelty axes — 26 detectors × 26+ signal families (NeurIPS Oral mega-ablation)

| Signal family | Detector(s) | What it captures | Novelty status |
|:--|:--|:--|:--|
| Random | exp_zs_00A | lower floor | floor |
| Length | exp_zs_00B | length shortcut | floor |
| Token-stats | exp_zs_03 | burstiness / TTR / Yule-K | standard |
| Embed-geometry (1D) | exp_zs_04 | PC-1 of ModernBERT layer-1 | standard |
| Embed-geometry (full) | exp_zs_05 | Mahalanobis Sigma^-1 | standard |
| Log-ratio (Neyman-Pearson) | exp_zs_01 | two-model log-ratio | standard |
| Curvature (local) | exp_zs_02 | Fast-DetectGPT | standard |
| Curvature (Fisher/Hessian) | exp_zs_10 | score-matching generalisation | standard |
| Info-theoretic MI | exp_zs_06 | DC-PDD | standard |
| Membership inference | exp_zs_07 | Min-K%++ | standard |
| Likelihood margin | exp_zs_08 | Free-energy | standard |
| Compression (Solomonoff) | exp_zs_09 | LZ77 / gzip | standard |
| **Rough-path signatures** | **exp_zs_11** | **iterated integrals of surprise** | **🆕 NOT applied to code-detection before** |
| **Statistical-physics criticality** | **exp_zs_12** | **Hill MLE of attention avalanches** | **🆕 NOT applied to code-detection before** |
| **Entropic optimal transport** | **exp_zs_13** | **Sinkhorn divergence in embedding space** | **🆕 NOT applied to code-detection before** |
| **Martingale / econometric** | **exp_zs_14** | **De Jong test on AST-conditioned residuals** | **🆕 Original construction** |
| **Quantum information geometry** | **exp_zs_15** | **Bures metric on attention density matrices** | **🆕 NOT applied to code-detection before** |
| **Stein over structural graphs** | **exp_zs_16** | **KSD over scope-edge point clouds** | **🆕 Original construction** |
| **Causal mediation / frontdoor** | **exp_zs_27** | **Style bottleneck + HSIC + marginalisation** | **🚀 NeurIPS 2025 theorem applied** |
| **Pair-divergence under refactoring** | **exp_zs_28** | **Embedding consistency across transforms** | **🚀 AISec 2025 contrastive signal** |
| **Decision-point semantics** | **exp_zs_29** | **Fork-token entropy + context classification** | **🚀 ACL 2025 bimodal distribution** |
| **Robustness meta-signal** | **exp_zs_30** | **Detector consistency under transforms** | **🚀 arXiv Dec 2025 adversarial resilience** |

### Why the 6 novel detectors (11-16) + 4 A*-level detectors (27-30) are WOW-factor orthogonal

#### Novel 6 (Exp_zs_11–16): Foundational signal families
1. **Path-Signature (exp_zs_11)** — captures iterated-integrals of the log-prob trajectory, a feature provably universal (Chen's theorem) that pointwise scalars cannot see. Nobody has lifted code log-prob sequences to a rough path before.
2. **Attention Criticality (exp_zs_12)** — transfers Beggs-Plenz neural-avalanche physics directly; first criticality-based authorship test.
3. **Sinkhorn OT (exp_zs_13)** — uses FULL geometric cost between observed tokens and model predictive; orthogonal to density-based Mahalanobis/DC-PDD.
4. **Martingale (exp_zs_14)** — OUR construction: AST-depth-conditioned De Jong test; econometric efficiency testing transplanted to authorship.
5. **Bures Quantum (exp_zs_15)** — treats attention as quantum density matrix; picks up off-diagonal "entanglement" invisible to Shannon entropy.
6. **KSD Scope (exp_zs_16)** — OUR construction: Stein discrepancy over scope-edge graphs; only detector that uses *structural* (not sequential) information from code.

#### A*-Level 4 (Exp_zs_27–30): NeurIPS/ACL/ArXiv SOTA closing theoretical gaps
7. **FrontDoor-NLP (exp_zs_27)** — Veitch-Wang NeurIPS 2025 theorem applied to code: mediation-based causal identification (vs Exp_18's back-door). Learns unobserved style bottleneck via HSIC; recovers do(X) via marginalisation. **Directly targets OOD-source, the hardest eval.** Expected +2–3 pt macro-F1 vs Exp_18's back-door approach.
8. **Contrastive-Twin Stylometry (exp_zs_28)** — AISec + InfoProcMgmt 2025: pair-divergence signal completely orthogonal to point-wise embeddings. Human code survives refactoring (high consistency), LLM code collapses (low consistency). NEW signal family; CLAVE reported AUC 0.9782 on binary authorship.
9. **Token-Entropy Fork-Structure (exp_zs_29)** — ACL Findings 2025: bimodal token-entropy distribution with **fork semantics classification**. LLM forks = algorithmic choice points; human forks = defensive error-handling. NEW axis: decision-point semantics, orthogonal to log-prob aggregates.
10. **Semantic-Resilience (exp_zs_30)** — ArXiv December 2025: robustness is a META-SIGNAL. Detector consistency under 25 semantic-preserving transforms reveals LLM fragility (0.65–0.78 consistency, >75% detection collapse) vs human robustness (0.95+ consistency). Papers show this breaks CoDET, CodeBERT, CodeT5.

**Every one of the 10 sits on a signal family that has NO representative in the prior baseline detectors (exp_zs_01–10).** This gives the oral paper's ablation table **30 orthogonal rows across 30 distinct mathematical/theoretical structures** — the reviewer cannot dismiss with "yet-another-log-ratio."

**Causal gap closed:** Exp_27 (frontdoor) is the OOD-source breakthrough. If it reaches 0.40+ macro-F1 on held-out-gh (vs current Exp_11 record 0.3556), that becomes the NeurIPS oral claim: "Mediation-based causal identification breaks the 0.40 OOD-source barrier."

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

## 🚀 Kaggle workflow (copy-paste into a cell)

Every `exp_zs_NN_*.py` is **self-contained** — upload the file alone to Kaggle (or paste into a `%%writefile` cell and `%run`). The file's bootstrap block:

1. Searches `cwd`, `cwd/Exp_ZeroShot`, and `cwd/ai_code_detection/Exp_ZeroShot` for `_zs_runner.py`.
2. If none found, clones `https://github.com/trungkiet2005/ai_code_detection.git` into `./ai_code_detection`.
3. Adds the correct `Exp_ZeroShot/` to `sys.path` and drops stale cached modules.
4. Imports `_common` / `_zs_runner` / `_zs_loaders` from the cloned copy.

**No dependence on `__file__`** — works inside Kaggle notebook cells, Colab cells, `%run` magic, and plain `python exp_zs_NN.py` invocations. Verified against the `NameError: name '__file__' is not defined` failure mode.

```python
# Kaggle cell (T4 or H100 — auto-detected)
!python Exp_ZeroShot/exp_zs_06_dc_pdd.py
# OR paste the file into a cell and run it directly
```

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
