# Performance Tracker - AICD-Bench

## Experiment Protocol (NeurIPS-oral target)

- Do **not** mix AICD and Droid in one training run.
- Train/evaluate **separate models per benchmark**:
  - `Benchmark A`: AICD-Bench (T1/T2/T3 as defined by benchmark).
  - `Benchmark B`: DroidCollection (T1 or T3 mapping).
- Report per-benchmark metrics independently, plus cross-benchmark transfer only as auxiliary study.

## Experiment Results

| Exp | Name | Task | Model | Epochs | Best Val F1 | Test F1 | Notes |
|-----|------|------|-------|--------|-------------|---------|-------|
| exp00 | CodeOrigin | T1 | ModernBERT-base | 5 | 0.9948 | 0.2625 | Severe overfitting: val F1 ~0.99 but test F1 ~0.26 |
| exp01-aicd | CodeOrigin (H100 profile + strict preflight) | T1 | ModernBERT-base | 3 | 0.9954 | 0.2877 (macro), 0.2721 (weighted) | AICD still shows severe OOD collapse despite better val |
| exp01-droid | CodeOrigin (H100 profile + strict preflight) | T1 | ModernBERT-base | 3 | 0.9693 | 0.9708 (macro), 0.9709 (weighted) | Droid benchmark reached strong in-distribution performance |

## Baseline Result (AICD T1)

| timestamp | benchmark | task | train | val | test | best_val_f1 | test_f1 |
|---|---|---|---:|---:|---:|---:|---:|
| 2026-03-29 12:24:05 | AICD-Bench | T1 | 100000 | 20000 | 50000 | 0.9948 | 0.2625 |
| 2026-03-29 14:42:00 | AICD-Bench | T1 | 100000 | 20000 | 50000 | 0.9954 | 0.2877 |

---

## Benchmark Suite Result (H100, strict preflight)

- **Run timestamp:** `2026-03-29 14:42:00`
- **Mode:** `RUN_MODE=both`, independent training per benchmark (fresh model each run)
- **Hardware profile:** `NVIDIA H100 80GB HBM3`, `precision=bf16`, `batch_size=64`, `grad_accum_steps=1`, `workers=8`
- **Dependency bootstrap:** auto-installed `tree-sitter` and `tree-sitter-languages` successfully.

### Preflight report snapshot

| Benchmark | Train | Val | Test | Label counts (train) | Avg chars | Avg lines | Result |
|---|---:|---:|---:|---|---:|---:|---|
| AICD (T1) | 100000 | 20000 | 50000 | 0:47691, 1:52309 | 835.56 | 38.02 | PASS |
| DROID (T1) | 100000 | 20000 | 50000 | 0:47269, 1:52731 | 1435.91 | 45.96 | PASS |

### Suite summary (copied from `SUITE_RESULTS_*` logs)

| benchmark | task | best_val_f1 | test_f1 | weighted_test_f1 |
|---|---|---:|---:|---:|
| AICD | T1 | 0.9954 | 0.2877 | 0.2721 |
| DROID | T1 | 0.9693 | 0.9708 | 0.9709 |

### Interpretation for paper drafting

- **Key contrast:** same method family shows opposite behavior across benchmarks: strong on Droid, weak on AICD test-OOD.
- **AICD evidence of OOD failure:** val remains very high while test macro-F1 stays low (`0.2877`), consistent with benchmark difficulty claim.
- **Droid evidence of fit to benchmark distribution:** test macro/weighted-F1 around `0.97`, indicating stable detection on this split protocol.
- **Research implication:** prioritize OOD-focused innovations for AICD while keeping Droid as strong-support benchmark for method capacity.

---

## Paper-to-Method Comparison (AICD + Droid)

### AICD paper baselines vs our exp01-aicd (T1)

Reference: `paper_AICD.md` Table 4 (Macro-F1, values in %).

| Method | Reported T1 Macro-F1 (%) |
|---|---:|
| Random baseline | 45.73 |
| SVM (TF-IDF) | 43.05 |
| DeBERTa | 34.13 |
| ModernBERT baseline | 30.61 |
| **CodeOrigin exp01-aicd** | **28.77** |

Notes:
- Our `exp01-aicd` improves over prior local `exp00` (26.25 -> 28.77), but is still below strong AICD baselines from the paper.
- This confirms T1 OOD remains the main bottleneck for current method design.

### Droid paper baselines vs our exp01-droid (binary setting)

Reference: `paper_Droid.md` Table 3/4 (Weighted-F1, values in %).

| Method | 2-class weighted-F1 (%) |
|---|---:|
| Fast-DetectGPT (avg) | 67.85 |
| CoDet-M4FT (avg) | 99.08 |
| DroidDetectCLS-Base (avg) | 99.11 |
| DroidDetectCLS-Large (avg) | 99.23 |
| **CodeOrigin exp01-droid** | **97.09** |

Notes:
- Our Droid binary result is strong and clearly above zero-shot baselines, but still below DroidDetectCLS full-training SOTA in the paper.
- Direct comparability is partial because current run uses benchmark subsampling and does not yet include 3-class/4-class + adversarial recall protocol.

### Protocol audit against papers (run readiness)

| Check item | Status | Comment |
|---|---|---|
| Separate benchmark training | PASS | Independent runs for AICD and Droid, fresh model per benchmark. |
| Preflight before training | PASS | Dataset/schema/tokenizer/feature checks all pass. |
| Tree-sitter AST parsing required | PASS | Auto-install and strict preflight enabled. |
| H100 optimization | PASS | BF16 + batch 64 + workers 8 + TF32 backend applied. |
| Epoch count aligned with papers | PASS | Set to 3 epochs. |
| Metric logging breadth | PASS | Macro-F1 + Weighted-F1 both logged. |
| AICD T1 split-wise OOD reporting | PARTIAL | Aggregate test reported; split-by-language/domain tables not yet emitted. |
| Droid 3-class / 4-class protocol | PARTIAL | Current run is T1 binary only. |
| Droid adversarial robustness table (human vs adversarial recall) | MISSING | Needed to mirror Table 5 style claim in `paper_Droid.md`. |

### Immediate insights (for next exp wave)

1. **AICD OOD robustness is the blocker**: high val but low test indicates shortcut learning under domain/language shift.
2. **Droid binary is already competitive**: optimize toward Droid 3-class/4-class + adversarial protocol to claim broader robustness.
3. **To align paper-quality evidence**, next must-have outputs:
   - AICD T1 split-wise OOD metrics (seen/unseen language/domain),
   - Droid 3-class + 4-class weighted-F1,
   - adversarial recall table (human vs adversarial).

---

## EDA Snapshot (2026-03-29)

Source file: `docs/eda_two_benches.json` (API-sampled).

### Dataset scale

- **AICD-Bench**
  - T1: train 500,000 / val 100,000 / test 1,108,207
  - T2: train 502,149 / val 101,176 / test 507,874
  - T3: train 900,000 / val 200,000 / test 1,000,000
- **DroidCollection**
  - train 846,598 / dev 105,824 / test 105,826

### AICD-Bench T1 sample stats

- Sampled rows: train 2,000 / val 2,000 / test 200 (API-rate limited on test window).
- Label ratio (train): class `0`=933, class `1`=1067 (near-balanced).
- Code length (train): mean 805 chars, p50 456, p95 2648.
- LOC (train): mean 34.3, p50 21, p95 104.
- Style signal proxies:
  - comment-like presence: 42.7%
  - tab indentation presence: 43.7%
  - near-duplicate proxy: ~0.05%

### DroidCollection sample stats

- Train quick sample 500:
  - labels: MACHINE_GENERATED 211, MACHINE_REFINED 110, HUMAN_GENERATED 179
  - top languages: Python 182, Java 113, C# 52, C++ 50, JavaScript 43
  - top sources: STARCODER_DATA 109, THEVAULT_FUNCTION 100, TACO 97, DROID_PERSONAHUB 81
- Dev sample 2,000:
  - labels: MACHINE_GENERATED 825, MACHINE_REFINED 435, HUMAN_GENERATED 740
  - generation mode: INSTRUCT 1162, Human_Written 740, COMPLETE 98
  - avg size: 1285 chars, 43.5 lines
  - comment-like presence: 59.9%
- Test sample 1,600:
  - labels: MACHINE_GENERATED 670, MACHINE_REFINED 344, HUMAN_GENERATED 586
  - avg size: 1264 chars, 43.7 lines
  - comment-like presence: 62.5%

---

## Detailed Results

### exp00 - CodeOrigin (Style-Content Disentanglement + Hierarchical Contrastive Learning)

- **Task:** T1 (binary, 2 classes)
- **Backbone:** `answerdotai/ModernBERT-base` (153M params, all trainable)
- **Batch size:** 32 x 2 = 64
- **FP16:** True
- **Device:** CUDA

#### Training Loss Progression

| Epoch | Total Loss | Task | Proto | Contrastive | Disentangle | Adversarial | Recon | MI | KL |
|-------|-----------|------|-------|-------------|-------------|-------------|-------|------|-------|
| 1 | 0.6029 | 0.0316 | 0.0902 | 1.6528 | 0.2714 | 0.0156 | 0.1346 | 0.0216 | 24.977 |
| 2 | 0.5089 | 0.0085 | 0.0267 | 1.5444 | 0.2128 | 0.0006 | 0.0753 | 0.0014 | 21.141 |
| 3 | 0.4962 | 0.0049 | 0.0169 | 1.5214 | 0.2189 | 0.0001 | 0.0780 | 0.0007 | 21.823 |
| 4 | 0.4846 | 0.0026 | 0.0089 | 1.5019 | 0.2196 | 0.0001 | 0.0671 | 0.0011 | 21.849 |
| 5 | 0.4778 | 0.0012 | 0.0039 | 1.4905 | 0.2202 | 0.0001 | 0.0629 | 0.0009 | 21.932 |

#### Validation F1 Progression

| Epoch | Val Loss | Val Macro-F1 | Best? |
|-------|----------|-------------|-------|
| 1 | 0.0828 | 0.9916 | |
| 2 | 0.0569 | 0.9936 | * |
| 3 | 0.0571 | 0.9942 | * |
| 4 | 0.0509 | 0.9948 | * |
| 5 | 0.0483 | 0.9945 | |

- **Best checkpoint:** Epoch 4 end (Val Macro-F1: 0.9948)

#### Test Results (from best checkpoint)

| Metric | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|--------|---------|---------|-----------|--------------|
| Precision | 0.6583 | 0.1977 | 0.4280 | 0.5574 |
| Recall | 0.1254 | 0.7681 | 0.4467 | 0.2662 |
| F1-score | 0.2107 | 0.3144 | 0.2625 | 0.2334 |
| Support | 39,045 | 10,955 | 50,000 | 50,000 |

- **Test Accuracy:** 0.2662
- **Test Macro-F1:** 0.2625
- **Test Loss:** 2.2274

### exp01-aicd - CodeOrigin (H100 profile, 3 epochs)

- **Task:** T1 (binary, 2 classes)
- **Backbone:** `answerdotai/ModernBERT-base`
- **Hardware:** `H100 80GB`, `bf16`, effective batch `64`
- **Best Val Macro-F1:** `0.9954`
- **Test:** `Macro-F1 0.2877`, `Weighted-F1 0.2721`, `Loss 2.0574`
- **Observation:** slight improvement over exp00 test macro (+0.0252), but still severe train/val-test mismatch.

### exp01-droid - CodeOrigin (H100 profile, 3 epochs)

- **Task:** T1 (binary, 2 classes, Droid mapping)
- **Backbone:** `answerdotai/ModernBERT-base`
- **Hardware:** `H100 80GB`, `bf16`, effective batch `64`
- **Best Val Macro-F1:** `0.9693`
- **Test:** `Macro-F1 0.9708`, `Weighted-F1 0.9709`, `Loss 0.1317`
- **Observation:** high and stable performance on Droid split; no large val-test collapse observed.

#### GRL-lambda Schedule

| Epoch | GRL-lambda |
|-------|-----------|
| 1 | 0.000 |
| 2 | 0.762 |
| 3 | 0.964 |
| 4 | 0.995 |
| 5 | 0.999 |

#### Key Observations
- **Massive train/val vs test gap:** Val Macro-F1 reaches 0.9948 but test drops to 0.2625 - strong distribution shift between val and test sets
- Class 0 (human) has very low recall (0.1254) on test - model predicts class 1 too aggressively
- Class imbalance in test: 39,045 (class 0) vs 10,955 (class 1) ~78:22 ratio
- Task loss converges very quickly (0.0316 -> 0.0012), suggesting the model memorizes training patterns rather than learning generalizable features
- The domain-invariant features (adversarial loss near 0) may not transfer to the test distribution

---

## Modeling Insights

1. **Length/domain shift is real**: Droid samples are substantially longer than AICD T1 samples.
   -> Keep max length 512 but add length-aware batching and report by length buckets.
2. **Human vs machine priors differ per benchmark**: AICD T1 close to balanced, Droid has strong MACHINE_GENERATED + MACHINE_REFINED mass.
   -> Use benchmark-specific class weights and threshold calibration.
3. **Source heterogeneity in Droid is high** (TACO/TheVault/PersonaHub).
   -> Include source/domain-aware stratified validation and per-source metrics.
4. **Comments are a strong but risky shortcut** (Droid comment-like ratio is higher).
   -> Run ablation with comment stripping / docstring masking to avoid leakage.
5. **Tab-indentation behavior differs between benchmarks**.
   -> Keep structural/style features, but regularize to prevent overfitting to formatting artifacts.

## Deep Methods Suite (Exp_DM)

6 advanced experiments created from Gemini DeepResearch Round 1 portfolio, targeting NeurIPS 2026 ORAL.
All files in `Exp_DM/` folder, standalone Kaggle-runnable. Full tracker at `Exp_DM/dm_tracker.md`.

| Exp | Method | Core Innovation | Tier | Target Bottleneck |
|-----|--------|----------------|------|-------------------|
| exp01 | **CausAST** | Frobenius norm orthogonal covariance (token vs AST) + Batch-Hard Triplet | A (Flagship 1) | AICD OOD generalization |
| exp02 | **TTA-Evident** | Evidential Deep Learning (Dirichlet) + Test-Time MLM Adaptation | A (Flagship 2) | Calibration + OOD uncertainty |
| exp03 | **AP-NRL** | SupCon + Code Augmenter (humanization sim) + dual-view consistency | A (Flagship 3) | Droid adversarial robustness |
| exp04 | **BH-SCM** | Batch-Hard Triplet multi-view + Cross-View Consistency | A (Must-Run) | Multi-view alignment |
| exp05 | **OSCP** | Orthogonal projection + Variance-Invariant Whitening + SupCon | A (Must-Run) | Style-content disentanglement |
| exp06 | **AST-IRM** | IRM across language environments + annealed penalty | B (High Novelty) | Cross-language causal invariance |

### Execution Priority
1. exp01 CausAST (AICD T1) → core OOD claim
2. exp02 TTA-Evident (AICD T1) → calibration claim
3. exp05 OSCP (AICD T1) → disentanglement ablation
4. exp03 AP-NRL (DROID T1) → adversarial robustness
5. exp04 BH-SCM (AICD T1) → multi-view baseline
6. exp06 AST-IRM (AICD T1) → causal invariance (high risk/reward)

---

## Next modeling plan

- Train 2 independent tracks:
  - Track A: `benchmark=aicd` (primary AICD leaderboard).
  - Track B: `benchmark=droid` (primary Droid benchmark).
- For each track:
  - standard run + comment-masked ablation
  - length-bucket evaluation
  - confusion matrix on hardest class pairs
