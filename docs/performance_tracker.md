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
| exp00-aicd-v2 | CodeOrigin (regex AST, 5 epoch) | T1 | ModernBERT-base | 5 | 0.9952 | 0.2839 (macro), 0.2646 (weighted) | Regex AST fallback, consistent with prior runs |
| exp00-droid-v2 | CodeOrigin (regex AST, 5 epoch) | T1 | ModernBERT-base | 5 | 0.9717 | 0.9718 (macro), 0.9718 (weighted) | Droid stable ~0.97, confirms prior result |
| exp01-aicd | CodeOrigin (H100 profile + strict preflight) | T1 | ModernBERT-base | 3 | 0.9954 | 0.2877 (macro), 0.2721 (weighted) | AICD still shows severe OOD collapse despite better val |
| exp01-droid | CodeOrigin (H100 profile + strict preflight) | T1 | ModernBERT-base | 3 | 0.9693 | 0.9708 (macro), 0.9709 (weighted) | Droid benchmark reached strong in-distribution performance |
| exp01-simple | Simple Baseline + OOD Regularization | T1 | ModernBERT-base | 3 | 0.9959 | *pending* | R-Drop α=5, label smoothing 0.1, multi-dropout K=5, SWA, max_len=1024, train=200k |
| exp02 | Code Stylometry + LightGBM | T1 | LightGBM (no neural) | 212 iters | 0.9918 | 0.2078 | 140 stylometric + char/word TF-IDF SVD = 1140 features. Worst test F1 yet |
| exp03 | Frozen Encoder + Group DRO | T1 | ModernBERT-base (frozen) | 10 | 0.9857 | *pending* | Only 1.6% params trainable (2.4M/151M). DRO α=0.2, augment 0.5, layers [-1,-4,-8] |
| exp04 | Multi-View Stacking Ensemble | T1 | 4x LightGBM + LogReg meta | 5-fold | 0.9934 | 0.2053 | 4 views stacked: char-ngram, word-ngram, stylometry, ModernBERT-emb. Worst test yet |
| exp00-T2 | CodeOrigin | **T2** | ModernBERT-base | 3 | 0.4734 | 0.3206 (macro), 0.6684 (weighted) | 12-class family attribution. Extreme class imbalance (class 0 = 88%). H100 bf16 |
| exp00-T3-aicd | CodeOrigin | **T3** | ModernBERT-base | 3 | 0.8231 | 0.5832 (macro), 0.6894 (weighted) | AICD 4-class: human/machine/hybrid/adversarial. OOD gap still present |
| exp00-T3-droid | CodeOrigin | **T3** | ModernBERT-base | 3 | 0.8921 | 0.8925 (macro), 0.9078 (weighted) | Droid 3-class: human/gen/refined. Strong performance, val≈test |
| **exp06-aicd** | **AST-IRM** | T1 | ModernBERT-base | 3 | 0.9944 | 0.2930 (macro), 0.2725 (weighted) | IRM penalty explodes epoch 3 (λ=5000, loss~365). No OOD improvement |
| **exp06-droid** | **AST-IRM** | T1 | ModernBERT-base | 3 | 0.9674 | 0.9685 (macro), 0.9685 (weighted) | Droid stable. IRM penalty ~0.073 epoch 3 |
| **exp05-aicd** | **OSCP** | T1 | ModernBERT-base | 3 | 0.9938 | 0.2634 (macro), 0.2469 (weighted) | Whitening loss dominates (~206). Ortho converges to 0.002. Worse than baseline |
| **exp05-droid** | **OSCP** | T1 | ModernBERT-base | 3 | 0.9640 | 0.9649 (macro), 0.9649 (weighted) | Droid slightly below baseline (0.9649 vs 0.9718) |
| exp04-irm-aicd | AST-IRM (w/ proto+contrastive) | T1 | ModernBERT-base | 3 | 0.9948 | 0.2779 (macro), 0.2558 (weighted) | Full CodeOrigin arch + IRM. Same λ explosion. Worse than exp06 |
| exp04-irm-droid | AST-IRM (w/ proto+contrastive) | T1 | ModernBERT-base | 3 | 0.9689 | 0.9687 (macro), 0.9688 (weighted) | Droid stable |
| **exp04-bh-aicd** | **BH-SCM** | T1 | ModernBERT-base | 3 | 0.9952 | 0.2811 (macro), 0.2670 (weighted) | Clean loss curve. xview→0.0002 by epoch 2. No OOD gain |
| **exp04-bh-droid** | **BH-SCM** | T1 | ModernBERT-base | 3 | 0.9698 | 0.9700 (macro), 0.9701 (weighted) | Droid stable, ngang baseline |
| **exp03-aicd** | **AP-NRL** | T1 | ModernBERT-base | 3 | 0.9951 | 0.2842 (macro), 0.2650 (weighted) | SupCon ~4.18 plateau. Consistency→0. Baseline-level |
| **exp03-droid** | **AP-NRL** | T1 | ModernBERT-base | 3 | 0.9702 | 0.9699 (macro), 0.9699 (weighted) | Droid stable |
| **exp01-aicd** | **CausAST** | T1 | ModernBERT-base | 3 | 0.9949 | 0.2753 (macro), 0.2574 (weighted) | Ortho converges (1.12→0.0002). Worst DM on AICD |
| **exp01-droid** | **CausAST** | T1 | ModernBERT-base | 3 | 0.9672 | 0.9686 (macro), 0.9687 (weighted) | Droid stable |

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

### AICD T2 (12-class family attribution) vs paper baselines

Reference: AICD paper Table 5 (Macro-F1).

| Method | T2 Macro-F1 (%) |
|---|---:|
| Random baseline | ~8.3 |
| **CodeOrigin exp00-T2** | **32.06** |

### exp00-T2 per-class breakdown

| Class | Support | Precision | Recall | F1 | Train % |
|-------|---------|-----------|--------|------|---------|
| 0 (human) | 23,870 | 0.9373 | 0.9825 | 0.9594 | 87.9% |
| 10 | 10,260 | 0.8401 | 0.6389 | 0.7258 | 2.1% |
| 11 | 810 | 0.4397 | 0.5630 | 0.4938 | 0.5% |
| 3 | 853 | 0.1956 | 0.3552 | 0.2523 | 0.6% |
| 4 | 381 | 0.1412 | 0.3727 | 0.2048 | 0.4% |
| 6 | 2,062 | 0.2555 | 0.2270 | 0.2404 | 1.2% |
| 8 | 1,110 | 0.1687 | 0.3009 | 0.2162 | 1.6% |
| 9 | 1,407 | 0.1490 | 0.2623 | 0.1900 | 0.9% |
| 5 | 2,105 | 0.2621 | 0.1492 | 0.1901 | 0.4% |
| 2 | 2,617 | 0.1866 | 0.1525 | 0.1678 | 1.9% |
| 7 | 3,593 | 0.2190 | 0.0918 | 0.1294 | 1.7% |
| 1 | 932 | 0.0534 | 0.1406 | 0.0774 | 0.8% |

Key observations:
- **Extreme class imbalance:** class 0 (human) = 87.9% of train, dominates predictions
- **Class 0 & 10 perform well** (F1 > 0.72) — these are the two largest classes
- **Minority classes are near-random** (F1 0.07-0.25) — model can't distinguish rare families
- **Val Macro-F1 (0.47) vs Test Macro-F1 (0.32):** same OOD gap pattern as T1
- **Weighted-F1 (0.67)** much higher than Macro-F1 (0.32) due to class 0 dominance

### AICD & Droid T3 (multi-class) baseline results

**AICD T3** (4-class: human / machine / hybrid / adversarial):

| Class | Support | Precision | Recall | F1 | Train % |
|-------|---------|-----------|--------|------|---------|
| 0 (human) | 33,668 | 0.9762 | 0.6822 | 0.8031 | 53.8% |
| 1 (machine) | 5,841 | 0.3855 | 0.7696 | 0.5137 | 23.5% |
| 2 (hybrid) | 7,293 | 0.2406 | 0.3603 | 0.2886 | 9.6% |
| 3 (adversarial) | 3,198 | 0.6628 | 0.8064 | 0.7276 | 13.1% |

- **Macro-F1: 0.5832** — much better than T1 (0.28) and T2 (0.32)
- **Hybrid class (2) is hardest** — F1 only 0.29, confused with machine
- **Adversarial class (3) surprisingly good** — F1 0.73, strong signal
- Val-test gap reduced but still present (0.82 → 0.58)

**Droid T3** (3-class: human / machine_generated / machine_refined):

| Class | Support | Precision | Recall | F1 |
|-------|---------|-----------|--------|------|
| 0 (human) | 23,648 | 0.9630 | 0.9707 | 0.9668 |
| 1 (machine_gen) | 12,562 | 0.8702 | 0.8616 | 0.8659 |
| 2 (machine_refined) | 13,790 | 0.8468 | 0.8429 | 0.8448 |

- **Macro-F1: 0.8925** — excellent, val≈test (no OOD gap)
- All 3 classes perform well (F1 > 0.84)
- Machine_refined is hardest but still strong (0.84)
- **First multi-class Droid result** — confirms 3-class protocol works

### Cross-task comparison (CodeOrigin baseline)

| Benchmark | Task | Classes | Val F1 | Test F1 | Val-Test Gap |
|-----------|------|---------|--------|---------|-------------|
| AICD | T1 | 2 | 0.9952 | 0.2839 | -0.711 |
| AICD | T2 | 12 | 0.4734 | 0.3206 | -0.153 |
| AICD | T3 | 4 | 0.8231 | 0.5832 | -0.240 |
| Droid | T1 | 2 | 0.9717 | 0.9718 | +0.001 |
| Droid | T3 | 3 | 0.8921 | 0.8925 | +0.000 |

**Key insight:** AICD OOD gap is worst on T1 (binary) and progressively smaller on T2/T3. Droid has zero gap across all tasks. This suggests AICD T1's extreme val-test gap is partly due to test set distribution shift, not just model weakness.

### exp06 - AST-IRM (Exp_DM) Results

**First DM experiment completed.** AST-driven Invariant Risk Minimization across language environments.

| Benchmark | Task | Best Val F1 | Test Macro-F1 | Test Weighted-F1 |
|-----------|------|-------------|---------------|------------------|
| AICD | T1 | 0.9944 | **0.2930** | 0.2725 |
| Droid | T1 | 0.9674 | **0.9685** | 0.9685 |

vs CodeOrigin baseline: AICD 0.2930 vs 0.2839 (+0.009), Droid 0.9685 vs 0.9718 (-0.003)

**IRM penalty behavior:**
- Epochs 1-2: `irm_lambda=0.0` (pure ERM phase) — model trains normally
- Epoch 3: `irm_lambda=5000.0` — **penalty explodes**: total loss jumps from ~0.02 to ~365
- IRM penalty magnitude: ~0.07 per environment, but multiplied by λ=5000 → ~350 total loss
- Despite explosion, val F1 remains stable (0.9943) — the best checkpoint was saved at epoch 2

**Key observations:**
- **IRM did NOT improve AICD OOD** — test F1 0.293 is within noise of baseline 0.284
- **IRM annealing too aggressive**: jumping from 0 to 5000 in one epoch destabilizes training
- **KL divergence spikes** under IRM: 6.48 → 69.78, suggesting latent space disruption
- **Droid unaffected**: IRM penalty doesn't hurt in-distribution performance
- **Recommendation**: reduce `irm_penalty_max` from 1e4 to 10-100, extend annealing over all epochs

### exp04-irm - AST-IRM variant (experiment/exp04_irm_ast.py)

Full CodeOrigin architecture (proto + contrastive + disentangle) + IRM penalty. Distinct from exp06 which stripped proto/contrastive.

| Benchmark | Task | Best Val F1 | Test Macro-F1 | Test Weighted-F1 |
|-----------|------|-------------|---------------|------------------|
| AICD | T1 | 0.9948 | **0.2779** | 0.2558 |
| Droid | T1 | 0.9689 | **0.9687** | 0.9688 |

**Same IRM explosion:** epoch 3 λ=5000, total loss ~380 (AICD) / ~399 (Droid). IRM penalty ~0.076-0.080.

**Comparison of IRM variants:**

| Variant | Architecture | AICD Test F1 | Droid Test F1 |
|---------|-------------|-------------|---------------|
| exp06 (Exp_DM) | Simple (IRM + disentangle only) | 0.2930 | 0.9685 |
| exp04 (experiment/) | Full (IRM + proto + contrastive + disentangle) | 0.2779 | 0.9687 |
| Baseline (CodeOrigin) | Full (no IRM) | 0.2839 | 0.9718 |

**Insight:** Adding proto+contrastive to IRM actually **hurts** AICD (0.2779 < 0.2930). The simpler exp06 variant performed better, suggesting extra loss components compete with IRM signal.

### exp04-bh - BH-SCM (Exp_DM) Results

Batch-Hard Supervised Contrastive Multi-View: triplet loss + cross-view consistency.

| Benchmark | Task | Best Val F1 | Test Macro-F1 | Test Weighted-F1 |
|-----------|------|-------------|---------------|------------------|
| AICD | T1 | 0.9952 | **0.2811** | 0.2670 |
| Droid | T1 | 0.9698 | **0.9700** | 0.9701 |

**Loss progression (AICD):**

| Epoch | Total | Task | Triplet | Cross-View |
|-------|-------|------|---------|------------|
| 1 | 0.302 | 0.028 | 0.523 | 0.040 |
| 2 | 0.068 | 0.007 | 0.122 | **0.0002** |
| 3 | 0.025 | 0.003 | 0.043 | **0.0002** |

**Key observations:**
- **Cleanest training** of all DM experiments — no loss explosion, stable convergence
- **Cross-view consistency converges instantly** (0.04 → 0.0002 by epoch 2) — token and AST views align trivially
- **Triplet loss decreases steadily** (0.52 → 0.04) — metric learning works, but doesn't help OOD
- AICD test 0.2811 ≈ baseline 0.2839 (within noise)
- Droid test 0.9700 ≈ baseline 0.9718 (within noise)
- **149M params** (vs 153M for CodeOrigin) — simpler architecture, same result

### exp05 - OSCP (Exp_DM) Results

Orthogonal Style-Content Projection + Variance-Invariant Latent Whitening + SupCon.

| Benchmark | Task | Best Val F1 | Test Macro-F1 | Test Weighted-F1 |
|-----------|------|-------------|---------------|------------------|
| AICD | T1 | 0.9938 | **0.2634** | 0.2469 |
| Droid | T1 | 0.9640 | **0.9649** | 0.9649 |

vs CodeOrigin baseline: AICD 0.2634 vs 0.2839 (**-0.021, worse**), Droid 0.9649 vs 0.9718 (-0.007)

**Loss component analysis:**

| Component | Epoch 1 | Epoch 2 | Epoch 3 | Notes |
|-----------|---------|---------|---------|-------|
| Total | 88.3 | 21.9 | 21.9 | Dominated by whitening |
| Task (CE) | 0.105 | 0.044 | 0.035 | Converges normally |
| Ortho (Frobenius) | 61.5 | **0.005** | **0.002** | Converges to near-zero — orthogonality achieved |
| Whiten | 254.0 | **206.0** | **206.0** | Plateaus at ~206, never converges |
| SupCon | 4.19 | 4.14 | 4.14 | Stable, doesn't improve |
| Recon | 0.094 | 0.051 | 0.037 | Converges |

**Key observations:**
- **Whitening loss dominates** (206/21.9 = 94% of total loss) — model spends all capacity trying to push covariance→identity but fails
- **Orthogonal penalty works perfectly** (61.5 → 0.002) — style and content ARE orthogonally separated
- **But orthogonality alone doesn't help OOD** — test F1 0.2634 is worse than baseline 0.2839
- **SupCon stuck at 4.14** — no contrastive improvement after epoch 1
- **OSCP slightly hurts Droid** (0.9649 vs 0.9718) — aggressive regularization reduces in-distribution capacity
- **Recommendation:** reduce `lambda_whiten` drastically (0.1→0.001) or remove whitening entirely, keep only ortho+supcon

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

## exp04 - Multi-View Stacking Ensemble (AICD T1)

- **File:** `experiment/exp04_multiview_ensemble.py`
- **Task:** T1 (binary, 2 classes)
- **Model:** 4-view LightGBM stacking + Logistic Regression meta-learner
- **Train samples:** 200,000 | **Val:** 20,000 | **Test:** 50,000
- **Stacking:** 5-fold CV for meta-feature generation

#### Views and Per-View Performance

| View | Feature Dim | Val Macro-F1 | Test Macro-F1 | Notes |
|------|------------|-------------|---------------|-------|
| Char n-grams | 300 (SVD) | 0.9253 | 0.2606 | ~1000 LightGBM iterations |
| Word n-grams | 300 (SVD) | 0.8704 | 0.2832 | Weakest val, decent test |
| Stylometry | 96 | 0.9914 | 0.2150 | Best val, worst test — classic overfit |
| ModernBERT embeddings | 1,536 (frozen) | 0.9850 | 0.2986 | Best individual test F1 |
| **Stacked Ensemble** | **8 (meta)** | **0.9934** | **0.2053** | **Worse than any individual view on test** |

#### Test Results (Stacked Ensemble)

| Metric | Class 0 | Class 1 | Macro Avg | Weighted Avg |
|--------|---------|---------|-----------|--------------|
| Precision | 0.4654 | 0.1728 | 0.3191 | 0.4008 |
| Recall | 0.0790 | 0.6794 | 0.3792 | 0.2115 |
| F1-score | 0.1351 | 0.2755 | 0.2053 | 0.1660 |
| Support | 38,969 | 11,031 | 50,000 | 50,000 |

#### Key Observations
- **Stacking makes OOD worse:** ensemble (0.2053) < every individual view on test
- **Stylometry paradox:** highest val F1 (0.9914) but lowest test F1 (0.2150) — pure shortcut learning
- **ModernBERT frozen embeddings** are the best individual view on test (0.2986) — pre-trained representations generalize better than handcrafted features
- **Word n-grams** have lowest val F1 (0.8704) but second-best test F1 (0.2832) — less overfitting = better OOD
- **Negative transfer in stacking:** meta-learner learns to trust overfit views (stylometry) over generalizable ones (word n-grams), degrading ensemble test performance
- **Same class 0 recall catastrophe:** 0.0790 — model overwhelmingly predicts class 1 on test

#### Insight: Inverse val-test correlation
| View | Val F1 | Test F1 | Gap |
|------|--------|---------|-----|
| Stylometry | 0.9914 | 0.2150 | -0.776 |
| ModernBERT | 0.9850 | 0.2986 | -0.686 |
| Char n-gram | 0.9253 | 0.2606 | -0.665 |
| Word n-gram | 0.8704 | 0.2832 | -0.587 |

**Pattern:** Higher val F1 → worse test F1. This confirms that in-distribution performance actively harms OOD generalization on AICD T1.

---

## exp03 - Frozen Encoder + Group DRO (AICD T1)

- **File:** `experiment/exp03_frozen_dro.py`
- **Task:** T1 (binary, 2 classes)
- **Model:** ModernBERT-base (frozen encoder, trainable head only)
- **Total params:** 151,376,130 | **Trainable:** 2,361,858 (1.6%)
- **Train samples:** 200,000
- **Epochs:** 10 (log truncated at epoch 4)
- **Batch size:** 32 x 2 = 64

#### Configuration
| Setting | Value |
|---------|-------|
| DRO alpha | 0.2 |
| Augmentation prob | 0.5 |
| Extract layers | [-1, -4, -8] (multi-layer feature extraction) |
| FP16 | True |
| Encoder | Frozen (only classification head trains) |

#### Training Progression

| Epoch | Total Loss | ERM | DRO | Worst Group | Val Loss | Val Macro-F1 | Best? |
|-------|-----------|-----|-----|-------------|----------|-------------|-------|
| 1 | 0.2925 | 0.2968 | 0.2755 | 0.4714 | 0.1002 | 0.9729 | * |
| 2 | 0.2526 | 0.2547 | 0.2438 | 0.3912 | 0.0959 | 0.9802 | |
| 2 (mid) | - | - | - | - | 0.0951 | 0.9822 | * |
| 3 | 0.2298 | 0.2316 | 0.2224 | 0.3388 | 0.0759 | 0.9857 | ** |
| 4 (partial) | ~0.226 | ~0.227 | ~0.220 | ~0.322 | 0.0933 | 0.9826 | |

- **Best Val Macro-F1:** 0.9857 (epoch 3 end)
- **Test Results:** *pending (log truncated at epoch 4)*

#### Group Weight Evolution
DRO group weights shift dramatically during training:
- **Epoch 1:** `[0.20, 0.18, 0.25, 0.13, 0.12, 0.12]` → `[0.59, 0.01, 0.40, 0.00, 0.00, 0.00]`
- **Epoch 3:** `[0.23, 0.00, 0.77, 0.00, 0.00, 0.00]`
- **Epoch 4:** `[0.15, 0.00, 0.85, 0.00, 0.00, 0.00]`

**Interpretation:** DRO collapses to 2 dominant groups (groups 0 and 2), with groups 1/3/4/5 getting zero weight. This means DRO is effectively only optimizing for 2 subgroups, not distributing robustness broadly.

#### Key Observations
- **Lower val F1 ceiling** (0.9857) vs full fine-tune baselines (~0.995) — frozen encoder limits capacity
- **DRO group collapse:** by epoch 2, only 2 of 6 groups retain non-zero weight — DRO fails to maintain broad subgroup coverage
- **Worst-group loss decreasing** (0.47 → 0.34) suggests some worst-case improvement, but group collapse undermines the theoretical benefit
- **Training is fast** (~8s/100 steps vs ~24s for full fine-tune) due to frozen encoder
- **Val loss decreasing steadily** (0.1002 → 0.0759) but val-test gap pattern likely persists

---

## exp02 - Code Stylometry + LightGBM (AICD T1)

- **File:** `experiment/exp02_stylometric_gbm.py`
- **Task:** T1 (binary, 2 classes)
- **Model:** LightGBM (non-neural baseline)
- **Train samples:** 200,000

#### Feature Engineering
| Feature Type | Dimensionality | Notes |
|-------------|---------------|-------|
| Stylometric features | 140 | Code-specific structural/style metrics |
| Char TF-IDF → SVD | 500 | Explained variance: 0.674 |
| Word TF-IDF → SVD | 500 | Explained variance: 0.461 |
| **Total** | **1,140** | |

#### Results
| Split | Macro-F1 | Class 0 F1 | Class 1 F1 | Accuracy |
|-------|----------|-----------|-----------|----------|
| Val | 0.9918 | 0.9915 | 0.9922 | 0.9919 |
| Test | **0.2078** | 0.1401 | 0.2755 | 0.2136 |

- **Best LightGBM iteration:** 212 (early stopping, patience 100)
- **Top feature:** `stylo_20` (gain 2,063,632) — dominates by 10x over #2

#### Key Observations
- **Worst test F1 across all experiments** (0.2078 < exp00's 0.2625)
- Non-neural approach confirms: **OOD gap is not a neural network problem** — it's a fundamental distribution shift
- Val F1 0.9918 vs Test F1 0.2078 = same overfitting pattern as neural methods
- Class 0 (human) recall catastrophically low: 0.0822 — model predicts class 1 for most test samples
- Test class imbalance: 38,969 (class 0) vs 11,031 (class 1) = 78:22
- **Stylometric features overfit to training distribution** just like neural embeddings
- Feature importance heavily concentrated (stylo_20 alone = 70%+ of total gain) → model relies on a single shortcut feature

---

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

## exp01-simple - Simple ModernBERT Baseline with OOD Regularization (AICD T1)

- **File:** `experiment/exp01_simple_baseline.py`
- **Task:** T1 (binary, 2 classes)
- **Backbone:** `answerdotai/ModernBERT-base` (149.6M params, all trainable)
- **Hardware:** Kaggle GPU, FP16
- **Batch size:** 16 x 4 = 64 (effective)
- **Max length:** 1024 (vs 512 in prior experiments)
- **Train samples:** 200,000 (vs 100k in prior experiments)
- **Epochs:** 3

#### OOD Regularization Techniques
| Technique | Setting |
|-----------|---------|
| R-Drop (KL divergence) | α = 5.0 |
| Label smoothing | 0.1 |
| Multi-sample dropout | K = 5 |
| SWA (Stochastic Weight Averaging) | start epoch 2 |

#### Training Loss Progression

| Epoch | Final Total Loss | Final CE | Final KL | Val Loss | Val Macro-F1 | Best? |
|-------|-----------------|----------|----------|----------|-------------|-------|
| 1 | 0.2467 | 0.2375 | 0.0018 | 0.0587 | 0.9946 | * |
| 2 | 0.2067 | 0.2062 | 0.0001 | 0.0601 | 0.9944 (end) | |
| 2 (mid) | - | - | - | 0.0568 | 0.9959 | ** (best) |
| 3 | 0.2038 | 0.2035 | 0.0001 | 0.0590 | 0.9955 | |

- **Best Val Macro-F1:** 0.9959 (epoch 2, mid-epoch eval)
- **Test Results:** *pending (log truncated)*

#### Key Observations
- **R-Drop KL drops quickly:** from 0.0226 → 0.0001 by epoch 2, suggesting R-Drop regularization effect fades as model converges
- **CE loss plateau:** epoch 2 CE ~0.2062, epoch 3 CE ~0.2035 — minimal improvement after epoch 2
- **Val F1 plateau:** val F1 oscillates 0.9944-0.9959, same overfitting pattern as prior exps
- **SWA activated epoch 2** but val loss slightly increased at epoch end (0.0601 vs 0.0587)
- **2x training data (200k)** and **2x sequence length (1024)** did not break the OOD pattern
- **Label smoothing** may help test generalization — need test results to confirm

#### Comparison with baseline
| Metric | exp00 CodeOrigin | exp01-simple |
|--------|-----------------|-------------|
| Train samples | 100k | 200k |
| Max length | 512 | 1024 |
| Best Val F1 | 0.9948 | 0.9959 |
| Regularization | Style-content disentangle + GRL | R-Drop + label smooth + SWA |
| Test F1 | 0.2625 | *pending* |

---

## Next modeling plan

- Train 2 independent tracks:
  - Track A: `benchmark=aicd` (primary AICD leaderboard).
  - Track B: `benchmark=droid` (primary Droid benchmark).
- For each track:
  - standard run + comment-masked ablation
  - length-bucket evaluation
  - confusion matrix on hardest class pairs
