# DM Tracker - Deep Methods Experiment Suite

## Overview

Advanced architectural experiments targeting **NeurIPS 2026 ORAL**.
All methods build on `exp00_codeorigin.py` (CodeOrigin baseline) with novel components from the Gemini DeepResearch portfolio.

**Protocol:**
- Train/evaluate **separate models per benchmark** (AICD-Bench / DroidCollection).
- Each file is **standalone** and Kaggle-runnable (`python exp0X_xxx.py`).
- **Paper-aligned metrics by benchmark:**
  - **AICD-Bench:** `Macro-F1` = primary.
  - **DroidCollection:** `Weighted-F1` = primary (report `Macro-F1` as auxiliary), and include recall-focused view for adversarial robustness.
- Always export full metric pack to logs/markdown: `Primary`, `Macro-F1`, `Weighted-F1`, `Macro-Recall`, `Weighted-Recall`, `Accuracy`, `per-class report`.
- Hardware target: Kaggle H100 80GB, BF16, effective batch 64.

---

## Experiment Registry

| Exp | Method | Core Innovation | Loss Formula | Tier | Status |
|-----|--------|----------------|--------------|------|--------|
| exp01 | **CausAST** | Orthogonal covariance penalty (Frobenius norm) between token & AST views + Batch-Hard Triplet | `L_CE + λ₁·L_Triplet + λ₂·‖Cov(H_tok,H_ast)‖²_F` | A (Flagship 1) | Pending |
| exp02 | **TTA-Evident** | Evidential Deep Learning (Dirichlet head) + Test-Time MLM Adaptation + ECE/Brier | `L_EDL + KL(Dir‖Dir_prior) + L_disentangle` | A (Flagship 2) | Pending |
| exp03 | **AP-NRL** | SupCon Loss + Code Augmenter (humanization simulation) + dual-view consistency | `L_CE + λ_supcon·L_SupCon + λ_consist·L_Consist` | A (Flagship 3) | Pending |
| exp04 | **BH-SCM** | Batch-Hard Triplet on multi-view + Cross-View Consistency (token-AST alignment) | `L_CE + λ_triplet·L_BH + λ_xview·L_CrossView` | A (Must-Run) | Pending |
| exp05 | **OSCP** | Frobenius norm orthogonal penalty + Variance-Invariant Latent Whitening + SupCon | `L_CE + λ_ortho·‖Cov‖²_F + λ_whiten·L_W + λ_supcon·L_SupCon` | A (Must-Run) | Pending |
| exp06 | **AST-IRM** | Invariant Risk Minimization across language environments + annealed IRM penalty | `Σ_e L_CE(e) + λ_irm·Σ_e ‖∇_{w=1}L_CE(e)·w‖²` | B (High Novelty) | Pending |
| exp07 | **DomainMix** | Comment stripping + variable normalization + embedding Mixup. Attacks domain shortcuts directly | `L_focal + λ_mixup·L_mixup` | A+ (Paper-guided) | Done (suite 5 runs) |
| exp08 | **MoE-Code** | Mixture-of-Experts head (K=4, top-2 routing) with load balancing. Specialized expert per domain/language | `L_focal + λ_balance·L_load_balance` | A+ (Architectural) | Done (suite 5 runs) |
| exp09 | **TokenStat** | Token distribution statistics (entropy, burstiness, TTR, Yule-K) + neural dual-head with soft gate | `L_focal (gated) + 0.3·L_neural + 0.3·L_stat` | A+ (Paper-guided) | Done (suite 5 runs) |
| exp10 | **MetaDomain** | Reptile meta-learning: languages-as-tasks, learns domain-agnostic initialization | `Reptile: θ += β·(θ' - θ)` | S (Paradigm shift) | Pending |
| exp11 | **SpectralCode** | FFT spectral analysis on token sequences. Multi-scale (32-256) frequency band features | `L_focal (gated) + 0.3·L_neural + 0.3·L_spectral` | S (Cross-domain transfer) | Done (suite 5 runs) |
| exp12 | **WatermarkStat** | Watermark-detection-inspired: green-list proxy, n-gram entropy, Zipf deviation, chi-sq | `L_focal (gated) + 0.3·L_neural + 0.3·L_wmark` | S (Cross-domain transfer) | Pending |
| exp13 | **SlotCode** | Slot Attention decomposes code into K structural slots. Slot consistency = detection signal | `L_task + 0.3·L_agg + 0.2·L_consist` | S (Object-centric) | Done (suite 5 runs) |
| exp18 | **HierTreeCode** | Hierarchical affinity constraint on latent groups (task-dependent; inactive on binary) with spectral-neural backbone | `L_task + 0.3·L_neural + 0.3·L_spectral (+ L_hier)` | S (Cross-domain stress-test) | Done (suite 5 runs) |
| exp19 | **KANCode** | KAN (B-spline learnable activations) replaces MLP heads. Captures nonlinear boundaries between human/AI code | `L_focal + 0.3·L_neural_kan + 0.3·L_spectral_kan` | S+ (ICLR 2025 Oral arch) | Pending |
| exp20 | **HyperCode** | Poincaré ball embeddings for hierarchy-aware classification. Centering loss organizes Human→AI family tree | `L_focal + 0.3·L_hyper_dist + 0.2·L_centering + 0.3·L_spectral` | S (NeurIPS 2024 geometry) | Pending |
| exp21 | **IBCode** | Variational Information Bottleneck compresses away domain shortcuts. Anti-OOD collapse | `L_focal + β·L_ib_kl + 0.3·L_neural + 0.3·L_spectral` | A+ (Anti-shortcut) | Pending |
| exp22 | **TTLCode** | Test-Time LoRA: adapts backbone at test time via MLM on unlabeled batches. OOD fix | `L_focal + 0.3·L_neural + 0.3·L_spectral (+ L_mlm@test)` | A+ (ICML 2025 OOD) | Pending |
| exp23 | **TopoCode** | Topological persistence (Betti numbers, H0/H1) from AST filtration graphs | `L_focal + 0.3·L_neural + 0.3·L_topo` | S (NeurIPS 2025 TDA) | Pending |
| exp24 | **MambaCode** | Selective SSM (Mamba) + MoE routing replaces cross-attention fusion. O(n) complexity | `L_focal + λ_balance·L_load + 0.2·L_ssm_aux + 0.3·L_spectral` | S+ (ICLR 2025 SSM) | Pending |
| exp25 | **EnergyCode** | Energy-based OOD detection. Energy margin training with pseudo-OOD noise injection | `L_focal + λ_energy·L_energy_margin + 0.3·L_neural + 0.3·L_spectral` | A+ (ICLR 2020/2025 OOD) | Pending |
| exp26 | **WaveCLCode** | Discrete Wavelet Transform (Haar) + class-aware frequency band selection | `L_focal + 0.3·L_neural + 0.3·L_wavelet` | S (NeurIPS 2024 freq) | Pending |
| exp27 | **DeTeCtiveCode** | Exp18 backbone + **multi-level SupCon** (neural + spectral heads) + optional **kNN blend** at test (same recipe as CoDET `run_codet_m4_exp27_detective.py`) | `L_task + 0.3L_n + 0.3L_s + λ_hier L_hier + λ_sup SupCon` | S (3-bench unified method) | Pending run |
| exp28 | **HardNegCode** | Exp27 variant: stronger supervised contrastive pressure and larger contrast head, no retrieval blend | `L_task + 0.3L_n + 0.3L_s + λ_hier L_hier + 0.20·SupCon` | S (representation-focused) | Pending run |
| exp29 | **RetrievalCalibCode** | Exp27 variant: lighter SupCon + stronger kNN blending (`k=48`, `alpha=0.35`, larger bank) | `L_task + 0.3L_n + 0.3L_s + λ_hier L_hier + 0.08·SupCon + kNN blend` | S (retrieval-calibration) | Pending run |
| exp30 | **HierFocusCode** | Exp27 variant: stronger family-tree constraints (`lambda_hier=0.55`, `margin=0.40`) | `L_task + 0.3L_n + 0.3L_s + 0.55·L_hier + λ_sup SupCon` | S (hierarchy-focused) | Pending run |

**3-bench unified protocol (CoDET-M4 + AICD + Droid):** train `run_codet_m4_exp27_detective.py` (IID binary + author; optional full suite) and `exp27_detective_code.py` with `RUN_MODE=full` so one method family is evaluated on all three benchmarks.

**Additional 3-bench variants ready:** `run_codet_m4_exp28_hardneg.py` + `exp28_hardneg_code.py`, `run_codet_m4_exp29_retrievalcalib.py` + `exp29_retrievalcalib_code.py`, `run_codet_m4_exp30_hierfocus.py` + `exp30_hierfocus_code.py`.

---

## Baseline Reference (from exp00 CodeOrigin)

| Benchmark | Task | Best Val F1 | Test F1 | Test Weighted F1 | Notes |
|-----------|------|-------------|---------|------------------|-------|
| AICD | T1 | 0.9954 | 0.2877 | 0.2721 | Severe OOD collapse (val-test gap) |
| DROID | T1 | 0.9693 | 0.9708 | 0.9709 | Strong in-distribution performance |

**Key bottleneck:** AICD T1 OOD robustness. All DM experiments target this gap.

---

## Results

### AICD-Bench (Benchmark A, Primary = Macro-F1)

| Exp | Method | Task | Epochs | Best Val F1 | Test Macro-F1 | Test Weighted-F1 | ECE | Brier | Notes |
|-----|--------|------|--------|-------------|---------------|------------------|-----|-------|-------|
| baseline | CodeOrigin | T1 | 3 | 0.9954 | 0.2877 | 0.2721 | - | - | Shortcut learning under OOD |
| exp01 | CausAST | T1 | 3 | 0.9949 | 0.2753 | 0.2574 | - | - | Ortho→0.0002. Worst DM AICD result |
| exp02 | TTA-Evident | T1 | 3 | - | - | - | - | - | |
| exp03 | AP-NRL | T1 | 3 | 0.9951 | 0.2842 | 0.2650 | - | - | SupCon plateau ~4.18. Consistency→0. Baseline-level |
| exp04 | BH-SCM | T1 | 3 | 0.9952 | 0.2811 | 0.2670 | - | - | Clean training. xview→0. Baseline-level |
| exp05 | OSCP | T1 | 3 | 0.9938 | 0.2634 | 0.2469 | - | - | Whitening loss dominates (~206). Worse than baseline (-0.02) |
| exp06 | AST-IRM | T1 | 3 | 0.9944 | 0.2930 | 0.2725 | - | - | IRM penalty explodes epoch 3 (λ=5000). No OOD gain vs baseline |
| exp07 | DomainMix | T1/T2/T3 | 3 | 0.9880 / 0.0965 / 0.6443 | 0.3088 / 0.1048 / 0.4454 | 0.2755 / 0.0935 / 0.5641 | - | - | Severe val-test gap on AICD (esp. T1,T2); best AICD test on T3 |
| exp08 | MoE-Code | T1/T2/T3 | 3 | 0.9946 / 0.1467 / 0.7676 | 0.2825 / 0.1455 / 0.5440 | 0.2683 / 0.1250 / 0.6286 | - | - | Better than exp07 on T2/T3, but AICD T1 OOD gap remains severe |
| exp09 | TokenStat | T1/T2/T3 | 3 | 0.9955 / 0.1747 / 0.7813 | 0.2589 / 0.1787 / 0.5521 | 0.2312 / 0.1666 / 0.6455 | - | - | Strong gains on AICD T2/T3 vs exp08; T1 still severe OOD collapse |
| exp10 | MetaDomain | T1 | 3 | - | - | - | - | - | Reptile meta-learning |
| exp11 | SpectralCode | T1/T2/T3 | 3 | 0.9956 / 0.1785 / 0.7848 | 0.2983 / 0.1893 / 0.5631 | 0.2918 / 0.1797 / 0.6524 | - | - | Best AICD T1 among exp08/09/11; T2/T3 slightly stronger than exp09 |
| exp12 | WatermarkStat | T1 | 3 | - | - | - | - | - | Watermark-inspired stats |
| exp13 | SlotCode | T1/T2/T3 | 3 | 0.9951 / 0.1485 / 0.7620 | 0.2569 / 0.1596 / 0.5706 | 0.2156 / 0.1571 / 0.6404 | - | - | Improves AICD T3 but underperforms exp11 on T1/T2 and keeps large T1 OOD gap |
| exp18 | HierTreeCode | T1/T2/T3 | 3 | 0.9949 / 0.2543 / 0.7783 | 0.2572 / 0.2071 / 0.5502 | 0.2235 / 0.1802 / 0.6407 | - | - | Strongest AICD T2 so far; T1 still catastrophic val-test collapse |

### DroidCollection (Benchmark B, Primary = Weighted-F1)

| Exp | Method | Task | Epochs | Best Val (Primary) | Test Primary (Weighted-F1) | Test Macro-F1 | Notes |
|-----|--------|------|--------|--------------------|-----------------------------|---------------|-------|
| baseline | CodeOrigin | T1 | 3 | 0.9693 | 0.9709 | 0.9708 | Strong baseline |
| exp01 | CausAST | T1 | 3 | 0.9672 | 0.9687 | 0.9686 | Droid stable |
| exp02 | TTA-Evident | T1 | 3 | - | - | - | |
| exp03 | AP-NRL | T1 | 3 | 0.9702 | 0.9699 | 0.9699 | Droid stable |
| exp04 | BH-SCM | T1 | 3 | 0.9698 | 0.9701 | 0.9700 | Stable, ngang baseline |
| exp05 | OSCP | T1 | 3 | 0.9640 | 0.9649 | 0.9649 | Slightly below baseline. Whitening hurts capacity |
| exp06 | AST-IRM | T1 | 3 | 0.9674 | 0.9685 | 0.9685 | Stable. IRM penalty ~0.073 |
| exp07 | DomainMix | T3/T4 | 3 | 0.7273 / 0.7071 | 0.7930 / 0.7706 | 0.7278 / 0.7036 | Stable and strong on Droid; close val-test alignment |
| exp08 | MoE-Code | T3/T4 | 3 | 0.8504 / 0.8460 | 0.8914 / 0.8785 | 0.8526 / 0.8454 | Strongest so far on Droid; val-test alignment is very close |
| exp09 | TokenStat | T3/T4 | 3 | 0.8574 / 0.8478 | 0.8941 / 0.8815 | 0.8556 / 0.8488 | Very strong and stable on Droid; best among current methods on T3/T4 |
| exp10 | MetaDomain | T1 | 3 | - | - | - | |
| exp11 | SpectralCode | T3/T4 | 3 | 0.8500 / 0.8494 | 0.8877 / 0.8802 | 0.8473 / 0.8467 | Strong on Droid but slightly below exp09 on both tasks |
| exp12 | WatermarkStat | T1 | 3 | - | - | - | |
| exp13 | SlotCode | T3/T4 | 3 | 0.8429 / 0.8388 | 0.8821 / 0.8679 | 0.8432 / 0.8337 | Solid Droid performance but below exp09/exp11 across both tasks |
| exp18 | HierTreeCode | T3/T4 | 3 | 0.8560 / 0.8474 | 0.8917 / 0.8793 | 0.8531 / 0.8453 | Very strong and stable on Droid; runner-up on T3, below exp09 on aggregate |

---

### Droid Paper Head-to-Head (Weighted-F1, Table 3/4 alignment)

Reference from `paper_Droid.md`:
- Droid paper reports **Weighted-F1** as the main table metric (2-class/3-class setups).
- Closest comparable line to our DM multi-class Droid runs is **3-class Full Training avg Weighted-F1**.

| Model/System | Droid paper 3-class Weighted-F1 (Avg) | Our closest run (Droid T3 Weighted-F1) | Delta (ours - paper) |
|---|---:|---:|---:|
| M4FT | 0.7350 | 0.8941 (exp09 TokenStat) | +0.1591 |
| GPT-SnifferFT | 0.8275 | 0.8941 (exp09 TokenStat) | +0.0666 |
| CoDet-M4FT | 0.8325 | 0.8941 (exp09 TokenStat) | +0.0616 |
| DroidDetectCLS-Base | 0.8676 | 0.8941 (exp09 TokenStat) | +0.0265 |
| DroidDetectCLS-Large | 0.8878 | 0.8941 (exp09 TokenStat) | +0.0063 |

Notes for paper writing:
- This is a **near-aligned** comparison (both Weighted-F1 centered), but not a perfect protocol match because our tracker currently logs task-level `T3/T4` while Droid table is averaged by domain/split setting.
- For strict camera-ready comparison, keep both:
  - **Main**: paper-protocol metric/setting match.
  - **Supplementary**: our full metric pack (`Macro/Weighted F1`, `Macro/Weighted Recall`, `Accuracy`) exported by DM scripts.

---

## Leaderboard (Current Runs)

Methods included: `exp07`, `exp08`, `exp09`, `exp11`, `exp13`, `exp18` (all with 5-run suite done).

| Rank | Task | Best Method | Test Macro-F1 | Runner-up | Delta |
|------|------|-------------|---------------|-----------|-------|
| 1 | AICD T1 | exp07 DomainMix | 0.3088 | exp11 SpectralCode (0.2983) | +0.0105 |
| 1 | AICD T2 | exp18 HierTreeCode | 0.2071 | exp11 SpectralCode (0.1893) | +0.0178 |
| 1 | AICD T3 | exp13 SlotCode | 0.5706 | exp11 SpectralCode (0.5631) | +0.0075 |
| 1 | DROID T3 | exp09 TokenStat | 0.8556 | exp08 MoE-Code (0.8526) | +0.0030 |
| 1 | DROID T4 | exp09 TokenStat | 0.8488 | exp11 SpectralCode (0.8467) | +0.0021 |

| Rank | Method | Avg Macro-F1 (5 tasks) | Avg AICD (T1/T2/T3) | Avg DROID (T3/T4) | Status |
|------|--------|--------------------------|----------------------|-------------------|--------|
| 1 | exp11 SpectralCode | 0.5489 | 0.3502 | 0.8470 | Best overall currently |
| 2 | exp18 HierTreeCode | 0.5426 | 0.3382 | 0.8492 | Best AICD T2; strong Droid stability |
| 3 | exp09 TokenStat | 0.5388 | 0.3299 | 0.8522 | Best on Droid aggregate |
| 4 | exp08 MoE-Code | 0.5340 | 0.3240 | 0.8490 | Strong, balanced |
| 5 | exp13 SlotCode | 0.5328 | 0.3290 | 0.8385 | Best AICD T3 single-task |
| 6 | exp07 DomainMix | 0.4581 | 0.2863 | 0.7157 | Baseline among new methods |

Quick take:
- If you want **best overall now** -> `exp11 SpectralCode`.
- If you want **best AICD T2 single-task** -> `exp18 HierTreeCode`.
- If you want **best Droid robustness** -> `exp09 TokenStat`.
- For **AICD T1 OOD bottleneck**, no method solves it yet (all still show large val-test collapse).

---

## Method Details

### exp01 - CausAST (Flagship 1)

**Paper claim:** "Disentangling AST structural dynamics from token semantics mathematically improves cross-language generalization."

- Dual-stream: ModernBERT (tokens) + BiLSTM AST encoder (structure)
- `OrthogonalProjection`: projects token and AST features into orthogonal subspaces
- `BatchHardTripletLoss`: mines hardest pos/neg per anchor in batch
- Frobenius norm penalty: `‖Cov(H_tok, H_ast)‖²_F → 0` forces statistical independence
- Key hyperparams: `lambda_triplet=0.5`, `lambda_ortho=1.0`, `triplet_margin=0.3`

### exp02 - TTA-Evident (Flagship 2)

**Paper claim:** "Standard softmax detectors are poorly calibrated; EDL + TTA provides superior epistemic uncertainty."

- `EvidentialHead`: outputs Dirichlet parameters `α = softplus(f(x)) + 1`
- `edl_loss`: Type II Maximum Likelihood + annealed KL to uniform Dirichlet prior
- Test-Time Adaptation: 1 step of masked token prediction (MLM) on test batch
- Calibration metrics: ECE (15-bin), Brier Score
- Key hyperparams: `edl_lambda_kl=0.1`, `tta_lr=1e-5`, `tta_mlm_prob=0.15`

### exp03 - AP-NRL (Flagship 3)

**Paper claim:** "Superficial humanization edits don't alter the generative manifold."

- `CodeAugmenter`: simulates adversarial humanization (rename vars, reformat, add comments)
- `SupConLoss`: supervised contrastive with hard negative mining
- Dual-view: original + augmented code → contrastive alignment
- Consistency loss: `1 - cos_sim(proj_z, aug_proj_z)`
- Key hyperparams: `lambda_supcon=0.5`, `lambda_consistency=0.3`, `augment_prob=0.3`

### exp04 - BH-SCM (Tier A Must-Run)

**Paper claim:** "Multi-view contrastive learning creates latent space robust to token perturbations."

- `CrossViewConsistency`: aligns token and AST views in shared projection space
- `BatchHardTripletLoss`: batch-hard mining on fused multi-view embedding
- Concat fusion: `[token_repr; ast_repr; struct_repr]` → classifier
- Key hyperparams: `lambda_triplet=0.5`, `lambda_cross_view=0.3`, `triplet_margin=0.3`

### exp05 - OSCP (Tier A Must-Run)

**Paper claim:** "Strict orthogonal projection isolates style from content, preventing shortcut learning."

- `compute_ortho_loss`: Frobenius norm of cross-covariance between z_style and z_content
- `compute_whitening_loss`: push style covariance toward identity matrix (VILW)
- SupCon on style space for discriminative clustering
- Key hyperparams: `lambda_ortho=1.0`, `lambda_whiten=0.1`, `lambda_supcon=0.3`

### exp06 - AST-IRM (Tier B High Novelty)

**Paper claim:** "IRM across language environments forces causal, language-agnostic detection features."

- Languages as environments: heuristic inference or dataset field
- `compute_irm_penalty`: IRMv1 `‖∇_{w=1} CE(w·logits, y)‖²` per environment
- Annealed schedule: pure ERM for first epoch, then ramp IRM penalty
- NaN safety: fallback to focal CE if IRM causes instability
- Key hyperparams: `lambda_irm=1.0`, `irm_anneal_epochs=1`, `irm_penalty_max=1e4`

### exp07 - DomainMix (Tier A+ Paper-guided)

**Run timestamp:** `2026-03-30 06:38:21` (full suite done)

- Preflight passed for 5 runs on H100 BF16: `aicd_T1`, `aicd_T2`, `aicd_T3`, `droid_T3`, `droid_T4`
- AICD final test Macro-F1: `T1=0.3088`, `T2=0.1048`, `T3=0.4454`
- Droid final test Macro-F1: `T3=0.7278`, `T4=0.7036`
- Notable behavior: very high val on AICD T1 (`0.9880`) but low test (`0.3088`) indicates unresolved OOD shortcut issue
- Overall: DomainMix is robust on Droid, but does not solve AICD OOD generalization (except partial gain on T3)

### exp08 - MoE-Code (Tier A+ Architectural)

**Run timestamp:** `2026-03-30 11:05:45` (full suite done)

- Preflight + full suite completed for 5 runs on H100 BF16: `aicd_T1`, `aicd_T2`, `aicd_T3`, `droid_T3`, `droid_T4`
- AICD final test Macro-F1: `T1=0.2825`, `T2=0.1455`, `T3=0.5440`
- Droid final test Macro-F1: `T3=0.8526`, `T4=0.8454`
- Best Val Macro-F1: `AICD_T1=0.9946`, `AICD_T2=0.1467`, `AICD_T3=0.7676`, `DROID_T3=0.8504`, `DROID_T4=0.8460`
- Overall: MoE routing substantially improves Droid and lifts AICD T2/T3 vs exp07, but still fails to close the AICD T1 OOD val-test gap

### exp09 - TokenStat (Tier A+ Paper-guided)

**Run timestamp:** `2026-03-30 11:17:24` (full suite done)

- Preflight + full suite completed for 5 runs on H100 BF16: `aicd_T1`, `aicd_T2`, `aicd_T3`, `droid_T3`, `droid_T4`
- AICD final test Macro-F1: `T1=0.2589`, `T2=0.1787`, `T3=0.5521`
- Droid final test Macro-F1: `T3=0.8556`, `T4=0.8488`
- Best Val Macro-F1: `AICD_T1=0.9955`, `AICD_T2=0.1747`, `AICD_T3=0.7813`, `DROID_T3=0.8574`, `DROID_T4=0.8478`
- Overall: Token statistics + gated fusion improves AICD T2/T3 and sets strongest Droid performance so far, but AICD T1 OOD shortcut issue remains unresolved

### exp11 - SpectralCode (Tier S Cross-domain transfer)

**Run timestamp:** `2026-03-30 11:16:35` (full suite done)

- Preflight + full suite completed for 5 runs on H100 BF16: `aicd_T1`, `aicd_T2`, `aicd_T3`, `droid_T3`, `droid_T4`
- AICD final test Macro-F1: `T1=0.2983`, `T2=0.1893`, `T3=0.5631`
- Droid final test Macro-F1: `T3=0.8473`, `T4=0.8467`
- Best Val Macro-F1: `AICD_T1=0.9956`, `AICD_T2=0.1785`, `AICD_T3=0.7848`, `DROID_T3=0.8500`, `DROID_T4=0.8494`
- Overall: spectral features give the strongest AICD performance so far (especially T1/T3), but Droid remains slightly better with TokenStat and the AICD T1 OOD gap is still present

### exp13 - SlotCode (Tier S Object-centric)

**Run timestamp:** `2026-03-30 11:07:11` (full suite done)

- Preflight + full suite completed for 5 runs on H100 BF16: `aicd_T1`, `aicd_T2`, `aicd_T3`, `droid_T3`, `droid_T4`
- AICD final test Macro-F1: `T1=0.2569`, `T2=0.1596`, `T3=0.5706`
- Droid final test Macro-F1: `T3=0.8432`, `T4=0.8337`
- Best Val Macro-F1: `AICD_T1=0.9951`, `AICD_T2=0.1485`, `AICD_T3=0.7620`, `DROID_T3=0.8429`, `DROID_T4=0.8388`
- Overall: slot decomposition helps on AICD T3, but ranking is below SpectralCode/TokenStat on most tasks and does not address AICD T1 OOD collapse

### exp18 - HierTreeCode (Cross-domain stress-test)

**Run timestamp:** `2026-04-01 05:24:40 → 07:22:11` (full suite done, ~2h on H100 BF16)

- Preflight + full suite completed for 5 runs: `aicd_T1`, `aicd_T2`, `aicd_T3`, `droid_T3`, `droid_T4`
- AICD final test Macro-F1: `T1=0.2572`, `T2=0.2071`, `T3=0.5502`
- Droid final test Macro-F1: `T3=0.8531`, `T4=0.8453`
- Best Val Macro-F1: `AICD_T1=0.9949`, `AICD_T2=0.2543`, `AICD_T3=0.7783`, `DROID_T3=0.8560`, `DROID_T4=0.8474`
- Overall: strongest AICD T2 so far, strong Droid stability, but AICD T1 remains severe val-test collapse (0.9949 → 0.2572)
- Params (task-dependent head): T1=151.74M / T2=151.75M / T3=151.74M / Droid-T3=151.74M

#### Hier loss activity (confirms design correctness)

| Task | # Classes | Hier ep1 | Hier ep3 | Δ | Behavior |
|------|-----------|---------|---------|---|----------|
| AICD T1 | 2 (binary) | **0.0000** | **0.0000** | 0% | ✅ **Correctly disabled** (family tree inactive on binary) |
| AICD T2 | 12 | 0.3643 | 0.1944 | **-47%** | ✅ Strongest convergence; family constraint actively organizing |
| AICD T3 | 4 | 0.3946 | 0.3072 | -22% | ✅ Moderate convergence |
| DROID T3 | 3 | 0.3798 | 0.3035 | -20% | ✅ Converges to plateau |

#### AICD T2 — per-class test F1 (12-class family attribution)

| Class | Support | F1 | Status |
|:------|--------:|:--:|:-------|
| 0 | 23 870 | **0.0000** | ⚠️ **Full collapse** (majority class; never predicted) |
| 10 | 10 260 | **0.5935** | Strongest |
| 11 | 810 | 0.3340 | — |
| 9 | 1 407 | 0.2579 | — |
| 5 | 2 105 | 0.2531 | — |
| 3 | 853 | 0.2398 | — |
| 6 | 2 062 | 0.2237 | — |
| 2 | 2 617 | 0.1799 | — |
| 4 | 381 | 0.1430 | — |
| 8 | 1 110 | 0.1372 | — |
| 7 | 3 593 | 0.1110 | — |
| 1 | 932 | 0.0121 | ⚠️ Near-zero (minority + LLM confusion) |

> **Key issue:** class 0 (47.7% of test) is never predicted → severe minority class bias from class-weighted focal + label imbalance. Macro-F1 (0.2071) ≪ weighted-F1 (0.1802) ∴ improvement requires rebalancing, not more architecture.

#### AICD T3 — per-class test F1 (4-class: human/machine/hybrid/adversarial)

| Class | Support | Precision | Recall | F1 |
|:------|--------:|:---------:|:------:|:--:|
| 0 (human) | 33 668 | 0.9879 | 0.5852 | **0.7350** |
| 1 | 5 841 | 0.4030 | 0.6828 | 0.5068 |
| 3 (adversarial) | 3 198 | 0.5248 | 0.8580 | **0.6512** |
| 2 (hybrid) | 7 293 | 0.2289 | 0.4685 | **0.3075** ← weakest |

> Human separation strong (0.7350); **hybrid class (class 2) is the bottleneck** — adversarial code is easier to flag than mixed human-AI authorship.

---

## Execution Priority

1. **exp01 CausAST** (AICD T1) - core architectural claim
2. **exp02 TTA-Evident** (AICD T1) - calibration + TTA claim
3. **exp05 OSCP** (AICD T1) - orthogonal disentanglement ablation
4. **exp03 AP-NRL** (DROID T1) - adversarial robustness claim
5. **exp04 BH-SCM** (AICD T1) - multi-view contrastive baseline
6. **exp06 AST-IRM** (AICD T1) - causal invariance (high risk/reward)

---

## Ablation Plan (after main runs)

| Ablation | What it tests | Expected outcome |
|----------|--------------|-----------------|
| CausAST without ortho penalty | Is orthogonality necessary? | OOD F1 drops >5% |
| CausAST without triplet (CE only) | Is metric learning needed? | Moderate F1 drop |
| TTA-Evident without TTA | Is test-time adaptation key? | ECE increases significantly |
| AP-NRL without augmenter | Is augmentation-robustness real? | Adversarial recall drops |
| OSCP without whitening | Is VILW needed beyond ortho? | Small drop, validates addition |
| AST-IRM with IRM from epoch 0 | Is annealing needed? | Training instability / NaN |

---

## Paper Table Mapping

- **Table 1** (Main OOD Robustness): exp01 CausAST + exp06 AST-IRM on AICD T1/T2
- **Table 2** (Adversarial Stress): exp03 AP-NRL on DROID adversarial splits
- **Table 3** (Calibration): exp02 TTA-Evident ECE/Brier
- **Table 4** (Transfer): train on DROID → eval on AICD (zero-shot)
- **Table 5** (Efficiency): params, VRAM, train time, inference time
