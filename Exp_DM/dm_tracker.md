# DM Tracker - Deep Methods Experiment Suite

## Overview

Advanced architectural experiments targeting **NeurIPS 2026 ORAL**.
All methods build on `exp00_codeorigin.py` (CodeOrigin baseline) with novel components from the Gemini DeepResearch portfolio.

**Protocol:**
- Train/evaluate **separate models per benchmark** (AICD-Bench / DroidCollection).
- Each file is **standalone** and Kaggle-runnable (`python exp0X_xxx.py`).
- Metrics: Macro-F1 (primary), Weighted-F1, per-class report.
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

---

## Baseline Reference (from exp00 CodeOrigin)

| Benchmark | Task | Best Val F1 | Test F1 | Test Weighted F1 | Notes |
|-----------|------|-------------|---------|------------------|-------|
| AICD | T1 | 0.9954 | 0.2877 | 0.2721 | Severe OOD collapse (val-test gap) |
| DROID | T1 | 0.9693 | 0.9708 | 0.9709 | Strong in-distribution performance |

**Key bottleneck:** AICD T1 OOD robustness. All DM experiments target this gap.

---

## Results

### AICD-Bench (Benchmark A)

| Exp | Method | Task | Epochs | Best Val F1 | Test Macro-F1 | Test Weighted-F1 | ECE | Brier | Notes |
|-----|--------|------|--------|-------------|---------------|------------------|-----|-------|-------|
| baseline | CodeOrigin | T1 | 3 | 0.9954 | 0.2877 | 0.2721 | - | - | Shortcut learning under OOD |
| exp01 | CausAST | T1 | 3 | - | - | - | - | - | |
| exp02 | TTA-Evident | T1 | 3 | - | - | - | - | - | |
| exp03 | AP-NRL | T1 | 3 | - | - | - | - | - | |
| exp04 | BH-SCM | T1 | 3 | - | - | - | - | - | |
| exp05 | OSCP | T1 | 3 | - | - | - | - | - | |
| exp06 | AST-IRM | T1 | 3 | - | - | - | - | - | |

### DroidCollection (Benchmark B)

| Exp | Method | Task | Epochs | Best Val F1 | Test Macro-F1 | Test Weighted-F1 | Notes |
|-----|--------|------|--------|-------------|---------------|------------------|-------|
| baseline | CodeOrigin | T1 | 3 | 0.9693 | 0.9708 | 0.9709 | Strong baseline |
| exp01 | CausAST | T1 | 3 | - | - | - | |
| exp02 | TTA-Evident | T1 | 3 | - | - | - | |
| exp03 | AP-NRL | T1 | 3 | - | - | - | |
| exp04 | BH-SCM | T1 | 3 | - | - | - | |
| exp05 | OSCP | T1 | 3 | - | - | - | |
| exp06 | AST-IRM | T1 | 3 | - | - | - | |

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
