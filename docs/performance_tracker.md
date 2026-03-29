# Performance Tracker - AICD-Bench

## Experiment Results

| Exp | Name | Task | Model | Epochs | Best Val F1 | Test F1 | Notes |
|-----|------|------|-------|--------|-------------|---------|-------|
| exp00 | CodeOrigin | T1 | ModernBERT-base | 5 | 0.9948 | 0.2625 | Severe overfitting: val F1 ~0.99 but test F1 ~0.26 |

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
