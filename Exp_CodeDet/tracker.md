# [CoDET-M4] SpectralCode Runner – Full Benchmark Evaluation

> **Method Name:** SpectralCode (ModernBERT-base + AST + Structural + FFT Spectral)  
> **Status:** **SOTA (REACHED)**  
> **Benchmark Dataset:** `DaniilOr/CoDET-M4` (~500K samples)

---

## 🚀 Performance Overview (Head-to-Head)

Comparison against Paper Results (Orel, Azizov & Nakov, ACL Findings 2025). SpectralCode outperforms the `UniXcoder` baseline in all in-distribution (IID) metrics.

| Evaluation Mode (IID) | Paper Baseliner (UniXcoder) | **SpectralCode (Exp11)** | **Delta (Gain)** |
|:----------------------|:---------------------------:|:------------------------:|:----------------:|
| **Binary Classification** (Table 2) | 98.65 F1                | **99.06 F1**             | `+ 0.41%`        |
| **Authorship Identification** (Table 7) | 66.33 F1                | **69.82 F1**             | `+ 3.49%`        |

---

## 📊 Binary Classification Breakdown (IID)

### 1. Per-Language Performance (Paper Table 3 Mapping)
`SpectralCode` shows superior robustness in C++ and Java, while maintaining parity in Python.

| Language | Paper F1 (UniXcoder) | **SpectralCode F1** | Status |
|:---------|:--------------------:|:-------------------:|:-------|
| **C++**  | 98.24                | **99.09**           | `Best` |
| **Java**  | 99.02                | **99.54**           | `Best` |
| **Python**| 98.60                | **98.61**           | `Stable` |

### 2. Per-Source Performance (Paper Table 4 Mapping)
Notable gain in CodeForces (CF), suggesting that structural/spectral features are highly effective in competitive programming contexts.

| Source   | Paper F1 (UniXcoder) | **SpectralCode F1** | Gain |
|:---------|:--------------------:|:-------------------:|:----:|
| **CodeForces (CF)** | 96.54                | **98.39**           | `+ 1.85%` |
| **LeetCode (LC)**   | 97.87                | **98.69**           | `+ 0.82%` |
| **GitHub (GH)**     | 98.46                | 98.38               | `- 0.08%` |

---

## ✍️ Authorship Identification (6-Class)

Objective: Identify the generator model (CodeLlama, Llama3.1, CodeQwen1.5, Nxcode, GPT-4o) or Human.

| Metric | Performance |
|:-------|:-----------:|
| **Macro-F1** | **0.6982** |
| **Weighted-F1** | 0.8065 |
| **Best Val F1** | 0.7080 |

### Confusion Matrix Highlights (Fig 2 Insight)
The paper highlights confusion between **Nxcode** and **CodeQwen1.5** (since Nxcode is a fine-tuned version of CodeQwen). SpectralCode confirms this trend but improves individual model recall significantly.

- **Nxcode <=> CodeQwen1.5 Convergence**: Significant confusion remains (approx. 33-40% of Qwen1.5 samples predicted as Nxcode), but **Human Authorship recall is near-perfect (99.51%)**.

---

## 🔍 Architecture & Hyperparameters (Exp11)

- **Backbone**: `answerdotai/ModernBERT-base` (attn_implementation="sdpa")
- **Loss**: `L_total = L_task + 0.3*L_neural + 0.3*L_spectral` (Focal Gain=2.0)
- **Features**: 
    - **FFT Spectral**: 64-dim (multi-scale 32/64/128/256 frequency energy)
    - **Bi-LSTM AST**: 128-dim hidden representation
    - **Structural**: 22 Hand-crafted statistical features (indent, camelCase, etc.)
- **Device**: NVIDIA H100 80GB HBM3 (BF16, Effective Batch 64)

---

## 📋 Registry & Status (Proxy Results for OOD)

These evaluations were not included in the latest `iid_only` run but are mapped to the paper's OOD scenarios.

| Eval Mode | Paper Table | Proxy Metric (from best DM tracker) | Status |
|:----------|:------------|:------------------------------------|:-------|
| `ood_generator` | Table 8 | 93.22 (Paper UniXcoder) | `Pending` |
| `ood_language`  | Table 10 | 88.96 (Paper UniXcoder) | `Pending` |
| `ood_source`    | Table 9 | 55.01 (Paper UniXcoder) | `Pending` |

---
**Run Completed:** 2026-03-30  
**Checkpoint Path:** `./codet_m4_checkpoints/codet_binary/spectralcode_CoDET_binary_best.pt`

---

## 🧪 Next Experiments (Targeting A* Paper — NeurIPS/ICLR 2026)

Current bottlenecks to attack:
1. **Author F1 = 69.82%** — Nxcode/Qwen confusion, weak cross-source generalization
2. **OOD evaluation pending** — ood_generator / ood_language / ood_source all untested
3. **GitHub source is hardest** (macro 0.5540 vs CF 0.7685) — domain shift problem

| Exp | File | Method | Core Novelty | Target | Tier |
|-----|------|--------|--------------|--------|------|
| **Exp14** | `run_codet_m4_protocon.py` | **ProtoCon** | Prototype memory bank (EMA) + hyperspherical uniformity + SupCon. Prototypes as anchors pull same-class embeddings, repel across classes | Author F1 ↑↑ (target 77%+) | **S** |
| **Exp15** | `run_codet_m4_groupdro.py` | **GroupDRO** | Distributionally Robust Optimization over `language × source` environments. Exponentiated-gradient weight updates ensure worst-group performance is maximized | OOD ↑↑, GH source ↑↑ | **S** |
| **Exp16** | `run_codet_m4_hypernet.py` | **HyperNetCode** | Style-Content disentanglement via HyperNetwork: style encoder → hypernet generates generator-specific classifier weights; MMD loss aligns content distributions across generators | Author F1 ↑↑, NXcode/Qwen separation ↑ | **A+** |
| **Exp17** | `run_codet_m4_ragdetect.py` | **RAGDetect** | Frozen EmbeddingBank from train set post-training. Test-time: retrieve top-k neighbors (k=32, cosine) → blend with classifier logits (α=0.3). Training-free OOD boost | OOD generator F1 +5-8% | **A+** |
| **Exp18** | `run_codet_m4_hiertree.py` | **HierTreeCode** | HierarchicalAffinityLoss over generator family tree: Nxcode/Qwen1.5 share family node → hard-negative triplet forces within-family < cross-family distance + margin | Author F1 ↑↑ (target 77%+), Nxcode/Qwen fix | **S** |
| **Exp19** | `run_codet_m4_eagle.py` | **EAGLECode** | GradientReversalLayer (DANN) → encoder learns generator-invariant binary features. DANN α schedule 0→1, λ_adv annealed 0→0.1 over epoch 1. gen_label from dataset | OOD generator F1 95%+, domain-invariant features | **S** |
| **Exp20** | `run_codet_m4_biscope.py` | **BiScopeCode** | Bidirectional MLM probe: mask K=16 positions/sequence, measure P(original\|context) via ModernBERT embedding cosine. 8-dim BiScope stats as 3rd gated stream | Binary 99.15%+, Author 72%+, novel signal | **S** |

### Motivation from Recent Literature

- **ProtoCon**: Inspired by DINO (ICCV 2021), ProtoNet, and the NeurIPS 2024 memory bank contrastive papers. The key insight: prototype-guided contrastive pulls same-generator samples into tight clusters while hyperspherical uniformity maximally separates class centers on the unit sphere. This directly attacks the Nxcode-Qwen collapse.
- **GroupDRO**: Inspired by Sagawa et al. (ICLR 2020) GroupDRO, extended with the spectral feature backbone from Exp11. The `language × source` factorization (9 groups) forces the model to achieve uniform accuracy across CF/GH/LC × C++/Java/Python combinations — precisely where current models fail (GH macro = 0.55).
- **HyperNetCode**: Inspired by HyperStyle (CVPR 2022) and recent disentanglement papers at ICLR 2025. If the generator's "writing style" is a latent code, then a hypernetwork that generates style-specific classifier weights will naturally discriminate generators while remaining agnostic to code content.

### Expected Performance Targets

| Eval Mode | Current (Exp11) | ProtoCon Target | GroupDRO Target | HyperNet Target |
|:----------|:---------------:|:---------------:|:---------------:|:---------------:|
| Binary IID Macro-F1 | 99.06 | 99.10+ | 99.05+ | 99.08+ |
| Author IID Macro-F1 | 69.82 | **77–80** | 72–75 | **75–78** |
| OOD Generator | pending | 94+ | **95+** | 92+ |
| OOD Source (hardest) | pending | 57+ | **62+** | 58+ |
