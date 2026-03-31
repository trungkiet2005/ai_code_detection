# [CoDET-M4] Experiment Tracker — NeurIPS 2026

> **Benchmark Dataset:** `DaniilOr/CoDET-M4` (~500K samples, Python/Java/C++)  
> **Paper Baseline:** Orel, Azizov & Nakov, ACL Findings 2025 (UniXcoder)  
> **Current Best:** **Exp18 HierTreeCode** — Author IID 70.55 F1 (new SOTA)

---

## 🏆 Leaderboard (IID Results — All Experiments)

| Exp | Method | Binary Macro-F1 | Author Macro-F1 | Author Val F1 | Status |
|:----|:-------|:---------------:|:---------------:|:-------------:|:------:|
| Paper (UniXcoder) | Baseline | 98.65 | 66.33 | — | reference |
| **Exp11** | SpectralCode | **99.06** | 69.82 | 70.80 | ✅ Done |
| **Exp18** | HierTreeCode | **99.06** | **70.55** | **71.88** | ✅ Done |
| Exp14 | ProtoCon | — | — | — | 🔲 Pending |
| Exp15 | GroupDRO | — | — | — | 🔲 Pending |
| Exp16 | HyperNetCode | — | — | — | 🔲 Pending |
| Exp17 | RAGDetect | — | — | — | 🔲 Pending |
| Exp19 | EAGLECode | — | — | — | 🔲 Pending |
| Exp20 | BiScopeCode | — | — | — | 🔲 Pending |

---

## 📊 Exp11 — SpectralCode (Baseline SOTA)

**Run:** 2026-03-30 | ModernBERT-base + AST + Structural + FFT Spectral  
**Loss:** `L_task + 0.3*L_neural + 0.3*L_spectral`

### IID Binary
| Metric | Value |
|:-------|:-----:|
| Test Macro-F1 | **0.9906** |
| Test Weighted-F1 | 0.9906 |
| Best Val F1 | 0.9893 |

| Language | F1 | Source | F1 |
|:---------|:--:|:-------|:--:|
| C++ | 99.09 | CF | 98.39 |
| Java | **99.54** | LC | 98.69 |
| Python | 98.61 | GH | 98.38 |

### IID Author (6-class)
| Metric | Value |
|:-------|:-----:|
| Test Macro-F1 | **0.6982** |
| Test Weighted-F1 | 0.8065 |
| Best Val F1 | 0.7080 |

| Class | F1 | | Class | F1 |
|:------|:--:|-|:------|:--:|
| Human | 0.9764 | | Llama3.1 | 0.8217 |
| CodeLlama | 0.7359 | | Nxcode | 0.4988 |
| GPT | 0.7434 | | Qwen1.5 | **0.4129** ← weakest |

**Key weakness:** Nxcode/Qwen1.5 confusion — Qwen recall only 36.28%

---

## 📊 Exp18 — HierTreeCode ⭐ NEW BEST

**Run:** 2026-03-31 | SpectralCode backbone + HierarchicalAffinityLoss  
**Loss:** `L_task + 0.3*L_neural + 0.3*L_spectral + 0.4*L_hier`  
**Key novelty:** Generator family tree constraint — Nxcode/Qwen1.5 forced to cluster together (within-family < cross-family + margin)

### IID Binary
| Metric | Value | vs Exp11 |
|:-------|:-----:|:--------:|
| Test Macro-F1 | **0.9906** | `=` |
| Test Weighted-F1 | 0.9906 | `=` |
| Best Val F1 | 0.9897 | `+0.0004` |

> Note: `Hier=0.0000` for binary task (correct — loss inactive when num_classes=2)

### IID Author (6-class)
| Metric | Value | vs Exp11 | Delta |
|:-------|:-----:|:--------:|:-----:|
| Test Macro-F1 | **0.7055** | `+0.0073` | `+0.73%` |
| Test Weighted-F1 | 0.8133 | `+0.0068` | `+0.68%` |
| Best Val F1 | **0.7188** | `+0.0108` | `+1.08%` |

| Class | F1 (Exp18) | F1 (Exp11) | Δ |
|:------|:----------:|:----------:|:--:|
| Human | 0.9820 | 0.9764 | `+0.0056` ↑ |
| CodeLlama | 0.7429 | 0.7359 | `+0.0070` ↑ |
| GPT | 0.7481 | 0.7434 | `+0.0047` ↑ |
| Llama3.1 | 0.8153 | 0.8217 | `-0.0064` ↓ |
| Nxcode | 0.5015 | 0.4988 | `+0.0027` ↑ |
| **Qwen1.5** | **0.4431** | **0.4129** | **`+0.0302` ↑↑** |

**Key finding:** Hier loss actively converges for author task (0.397 → 0.302 over 3 epochs). Qwen1.5 gains most (+3.02%) — family constraint working. Nxcode/Qwen confusion still significant but improving. Human recall jumps to 96.96% (was 95.83%).

#### Per-source breakdown (Author)
| Source | Exp18 | Exp11 | Δ |
|:-------|:-----:|:-----:|:--:|
| CF | 0.7717 | 0.7685 | `+0.0032` |
| GH | 0.5618 | 0.5540 | `+0.0078` |
| LC | 0.6035 | 0.5965 | `+0.0070` |

**Checkpoint:** `./codet_m4_checkpoints/codet_author/hiertreecode_CoDET_author_best.pt`

---

## 🚀 Performance vs Paper (Head-to-Head)

| Evaluation Mode (IID) | Paper (UniXcoder) | Exp11 SpectralCode | **Exp18 HierTreeCode** | Best Delta |
|:----------------------|:-----------------:|:------------------:|:----------------------:|:----------:|
| Binary F1 (Table 2) | 98.65 | 99.06 | **99.06** | `+0.41%` |
| Author F1 (Table 7) | 66.33 | 69.82 | **70.55** | `+4.22%` |

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
