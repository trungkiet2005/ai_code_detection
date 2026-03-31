# [CoDET-M4] Experiment Tracker — NeurIPS 2026

> **Benchmark Dataset:** `DaniilOr/CoDET-M4` (~500K samples, Python/Java/C++)  
> **Paper Baseline:** Orel, Azizov & Nakov, ACL Findings 2025 (UniXcoder)  
> **Current Best:** **Exp18 HierTreeCode** — Author IID 70.55 F1 (new SOTA, Exp14 ProtoCon at 70.13 close second)

---

## 🏆 Leaderboard (IID Results — All Experiments)

| Exp | Method | Binary Macro-F1 | Author Macro-F1 | Author Val F1 | Status |
|:----|:-------|:---------------:|:---------------:|:-------------:|:------:|
| Paper (UniXcoder) | Baseline | 98.65 | 66.33 | — | reference |
| **Exp11** | SpectralCode | **99.06** | 69.82 | 70.80 | ✅ Done |
| **Exp18** | HierTreeCode | **99.06** | **70.55** | **71.88** | ✅ Done |
| **Exp14** | ProtoCon | **99.06** | 70.13 | 71.26 | ✅ Done |
| **Exp15** | GroupDRO | 99.06 | 70.17 | 70.59 | ✅ Done |
| Exp16 | HyperNetCode | — | — | — | 🔲 Pending |
| **Exp17** | **RAGDetect** | **99.09** | **70.46** | **70.99** | ✅ Done |
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

## 📊 Exp17 — RAGDetect (Train+Test-Time Blending)

**Run:** 2026-03-31 | ModernBERT-base (bf16, H100)  
**Training:** IID (binary + author), `epochs=3`, `batch=64x1`, `workers=8`  
**Test-time RAG:** embedding bank `max=50000`, `dim=512`; retrieve `k=32`, blend `alpha=0.3`

### IID Binary (2-class)
| Metric | Value |
|:-------|:-----:|
| Test Macro-F1 | **0.9909** |
| Test Weighted-F1 | 0.9909 |
| Best Val F1 | 0.9901 |

| Language | Macro-F1 |
|:---------|:---------:|
| C++ | 0.9916 |
| Java | **0.9952** |
| Python | 0.9860 |

**Insight:** Binary is near ceiling; the harder part shows up clearly in **author (6-class)**.

### IID Author (6-class)
| Metric | Value |
|:-------|:-----:|
| Test Macro-F1 | **0.7046** |
| Test Weighted-F1 | 0.8109 |
| Best Val F1 | 0.7099 |

| Language | Macro-F1 |
|:---------|:---------:|
| C++ | 0.6994 |
| Java | **0.7526** |
| Python | 0.6670 |

| Source | Macro-F1 |
|:-------|:---------:|
| CF | **0.7636** |
| GH | 0.5680 |
| LC | 0.5973 |

**Classwise (Author, final test):**
| Class (generator) | F1 |
|:------------------|:--:|
| Human | 0.9818 |
| CodeLlama (codellama) | 0.7456 |
| GPT (gpt) | 0.7667 |
| Llama3.1 | 0.8164 |
| Nxcode (nxcode) | **0.4389** |
| Qwen1.5 (qwen1.5) | **0.4783** |

**Key failure mode (most important insight):** Strong confusion between **Nxcode <-> Qwen1.5**.
- `true=nxcode -> pred=qwen1.5` = `2442/5537`
- `true=qwen1.5 -> pred=nxcode` = `1482/5254`

**Interpretation:** Human is very strong (F1 0.9818) but generated classes remain poorly separated, which drags down `macro-F1` (macro 0.7046 < weighted 0.8109).

### Infra / Reliability Notes (from logs)
- HF Hub warning: unauthenticated requests (set `HF_TOKEN` to improve rate limits/speed).
- `ModernBertModel LOAD REPORT`: there are `UNEXPECTED` keys (e.g., `head.dense.weight`, `decoder.bias`, `head.norm.weight`) - may be ignorable if architectures/tasks differ, but worth tracking.
- Dataset loading: some `404` for `.py`/`dataset_infos.json` paths, but the pipeline still uses the **built-in split column** from CoDET-M4.

---
---

## 📊 Exp14 — ProtoCon (Prototype Contrastive Learning)

**Run:** 2026-03-31 | ModernBERT-base + EMA Prototype Bank + Hyperspherical Uniformity + SupCon  
**Loss:** `L_focal + 0.3*L_proto_attract + 0.1*L_unif + 0.2*L_supcon`  
**Key novelty:** Prototype memory bank pulled toward class centroids per batch; uniformity loss maximizes sphere separation
**Infra:** H100 80GB BF16 | batch=64x1 | epochs=3 | workers=8 | 149.5M params

### IID Binary
| Metric | Value | vs Exp18 |
|:-------|:-----:|:--------:|
| Test Macro-F1 | **0.9906** | `=` |
| Test Weighted-F1 | 0.9906 | `=` |
| Best Val F1 | 0.9902 | `=` |

| Language | F1 | Source | F1 |
|:---------|:--:|:-------|:--:|
| C++ | 0.9903 | CF | 0.9824 |
| Java | **0.9947** | LC | 0.9841 |
| Python | 0.9869 | GH | 0.9853 |

### IID Author (6-class)
| Metric | Value | vs Exp18 | Delta |
|:-------|:-----:|:--------:|:-----:|
| Test Macro-F1 | **0.7013** | `-0.0042` | `-0.42%` |
| Test Weighted-F1 | 0.8130 | `-0.0003` | `~` |
| Best Val F1 | **0.7126** | `-0.0062` | `-0.62%` |

| Class | F1 (Exp14) | F1 (Exp18) | Δ |
|:------|:----------:|:----------:|:--:|
| Human | 0.9826 | 0.9820 | `+0.0006` ↑ |
| CodeLlama | 0.7415 | 0.7429 | `-0.0014` ↓ |
| GPT | **0.7188** | 0.7481 | `-0.0293` ↓ |
| Llama3.1 | **0.8246** | 0.8153 | **`+0.0093` ↑↑** |
| Nxcode | **0.5073** | 0.5015 | **`+0.0058` ↑** |
| Qwen1.5 | 0.4329 | **0.4431** | `-0.0102` ↓ |

#### Per-source breakdown (Author)
| Source | Exp14 | Exp18 | Δ |
|:-------|:-----:|:-----:|:--:|
| CF | 0.7618 | **0.7717** | `-0.0099` |
| GH | 0.5591 | **0.5618** | `-0.0027` |
| LC | **0.6047** | 0.6035 | `+0.0012` |

**Key insights:**
- ProtoCon slightly below HierTreeCode overall (70.13 vs 70.55) but **Llama3.1 is best ever (+0.93%)** and Nxcode improves
- Prototype uniformity converges (unif: -0.08 → -1.03 over 3 epochs) — sphere separation working  
- SupCon stays high (3.62 → 3.06): author-level discrimination still hard, confirming Nxcode/Qwen confusion as main bottleneck
- Confusion: nxcode→qwen (1886 samples) and qwen→nxcode (1961 samples) — comparable to Exp18

### OOD Generator
**Status: ALL FAILED** — Known infrastructure issue: CoDET-M4 test split does not expose generator labels for LOO filtering. `test_ood=0` for all 5 generators.  
→ OOD generator evaluation requires custom dataset split; flag for fix in future runs.

### OOD Language (partial — log truncated)
`cpp`, `java`, `python` LOO runs started but log was truncated. Results pending.

**Checkpoint:** `./codet_m4_checkpoints/codet_author/protocon_CoDET_author_best.pt`

---

---

## 📊 Exp15 — GroupDRO (Distributionally Robust Optimization)

**Run:** 2026-03-31 | GroupDRO over 9 `language × source` groups | SpectralCode backbone  
**Loss:** `L_DRO (exponentiated gradient worst-group upweighting) + 0.3*L_neural + 0.3*L_spectral`  
**Key novelty:** 9 groups `{cpp,java,python} × {cf,gh,lc}` — upweights highest-loss group each step  
**Infra:** H100 80GB BF16 | batch=64x1 | epochs=3 | dro_eta=0.01 | 152.2M params

### IID Binary
| Metric | Value | vs Exp18 |
|:-------|:-----:|:--------:|
| Test Macro-F1 | **0.9898** | `-0.0008` |
| Test Weighted-F1 | 0.9898 | `-0.0008` |
| Best Val F1 | 0.9901 | `=` |

| Language | F1 | Source | F1 |
|:---------|:--:|:-------|:--:|
| C++ | 0.9903 | CF | 0.9817 |
| Java | **0.9949** | LC | 0.9827 |
| Python | 0.9845 | **GH** | **0.9836** |

> DRO binary worst-group: `python_gh` dominates early (w=0.14 ep1) → shifts to `cpp_cf` (w=0.155 ep3)

### IID Author (6-class)
| Metric | Value | vs Exp18 | Delta |
|:-------|:-----:|:--------:|:-----:|
| Test Macro-F1 | **0.7017** | `-0.0038` | `-0.38%` |
| Test Weighted-F1 | 0.8146 | `+0.0013` | `+0.13%` |
| Best Val F1 | **0.7059** | `-0.0129` | `-1.29%` |

| Class | F1 (Exp15) | F1 (Exp18) | Δ |
|:------|:----------:|:----------:|:--:|
| Human | 0.9881 | 0.9820 | `+0.0061` ↑ |
| CodeLlama | 0.7379 | 0.7429 | `-0.0050` ↓ |
| GPT | 0.7267 | **0.7481** | `-0.0214` ↓ |
| Llama3.1 | **0.8218** | 0.8153 | `+0.0065` ↑ |
| Nxcode | 0.4560 | **0.5015** | **`-0.0455` ↓↓** |
| **Qwen1.5** | **0.4798** | 0.4431 | **`+0.0367` ↑↑ best ever!** |

#### Per-source breakdown (Author)
| Source | Exp15 | Exp18 | Δ |
|:-------|:-----:|:-----:|:--:|
| CF | 0.7385 | **0.7717** | `-0.0332` |
| **GH** | **0.5778** | 0.5618 | **`+0.0160` ↑ GroupDRO working!** |
| LC | 0.6032 | **0.6035** | `~` |

#### Per-group breakdown (Author test)
| Group | Macro-F1 | Group | Macro-F1 |
|:------|:--------:|:------|:--------:|
| cpp_cf | 0.7270 | java_cf | **0.7642** |
| cpp_gh | 0.6645 | java_gh | 0.5864 |
| cpp_lc | 0.5596 | java_lc | **0.7693** |
| python_cf | 0.4467 | python_gh | 0.5362 |
| python_lc | 0.6107 | — | — |

> DRO author worst-group: `python_gh` weight escalates to **0.644** by ep3 — confirms Python/GitHub is hardest combination

### OOD Generator
**Status: ALL FAILED** — Same `test_ood=0` infrastructure issue as Exp14. Confirmed systematic.

### Key Insights
- **GroupDRO successfully improves GH source** (+1.60% Author) — worst-group upweighting targeting Python/GH combo works
- **Trade-off: Nxcode drops significantly** (-4.55%) while Qwen improves (+3.67%) — DRO optimization target conflicts with fine-grained generator separation
- **Binary slightly below baseline** (98.98 vs 99.06) — DRO regularization costs a small IID penalty
- **python_gh dominates** as worst group throughout — key target for future experiments

**Checkpoint:** `./codet_m4_groupdro_checkpoints/iid_author/best.pt`

---

## 🚀 Performance vs Paper (Head-to-Head)

| Evaluation Mode (IID) | Paper (UniXcoder) | Exp11 SpectralCode | Exp14 ProtoCon | Exp15 GroupDRO | **Exp18 HierTreeCode** | Best Delta |
|:----------------------|:-----------------:|:------------------:|:--------------:|:--------------:|:----------------------:|:----------:|
| Binary F1 (Table 2) | 98.65 | 99.06 | 99.06 | 98.98 | **99.06** | `+0.41%` |
| Author F1 (Table 7) | 66.33 | 69.82 | 70.13 | 70.17 | **70.55** | `+4.22%` |

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
1. **Author F1 = 70.55% (best: Exp18) / 70.46% (Exp17)** — Nxcode/Qwen confusion persists; generated-class separation still the main failure mode
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

| Eval Mode | Current (best so far) | ProtoCon Target | GroupDRO Target | HyperNet Target |
|:----------|:---------------:|:---------------:|:---------------:|:---------------:|
| Binary IID Macro-F1 | 99.06 | 99.10+ | 99.05+ | 99.08+ |
| Author IID Macro-F1 | **70.55** | **77–80** | 72–75 | **75–78** |
| OOD Generator | pending | 94+ | **95+** | 92+ |
| OOD Source (hardest) | pending | 57+ | **62+** | 58+ |

---

## 🧪 New Experiments — Batch 2 (Exp21–Exp26)

Designed 2026-03-31 | Based on ICML 2025, NeurIPS 2024, CVPR/ICCV/AAAI survey

> **Research sources:** DeTeCtive (NeurIPS 2024), MH-MoE (NeurIPS 2024), TLM-LoRA (ICML 2025),
> FA-AST+GNN (AAAI/ICSE 2024), BYOL/DINO (NeurIPS 2020/2021), CosFace (CVPR 2019),
> CharCNN (NIPS 2015), DinoSR (NeurIPS 2023), SAR (ICLR 2023)

| Exp | File | Method | Core Novelty | Target | Inspired By |
|-----|------|--------|--------------|--------|-------------|
| **Exp21** | `exp21_moe.py` | **MoECode** | Sparse MoE (4 experts, top-2 routing) over fusion repr. Load-balance auxiliary loss. Each expert specializes: syntax/semantic/style/frequency | Author 72%+ OOD robustness | MH-MoE (NeurIPS 2024), Mixtral, Switch Transformer |
| **Exp22** | `exp22_tta.py` | **TTACode** | Test-Time Adaptation via entropy minimization. Updates only LayerNorm params at inference (TENT-style). SAR filter for high-entropy samples | OOD +5-10% F1, no labels needed | TENT (ICLR 2021), SAR (ICLR 2023), TLM (ICML 2025) |
| **Exp23** | `exp23_graphstyle.py` | **GraphStyleCode** | Replaces BiLSTM AST with Graph Attention Network (GAT). GraphSAGE-style concat+project, 2-layer stacked, attention-weighted global pooling | GH source +4-8%, structural OOD | FA-AST+GNN (AAAI 2020/ICSE 2024), GraphCodeBERT (ICLR 2021) |
| **Exp24** | `exp24_cosineproto.py` | **CosineProtoCode** | Cosine similarity to learnable class prototypes (EMA-updated). Learnable per-class temperature. Hyperspherical uniformity regularization | Author 74%+, Nxcode/Qwen separation | CosFace/ArcFace (CVPR 2018/19), ProtoNet (NeurIPS 2017), Hyperspherical Uniformity (NeurIPS 2018) |
| **Exp25** | `exp25_multigran.py` | **MultiGranCode** | Adds CharCNN (3 parallel Conv1D, k=3,5,7) as 3rd feature stream. 3-way gated fusion: neural+spectral+char. Char stream captures micro-stylometry | Binary 99.1%+, Author 71%+ | CharCNN (NIPS 2015), TextCNN (EMNLP 2014), Multi-granularity NLP (ACL 2020) |
| **Exp26** | `exp26_selfdistill.py` | **SelfDistillCode** | BYOL-style EMA teacher self-distillation. Student matches teacher (its own past self) representation — no labels for distillation loss. Teacher updated as EMA after each step | OOD robustness 72%+, stable repr | BYOL (NeurIPS 2020), DINO (ICCV 2021), Mean Teacher (NeurIPS 2017), DinoSR (NeurIPS 2023) |

### Leaderboard Update (Exp21-26 all Pending)

| Exp | Method | Binary F1 | Author F1 | Val F1 | Status |
|:----|:-------|:---------:|:---------:|:------:|:------:|
| Exp18 | HierTreeCode | 99.06 | **70.55** | **71.88** | ✅ SOTA |
| Exp21 | MoECode | — | — | — | 🔲 Pending |
| Exp22 | TTACode | — | — | — | 🔲 Pending |
| Exp23 | GraphStyleCode | — | — | — | 🔲 Pending |
| Exp24 | CosineProtoCode | — | — | — | 🔲 Pending |
| Exp25 | MultiGranCode | — | — | — | 🔲 Pending |
| Exp26 | SelfDistillCode | — | — | — | 🔲 Pending |

### Research Agent Findings (2026-03-31)

Top-rated paper ideas from NeurIPS/ICML/CVPR survey for NeurIPS 2026 ORAL:

| Priority | Paper | Venue | Year | Addresses |
|:---------|:------|:-----:|:----:|:----------|
| **S-tier** | DeTeCtive: Multi-Level Contrastive + kNN inference | NeurIPS | 2024 | Nxcode/Qwen, OOD, GH |
| **S-tier** | TLM: Test-Time Learning via LoRA perplexity min | ICML | 2025 | OOD collapse (0.29→0.7?) |
| **A-tier** | MH-MoE: Sub-token routing (h=4, 8 experts) | NeurIPS | 2024 | Nxcode/Qwen, GH |
| **A-tier** | FA-AST+CPG: AST+CFG+DFG graph, GATv2 | AAAI/ICSE | 2024 | GH source, structural |
| **A-tier** | SHAP Attribution Meta-features | SciReports | 2025 | Nxcode/Qwen, interpretability |
| **A-tier** | Deep Ensemble Disagreement (5 seeds, mutual info) | NeurIPS | 2017 | OOD flagging, uncertainty |
| **B-tier** | DiffPath: Diffusion score curvature for OOD | NeurIPS | 2024 | Novel signal, no retraining |
| **B-tier** | Span-level CRF segmentation | arXiv | 2025 | GH mixed-authorship, T3 |

> **Next priority after Exp21-26:** Implement DeTeCtive multi-level contrastive (Exp27) and
> LoRA-TTA upgrade of Exp22 (replaces norm-layer TTA with full LoRA adaptation → Exp27 or v2).
> **Ensemble of Exp18×5 seeds** (zero training cost) estimated +1-2% from mutual-info reranking.
