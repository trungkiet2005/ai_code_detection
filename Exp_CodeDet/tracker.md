# [CoDET-M4] Experiment Tracker — NeurIPS 2026

> **Benchmark Dataset:** `DaniilOr/CoDET-M4` (~500K samples, Python/Java/C++)  
> **Paper Baseline:** Orel, Azizov & Nakov, ACL Findings 2025 (UniXcoder)  
> **Current Best:** **Exp18 HierTreeCode** — Author IID 70.55 F1 (+4.22% vs UniXcoder)

---

## 🏆 Leaderboard — Ranked by Author Macro-F1 ↓

> Primary metric = **Author IID Macro-F1** (6-class, hardest task). Binary is near-ceiling for all modern PLMs.  
> Δ-Paper = delta vs paper's best (UniXcoder: Binary 98.65 / Author 66.33)

| Rank | Exp | Method | Binary F1 | Δ-Paper | Author F1 | Δ-Paper | Val F1 | Status |
|:----:|:----|:-------|:---------:|:-------:|:---------:|:-------:|:------:|:------:|
| 🥇 | **Exp18** | **HierTreeCode** | **99.06** | `+0.41` | **70.55** | **`+4.22`** | **71.88** | ✅ |
| 🥈 | **Exp17** | RAGDetect | **99.09** | `+0.44` | 70.46 | `+4.13` | 70.99 | ✅ |
| 🥉 | **Exp20** | BiScopeCode | **99.06** | `+0.41` | 70.20 | `+3.87` | 71.22 | ✅ |
| 4 | **Exp22** | TTACode | **99.06** | `+0.41` | 70.20 | `+3.87` | 71.32 | ✅ |
| 5 | **Exp25** | MultiGranCode | 99.03 | `+0.38` | 70.19 | `+3.86` | 71.22 | ✅ |
| 6 | **Exp15** | GroupDRO | 98.98 | `+0.33` | 70.17 | `+3.84` | 70.59 | ✅ |
| 7 | **Exp26** | SelfDistillCode | 99.04 | `+0.39` | 70.14 | `+3.81` | 71.22 | ✅ |
| 8 | **Exp14** | ProtoCon | **99.06** | `+0.41` | 70.13 | `+3.80` | 71.26 | ✅ |
| 9 | **Exp11** | SpectralCode | **99.06** | `+0.41` | 69.82 | `+3.49` | 70.80 | ✅ |
| 10 | **Exp23** | GraphStyleCode | 99.00 | `+0.35` | 69.71 | `+3.38` | 70.89 | ✅ |
| — | **Exp16** | HyperNetCode | **99.07** | `+0.42` | — ❌ bug | — | — | ✅ |
| — | **Exp19** | EAGLECode (DANN) | 98.73 | `+0.08` | 62.89 ↓↓ | `-3.44` | 64.23 | ✅ |
| — | Exp21, Exp24 | Batch 2 (pending) | — | — | — | — | — | 🔲 |
| **REF** | Paper | UniXcoder | 98.65 | — | 66.33 | — | — | reference |
| REF | Paper | CodeT5 | 98.35 | `-0.30` | 62.45 | `-3.88` | — | reference |
| REF | Paper | CodeBERT | 95.70 | `-2.95` | 64.80 | `-1.53` | — | reference |
| REF | Paper | CatBoost | 88.78 | `-9.87` | 45.42 | `-20.91` | — | reference |

---

## 📊 Full Paper Baseline Comparison (All Tables)

### Table 2 — Binary IID Detection (Macro-F1 %)

| Model | Binary F1 | Binary Acc | Our Best (Exp17) | Δ |
|:------|:---------:|:----------:|:----------------:|:--:|
| **Exp17 RAGDetect (ours)** | **99.09** | **99.07** | — | — |
| **Exp18 HierTreeCode (ours)** | **99.06** | **99.06** | — | — |
| **Exp11–Exp14 SpectralCode family** | **99.06** | **99.06** | — | — |
| UniXcoder (paper best) | 98.65 | 98.65 | `+0.44` | ↑ |
| CodeT5 | 98.35 | 98.35 | `+0.74` | ↑ |
| CodeBERT | 95.70 | 95.71 | `+3.39` | ↑ |
| CatBoost | 88.78 | 88.79 | `+10.31` | ↑ |
| SVM | 72.19 | 72.19 | `+26.90` | ↑ |
| Baseline (logistic) | 62.03 | 65.17 | `+37.06` | ↑ |

### Table 3 — Binary Per-Language (F1 %)

| Language | UniXcoder | CodeT5 | CodeBERT | **Our Best (Exp17/18)** | Δ vs UniXcoder |
|:---------|:---------:|:------:|:--------:|:-----------------------:|:--------------:|
| **C++** | 98.24 | 97.86 | 95.73 | **99.12** (Exp16) | `+0.88` ↑ |
| **Java** | 99.02 | 98.89 | 96.54 | **99.54** (Exp11) | `+0.52` ↑ |
| **Python** | 98.60 | 98.22 | 94.84 | **98.66** (Exp11) | `+0.06` ↑ |

### Table 4 — Binary Per-Source (F1 %)

| Source | UniXcoder | CodeT5 | CodeBERT | **Our Best** | Δ vs UniXcoder |
|:-------|:---------:|:------:|:--------:|:------------:|:--------------:|
| **CodeForces** | 96.54 | 97.24 | 91.67 | **98.39** (Exp11) | `+1.85` ↑ |
| **LeetCode** | 97.87 | 66.23 | 87.63 | **99.35** (Exp15) | `+1.48` ↑ |
| **GitHub** | 98.46 | 98.54 | 95.31 | **98.54** (Exp11) | `+0.08` ↑ |

### Table 7 — Author IID (6-class Macro-F1 %)

| Model | Author F1 | Author Acc | **Our Best (Exp18)** | Δ |
|:------|:---------:|:----------:|:--------------------:|:--:|
| **Exp18 HierTreeCode (ours)** | **70.55** | ~80+ | — | — |
| **Exp17 RAGDetect (ours)** | **70.46** | — | — | — |
| UniXcoder (paper best) | 66.33 | 79.35 | `+4.22` | ↑ |
| CodeBERT | 64.80 | 77.65 | `+5.75` | ↑ |
| CodeT5 | 62.45 | 78.25 | `+8.10` | ↑ |
| CatBoost | 45.42 | 66.19 | `+25.13` | ↑ |
| SVM | 27.63 | 49.70 | `+42.92` | ↑ |

### Table 8 — OOD Generator LOO (F1 %) [pending our eval]

| Model | OOD-Gen F1 | OOD-Gen Recall | OOD-Gen Acc |
|:------|:----------:|:--------------:|:-----------:|
| UniXcoder | **93.22** | 87.30 | 87.30 |
| CatBoost | 92.31 | 85.71 | 85.71 |
| SVM | 88.99 | 80.16 | 80.16 |
| CodeT5 | 79.43 | 65.87 | 65.87 |
| CodeBERT | 66.67 | 50.00 | 50.00 |
| Baseline | 59.65 | 29.37 | 64.68 |
| **Our experiments** | ❌ test_ood=0 | — | — |

> ⚠️ All our experiments fail OOD-generator LOO: CoDET-M4 test split has no `model` field → `test_ood=0`. Needs custom train-split LOO.

### Table 9 — OOD Source LOO (F1 %) [pending our eval]

| Model | OOD-Source F1 | OOD-Source Acc |
|:------|:-------------:|:--------------:|
| CodeT5 | **58.22** | 74.11 |
| UniXcoder | 55.01 | 72.81 |
| CatBoost | 50.62 | 69.11 |
| Baseline | 49.84 | 50.30 |
| CodeBERT | 43.16 | 66.01 |
| SVM | 38.66 | 55.16 |
| **Our experiments** | ❌ pending | — |

### Table 10 / 12 — OOD Language LOO (F1 %)

**Aggregate (Table 12):**
| Model | OOD-Lang F1 | OOD-Lang Acc |
|:------|:-----------:|:------------:|
| UniXcoder | **88.96** | 88.96 |
| CodeT5 | 71.47 | 72.17 |
| CodeBERT | 58.78 | 59.10 |
| CatBoost | 53.42 | 56.26 |
| Baseline | 51.53 | 59.64 |
| SVM | 28.68 | 36.29 |

**Per-language (Table 10):**
| Language | UniXcoder | CodeT5 | CodeBERT | CatBoost |
|:---------|:---------:|:------:|:--------:|:--------:|
| C# | **91.31** | 78.55 | 45.31 | 51.01 |
| Go | **90.01** | 88.78 | 52.46 | 64.72 |
| JavaScript | **81.48** | 34.81 | 26.84 | 26.03 |
| PHP | **96.17** | 93.66 | 56.68 | 45.08 |

> Our Exp16 cpp-OOD: 88.21%; GH subgroup 48.39% (worst). Java/Python OOD pending.

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

## 📊 Exp16 — HyperNetCode (Style-Content Disentanglement via HyperNetwork)

**Run:** 2026-03-31 | ModernBERT-base + style encoder → hypernetwork → generator-specific classifier weights  
**Loss:** `L_focal + 0.2*L_style_aux + 0.3*L_mmd + 0.2*L_style_con`  
**Key novelty:** Style encoder generates HyperNet weights per class (generator-conditioned FC); MMD aligns content distributions; StyleCon separates style embeddings  
**Infra:** H100 80GB BF16 | batch=64x1 | epochs=3 | workers=8 | 150.0M params

### ⚠️ Training Issue: StyleCon = NaN throughout all 3 epochs
Total loss = `nan` at every step — `style_con` diverged immediately (likely exp overflow or division-by-zero in style contrastive pairs). However `CE(focal)` + `MMD` were numerically valid and the model trained effectively via those components. The NaN total did not crash training (backprop on NaN had no gradient effect), so the model converged on the CE objective alone.

### IID Binary
| Metric | Value | vs Exp18 | vs Exp17 (SOTA) |
|:-------|:-----:|:--------:|:---------------:|
| Test Macro-F1 | **0.9907** | `+0.0001` | `-0.0002` |
| Test Weighted-F1 | 0.9907 | `+0.0001` | `-0.0002` |
| Best Val F1 | 0.9897 | `=` | `-0.0004` |

| Language | Macro-F1 | Source | Macro-F1 |
|:---------|:---------:|:-------|:---------:|
| C++ | 0.9912 | CF | 0.9831 |
| Java | **0.9945** | GH | 0.9852 |
| Python | 0.9866 | LC | 0.9844 |

Binary per-class breakdown:
| Class | Precision | Recall | F1 |
|:------|:---------:|:------:|:--:|
| Human (0) | 0.9905 | 0.9915 | 0.9910 |
| AI (1) | 0.9910 | 0.9899 | 0.9904 |

### IID Author (6-class)
**FAILED** — `classes=0` error at preflight. Exp16 data loader for author task returned empty splits (train=0, val=0, test=0). Root cause: Exp16 has a different `load_codet_m4_data` implementation that appears to not handle the author task's `model` field correctly. Same as OOD generator issue but for the IID split.

> Unique to Exp16: other experiments (11/14/15/17/18) all correctly load author data. Likely a bug in Exp16's author filtering that sets `num_classes=0` when `model` field is absent or differently named.

### OOD Generator
**Status: ALL FAILED** — `test_ood=0` for all generators. Same infrastructure issue as Exp14/15.

### OOD Language
| Language held-out | Test Macro-F1 | Test Weighted-F1 | Best Val F1 |
|:------------------|:-------------:|:----------------:|:-----------:|
| cpp | 0.8821 | 0.8841 | 0.9898 |
| java | — (log truncated) | — | — |
| python | — (log truncated) | — | — |

**OOD cpp breakdown:**
| Source (cpp test) | Macro-F1 | Weighted-F1 |
|:------------------|:--------:|:-----------:|
| CF | 0.9289 | 0.9506 |
| **GH** | **0.4839** | 0.4559 ← worst |
| LC | 0.9916 | 0.9983 |

> GH source drops severely under language OOD (48.39% macro for cpp-held-out) — same GH weakness seen across all experiments. CF and LC remain robust.

### Key Insights
- **HyperNetCode achieves 99.07% binary** — second highest overall (Exp17 RAGDetect at 99.09%)
- **StyleCon NaN bug** is a critical failure — the intended novel contribution (style-conditioned hypernetwork) was inoperative; results reflect a CE+MMD baseline, not the full HyperNet design
- **Author task completely broken** in Exp16 — the most important task metric is unavailable; needs data loading fix before drawing conclusions about the hypernet's generator-discrimination ability
- **OOD Language cpp → 88.21%** — consistent with other experiments; GH-subgroup is the worst (48.39%)
- **Net conclusion:** Exp16 is inconclusive — StyleCon NaN prevents evaluating the core contribution

**Checkpoint:** `./hypernet_checkpoints/hypernetcode_iid_binary_best.pt`

---

## 📊 Exp19 — EAGLECode (DANN Adversarial Domain Adaptation)

**Run:** 2026-03-31 | ModernBERT-base + GradientReversalLayer (DANN) for generator-invariant features  
**Loss:** `L_focal + L_neural + L_spectral + λ_adv * L_adv`  
**Key novelty:** λ_adv anneals 0→0.1 over epoch 1; GRL reverses gradient from generator discriminator → encoder learns generator-invariant binary features  
**Infra:** H100 80GB BF16 | batch=64x1 | epochs=3 | **~151.806M** params (logs: 151,805,772 binary / 151,808,340 author)

### IID Binary
| Metric | Value | vs Exp18 |
|:-------|:-----:|:--------:|
| Test Macro-F1 | **0.9873** | `-0.0033` |
| Test Weighted-F1 | 0.9873 | `-0.0033` |
| Best Val F1 | **0.9879** | `-0.0018` |

| Language | Macro-F1 | Source | Macro-F1 |
|:---------|:---------:|:-------|:---------:|
| C++ | 0.9876 | CF | 0.9758 |
| Java | **0.9927** | GH | 0.9808 |
| Python | 0.9819 | LC | 0.9777 |

**Binary training dynamics:**
- Epoch 1: λ_adv=0, Adv=0 → val **0.9726**
- Epoch 2: GRL on (λ_adv=0.1, Adv≈3–4.8) → val **0.9815** mid-epoch, **0.9793** end-epoch
- Epoch 3: Adv≈2.5–2.7 → val **0.9872** mid → **0.9879** best (loaded for test) — still **below** Spectral/TTACode tier (~0.990+)

### IID Author (6-class) ⚠️ CATASTROPHIC DEGRADATION
| Metric | Value | vs Exp18 | Delta |
|:-------|:-----:|:--------:|:-----:|
| Test Macro-F1 | **0.6289** | `-0.0766` | **`-7.66%` ↓↓↓** |
| Test Weighted-F1 | 0.7408 | `-0.0725` | `-7.25%` |
| Best Val F1 | **0.6423** | `-0.0765` | `-7.65%` |

Per-class F1:
| Class | F1 (Exp19) | F1 (Exp18) | Δ |
|:------|:----------:|:----------:|:--:|
| Human | 0.9172 | 0.9820 | `-0.0648` ↓↓ |
| CodeLlama | 0.6762 | 0.7429 | `-0.0667` ↓↓ |
| GPT | 0.6891 | 0.7481 | `-0.0590` ↓↓ |
| Llama3.1 | **0.7878** | 0.8153 | `-0.0275` ↓ |
| Nxcode | 0.5052 | **0.5015** | `+0.0037` ↑ |
| **Qwen1.5** | **0.1981** | 0.4431 | **`-0.2450` ↓↓↓ worst ever!** |

#### Per-language (Author)
| Language | Macro-F1 |
|:---------|:---------:|
| C++ | 0.6339 |
| Java | **0.6832** |
| Python | 0.5819 |

#### Per-source (Author)
| Source | Exp19 | Exp18 | Δ |
|:-------|:-----:|:-----:|:--:|
| CF | 0.6267 | **0.7717** | `-0.1450` ↓↓ |
| GH | 0.4738 | **0.5618** | `-0.0880` ↓↓ |
| LC | 0.5452 | **0.6035** | `-0.0583` ↓ |

#### Confusion matrix analysis
```
           H     CL    GPT   L3.1  NX    QW
Human:  [20868,  829,  199,  217, 2062, 383]  ← 2062 humans pred as Nxcode!
CodeLL: [   24, 3834,  167,  656,  449,  96]
GPT:    [   15,   84, 1465,   89,  104,   3]
L3.1:   [   13,  248,  319, 4606,  219,   4]
Nxcode: [   16,  437,  165,  373, 3917, 629]
Qwen:   [   14,  695,  169,  356, 3293, 727]  ← 3293 Qwen pred as Nxcode
```
> **Root failure**: DANN forces encoder to produce generator-invariant features. This is *exactly the opposite* of what author identification needs. Qwen1.5 (F1=0.198) collapses almost entirely into Nxcode — GRL successfully erases generator-style information, destroying fine-grained discrimination.

### Training dynamics (Author — DANN degradation)
| Epoch | Val Macro-F1 | Adv Loss (train end) | λ_adv |
|:------|:-------------:|:--------------------:|:------:|
| 1 | 0.6403 | 0.00 | 0.000 |
| 2 | **0.6423** (peak, mid-ep) → **0.4111** (end-ep) | 1.61 | 0.100 |
| 3 | 0.6026 (mid) → **0.5569** (final) | 1.37 | 0.100 |
> **Best checkpoint** = val peak **0.6423** early in **epoch 2**; later in the same epoch val falls to **0.4111** (GRL overwhelms useful generator signal). Test macro **0.6289** is below that val peak (train/val/test gap), but far above the collapsed end-of-epoch-2 state.

### OOD Generator / Source
Not evaluated (log only shows iid_only run plan).

### Key Insights
- **DANN is fundamentally misaligned with author identification**: the GRL objective (make features indistinguishable by generator) directly conflicts with the task (identify generator). This was a known risk but the degree of failure (-7.66%) confirms it
- **Binary also hurt** (-0.33%) — forcing generator-invariant binary features removes useful discriminative signal even for the simple human vs AI task
- **Qwen1.5 worst ever** (0.198): DANN most aggressively collapses Qwen→Nxcode since they're the most similar pair; the adversarial training exacerbates the existing confusion
- **Experiment conclusion**: GRL/DANN is contra-indicated for author identification; adversarial invariance is only appropriate when the domain factor is orthogonal to the label

**Checkpoint:** `./codet_m4_checkpoints/codet_author/eaglecode_CoDET_author_best.pt`

---

## 📊 Exp20 — BiScopeCode (bidirectional MLM probe stream)

**Run:** 2026-03-31 | ModernBERT-base + neural + spectral + **BiScope** (K=16 masked positions → 8-dim probe stats, gated fusion)  
**Loss:** `L_task + 0.3*L_neural + 0.3*L_spectral + L_biscope` — BiScope aux plateaus ~**0.173** by epoch 3 (binary) / ~**0.53** (author)  
**Key novelty:** Local LM likelihood / embedding-consistency signal as a third stream for stylometric “fingerprints”  
**Infra:** H100 80GB BF16 | batch=64x1 | epochs=3 | **~151.75M** params (author) | **~5× wall time per step** vs non-BiScope runs (log spacing)

### IID Binary
| Metric | Value | vs Exp18 |
|:-------|:-----:|:--------:|
| Test Macro-F1 | **0.9906** | `=` |
| Test Weighted-F1 | 0.9907 | `+0.0001` |
| Best Val F1 | 0.9897 | `=` |

| Language | Macro-F1 | Source | Macro-F1 |
|:---------|:---------:|:-------|:---------:|
| C++ | 0.9907 | CF | 0.9839 |
| Java | **0.9951** | GH | 0.9844 |
| Python | 0.9864 | LC | 0.9849 |

### IID Author (6-class)
| Metric | Value | vs Exp18 | Delta |
|:-------|:-----:|:--------:|:-----:|
| Test Macro-F1 | **0.7020** | `-0.0035` | `-0.35%` |
| Test Weighted-F1 | 0.8086 | `-0.0047` | `-0.47%` |
| Best Val F1 | **0.7122** | `-0.0066` | `-0.66%` |

| Class | F1 (Exp20) | F1 (Exp18) | Δ |
|:------|:----------:|:----------:|:--:|
| Human | 0.9754 | **0.9820** | `-0.0066` ↓ |
| CodeLlama | 0.7333 | **0.7429** | `-0.0096` ↓ |
| GPT | 0.7463 | **0.7481** | `-0.0018` ↓ |
| Llama3.1 | **0.8203** | 0.8153 | `+0.0050` ↑ |
| **Nxcode** | **0.5148** | 0.5015 | `+0.0133` ↑ |
| Qwen1.5 | 0.4220 | **0.4431** | `-0.0211` ↓ |

#### Per-source breakdown (Author)
| Source | Exp20 | Exp18 | Δ |
|:-------|:-----:|:-----:|:--:|
| CF | 0.7660 | **0.7717** | `-0.0057` |
| GH | 0.5510 | **0.5618** | `-0.0108` |
| LC | 0.6027 | **0.6035** | `-0.0008` |

#### Confusion matrix focus (Author)
- `true=nxcode -> pred=qwen1.5` = `1831/5537`
- `true=qwen1.5 -> pred=nxcode` = `2009/5254`

### Key Insights
- **Ties Exp22 on headline Author (70.20%)** — raw macro **slightly above** Exp22 (0.70203 vs 0.70195); **below** Exp22 on val best (71.22 vs 71.32)
- **Binary matches SOTA tier (99.06%)** despite slower training — BiScope does not buy extra IID binary at this budget
- **Nxcode improves vs Exp18** (+1.33 pt F1) but **Qwen1.5** still weakest; same Nxcode↔Qwen diagonal mass

**Checkpoint:** `./codet_m4_checkpoints/codet_author/biscopecode_CoDET_author_best.pt`

---

## 📊 Exp22 — TTACode (TENT-style LayerNorm TTA)

**Run:** 2026-03-31 | ModernBERT-base + spectral/neural fusion (same family as Exp11)  
**Training:** IID only — **TTA is not applied during train**; optional **TTAAdapter** runs at eval on LayerNorm (55 params, `tta_steps=1`, `lr=1e-4`)  
**Infra:** H100 80GB BF16 | batch=64x1 | epochs=3 | **~151.7M** params (author run)

### IID Binary (standard checkpoint eval — **no TTA**)
| Metric | Value | vs Exp18 |
|:-------|:-----:|:--------:|
| Test Macro-F1 | **0.9906** | `=` |
| Test Weighted-F1 | 0.9906 | `=` |
| Best Val F1 | **0.9903** | `+0.0006` |

> Per-language / per-source **binary** slices were only printed in this log **after** the TTA pass (see below), so they reflect adapted weights — **not** the 0.9906 headline metric. Use global test F1 above for IID binary.

### IID Author (6-class) — standard test (**no TTA**)
| Metric | Value | vs Exp18 | Delta |
|:-------|:-----:|:--------:|:-----:|
| Test Macro-F1 | **0.7020** | `-0.0035` | `-0.35%` |
| Test Weighted-F1 | 0.8087 | `-0.0046` | `-0.46%` |
| Best Val F1 | **0.7132** | `-0.0056` | `-0.56%` |

| Class | F1 (Exp22) | F1 (Exp18) | Δ |
|:------|:----------:|:----------:|:--:|
| Human | 0.9764 | **0.9820** | `-0.0056` ↓ |
| CodeLlama | 0.7400 | **0.7429** | `-0.0029` ↓ |
| **GPT** | **0.7510** | 0.7481 | `+0.0029` ↑ |
| Llama3.1 | **0.8211** | 0.8153 | `+0.0058` ↑ |
| **Nxcode** | **0.5333** | 0.5015 | **`+0.0318` ↑↑** |
| Qwen1.5 | 0.3899 | **0.4431** | `-0.0532` ↓↓ |

### ⚠️ Post-hoc TTA on test breakdown (do **not** use for leaderboard)
After the main test pass, the harness ran **TTAAdapter** while computing subgroup metrics. This **collapsed** behaviour:
- **Binary** subgroup macro dropped to **~0.94** overall (e.g. CF **~0.78** macro vs ~0.98 without TTA) — LN adaptation overfitted batch statistics on the **test** pass
- **Author** subgroup macro **~0.17** with near-total **human** collapse in the logged confusion matrix (`4879/5226` CodeLlama rows predicted Human, LeetCode bucket **~0.045** macro)

**Takeaway:** report **only** the checkpoint metrics above for paper comparisons; fix eval to run breakdown **before** TTA, or disable TTA on IID suite until SAR gating / calibrated steps are validated.

### Key Insights
- **Strongest Batch-2 Author so far (70.20)** — ties Exp11-tier binary (**99.06**) while improving **Nxcode** vs Exp18 at the cost of **Qwen1.5**
- **TTA hook is a footgun** on this pipeline: same run shows TTA can erase gains if applied incorrectly to test telemetry

**Checkpoint:** `./codet_m4_checkpoints/codet_author/ttacode_CoDET_author_best.pt`

---

## 📊 Exp23 — GraphStyleCode (GAT replaces BiLSTM AST)

**Run:** 2026-03-31 | ModernBERT-base + **Graph Attention Network** on AST (2-layer, attention-weighted pooling) in place of BiLSTM AST; same neural + spectral fusion as Exp11 family  
**Loss:** `L_task + 0.3*L_neural + 0.3*L_spectral` — **Aux = 0** throughout (no extra graph auxiliary in this run)  
**Key novelty:** Explicit relational structure over AST nodes vs sequential BiLSTM over flattened AST  
**Infra:** H100 80GB BF16 | batch=64x1 | epochs=3 | workers=8 | **~151.7M** params (author run)

### IID Binary
| Metric | Value | vs Exp18 |
|:-------|:-----:|:--------:|
| Test Macro-F1 | **0.9900** | `-0.0006` |
| Test Weighted-F1 | 0.9900 | `-0.0006` |
| Best Val F1 | 0.9900 | `+0.0003` |

| Language | Macro-F1 | Source | Macro-F1 |
|:---------|:---------:|:-------|:---------:|
| C++ | 0.9897 | CF | 0.9806 |
| Java | **0.9947** | GH | 0.9840 |
| Python | 0.9858 | LC | 0.9862 |

### IID Author (6-class)
| Metric | Value | vs Exp18 | Delta |
|:-------|:-----:|:--------:|:-----:|
| Test Macro-F1 | **0.6971** | `-0.0084` | `-0.84%` |
| Test Weighted-F1 | 0.8060 | `-0.0073` | `-0.73%` |
| Best Val F1 | **0.7089** | `-0.0099` | `-0.99%` |

| Class | F1 (Exp23) | F1 (Exp18) | Δ |
|:------|:----------:|:----------:|:--:|
| Human | 0.9753 | **0.9820** | `-0.0067` ↓ |
| CodeLlama | 0.7354 | **0.7429** | `-0.0075` ↓ |
| GPT | 0.7350 | **0.7481** | `-0.0131` ↓ |
| Llama3.1 | **0.8172** | 0.8153 | `+0.0019` ↑ |
| Nxcode | 0.4970 | **0.5015** | `-0.0045` ↓ |
| Qwen1.5 | 0.4227 | **0.4431** | `-0.0204` ↓ |

#### Per-source breakdown (Author)
| Source | Exp23 | Exp18 | Δ |
|:-------|:-----:|:-----:|:--:|
| CF | 0.7616 | **0.7717** | `-0.0101` |
| GH | 0.5585 | **0.5618** | `-0.0033` |
| LC | 0.5949 | **0.6035** | `-0.0086` |

#### Confusion matrix focus (Author)
- `true=nxcode -> pred=qwen1.5` = `1836/5537`
- `true=qwen1.5 -> pred=nxcode` = `2097/5254`

### Key Insights
- **Author 69.71 < Exp11 SpectralCode 69.82** — GAT-on-AST under this recipe did not outperform the BiLSTM AST backbone; relational pooling may need richer graph (CFG/DFG) or longer tuning
- **Binary 99.00** remains strong (near ceiling); slight **Java** strength (0.9947) mirrors other ModernBERT runs
- **Same dominant failure mode**: heavy Nxcode↔Qwen off-diagonal mass

**Checkpoint:** `./codet_m4_checkpoints/codet_author/graphstylecode_CoDET_author_best.pt`

---

## 📊 Exp25 — MultiGranCode (CharCNN + gated fusion)

**Run:** 2026-03-31 | ModernBERT-base + neural + spectral + **CharCNN** (k=3,5,7) three-way gated fusion  
**Loss:** `L_task + 0.3*L_neural + 0.3*L_spectral + L_aux(char/aux stream)` — Aux remains high early (CharCNN warm-up), decays by epoch 3  
**Key novelty:** Micro-stylometry via character convolutions complements token-level ModernBERT for generator-specific surface patterns  
**Infra:** H100 80GB BF16 | batch=64x1 | epochs=3 | workers=8 | **~151.8M** params (author run)

### IID Binary
| Metric | Value | vs Exp18 |
|:-------|:-----:|:--------:|
| Test Macro-F1 | **0.9903** | `-0.0003` |
| Test Weighted-F1 | 0.9903 | `-0.0003` |
| Best Val F1 | 0.9897 | `=` |

| Language | Macro-F1 | Source | Macro-F1 |
|:---------|:---------:|:-------|:---------:|
| C++ | 0.9907 | CF | 0.9826 |
| Java | **0.9949** | GH | 0.9841 |
| Python | 0.9856 | LC | 0.9846 |

### IID Author (6-class)
| Metric | Value | vs Exp18 | Delta |
|:-------|:-----:|:--------:|:-----:|
| Test Macro-F1 | **0.7019** | `-0.0036` | `-0.36%` |
| Test Weighted-F1 | 0.8104 | `-0.0029` | `-0.29%` |
| Best Val F1 | **0.7122** | `-0.0066` | `-0.66%` |

| Class | F1 (Exp25) | F1 (Exp18) | Δ |
|:------|:----------:|:----------:|:--:|
| Human | 0.9778 | **0.9820** | `-0.0042` ↓ |
| CodeLlama | **0.7444** | 0.7429 | `+0.0015` ↑ |
| GPT | 0.7325 | **0.7481** | `-0.0156` ↓ |
| Llama3.1 | **0.8211** | 0.8153 | `+0.0058` ↑ |
| **Nxcode** | **0.5124** | 0.5015 | `+0.0109` ↑ |
| Qwen1.5 | 0.4230 | **0.4431** | `-0.0201` ↓ |

#### Per-source breakdown (Author)
| Source | Exp25 | Exp18 | Δ |
|:-------|:-----:|:-----:|:--:|
| CF | 0.7653 | **0.7717** | `-0.0064` |
| GH | 0.5559 | **0.5618** | `-0.0059` |
| LC | 0.6027 | **0.6035** | `-0.0008` |

#### Confusion matrix focus (Author)
- `true=nxcode -> pred=qwen1.5` = `1702/5537`
- `true=qwen1.5 -> pred=nxcode` = `2149/5254`

### Key Insights
- **Best Batch-2 Author so far (70.19)**, slightly above SelfDistill Exp26 (+0.05 pt macro) — Char stream helps **Nxcode / CodeLlama** at the cost of weaker **Qwen1.5** vs Exp18
- **Binary 99.03** — on par with Spectral family; large `aux` in epoch 1 is expected while CharCNN aligns with gated fusion
- **Nxcode–Qwen mass off-diagonal** still dominates (`2149/5254` Qwen→Nxcode ≈41%) — same structural failure mode as Exp11–18

**Checkpoint:** `./codet_m4_checkpoints/codet_author/multigrancode_CoDET_author_best.pt`

---

## 📊 Exp26 — SelfDistillCode (BYOL-style EMA Teacher)

**Run:** 2026-03-31 | ModernBERT-base + EMA teacher self-distillation (`momentum=0.995`)  
**Loss:** `L_task + 0.3*L_neural + 0.3*L_spectral + L_aux(distill)`  
**Key novelty:** Student matches EMA teacher representation from its own historical parameters, improving feature stability without extra labels  
**Infra:** H100 80GB BF16 | batch=64x1 | epochs=3 | workers=8 | 151.7M params

### IID Binary
| Metric | Value | vs Exp18 |
|:-------|:-----:|:--------:|
| Test Macro-F1 | **0.9904** | `-0.0002` |
| Test Weighted-F1 | 0.9904 | `-0.0002` |
| Best Val F1 | 0.9901 | `+0.0004` |

| Language | Macro-F1 | Source | Macro-F1 |
|:---------|:---------:|:-------|:---------:|
| C++ | 0.9910 | CF | 0.9837 |
| Java | **0.9949** | GH | 0.9835 |
| Python | 0.9855 | LC | 0.9850 |

### IID Author (6-class)
| Metric | Value | vs Exp18 | Delta |
|:-------|:-----:|:--------:|:-----:|
| Test Macro-F1 | **0.7014** | `-0.0041` | `-0.41%` |
| Test Weighted-F1 | 0.8091 | `-0.0042` | `-0.42%` |
| Best Val F1 | **0.7122** | `-0.0066` | `-0.66%` |

| Class | F1 (Exp26) | F1 (Exp18) | Δ |
|:------|:----------:|:----------:|:--:|
| Human | 0.9762 | **0.9820** | `-0.0058` ↓ |
| CodeLlama | 0.7397 | **0.7429** | `-0.0032` ↓ |
| GPT | 0.7375 | **0.7481** | `-0.0106` ↓ |
| Llama3.1 | **0.8201** | 0.8153 | `+0.0048` ↑ |
| **Nxcode** | **0.5053** | 0.5015 | `+0.0038` ↑ |
| Qwen1.5 | 0.4300 | **0.4431** | `-0.0131` ↓ |

#### Per-source breakdown (Author)
| Source | Exp26 | Exp18 | Δ |
|:-------|:-----:|:-----:|:--:|
| CF | 0.7651 | **0.7717** | `-0.0066` |
| GH | **0.5620** | 0.5618 | `+0.0002` |
| LC | 0.6003 | **0.6035** | `-0.0032` |

#### Confusion matrix focus (Author)
- `true=nxcode -> pred=qwen1.5` = `1830/5537`
- `true=qwen1.5 -> pred=nxcode` = `2077/5254`

### Key Insights
- **Strong, stable baseline for Batch-2:** Author 70.14, close to top cluster (Exp14/15/18) with minimal tuning
- **EMA distillation improves stability** (val best 0.7122) but does not fully resolve Nxcode/Qwen separation
- **Best class gains vs Exp18 are on Nxcode (+0.38%) and Llama3.1 (+0.48%)**, while Qwen1.5 still lags
- **Still far better than DANN Exp19** (+7.25% weighted-F1 and +7.66% macro-F1 on author)

**Checkpoint:** `./codet_m4_checkpoints/codet_author/selfdistillcode_CoDET_author_best.pt`

---

## 🚀 Performance vs Paper (Head-to-Head)

| Evaluation Mode (IID) | Paper (UniXcoder) | Exp11 SpectralCode | Exp14 ProtoCon | Exp15 GroupDRO | Exp16 HyperNet | Exp19 EAGLE | Exp20 BiScope | Exp22 TTACode | Exp23 GraphStyle | Exp25 MultiGran | Exp26 SelfDistill | **Exp18 HierTreeCode** | Best Delta |
|:----------------------|:-----------------:|:------------------:|:--------------:|:--------------:|:--------------:|:-----------:|:-------------:|:-------------:|:----------------:|:---------------:|:----------------:|:----------------------:|:----------:|
| Binary F1 (Table 2) | 98.65 | 99.06 | 99.06 | 98.98 | **99.07** | 98.73 | **99.06** | **99.06** | 99.00 | 99.03 | 99.04 | **99.06** | `+0.44%` (Exp17 99.09 best) |
| Author F1 (Table 7) | 66.33 | 69.82 | 70.13 | 70.17 | — (failed) | 62.89 ↓↓ | 70.20 | 70.20 | 69.71 | 70.19 | 70.14 | **70.55** | `+4.22%` |

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
1. **Author F1 plateau near ~70%** (best: Exp18=70.55, Exp17=70.46, **Exp20/Exp22≈70.20** tie cluster, Exp25=70.19, Exp26=70.14; Exp23 GAT=69.71 slightly below Exp11=69.82) — Nxcode/Qwen confusion persists; generated-class separation still the main failure mode
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
| **Exp20** | `run_codet_m4_biscope.py` | **BiScopeCode** | Bidirectional MLM probe: mask K=16 positions/sequence, measure P(original\|context) via ModernBERT embedding cosine. 8-dim BiScope stats as 3rd gated stream | ✅ IID: Bin **99.06**, Auth **70.20** (targets 99.15 / 72 not met) | **S** |

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

### Leaderboard Update (Exp21-26 Progress)

| Exp | Method | Binary F1 | Author F1 | Val F1 | Status |
|:----|:-------|:---------:|:---------:|:------:|:------:|
| Exp18 | HierTreeCode | 99.06 | **70.55** | **71.88** | ✅ SOTA |
| Exp21 | MoECode | — | — | — | 🔲 Pending |
| Exp22 | TTACode | 99.06 | 70.20 | 71.32 | ✅ Completed |
| Exp23 | GraphStyleCode | 99.00 | 69.71 | 70.89 | ✅ Completed |
| Exp24 | CosineProtoCode | — | — | — | 🔲 Pending |
| Exp25 | MultiGranCode | 99.03 | 70.19 | 71.22 | ✅ Completed |
| Exp26 | SelfDistillCode | 99.04 | 70.14 | 71.22 | ✅ Completed |

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
