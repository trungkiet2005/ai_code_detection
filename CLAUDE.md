---
description: 
alwaysApply: true
---

# CLAUDE.md — Repo Briefing (EMNLP 2026 Main target)

> **Read this file first.** One-stop theoretical + operational context.
> If anything here disagrees with the code, trust the code and fix this file.

---

## 0. Current submission target -- UPDATED 2026-05-10 (theory-pivot locked + extended novelty framework)

> **Venue:** EMNLP 2026 Main (long paper, 8 pages). **Reach for Oral.**
> **Deadline:** ~2026-05-26.
> **Status (2026-05-10):** Few-shot CoDET-M4 headline locked; theory framing is now the paper spine.
> **Working title:** *Few-Shot AI-Code Attribution via Hierarchical Target-Kernel Learning*

**Single sentence:**

> We are not just improving an AI-code detector. We are using CoDET-M4 authorship and AICD model-family attribution as public testbeds for one theory contribution: **model genealogy can be written as a target kernel, and aligning few-shot representations to that kernel gives a scale-aware inductive bias for LLM attribution.**

The benchmark numbers are evidence. The contribution is the attribution principle.

**4 contributions (updated):**
- **C1 (theory headline)** -- **Hierarchical Target-Kernel Alignment (HTKA / Hier-NTK)**: define a genealogy-aware target kernel over attribution labels and align the encoder representation Gram matrix to it. This is the paper's new object, not a bag of tricks.
- **C2 (empirical headline)** -- With **5% of training data (~25K samples)** on ModernBERT-base, Hier-NTK reaches **0.6709 CoDET-M4 6-class Author Macro-F1**, exceeding paper UniXcoder's full-data 0.6633 by **+0.0076 with 1/20 the budget**.
- **C3 (regime law)** -- Below ~5K samples, encoder pretraining dominates; above ~5K samples, the genealogy target-kernel prior dominates. This phase transition is a claim to explain, not just a table entry.
- **C4 (scope validation)** -- AICD T2 model-family attribution is the secondary authorship-like stress test. Droid T3/T4 are detection/adversarial robustness checks only, not the paper's central authorship claim.

**Locked headline numbers (CoDET-M4 6-class Author IID Macro-F1):**

| Bench / Setup | Method | Score | Δ vs paper UniXcoder full (0.6633) |
|:--|:--|:-:|:-:|
| 5% training data | **🥇 FS-Hier-NTK** | **0.6709** | **+0.0076** |
| 5% training data | 🥈 FS-HierTree | 0.6682 | +0.0049 |
| 5% training data | 🥉 FS-NTKAlign | 0.6652 | +0.0019 |
| 5% training data | 4 FS-Focal | 0.6616 | −0.0017 |
| 5% training data | 5 FS-Baseline-UniXcoder reimpl | 0.6512 | −0.0121 |
| 1% training data | 🥇 FS-Baseline-UniXcoder reimpl | 0.5744 | (encoder-pretrain regime) |
| K=128 (~768 samples) | 🥇 FS-Focal | 0.3749 | (class-imbalance regime) |
| 20% training data (existing Exp_Climb) | Exp_13 NTKAlign | 0.7103 | +4.70 |
| Full data (existing Exp_CodeDet) | Exp_27 DeTeCtive | 0.7153 | +5.20 |
| Droid T3 (lean 20%, existing) | Exp_04 Poincare | 0.8976 W-F1 | +0.98 vs DroidDetectCLS-Large 88.78 |

**Out-of-scope (DO NOT add to headline):**
- AICD T1 binary robust detection (universal val->test collapse, mention in Limitations only).
- AICD T3 fine-grained human/machine/hybrid/adversarial classification (appendix only if run).
- Droid T3/T4 as "authorship" (they are detection/robustness tasks, not generator authorship).
- Zero-Shot suite (in `legacy/`, reproduction gap −32 pt vs paper FDG).
- OOD-Source-gh > 0.40 (NeurIPS Oral threshold; for EMNLP Main framed as "characterisation" not "claim").

**Repo cleanup done 2026-05-08 (commit `89ba63d`):**
- `Exp_ZeroShot/`, `Exp_TK/`, 21 weak `Exp_DM/` files moved to `legacy/`.
- Active suites: `Exp_Climb/` + `Exp_CodeDet/` + `Exp_DM/` (5 champions: exp07/09/11/13/18) + `Exp_FewShot/` (new).

**Hardware switch:** Kaggle **T4 16GB** is now first-class (Exp_FewShot is T4-native bs=16 fp16 seq=384). H100 still preferred for `Exp_Climb/` runs but no longer required.

**⚡ RTX6000 Ada Lovelace (48GB) speedup:** Detected automatically → bs=64, bf16, seq=512. **~3-4× faster** than T4.
**⚡ RTX 96GB speedup:** Detected automatically → bs=128, bf16, seq=512. **~6-8× faster** than T4.

**📦 Kaggle offline datasets (no-internet, avoids conflict + bias):**
- CoDET-M4: `/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet`
- DroidCollection: `/kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/`
- AICD-Bench: `/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench/`
  - Structure: `T1/`, `T2/`, `T3/` subdirectories with parquet files
  - Each task loads from its respective subdirectory automatically
- Encoders: `/kaggle/input/datasets/chiboiz/ai-detection-encoders/models/`

**Out-of-scope for this paper (do NOT add to headline):**
- Zero-shot detection (Exp_ZeroShot, **moved to legacy/** — reproduction gap −32 pt vs paper Fast-DetectGPT).
- AICD T1 binary robust detection (universal val->test collapse, **discussion-only mention** in Limitations).
- AICD T3 and Droid T4 unless used as appendix robustness stress tests.
- OOD-Source-gh > 0.40 (NeurIPS Oral threshold; for EMNLP Main framed as "characterisation" not "claim").

**Repo cleanup done 2026-05-08 (commit `89ba63d`):**
- `Exp_ZeroShot/`, `Exp_TK/`, 21 weak `Exp_DM/` files moved to `legacy/`.
- Active suites: `Exp_Climb/` + `Exp_CodeDet/` + `Exp_DM/` (5 champions: exp07/09/11/13/18) + `Exp_FewShot/` (new).

---

## 0bis. Original NeurIPS framing (kept for context)

The repository was originally built targeting NeurIPS 2026 Oral with the source-confounding research question. We reframed to EMNLP Main when:
- 17 days to NeurIPS-like deadline made oral-tier breakthrough (OOD-gh > 0.40) unrealistic.
- Existing 20%-data SOTA (+4.70 / +5.20 / +0.98) is already a publishable EMNLP claim.
- Source confounding becomes §5 Analysis ("characterise") instead of §3 Method ("solve").

The §1–§4 narrative below remains the **why** behind the methods. It now serves as paper §5 Analysis material rather than the headline pitch.

---

## 1. The central research question

> **How should a few-shot attribution model use the known genealogy of LLM generators instead of treating every author label as unrelated?**

Prior work on CoDET-M4, DroidCollection, and AICD-Bench mostly frames AI-code work as detection accuracy: human vs AI, or flat multi-class author labels. That misses the structure reviewers will recognise as real: generator labels are not exchangeable. GPT-family, Llama-family, CodeLlama-family, Qwen-family, and fine-tuned descendants form a prior over which mistakes are semantically closer than others.

The paper's core claim is therefore not "we tried a stronger loss." The claim is that **LLM attribution is a structured-label problem**, and model genealogy gives a target similarity geometry that can be imposed even in the few-shot regime.

## 2. Thesis (one sentence)

> **Few-shot AI-code attribution fails when it learns a flat simplex over generator labels; it improves when representation geometry is aligned to a genealogy-derived target kernel over authors/model families.**

CoDET-M4 Author is the main generator-level attribution benchmark. AICD T2 model-family attribution is the natural coarser version of the same idea. Droid T3/T4 are useful only as robustness checks because their labels are detection/adversarial states, not model authors.

### 2.1 The new object

For a batch of examples with labels $y_i$, define a target genealogy kernel:

```
T_ij = k_tree(y_i, y_j)
```

where `k_tree` is high for the same author, intermediate for related model families, and low for unrelated human/model branches. Given normalized embeddings `z_i`, the method aligns the representation Gram matrix to this target:

```
L_htka = 1 - cos(vec(ZZ^T), vec(T))
```

This is the current paper object: **Hierarchical Target-Kernel Alignment**. HierTree is the prior; NTK/Gram alignment is the witness that the representation actually obeys it.

### 2.2 Theory-grounded novelty bar

Every "novel" method for this paper must be stated as one mathematical object with one falsifiable property. **This is no longer just hierarchical — we welcome ANY theory-grounded approach** from:

**Theory Sources (arxiv-anchored):**
1. **Kernel Methods:** TKM (`NeurIPS 2024`, `2410.06171`) — kernel complexity vs alignment
2. **Neural Collapse:** NC (`ICLR 2025`, `2410.04887`) — ETF geometry, class-mean collapse
3. **Information Theory:** VIB (`Alemi ICLR 2017`), CIB (`2410.00535`) — KL regularisation
4. **Contrastive Learning:** SupCon (`IEEE 2020`), SimCLR (`2020`) — contrastive alignment
5. **Causal Inference:** IRM (`Arjovsky 2019`), CausalIB (`ICLR 2025`) — invariant learning
6. **Geometric Deep Learning:** CUSP (`ICLR 2025`, `2502.00401`) — hyperbolic/curvature
7. **Representation Alignment:** Unified alignment theory (`ICLR 2025`, `2502.14047`)

**Required proposal block for new theory-track experiments:**

```
NAME            : <Capitalised noun phrase. The new object.>
ARXIV_ID        : <Relevant paper(s) from above with arxiv ID>
ONE-LINE CLAIM  : <The thesis. No "and".>
EQUATION        : <The defining equation or operator.>
PROPERTY        : <The provable or testable property that justifies attribution.>
WHY NOT BEFORE  : <Closest prior work and the exact gap.>
FALSIFIER       : <The single experiment that would refute the claim.>
```

For Hier-NTK, the falsifier is concrete: if the representation-target alignment score rises while sibling/family attribution errors do not fall, the genealogy-kernel story is wrong and the result is only empirical tuning.

Canonical header style for the current headline method:

```python
# =============================================================================
# Theory-Track exp -- Hierarchical Target-Kernel Alignment (HTKA / Hier-NTK):
# closed-form witness for the genealogy prior in few-shot AI-code attribution.
#
# ARXIV_ID      : NeurIPS 2024 TKM (2410.06171) for kernel alignment theory;
#                 ICLR 2025 NC (2410.04887) for class-mean geometry.
# NAME          : Hierarchical Target-Kernel Alignment (HTKA).
# ONE-LINE CLAIM: Few-shot LLM attribution improves when representation
#                 similarity is aligned to model genealogy.
# EQUATION      : L_htka = 1 - cos(vec(ZZ^T), vec(T_tree))
#                 where z_i are L2-normalised embeddings and
#                 T_tree[i,j] = k_tree(y_i, y_j).
# PROPERTY      : If attribution labels are generated from a hierarchical
#                 model-family prior, then T_tree is a lower-variance target
#                 than one-hot labels in the few-shot regime; aligning ZZ^T
#                 to T_tree reduces sibling/family confusion without requiring
#                 full-data estimation of every class boundary.
# WHY NOT BEFORE: Code authorship baselines treat generator labels as a flat
#                 simplex. Hierarchical classifiers use the tree only at the
#                 output. HTKA uses genealogy as a representation-level target
#                 kernel and tests the alignment/error relation directly.
# FALSIFIER     : Track target-kernel alignment, sibling-pair error, and
#                 Macro-F1. If alignment rises but sibling/family errors do
#                 not fall, the genealogy-kernel theory is false.
# =============================================================================
```

### 2.3 New experiments (2026-05-10, theory-anchored)

New `exp_nNN_*.py` files with arxiv-grounded theory:

| File | Method | ArXiv | Theory Hook | Key Equation |
|:--|:--|:--|:--|:--|
| `exp_n05_focal.py` | Focal-CE | Lin 2017 | Hard example mining | \(L_focal = -(1-p_t)^\gamma \log(p_t)\) |
| `exp_n06_attn_pool.py` | HAP | Lee 2017 | Hierarchical attention | \(z = \sum_i \alpha_i h_i\) |
| `exp_n07_mixup_align.py` | MGA | Arjovsky 2019 | IRM source invariance | \(L + \lambda \|\nabla_z L - E[\nabla_z L]\|^2\) |
| `exp_n08_ortho_clf.py` | Ortho-CLF | Papyan 2020 | Neural collapse prevention | Gram-Schmidt orthogonalisation |
| `exp_n09_etf_simplex.py` | ETF-Simplex | NC theory | Optimal K-class ETF | \(W_{etf} = \sqrt{K/(K-1)}(e_i - 1/K)\) |
| `exp_n10_vib.py` | VIB | Alemi 2017 | Info bottleneck KL | \(L_{vib} = CE + \beta \cdot KL(q\|p)\) |
| `exp_n11_mixup_ce.py` | Mixup-CE | Zhang 2018 | Boundary smoothing | \(\lambda y_1 + (1-\lambda)y_2\) |
| `exp_n12_label_smooth.py` | LabelSmooth | Pereyra 2017 | Uncertainty calibration | \(t_i = (1-\epsilon)y_i + \epsilon/K\) |

## 2.4 Novel Theorems (self-derived for EMNLP Oral)

**IMPORTANT:** For EMNLP Oral, we can and SHOULD derive our OWN theorems. This is stronger than citing existing work because:
- Existing work → reviewers already know
- Novel theorem → reviewers must read carefully, harder to reject

### Theorem 1: Few-Shot Kernel Alignment Generalization Bound

**NAME:** Sample Complexity of Genealogical Target Kernel Alignment

**STATEMENT:**
Let \(T \in \mathbb{R}^{K \times K}\) be the genealogical target kernel with condition number \(\kappa(T) = \lambda_{\max}(T) / \lambda_{\min}(T)\). Let \(Z \in \mathbb{R}^{n \times h}\) be the embedding matrix. Then with probability at least \(1-\delta\) over the training set:

\[
R(\hat{h}) \leq \underbrace{O\left(\sqrt{\frac{h \cdot \log(1/\delta)}{n}}\right)}_{\text{standard bound}} + \underbrace{\frac{C}{\kappa(T)} \cdot \|\Delta_K\|_2}_{\text{genealogical correction}}
\]

where \(\Delta_K = ZZ^T - T\) is the kernel alignment error.

**WHY NOVEL:** Existing kernel methods (TKM, EigenPro) assume stationary kernels. Our bound shows genealogical structure (via \(\kappa(T)\)) directly modulates the excess risk.

**IMPLICATION:** Classes closer in the genealogy tree (smaller tree distance) have tighter bounds → explains WHY Hier-NTK helps on sibling pairs.

**FALSIFIER:** If \(\|\Delta_K\| ↑\) but test error does not ↑ proportionally, the genealogical correction term is wrong.

---

### Theorem 2: Phase Transition Threshold

**NAME:** Critical Sample Complexity for Genealogical Prior Dominance

**STATEMENT:**
There exists a critical sample size \(n^*\) such that:

\[
n^* = \Theta\left(\frac{K \cdot h}{\lambda_{\min}(T)^2}\right)
\]

For \(n \ll n^*\): encoder pretraining dominates (embedding quality matters).
For \(n \gg n^*\): genealogical prior dominates (label structure matters).

**WHY NOVEL:** Existing few-shot theory (e.g., Many-Shot ICL) does not predict phase transitions based on label structure. Our theorem links the phase transition to the kernel condition number.

**IMPLICATION:** Predicts WHY we observe transition at ~5K samples on CoDET-M4.

**FALSIFIER:** If phase transition does not occur at \(n^*\) empirically, the threshold formula is wrong.

---

### Theorem 3: Sibling Confusion Bound

**NAME:** Bounding Generational Pair Confusion Error

**STATEMENT:**
For any sibling pair \((i, j)\) in the genealogical tree (sharing parent node), let \(d_T(i,j)\) be their tree distance. Then:

\[
P(\hat{y} = j \mid y = i) \leq \frac{d_T(i,j)^{-1}}{\sum_{k \neq i} d_T(i,k)^{-1}} \cdot (1 + \epsilon_n)
\]

where \(\epsilon_n = O(\sqrt{\log K / n})\) is the sampling error.

**WHY NOVEL:** Existing attribution bounds (e.g., nearest-neighbor, SVM margins) do not account for genealogical proximity. Our bound shows sibling confusion is inversely proportional to tree distance.

**IMPLICATION:** Quantifies WHY Qwen↔Nxcode (sibling pair) is the hardest confusion pair.

**FALSIFIER:** If sibling pairs with larger tree distance have higher confusion error, the tree-distance formulation is wrong.

---

### 2.5 How to derive novel theorems

**Process:**
1. Start from a known bound (Rademacher complexity, covering numbers, etc.)
2. Introduce genealogical structure via the target kernel \(T\)
3. Relate the new terms to observable quantities (tree depth, kernel eigenvalues)
4. Derive testable predictions (phase transition, confusion hierarchy)

**Template for new theorem docstring:**

```python
# =============================================================================
# THEORY ORIGINAL — [Theorem Name]:
# [1-2 sentence description]
#
# NAME          : [Theorem name]
# TYPE          : generalization_bound | phase_transition | confusion_bound
# STATEMENT     : [Formal mathematical statement]
# KEY_INSIGHT   : [Why this matters for attribution]
# DERIVATION    : [Sketch of proof from known results]
# PREDICTIONS   : [Testable empirical predictions]
# FALSIFIER    : [What experiment would refute this]
# =============================================================================
```

## 3. Structural causal model we operate in

```
        ┌────────────┐
        │   Author   │  Y  ∈ {human, gpt, llama3.1, codellama, nxcode, qwen1.5}
        └─────┬──────┘
              │ generative process
              ▼
┌──────────┐  ┌────────────┐  ┌──────────┐
│ Language │─►│  Code X    │◄─│  Source  │   S ∈ {cf, lc, gh}
│    L     │  │            │  │   (dom.) │
└──────────┘  └─────┬──────┘  └──────────┘
                    │
                    ▼
                 Detector
                  f(X) → Ŷ
```

- $S$ is a **fork / confounder** on $(X, Y)$: source selection affects both which authors submit (CF/LC favour competitive problems human-written and LLM-refined) and the surface style of $X$ (indentation, identifier conventions, template boilerplate).
- Identifying $P(Y \mid \operatorname{do}(X))$ requires blocking the back-door path $Y \leftarrow S \to X$. The methods in this repo implement three complementary back-door interventions (counterfactual swap, backdoor adjustment, IV orthogonality), plus non-causal priors that compensate when do-operations are weak.
- **Null alternative the reviewers will raise:** "Is this just a covariate-shift / class-imbalance story?" The ablation plan (§6) explicitly separates the three.

## 4. Theoretical axes — our ablation spine

Every method in `Exp_Climb/exp_NN_*.py` is a controlled perturbation along **exactly one** of these axes. The paper's Table 2 is the matrix of (axis × method × Δ-F1); the paper's *story* is which axis moves OOD the most.

| Axis | Hypothesis it tests | Current evidence |
|:--|:--|:--|
| **A. Genealogy prior** | $P(Y)$ factorises along the fine-tune tree; pulling siblings together breaks the Qwen↔Nxcode confusion | HierTree alone explains +3.6 pt IID Author over UniXcoder (insight #2) |
| **B. Spectral / multi-scale** | Human ≠ LLM frequency statistics in AST / token streams | Persistent Homology (Exp_11) sets current GH-OOD record 35.56 |
| **C. Source-invariance (do(S))** | Back-door adjust on $S$ to recover $P(Y \mid \operatorname{do}(X))$ | Exp_02 IRM gains GH; Exp_18 causal stack under-delivers on IID author (70.19), OOD-GH not yet run — **open** |
| **D. Density / generative margin** | Class-conditional manifolds carry author signal | Flow-matching (Exp_06) is climb #2, 70.90 Author |
| **E. Compressibility / info** | LLM outputs are lower-entropy / more predictable than human | Epiplexity (Exp_09) climb #3, best Droid T4 |
| **F. Optimisation geometry** | Flat minima / NTK alignment improve OOD generalisation | NTKAlign (Exp_13) climb #1, 71.03 Author |
| **G. Data distribution** | Training distribution itself is the leak; reshape it | Exp_14 GHCurriculum pending — first data-side intervention |
| **H. Test-time adaptation** | BN/LN stats on GH never match train; adapt at inference | Exp_17 has partial ablation-only CoDET IID result (no_teacher_distill 70.89); full OOD/Droid evaluation pending |

The oral claim must reduce to one sentence of the form "axis X gives $\Delta$ on OOD when controlling for axes Y, Z". The tracker (`Exp_Climb/tracker.md`) is the running evidence log.

## 5. Benchmarks (role-specified)

| Short name | Role in the paper | Primary metric | Why this metric |
|:--|:--|:--|:--|
| **CoDET-M4 Author** (ACL'25) | **Main evaluation** — generator-level AI-code authorship attribution | **Macro-F1** (6-class author) | Class-balanced; directly matches the paper's attribution claim; published UniXcoder baseline |
| **AICD-Bench T2** | **Secondary attribution stress test** — model-family attribution | **Macro-F1** (12-class family) | Authorship-like but coarser than CoDET; tests whether the genealogy-kernel idea survives a different label taxonomy |
| **DroidCollection T3** (EMNLP'25) | Supporting detection benchmark | **Weighted-F1** (3-class) | Detection/machine-refinement robustness under class imbalance; do not call this authorship |
| **DroidCollection T4** | Appendix robustness stress test | **Weighted-F1** (4-class incl. adversarial) | Adversarial/humanised-machine robustness; useful only after CoDET + AICD T2 are stable |
| **AICD-Bench T1/T3** | Limitation / appendix only | Macro-F1 | T1 has known val->test collapse; T3 is fine-grained detection, not central attribution |

**Protocol (non-negotiable):**
- Train separate model per benchmark. Never mix.
- Report the full metric pack: Primary, Macro-F1, Weighted-F1, Macro-R, Weighted-R, Accuracy, per-class.
- Report **val and test** side-by-side. The val–test gap is itself a diagnostic (insight #4).
- Hardware: **H100 80GB, BF16, batch 64×1, seq 512**.
- Each `exp_NN_*.py` is **standalone** and Kaggle-runnable.
- Data-efficiency framing: **train on 5% for the headline**, evaluate on 100% test. Any positive delta vs a full-data paper baseline is a paper claim only when the task matches the paper scope.

**Experiment priority order:**
1. `codet_m4` -- headline generator-level authorship attribution.
2. `aicd_t2` -- secondary model-family attribution stress test.
3. `droid_t3` -- supporting detection transfer.
4. `droid_t4` -- appendix adversarial robustness.
5. `aicd_t3` -- optional appendix only.

## 6. Ablation plan (for the Oral)

An Oral needs ablations that falsify hypotheses, not ablations that tune λ's. For each axis in §4:

1. **Main effect.** Method with all components vs UniXcoder / DroidDetectCLS-Large paper baseline on IID + OOD.
2. **Component drop.** One λ → 0 per row. Δ tells us whether that component carries the axis.
3. **Cross-axis control.** Swap the axis-defining component for a non-causal / non-theoretical equivalent (e.g. replace `lambda_cf` with a random-pair swap). Must eliminate the gain.
4. **Null / shortcut probe.** Train a 1-layer linear probe on the final embedding to predict $S$. The theory says $\Pr(\hat S \mid \phi(X))$ should drop under axis-C methods. If the probe still succeeds, axis C did not identify $\operatorname{do}(S)$.

**Current state of §6 for Exp_18 (example, see tracker):** (1) done (70.19 IID), (2) done (component Δ ≤ 0.3 pt each), (3) **not run**, (4) **not run**. Two missing pieces before a causal claim.

## 7. Repo layout (post-cleanup 2026-05-08)

```
ai_code_detection/
├── README.md
├── Exp_Climb/                           # MAIN training suite — 20%-data climb (CoDET + Droid)
│   ├── tracker.md                       # Primary evidence log. 14 methods cluster ≥70 Author IID.
│   ├── _common.py / _features.py / _model.py / _trainer.py
│   ├── _data_codet.py / _data_droid.py
│   ├── _climb_runner.py / _paper_table.py / _ablation.py
│   └── exp_NN_<method>.py               # ONE method per file, thin wrapper
├── Exp_CodeDet/                         # CoDET-M4 full-data runs (Exp27 DeTeCtive 71.53 lives here)
├── Exp_DM/                              # 5 champions on AICD + Droid (exp07/09/11/13/18). dm_tracker.md.
├── Exp_FewShot/                         # NEW (2026-05-08) — K-shot pivot, T4-native
│   ├── tracker.md                       # K-sweep leaderboard + decision-gate criteria
│   ├── _common_fs.py / _fewshot_sampler.py / _data_codet_fs.py
│   ├── _model_fs.py / _trainer_fs.py
│   ├── exp_fs_00_baseline.py            # CE only (floor)
│   └── exp_fs_01_ntkalign.py            # CE + NTK target-kernel alignment
├── Paper/                               # NEW (2026-05-08) — EMNLP 2026 submission
│   ├── outline.md                       # 1-page section outline + day-by-day plan
│   ├── README.md / formatting.md
│   └── latex/
│       ├── main.tex                     # OUR draft (5 pages, storytelling, committed 2268691)
│       ├── custom.bib                   # 25+ real citations
│       ├── acl.sty / acl_natbib.bst     # ACL/EMNLP template (read-only)
│       └── acl_latex.tex                # template reference (read-only)
├── legacy/                              # Archived — read-only, see legacy/README.md
│   ├── Exp_ZeroShot/                    # 31 ZS detectors, dropped (reproduction gap −32 pt vs paper FDG)
│   ├── Exp_TK/                          # consolidated mirror, redundant
│   ├── Exp_DM_weak/                     # 21 underperformers from Exp_DM
│   ├── experiment/                      # original exp00–04 baselines
│   └── performance_tracker.md           # superseded by per-suite trackers
├── docs/
│   ├── CLAUDE.md                        # THIS FILE
│   └── references/                      # Read-only: paper_AICD.md, paper_Droid.md, paper_CodeDet_M4.md
├── Slide/                               # Vietnamese progress-report deck
└── Formatting_Instructions_For_NeurIPS_2026/   # OLD template, kept for reference
```

**Single sources of truth:**
- EMNLP 2026 paper draft → [Paper/latex/main.tex](../Paper/latex/main.tex) (5 pages, story locked 2026-05-08)
- Section outline + day-by-day plan → [Paper/outline.md](../Paper/outline.md)
- 20%-data climb leaderboard → [Exp_Climb/tracker.md](../Exp_Climb/tracker.md)
- CoDET-M4 full-data leaderboard → [Exp_CodeDet/tracker.md](../Exp_CodeDet/tracker.md)
- Few-shot K-sweep + decision gate → [Exp_FewShot/tracker.md](../Exp_FewShot/tracker.md)
- AICD + Droid method history → [Exp_DM/dm_tracker.md](../Exp_DM/dm_tracker.md)
- Archived suites + rationale → [legacy/README.md](../legacy/README.md)

## 8. Current standing (as of 2026-05-09 night, 31 Kaggle runs banked)

**EMNLP-LOCKED NUMBERS (CoDET-M4 6-class Author IID Macro-F1):**

The few-shot phase-transition curve is now FILLED:

```
K=32 (192 samples)   = 0.183  (random ≈ 0.167)
K=128 (768 samples)  = 0.293
1% (~5K samples)     = 0.570
5% (~25K samples)    = 0.671  ← 🏆 BEATS PAPER UniXcoder full (0.6633)
20% (~100K samples)  = 0.710  (Exp_13 NTKAlign)
Full data            = 0.715  (Exp_27 DeTeCtive)
```

Top 5 at fraction=0.05:
- 🥇 FS-Hier-NTK 0.6709 (+0.0076 vs paper)
- 🥈 FS-HierTree 0.6682 (+0.0049)
- 🥉 FS-NTKAlign 0.6652 (+0.0019)
- 4 FS-Focal 0.6616 (−0.0017)
- 5 FS-Baseline-UniXcoder reimpl 0.6512 (−0.0121)

Regime split:
- **At fraction ≥ 5% (loss-design regime):** Hier-NTK / HierTree / NTKAlign / Focal all beat paper UniXcoder full data.
- **At fraction = 1% (~5K, encoder-pretrain regime):** FS-Baseline-UniXcoder reimpl (0.5744) wins narrowly; ours methods 0.55-0.57.
- **At K=128 (~768, class-imbalance regime):** Focal (0.3749) wins; Hier-NTK (0.2775) drops below baseline UniXcoder (0.3727) because the kernel target is too sparse with bs=16.
- **At K=32 (~192, information-floor regime):** all methods near random.

**Cross-bench (Droid T3 from earlier Exp_Climb runs, 20% data):**
- Exp_04 Poincare 0.8976 W-F1 (+0.98 vs DroidDetectCLS-Large 0.8878)
- Few-shot dispatch now lives in self-contained `Exp_FewShot/testing_chis/exp_*.py` and `baseline_*.py` files.

**Open problems (paper §5 Analysis + §6 Limitations):**
- **OOD-Source-gh:** best Exp_11 PH 35.56, far below 0.40. Frame as characterisation, not solution.
- **Sibling generator confusion:** Qwen↔Nxcode pair = >50% of errors in DeTeCtive. Hier-NTK helps; n01 SRD + n06 PCS are dedicated novel attacks.
- **AICD T2:** run as the authorship-like model-family attribution stress test.
- **AICD T1:** universal val->test collapse. Excluded from headline.

**Few-shot suite status:**
- 31 Kaggle T4 runs banked, summary.json aggregated.
- 4 ours methods exceed paper UniXcoder full-data 0.6633 at 5% data.
- 8 novel methods (n01-n10) active; 0 currently failed Novelty Filter.
- Active next target is `aicd_t2`; `droid_t3` is supporting evidence; `droid_t4`/`aicd_t3` are appendix candidates.

**⚡ Operational improvements (2026-05-10):**
1. **Remote refactor accepted** -- portfolio runners were split into self-contained copy-runnable files under `Exp_FewShot/testing_chis/`.
2. **RTX6000 Ada (48GB) auto-detection** — bs=64, bf16, seq=512 → ~3-4× faster than T4. Detection: `mem_gb >= 40`.
3. **Offline loading (no-internet)** — All models/datasets from Kaggle input paths. No HuggingFace download → eliminates conflict + bias from shared cache.

## 9. EMNLP rubric self-check

EMNLP Main needs (vs the original NeurIPS Oral plan):
1. ✅ **Strong empirical results** — +5.20 Author / +4.70 lean / +0.98 Droid all locked.
2. ✅ **Multi-benchmark coverage** — CoDET-M4 + DroidCollection (2 venues' SOTA each).
3. ✅ **Honest limitations** — paper §6 has 4 limitations explicitly named.
4. 🟡 **Method novelty** — must be framed as Hierarchical Target-Kernel Alignment, not "NTK loss + hierarchy." The paper needs the target-kernel object, its property, and its falsifier.
5. 🟡 **OOD story** — characterisation only; reviewer may ask for solution.

What would still upgrade this to NeurIPS-tier (post-EMNLP, not for current paper):
1. Break OOD-SRC-gh > 0.40 (exp_20–26 pending pipeline).
2. Causal identification claim via linear probe Pr(Ŝ|φ).
3. Data-efficiency theorem under (S,Y)-DAG.

The NeurIPS axes (§4 above) remain the **scientific spine**, but for EMNLP we promote them from "framework" to "diagnosis material" in §5 Analysis.

## 10. How to work in this repo

### Run an experiment
```bash
python Exp_Climb/exp_NN_<method>.py           # dual-bench data-efficient climb (preferred)
python Exp_DM/expNN_<method>.py               # AICD/Droid single-bench
python Exp_CodeDet/run_codet_m4_expNN_*.py    # CoDET-M4 single-bench
```
Each file is self-contained. Kaggle-first: scripts assume H100 BF16, auto-install `tree-sitter`, `tree-sitter-languages`. Outputs go to local `logs/`, `results/`, `codet_m4_checkpoints/` (gitignored).

### Add a new method
1. First write the 6-line proposal block from §2.2. If NAME / EQUATION / PROPERTY / FALSIFIER cannot be filled, do not code it.
2. Keep one idea per file. The implementation must isolate one mathematical object or one loss witness.
3. Copy the closest current `Exp_FewShot/testing_chis/exp_*.py` file; change only the method-specific loss/object and the result filename.
4. Register ablation toggles only if they falsify the stated property. Do not add λ sweeps as "novelty."
5. After the run, append: (i) the leaderboard row, (ii) the falsifier/proxy metric, (iii) a *Theory / Mechanism / Evidence* block in tracker.md. Never rewrite historical rows.

### Conventions
- Experiment IDs never reused.
- Primary metric is benchmark-specific: Macro-F1 for AICD/CoDET-M4, Weighted-F1 for Droid. Do not substitute.
- Always report val + test + Δ together.
- Ablations live in `Exp_DM/` and `Exp_CodeDet/`; **climb files also carry built-in ablations** starting at Exp_14 (this is new and should stay).

## 11. Paper artefacts

- **EMNLP 2026 draft**: [Paper/latex/main.tex](../Paper/latex/main.tex) (5 pages, storytelling, committed `2268691` 2026-05-08). Compiles via pdflatex + bibtex.
- **Section outline + 18-day plan**: [Paper/outline.md](../Paper/outline.md).
- **Proposal slides + script**: `Slide/proposal.tex`, `Slide/proposal.pdf`, `Slide/script_thuyet_trinh.md` (Vietnamese narration).
- **Benchmark papers**: `docs/references/paper_AICD.md`, `paper_Droid.md`, `paper_CodeDet_M4.md` (read-only references).

## 12. Time-wasters to avoid

- **DO NOT pivot scope.** Headline numbers locked 2026-05-08. Any new experiment that would change them must wait for camera-ready.
- **DO NOT run new training experiments without checking the decision gate** (Exp_FewShot Phase B Day 5). Kaggle quota is the bottleneck.
- `legacy/` is archived. Do not import from it, do not run scripts from it. Reference only.
- `.claude/`, `.cursor/`, `codet_m4_checkpoints/`, `*.pt`, `logs/`, `results/`, LaTeX `*.aux/.log/.pdf` are gitignored. Don't commit.
- Large negative Δ on AICD T1 test is expected (val→test collapse). Mention only in §6.4 Limitations, not headline.
- **OOD-Generator LOO macro-F1 is ceiling-bound ~0.5** (test = only the held-out class). Report weighted-F1 and per-class recall for that column; macro is a regression detector only.
- The original NeurIPS Oral framing (axes §4, OOD-gh > 0.40) is **archived as scientific spine**. For the EMNLP paper, keep that material in §5 Analysis, not §1 Introduction.

## 13. Cold-start checklist (2026-05-09 EMNLP-mode, post-breakthrough)

1. Read §0 (current submission target) and §8 (locked numbers) of this file.
2. **Headline is locked: Hier-NTK + 5% data = 0.6709, beats paper UniXcoder 0.6633.** Don't re-litigate the choice; build on it.
3. Open [Exp_FewShot/tracker.md](../Exp_FewShot/tracker.md) — full 31-run leaderboard + 11 insights + evolution timeline. This is the most current source of truth.
4. Open [Paper/latex/main.tex](../Paper/latex/main.tex) — current draft built on the OLD NTKAlign-only headline. Needs update to feature Hier-NTK as main method.
5. Open [Exp_FewShot/novel/README.md](../Exp_FewShot/novel/README.md) — 5-gate Novelty Filter + 8-method catalog + recommended Droid runs.
6. If the user asks about results: cross-check the tracker first. Be careful — the OLD `Exp_Climb/` numbers (Exp_13 71.03, Exp_27 71.53) live at 20% data with a heavier backbone (HierTree+spectral); the NEW `Exp_FewShot/` numbers (Hier-NTK 0.6709 at 5%) live on a lean ModernBERT-only stack. Different experiments — quote both with the data-budget annotation.
7. If the user asks about Droid: most Exp_FewShot Droid runs are still queued. Existing Droid number is Exp_04 Poincare 0.8976 W-F1 at lean 20% data from `Exp_Climb/`.
8. If the user asks about something in `legacy/`: confirm before re-activating — those files were archived for a reason.

## 14. Decision rules (what NOT to do unprompted)

- **Never** add a new method to `Exp_Climb/` / `Exp_CodeDet/` / `Exp_DM/` without user approval. Those suites are frozen for paper.
- **Never** re-open Zero-Shot work — it's archived. If user asks "can we revive ZS?", point at the −32 pt FDG reproduction gap and require a justification.
- **Never** mix benchmarks in a single training run (CLAUDE.md root rule).
- **Never** reuse an `expNN` ID across or within suites.
- **Never** rewrite historical tracker rows. Append-only.
- **Always** report val-test gap alongside test metrics.
- **Always** keep the existing 71.03 / 71.53 / 89.76 numbers as the safety net. New runs in `Exp_FewShot/` go to new cells, never overwrite.
- For paper edits: **prefer Edit on `Paper/latex/main.tex` over Write** — keeps history of section evolution clean in `git log -p`.

## 15. Exp_FewShot rules (active suite, 2026-05-10)

The few-shot suite is split into two physically separate folders. Different rules apply to each.

### `Exp_FewShot/testing_chis/` -- active self-contained experiment track

- Hyperparameter points, encoder swaps, paper-baseline reimplementations,
  combinations of existing losses.
- **No novelty gate for baselines**, but all paper-claim methods still need the theory/falsifier block.
- Current naming is explicit and copy-runnable:
  - `exp_01_hier_tree.py`
  - `exp_02_ntk_align.py`
  - `exp_03_hier_ntk.py`
  - `exp_04_ce_baseline.py`
  - `baseline_01_codet5.py` ... `baseline_04_style.py`
- This track is the safe-floor for EMNLP Main.

### `Exp_FewShot/novel/` — theory-driven oral track

- Each file proposes ONE new mathematical object with a falsifiable claim.
- Filename pattern: `exp_nNN_<concept>.py` (sequential).
- **Must pass 5-gate Novelty Filter** before landing here:
  1. Theory-driven (**arxiv paper OR self-derived theorem** — see §2.4)
  2. ≤ 2 novel components
  3. Falsifiable claim
  4. Differentiated from prior work
  5. Compute-feasible (≤ 1 T4-hour per data point)
- **Mandatory header docstring** with: ARXIV_ID (or `THEORY_ORIGINAL`) / NAME / ONE-LINE CLAIM / EQUATION / THEORY HOOK / WHY NOT BEFORE / FALSIFIER.
- Failed runs go to `legacy/novel_failed_NN_*.py`. Never delete; document the negative result in tracker.

**Two paths to novelty:**

**Path A — Arxiv-grounded** (safer, easier to defend):
- Cite existing theory paper with arxiv ID
- Apply to code attribution domain

**Path B — Self-derived theorem** (stronger, EMNLP Oral-tier):
- Derive NEW theorem from first principles
- Must have: formal statement, key insight, derivation sketch, falsifiable predictions
- See §2.4 for templates and examples

**Current novel entries (arxiv-grounded, 2026-05-10):**

| File | Method | ArXiv ID | Theory Hook | Key Contribution |
|:--|:--|:--|:--|:--|
| `exp_n05_focal.py` | **Focal-CE** | Lin 2017 | Hard example mining | Down-weights easy examples; γ=2 |
| `exp_n06_attn_pool.py` | **HAP** | Lee 2017 | Hierarchical attention | \(z = \sum_i \alpha_i h_i\) |
| `exp_n07_mixup_align.py` | **MGA** | Arjovsky 2019 | IRM invariance | Gradient penalty for do(S) |
| `exp_n08_ortho_clf.py` | **Ortho-CLF** | Papyan 2020 | Neural collapse | Gram-Schmidt classifier |
| `exp_n09_etf_simplex.py` | **ETF-Simplex** | NC theory | Optimal ETF geometry | \(W_{etf} = \sqrt{K/(K-1)}(e_i - 1/K)\) |
| `exp_n10_vib.py` | **VIB** | Alemi 2017 | Info bottleneck | KL(q‖p) regularisation |
| `exp_n11_mixup_ce.py` | **Mixup-CE** | Zhang 2018 | Boundary smoothing | Soft label interpolation |
| `exp_n12_label_smooth.py` | **LabelSmooth** | Pereyra 2017 | Calibration | ECE improvement |

**Novel theorems (self-derived, EMNLP Oral-tier):**

| File | Theorem | Type | Key Prediction |
|:--|:--|:--|:--|
| *(to be created)* | **Kernel Alignment Bound** | §2.4 Thm 1 | Genealogical correction modulates excess risk |
| *(to be created)* | **Phase Transition** | §2.4 Thm 2 | Transition at \(n^* = \Theta(K \cdot h / \lambda_{\min}^2)\) |
| *(to be created)* | **Sibling Confusion** | §2.4 Thm 3 | Confusion ∝ 1/tree_distance |

**Theory Sources (key papers):**

| Theory | Paper | ArXiv | Venue | Relevance |
|:--|:--|:--|:--|:--|
| **Kernel Methods** | Truncated Kernel Methods (TKM) | `2410.06171` | NeurIPS 2024 | Kernel complexity vs alignment |
| **Neural Collapse** | Deep NC with Weight Decay | `2410.04887` | ICLR 2025 | ETF geometry, class collapse |
| **Target Alignment** | Unified Representation Alignment | `2502.14047` | ICLR 2025 | Gram matrix → kernel alignment |
| **Information Bottleneck** | Causal IB (CIB) | `2410.00535` | arXiv 2024 | KL regularisation for invariance |
| **Contrastive Learning** | SupCon | IEEE 2020 | — | Contrastive alignment |
| **Geometric DL** | CUSP | `2502.00401` | ICLR 2025 | Hyperbolic/curvature methods |
| **Causal Inference** | IRM | Arjovsky 2019 | — | Invariant risk minimisation |

### Benchmark dispatch (2026-05-10)

Every novel + testing file accepts `FS_BENCHMARK` env var:

| `FS_BENCHMARK` | Bench / Task | Classes | Primary metric | Paper baseline |
|:--|:--|:-:|:--|:-:|
| `codet_m4` (default) | CoDET-M4 6-class Author IID | 6 | Macro-F1 | UniXcoder 66.33 |
| `aicd_t2` | AICD-Bench Task 2 model-family attribution | 12 | Macro-F1 | AICD T2 paper baseline |
| `droid_t3` | DroidCollection T3 (3-class) | 3 | Weighted-F1 | DroidDetectCLS-Large 88.78 |
| `droid_t4` | DroidCollection T4 (4-class incl. adversarial) | 4 | Weighted-F1 | DroidDetectCLS-Large 94.30 |
| `aicd_t3` | AICD-Bench Task 3 fine-grained detection | 4 | Macro-F1 | Appendix only |

Cross-bench portability of methods:

| Method | CoDET-M4 | Droid T3 | Droid T4 |
|:--|:-:|:-:|:-:|
| n01 SRD / n06 PCS | ✅ ideal | ⚠️ no sibling pair | ⚠️ same |
| n02 FSM | ✅ | ✅ | ✅ |
| n04 EFS | ✅ K=6 | ✅ K=3 | ✅ K=4 |
| n05 MIF | ✅ | ⚠️ less interesting at K=3 | ⚠️ |
| n07 CMP | ✅ | ✅ | ✅✅ T4 adversarial sweet spot |
| n08 SEA | ✅ K=6 | ⚠️ K=3 trivial | ⚠️ |
| n10 VIB | ✅ | ✅ | ✅ |
| Hier / NTK / HierTree+NTK | ✅ ideal | ⚠️ no sibling pair | ⚠️ same |

Recommended run order: `codet_m4` first, then `aicd_t2`, then `droid_t3`. Run `droid_t4` only when robustness appendix is needed. Run `aicd_t3` only if there is spare compute.

### When to lock the paper headline

**Already locked (2026-05-09 night):** Hier-NTK at 5% data = 0.6709 is the main result. Earlier policy "wait until matrix half-filled" satisfied (31 runs across 9 methods × 3 budgets).

### Anti-patterns (do NOT add to `novel/`)

- Loss-weight tuning ("we tried λ=0.4, 0.6, 0.8")
- Feature stacking ("AST + token-stat + spectral all at once")
- Augmentation cocktails
- Hyperparameter sweeps presented as method
- "Apply existing method to new domain" alone
- **ANY method without arxiv citation** — must be grounded in theory

If the idea is "stronger version of an existing method", ship under `testing/`.

**What IS welcome:**
- Novel combinations of existing arxiv-grounded methods
- Theoretical extensions with mathematical justification
- Ablation of components from arxiv methods
- Transfer from other domains if arxiv-grounded (e.g., VIB from NLP→code, ETF from vision→code)

### Five open problems where novelty is welcome

(Theory-grounded, not just hierarchical. Any approach from §2.2 is welcome.)

1. **K-shot collapse on codellama↔nxcode siblings** — Try: Focal loss (n05), Attention pooling (n06), Fisher/SRD
2. **Source-confounding (CF/LC → GitHub OOD)** — Try: MGA gradient alignment (n07), VIB (n10), IRM
3. **Phase transition at ~5K samples** — Try: Mixup (n11), Label smoothing (n12), Information theory bounds
4. **Neural collapse explicit parameterisation** — Try: ETF-Simplex (n09), Ortho-CLF (n08)
5. **Information-theoretic floor I(Y; X) for CoDET-M4** — Try: VIB (n10), CIB, MINE

## 16. Original portfolio rules (kept for legacy reference)

The few-shot portfolio is **exploratory until the matrix is half-filled**. Until then, treat it like a search, not a commitment.

### File contract (every `exp_fs_NN_*.py`)
1. **Self-contained.** Each file auto-clones the repo from GitHub if `_common_fs.py` is absent locally. A Kaggle cell can `!python exp_fs_NN.py` with no setup.
2. **Standalone per CLAUDE.md.** Imports come from `Exp_FewShot/_*.py`, never from `Exp_Climb/`, `Exp_DM/`, `Exp_CodeDet/`.
3. **One method, one file.** Hyperparameters surface as env vars (`FS_K_SHOT`, `FS_TRAIN_FRACTION`, `FS_LAMBDA_NTK`, `FS_TEMP`, `FS_LR_HEADS`, ...).
4. **Two regimes via env vars:**
   - `FS_K_SHOT=N` (N≥1) → K-shot per-class, 1 epoch, no LR schedule.
   - `FS_TRAIN_FRACTION=f` (with `FS_K_SHOT=0`) → fraction of train, 3 epochs, cosine LR + 10% warmup.
5. **Always emit `BEGIN_FS_TABLE … END_FS_TABLE`** — that's how the tracker matrix gets updated.

### Sweep contract (`run_fs_portfolio.py`)
- Spawn each `(method, regime, value)` as a subprocess; pipe stdout to `logs/<script>_<label>.log`.
- `FS_METHODS`, `FS_KS`, `FS_FRACTIONS` env vars override the sweep grid.
- Default sweep is conservative: 4 methods × {K=32, K=128, 1%, 5%} = 16 runs.

### Hardware contract
- T4 first (Kaggle free tier): bs=16, fp16, seq=384, 1 epoch (K-shot) or 3 epochs (%-fraction).
- Auto-upgrade on H100/A100/V100 if available (bs/seq lifted).
- CPU mode = smoke test only.

### When to lock the paper headline
- Lock when **at least 50%** of the portfolio matrix is filled AND one cell beats `Exp_13 NTKAlign 0.7103` OR clearly beats UniXcoder full-data 0.6633 with substantially less data.
- If after the full sweep no cell improves on the safety net, paper draft v1 ships unchanged with the data-efficient curve as a supplementary figure.
