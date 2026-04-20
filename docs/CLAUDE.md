# CLAUDE.md — Theory-Driven Repo Briefing (NeurIPS 2026 Oral target)

> **Read this file first.** One-stop theoretical + operational context. Oral-quality papers lead with a hypothesis, not a leaderboard. This file states the hypothesis, the causal model, the evidence axes, and where each artefact in the repo sits on those axes.
> If anything here disagrees with the code, trust the code and fix this file.

---

## 1. The central research question

> **Why do state-of-the-art AI-code detectors generalise to unseen generators and languages but collapse on unseen *sources* (CodeForces → GitHub)?**

Prior work (CoDET-M4 ACL'25; DroidCollection EMNLP'25; AICD-Bench) reports strong IID Macro-F1 (≥98% binary, 66–71% 6-class author) but shows order-of-magnitude drops on held-out sources — on CoDET-M4 held-out-GitHub, macro-Author-F1 collapses to ≈0.28 across 14 independent methods we measured. Our thesis is that this is **not** a capacity or a representation problem; it is a **confounding** problem.

## 2. Thesis (one sentence)

> **AI-code detectors fail on unseen sources because source style $S$ is a confounder of both the input $X$ and the author label $Y$, and existing methods optimise $P(Y \mid X)$ instead of $P(Y \mid \operatorname{do}(X))$.**

Equivalently: training on CF + LC competitive-programming templates hands the model a shortcut that is *spuriously* predictive of $Y$ on the training support and *uninformative* off it. Our paper argues — and then empirically tests — that progress on AI-code detection requires interventions along one of eight theoretical axes (§4), and that only a subset of those axes move OOD metrics without breaking IID.

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

## 5. Benchmarks (three in play, role-specified)

| Short name | Role in the paper | Primary metric | Why this metric |
|:--|:--|:--|:--|
| **CoDET-M4** (ACL'25) | Main evaluation — has the crucial per-source / per-language LOO splits that expose the $S$ back-door | **Macro-F1** (author 6-class) | Class-balanced; standard ACL metric; paper baselines published |
| **DroidCollection** (EMNLP'25) | Secondary — cross-domain stability + adversarial robustness | **Weighted-F1** (T3 3-class) | Severe class imbalance; matches paper protocol |
| **AICD-Bench** | **Open challenge / negative result** — val 0.99 → test 0.25 universal collapse on T1 across 23 Exp_DM methods | Macro-F1 | We deliberately exclude AICD from climb claims; cite as evidence that some OOD problems are *dataset* properties |

**Protocol (non-negotiable):**
- Train separate model per benchmark. Never mix.
- Report the full metric pack: Primary, Macro-F1, Weighted-F1, Macro-R, Weighted-R, Accuracy, per-class.
- Report **val and test** side-by-side. The val–test gap is itself a diagnostic (insight #4).
- Hardware: **H100 80GB, BF16, batch 64×1, seq 512**.
- Each `exp_NN_*.py` is **standalone** and Kaggle-runnable.
- Data-efficiency framing: **train on 20%**, evaluate on 100% test. Any +Δ vs a full-data paper baseline is a paper claim.

## 6. Ablation plan (for the Oral)

An Oral needs ablations that falsify hypotheses, not ablations that tune λ's. For each axis in §4:

1. **Main effect.** Method with all components vs UniXcoder / DroidDetectCLS-Large paper baseline on IID + OOD.
2. **Component drop.** One λ → 0 per row. Δ tells us whether that component carries the axis.
3. **Cross-axis control.** Swap the axis-defining component for a non-causal / non-theoretical equivalent (e.g. replace `lambda_cf` with a random-pair swap). Must eliminate the gain.
4. **Null / shortcut probe.** Train a 1-layer linear probe on the final embedding to predict $S$. The theory says $\Pr(\hat S \mid \phi(X))$ should drop under axis-C methods. If the probe still succeeds, axis C did not identify $\operatorname{do}(S)$.

**Current state of §6 for Exp_18 (example, see tracker):** (1) done (70.19 IID), (2) done (component Δ ≤ 0.3 pt each), (3) **not run**, (4) **not run**. Two missing pieces before a causal claim.

## 7. Repo layout (post-cleanup, theory-axes aligned)

```
ai_code_detection/
├── README.md
├── experiment/                          # Legacy baselines (exp00–04). Only exp00_codeorigin.py is load-bearing.
│   └── exp00_codeorigin.py
├── Exp_Climb/                           # MAIN suite — axis-aligned, dual-bench data-efficient climb
│   ├── tracker.md                       # Primary evidence log. Axis table + leaderboard + ablation matrix + per-exp theory blocks.
│   ├── _common.py / _features.py / _model.py / _trainer.py
│   ├── _data_codet.py / _data_droid.py
│   ├── _climb_runner.py / _paper_table.py / _ablation.py
│   └── exp_NN_<method>.py               # ONE method per file, thin wrapper, each targets a named axis (see §4)
├── Exp_DM/                              # 30 methods on AICD + Droid (historical). dm_tracker.md.
├── Exp_CodeDet/                         # CoDET-M4 adaptation of DM methods. tracker.md.
├── docs/
│   ├── CLAUDE.md                        # THIS FILE
│   ├── performance_tracker.md           # Legacy. Superseded by Exp_Climb/tracker.md + Exp_DM/dm_tracker.md.
│   └── references/                      # Read-only: paper_AICD.md, paper_Droid.md, paper_CodeDet_M4.md, dataset_links.md
├── Slide/                               # Proposal deck (Vietnamese presentation script + .tex + .pdf)
└── Formatting_Instructions_For_NeurIPS_2026/   # DO NOT MODIFY .sty
```

**Single sources of truth:**
- Axis spine + ablation matrix → [Exp_Climb/tracker.md](../Exp_Climb/tracker.md) §Theory axes, §Ablation matrix
- AICD + Droid method history → [Exp_DM/dm_tracker.md](../Exp_DM/dm_tracker.md)
- CoDET-M4 baseline leaderboard → [Exp_CodeDet/tracker.md](../Exp_CodeDet/tracker.md)
- Backbone reference implementation → [experiment/exp00_codeorigin.py](../experiment/exp00_codeorigin.py)

## 8. Current standing (as of 2026-04-19)

- **CoDET-M4 Author IID (primary):** climb leader **Exp_13 NTKAlign 71.03** (+4.70 vs UniXcoder 66.33). Runner-up Exp_06 Flow 70.90. **Exp_17 no_teacher_distill (ablation variant) reaches 70.89** but is partial-scope (single-task only, no OOD/Droid), so climb ranking remains anchored to complete lean/full runs. **Exp_18 CausalIntervention 70.19** is also ablation-only scope (OOD-GH not yet evaluated → cannot yet claim axis-C SOTA).
- **CoDET-M4 OOD-Source (held-out GH), hardest subgroup:** **Exp_11 PersistentHomology 35.56** (climb record). No method has crossed the 0.40 threshold we set as the NeurIPS headline for axis C / G / H.
- **Droid T3 (3-class W-F1):** climb leader **Exp_04 PoincareGenealogy 89.76** (+0.98 vs DroidDetectCLS-Large 88.78). Droid is stable across methods (insight #11); it's a sanity check, not a discriminator.
- **AICD T1:** open challenge. val 0.99 → test 0.25–0.31 across 23 methods. We report it as a negative result.

## 9. What would make this an Oral vs. a Poster

An oral needs one of:
1. **Break OOD-SRC-gh > 0.40** with a method whose ablation isolates *why* (currently at 0.3556; +0.05 absolute on the hardest subgroup). Axes C, G, H are the live candidates.
2. **Causal identification claim.** Show empirically that $P(\hat Y \mid X, s)$ is invariant across $s$ under our method via the §6 shortcut probe — not just that Macro-F1 improves.
3. **Data-efficiency theorem.** Formalise why 20%-train is sufficient under the $(S, Y)$-DAG; connect to the causal-sufficiency literature.

Poster-level content (incremental +F1 without a mechanism) is not enough. Every new `exp_NN_*.py` should state which of (1/2/3) it contributes to — tracker's per-method block now requires a *Theory / Mechanism / Evidence* triple.

## 10. How to work in this repo

### Run an experiment
```bash
python Exp_Climb/exp_NN_<method>.py           # dual-bench data-efficient climb (preferred)
python Exp_DM/expNN_<method>.py               # AICD/Droid single-bench
python Exp_CodeDet/run_codet_m4_expNN_*.py    # CoDET-M4 single-bench
```
Each file is self-contained. Kaggle-first: scripts assume H100 BF16, auto-install `tree-sitter`, `tree-sitter-languages`. Outputs go to local `logs/`, `results/`, `codet_m4_checkpoints/` (gitignored).

### Add a new method
1. Pick **one** axis from §4. State the hypothesis in the file's docstring.
2. Copy the closest `exp_NN_*.py`; change only the model class + loss + (optionally) the λ-registry.
3. Register ablation toggles so the same file emits `BEGIN_ABLATION_TABLE`.
4. After the run, append: (i) the leaderboard row, (ii) the ablation matrix row, (iii) a *Theory / Mechanism / Evidence* block in tracker.md. Never rewrite historical rows.

### Conventions
- Experiment IDs never reused.
- Primary metric is benchmark-specific: Macro-F1 for AICD/CoDET-M4, Weighted-F1 for Droid. Do not substitute.
- Always report val + test + Δ together.
- Ablations live in `Exp_DM/` and `Exp_CodeDet/`; **climb files also carry built-in ablations** starting at Exp_14 (this is new and should stay).

## 11. Paper artefacts

- **Draft**: will live under `Formatting_Instructions_For_NeurIPS_2026/` once writing starts. `references/paper_*.md` are *other people's* papers we cite.
- **Proposal slides + script**: `Slide/proposal.tex`, `Slide/proposal.pdf`, `Slide/script_thuyet_trinh.md`.
- **Benchmark papers**: `docs/references/paper_AICD.md`, `paper_Droid.md`, `paper_CodeDet_M4.md`.

## 12. Time-wasters to avoid

- `docs/performance_tracker.md` is legacy; trust `Exp_Climb/tracker.md` → `Exp_DM/dm_tracker.md` in that order.
- `experiment/` only contains legacy baselines. The active suite is `Exp_Climb/` (climb) and `Exp_DM/` / `Exp_CodeDet/` (historical ablations).
- Some `Exp_DM/` files (exp28, 29, 30) are tiny wrappers over exp27. Not a bug.
- `.claude/`, `.cursor/`, `codet_m4_checkpoints/`, `*.pt`, `logs/`, `results/` are gitignored. Don't commit.
- Large negative Δ on AICD T1 test is expected, not a bug — it's §1's motivation.
- **OOD-Generator LOO macro-F1 is ceiling-bound ~0.5** (test = only the held-out class). Report weighted-F1 and per-class recall for that column; macro is a regression detector only.

## 13. Cold-start checklist

1. Read this file (§1–§4 minimum).
2. Skim `Exp_Climb/tracker.md` — axes table + leaderboard top 5 + ablation matrix.
3. Check `Exp_CodeDet/tracker.md` for CoDET-M4 non-climb runs.
4. Before proposing a new method: name the axis from §4 and the evidence gap (§8 vs §9).
5. Before running anything: confirm benchmark = CoDET-M4 ∨ Droid ∨ AICD. The answer changes the primary metric, the script, and the tracker file.
