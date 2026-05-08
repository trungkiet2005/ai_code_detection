# CLAUDE.md — Repo Briefing (EMNLP 2026 Main target)

> **Read this file first.** One-stop theoretical + operational context.
> If anything here disagrees with the code, trust the code and fix this file.

---

## 0. Current submission target — LOCKED 2026-05-08

> **Venue:** EMNLP 2026 Main (long paper, 8 pages).
> **Deadline:** ~2026-05-26 (18 days from 2026-05-08).
> **Status (as of 2026-05-08):** Day 2 of 18. Paper draft v1 = 5 pages, committed `2268691`.
> **Working title:** *Who Wrote This Code? Data-Efficient AI-Code Authorship Detection via Hierarchical Family-Aware Learning*

**3 contributions:**
- **C1** — HierTree backbone + NTK target-kernel alignment → **71.03 Author Macro-F1 with 20% data** (+4.70 vs UniXcoder full-data 66.33). Full-data ceiling 71.53 (Exp27 DeTeCtive, +5.20).
- **C2** — Same backbone, retrained on Droid → **89.76 T3 Weighted-F1** (+0.98 vs DroidDetectCLS-Large 88.78).
- **C3** — Source-confounding failure mode characterised: 0.71 IID → 0.36 held-out GitHub across 14 methods.

**Locked headline numbers (DO NOT touch unless rerunning logs invalidates them):**
| Bench / Task | Method | Score | Δ |
|:--|:--|:-:|:-:|
| CoDET-M4 Author IID (full) | Exp27 DeTeCtive | **71.53** | +5.20 vs UniXcoder |
| CoDET-M4 Author IID (lean 20%) | Exp_13 NTKAlign | **71.03** | +4.70 vs UniXcoder |
| Droid T3 (lean 20%) | Exp_04 Poincare | **89.76** | +0.98 vs DroidDetectCLS-Large |
| OOD-SRC-gh (best) | Exp_11 PH | 35.56 | best in suite, but **does not break 0.40 oral threshold** |

**Active pivot — Few-Shot extension (Exp_FewShot/, Phase A complete):**
- Hypothesis: K=64 or K=128 examples per class can match UniXcoder full-data (66.33).
- Phase B Day 4 = Kaggle T4 K-sweep on `exp_fs_01_ntkalign.py`. Phase B Day 5 = decision gate.
- **Decision rule:** K=64 ≥ 65 → full pivot. K=128 < 60 → fall back to 20% story (paper draft v1 still works).
- See `Exp_FewShot/tracker.md` for decision criteria.

**Hardware switch:** Kaggle **T4 16GB** is now first-class (Exp_FewShot is T4-native bs=16 fp16 seq=384). H100 still preferred for `Exp_Climb/` runs but no longer required.

**Out-of-scope for this paper (do NOT add to headline):**
- Zero-shot detection (Exp_ZeroShot, **moved to legacy/** — reproduction gap −32 pt vs paper Fast-DetectGPT).
- AICD-Bench (universal val→test collapse, **discussion-only mention** in §6 Limitations).
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

## 1. The central research question (paper §5 Analysis)

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

## 8. Current standing (as of 2026-05-08)

**EMNLP-locked numbers (paper §4):**
- **CoDET-M4 Author IID full data:** **Exp27 DeTeCtive 71.53** (+5.20 vs UniXcoder 66.33). Lives in `Exp_CodeDet/`.
- **CoDET-M4 Author IID 20% data:** **Exp_13 NTKAlign 71.03** (+4.70 vs UniXcoder). Lives in `Exp_Climb/`.
- **Droid T3 W-F1 (3-class) 20% data:** **Exp_04 Poincare 89.76** (+0.98 vs DroidDetectCLS-Large 88.78).
- **20+ methods cluster ≥70.0** Author IID — interpreted as evidence that the gain is from the HierTree prior, not any single auxiliary loss (paper §4.2 "cluster-break").
- **7+ methods beat DroidDetectCLS-Large** on T3 — cross-bench robustness for paper §4.3.

**Open problems (paper §5 Analysis + §6 Limitations):**
- **OOD-Source-gh:** best **Exp_11 PH 35.56**, far below 0.40. Across 14 methods 0.71→0.36 → frame as *characterisation*, not *solution*.
- **Sibling generator confusion:** Qwen↔Nxcode pair = >50% of all errors in DeTeCtive (1884 + 1962 swaps). Hierarchical prior helps but doesn't close the gap.
- **AICD-Bench:** universal val 0.99→test 0.25 across 23 methods. **Excluded from headline**; brief mention in §6.4.

**Few-shot pivot status (Exp_FewShot/, Phase A done):**
- Infrastructure committed `a0a0798` (T4-native sampler, model, trainer).
- Phase B Day 4 = Kaggle T4 K-sweep on K∈{8,16,32,64,128} for `exp_fs_01_ntkalign.py`.
- Phase B Day 5 = decision gate. If K=128 < 60 Author F1 → fall back to 20% story (paper draft v1 still works).

## 9. EMNLP rubric self-check

EMNLP Main needs (vs the original NeurIPS Oral plan):
1. ✅ **Strong empirical results** — +5.20 Author / +4.70 lean / +0.98 Droid all locked.
2. ✅ **Multi-benchmark coverage** — CoDET-M4 + DroidCollection (2 venues' SOTA each).
3. ✅ **Honest limitations** — paper §6 has 4 limitations explicitly named.
4. 🟡 **Method novelty** — NTK alignment for code authorship is novel; reviewer may push back as incremental. Mitigation: cluster-break observation across 14 methods.
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

## 13. Cold-start checklist (2026-05-08 EMNLP-mode)

1. Read §0 (current submission target) and §8 (locked numbers) of this file.
2. Open [Paper/outline.md](../Paper/outline.md) — see what day we are on and what's next.
3. Open [Paper/latex/main.tex](../Paper/latex/main.tex) — read the abstract + §1 to absorb the story.
4. If the user asks about results: cross-check [Exp_Climb/tracker.md](../Exp_Climb/tracker.md) and [Exp_CodeDet/tracker.md](../Exp_CodeDet/tracker.md) before relying on memory.
5. If the user asks about few-shot: check [Exp_FewShot/tracker.md](../Exp_FewShot/tracker.md). Phase B (Day 4 Kaggle) decides whether few-shot becomes headline.
6. If the user asks about something in `legacy/`: confirm with them before re-activating — most legacy files were archived for a reason.

## 14. Decision rules (what NOT to do unprompted)

- **Never** add a new method to `Exp_Climb/` / `Exp_CodeDet/` / `Exp_DM/` without user approval. Suite is frozen for paper.
- **Never** re-open Zero-Shot work — it's archived. If user asks "can we revive ZS?", point at the −32 pt FDG reproduction gap and require a justification.
- **Never** mix benchmarks in a single training run (CLAUDE.md root rule).
- **Never** reuse an `expNN` ID across or within suites.
- **Never** rewrite historical tracker rows. Append-only.
- **Always** report val-test gap alongside test metrics.
- **Always** keep the locked headline numbers stable. New runs go in new exp IDs.
- For paper edits: **prefer Edit on `Paper/latex/main.tex` over Write** — keeps history of section evolution clean in `git log -p`.
