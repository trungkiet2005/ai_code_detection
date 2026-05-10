# CLAUDE.md — Repo Briefing (EMNLP 2026 Main target)

> **Read this file first.** One-stop theoretical + operational context.
> If anything here disagrees with the code, trust the code and fix this file.

---

## 0. Current submission target — UPDATED 2026-05-09 (after 31 Kaggle runs)

> **Venue:** EMNLP 2026 Main (long paper, 8 pages). **Reach for Oral.**
> **Deadline:** ~2026-05-26.
> **Status (2026-05-09 night):** Day 3 of 18. 31 Kaggle T4 runs banked.
> **Working title:** *Few-Shot AI-Code Authorship Detection via Hierarchical-NTK Family-Aware Learning*

**4 contributions (updated):**
- **C1 (NEW headline)** — **Hier-NTK** = HierTree family prior (Galanti-Poggio 2025) + NTK target-kernel alignment. With **5% of training data (~25K samples)** on a ModernBERT-base encoder, reaches **0.6709 CoDET-M4 6-class Author Macro-F1** — exceeding paper UniXcoder's full-data 0.6633 by **+0.0076 with 1/20 the budget**.
- **C2 (cluster claim)** — Four of our methods (Hier-NTK, HierTree, NTKAlign, Focal) cluster ≥ 0.6616 at 5% data; gain comes from the **hierarchical prior + ModernBERT backbone**, not any single trick.
- **C3 (regime separation)** — Below 5K samples, paper baseline (UniXcoder reimpl) wins; above 5K, Hier-NTK wins. Phase transition at ~5K samples / 1% of train.
- **C4 (cross-bench, in progress)** — Same recipe transfers to DroidCollection T3/T4 (runs queued via `FS_BENCHMARK=droid_t3` env var).

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
- AICD-Bench (universal val→test collapse, mention in §6 Limitations only).
- Zero-Shot suite (in `legacy/`, reproduction gap −32 pt vs paper FDG).
- OOD-Source-gh > 0.40 (NeurIPS Oral threshold; for EMNLP Main framed as "characterisation" not "claim").

**Repo cleanup done 2026-05-08 (commit `89ba63d`):**
- `Exp_ZeroShot/`, `Exp_TK/`, 21 weak `Exp_DM/` files moved to `legacy/`.
- Active suites: `Exp_Climb/` + `Exp_CodeDet/` + `Exp_DM/` (5 champions: exp07/09/11/13/18) + `Exp_FewShot/` (new).

**Hardware switch:** Kaggle **T4 16GB** is now first-class (Exp_FewShot is T4-native bs=16 fp16 seq=384). H100 still preferred for `Exp_Climb/` runs but no longer required.

**⚡ RTX6000 Ada Lovelace (48GB) speedup:** Detected automatically → bs=64, bf16, seq=512. **~3-4× faster** than T4.
**⚡ RTX 96GB speedup:** Detected automatically → bs=128, bf16, seq=512. **~6-8× faster** than T4.

**📦 Offline loading (no-internet):** All models and datasets loaded from Kaggle input paths:
- Encoders: `/kaggle/input/datasets/chiboiz/ai-detection-encoders/models/`
- CoDET-M4: `/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet`
- DroidCollection: `/kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/`
No HuggingFace download → eliminates conflict/bias from model caching on shared Kaggle instances.

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
- Few-shot dispatching to Droid T3/T4 via FS_BENCHMARK env var (queued runs).

**Open problems (paper §5 Analysis + §6 Limitations):**
- **OOD-Source-gh:** best Exp_11 PH 35.56, far below 0.40. Frame as characterisation, not solution.
- **Sibling generator confusion:** Qwen↔Nxcode pair = >50% of errors in DeTeCtive. Hier-NTK helps; n01 SRD + n06 PCS are dedicated novel attacks.
- **AICD-Bench:** universal val→test collapse. Excluded from headline.

**Few-shot suite status:**
- 31 Kaggle T4 runs banked, summary.json aggregated.
- 4 ours methods exceed paper UniXcoder full-data 0.6633 at 5% data.
- 8 novel methods (n01-n10) active; 0 currently failed Novelty Filter.
- Both benches dispatched via FS_BENCHMARK env var; cross-bench Droid runs queued.

**⚡ Operational improvements (2026-05-10):**
1. **Merge code for speed** — `run_hier_ntk_portfolio.py` chains multiple configs sequentially in one cell (3 encoders × 3 fractions = 9 runs).
2. **RTX6000 Ada (48GB) auto-detection** — bs=64, bf16, seq=512 → ~3-4× faster than T4. Detection: `mem_gb >= 40`.
3. **Offline loading (no-internet)** — All models/datasets from Kaggle input paths. No HuggingFace download → eliminates conflict + bias from shared cache.

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

## 15. Exp_FewShot two-track rules (active suite, 2026-05-09 night)

The few-shot suite is split into two physically separate folders. Different rules apply to each.

### `Exp_FewShot/testing/` — setup-variation track

- Hyperparameter points, encoder swaps, paper-baseline reimplementations,
  combinations of existing losses.
- **No novelty gate.** Add cells liberally to fill the data-efficiency matrix.
- Filename free-form (e.g. `exp_fs_baseline_unixcoder.py`).
- 14+ files: 6 method variants, 5 paper-baseline reimpls, 3 combos.
- This track is the safe-floor for EMNLP Main.

### `Exp_FewShot/novel/` — theory-driven oral track

- Each file proposes ONE new mathematical object with a falsifiable claim.
- Filename pattern: `exp_nNN_<concept>.py` (sequential).
- **Must pass 5-gate Novelty Filter** before landing here:
  1. Theory-driven (theorem behind it)
  2. ≤ 2 novel components
  3. Falsifiable claim
  4. Differentiated from prior work
  5. Compute-feasible (≤ 1 T4-hour per data point)
- **Mandatory header docstring** with: NAME / ONE-LINE CLAIM / EQUATION / THEORY HOOK / WHY NOT BEFORE / FALSIFIER.
- Failed runs go to `legacy/novel_failed_NN_*.py`. Never delete; document the negative result in tracker.

**Current 8 novel entries:**

| File | Method | Theory hook | Open problem |
|:--|:--|:--|:--|
| `exp_n01_sibling_residual.py` | SRD — Fisher in residual subspace | Galanti-Poggio 2025 | K-shot codellama↔nxcode collapse |
| `exp_n02_frontdoor_style.py` | FSM — HSIC front-door | Veitch-Wang NeurIPS 2025 | Source-confounding (CF/LC→GH) |
| `exp_n04_etf_simplex.py` | EFS — frozen ETF classifier | Galanti-Poggio + Papyan-Han-Donoho | Explicit ETF parameterisation |
| `exp_n05_mi_floor.py` | MIF — MINE info ceiling | Belghazi MINE ICML 2018 + Fano | I(Y;X) ceiling for 6-class |
| `exp_n06_proximal_sibling.py` | PCS — kernel-ridge proxy | Mastouri-Gretton JMLR 2025 | Sibling pair via causal proxy |
| `exp_n07_conformal_mondrian.py` | CMP — class-conditional conformal | Vovk 2005; Romano-Patterson-Candès 2020 | Per-class FNR guarantee |
| `exp_n08_spectral_eigengap.py` | SEA — Cheeger eigengap | Cheeger 1970; Lee-Oveis-Trevisan 2014 | Phase-transition predictor |
| `exp_n10_vib.py` | VIB — variational info bottleneck | Tishby IB 2000; Alemi VIB ICLR 2017 | Data-efficient regulariser |

(n03, n09 reserved.)

### Two-benchmark dispatch (2026-05-09)

Every novel + testing file accepts `FS_BENCHMARK` env var:

| `FS_BENCHMARK` | Bench / Task | Classes | Primary metric | Paper baseline |
|:--|:--|:-:|:--|:-:|
| `codet_m4` (default) | CoDET-M4 6-class Author IID | 6 | Macro-F1 | UniXcoder 66.33 |
| `droid_t3` | DroidCollection T3 (3-class) | 3 | Weighted-F1 | DroidDetectCLS-Large 88.78 |
| `droid_t4` | DroidCollection T4 (4-class incl. adversarial) | 4 | Weighted-F1 | DroidDetectCLS-Large 94.30 |

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

Recommended Droid runs (subset that transfers cleanly): n02 FSM, n04 EFS, n07 CMP, n10 VIB. Plus the headline winner Hier-NTK to validate cross-bench transfer.

### When to lock the paper headline

**Already locked (2026-05-09 night):** Hier-NTK at 5% data = 0.6709 is the main result. Earlier policy "wait until matrix half-filled" satisfied (31 runs across 9 methods × 3 budgets).

### Anti-patterns (do NOT add to `novel/`)

- Loss-weight tuning ("we tried λ=0.4, 0.6, 0.8")
- Feature stacking ("AST + token-stat + spectral all at once")
- Augmentation cocktails
- Hyperparameter sweeps presented as method
- "Apply existing method to new domain" alone

If the idea is "stronger version of an existing method", ship under `testing/`.

### Five open problems where novelty is welcome

(See [Exp_FewShot/novel/README.md](../Exp_FewShot/novel/README.md) for full text.)
1. K-shot collapse on codellama↔nxcode siblings (n01 attacks this).
2. Source-confounding (CF/LC → GitHub OOD), the original NeurIPS thesis.
3. Phase transition at ~5K samples — sample-complexity floor theorem.
4. Hierarchical neural collapse explicit parameterisation.
5. Information-theoretic floor I(Y; X) for CoDET-M4.

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
