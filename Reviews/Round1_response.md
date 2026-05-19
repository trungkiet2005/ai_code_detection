# Round 1 Response — TRACO Paper

> Reviewer date: 2026-05-19. This document maps each reviewer concern to a
> concrete action item, an experiment file (under
> `Reviews/R1_responses/`), and the expected rebuttal answer.

---

## At-a-glance: 9 rebuttal experiments

| # | Concern (W = weakness, Q = question) | File | Status |
|:-:|:--|:--|:--|
| W1 / Q1 | Sensitivity to errors in the family tree | `R1_responses/R1_01_treenoise.py` | drafted |
| Q2 | Hyperparameter sensitivity (γ, τ, λ) | `R1_responses/R1_02_hpsens.py` | drafted |
| W4 / Q6 | Open-set / unseen-generator evaluation | `R1_responses/R1_03_openset.py` | drafted |
| W6 / Q7 | Stronger encoder baseline (ModernBERT) | `R1_responses/R1_04_modernbert.py` | drafted |
| W7 / Q3 | AICD-T2 Macro-vs-Weighted F1 gap, per-class breakdown | `R1_responses/R1_05_perclass.py` | drafted |
| W2 / Q5 | Ad hoc HUMAN-vs-AI = 3 distance choice | `R1_responses/R1_06_treedist.py` | drafted |
| W5 | Held-out-domain (train Codeforces, test GitHub) | `R1_responses/R1_07_oodsrc.py` | drafted |
| W3 / Q4 | Augmentation semantic preservation rate | `R1_responses/R1_08_augsem.py` | drafted |
| Q8 | Hierarchical cross-entropy direct baseline | `R1_responses/R1_09_hierce.py` | drafted |

---

## Detailed response per concern

### W1 / Q1 — Sensitivity to tree noise
> "Sensitivity to errors in the family tree is unmeasured; real deployments
> often involve incomplete/ambiguous genealogy (closed-source models), and
> small tree mistakes could harm performance."

**Action:** `R1_01_treenoise.py`. We perturb the family tree at five noise
levels: 0%, 10%, 25%, 50%, 100% (i.e. completely random tree). For each, we
re-run TRACO at 1% CoDET-M4 and 1% AICD-T2 (the two slots most sensitive
to the prior) and report Macro-F1 + the Spearman of the noisy tree with
the clean one.

**Expected answer:** Macro-F1 degrades gracefully from 0% to 25% noise
(within $\pm$ 0.02), then collapses toward the CE-only baseline as noise
approaches 100%. This documents that the method tolerates the
closed-source-genealogy case (a few wrong edges) but cannot run without
SOME prior, which matches the paper's central claim.

### Q2 — Hyperparameter sensitivity
> "Can you provide a hyperparameter sensitivity study for γ, τ, and λ?"

**Action:** `R1_02_hpsens.py`. Grid sweep at 5% CoDET-M4 (most stable
slot) over $\gamma\!\in\!\{0.25, 0.5, 1.0, 2.0, 4.0\}$,
$\tau\!\in\!\{0.05, 0.1, 0.2, 0.5\}$, $\lambda\!\in\!\{0.1, 0.3, 0.5, 1.0\}$.

**Expected answer:** report Macro-F1 surface; identify the robust plateau
region around the paper's $(\gamma, \tau, \lambda) = (1.0, 0.1, 0.5)$
choice; report best- and worst-case Macro-F1 over the grid to bound the
sensitivity.

### W4 / Q6 — Open-set / unseen-generator
> "How does TRACO perform under open-set conditions ... A small-scale
> leave-one-generator/family-out evaluation would strengthen the
> deployment relevance."

**Action:** `R1_03_openset.py`. Leave-one-generator-out on CoDET-M4 (6
runs, each with 5 train classes + 1 held-out) and leave-one-family-out
on AICD-T2 (4 runs, each with 3 train families + 1 held-out). Evaluate
at 5% training data on the K-1 closed-set, then report the held-out
class as a NOT-IN-TRAIN bin (Macro-F1 over K-1 known + "unseen"
flag accuracy via prediction-entropy thresholding).

**Expected answer:** TRACO retains $\ge 85\%$ of its in-distribution
Macro-F1 over the K-1 known classes and identifies the unseen-class
samples with prediction-entropy AUC $\ge 0.75$, supporting deployment
relevance.

### W6 / Q7 — ModernBERT encoder
> "Could you add stronger encoder baselines (e.g., ModernBERT) under the
> same protocol, and/or show that TRACO's relative gains persist with a
> stronger backbone?"

**Action:** `R1_04_modernbert.py`. Drop in `ModernBERT-base` as the
encoder, keep everything else identical. Run all 6 slots (CoDET 1/5/20
+ AICD 1/5/20).

**Expected answer:** TRACO with ModernBERT either matches or slightly
exceeds the UniXcoder TRACO numbers; the relative gain over CE-only
remains at extreme few-shot, confirming the method is encoder-agnostic.

### W7 / Q3 — AICD-T2 Macro vs Weighted F1 gap
> "The AICD-T2 results exhibit large Macro vs. Weighted F1 gaps. Are
> classes heavily imbalanced ... Please include per-class breakdowns."

**Action:** `R1_05_perclass.py`. Diagnostic-only: load
`exp76_traco_results.json`, dump per-class F1, precision, recall, support
on AICD-T2 at all three fractions. Plot class-support distribution to
expose any imbalance. Compare to CE-only baseline (exp65_abl).

**Expected answer:** AICD-T2 has substantial class-size imbalance
(human-authored class is much larger than per-LLM classes); the
Weighted-F1 high value reflects correct predictions on the large class,
the Macro-F1 reflects difficulty on the smaller LLM classes. TRACO's
gains spread across most classes; we will add this breakdown to the
paper's Appendix.

### W2 / Q5 — HUMAN-vs-AI distance choice
> "You assign HUMAN–AI distance = 3 and cross-family = 4 on CoDET-M4. Why
> is human 'closer' to each AI family than families are to each other?"

**Action:** `R1_06_treedist.py`. Ablate the HUMAN-vs-AI default distance
$d_{\mathrm{hum}}\!\in\!\{1, 2, 3, 4, 5, \text{learned-scalar}\}$, keep
all other distances fixed. Re-run all 6 slots.

**Expected answer:** The choice $d_{\mathrm{hum}}\!=\!3$ was not the
strongest; we will replace it with the empirically best value (likely
$d_{\mathrm{hum}}\!=\!2$ or $3$) and report a 0.005-band of insensitivity
around the chosen default. We will rewrite the appendix paragraph
that introduces this constant to be a hyperparameter, not a hand-picked
prior.

### W5 — Held-out-domain robustness
> "There is no held-out-domain evaluation protocol (train on Codeforces,
> test on GitHub, etc.) to systematically quantify domain shift."

**Action:** `R1_07_oodsrc.py`. Train on Codeforces-only samples
(stratified per class), test on GitHub-only and LeetCode-only test
splits. Compare TRACO vs CE-only.

**Expected answer:** Both methods drop substantially on cross-domain
test (matching the post-hoc finding in our §7); TRACO's drop is similar
to CE-only's, confirming that source confounding is orthogonal to the
tree-weighting contribution.

### W3 / Q4 — Augmentation semantic preservation
> "How often do your augmentations change code semantics in practice?
> Can you report a small-scale compile/run check (per language) to
> quantify semantic preservation rates?"

**Action:** `R1_08_augsem.py`. Sample 500 Python snippets from CoDET-M4
training set. For each of the 4 augmentations, apply once, then
attempt `ast.parse` on both original and augmented. Report:
(i) parse-rate of augmented; (ii) AST-edit distance (when both parse);
(iii) for a subset of compile-and-run-eligible snippets, run-pass rate.

**Expected answer:** `comment_strip` and `ws_jitter` are 100%
parse-preserving; `id_rename` is 100% parse-preserving (since we only
rename non-reserved identifiers consistently); `token_dropout` breaks
parse rate to roughly 60%, but the encoder is trained to be invariant
to this surface noise -- semantic equivalence is not the right criterion
for encoder view augmentation. Discussion: contrastive view-augmentation
is a training-time regulariser; semantic equivalence is sufficient but
not necessary.

### Q8 — Hierarchical CE direct baseline
> "Have you tried a learned distance-bucket weighting (instead of fixed
> exp(-γ d_T)) or hierarchical cross-entropy as a direct baseline with
> the same encoder?"

**Action:** `R1_09_hierce.py`. Implement hierarchical cross-entropy:
two-stage softmax (family then within-family) on the SAME UniXcoder
encoder. Run all 6 slots and report side by side with TRACO.

**Expected answer:** Hierarchical CE lifts CE-only by $\approx 0.5$
Macro-F1 at 1% and saturates with the CE baseline at 20%, well below
TRACO. The contrastive form of hierarchy injection beats the
classification-head form at the same encoder budget, which we already
hinted at with TKL (exp63) but had not stated cleanly. We will add a
single row to Table 1 for hierarchical CE.

---

## Notes the reviewer made that we agree with

- "Mixes strong rhetorical claims with nuanced results." We will tone
  down "only ingredient that matters" to "the ingredient that
  contributes most across the slots we tested" in the next revision.
- "Negative baselines described briefly." We will expand the
  hyperbolic-prototype paragraph and the SetFit paragraph in the
  appendix with the specific hyperparameter sweep we ran for each.

## What we are NOT changing

- We will keep `unixcoder-base` as the default headline encoder; the
  ModernBERT comparison is for the appendix.
- We will keep the published-baseline numbers as is; the new ModernBERT
  TRACO row is supplementary.

---

## Folder layout

```
Reviews/
├── Round1.md              ← the original reviewer text
├── Round1_response.md     ← this file
└── R1_responses/          ← 9 self-contained Python rebuttal experiments
    ├── README.md          ← summary of all 9 with one-line takeaways
    ├── R1_01_treenoise.py
    ├── R1_02_hpsens.py
    ├── R1_03_openset.py
    ├── R1_04_modernbert.py
    ├── R1_05_perclass.py
    ├── R1_06_treedist.py
    ├── R1_07_oodsrc.py
    ├── R1_08_augsem.py
    └── R1_09_hierce.py
```

All experiment files follow the same Kaggle-friendly structure as
`Exp_FewShot/testing_chis/exp76_traco.py`: theory block at top, single
self-contained file, single JSON output. They live under `Reviews/`
rather than `Exp_FewShot/testing_chis/` because they are specifically
rebuttal artefacts for Round 1 and we want to keep them findable as a
unit.
