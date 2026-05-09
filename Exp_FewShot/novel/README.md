# Exp_FewShot/novel/ — Theory-Driven Oral Bets

> **Goal:** EMNLP 2026 Oral submission. Each file in this folder must propose
> **one new mathematical object** that targets an open problem we cannot
> already solve, with a falsifiable claim and a theory hook.

## What lives here

- **One file = one novel claim.** Filenames `exp_nNN_<concept>.py` (n01, n02, …).
- **Self-contained inline files** — paste into a Kaggle cell, run.
- **Header docstring contract** (every file):
  - `NAME`
  - `ONE-LINE CLAIM`
  - `EQUATION` — exact math of the new object
  - `THEORY HOOK` — theorem / lemma / known result that motivates the choice
  - `WHY NOT BEFORE` — what changed (data, domain, formulation) that makes
    this newly applicable
  - `FALSIFIER` — the experimental result that would kill the claim

## Novelty Filter — the 5 gates

Every idea must clear all 5 before a file lands in this folder.

| Gate | Pass criterion |
|:--|:--|
| **G1 Theory-driven** | Is there a theorem/derivation that says it should work? Not "empirically helps." |
| **G2 ≤ 2 novel components** | Count new modules/losses vs the closest baseline. >2 → reject (engineering, not novelty). |
| **G3 Falsifiable claim** | One concrete experimental outcome would make the claim wrong. State it. |
| **G4 Differentiated** | Search 2–3 recent papers; what's the delta? "New domain alone" is not enough. |
| **G5 Compute-feasible** | ≤ 1 H100-hour OR ≤ 1 T4-hour per data point. >8h → cut something. |

Output every novelty check as the standard table:

```
## Novelty Check — <Idea Name>

| Gate | Pass? | Notes |
|------|-------|-------|
| Theory-driven | ✅/❌ | … |
| ≤2 novel components | ✅/❌ | … |
| Falsifiable claim | ✅/❌ | … |
| Differentiated | ✅/❌ | … |
| Compute-feasible | ✅/❌ | … |

**Verdict:** ✅ Implement / ⚠️ Fix [X] first / ❌ Reject
**Reason:** …
**If implement → Next:** exp_nNN_<name>, baseline=…, metric=…, est. …h
```

## Anti-patterns (do NOT add these)

- Loss-weight tuning ("we tried λ=0.4, 0.6, 0.8")
- Feature stacking ("add AST + token-stat + spectral all at once")
- Augmentation cocktails ("mixup + cutmix + random erase")
- Hyperparameter sweeps presented as method
- Any file whose value proposition is "we tried more components"

If the idea is "stronger version of an existing method", ship it under
`testing/` instead.

## Open problems (where novelty is welcome for our paper)

These are the unanswered questions where one new theorem-grounded object
could move the EMNLP story to oral-tier:

1. **K-shot collapse on AI-vs-AI siblings.** At K=32, codellama (1) and
   nxcode (4) — fine-tune siblings — both have F1 ≤ 0.04. CE has no
   information to separate them. **What single new mathematical object
   would identify the sibling-discriminating direction?**

2. **Source-confounding (CF/LC → GitHub OOD).** Train→test gap 0.71 →
   0.36 across 14 methods. Confound graph: \(Y \leftarrow S \rightarrow X\).
   **What back-door / front-door / proxy-causal object identifies
   \(P(Y \mid \operatorname{do}(X))\) without losing IID?**

3. **Phase transition at ~5K samples.** NTKAlign jumps from 0.29 (K=128 ≈
   768 samples) to 0.57 (1% ≈ 5K) to 0.665 (5% ≈ 25K). **Why does the
   phase transition happen there? Is there a theorem that predicts the
   sample-complexity floor for K-way authorship under family-genealogy
   structure?**

4. **Hierarchical neural collapse for generator families.** Galanti-Poggio
   2025: same-family classes share a parent simplex; sibling residuals
   are orthogonal in the parent's tangent space. **Can we explicitly
   parameterise this geometry and verify the orthogonality empirically?**

5. **Information-theoretic floor for 6-class authorship.** What is
   \(I(Y; X)\) for the CoDET-M4 distribution? If the empirical mutual
   information ceiling is below 0.85, a 0.71 detector is already near-
   optimal — and that's a contribution by itself.

A novel idea answers ONE of these with one new mathematical object.

## Workflow

1. Brainstorm an idea targeting one of the open problems above.
2. Run the Novelty Filter table — must pass 5/5 to land here.
3. Pick the next free `n` number, create `exp_nNN_<concept>.py`.
4. Header docstring includes the 6 mandatory fields.
5. Smoke-test on Kaggle T4. Paste log → tracker.md `novel/` section.
6. If FALSIFIER condition triggers, document the negative result and move
   the file to `legacy/novel_failed_<NN>_<concept>.py`. Do not delete.

## Existing entries

| File | Name | Targets open problem | New mathematical object | Theory hook |
|:--|:--|:-:|:--|:--|
| `exp_n01_sibling_residual.py` | SRD (Sibling-Residual Discriminant) | 1 | Fisher ratio in residual subspace orthogonal to family centroid | Galanti-Poggio Hierarchical Neural Collapse, arXiv:2501.09211 |
| `exp_n02_frontdoor_style.py` | FSM (Front-Door Style Mediator) | 2 | HSIC(z_style, S) on the source variable | Veitch-Wang front-door criterion, NeurIPS 2025 |
| `exp_n04_etf_simplex.py` | EFS (ETF Frozen Simplex) | 4 | Frozen K×D ETF classifier with ⟨W_i, W_j⟩ = -1/(K-1) | Galanti-Poggio 2025; Papyan-Han-Donoho neural collapse |
| `exp_n05_mi_floor.py` | MIF (Mutual-Info Floor) | 5 | MINE Donsker-Varadhan lower bound on I(Y; φ(X)) | Belghazi MINE ICML 2018; Fano's inequality |
| `exp_n06_proximal_sibling.py` | PCS (Proximal Causal Sibling) | 1 | 2-stage kernel-ridge proxy estimator using sibling as negative control | Mastouri-Gretton JMLR 2025 proximal causal inference |
| `exp_n07_conformal_mondrian.py` | CMP (Conformal Mondrian Prediction) | — | Per-class threshold τ_c calibrated on val s.t. FNR_c ≤ α | Vovk 2005; Romano-Patterson-Candès JASA 2020 |
| `exp_n08_spectral_eigengap.py` | SEA (Spectral Eigengap Authorship) | 3 | k-th eigengap of normalised label-similarity Laplacian | Cheeger inequality 1970; Lee-Oveis-Trevisan 2014 |
| `exp_n10_vib.py` | VIB (Variational Information Bottleneck) | 5 | Gaussian variational posterior + KL(q || N(0,I)) constraint | Tishby IB 2000; Alemi VIB ICLR 2017 |

> "Open problem" column indexes the 5 problems in the section above.
> n03, n09 reserved for future ideas (sample-complexity bound theorem;
> score-matching density estimator).
>
> Generator: `_generate_novel_batch.py` — reads `exp_n01_sibling_residual.py`
> as the template, substitutes per-method blocks. To add a new entry, append
> to the SPECS list in that file.
