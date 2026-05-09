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

## Existing entries (17 active novel methods, 2026-05-10)

| File | Name | Targets open problem | New mathematical object | Theory hook |
|:--|:--|:-:|:--|:--|
| `exp_n01_sibling_residual.py` | SRD | 1 | Fisher ratio in residual subspace orthogonal to family centroid | Galanti-Poggio Hierarchical Neural Collapse 2025 |
| `exp_n02_frontdoor_style.py` | FSM | 2 | HSIC(z_style, S) on the source variable | Veitch-Wang front-door criterion NeurIPS 2025 |
| `exp_n04_etf_simplex.py` | EFS | 4 | Frozen K×D ETF classifier with ⟨W_i, W_j⟩ = -1/(K-1) | Galanti-Poggio 2025; Papyan-Han-Donoho |
| `exp_n05_mi_floor.py` | MIF | 5 | MINE Donsker-Varadhan lower bound on I(Y; φ(X)) | Belghazi MINE ICML 2018; Fano |
| `exp_n06_proximal_sibling.py` | PCS | 1 | 2-stage kernel-ridge proxy estimator | Mastouri-Gretton JMLR 2025 |
| `exp_n07_conformal_mondrian.py` | CMP | — | Per-class threshold τ_c with FNR_c ≤ α | Vovk 2005; Romano-Patterson-Candès JASA 2020 |
| `exp_n08_spectral_eigengap.py` | SEA | 3 | k-th eigengap of normalised label-similarity Laplacian | Cheeger 1970; Lee-Oveis-Trevisan 2014 |
| **`exp_n09_pac_bayes_floor.py`** 🆕 | PSF | **3** | Empirical KL between author posterior and uniform prior; predicts N* | McAllester 1999; Catoni 2007 PAC-Bayes |
| `exp_n10_vib.py` | VIB | 5 | Gaussian variational posterior + KL(q ‖ N(0,I)) | Tishby IB 2000; Alemi VIB ICLR 2017 |
| **`exp_n11_vic_source.py`** 🆕 | VICS | **2** | Per-source variance hinge max(0, γ − √Var(z_s)) | Bardes-Ponce-LeCun ICLR 2022 (VICReg) |
| **`exp_n13_tree_wasserstein.py`** 🆕 | TWG | **1** | Closed-form tree-Wasserstein on 6-leaf genealogy | Le-Yamada-Cuturi NeurIPS 2019 (Tree-W) |
| **`exp_n14_sliced_wasserstein.py`** 🆕 | SWC | — | SW² over n=64 random projections per class pair | Bonneel-Rabin-Peyré-Pfister 2015 |
| **`exp_n15_tent_tta.py`** 🆕 | TENT | — | One-step entropy-min on LayerNorm at test time | Wang-Shelhamer ICLR 2021 (TENT) |
| **`exp_n16_energy_ood.py`** 🆕 | EBO | — | Free-energy margin loss + pseudo-OOD permutation | Liu et al. NeurIPS 2020 (Energy OOD) |
| **`exp_n17_prototypical.py`** 🆕 | ProtoCC | **1, 3** | EMA per-class prototype + cosine-CE | Snell NeurIPS 2017; Khosla 2020 (SupCon) |
| **`exp_n18_datamaps_curriculum.py`** 🆕 | DMC | — | Per-sample ambiguity = std of confidence across steps | Swayamdipta EMNLP 2020 (DataMaps) |
| **`exp_n19_irm.py`** 🆕 | IRM | **2** | Per-environment gradient-norm penalty | Arjovsky-Bottou-Gulrajani-Lopez-Paz 2019 |
| **`exp_n20_mlm_auxiliary.py`** 🆕 | MLM-Aux | — | Feature-level masked-prediction proxy | Devlin BERT 2019; Howard-Ruder ULMFiT 2018 |

> 🆕 = added 2026-05-10. Open problems: 1=K-shot sibling collapse · 2=Source-confounding (CF/LC→GH) · 3=Phase transition at ~5K · 4=Hierarchical neural collapse · 5=I(Y;X) ceiling.

### Coverage map (17 methods × 5 open problems)

| Problem | Targeting methods |
|:--|:--|
| **1 — K-shot sibling collapse (codellama↔nxcode)** | n01 SRD, n06 PCS, n13 TWG, n17 ProtoCC |
| **2 — Source-confounding (CF/LC→GH OOD)** | n02 FSM, n11 VICS, n19 IRM |
| **3 — Phase transition ~5K samples** | n08 SEA, n09 PSF, n17 ProtoCC |
| **4 — Hierarchical neural collapse parameterisation** | n04 EFS, n13 TWG |
| **5 — Information-theoretic ceiling** | n05 MIF, n10 VIB |
| Calibration / robustness / curriculum / TTA | n07 CMP, n14 SWC, n15 TENT, n16 EBO, n18 DMC, n20 MLM-Aux |
>
> Generator: `_generate_novel_batch.py` — reads `exp_n01_sibling_residual.py`
> as the template, substitutes per-method blocks. To add a new entry, append
> to the SPECS list in that file.

## 🎚️ Two-benchmark dispatch (2026-05-09)

Every novel file now supports BOTH benchmarks via `FS_BENCHMARK` env var:

| `FS_BENCHMARK` | Bench / Task | Classes | Primary metric | Paper baseline |
|:--|:--|:-:|:--|:-:|
| `codet_m4` (default) | CoDET-M4 6-class Author IID | 6 | Macro-F1 | UniXcoder 66.33 |
| `droid_t3` | DroidCollection T3 (3-class) | 3 | Macro-F1 / W-F1 | DroidDetectCLS-Large 88.78 |
| `droid_t4` | DroidCollection T4 (4-class incl. adversarial) | 4 | Macro-F1 / W-F1 | DroidDetectCLS-Large 94.30 |

```python
# Cell on Kaggle:
import os
os.environ["FS_BENCHMARK"] = "droid_t3"   # switch to Droid T3
# Paste exp_n02_frontdoor_style.py and run
```

JSON output filename encodes the benchmark:
```
exp_n02_frontdoor_style_K128_seed42.json            # CoDET (default)
exp_n02_frontdoor_style_droid_t3_K128_seed42.json   # Droid T3
exp_n02_frontdoor_style_droid_t4_K128_seed42.json   # Droid T4
```

**Cross-bench portability of each method:**

| Method | CoDET | Droid T3 | Droid T4 | Why |
|:--|:-:|:-:|:-:|:--|
| n01 SRD | ✅ ideal | ⚠️ no sibling pair in 3-class | ⚠️ same | sibling-restricted to codellama-nxcode |
| n02 FSM | ✅ | ✅ Droid has 3 domains too | ✅ | source-confounding general |
| n04 EFS | ✅ K=6 | ✅ K=3 | ✅ K=4 | ETF works for any K |
| n05 MIF | ✅ | ✅ but I(Y;X) less interesting at K=3 | ✅ | MINE is generic |
| n06 PCS | ✅ ideal | ⚠️ no sibling | ⚠️ same | proxy-causal needs sibling pair |
| n07 CMP | ✅ | ✅ | ✅✅ T4 adversarial sweet-spot | conformal calibration is generic |
| n08 SEA | ✅ K=6 | ⚠️ K=3 eigengap less informative | ⚠️ K=4 | Cheeger bound is K-sensitive |
| n10 VIB | ✅ | ✅ | ✅ | VIB is K-agnostic |

> Recommended Droid runs (subset that transfers cleanly):
> n02 FSM, n04 EFS, n07 CMP, n10 VIB. Skip n01/n06/n08 on Droid.
