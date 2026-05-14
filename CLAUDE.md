---
description: 
alwaysApply: true
---

# CLAUDE.md — Repo Briefing (EMNLP 2026 Main target)

> **Read this file first.** One-stop theoretical + operational context.
> If anything here disagrees with the code, trust the code and fix this file.

---

## 0. Current submission target — UPDATED 2026-05-14

> **Venue:** EMNLP 2026 Main (long paper, 8 pages). **Aim: Oral.**
> **Deadline:** ~2026-05-26.
> **Mode:** Open portfolio. We are NOT locked to a single named method. We run
> many genealogy-aware objects (kernels, residuals, contrastive variants,
> temperatures, factorizations, alignments), let the leaderboard pick the
> empirical winners, and let the **ablation experiment (`exp65_abl`)** decide
> which **theoretical components** carry the gain. The paper writes itself
> around the survivors.

### 0.1 Frozen operational protocol (these are NOT design space)

These are settings every new experiment file must obey — they ensure runs are
comparable. They are NOT contributions and are NOT subject to change without
explicit user approval.

- **Encoder:** `unixcoder-base` only, `local_files_only=True` (ModernBERT dropped 2026-05-13).
- **Benchmarks:** `codet_m4` (Author 6-class, **Macro-F1**) and `aicd_t2` (model-family 12-class, **Macro-F1**). Droid T3/T4 skipped per 2026-05-10 directive.
- **Fractions:** `[0.01, 0.05, 0.20]` of the train split per class.
- **`_load_aicd` STRICT:** `FileNotFoundError` if T2 dir missing. No HF fallback.
- **AMP:** `torch.autocast(device_type='cuda', dtype=torch.bfloat16)`.
- **HW knobs:** `bs=256`, `seq=512`, `num_workers=4`, `pin_memory=True`. `_hw(cfg)` auto-downscales for <40GB VRAM.
- **AST features:** legacy-aligned 22-feature `extract_ast_features` (no tree-sitter dependency).
- **Schedule:** Regime-adaptive (`adaptive_schedule(cfg)`): 1%→10ep, 5%→6ep, 20%→6ep; LR sqrt-scaled for bs=256; cosine + warmup. This is the **engineering fix** isolated in exp56, not a contribution.
- **Always report `val_macro`, `test_macro`, `val_test_gap`.** Per repo-rules hook.
- **Output schema:** every new exp dumps a single combined `{expNN_method}_results.json` with full `eval_pack` (per-class P/R/F1, per-language F1, per-source F1, confusion matrix, sibling_confusion_rate, cross_family_confusion_rate, val_history).

> ⚠️ **DATA BUG (exp1–exp17 / baseline_01–04 / exp_n05–n16 original runs):**
> `_load_aicd()` path bug → AICD results from those experiments were T1 binary
> instead of T2 12-class. **Quote AICD results only from exp18+.** CoDET-M4
> results from those experiments are still reliable.

### 0.2 Paper strategy — Single-method hero, diverse baselines (Option 1)

> **Submission shape (decided 2026-05-14):** **EMNLP single-method paper**.
> One hero method in §3 Method, a diverse pool of competing methods in §4 as
> baselines + §5 ablation. NOT a multi-method portfolio / survey / benchmark
> paper.
>
> Why: EMNLP/ACL/NAACL convention is one clean contribution per paper. Oral
> papers at these venues are almost always single-method. Multi-method
> portfolio shapes belong to NeurIPS Datasets & Benchmarks, not main NLP.

### 0.2.1 Diversity = discovery tool, NOT contribution

The diversity matrix in §2.5 and the future expNN slate exist to **find** the
hero method, **validate** it against family-diverse alternatives, and **rule
out** that the gain comes from a single equation surface. Diversity is
*means*, not *end*:

| Role in paper | What goes there |
|:--|:--|
| §3 Method (hero) | ONE method, picked after exp65_abl + exp66-73 round |
| §4 Experiments table | Hero + 5–8 family-diverse competitors as baselines |
| §5 Ablation | exp65_abl + component decomposition of the hero |
| §6 Analysis | sibling/cross-family confusion, per-language, per-source |
| Appendix | Full diversity portfolio table (20 methods like legacy) |

### 0.2.2 No hero is locked yet — keep searching

> **DO NOT commit to a headline method until the user says "chốt".**
> Current standing (multiple methods tied at ~0.7186 on CoDET-M4 20%) is NOT
> a stable signal. We need:
>
>   1. **exp65_abl results** to decompose which component carries gain.
>   2. **exp66+ runs** (retrieval, MoE, hypernet, KAN, SSM, energy, distill,
>      IB) to populate the family-diverse alternatives and stress-test the
>      saturation hypothesis.
>   3. **Falsifier checks** (sibling_confusion_rate, learned hyperparams) for
>      every method in the pool.
>   4. **User decision.** The user says "chốt" → we then pick the hero based
>      on the lattice of (test macro-F1, ablation contribution, falsifier
>      passage, mechanism cleanliness). Until that moment, treat the headline
>      as an OPEN question.

### 0.2.3 What contribution claims the paper will make (still open form)

These are the contribution SLOTS that will be filled once a hero is picked.
They are NOT pre-decided claims:

- **A theoretical claim** — genealogy structure on the LABEL space defines new
  mathematical objects (target kernels, hierarchical losses, learnable distance
  weights, heat diffusion on label graphs, family-conditional temperatures,
  residual constraints, retrieval-stratified banks, MoE family routers,
  hypernet-generated weights, family-anchored energies). Any of these can
  carry the headline if it wins empirically AND survives ablation.
- **An empirical claim** — those objects beat the published full-data encoder
  baselines (UniXcoder 0.6633 on CoDET-M4) with a small fraction of training
  data. **Current 20% SOTA: 0.7186 (+0.0553 vs paper)**, achieved by multiple
  methods that converge to the same band — see tracker §0 "schedule was the
  dominant bottleneck". The hero must beat this band, not just match it.
- **A regime claim** — performance is dominated by encoder pretraining at
  small n and by the genealogy prior at large n; PTR (`exp61`) tests this
  directly. The phase transition is itself a contribution candidate.
- **A falsifier line** — every method in the pool commits to a falsifier
  metric (sibling_confusion_rate, learned hyperparam, etc.) logged in JSON.
  The hero must pass its falsifier; competing methods that pass their own
  falsifiers strengthen the family-diversity story.
- **A diagnostic ablation** — `exp65_abl` toggles components (CE / SSL / HTKA
  / GSCE / SCR / combinations) on the same backbone+schedule to isolate
  which loss object owns which fraction of the gain. THIS is what tells us
  which hero to pick.
- **A family-diverse baseline table** — competing methods from §2.5 cover at
  least 6 of the 20 families, so the hero is positioned against retrieval,
  MoE, hypernet, KAN, distill, IB alternatives — not just sibling-CE variants.
  Diversity makes the paper read as a portfolio, but only one method gets
  the §3 spotlight.

> **Working stance until "chốt":** Keep generating diverse, novel experiments
> (`exp66+`). Keep logging falsifiers. Keep filling the §2.5 family matrix.
> Resist any prose that promotes one method above the others. The hero is the
> last decision, not the first.

### 0.3 Current leaderboard snapshot (CoDET-M4 Author 20%, Macro-F1)

| Rank | Method | Score | Δ vs paper UniXcoder 0.6633 | Source |
|:-:|:--|:-:|:-:|:--|
| 🥇 | SSL-RAS    (exp56) | 0.7186 | +0.0553 | tracker 2026-05-14 |
| 🥈 | GSCE       (exp57) | 0.7185 | +0.0552 | tracker 2026-05-14 |
| 🥉 | RASL       (exp58) | 0.7181 | +0.0548 | tracker 2026-05-14 |
| 4 | GFTS       (exp59) | 0.7177 | +0.0544 | tracker 2026-05-14 |
| ref | SSL (exp42, old sched) | 0.6796 | +0.0163 | pre-RAS schedule |
| ref | UniXcoder full | 0.6633 | 0.0000 | paper baseline |

**The takeaway in the tracker:** all 4 methods land within 0.001 of each other
at 20% CoDET-M4. This is a signal that the schedule + the genealogy prior in
*aggregate* drives the gain, not any single loss object. The next round of
experiments (exp60–65) is designed to isolate WHICH genealogy prior, with
ablation `exp65_abl` providing the decomposition.

---

## 0bis. Original NeurIPS framing (kept for context)

The repository was originally built targeting NeurIPS 2026 Oral. We reframed to EMNLP Main when:
- 17 days to NeurIPS-like deadline made oral-tier breakthrough (OOD-gh > 0.40) unrealistic.
- Existing 20%-data SOTA (+4.70 / +5.20 / +0.98) is already a publishable EMNLP claim.
- Source confounding becomes §5 Analysis instead of §3 Method.

---

## 1. The central research question

> **How should a few-shot attribution model use the known genealogy of LLM generators instead of treating every author label as unrelated?**

## 2. Thesis — open formulation (2026-05-14 update)

The flat-simplex baseline is wrong on a label space that has metric structure.
Once we admit genealogy structure on labels, several new mathematical objects
become available, none of which are reducible to standard losses:

| Object | Surface | New under genealogy |
|:--|:--|:--|
| **Target kernel `T_ij = κ(y_i, y_j)`** | Representation Gram | `L = 1 − cos(vec(ZZᵀ), vec(T))`  (HTKA / DGK) |
| **Tree-tempered soft label** | Output distribution | `q_c = (1−ε)·1[c=y] + ε·exp(−α·d_tree)/Z` (GSCE) |
| **Sibling-weighted per-sample CE** | Logit error | `L = E[w_d · CE]`  (SSL, TKL learns w) |
| **Family-conditional temperature** | Softmax | `τ_c = exp(log τ₀ + β·s_c)` (GFTS) |
| **Genealogy residual constraint** | AST embedding | `R = AST − μ_gene[y]` (GRA, RASL) |
| **Heat-diffusion kernel** | Graph of labels | `K = exp(−tL)` (DGK) |
| **Sibling confusion regularizer** | Soft predictions | `L = E[Σ_sib softmax]` (SCR) |
| **Phase-transition curriculum** | Loss weight | `λ(n) = σ((n − n_c)/scale)` (PTR) |

None of these has a definition when labels are unstructured. This is the
**novelty floor**: every new experiment must add to this table or sharpen an
existing row. Reviewers cannot reduce any row to "X applied to Y" because
the X is undefined without the tree.

### 2.1 Headline object — undecided, search ongoing

The repository was originally framed around HTKA. After RAS-schedule rollout
and the exp56-59 results, the empirical SOTA is shared across SSL-RAS, GSCE,
RASL, GFTS within 0.001 Macro-F1 on CoDET-M4 20%. We DO NOT commit to a
single headline; the headline is decided by `exp65_abl`, by `exp66+`
runs (still pending — retrieval / MoE / hypernet / KAN / SSM / energy /
distill / IB families per §2.5), and by the falsifier metrics — not by the
order in which we implemented the methods.

**Hero-locking gate:** the hero is selected only when the user explicitly
says "chốt". Until then, treat HTKA / SSL-RAS / GSCE / RASL / GFTS as
co-equal candidates. New experiments should expand the candidate pool, not
narrow it.

### 2.2 Theory-grounded novelty bar — NOVELTY-FIRST PRINCIPLE

> 🚨 **EMNLP Oral demands NEW mathematical objects, not generic ML techniques applied to a new domain.**
> Every experiment must introduce something that **ONLY makes sense** in the context of
> AI-code attribution with genealogy structure. If a reviewer can say "this is just X applied to Y",
> the experiment is NOT novel enough.
>
> **Engineering knobs (schedule, LR, batch, AMP) are NOT contributions.** They
> live in §0.1 (frozen protocol). If a method's gain disappears under matched
> schedule, that method does not survive ablation. This is enforced by
> `exp65_abl` running all toggles under the SAME RAS schedule.
>
> **Family-diversity rule (added 2026-05-14):** when proposing the next exp,
> first check §2.5 family map. **Do NOT propose another variant of an
> already-saturated family** (F1 loss-reshape has 3 active variants — enough;
> F3 kernel alignment has 2 — enough). Pick a family marked ❌ or 🟡, or a
> family not yet on the matrix. A paper with 20 methods in 7 families ranks
> better than a paper with 4 methods in 1 family, even at identical SOTA.
>
> **Problem-specificity rule (added 2026-05-14):** every proposal MUST cite ≥ 1
> of the S1–S10 structural facts in §2.5.0 (dual-tree, decoding fingerprint,
> provenance, prompt sharing, AST motifs, multi-LM likelihood, temperature
> nuisance, source confounding, few-shot regime, intra-human substructure).
> Methods that work the same way for ImageNet+label-tree as for code
> attribution are REJECTED at the novelty gate. The acceptance test: the
> proposed equation contains ≥ 1 of `{AST_i, prompt_embed, log p_LM,
> source domain, decoding parameter}`.

**Anti-patterns (DO NOT propose):**
- SupCon, ArcFace, R-Drop, LoRA, SimCSE, Center Loss — anyone can think of these
- "Apply existing method to code attribution" alone — must have genealogy-specific twist
- Loss-weight tuning presented as novelty
- Feature stacking without new theory

**Required: each experiment must have a NEW MATHEMATICAL OBJECT** with:
- NAME / ARXIV_ID / ONE-LINE CLAIM / EQUATION / PROPERTY / WHY NOT BEFORE / FALSIFIER
- The "WHY NOT BEFORE" must explain why this object is specific to genealogy-structured attribution

```python
# =============================================================================
# Theory-Track exp -- Hierarchical Target-Kernel Alignment (HTKA / Hier-NTK):
#
# ARXIV_ID      : NeurIPS 2024 TKM (2410.06171); ICLR 2025 NC (2410.04887)
# NAME          : Hierarchical Target-Kernel Alignment (HTKA)
# ONE-LINE CLAIM: Few-shot LLM attribution improves when representation
#                 similarity is aligned to model genealogy.
# EQUATION      : L_htka = 1 - cos(vec(ZZ^T), vec(T_tree))
# PROPERTY      : T_tree is a lower-variance target than one-hot in few-shot.
# WHY NOT BEFORE: Code authorship baselines treat labels as a flat simplex.
# FALSIFIER     : If alignment rises but sibling/family errors do not fall.
# =============================================================================
```

### 2.3 Active experiments (as of 2026-05-12)

**Top-3 Novel (arxiv-grounded, proven on CoDET-M4):**

| File | Method | ArXiv | Theory Hook | Status |
|:--|:--|:--|:--|:--|
| `exp_n06_attn_pool.py` | HAP | Lee 2017 | Hierarchical attention pooling | ✅ Active |
| `exp_n07_mixup_align.py` | MGA | Arjovsky 2019 | IRM source invariance | ✅ Active |
| `exp_n09_etf_simplex.py` | ETF-Simplex | NC theory | Optimal K-class ETF geometry | ✅ Active (best novel) |

**Rescued theory-track (fraction protocol):**

| File | Method | ArXiv | Status |
|:--|:--|:--|:--|
| `exp20_proto.py` | PPN | NeurIPS 2017 | ✅ Rescued |
| `exp25_deconfound.py` | DRL/IRM | ICML 2019 | ✅ Rescued (penalty_anneal=50) |
| `exp26_graph.py` | GNA | NeurIPS 2017 | ✅ Rescued |
| `exp27_cmixup.py` | CMA | arXiv 2017 | ✅ Active |
| `exp28_spectral.py` | SCL | ICML 2021 | ✅ Active |
| `exp29_causal_trace.py` | CFT | ICLR 2021 | ✅ Active |

**🔬 Oral-Tier Novel Experiments (NEW — genealogy-specific objects):**

| File | Method | New Object | Why Genuinely Novel |
|:--|:--|:--|:--|
| `exp31_got.py` | GOT | Wasserstein loss with tree-metric ground cost | OT ground metric from genealogy — misclass cost ∝ tree distance |
| `exp32_spectral_gene.py` | SGE | Laplacian eigenvectors of label graph as target | Spectral methods on LABEL structure, not data graph |
| `exp33_renyi.py` | RAD | Rényi-α loss with regime-dependent optimal α | α-regime connection (α* ∝ log(n/K)) is new |
| `exp34_hyper_proto.py` | HPA | Prototypes in Poincaré ball for tree labels | Trees embed with zero distortion in hyperbolic space |
| `exp35_kac.py` | KAC | Phase-transition curriculum from OUR Theorem 2 | Tests our own theorem — ultimate self-falsifier |
| `exp36_mmdg.py` | MMDG | MMD with genealogy-defined kernel | Distribution matching with tree kernel |
| `exp37_igg.py` | IGG | Natural gradient with genealogy Fisher target | Optimisation geometry from label tree |

**🔬 Dual-Tree / Genealogy-Aware Experiments (exp38–exp55, post-refactor 2026-05-13):**

> **Motivation:** These experiments define NEW mathematical objects that only make sense
> when combining AST structure (how code is written) with genealogy structure (who wrote it).
> Six were refactored on 2026-05-13 from generic ML wrappers into genuinely novel
> tree-conditioned objects (TPNL, BGB, TKE, GIBA, HETE, GFR — see table below).

| File | Method (in code) | New Object | Theory Basis / Equation |
|:--|:--|:--|:--|
| `exp38_gaka.py` | GAKA | Cross-kernel alignment | `L = 1 − ⟨K_AST, K_GENE⟩_F / √(‖K_AST‖²·‖K_GENE‖²)` |
| `exp39_sto.py` | STO | Structural Transfer Operator | Transfer AST patterns weighted by tree distance |
| `exp40_hcdt.py` | **TPNL** *(was HCDT)* | Tree-Path Negative Lifting | InfoNCE w/ neg weights `exp(−α·d_tree(y_p, y_n))` |
| `exp41_cfi.py` | CFI | Code-Family Invariance | Invariance loss for AST-preserving perturbation |
| `exp42_ssl.py` | SSL | Sibling-Sensitive weighted CE | Per-sample CE × genealogy-distance weight matrix |
| `exp43_cta.py` | **BGB** *(was CTA)* | Batch Genealogy Bridge | Cross-sample attn modulated by `exp(−α·d_tree)` |
| `exp44_sgd.py` | **TKE** *(was SGD)* | Tree-Kernel Embedding (Bourgain) | `L = E[(‖z_i−z_j‖² − β·d_tree(y_i,y_j))²]` |
| `exp45_gra.py` | GRA | Genealogical residual | `R(x) = AST(x) − E[AST \| gene(x)]` |
| `exp46_cie.py` | CIE | Backdoor adjustment | `P(Y \| do(AST)) = Σ_s P(Y \| AST, S=s)·P(S=s)` |
| `exp47_ift.py` | **GIBA** *(was IFT)* | Genealogy InfoNCE Bottleneck | `CE − μ·I_NCE(z;fam) + ν·I_CLUB(z;src) + β·KL` |
| `exp48_dtr.py` | **HETE** *(was DTR)* | Heterogeneous Treatment Effects | `logit_y = base_y(φ) + Σ_k T_k(x)·τ_k(φ, y)` |
| `exp49_paca.py` | PAC-A | PAC-Bayes bound | Gaussian posterior + KL(Q‖P) regularisation |
| `exp50_sie.py` | SIE | Structural Invariance Equation | Z_2 sign-flip group action averaged over views |
| `exp51_frc.py` | FRC | Functional Representation Consistency | `‖proj_auth(emb) − proj_func(emb)‖²` |
| `exp52_cpo.py` | CPO | Counterfactual Permutation | Triplet-margin on AST-swapped counterfactuals |
| `exp53_rce.py` | RCE | Representation Causal Effect | Learnable causal gates on sem/ast components |
| `exp54_rde.py` | **GFR** *(was RDE)* | Genealogy-Factorized Representation | β-TCVAE + FactorVAE TC disc + family supervision on z_sty |
| `exp55_irm.py` | IRM | Invariant Risk Minimization | `‖∇_w R(w·φ)‖²` penalty via `torch.autograd.grad` |

> **Why genuinely novel:** Each requires either BOTH AST + genealogy structure, or a
> tree-conditioned mathematical object that is undefined when labels lack metric structure.

**🚨 Refactor manifest (2026-05-13) — 6 weak methods promoted to oral-tier objects:**

| Old (weak, in commit ≤ 87317ce) | New (oral-tier, in commit 1a5d3c5) | What was broken |
|:--|:--|:--|
| exp40 HCDT (non-canonical loss, O(B²) Python loop) | **TPNL** Tree-Path Negative Lifting | Loss = `−pos_term`, not real InfoNCE |
| exp43 CTA (per-sample `.item()` compat scalar) | **BGB** Batch Genealogy Bridge | Genealogy multiplier detached from autograd |
| exp44 SGD (Python `gene_distance` loop, ad-hoc α) | **TKE** Tree-Kernel Embedding | No metric anchoring; only positive margin |
| exp47 IFT (correlation, not MI) | **GIBA** Genealogy InfoNCE Bottleneck | Fake MI = `‖z·s‖`; no real estimator |
| exp48 DTR (trivial scalar τ per feature) | **HETE** Heterogeneous Treatment Effects | No class-conditional treatment effects |
| exp54 RDE (3 KL terms, no real independence) | **GFR** Genealogy-Factorized Representation | No TC penalty, no genealogy anchor on z_sty |

### 2.4 Novel Theorems (self-derived for EMNLP Oral)

See detailed statements in §2.4 of the original CLAUDE.md version (theorems 1–3 on kernel alignment bound, phase transition, sibling confusion bound). These are self-derived and represent the Oral-tier contribution.

### 2.5 Method-Diversity Matrix — PROBLEM-SPECIFIC DISCOVERY MANDATE (rewritten 2026-05-14)

> **Purpose of diversity:** to FIND the hero method that will sit in §3 of the
> paper, and to populate §4's baseline table with family-diverse alternatives.
> Diversity is a search tool and a baseline source, **NOT a contribution by
> itself**. The paper is single-method (§0.2). The DISCOVERY phase, however,
> must be diverse — AND every entry in the discovery pool must be ANCHORED
> in the structure of AI-code attribution, not generic ML applied to code.
>
> **Lesson from tracker analysis 2026-05-14:** legacy `Exp_CodeDet` had **20 methods spanning 7+ families**, with author-F1 spread of **1.7 points** (top: DeTeCtive 71.53 → bottom: CosineProto 67.80). Current `testing_chis` RAS-track (exp56-59) has **4 methods all in ONE family** (sibling-weighted CE variants on the same backbone) → spread of **0.001 points** — TIED.
>
> If the spread stays tied across ALL families, the legitimate paper story is "schedule + any genealogy prior saturates"; if a non-loss family breaks the tie, that family's representative becomes the hero candidate. Either way we need the diverse pool BEFORE we can defend any single-method claim.

### 2.5.0 What is structurally unique about AI-code attribution

These are the structural facts that make this problem NOT just "text
classification with a tree label space". Every method proposal must exploit
at least one of them; methods that ignore them are reduced to "X applied to Y"
and fail the §2.2 novelty bar.

| # | Structural fact about AI-code attribution | What new objects it enables |
|:-:|:--|:--|
| **S1** | **Two trees co-exist.** Each sample carries an AST (syntax tree). Each label carries a genealogy tree (model family). Cross-domain rare | Dual-tree kernels, AST × genealogy joint geometry, AST-conditional family priors |
| **S2** | **Generators have known decoding fingerprints.** Temperature, top-p, repetition penalty, max-tokens are model-specific and leak into output | Decoding-artifact probes, temperature regression heads, repetition pattern features |
| **S3** | **Generators trained on overlapping but distinct corpora** (Stack, BigCode, code books, GitHub). Pretraining provenance leaks into generations | Provenance-trace heads, corpus-style classifiers, cross-corpus invariants |
| **S4** | **Prompts are shared across generators in standard benchmarks** (CodeT-M4, AICD generate from the SAME problem statements). Prompt content is a shared confound | Prompt-invariant attribution, prompt-residual encoding, problem-conditional style |
| **S5** | **Code under same author has CHARACTERISTIC AST motifs** (specific loop styles, naming conventions, error-handling idioms) | AST subtree mining, motif retrieval banks, family-specific syntactic priors |
| **S6** | **The detector can re-encode the input** under each candidate LM and obtain a likelihood vector. This is unique to generative-content attribution | Multi-LM likelihood features, cross-decoder consistency, perplexity signatures |
| **S7** | **Same prompt + different temperatures from same model produce different outputs.** Author identity is invariant; temperature is a nuisance | Temperature-augmented training, invariance to decoding hyperparam |
| **S8** | **Source domain (cf / lc / gh) is observed and confounding.** GitHub code is more diverse, LeetCode is templated, Codeforces is competitive | Source-stratified backdoor adjustment, source-conditional family priors, OOD-source held-out evaluation |
| **S9** | **Few-shot regime is the operating point.** Production detectors must adapt to NEW generators added monthly | Few-shot meta-attribution, prototype generators, retrieval-as-attribution |
| **S10** | **Human code is a class, not just "not AI".** Human ≠ flat distribution; humans cluster by skill / language / source | Human-cluster prototypes, intra-human substructure as auxiliary signal |

> **The novelty bar (§2.2) translated through this matrix:** A proposed method
> is acceptable only if its NEW MATHEMATICAL OBJECT requires ≥ 1 of S1–S10 to
> even be definable. If you can write the same equation for image
> classification with a label tree, the method is rejected at the novelty
> gate.

**Family map of what we have / lack:**

| # | Family | Mechanism | Existing (active) | Status | Gap-fill target |
|:-:|:--|:--|:--|:-:|:--|
| F1 | Loss reshape — sibling-weighted CE | per-sample CE × `w_d(y, ŷ)` | SSL, TKL, SCR | ✅ saturated | — (already 3 variants) |
| F2 | Loss reshape — soft labels | KL with tree-tempered target | GSCE | ✅ |  |
| F3 | Kernel / Gram alignment | `cos(ZZᵀ, T)` or `‖ZZᵀ − T‖²` | HTKA, DGK | ✅ |  |
| F4 | Contrastive / InfoNCE | tree-weighted positives/negatives | TPNL, GIBA, BGB | ✅ |  |
| F5 | Temperature / output scaling | per-class `τ_c` from topology | GFTS | ✅ |  |
| F6 | Residual / disentangle | AST − E[AST \| gene] | GRA, RASL, GFR | ✅ |  |
| F7 | Curriculum / phase-transition | `λ(n) = σ((n−n_c)/scale)` | PTR, KAC | ✅ |  |
| F8 | Causal / backdoor adjustment | `P(Y\|do(X)) = Σ_s P(Y\|X,S)P(S)` | CFT, CIE, HETE, RCE | ✅ |  |
| F9 | Optimal transport | Wasserstein on label-tree ground cost | GOT, MMDG | ✅ |  |
| F10 | Spectral / Laplacian | label-graph eigenvectors | SGE, DGK | ✅ |  |
| F11 | Hyperbolic / non-Euclidean | Poincaré prototypes | HPA | ✅ |  |
| F12 | Bayesian / uncertainty | PAC-Bayes, MC dropout | PAC-A, BUA | ✅ |  |
| **F13** | **Retrieval-augmented** | RAG bank keyed by gene clusters | **(none active)** | ❌ MISSING | **exp66_retag** |
| **F14** | **Mixture-of-Experts** | one expert / family, gene-gated | (legacy MoECode only) | ❌ MISSING | **exp67_gmoe** |
| **F15** | **Hypernetwork** | weights generated from family ID | (legacy HyperCode only) | ❌ MISSING | **exp68_hyper** |
| **F16** | **KAN / non-linear heads** | learnable per-feature splines | (legacy KANCode only) | ❌ MISSING | **exp69_kang** |
| **F17** | **State-space backbones** | Mamba / S4 over code tokens | (legacy MambaCode only) | ❌ MISSING | **exp70_smamba** |
| **F18** | **Energy / score-based** | `E(x) = ‖z − μ_y‖² + family term` | (legacy EnergyCode only) | ❌ MISSING | **exp71_genergy** |
| **F19** | **Self-distillation** | family-teacher → author-student | (legacy SelfDistill only) | ❌ MISSING | **exp72_gdistill** |
| **F20** | **Information bottleneck** | I(Z; Y) − β·I(Z; X) with genealogy-conditioned prior | GIBA partial; legacy IBCode | 🟡 | exp73_gib_full |

**Diversity target before EMNLP submission:** **≥ 10 families with active runs**. Currently we have 12 families covered but 5 of them are LEGACY-only (architecture-driven gains that we never ported under the RAS schedule). The 7 ❌MISSING and 1 🟡PARTIAL families are the priority slots.

**Future expNN slate (do NOT reuse IDs; PROBLEM-SPECIFIC, not generic ML):**

Each proposal below cites at least one S1–S10 structural fact it exploits.
If a proposal cannot cite ≥ 1, it does not belong in this slate.

| ExpID | Method (proposed) | Exploits | New object — UNDEFINED for flat classification |
|:--|:--|:-:|:--|
| `exp66_dtke` | **DTKE** Dual-Tree Kernel Embedding | S1 | `K_dual(x_i, x_j) = K_ast(AST_i, AST_j) · K_gene(y_i, y_j)`. Joint kernel over per-sample AST tree AND per-label genealogy tree. Both trees are required to define K_dual. Aligns ZZᵀ to K_dual. |
| `exp67_provtr` | **PROVTR** Provenance-Trace Head | S3 | Auxiliary head predicts pretraining-corpus signature (Stack / TheStack / BigCode / human-GitHub) from AST + token n-grams. Loss = CE on attribution + γ · CE on corpus probe. Family identity is read OFF the provenance signature. |
| `exp68_decofp` | **DECOFP** Decoding-Fingerprint Regressor | S2 | Regression head predicts (estimated_temperature, estimated_top_p, repetition_entropy) from the code. These quantities differ per generator family by design. Attribution head conditioned on the predicted decoding triple. |
| `exp69_promptinv` | **PROMPTINV** Prompt-Invariant Attribution | S4 | Residual `z_attr(x) = z(x) − z_prompt_anchor(family_of(y))`. The prompt embedding is the shared confound; subtracting a family-anchored prompt prototype isolates author-specific style. |
| `exp70_astmotif` | **ASTMOTIF** AST Subtree Motif Retrieval | S5 | Mine top-K AST subtree motifs per family (frequent in family `f`, rare elsewhere). Test-time: hash query AST → sparse vote over family motifs + dense logit fusion. Fully discrete + dense hybrid. |
| `exp71_multilm` | **MULTILM** Multi-LM Likelihood Signature | S6 | Feature vector = `[ℓog p_GPT(x), log p_CodeLlama(x), log p_Qwen(x), log p_Nxcode(x), log p_Human-LM(x)]`. The detector classifies on this likelihood SIGNATURE, not on raw embeddings. |
| `exp72_tempaug` | **TEMPAUG** Temperature-Invariant Training | S7 | Train with paired (x, x_alt) where x_alt is the same author re-sampled at higher T. Add invariance loss `‖z(x) − z(x_alt)‖²`. Author identity is the invariant; T is the nuisance. |
| `exp73_srcbdoor` | **SRCBDOOR** Source-Stratified Backdoor Adjustment | S8 | `P(Y \| do(X)) = Σ_s P(Y \| X, S=s) · P(S=s)`. Train one classifier head per source domain; aggregate via observed source marginal. Targets the cross-source confound directly. |
| `exp74_protogen` | **PROTOGEN** Prototype-Generator Few-Shot Meta | S9 | Each generator gets a prototype embedding learned via episodic meta-attribution. Test-time attribution = nearest prototype. Designed for the "new generator added monthly" production regime. |
| `exp75_humclust` | **HUMCLUST** Intra-Human Substructure | S10 | Cluster the human class into K subclusters (by skill / source / language signal). Train (K+5)-way head; collapse to original 6-way at inference. Human-class richness as auxiliary signal. |

These cover families F13–F20 from the abstract diversity matrix but each row
is now a CODE-ATTRIBUTION-SPECIFIC mechanism, not "MoE applied to code". A
reviewer cannot say "this is X applied to Y" for any of them because the X is
undefined without the matching S-structural fact.

> **Picking the next exp from this slate (when user says "tạo thêm exp"):**
>   1. Open `tracker.md` and list which S-facts are currently UNEXPLOITED.
>   2. Pick a proposal from the slate above that exploits the most-unexploited
>      S-fact.
>   3. Confirm the new object is UNDEFINED under flat-simplex classification.
>      If you can write the same equation for ImageNet with a label tree, the
>      proposal fails — pick a different S-fact.
>   4. Write the file with the same self-contained structure as `exp60_htka.py`
>      (Kaggle paths, STRICT `_load_aicd`, AMP bf16, RAS schedule, full
>      `eval_pack`).
>
> **Anti-patterns at the proposal stage:**
> - "Apply Mamba/MoE/KAN to code" — generic; reject unless tied to a specific
>   S-fact and the architectural change is REQUIRED by that fact.
> - "Hypernet generates classifier weights from family ID" — possible but
>   weak: family ID is a categorical, hypernet adds capacity not structural
>   bias. Reject unless tied to a deeper S-fact.
> - "Add genealogy-aware regularisation to MoE" — feature stacking, not new
>   object.
> - "Use a graph neural network on the genealogy tree" — generic spectral
>   trick, already covered by SGE / DGK.
>
> **Acceptance pattern:** the proposed object's equation contains ≥ 1 of
> { AST_i, prompt_embed, log p_LM, source domain, decoding parameter }. These
> are the IRREDUCIBLE tokens of AI-code attribution — they cannot be defined
> away.

---

## 3. Structural causal model

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

## 5. Benchmarks (role-specified)

| Short name | Role | Primary metric | Paper baseline |
|:--|:--|:--|:--|
| **CoDET-M4 Author** | **Main evaluation** — generator-level attribution | **Macro-F1** (6-class) | UniXcoder 66.33 |
| **AICD-Bench T2** | **Secondary stress test** — model-family attribution | **Macro-F1** (12-class) | AICD T2 paper |
| **DroidCollection T3/T4** | SKIPPED (per 2026-05-10 directive) | — | — |
| **AICD T1/T3** | Limitation / appendix only | — | T1 has known val→test collapse |

**Protocol (non-negotiable, as of 2026-05-13):**
- Encoder: `unixcoder-base` only
- Fractions: `[0.01, 0.05, 0.20]`
- Benchmarks: `codet_m4`, `aicd_t2`
- **`_load_aicd()` is STRICT:** `FileNotFoundError` if target task dir missing. NO fallback to other tasks. NO HuggingFace fallback.
- **All model + tokenizer loads:** `local_files_only=True` (offline). Paths only from `KAGGLE_*` constants.
- **AMP:** `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` — native on Ada (RTX Pro 6000), avoids GradScaler overflow.
- **DataLoader:** `num_workers=4`, `pin_memory=True`, `bs=256`, `seq=512` (auto-down-scaled by `_hw(cfg)` if VRAM < 40 GB).
- **AST features:** legacy-aligned 22-feature structural vector (`extract_ast_features`, mirrors `legacy/Exp_DM_weak/exp06_ast_irm.py::extract_structural_features`), padded to `max_len`. No tree-sitter.
- Train separate model per benchmark. Never mix.
- Report the full metric pack: Primary, Macro-F1, Weighted-F1, per-class.
- **Final evaluation is always on TEST set** (val set used only for checkpoint selection).
- **Always report val-test gap** (per repo-rules hook).

**JSON Output Schema (simplified for fraction protocol):**
```json
{
  "tag": "exp_n09_etf_unixcoder_codet_m4_f0.05",
  "enc": "unixcoder-base",
  "bench": "codet_m4",
  "frac": 0.05,
  "macro": 0.67,
  "weighted": 0.71,
  "acc": 0.72,
  "dpaper": 0.007,
  "per_class_f1": [...],
  "confusion_matrix": [[...]],
  "train_history": [...],
  "val_history": [...],
  "wall": 120.5,
  "timestamp": "2026-05-12 18:00:00"
}
```

## 7. Repo layout (post-refactor 2026-05-13)

```
ai_code_detection/
├── README.md
├── CLAUDE.md                            ← THIS FILE
├── Exp_Climb/                           # 20%-data climb (CoDET + Droid), FROZEN for paper
│   └── tracker.md
├── Exp_CodeDet/                         # CoDET-M4 full-data runs, FROZEN
├── Exp_DM/                              # 5 champions on AICD + Droid, FROZEN
├── Exp_FewShot/
│   └── testing_chis/                   # ← ACTIVE EXPERIMENT TRACK
│       ├── tracker.md                  # Primary leaderboard for fraction-protocol runs
│       ├── legacy/                     # Archived weak/failed experiments (read-only)
│       │
│       │  # === BASELINES (published) ===
│       ├── baseline_01_codet5.py       # CodeT5-Authorship (AISec 2025)
│       ├── baseline_02_detective.py    # DeTeCtive (arXiv 2410.20964)
│       ├── baseline_03_faid.py         # FAID (EACL 2026)
│       │
│       │  # === OUR METHODS ===
│       ├── exp_n13_hier_tree.py        # HierTree only
│       ├── exp_n14_ntk_align.py        # NTK only
│       ├── exp_n15_hier_ntk.py         # Hier-NTK (MAIN METHOD)
│       ├── exp_n16_ce_baseline.py      # CE baseline (floor)
│       │
│       │  # === TOP-3 NOVEL (arxiv-grounded) ===
│       ├── exp_n06_attn_pool.py        # HAP
│       ├── exp_n07_mixup_align.py      # MGA
│       ├── exp_n09_etf_simplex.py      # ETF-Simplex (best novel)
│       │
│       │  # === RESCUED THEORY-TRACK (fraction protocol) ===
│       ├── exp20_proto.py              # PPN (Prototypical Networks)
│       ├── exp25_deconfound.py         # DRL/IRM (penalty_anneal=50)
│       ├── exp26_graph.py              # GNA (Graph Neural Attribution)
│       ├── exp27_cmixup.py             # CMA (Family-aware Mixup)
│       ├── exp28_spectral.py           # SCL (Spectral Contrastive)
│       ├── exp29_causal_trace.py       # CFT (Causal Feature Tracing)
│       │
│       │  # === ORAL-TIER NOVEL (single-tree, exp31–37) ===
│       ├── exp31_got.py                # GOT (OT w/ tree ground cost)
│       ├── exp32_spectral_gene.py      # SGE (Laplacian eigvecs of label graph)
│       ├── exp33_renyi.py              # RAD (Rényi-α, regime-dependent)
│       ├── exp34_hyper_proto.py        # HPA (Poincaré-ball prototypes)
│       ├── exp35_kac.py                # KAC (phase-transition curriculum)
│       ├── exp36_mmdg.py               # MMDG (MMD w/ genealogy kernel)
│       ├── exp37_igg.py                # IGG (natural grad w/ genealogy Fisher)
│       │
│       │  # === DUAL-TREE + ORAL-TIER REFACTORS (exp38–exp55) ===
│       ├── exp38_gaka.py               # GAKA  cross-kernel alignment
│       ├── exp39_sto.py                # STO   structural transfer operator
│       ├── exp40_hcdt.py               # TPNL  tree-path negative lifting (refactored 2026-05-13)
│       ├── exp41_cfi.py                # CFI   code-family invariance
│       ├── exp42_ssl.py                # SSL   sibling-sensitive weighted CE (grad fix 2026-05-13)
│       ├── exp43_cta.py                # BGB   batch genealogy bridge (refactored 2026-05-13)
│       ├── exp44_sgd.py                # TKE   tree-kernel embedding (refactored 2026-05-13)
│       ├── exp45_gra.py                # GRA   genealogical residual
│       ├── exp46_cie.py                # CIE   backdoor adjustment (dim fix 2026-05-13)
│       ├── exp47_ift.py                # GIBA  genealogy InfoNCE bottleneck (refactored 2026-05-13)
│       ├── exp48_dtr.py                # HETE  heterogeneous treatment effects (refactored 2026-05-13)
│       ├── exp49_paca.py               # PAC-A PAC-Bayes (grad fix 2026-05-13)
│       ├── exp50_sie.py                # SIE   Z_2 sign-flip group invariance (crash fix 2026-05-13)
│       ├── exp51_frc.py                # FRC   functional representation consistency
│       ├── exp52_cpo.py                # CPO   counterfactual permutation (method fix 2026-05-13)
│       ├── exp53_rce.py                # RCE   representation causal effect
│       ├── exp54_rde.py                # GFR   genealogy-factorized representation (refactored 2026-05-13)
│       └── exp55_irm.py                # IRM   invariant risk minimization
├── Paper/                              # EMNLP 2026 submission
│   ├── outline.md
│   └── latex/main.tex
├── legacy/                             # Archived suites (read-only)
│   ├── Exp_ZeroShot/
│   ├── Exp_TK/
│   └── Exp_DM_weak/
└── docs/
    └── references/
```

**Single sources of truth:**
- Fraction-protocol leaderboard → [Exp_FewShot/testing_chis/tracker.md](Exp_FewShot/testing_chis/tracker.md)
- 20%-data climb leaderboard → [Exp_Climb/tracker.md](Exp_Climb/tracker.md)
- Paper draft → [Paper/latex/main.tex](Paper/latex/main.tex)

## 8. Current standing (as of 2026-05-12)

**CoDET-M4 results (fraction protocol, unixcoder-base — VALID):**

```
1% (~5K samples)    unixcoder CE-baseline ≈ 0.383
5% (~25K samples)   unixcoder CE-baseline ≈ 0.573
                    Hier-NTK              ≈ 0.671  🏆 BEATS PAPER UniXcoder full (0.6633)
20% (~100K samples) CE-baseline           ≈ 0.680
                    ETF-Simplex           ≈ 0.683
```

**AICD-T2 results (fraction protocol, unixcoder-base — PENDING re-run with correct T2 loader):**
- All prior AICD results (from exp1–17 original runs) are INVALID (T1 binary data). Re-runs pending.

**Theory-track (exp20, exp25, exp26, exp27, exp28, exp29):**
- Now on fraction protocol. Results pending (first runs under new protocol).
- DRL/IRM: `penalty_anneal` fixed to 50 (was 500 — IRM never activated before).
- Proto/GNA: fraction-based sampling should provide enough sample diversity.

## 9. EMNLP rubric self-check

1. ✅ **Strong empirical results** — +5.20 Author / +4.70 lean / +0.98 Droid all locked.
2. ✅ **Multi-benchmark coverage** — CoDET-M4 + AICD-T2 (pending clean re-run).
3. ✅ **Honest limitations** — AICD T1 collapse, OOD-gh far below 0.40, documented.
4. 🟡 **Method novelty** — must be framed as HTKA, not "NTK loss + hierarchy."
5. 🟡 **OOD story** — characterisation only; reviewer may ask for solution.

## 10. How to work in this repo

### Run an experiment
```bash
# All self-contained, Kaggle-runnable
cd Exp_FewShot/testing_chis/
python exp_n15_hier_ntk.py       # main method
python exp_n09_etf_simplex.py    # best novel
python exp20_proto.py            # rescued theory
```

Each file runs: `unixcoder-base × [codet_m4, aicd_t2] × [1%, 5%, 20%]` = 6 runs.
Results → `results/expXX_YYY_results.json`.

### Add a new experiment
1. Write the 6-line theory block (NAME / ARXIV_ID / EQUATION / PROPERTY / WHY NOT BEFORE / FALSIFIER).
2. Copy closest active `exp_nXX_*.py` or one of the oral-tier refactors (`exp40_hcdt` for TPNL-style contrastive, `exp43_cta` for BGB-style batch attention, `exp44_sgd` for TKE-style metric, `exp47_ift` for GIBA-style InfoNCE-MI, `exp48_dtr` for HETE-style per-class CATE, `exp54_rde` for GFR-style disentanglement). Change only the method-specific loss / model.
3. Use fraction-based `build_dls(cfg)` with `cfg.frac`. Do NOT use `FIXED_TOTAL_TRAIN`.
4. Use `unixcoder-base` as the encoder. Do NOT add ModernBERT.
5. `from_pretrained(..., local_files_only=True)` for both `AutoModel` and `AutoTokenizer`.
6. Wrap forward+loss in `torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=(cfg.device == 'cuda'))`.
7. Use `num_workers=4`, `pin_memory=True` in `DataLoader`.
8. Match the `_hw(cfg)`, `=== {tag} ===`, `[epoch N] val=...`, and single-combined-JSON save pattern from `exp_n19_detective_lite.py` / `exp40_hcdt.py`.
9. Register in tracker.md after run with val + test + Δ.

### Conventions
- Experiment IDs never reused.
- Primary metric: Macro-F1 for CoDET-M4 and AICD-T2.
- Always report val + test + Δ together.
- `_load_aicd(task)` is STRICT: call with `"t2"` for AICD-T2, raises immediately if dir missing.
- `legacy/` is read-only. Do not run scripts from it.

## 11. Paper artefacts

- **EMNLP 2026 draft**: [Paper/latex/main.tex](Paper/latex/main.tex)
- **Section outline**: [Paper/outline.md](Paper/outline.md)
- **Benchmark papers**: `docs/references/paper_AICD.md`, `paper_Droid.md`, `paper_CodeDet_M4.md`

## 12. Time-wasters to avoid

- **DO NOT** add `FIXED_TOTAL_TRAIN` or `ModernBERT` to any new experiment.
- **DO NOT** add HuggingFace fallback back into `_load_aicd()`. STRICT mode is intentional.
- **DO NOT** quote AICD results from exp1–17 — they are T1 binary (invalid).
- **DO NOT** pivot scope. Headline numbers locked.
- **DO NOT** mix benchmarks in a single training run.
- **DO NOT** reuse an `expNN` ID.
- **DO NOT** rewrite historical tracker rows. Append-only.
- `legacy/` is archived. Do not import from it.
- Large negative Δ on AICD T1 is expected (val→test collapse). §6 Limitations only.

## 13. Cold-start checklist (2026-05-13)

1. Read §0 (protocol, data bug warning) and §8 (locked numbers).
2. Open [Exp_FewShot/testing_chis/tracker.md](Exp_FewShot/testing_chis/tracker.md) — current leaderboard.
3. Verify `_load_aicd("t2")` in any file you touch — confirm it is STRICT (no fallback, raises FileNotFoundError if T2 dir missing).
4. Active files are in §7 layout. Only edit files listed there.
5. If asked about AICD results: **only results from exp18+ (with correct T2 loader) are valid**. Earlier runs were T1 binary.
6. `Exp_Climb/`, `Exp_CodeDet/`, `Exp_DM/` are FROZEN. Do not add new experiments there.
7. **2026-05-13 refactor note:** if you see `HCDT`, `CTA`, `SGD`, `IFT`, `DTR`, or `RDE`
   referenced in old PRs / tracker rows, they have been refactored in-place to
   **TPNL / BGB / TKE / GIBA / HETE / GFR** respectively. Files keep the same expNN ID
   (`exp40_hcdt.py` etc.) but the `method` field in JSON output, the JSON filename
   (`exp40_tpnl_results.json`), and the in-code logger name (`exp40_tpnl`) reflect the
   new object. See §2.3 refactor manifest.
8. **GPU:** code is tuned for RTX Pro 6000 (96 GB Ada). `_hw(cfg)` auto-down-scales for
   smaller VRAM. AMP is `bf16` (no GradScaler dependence even if `GradScaler(enabled=False)`
   is still constructed).

## 14. Decision rules (what NOT to do unprompted)

- **Never** add a new method to `Exp_Climb/` / `Exp_CodeDet/` / `Exp_DM/` without user approval.
- **Never** re-open Zero-Shot work — archived for −32pt reproduction gap.
- **Never** add `dirs_to_try` fallback or HuggingFace fallback to `_load_aicd()`.
- **Never** drop `local_files_only=True` from `AutoModel.from_pretrained` / `AutoTokenizer.from_pretrained`.
- **Never** revert AMP back to fp16 + GradScaler on RTX Pro 6000 — bf16 is preferred (no scaler overhead, wider dynamic range).
- **Never** re-introduce the per-tag JSON save inside `run_exp` (one combined `{expNN_method}_results.json` per file, all 6 rows).
- **Never** quote AICD numbers from exp1–17 as T2 results — they are T1 (binary).
- **Always** report val-test gap alongside test metrics.
- **Always** use fraction sampling, not fixed sample count.
- **Always** match the legacy-aligned `extract_ast_features` schema (22 features, padded) — do not regress to the 12-feature stub.
- **Never** declare a "headline method" or "hero method" in prose, commit
  messages, or section headers until the user has explicitly said **"chốt"**.
  Before that: every method is a candidate. Use neutral language ("current
  top performer", "leading candidate") rather than "our method" / "the method".
- **Never** narrow the candidate pool unprompted. If `exp65_abl` or a new
  exp produces a winner, log it but keep the others as live candidates until
  user instruction.
- **Always** keep proposing experiments from ❌ / 🟡 families in §2.5
  (retrieval, MoE, hypernet, KAN, SSM, energy, distill, IB) when the user
  asks for "more exps" — these are the gaps in the discovery portfolio.

## 15. Exp_FewShot/testing_chis rules (active suite, 2026-05-13)

### Protocol
- Encoder: `unixcoder-base` only (loaded with `local_files_only=True`)
- Fractions: `[0.01, 0.05, 0.20]`
- Benchmarks: `codet_m4`, `aicd_t2` (T2 = 12-class model-family)
- Sampling: per-class fraction sampling (not fixed total)
- `_load_aicd` STRICT — no fallback, no HF download
- AMP: `bfloat16` autocast; `DataLoader(num_workers=4, pin_memory=True)`
- AST: 22-feature legacy-aligned `extract_ast_features` (no tree-sitter)

### Active file inventory (updated 2026-05-13)

| Category | Files | Notes |
|:--|:--|:--|
| Published baselines | `baseline_01`, `02`, `03` | CodeT5, DeTeCtive, FAID |
| Ours | `exp_n13`, `n14`, `n15`, `n16` | HierTree, NTK, Hier-NTK, CE-base |
| Top-3 Novel | `exp_n06`, `n07`, `n09` | HAP, MGA, ETF-Simplex |
| Rescued theory | `exp20`, `25`, `26`, `27`, `28`, `29` | Fraction protocol, no FIXED_TOTAL |
| Oral-tier single-tree | `exp31–exp37` | GOT, SGE, RAD, HPA, KAC, MMDG, IGG |
| Dual-tree + oral-tier refactors | `exp38–exp55` | See §2.3; 6 of these (40/43/44/47/48/54) refactored 2026-05-13 |
| Reference loader/log style | `exp_n19_detective_lite.py` | Match for new files |
| Legacy (inactive) | `testing_chis/legacy/*.py` | Do not run |

### Anti-patterns
- `FIXED_TOTAL_TRAIN` — replaced by `frac`
- `ModernBERT-base` as encoder — dropped
- `dirs_to_try` fallback in `_load_aicd` — replaced by STRICT loader
- HuggingFace fallback in `_load_aicd` — removed
- Missing `local_files_only=True` on `from_pretrained` — banned
- `torch.autocast(...)` without `dtype=torch.bfloat16` — banned for new files
- `num_workers <= 2` for fraction-protocol training — banned (use 4)
- Per-tag JSON save inside `run_exp` — replaced by ONE combined `{expNN_method}_results.json`
- Python `O(B²)` pairwise loops with `.item()` — vectorise (see TPNL, TKE, BGB)
- `weights = torch.zeros_like(...)` then index-assign grad-tracked product — gradient is **detached**; use `(no_grad_w * grad_x).mean()` instead (see exp42 SSL fix)
- Loss-weight tuning presented as novelty
- Feature stacking without theory grounding
- 12-feature regex AST stub — replaced by 22-feature legacy-aligned extractor

## 16. GPU + AMP (RTX Pro 6000, 96 GB Ada Lovelace SM89)

**Tuned defaults (all 18 files in `testing_chis/exp{38..55}*.py`):**

| Setting | Value | Why |
|:--|:--|:--|
| `bs` | 256 | unixcoder-base (~125M params) at seq=512 uses ~15 GB / step — comfortable on 96 GB |
| `seq` | 512 | matches paper-protocol; `_hw(cfg)` auto-down-scales if VRAM < 40 GB |
| `num_workers` | 4 | matches Kaggle CPU cap; avoids IO bottleneck without oversubscription |
| `pin_memory` | True | required for true zero-copy host→device |
| AMP | `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` | Ada native bf16; no fp16 scaler overflow risk |
| `GradScaler` | constructed but effectively inert under bf16 | left in to keep `scaler.scale/step/update` call sites unchanged |
| `cudnn.benchmark` | True | seq-len fixed → kernel autotuning gives 10-20% speedup |
| `cuda.matmul.allow_tf32` | True | TF32 matmul on Ampere/Ada |
| `cudnn.allow_tf32` | True | TF32 conv on Ampere/Ada |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | reduces fragmentation across the 6 runs / file |

`_hw(cfg)` ladder (auto-selected by VRAM):
- ≥ 40 GB → `bs=256, seq=512`
- ≥ 10 GB → `bs=128, seq=384`
- else   → `bs=64,  seq=256`

## 17. Bug-fix audit log (2026-05-13, commit `1a5d3c5`)

| File | Severity | What was broken | Fix |
|:--|:--|:--|:--|
| `exp42_ssl.py` | HIGH (grad) | `weights = torch.zeros_like(ce); weights[i] = w * ce[i]` — index-assign into a non-grad tensor detached gradient → SSL term was dead | Replace with `(sample_w * ce).mean()` where `sample_w` is no-grad and `ce` carries grad. |
| `exp46_cie.py` | HIGH (dim) | `SourceEncoder(n_sources)` passed `n_sources` into `embed_dim` slot → `adjust_net` input dim mismatched downstream | Refactored to `SourceEncoder(n_sources, embed_dim)`; `use_adjustment=False` path now marginalises via uniform source prior. |
| `exp47_ift.py` | MED (logic) | `estimate_mi(z_ast, source_ids, source_ids)` passed source twice; labels never reached MI estimator | Added `labels` to `forward` signature, plumbed through `train_epoch`. Later refactored to GIBA with real InfoNCE-MI critic + CLUB upper bound. |
| `exp48_dtr.py` | HIGH (dim) | `self.proj = nn.Linear(hidden + ast_dim + n_treatments, …)` but `fused` was `cat([sem_proj(128), ast_emb(64), tau(9)])` | Fixed input dim to `128 + ast_dim + n_treatments`; later refactored to HETE per-class CATE. |
| `exp49_paca.py` | HIGH (NameError) | `torch.randn_like(self.weight_std)` and `self.bias_std` referenced attrs that did not exist | Use local `weight_std = self.weight_log_std.exp()` and `bias_std = self.bias_log_std.exp()`. |
| `exp50_sie.py` | HIGH (crash + dead reg) | `apply_group=True` branch called `GroupInvariantPool((B, D))` instead of `(B, G, D)` → dim crash at epoch ≥ 2; `logits_aug` always `None` → invariance reg dead | Replaced with real Z_2 sign-flip group action: `h_inv = (1/|G|) Σ_g f(g·h)`. |
| `exp52_cpo.py` | HIGH (method dead) | `cpo_loss(..., emb_cf=None, labels_cf=None)` always; counterfactual reg always 0; loss sign wrong | Train loop now permutes `ast_feat` within batch, computes `emb_cf` via second forward, `cpo_loss` is triplet-margin (same-label pull, diff-label push). |
| `exp52_cpo.py` | LOW (install) | `_ensure("sklearn")` is wrong pip name | Changed to `_ensure("scikit-learn")`. |
| All 18 (style) | MED (consistency) | mixed `bs=64` vs `bs=256`; `num_workers=2`; fp16 + GradScaler | Standardised to bs=256, num_workers=4, bf16 autocast; `_hw(cfg)` logs `[hw] mem=…GB bs=… seq=…`. |
| All 18 (logging) | MED (consistency) | Per-tag JSON save inside `run_exp`; no summary table; errors swallowed | Replaced with single combined JSON per file + summary table; `main()` logs `=== {tag} ===`, `[epoch N] val=…`, `[{tag}] MacroF1=… ({Δ:+.4f} vs paper)`. |

