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

### 0.2 The contribution slate (open, not locked)

We are aiming at an **oral-tier paper** with a **portfolio of genealogy-aware
contributions**, evidenced by:

- **A theoretical claim** — genealogy structure on the LABEL space defines new
  mathematical objects (target kernels, hierarchical losses, learnable distance
  weights, heat diffusion on label graphs) that are undefined for flat-simplex
  classification. Any of these can carry the headline if it wins empirically
  AND survives ablation.
- **An empirical claim** — those objects beat the published full-data encoder
  baselines (UniXcoder 0.6633 on CoDET-M4) with a small fraction of training
  data. **Current 20% SOTA: 0.7186 (+0.0553 vs paper)**, achieved by multiple
  methods (SSL-RAS, GSCE, RASL, GFTS) that converge to the same band — see
  tracker §0 "schedule was the dominant bottleneck".
- **A regime claim** — performance is dominated by encoder pretraining at small
  n and by the genealogy prior at large n; PTR (`exp61`) tests this directly.
- **A falsifier line** — every new exp must commit to a falsifier metric we
  log automatically (`sibling_confusion_rate`, learned hyperparam, etc.) so a
  reviewer can verify the mechanism, not just the bottom-line number.
- **A diagnostic ablation** — `exp65_abl` toggles components (CE / SSL / HTKA
  / GSCE / SCR / combinations) on the same backbone+schedule to isolate which
  loss object owns which fraction of the gain.

> The headline name is NOT pre-decided. It will be whichever method dominates
> the ablation while passing its falsifier.

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

### 2.1 Headline object — kept as a candidate, not a commitment

The repository was originally framed around HTKA. After RAS-schedule rollout
and the exp56-59 results, the empirical SOTA is shared across SSL-RAS, GSCE,
RASL, GFTS within 0.001 Macro-F1 on CoDET-M4 20%. We DO NOT commit to a
single headline; the headline is decided by `exp65_abl` and the falsifier
metrics, not by the order in which we implemented the methods.

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

