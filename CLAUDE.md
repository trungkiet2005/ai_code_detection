---
description:
alwaysApply: true
---

# CLAUDE.md — Repo Briefing (EMNLP 2026 Main target)

> Read this file first. If it disagrees with the code, trust the code and fix this file.

---

## -1. 12-rule template (global)

Apply to every task unless explicitly overridden. Bias: caution over speed on non-trivial work.

1. **Think Before Coding** — State assumptions. If uncertain, ask. Stop when confused.
2. **Simplicity First** — Minimum code. No features beyond ask. No abstractions for single-use code.
3. **Surgical Changes** — Touch only what you must. Don't refactor what isn't broken.
4. **Goal-Driven Execution** — Define success criteria, loop until verified.
5. **Use the model only for judgment calls** — If code can answer, code answers.
6. **Token budgets are not advisory** — Per-task 4k, per-session 30k. Summarize and restart if breached.
7. **Surface conflicts, don't average them** — Pick one (more recent / more tested). Flag the other.
8. **Read before you write** — Read callers and shared utilities before adding code.
9. **Tests verify intent, not just behavior** — Encode WHY behavior matters.
10. **Checkpoint after every significant step** — Summarize done / verified / left.
11. **Match the codebase's conventions** — Conformance > taste. If harmful, surface; don't fork silently.
12. **Fail loud** — "Completed" is wrong if anything was skipped silently.

**Project-specific addendum:** Every new `exp*.py` MUST open with `# expNN — Method` comment.

---

## 0. Submission target

- **Venue:** EMNLP 2026 Main (long, 8 pages). Aim: Oral.
- **Deadline:** ~2026-05-26.
- **Strategy:** Single-method hero in §3, family-diverse baselines in §4, ablation in §5. Hero NOT locked — chosen after ablation + diversity round.

### 0.1 Frozen operational protocol (NOT design space)

| Setting | Value |
|:--|:--|
| Encoder | `unixcoder-base` only, `local_files_only=True` |
| Benchmarks | `codet_m4` (Macro-F1, 6-class), `aicd_t2` (Macro-F1, 12-class) |
| Fractions | `[0.01, 0.05, 0.20]` per-class |
| AMP | `torch.autocast(device_type='cuda', dtype=torch.bfloat16)` |
| HW knobs | `bs=256, seq=512, num_workers=4, pin_memory=True`; `_hw(cfg)` auto-downscales |
| AST features | legacy-aligned 22-feature `extract_ast_features` |
| Schedule | regime-adaptive (1%→10ep, 5%→6ep, 20%→6ep); LR sqrt-scaled; cosine + warmup |
| Reporting | always `val_macro`, `test_macro`, `val_test_gap` |
| Output | one combined `{expNN_method}_results.json` with full `eval_pack` |

**`_load_aicd("t2")` is STRICT:** raises `FileNotFoundError` if T2 dir missing. No HF fallback.

⚠️ **DATA BUG (exp1–exp17):** old `_load_aicd` returned T1 binary instead of T2. Quote AICD results only from exp18+.

---

## 1. Research question

How should a few-shot attribution model use the known genealogy of LLM generators instead of treating every author label as unrelated?

**Thesis (open form):** The flat-simplex baseline is wrong on a label space with metric structure. Genealogy-aware objects (kernels, residuals, contrastive variants, temperatures, factorizations, alignments) define new mathematical objects that have no analogue under unstructured labels. The headline object is decided by ablation, NOT pre-locked.

---

## 2. Novelty bar — problem-specific only

Every new exp MUST introduce a NEW MATHEMATICAL OBJECT that is **undefined for flat-label classification** and exploits ≥ 1 of S1–S10 structural facts. If a reviewer can say "this is X applied to Y", REJECT.

### 2.1 Structural facts of AI-code attribution (the irreducible tokens)

| # | Fact | Enables |
|:-:|:--|:--|
| S1 | Two trees co-exist: AST (syntax) + genealogy (model family) | Dual-tree kernels, joint geometry |
| S2 | Decoding fingerprints (T, top-p, rep-penalty) leak into output | Decoding-artifact heads |
| S3 | Pretraining-corpus provenance (Stack/BigCode/GitHub) leaks | Provenance-trace heads |
| S4 | Prompts shared across generators in benchmarks | Prompt-invariant attribution |
| S5 | Authors have characteristic AST motifs | AST subtree mining, motif banks |
| S6 | Detector can re-encode under each candidate LM → likelihood vector | Multi-LM perplexity signature |
| S7 | Same prompt × different T from same model → different outputs; identity invariant | Temperature-augmented training |
| S8 | Source domain (cf/lc/gh) observed and confounding | Source-stratified backdoor adjustment |
| S9 | Few-shot is the operating regime (new generator added monthly) | Prototype-generator meta, tournament |
| S10 | Human is a class with internal cluster substructure | Human-cluster prototypes |

**Acceptance test:** the proposed equation contains ≥ 1 of `{AST_i, prompt_embed, log p_LM, source domain, decoding param, d_tree, family-anchor}`.

### 2.2 Anti-patterns (REJECT)

- "Apply SupCon/ArcFace/SimCSE/R-Drop/LoRA to code" — generic.
- Loss-weight tuning presented as novelty.
- Feature stacking without theory grounding.
- Engineering wins (schedule, LR, bs, AMP) presented as method.

---

## 3. Method-diversity matrix — discovery tool (NOT contribution)

Diversity finds the hero and populates §4 baselines. Paper is single-method (§3 spotlight goes to ONE method only).

| Family | Active in repo |
|:--|:--|
| Sibling-weighted CE (F1) | SSL, TKL, SCR — saturated |
| Soft labels (F2) | GSCE |
| Kernel/Gram alignment (F3) | HTKA, DGK |
| Contrastive InfoNCE (F4) | TPNL, GIBA, BGB |
| Output temperature (F5) | GFTS |
| Residual/disentangle (F6) | GRA, RASL, GFR |
| Curriculum / phase-transition (F7) | PTR, KAC |
| Causal / backdoor (F8) | CFT, CIE, HETE, RCE |
| Optimal transport (F9) | GOT, MMDG |
| Spectral (F10) | SGE, DGK |
| Hyperbolic (F11) | HPA |
| Bayesian (F12) | PAC-A, BUA |
| Retrieval (F13) | TBD exp67_tourn-adjacent |
| MoE / Hypernet / KAN / SSM (F14–F17) | gap — legacy-only |
| Stylometry fusion | exp68_stylo |
| Multi-LM likelihood | exp69_perpsig |
| Decoding-fingerprint | exp70_decofp |

---

## 4. Current leaderboard (best per slot across all rounds, Macro-F1)

### CoDET-M4 Author

| Frac | Best | Score | Δ vs paper 0.6633 | Runner-up |
|:-:|:--|:-:|:-:|:--|
| 1% | **TRACO** (exp76) | **0.5887** | — | RACO 0.5627 |
| 5% | **TRACO** (exp76) | **0.6622** | — | TKL 0.6549 |
| 20% | TKL / TOURN | 0.7198 | +0.0565 | TRACO 0.7186, SSL-RAS 0.7186 |

### AICD-T2 Model-Family

| Frac | Best | Score | Runner-up |
|:-:|:--|:-:|:--|
| 1% | TKL (exp63) | 0.3038 | TOURN 0.2972 |
| 5% | DECOFP (exp70) | 0.4108 | RACL 0.4030 |
| 20% | **DTKE** (exp66) | **0.4882** | CE-only 0.4881 ≈ tied |

### Critical findings

1. **TRACO (exp76) breaks the saturation band at 1% and 5% CoDET-M4** — first method to do so. Falsifier `view_cos = 0.97` confirms S7 invariance empirically achieved.

2. **CE-only @20% matches band ceiling on both benchmarks** (ablation exp65_abl): CoDET-M4 0.7118 / AICD-T2 0.4881. `DTKE − CE-only` on AICD = +0.0001 → saturation is encoder-level, not method-level. Adding structure-aware loss terms doesn't help at 20% data.

3. **TKL learned `sibling_weight @20% → 0`** — structure prior decays with n. Confirmed by exp65 ablation.

4. **RACO's `λ` GROWS with n** (0.64→0.82→0.97) — opposite direction from TKL. Loss-level vs rep-level structure priors are regime-dependent in OPPOSITE ways.

5. **GENEPRINT exp71 falsifier FAILED**: 3-channel orthogonal but zero-out drops < 0.025 → classifier ignores channels. Negative finding: author identity is NOT factorisable along S-fact channels.

6. **TIEH exp72 hyperbolic FAILED**: low-n collapse (AICD 1% = 0.17). Hyperbolic geometry too unstable for sparse data.

7. **TAPA exp73 collapse**: prototype Spearman anti-correlated with d_tree (-0.47 to -0.68). EMA prototypes conflict with tree-iso constraint.

---

## 5. Active slate (NEW round, sequential IDs)

| ID | Method | New object | S-fact | Reference (cite/baseline) |
|:--|:--|:--|:--|:--|
| `exp64_raco` | **RACO** Regime-Adaptive Contrastive: `L = λ(n)·SupCon(d_tree-weighted) + (1−λ(n))·CE`, λ learned | S9 + structure | Differs from LASCL (arXiv:2402.00232) by phase-transition gate |
| `exp65_abl` | Ablation: CE / SSL / HTKA / GSCE / SCR component decomposition | — | diagnostic |
| `exp66_dtke` | **DTKE** Dual-Tree Kernel: `K(x,y) = K_ast(AST_x, AST_y) · K_gene(y, y')` | S1 | undefined w/o both trees |
| `exp67_tourn` | **TOURN** Tournament few-shot, tree-aware bracket | S9 | arXiv:2501.08165 |
| `exp68_stylo` | **STYLO** Stylometry feature fusion (30+ features) ⊕ unixcoder | S5 | arXiv:2506.17323 |
| `exp69_perpsig` | **PERPSIG** Multi-LM likelihood signature classifier | S6 | new |
| `exp70_decofp` | **DECOFP** Decoding-fingerprint regressor + conditional attribution | S2 | new |
| `exp71_geneprint` | **GENEPRINT** ⭐ HERO: tri-channel disentangled `z=[z_T;z_D;z_M]` w/ HSIC orthogonality; channel-specific losses (topology / decoding / motif) | S1+S2+S5 | synthesises all Round-2 insights |
| `exp72_tieh` | **TIEH** Tree-Isometric Hyperbolic Embedding in Poincare ball; learnable prototypes constrained `d_H(p_i,p_j) ≈ d_tree(i,j)` | S1 (geometric) | extends Nickel & Kiela 2017 |
| `exp73_tapa` | **TAPA** Tree-Anchored Prototypical Attribution: EMA prototypes + multi-layer pooling + tree-iso constraint | S1+S9 | Snell 2017 + LIGHT arXiv:2503.00958 |
| `exp74_setfit_tw` | **SETFIT-TW** SetFit two-stage: SupCon w/ tree-weighted negatives → frozen-encoder linear head | S1+S9 | SetFit arXiv:2209.11055 |
| `exp75_racl` | **RACL** Retrieval-Augmented Code Logit: learned mix `β · param + (1−β) · kNN`, kNN with tree-aware weighting | S1+S9 | RAFC arXiv:2406.11148 + kNN-LM |
| `exp76_traco` | **TRACO** Token-level Robust Augmentation Contrastive: 2-view encoder enforces S7 invariance under {token_dropout, id_rename, ws_jitter, comment_strip} | S7+S1 | new (SimCLR/MoCo paradigm, S7-grounded for code) |
| `exp77_cronos` | **CRONOS** Single-encoder + 3 S-fact aux heads (tree-dist regressor + decoding-fp regressor + sibling-pair classifier) | S1+S2+S9 | multi-task aux co-training, fixes GENEPRINT failure |
| `exp78_cascade` | **CASCADE** Hierarchical family-then-sibling decoding: `p(y=k) = p(family) · p(sibling\|family)` | S1 | extends hierarchical softmax to LLM-genealogy attribution |
| `exp79_mage` | **MAGE** Genealogy-conditioned mixup: pair sampling ∝ exp(-γ·d_tree) → sibling-heavy embedding-mixup | S1 | extends mixup with label-tree-conditioned sampler |
| `exp80_tracod` | **TRACOD** TRACO + EMA teacher self-distillation (DINO-style centered targets) under code augmentation views | S7+stability | extends DINO arXiv:2104.14294 with supervised CE and TRACO augmentations |

**Killed (chưa run, generic ML):** `exp64_dgk` (kernel-alignment trùng HTKA), `exp73_srcbdoor` (trùng CIE/CFT/HETE), `exp75_humclust` (chỉ K-means trên human). **GENEPRINT exp71 / TIEH exp72: falsifier failed (run completed), kept as analysis evidence in §5.**

**Hero-locking gate:** chỉ chốt khi user nói **"chốt"**. Trước đó: mọi method = candidate.

---

## 6. Repo layout

```
Exp_FewShot/testing_chis/      ← ACTIVE
  tracker.md                   ← primary leaderboard
  baseline_0[1-3].py           ← CodeT5, DeTeCtive, FAID
  exp_n{06,07,09,13..19}.py    ← novel + baselines
  exp{20,25-29}.py             ← rescued theory
  exp{31..37}.py               ← oral-tier single-tree
  exp{38..55}.py               ← dual-tree refactors
  exp{56..63}.py               ← RAS-schedule round (results logged)
  exp{64..70}.py               ← new slate (pending)
  legacy/                      ← read-only
Exp_Climb/, Exp_CodeDet/, Exp_DM/   ← FROZEN
Paper/latex/main.tex                ← draft
legacy/                             ← archived
```

---

## 7. How to add a new experiment

1. Write the 6-line theory block: NAME / ARXIV_ID / ONE-LINE CLAIM / EQUATION / WHY NOT BEFORE / FALSIFIER.
2. Verify the new object exploits ≥ 1 S-fact and is undefined for flat labels.
3. Copy closest active exp (e.g. `exp42_ssl.py`, `exp60_htka.py`, `exp63_tkl.py`) as skeleton.
4. **First line of file:** `# expNN — Method name` comment.
5. Use `unixcoder-base`, `local_files_only=True`, fraction protocol, AMP bf16, `_hw(cfg)`.
6. Output ONE combined `{expNN_method}_results.json` with full `eval_pack`.
7. Register in `tracker.md` after run with val + test + Δ + val_test_gap.

---

## 8. Decision rules (DO NOT do unprompted)

- Don't reuse `expNN` IDs.
- Don't add HF fallback or `dirs_to_try` to `_load_aicd`.
- Don't drop `local_files_only=True`.
- Don't revert AMP to fp16+GradScaler on RTX Pro 6000.
- Don't quote AICD numbers from exp1–17 (T1-binary bug).
- Don't add experiments to `Exp_Climb/`, `Exp_CodeDet/`, `Exp_DM/` (frozen).
- Don't declare "hero method" until user says **"chốt"**.
- Don't propose generic ML — every method must cite ≥ 1 S-fact.
- Always report val_test_gap.

---

## 9. References

- Paper draft: [Paper/latex/main.tex](Paper/latex/main.tex)
- Section outline: [Paper/outline.md](Paper/outline.md)
- Benchmark refs: `docs/references/paper_{AICD,Droid,CodeDet_M4}.md`
- Active tracker: [Exp_FewShot/testing_chis/tracker.md](Exp_FewShot/testing_chis/tracker.md)
- Key external (lit-search 2026-05-17):
  - [LASCL — Label Hierarchy SupCon (arXiv:2402.00232)](https://arxiv.org/abs/2402.00232)
  - [HCAL — Hierarchy-Consistent Adaptive Loss (arXiv:2508.13452)](https://arxiv.org/html/2508.13452v1)
  - [Disentangled Authorship (arXiv:2604.21300)](https://arxiv.org/abs/2604.21300)
  - [LLM-AuthorBench Stylometry (arXiv:2506.17323)](https://arxiv.org/abs/2506.17323)
  - [Tournament Code Attribution (arXiv:2501.08165)](https://arxiv.org/html/2501.08165v1)
