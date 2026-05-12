---
description: 
alwaysApply: true
---

# CLAUDE.md — Repo Briefing (EMNLP 2026 Main target)

> **Read this file first.** One-stop theoretical + operational context.
> If anything here disagrees with the code, trust the code and fix this file.

---

## 0. Current submission target — UPDATED 2026-05-12

> **Venue:** EMNLP 2026 Main (long paper, 8 pages). **Reach for Oral.**
> **Deadline:** ~2026-05-26.
> **Status (2026-05-12):** Fraction-based few-shot protocol locked; `unixcoder-base` only; active files cleaned up; `_load_aicd` is now STRICT (no fallback).

**Few-Shot Protocol (FINAL as of 2026-05-12):**
- **Fractions:** `0.01` (1%), `0.05` (5%), `0.20` (20%) of the training split per class
- **Encoder:** `unixcoder-base` ONLY (ModernBERT dropped — unixcoder consistently best or equal, reduces compute)
- **Benchmarks:** `codet_m4` (headline), `aicd_t2` (stress test — 12-class model-family attribution)
- **Droid T3/T4:** SKIPPED (per 2026-05-10 directive)
- **Batch:** 256, seq=512
- **Total per experiment file:** 6 runs (1 encoder × 2 benchmarks × 3 fractions)

> ⚠️ **DATA BUG (exp1–exp17 / baseline_01–04 / exp_n05–n16 original runs):**
> Due to a `_load_aicd()` path bug, all AICD results from those experiments were
> **trained on T1 (binary 2-class)** instead of T2 (12-class model-family).
> **AICD "results" from those runs are INVALID and must not be quoted.**
> Only CoDET-M4 results from those runs are reliable.
> **Fix applied 2026-05-12:** `_load_aicd()` is now STRICT — raises `FileNotFoundError`
> immediately if target task dir is missing; NO fallback to other tasks or HuggingFace.

**Single sentence:**

> We are not just improving an AI-code detector. We are using CoDET-M4 authorship and AICD model-family attribution as public testbeds for one theory contribution: **model genealogy can be written as a target kernel, and aligning few-shot representations to that kernel gives a scale-aware inductive bias for LLM attribution.**

The benchmark numbers are evidence. The contribution is the attribution principle.

**4 contributions (updated):**
- **C1 (theory headline)** — **Hierarchical Target-Kernel Alignment (HTKA / Hier-NTK)**: define a genealogy-aware target kernel over attribution labels and align the encoder representation Gram matrix to it.
- **C2 (empirical headline)** — With **5% of training data** on unixcoder-base, Hier-NTK reaches **0.6709 CoDET-M4 6-class Author Macro-F1**, exceeding paper UniXcoder's full-data 0.6633 by **+0.0076 with 1/20 the budget**.
- **C3 (regime law)** — Below ~5K samples, encoder pretraining dominates; above ~5K samples, the genealogy target-kernel prior dominates. This phase transition is a claim to explain, not just a table entry.
- **C4 (scope validation)** — AICD T2 model-family attribution is the secondary authorship-like stress test.

**Locked headline numbers (CoDET-M4 6-class Author IID Macro-F1, unixcoder-base):**

| Bench / Setup | Method | Score | Δ vs paper UniXcoder full (0.6633) |
|:--|:--|:-:|:-:|
| 5% training data | **🥇 FS-Hier-NTK** | **0.6709** | **+0.0076** |
| 5% training data | 🥈 FS-HierTree | 0.6682 | +0.0049 |
| 5% training data | 🥉 FS-NTKAlign | 0.6652 | +0.0019 |
| 5% training data | 4 FS-ETF-Simplex | 0.6616 | −0.0017 |
| 5% training data | 5 FS-CE-Baseline | 0.5725 | −0.0908 |
| 1% training data | 🥇 FS-Baseline (unixcoder) | 0.5744 | (encoder-pretrain regime) |

---

## 0bis. Original NeurIPS framing (kept for context)

The repository was originally built targeting NeurIPS 2026 Oral. We reframed to EMNLP Main when:
- 17 days to NeurIPS-like deadline made oral-tier breakthrough (OOD-gh > 0.40) unrealistic.
- Existing 20%-data SOTA (+4.70 / +5.20 / +0.98) is already a publishable EMNLP claim.
- Source confounding becomes §5 Analysis instead of §3 Method.

---

## 1. The central research question

> **How should a few-shot attribution model use the known genealogy of LLM generators instead of treating every author label as unrelated?**

## 2. Thesis (one sentence)

> **Few-shot AI-code attribution fails when it learns a flat simplex over generator labels; it improves when representation geometry is aligned to a genealogy-derived target kernel over authors/model families.**

### 2.1 The new object

For a batch of examples with labels $y_i$, define a target genealogy kernel:

```
T_ij = k_tree(y_i, y_j)
```

where `k_tree` is high for the same author, intermediate for related model families, and low for unrelated human/model branches. Given normalized embeddings `z_i`, the method aligns the representation Gram matrix to this target:

```
L_htka = 1 - cos(vec(ZZ^T), vec(T))
```

### 2.2 Theory-grounded novelty bar — NOVELTY-FIRST PRINCIPLE

> 🚨 **EMNLP Oral demands NEW mathematical objects, not generic ML techniques applied to a new domain.**
> Every experiment must introduce something that **ONLY makes sense** in the context of
> AI-code attribution with genealogy structure. If a reviewer can say "this is just X applied to Y",
> the experiment is NOT novel enough.

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

**🔬 Dual-Tree Theory Experiments (NEW — AST + Genealogy intersection):**

> **Motivation:** These experiments define NEW mathematical objects that only make sense when combining AST structure (how code is written) with genealogy structure (who wrote it).

| File | Method | New Object | Theory Basis |
|:--|:--|:--|:--|
| `exp38_gaka.py` | GAKA | Cross-kernel alignment | Alignment between AST kernel and genealogy kernel |
| `exp39_sto.py` | STO | Structural Transfer Operator | Transfer AST patterns between generators |
| `exp40_hcdt.py` | HCDT | Dual-tree contrastive | Positive pairs as INTERSECTION of two trees |
| `exp41_cfi.py` | CFI | Family invariance | Invariance defined by genealogy tree |
| `exp42_ssl.py` | SSL | Weighted CE | Error weights from genealogy distance |
| `exp43_cta.py` | CTA | Genealogical attention | Attention modulated by genealogy |
| `exp44_sgd.py` | SGD | Unified distance | d_sgd combines AST + Genealogy trees |
| `exp45_gra.py` | GRA | Genealogical residual | R(x) = AST(x) - E[AST|gene(x)] |

> **Why genuinely novel:** Each requires BOTH AST structure AND genealogy structure.

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

**Protocol (non-negotiable, as of 2026-05-12):**
- Encoder: `unixcoder-base` only
- Fractions: `[0.01, 0.05, 0.20]`
- Benchmarks: `codet_m4`, `aicd_t2`
- **`_load_aicd()` is STRICT:** `FileNotFoundError` if target task dir missing. NO fallback to other tasks. NO HuggingFace fallback.
- Train separate model per benchmark. Never mix.
- Report the full metric pack: Primary, Macro-F1, Weighted-F1, per-class.
- **Final evaluation is always on TEST set** (val set used only for checkpoint selection).

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

## 7. Repo layout (post-cleanup 2026-05-12)

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
│       └── exp29_causal_trace.py       # CFT (Causal Feature Tracing)
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
2. Copy closest active `exp_nXX_*.py`. Change only the method-specific loss.
3. Use fraction-based `build_dls(cfg)` with `cfg.frac`. Do NOT use `FIXED_TOTAL_TRAIN`.
4. Use `unixcoder-base` as the encoder. Do NOT add ModernBERT.
5. Register in tracker.md after run.

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

## 13. Cold-start checklist (2026-05-12)

1. Read §0 (protocol, data bug warning) and §8 (locked numbers).
2. Open [Exp_FewShot/testing_chis/tracker.md](Exp_FewShot/testing_chis/tracker.md) — current leaderboard.
3. Verify `_load_aicd("t2")` in any file you touch — confirm it is STRICT (no fallback, raises FileNotFoundError if T2 dir missing).
4. Active files are in §7 layout. Only edit files listed there.
5. If asked about AICD results: **only results from exp18+ (with correct T2 loader) are valid**. Earlier runs were T1 binary.
6. `Exp_Climb/`, `Exp_CodeDet/`, `Exp_DM/` are FROZEN. Do not add new experiments there.

## 14. Decision rules (what NOT to do unprompted)

- **Never** add a new method to `Exp_Climb/` / `Exp_CodeDet/` / `Exp_DM/` without user approval.
- **Never** re-open Zero-Shot work — archived for −32pt reproduction gap.
- **Never** add `dirs_to_try` fallback or HuggingFace fallback to `_load_aicd()`.
- **Never** quote AICD numbers from exp1–17 as T2 results — they are T1 (binary).
- **Always** report val-test gap alongside test metrics.
- **Always** use fraction sampling, not fixed sample count.

## 15. Exp_FewShot/testing_chis rules (active suite, 2026-05-12)

### Protocol
- Encoder: `unixcoder-base` only
- Fractions: `[0.01, 0.05, 0.20]`
- Benchmarks: `codet_m4`, `aicd_t2` (T2 = 12-class model-family)
- Sampling: per-class fraction sampling (not fixed total)
- `_load_aicd` STRICT — no fallback

### Active file inventory

| Category | Files | Notes |
|:--|:--|:--|
| Published baselines | `baseline_01`, `02`, `03` | CodeT5, DeTeCtive, FAID |
| Ours | `exp_n13`, `n14`, `n15`, `n16` | HierTree, NTK, Hier-NTK, CE-base |
| Top-3 Novel | `exp_n06`, `n07`, `n09` | HAP, MGA, ETF-Simplex |
| Rescued theory | `exp20`, `25`, `26`, `27`, `28`, `29` | Fraction protocol, no FIXED_TOTAL |
| Legacy (inactive) | `testing_chis/legacy/*.py` | Do not run |

### Anti-patterns
- `FIXED_TOTAL_TRAIN` — replaced by `frac`
- `ModernBERT-base` as encoder — dropped
- `dirs_to_try` fallback in `_load_aicd` — replaced by STRICT loader
- HuggingFace fallback in `_load_aicd` — removed
- Loss-weight tuning presented as novelty
- Feature stacking without theory grounding

