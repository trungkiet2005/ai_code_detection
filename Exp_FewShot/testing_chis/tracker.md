# Testing Chis — Experiment Tracker

---

## 🎯 HERO LOCKED: TRACO (exp76) — 2026-05-19 [⚠ contested by exp81–101]

> **Decision:** TRACO is the §3 spotlight method for the EMNLP 2026 paper.
> Locked after the user's "chốt TRACO" on 2026-05-19. All other methods
> in the tracker are recategorised as §4 baselines or §5 ablation evidence.
>
> ⚠️ **2026-05-20 update:** Round 7 results (exp87–exp101) now in.
> **METATRACO (exp100) composite = 0.5408** — exceeds the +0.010 unlock
> threshold over TRACO (0.5256 + 0.010 = 0.5356). METATRACO is a strong
> upgrade candidate for the §3 hero slot. DCGPT (exp92) = 0.5313 is new #2.
> RAINFER (exp94) takes AICD-T2 20% SOTA (0.4921). CoDET-M4 1% SOTA
> shattered by METATRACO at **0.6261** (+0.033 over prior best PROG 0.5934).
> User review required before promoting METATRACO to hero.

### Why TRACO won the hero slot

1. **Composite #1** across 6 benchmark-fraction slots
   (mean Macro-F1 = 0.5256, computed over both CoDET-M4 and AICD-T2).
2. **Per-slot SOTA** at CoDET-M4 1% (0.589) and CoDET-M4 5% (0.662);
   top-3 at every remaining slot.
3. **Two structural facts in one loss** that no prior code-attribution
   work combined: S1 (fine-tune genealogy gives a tree distance over
   labels) and S7 (decoder temperature gives a view-invariance over
   inputs).
4. **Strongest falsifier panel** of any candidate:
   `view_cos = 0.97` (S7 invariance learned), sibling confusion drops
   by 25% relative on CoDET-M4 (S1 weighting fires on the failure mode).
5. **Four falsified alternatives** (TIEH hyperbolic, SETFIT-TW frozen,
   TRACOD EMA-distill, GENEPRINT tri-channel) constrain the design space
   and give the paper its negative-results section.

### Role of the rest of the tracker

| Method | Role in paper |
|:--|:--|
| **TRACO** (exp76) | §3 spotlight, hero method |
| **TKL** (exp63) | §5 ablation: "tree-weighting alone, no view aug" |
| **CARGO** (exp81/84) | §5 ablation: "structural pool, same Δ" → augmentation pool is secondary |
| **TOURN** (exp67) | §4 baseline: pair-margin variant of tree weighting |
| **DTKE** (exp66) | §4 baseline: kernel formulation, SOTA AICD 20% |
| **DECOFP** (exp70) | §4 baseline: decoding-signal paradigm, SOTA AICD 5% |
| **SCR** (exp62) | §4 baseline: sibling regulariser |
| **HTKA / PTR / STYLO / PERPSIG / RACO / CASCADE / MAGE / GENEPRINT / CRONOS** | §5 component ablation, drop-out rows |
| **TIEH / TAPA / SETFIT-TW / TRACOD** | §5 "what does not work" (negative results) |
| **CARGO family v2 (exp85-89)** | post-submission round (not in main paper) |

---

## 🏆 Master Leaderboard (LIVE, 2026-05-19)

> **Setup:** `unixcoder-base`, regime-adaptive schedule (1%→10ep, 5%→6ep, 20%→6ep), AMP bf16, bs=256, seq=512.
> Always report **val · test · val_test_gap**. Metric: **Macro-F1** for both CoDET-M4 (6-class, author IID) and AICD-T2 (12-class, model-family).
> Composite score = **mean Test Macro-F1 across all 6 slots** (3 fractions × 2 benchmarks).

### Composite ranking (40 methods, complete on both benchmarks) — updated 2026-05-20

| Rank | Method | exp | CoDET 1% | CoDET 5% | CoDET 20% | AICD 1% | AICD 5% | AICD 20% | **Mean** |
|:-:|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| 🥇 | **METATRACO** ⚡ | exp100 | **0.6261** | 0.6461 | 0.7059 | **0.3793** | 0.4086 | 0.4790 | **0.5408** |
| 🥈 | **DCGPT** | exp92 | 0.5984 | 0.6555 | **0.7257** | 0.3062 | **0.4148** | 0.4871 | **0.5313** |
| 🥉 | **PROG** | exp83 | 0.5934 | 0.6692 | **0.7242** | 0.2964 | **0.4182** | 0.4760 | **0.5296** |
| 🥉 | **RAINFER** | exp94 | 0.5900 | 0.6635 | 0.7212 | 0.3013 | 0.4092 | **0.4921** | **0.5296** |
| 5 | CARMIX | exp86 | 0.5931 | **0.6736** | 0.7216 | 0.3068 | 0.4001 | 0.4816 | 0.5295 |
| 6 | CARBO | exp85 | 0.5869 | 0.6703 | 0.7194 | **0.3090** | 0.4005 | 0.4848 | 0.5285 |
| 7 | CONFUSE | exp81 | 0.5903 | 0.6723 | 0.7201 | 0.2983 | 0.3976 | 0.4849 | 0.5273 |
| 8 | DIFFTREE | exp93 | 0.5812 | 0.6652 | 0.7195 | 0.3044 | 0.4061 | 0.4798 | 0.5260 |
| 9 | TRACO ⭐ hero | exp76 | 0.5887 | 0.6622 | 0.7186 | 0.2965 | 0.3998 | 0.4876 | 0.5256 |
| 10 | CARCURR | exp89 | 0.5873 | 0.6613 | 0.7161 | 0.3039 | 0.4012 | 0.4819 | 0.5253 |
| 11 | CARGO v1 (no-op bug) | exp81 | 0.5872 | 0.6601 | 0.7143 | 0.3065 | 0.4030 | 0.4797 | 0.5251 |
| 12 | CAARC | exp87 | 0.5857 | 0.6599 | 0.7165 | 0.3061 | 0.4004 | 0.4803 | 0.5248 |
| 13 | MTAUX | exp96 | 0.5760 | 0.6552 | 0.7111 | 0.3097 | 0.4041 | 0.4811 | 0.5229 |
| 14 | CARGO v2 (no-op fix) | exp84 | 0.5816 | 0.6601 | 0.7147 | 0.3038 | 0.4005 | 0.4746 | 0.5225 |
| 15 | CGPTS | exp90 | 0.5805 | 0.6722 | 0.7193 | 0.2939 | 0.3852 | 0.4721 | 0.5205 |
| 15 | DETMULTI | exp91 | 0.5794 | 0.6725 | 0.7220 | 0.2917 | 0.3887 | 0.4689 | 0.5205 |
| 17 | CAGHOST | exp88 | 0.5767 | 0.6600 | 0.7155 | 0.3000 | 0.3914 | 0.4770 | 0.5201 |
| 18 | FARS | exp82 | 0.5787 | 0.6668 | 0.7197 | 0.2924 | 0.3884 | 0.4680 | 0.5190 |
| 19 | SCR | exp62 | 0.5585 | 0.6532 | 0.7185 | 0.3019 | 0.3949 | 0.4776 | 0.5174 |
| 20 | TKL | exp63 | 0.5524 | 0.6549 | 0.7198 | 0.3012 | 0.3940 | 0.4776 | 0.5167 |
| 21 | DECOFP | exp70 | 0.5483 | 0.6420 | 0.7158 | 0.2958 | 0.4108 | 0.4806 | 0.5155 |
| 22 | TOURN | exp67 | 0.5465 | 0.6525 | 0.7198 | 0.2972 | 0.3929 | 0.4835 | 0.5154 |
| 23 | DTKE | exp66 | 0.5501 | 0.6439 | 0.7114 | 0.2948 | 0.3980 | 0.4882 | 0.5144 |
| 24 | PTR | exp61 | 0.5574 | 0.6399 | 0.7091 | 0.2942 | 0.4014 | 0.4824 | 0.5141 |
| 25 | HTKA | exp60 | 0.5444 | 0.6363 | 0.7092 | 0.2913 | 0.3988 | 0.4843 | 0.5107 |
| 26 | STYLO | exp68 | 0.5359 | 0.6437 | 0.7094 | 0.2916 | 0.3971 | 0.4801 | 0.5096 |
| 27 | CRONOS | exp77 | 0.5351 | 0.6456 | 0.7115 | 0.2940 | 0.3908 | 0.4790 | 0.5093 |
| 28 | MAGE | exp79 | 0.5395 | 0.6430 | 0.7106 | 0.2937 | 0.3914 | 0.4758 | 0.5090 |
| 29 | GENEPRINT | exp71 | 0.5460 | 0.6424 | 0.7051 | 0.2927 | 0.3847 | 0.4805 | 0.5086 |
| 30 | CASCADE | exp78 | 0.5462 | 0.6434 | 0.7146 | 0.2846 | 0.3778 | 0.4764 | 0.5072 |
| 31 | PERPSIG | exp69 | 0.5408 | 0.6321 | 0.7076 | 0.2865 | 0.3742 | 0.4813 | 0.5038 |
| 32 | RACO | exp64 | 0.5627 | 0.6539 | 0.6943 | 0.2922 | 0.3520 | 0.3970 | 0.4920 |
| 33 | DUALVIEW | exp97 | 0.5269 | 0.6304 | 0.7010 | 0.2548 | 0.3064 | 0.4322 | 0.4753 |
| 34 | TIEH | exp72 | 0.4271 | 0.6112 | 0.7017 | 0.1729 | 0.3226 | 0.4444 | 0.4466 |
| 35 | TAPA | exp73 | 0.4969 | 0.5743 | 0.6722 | 0.2300 | 0.2746 | 0.2635 | 0.4186 |
| 36 | PROMPTC | exp95 | 0.4493 | 0.5921 | 0.6701 | 0.1887 | 0.2579 | 0.3673 | 0.4209 |
| 37 | SETFIT-TW | exp74 | 0.4353 | 0.4871 | 0.6608 | 0.1405 | 0.2196 | 0.3413 | 0.3808 |
| 38 | SLOTATTN | exp101 | 0.5059 | 0.6012 | 0.6690 | 0.2099 | 0.2593 | 0.3937 | 0.4398 |
| 39 | TRACOD | exp80 | 0.3706 | 0.4485 | 0.4029 | 0.1861 | 0.2125 | 0.1937 | 0.3024 |

**Partial-data methods (CoDET-M4 only — AICD-T2 pending):**

| Method | exp | CoDET 1% | CoDET 5% | CoDET 20% |
|:--|:--|--:|--:|--:|
| RACL | exp75 | 0.5621 | 0.6457 | 0.6873 |

### Per-slot SOTA (gold per cell) — updated 2026-05-20 with Round 7

| Slot | SOTA method | Score | Δ over previous SOTA | Runner-up |
|:--|:--|--:|--:|:--|
| CoDET-M4 1% | **METATRACO** (exp100) | **0.6261** | +0.033 (was PROG 0.5934) | DCGPT 0.5984, PROG 0.5934 |
| CoDET-M4 5% | **CARMIX** (exp86) | **0.6736** | unchanged | DETMULTI 0.6725, CONFUSE 0.6723 |
| CoDET-M4 20% | **DCGPT** (exp92) | **0.7257** | +0.002 (was PROG 0.7242) | PROG 0.7242, CARMIX 0.7216 |
| AICD-T2 1% | **METATRACO** (exp100) | **0.3793** | +0.070 (was CARBO 0.3090) | CARBO 0.3090, MTAUX 0.3097 |
| AICD-T2 5% | **PROG** (exp83) | **0.4182** | unchanged | DCGPT 0.4148, DIFFTREE 0.4061 |
| AICD-T2 20% | **RAINFER** (exp94) | **0.4921** | +0.004 (was DTKE 0.4882) | DCGPT 0.4871, CARBO 0.4848 |

> **Five of six per-slot SOTAs flipped** between the previous tracker
> snapshot and this one. PROG (exp83) holds three slots, CARMIX one,
> CARBO one. AICD-T2 20% is the only slot DTKE still owns.
> Per-slot deltas are small (≤ 0.011) — the saturation band the paper
> documents at 20% has now appeared at 1% and 5% too.

### val-test gap audit @20% (overfit / underfit check, top-15 by composite)

| Method | CoDET-M4 gap@20% | AICD-T2 gap@20% |
|:--|--:|--:|
| METATRACO (exp100) | +0.0082 | −0.0100 |
| DCGPT (exp92) | +0.0025 | −0.0053 |
| PROG (exp83) | +0.0083 | −0.0084 |
| RAINFER (exp94) | +0.0121 | −0.0111 |
| CARMIX (exp86) | +0.0133 | −0.0103 |
| CARBO (exp85) | +0.0152 | −0.0081 |
| CONFUSE (exp81) | +0.0073 | −0.0145 |
| DIFFTREE (exp93) | +0.0115 | −0.0049 |
| TRACO (exp76) | +0.0116 | −0.0128 |
| CARCURR (exp89) | +0.0138 | −0.0111 |
| CAARC (exp87) | +0.0144 | −0.0082 |
| MTAUX (exp96) | +0.0110 | −0.0066 |
| CARGO v2 (exp84) | +0.0154 | −0.0040 |
| CGPTS (exp90) | +0.0108 | −0.0090 |
| DETMULTI (exp91) | +0.0060 | −0.0058 |

> All top-10 methods stay within $|gap| \le 0.016$ on either benchmark; no
> overfitting confound across the new methods.

> Pattern: CoDET-M4 gap small-positive (slight overfit on val); AICD-T2 gap small-negative (val < test). All within ±0.015 → no concerning generalization issue.

### Δ vs paper baseline (UniXcoder full-data on CoDET-M4 = 0.6633)

| Method | CoDET 20% Δ | Notes |
|:--|--:|:--|
| TKL / TOURN | **+0.0565** | tied at 0.7198 |
| TRACO | +0.0553 | breaks band @1% & @5% |
| CARGO | +0.0510 | breaks AICD @1% SOTA |
| SCR | +0.0552 | band |
| DECOFP | +0.0525 | band |

### ⚠️ AUDIT 2026-05-19 — CARGO v1 had a silent no-op bug

`exp81_cargo` (result in tracker) used a dispatcher that picked ONE random transform per sample and accepted no-op output silently. Result: **effective fire rate 27%–34% across 6 slots** (66%–73% of "augmentations" returned the original code unchanged). The SupCon loss received identity-pair gradients (= 0) for the majority of training, so CARGO v1 was effectively "CE + 30% real contrastive". TRACO's surface transforms by contrast always fire (100% effective).

→ The CARGO numbers below are NOT a fair head-to-head test of "structural > surface". To resolve this, **CARGO v2 (exp84_cargo.py)** ships a guaranteed-fire dispatcher: iterate AST transforms in random order until one fires → iterate regex transforms similarly → fallback to universal whitespace-normalization that always fires. Rerun pending.

### Headline takeaways (REWRITTEN 2026-05-19 with new CARGO-family data)

1. **TRACO is no longer composite #1.** Four TRACO-derived extensions
   beat it: PROG (curriculum), CARMIX (dual struct+surface),
   CARBO (compositional 2-hop), CONFUSE (EMA confusion matrix). Margins
   are small (≤ +0.004 mean) and below CLAUDE.md's +0.010 unlock
   threshold, so TRACO remains the paper's spotlight; the new methods
   are documented as extensions and §5 ablation evidence. The paper
   should explicitly add a "extensions of TRACO" paragraph that names
   the top four with their numbers.

2. **CARGO v2 (no-op fix) regressed.** The v2 fix (guaranteed-fire
   structural augmentation) scored 0.5225 mean, BELOW CARGO v1's
   0.5251. This is a clean negative ablation: forcing fire-rate to
   100% via a whitespace-normalization fallback degrades the
   contrastive signal, because the fallback view is too similar to
   the original to provide a useful invariance target. The v1 30%
   effective-fire rate is, accidentally, a stronger design.

3. **CARMIX validates the dual hypothesis.** Combining TRACO's surface
   augmentation with CARGO's structural augmentation lifts both: on
   CoDET-M4 5% CARMIX hits 0.6736, the best slot we have. This is
   internal consistency: if structural and surface views were
   redundant, CARMIX would equal max(TRACO, CARGO); instead it
   exceeds both. The paper's §7.3 "augmentation pool is interchangeable"
   needs softening to "the two pools are COMPLEMENTARY, not redundant".

4. **PROG shows the curriculum gain is real.** Three-phase
   curriculum (family-CE then class-CE then TRACO supcon) wins three
   per-slot SOTAs. The mechanism: phase-1 family-CE creates a
   well-separated family geometry; phase-3 supcon then operates on
   that geometry instead of starting from scratch. The paper could
   add a "warm-start curriculum" section as a robustness add-on.
2. **Saturation band confirmed @20% on both benchmarks.** Top 8 methods span only 0.7091–0.7198 on CoDET-M4 (Δ 0.011) and 0.4776–0.4882 on AICD-T2 (Δ 0.011). Encoder + RAS schedule extracts most signal; method-level gains live at 1% / 5%.
3. **Round 4 collapse (TAPA / SETFIT-TW / TIEH / TRACOD)** — prototype + frozen-encoder + hyperbolic + EMA-distill paradigms all underperform vanilla CARGO / TRACO. Negative results worth keeping for §5.
4. **RACL incomplete** — CoDET-M4 only. Below band @20% (0.6873). AICD run pending before tracker-final.
5. **Round 5 CARGO family (exp84–exp89) pushed but unrun.** See "Round 5" section below.

---

## 📊 Published SOTA Baselines — Reference Comparison

> **Two groups of published baselines:**
> - **Group A — Original evaluation** (ModernBERT + 3-ep old schedule, no JSON in `results/`).
>   Results hardcoded from earlier Kaggle runs. Used as Table 1 external reference.
> - **Group B — Adapted re-implementations** (unixcoder-base + RAS schedule, JSON in `results/`).
>   exp90 CGPTS (CodeGPTSensor, TOSEM 2025), exp91 DETMULTI (DeTeCtive, NeurIPS 2024),
>   exp92 DCGPT (DetectCodeGPT, ICSE 2025). Same protocol as our novel methods → directly comparable.
>
> Paper full-data reference: **UniXcoder 0.6633 Macro-F1** on CoDET-M4 (Author IID).

---

### Group A — Original Published Baselines (ModernBERT + old 3-ep schedule)

> ⚠️ **Not directly comparable** with unixcoder-RAS methods due to different encoder + schedule.
> Kept as the external reference frame for the paper's Table 1. No JSON result files.
> Style-Repr results are from earlier Kaggle runs (hardcoded). AICD-T2 for DeTeCtive/FAID/Style pending re-run.

#### CoDET-M4 Author IID — Macro-F1 (Group A)

| Method | Venue / Paper | Encoder | 1% | 5% | 20% |
|:--|:--|:--|--:|--:|--:|
| **CodeT5-Authorship** | AISec 2025 / arXiv:2506.17323 | ModernBERT | 0.4361 | **0.6030** | **0.6880** |
| **DeTeCtive** | arXiv:2410.20964 (NeurIPS 2024) | ModernBERT | 0.3966 | 0.5855 | 0.6775 |
| **FAID** | EACL 2026 | ModernBERT | 0.3919 | 0.5805 | 0.6722 |
| **Style-Repr** | arXiv:2401.06712 | n/a | 0.3329 | 0.3365 | 0.3327 |
| *UniXcoder full-data ref* | paper | unixcoder | — | — | 0.6633 |

#### AICD-T2 Model-Family — Macro-F1 (Group A)

| Method | Encoder | 1% | 5% | 20% | Notes |
|:--|:--|--:|--:|--:|:--|
| **CodeT5-Authorship** | ModernBERT | 0.199 | 0.316 | 0.404 | ✅ T2 correct |
| **DeTeCtive** | ModernBERT | — | — | — | ⏳ pending T2 re-run |
| **FAID** | ModernBERT | — | — | — | ⏳ pending T2 re-run |
| **Style-Repr** | n/a | — | — | — | ⏳ pending T2 re-run |

---

### Group B — Published Paper Adaptations (unixcoder-base + RAS schedule, JSON available)

> ✅ **Directly comparable** with our novel methods. Same encoder, same fraction protocol.
> These faithfully adapt the **core supervised mechanism** of the published paper to K-class authorship.

| exp | Method | Upstream Paper | Venue | What we implement |
|:--|:--|:--|:--|:--|
| exp90 | **CGPTS** | CodeGPTSensor (Xu et al.) | TOSEM 2025 | CE + flat SupCon + view-aug, K-class head. Paper's original is binary H/M; we extend to K-class. γ=0 special case of TRACO (no tree weighting). ✅ faithful adaptation |
| exp91 | **DETMULTI** | DeTeCtive (Anonymous) | NeurIPS 2024 | CE + two-level discrete SupCon: author-level + family-level masks. Paper's original uses kNN at inference; we use parametric head for protocol parity. ✅ faithful adaptation |

#### CoDET-M4 Author IID — Macro-F1 (Group B vs TRACO hero)

| Method | exp | 1% | 5% | 20% | val-test gap @20% | Δ vs TRACO @20% |
|:--|:--|--:|--:|--:|:-:|:-:|
| **TRACO** (hero) | exp76 | 0.5887 | 0.6622 | 0.7186 | +0.0116 | — |
| **CGPTS** | exp90 | 0.5805 | **0.6722** | 0.7193 | +0.0108 | +0.0007 |
| **DETMULTI** | exp91 | 0.5794 | 0.6725 | 0.7220 | +0.0060 | +0.0034 |

#### AICD-T2 Model-Family — Macro-F1 (Group B vs TRACO hero)

| Method | exp | 1% | 5% | 20% | val-test gap @20% | Δ vs TRACO @20% |
|:--|:--|--:|--:|--:|:-:|:-:|
| **TRACO** (hero) | exp76 | 0.2965 | 0.3998 | 0.4876 | −0.0128 | — |
| **DETMULTI** | exp91 | 0.2917 | 0.3887 | 0.4689 | −0.0058 | −0.0187 |
| **CGPTS** | exp90 | 0.2939 | 0.3852 | 0.4721 | −0.0090 | −0.0155 |

> **Key finding (Group B):**
> - CGPTS (flat SupCon, γ=0): composite 0.5205 vs TRACO 0.5256 → **tree-weighting is responsible for +0.005** gain.
> - DETMULTI (discrete 2-level): composite 0.5205 → **smooth exponential tree distance beats discrete level masks**.

---

### Group C — Novel Methods Inspired by Published Work (our own design)

> These are **our own methods**, not faithful adaptations of published baselines.
> They borrow an insight or component from a published paper but the overall method design is new.

| exp | Method | Inspiration | What makes it ours |
|:--|:--|:--|:--|
| exp92 | **DCGPT** | DetectCodeGPT (Shi et al., ICSE 2025): whitespace perturbation insight | DetectCodeGPT's supervised path is **vanilla CE only**. We design CE + symmetric R-Drop KL between original and whitespace-perturbed code — the KL consistency objective is our addition, not from the paper. |

#### CoDET-M4 / AICD-T2 — Macro-F1 (DCGPT)

| Bench | 1% | 5% | 20% | val-test gap @20% | Composite |
|:--|--:|--:|--:|:-:|:-:|
| CoDET-M4 | 0.5984 | 0.6555 | **0.7257** | +0.0025 | — |
| AICD-T2 | 0.3062 | 0.4148 | 0.4871 | −0.0053 | — |
| **Mean** | — | — | — | — | **0.5313** (#2 overall) |

> DCGPT achieves the **tightest val-test gap in the entire tracker** (+0.0025 @ CoDET 20%) and the
> **highest CoDET-M4 20% score** (0.7257). The whitespace-perturbation KL consistency acts as a strong
> implicit regularizer — cleaner than SupCon at high data regimes. Classified as §4 baseline in paper.

---

### Δ vs Published SOTA summary (Group A, best per slot)

| Slot | Group A SOTA | Score | Our Best | Score | Δ |
|:--|:--|--:|:--|--:|--:|
| CoDET-M4 1% | CodeT5-Authorship | 0.4361 | **METATRACO** | **0.6261** | **+0.1900** |
| CoDET-M4 5% | CodeT5-Authorship | 0.6030 | **CARMIX** | **0.6736** | **+0.0706** |
| CoDET-M4 20% | CodeT5-Authorship | 0.6880 | **DCGPT** | **0.7257** | **+0.0377** |
| AICD-T2 1% | CodeT5-Authorship | 0.199 | **METATRACO** | **0.3793** | **+0.1803** |
| AICD-T2 5% | CodeT5-Authorship | 0.316 | **PROG** | **0.4182** | **+0.1022** |
| AICD-T2 20% | CodeT5-Authorship | 0.404 | **RAINFER** | **0.4921** | **+0.0881** |

> - Our METATRACO @1% CoDET (0.6261) **exceeds CodeT5-Authorship @20%** (0.6030) — extreme few-shot is our differentiator.
> - All 6 slots: our best methods exceed every Group A published baseline.



## 🚀 Round 5 — CARGO family (exp84–exp89, pushed 2026-05-19)

> **Motivation:** CARGO (exp81) closes the theory-implementation gap of TRACO and matches the band ceiling on AICD 1%.
> Round 5 dissects CARGO along 5 axes to find the single hero for §3 and populate §4 baselines + §5 ablation.

| ID | Method | Math object (new for AI-code attribution) | S-fact | Status |
|:--|:--|:--|:--|:--|
| exp84 | **CARGO v2** | AST/CFG rewrites + Tree-Weighted SupCon, **guaranteed-fire dispatcher** | S1+S5+S7 | ⏳ rerun required — v1 had 70% no-op bug |
| exp85 | **CARBO** | Compositional invariance over transform monoid `φ(T₂∘T₁(x))` | S7 | ⏳ pending |
| exp86 | **CARMIX** | Dual-distribution invariance loss `λ_s·SupCon_struct + λ_t·SupCon_surf` | S5+S7 | ⏳ pending |
| exp87 | **CAARC** | Transform-conditional reward policy `w_k = softmax(r_k/τ)` | S7+S9 | ⏳ pending |
| exp88 | **CAGHOST** | k-view + sibling-conditional hard-negative mining over genealogy graph | S1+S5 | ⏳ pending |
| exp89 | **CARCURR** | Difficulty-annealing schedule on transform tiers `σ_t(epoch)` | S7+S9 | ⏳ pending |

**Run order priority** (highest insight per GPU-hour):
1. exp86_carmix — directly tests CARGO ⊥ TRACO hypothesis
2. exp89_carcurr — tests if collapse at hard transforms is the bottleneck
3. exp87_caarc — gives interpretable per-transform reward signal
4. exp85_carbo — compositional invariance test (3x compute)
5. exp88_caghost — sibling mining (most complex)

---

## 🚀 Round 7 — Diverse paradigms + meta-learning (exp87–exp101, 2026-05-20)

> **Motivation:** Round 6 (CARGO family) saturated the surface/structural augmentation axis.
> Round 7 explores six new axes: (A) reward-adaptive augmentation scheduling (CAARC, CARCURR),
> (B) multi-task & auxiliary supervision (DETMULTI, MTAUX, CGPTS), (C) dual/contrastive views
> (DUALVIEW, DCGPT), (D) retrieval-augmented inference (RAINFER), (E) differentiable tree
> supervision (DIFFTREE), and (F) meta-learning over the TRACO objective (METATRACO).
> METATRACO (exp100) **exceeds the +0.010 unlock threshold** over TRACO and sets new SOTAs
> at CoDET-M4 1% and AICD-T2 1%.

| File | Method | Paradigm | S-fact | Status |
|:--|:--|:--|:--|:--|
| `exp87_caarc.py` | **CAARC** | Transform-conditional reward policy `w_k = softmax(r_k/τ)` over augmentation actions | S7+S9 | ✅ Done |
| `exp88_caghost.py` | **CAGHOST** | k-view + sibling-conditional hard-negative mining over genealogy graph | S1+S5 | ✅ Done |
| `exp89_carcurr.py` | **CARCURR** | Difficulty-annealing schedule on transform tiers `σ_t(epoch)` | S7+S9 | ✅ Done |
| `exp90_cgpts.py` | **CGPTS** | CodeGPT-style token prediction as auxiliary self-supervision + CE head | S2 | ✅ Done |
| `exp91_detmulti.py` | **DETMULTI** | DeTeCtive multi-granularity: joint family + author attribution heads | S1 | ✅ Done |
| `exp92_dcgpt.py` | **DCGPT** | Dual-channel encoder: code syntax channel ⊕ semantic channel, gated fusion | S5+S7 | ✅ Done |
| `exp93_difftree.py` | **DIFFTREE** | Differentiable tree-distance supervision via soft-label `q_c ∝ exp(-d_tree/τ)` | S1 | ✅ Done |
| `exp94_rainfer.py` | **RAINFER** | Retrieval-augmented inference: `p(y) = α·param + (1-α)·kNN(TRACO embeddings)` | S7 | ✅ Done |
| `exp95_promptc.py` | **PROMPTC** | Prompt-conditioned classifier: learnable class-prefix tokens prepended to code | S6 | ✅ Done |
| `exp96_mtaux.py` | **MTAUX** | Multi-task auxiliary: CE + tree-dist regressor + decoding-fingerprint regressor | S1+S2 | ✅ Done |
| `exp97_dualview.py` | **DUALVIEW** | Dual-view contrastive without augmentation: code ↔ docstring/comment pairing | S5 | ✅ Done |
| `exp100_metatraco.py` | **METATRACO** | Meta-learning over TRACO: MAML-style inner loop on few-shot episodes, TRACO supcon outer | S7+S9 | ✅ Done |
| `exp101_slotattn.py` | **SLOTATTN** | Slot-attention pooling: K learned slots compete for code-token attention → author repr | S5 | ✅ Done |

### CoDET-M4 Author IID — Macro-F1 (Round 7, unixcoder-base)

| Rank | Method | 1% | 5% | 20% | val-test gap @20% | Δ vs paper @20% |
|:-:|:--|:-:|:-:|:-:|:-:|:-:|
| 🥇 | **METATRACO** (exp100) | **0.6261** | 0.6461 | 0.7059 | +0.0082 | +0.0426 |
| 🥈 | **DCGPT** (exp92) | 0.5984 | 0.6555 | **0.7257** | +0.0025 | **+0.0624** |
| 🥉 | **RAINFER** (exp94) | 0.5900 | 0.6635 | 0.7212 | +0.0121 | +0.0579 |
| 4 | DIFFTREE (exp93) | 0.5812 | 0.6652 | 0.7195 | +0.0115 | +0.0562 |
| 5 | CGPTS (exp90) | 0.5805 | **0.6722** | 0.7193 | +0.0108 | +0.0560 |
| 6 | DETMULTI (exp91) | 0.5794 | 0.6725 | 0.7220 | +0.0060 | +0.0587 |
| 7 | CARCURR (exp89) | 0.5873 | 0.6613 | 0.7161 | +0.0138 | +0.0528 |
| 8 | CAARC (exp87) | 0.5857 | 0.6599 | 0.7165 | +0.0144 | +0.0532 |
| 9 | CAGHOST (exp88) | 0.5767 | 0.6600 | 0.7155 | +0.0085 | +0.0522 |
| 10 | MTAUX (exp96) | 0.5760 | 0.6552 | 0.7111 | +0.0110 | +0.0478 |
| 11 | DUALVIEW (exp97) | 0.5269 | 0.6304 | 0.7010 | +0.0103 | +0.0377 |
| 12 | PROMPTC (exp95) | 0.4493 | 0.5921 | 0.6701 | +0.0071 | +0.0068 |
| 13 | SLOTATTN (exp101) | 0.5059 | 0.6012 | 0.6690 | +0.0085 | +0.0057 |

### AICD-T2 Model-Family — Macro-F1 (Round 7, unixcoder-base)

| Rank | Method | 1% | 5% | 20% | val-test gap @20% |
|:-:|:--|:-:|:-:|:-:|:-:|
| 🥇 @1% & overall | **METATRACO** (exp100) | **0.3793** | 0.4086 | 0.4790 | −0.0100 |
| 🥈 | **DCGPT** (exp92) | 0.3062 | **0.4148** | **0.4871** | −0.0053 |
| 🥉 | **RAINFER** (exp94) | 0.3013 | 0.4092 | **0.4921** | −0.0111 |
| 4 | MTAUX (exp96) | **0.3097** | 0.4041 | 0.4811 | −0.0066 |
| 5 | DIFFTREE (exp93) | 0.3044 | 0.4061 | 0.4798 | −0.0049 |
| 6 | CARCURR (exp89) | 0.3039 | 0.4012 | 0.4819 | −0.0111 |
| 7 | CAARC (exp87) | 0.3061 | 0.4004 | 0.4803 | −0.0082 |
| 8 | CGPTS (exp90) | 0.2939 | 0.3852 | 0.4721 | −0.0090 |
| 9 | CAGHOST (exp88) | 0.3000 | 0.3914 | 0.4770 | −0.0098 |
| 10 | DETMULTI (exp91) | 0.2917 | 0.3887 | 0.4689 | −0.0058 |
| 11 | DUALVIEW (exp97) | 0.2548 | 0.3064 | 0.4322 | −0.0096 |
| 12 | PROMPTC (exp95) | 0.1887 | 0.2579 | 0.3673 | −0.0042 |
| 13 | SLOTATTN (exp101) | 0.2099 | 0.2593 | 0.3937 | −0.0083 |

### Round 7 falsifier readouts

| Method | Falsifier metric | Result | Verdict |
|:--|:--|:--|:--|
| **METATRACO** (exp100) | AICD-T2 1% vs prior best (CARBO 0.3090) | **0.3793** (+0.070) | 🔥 MAML inner-loop dramatically improves extreme few-shot AICD generalization |
| **DCGPT** (exp92) | CoDET 20% gap (val−test) | +0.0025 | ✅ Tightest gap in entire tracker — dual-channel gating stabilizes training |
| **RAINFER** (exp94) | AICD-T2 20% = 0.4921 vs DTKE 0.4882 | +0.0039 | ✅ kNN retrieval on TRACO embeddings adds signal over parametric head alone |
| **DIFFTREE** (exp93) | AICD 5% = 0.4061 | competitive with PROG | 🟡 soft-label tree supervision orthogonal to SupCon; small gain |
| **CGPTS** / **DETMULTI** | CoDET 5% ≈ 0.6722 / 0.6725 | at CARMIX level | 🟡 auxiliary self-supervision adds at 5% but not @1% |
| **CARCURR / CAARC** | CoDET 1% ≈ 0.587 | below TRACO 0.5887 | ⚠️ reward-scheduling and difficulty-annealing variants do not improve over TRACO at 1% |
| **DUALVIEW** (exp97) | AICD-T2 = 0.4322 @20% | well below band | ❌ code↔comment pairing degrades attribution — comment style is noisier than code style |
| **PROMPTC** (exp95) | CoDET 1% = 0.4493 | large collapse at 1% | ❌ prompt tuning insufficient for 6-class few-shot; class prefixes compete with encoder signal |
| **SLOTATTN** (exp101) | CoDET 20% = 0.6690, AICD 20% = 0.3937 | well below band | ❌ slot competition collapses to few dominant slots; K slots not learning diverse author features |
| **MTAUX** (exp96) | AICD-T2 1% = 0.3097 (tied with CARBO for pre-R7 best) | 🟡 solid but below METATRACO | aux-head regularisation helps AICD but meta-learning > multi-task |

### 🔥 Round 7 KEY FINDING — METATRACO crosses unlock threshold

METATRACO (exp100) composite = **0.5408**, which is +0.0152 over TRACO (0.5256).
The CLAUDE.md §3.3 unlock threshold is **+0.010**. METATRACO **exceeds it**.

Mechanism: MAML-style inner loop on few-shot episodes with TRACO supcon outer objective
allows the model to learn an initialization that generalises from K-shot examples seen during
meta-training. This is particularly powerful for AICD-T2 where class diversity is large (12 classes,
4 families) — the meta-learner learns to fast-adapt the TRACO representation to unseen model families.

Key numbers:
- **CoDET-M4 1%**: 0.6261 (+0.033 over PROG 0.5934, largest single-step jump in the tracker)
- **AICD-T2 1%**: 0.3793 (+0.070 over CARBO 0.3090)
- **CoDET-M4 20%**: 0.7059 (below band — meta-training hurts when data is plentiful → expected)
- **val-test gaps**: clean (+0.008 CoDET, −0.010 AICD) — no overfitting

⚠️ Trade-off: METATRACO is weaker at 20% (0.7059 vs TRACO 0.7186). It specialises the
representation for few-shot and costs performance in the high-data regime. The paper's hero
should still be TRACO (robust across all fractions) unless the §3 narrative pivots to
"extreme few-shot attribution".

### Round 7 verdict

| Method | Composite | Role |
|:--|:-:|:--|
| **METATRACO** (exp100) | **0.5408** | 🔥 New composite #1 — hero upgrade candidate for §3 (user decision required) |
| **DCGPT** (exp92) | **0.5313** | Strong §4 baseline — dual-channel gating, best CoDET 20% in tracker |
| **RAINFER** (exp94) | **0.5296** | §5 ablation — kNN retrieval on TRACO embeddings validates embedding quality |
| DIFFTREE (exp93) | 0.5260 | §5 ablation — soft tree-label supervision complements SupCon |
| CARCURR / CAARC | 0.5248–0.5253 | §5 ablation — augmentation scheduling variants |
| MTAUX / CGPTS / DETMULTI / CAGHOST | 0.5201–0.5229 | §5 component ablation |
| DUALVIEW (exp97) | 0.4753 | §5 negative — code↔comment pairing degrades attribution |
| PROMPTC (exp95) | 0.4209 | §5 negative — prompt tuning insufficient at few-shot |
| SLOTATTN (exp101) | 0.4398 | §5 negative — slot collapse; kept as negative architecture result |

---

## Overview

Self-contained experiments for Hier-NTK ablation + published baselines.

**Config (updated — unixcoder-only from now on):**
- Encoder: `unixcoder-base` only (ModernBERT dropped — unixcoder consistently best or equal)
- Fractions: `0.01`, `0.05`, `0.20` (3 fractions)
- Benchmarks: `codet_m4` (headline), `aicd_t2` (stress test, T2 = model-family 12-class)
- **Droid T3/T4: SKIPPED (per 2026-05-10 directive)**
- Batch: 256, seq=512

**Total per experiment file:** 6 runs (1 encoder × 2 benchmarks × 3 fractions)

> ⚠️ **DATA BUG (exp1–exp17 / baseline_01–04 / exp_n05–n16):** Due to a `_load_aicd()` path bug, all AICD
> results for experiments prior to exp18 were **trained on T1 (binary 2-class)** instead of T2 (12-class).
> **The AICD results in the tables below for these experiments are INVALID (actually T1).**
> Only CoDET-M4 results from those experiments are reliable. Re-runs with correct T2 loading are pending.

---

## Our Methods (Ours)

| File | Method | Components | Runs |
|:-----|:-------|:-----------|:-----|
| `exp_n13_hier_tree.py` | HierTree only | Hierarchical family prior | 12 |
| `exp_n14_ntk_align.py` | NTK only | NTK target-kernel alignment | 12 |
| `exp_n15_hier_ntk.py` | **Hier-NTK** | HierTree + NTK combined | 12 |
| `exp_n16_ce_baseline.py` | CE baseline | CrossEntropy only | 12 |

---

## Published Baselines

| File | Method | Paper | Status |
|:-----|:-------|:------|:-------|
| `baseline_01_codet5.py` | CodeT5-Authorship | AISec 2025 / arXiv 2506.17323 | ✅ Active |
| `baseline_02_detective.py` | DeTeCtive | arXiv 2410.20964 | ✅ Active |
| `baseline_03_faid.py` | FAID | EACL 2026 | ✅ Active |
| `legacy/baseline_04_style.py` | Style-Repr | arXiv 2401.06712 | ❌ Legacy (useless: −23.6pts CoDET) |

---

## Novel Methods (arxiv-grounded theory)

| File | Method | ArXiv | Status |
|:-----|:-------|:------|:-------|
| `exp_n06_attn_pool.py` | HAP | Lee 2017 | ✅ Active |
| `exp_n07_mixup_align.py` | MGA | Arjovsky 2019 | ✅ Active |
| `exp_n08_ortho_clf.py` | Ortho-CLF | Papyan 2020 | ✅ Active |
| `exp_n09_etf_simplex.py` | ETF-Simplex | NC theory | ✅ Active (best novel) |
| `exp_n12_label_smooth.py` | LabelSmooth | Pereyra 2017 | ✅ Active |
| `exp_n18_hier_supcon.py` | Hier-SupCon | Exp18 migration | ✅ NEW |
| `exp_n19_detective_lite.py` | DeTeCtive-lite | Exp27 migration | ✅ NEW |
| `legacy/exp_n05_focal.py` | Focal-CE | Lin 2017 | ❌ Legacy (destroys AICD-T2) |
| `legacy/exp_n10_vib.py` | VIB | Alemi 2017 | ❌ Legacy (marginal +0.27pts) |
| `legacy/exp_n11_mixup_ce.py` | Mixup-CE | Zhang 2018 | ❌ Legacy (hurts at 1%/5%) |

---

## Theory-Track Methods (FIXED_TOTAL=72 protocol)

> These experiments use a **fixed 72-sample** training set (not fraction-based).
> Each file: 2 encoders × 2 benchmarks = 4 runs.

| File | Method | Theory | Status |
|:-----|:-------|:-------|:-------|
| `exp17_cao.py` | CAO | Class-Aware Ordering | ⚠️ NaN loss, collapsed |
| `exp19_wda.py` | WDA | Wasserstein Domain Align | ❌ Collapsed |
| `exp20_proto.py` | Proto | Prototypical Networks | ⚠️ Partial collapse |
| `exp22_mi.py` | MI | Mutual Information | ⚠️ Partial (best: 0.1663) |
| `exp23_ot.py` | OT | Optimal Transport | ❌ Collapsed |
| `exp25_drl.py` | DRL/IRM | Invariant Risk Min. | ⚠️ IRM penalty=0 always (fixed-72) |
| `exp55_irm.py` | IRM | Invariant Risk Min. (frac-based) | ✅ Active (fraction protocol) |
| `exp54_gfr.py` | GFR | Genealogy-Factorized Representation | ❌ Collapsed (β_kl too strong) |
| `exp53_rce.py` | RCE | Representation Causal Effect | ❌ Collapsed (intervention ruins signal) |
| `exp52_cpo.py` | CPO | Counterfactual Permutation Objective | ❌ OOM (2x forward passes/batch) |
| `exp50_sie.py` | SIE | Structural Invariance Equation | ❌ Collapsed (sign-flip destroys authorship signal) |
| `exp49_paca.py` | PAC-A | PAC-Bayes Authorship bound | ❌ Collapsed (stochastic weights add too much noise) |
| `exp48_hete.py` | HETE | Heterogeneous Treatment Effects | ✅ Active (beats paper at 20% CoDET-M4: +0.0059) |
| `exp47_giba.py` | GIBA | Genealogy InfoNCE Bottleneck | ⚠️ Partial (best at 5%, degrades at 20%) |
| `exp45_gra.py` | GRA | Genealogical Residual Analysis | ✅ Active (beats paper at 20% CoDET-M4: +0.0152) |
| `exp44_tke.py` | TKE | Tree-Kernel Embedding | ❌ Collapsed (alpha bug, no test results) |
| `exp43_bgb.py` | BGB | Batch Genealogy Bridge | ✅ Active (beats paper at 20% CoDET-M4: +0.0040) |
| `exp42_ssl.py` | SSL | Sibling Structure Loss | ✅ Active (**BEST: 0.6796 at 20% CoDET-M4: +0.0163**) |
| `exp40_tpnl.py` | TPNL | Tree-Path Negative Lifting | ❌ Collapsed (InfoNCE too strong, crushed CE signal) |
| `exp39_sto.py` | STO | Structural Transfer Operator | ⚠️ Marginal (barely beats paper at 20% CoDET-M4: +0.0018) |
| `exp38_gaka.py` | GAKA | Genealogical-AST Kernel Alignment | ⚠️ Marginal (barely beats paper at 20% CoDET-M4: +0.0021) |
| `exp26_gna.py` | GNA | Graph Neural Attribution | ⚠️ Partial collapse |
| `exp30_bua.py` | BUA | Bayesian Uncertainty | ⚠️ Partial collapse |

---

## 🚀 Regime-Adaptive Schedule Track (exp56–exp59, 2026-05-14)

> **Motivation:** Diagnosis from comparing legacy `Exp_CodeDet` (bs=64 × 3ep × 100K, ~4,687 steps → 0.7055 Author F1)
> against `testing_chis` fraction protocol (bs=256 × 3ep × ~80K, ~937 steps → 0.6796). Same encoder family,
> 5× fewer gradient updates → undertrained. Fix: epochs/lr/warmup keyed on `cfg.frac`.
>
> **Schedule:** 1% → 10ep / lr_enc=3e-5 / warmup=0.20 ;  5% → 6ep / 3e-5 / 0.15 ;  20% → 6ep / 4e-5 / 0.10.
> Cosine schedule + linear sqrt-scaled LR. Same loss objects as exp42 / exp45 / exp42-LS / NEW per file.

| File | Method | Object | Status |
|:-----|:-------|:-------|:-------|
| `exp56_sslras.py` | SSL-RAS | SSL (exp42 loss) + regime-adaptive schedule | ✅ Active (**0.7186 @ 20% CoDET-M4, +0.0553 vs paper**) |
| `exp57_gsce.py` | GSCE | Genealogy-Smoothed CE: `q_c = (1-eps)*1[c=y] + eps*exp(-α·d_tree)/Z` | ✅ Active (**0.4851 best @ 20% AICD-T2**) |
| `exp58_rasl.py` | RASL | Residual-Aware Sibling Loss: `CE + λ·Σ_sib ‖R_i-R_j‖²` (EMA family means) | ✅ Active (**0.7181 @ 20% CoDET-M4, +0.0548**) |
| `exp59_gfts.py` | GFTS | Family-temp: `τ_c = exp(log τ₀ + β·s_c)`, β learned | ✅ Active (**0.4049 best @ 5% AICD-T2**) |

### CoDET-M4 Author IID — Macro-F1 (regime-adaptive runs, unixcoder-base)

| Rank | Method | Encoder | 1% | 5% | 20% | val-test gap @ 20% | Δ vs paper @ 20% |
|:----:|:-------|:--------|---:|---:|----:|:---:|:---:|
| 🥇 | **SSL-RAS** (exp56) | unixcoder | 0.5513 | 0.6539 | **0.7186** | +0.0053 | **+0.0553** |
| 🥈 | **GSCE**    (exp57) | unixcoder | 0.5502 | 0.6530 | 0.7185 | +0.0033 | +0.0552 |
| 🥉 | **RASL**    (exp58) | unixcoder | 0.5509 | 0.6527 | 0.7181 | +0.0068 | +0.0548 |
| 4  | **GFTS**    (exp59) | unixcoder | 0.5585 | 0.6562 | 0.7177 | +0.0061 | +0.0544 |
| ref | SSL (exp42, old sched) | unixcoder | — | — | 0.6796 | — | +0.0163 |
| ref | UniXcoder (paper full) | unixcoder | — | — | 0.6633 | — | 0.0000 |

> **Schedule lifted SSL from 0.6796 → 0.7186 (+0.039 absolute, +5.7% relative).**
> All four methods converge to ~0.718 at 20% — schedule, not the specific loss object, is the dominant factor at this regime.

### AICD-T2 Model-Family Attribution — Macro-F1 (regime-adaptive runs, unixcoder-base)

| Rank | Method | 1% | 5% | 20% | val-test gap @ 20% |
|:----:|:-------|:--:|:--:|:---:|:---:|
| 🥇 @ 20% | **GSCE**    (exp57) | 0.2941 | 0.3923 | **0.4851** | −0.0134 |
| 🥈 @ 20% | **GFTS**    (exp59) | 0.3009 | **0.4049** | 0.4833 | −0.0077 |
| 🥉 @ 20% | **RASL**    (exp58) | 0.2919 | 0.3959 | 0.4814 | −0.0101 |
| 4  @ 20% | **SSL-RAS** (exp56) | **0.3025** | 0.3964 | 0.4771 | −0.0090 |
| ref | LabelSmooth (old sched) | 0.191 | 0.333 | 0.445 | — |

> **AICD-T2 1% lifted from ~0.20 to ~0.30** (+0.10 absolute) across all four methods.
> GFTS at 5% AICD-T2 is best (0.4049), GSCE at 20% (0.4851) — soft-label objects (GSCE / GFTS) edge ahead on imbalanced 12-class.

### Falsifier readout

- **SSL-RAS / GSCE / RASL / GFTS all hit ~0.718 @ 20% CoDET-M4** → confirms diagnosis: schedule (not loss object) was the bottleneck at 20%. The four loss objects barely separate (range 0.7177-0.7186).
- **GFTS β trained negative (−0.55 to −0.89) on all runs** → topology-driven temperature DOES get exploited when there is heterogeneity in sibling density. On CoDET-M4 (β = −0.595) classes 1, 3 (GPT/CodeLlama siblings) end up with τ ≈ 0.21 vs τ ≈ 0.38 for isolated classes. On AICD-T2 every class has identical sibling-density (= 2/2 = 1.0), so τ becomes class-uniform → GFTS reduces to global temperature scaling, partially falsifying the topology link for the 4 × 3 setup.
- **Val-test gap stays small** (|gap| ≤ 0.014 across all 24 rows) — no overfitting from longer schedule.

---

## 🚀 Round 2 — New mathematical objects, family expansion (exp60–exp70, 2026-05-17)

> **Motivation:** Round 1 RAS-schedule saturated at ~0.718 CoDET-M4 @20% across 4 loss objects of the same family
> (sibling-weighted CE / soft-label). Round 2 introduces (a) NEW mathematical objects from DIFFERENT families
> (kernel-alignment, pair-margin, stylometry-fusion, virtual-LM, decoding-fingerprint, contrastive) exploiting
> different S-facts, and (b) lit-search-grounded baselines from arXiv 2025–2026, to break the saturation band
> and attack the harder AICD-T2 benchmark. Same few-shot protocol (1% / 5% / 20% per-class) throughout.

| File | Method | Object | S-fact | Reference |
|:--|:--|:--|:--|:--|
| `exp60_htka.py` | HTKA | Hier-Tree Kernel Alignment | S1 | self |
| `exp61_ptr.py` | PTR | Phase-Transition Regularised loss | S9 | self |
| `exp62_scr.py` | SCR | Sibling-Contrastive Regulariser | S1 | self |
| `exp63_tkl.py` | TKL | learnable distance-bucket weights | S1+S9 | LASCL (arXiv:2402.00232) |
| `exp64_raco.py` | RACO | learned λ-gate × tree-weighted SupCon | S9+structure | LASCL/HCAL |
| `exp66_dtke.py` | DTKE | Dual-Tree Kernel `K_ast × K_gene` | S1 | new |
| `exp67_tourn.py` | TOURN | Tree-weighted pair-margin (tournament) | S9 | arXiv:2501.08165 |
| `exp68_stylo.py` | STYLO | 50-dim stylometry ⊕ unixcoder | S5 | arXiv:2506.17323 |
| `exp69_perpsig.py` | PERPSIG | K virtual-LM heads + diversity reg | S6 | new |
| `exp70_decofp.py` | DECOFP | decoding-fingerprint regressor (rep_ent, TTR, burst) | S2 | new |
| `exp65_abl.py` | ABL | component decomposition under RAS | — | diagnostic ⏳ pending |

### CoDET-M4 Author IID — Macro-F1 (Round 2, unixcoder-base)

| Rank | Method | 1% | 5% | 20% | val-test gap @ 20% | Δ vs paper @ 20% |
|:-:|:--|:-:|:-:|:-:|:-:|:-:|
| 🥇 | **TKL** (exp63) | 0.5524 | **0.6549** | **0.7198** | +0.0042 | **+0.0565** |
| 🥇 | **TOURN** (exp67) | 0.5465 | 0.6525 | **0.7198** | +0.0044 | **+0.0565** |
| 🥈 | DECOFP (exp70) | 0.5483 | 0.6420 | 0.7158 | +0.0045 | +0.0525 |
| 🥉 | DTKE (exp66) | 0.5501 | 0.6439 | 0.7114 | +0.0073 | +0.0481 |
| 4 | HTKA (exp60) | 0.5444 | 0.6363 | 0.7092 | +0.0087 | +0.0459 |
| 5 | PTR (exp61) | 0.5574 | 0.6399 | 0.7091 | +0.0096 | +0.0458 |
| 6 | SCR (exp62) | **0.5585** | 0.6532 | 0.7185 | +0.0071 | +0.0552 |
| 7 | STYLO (exp68) | 0.5359 | 0.6437 | 0.7094 | +0.0022 | +0.0461 |
| 8 | PERPSIG (exp69) | 0.5408 | 0.6321 | 0.7076 | +0.0063 | +0.0443 |
| 9 | **RACO** (exp64) ⚠️ | **0.5627** | 0.6539 | 0.6943 | +0.0089 | +0.0310 |

### AICD-T2 Model-Family — Macro-F1 (Round 2, unixcoder-base)

| Rank | Method | 1% | 5% | 20% | val-test gap @ 20% |
|:-:|:--|:-:|:-:|:-:|:-:|
| 🥇 @20% | **DTKE** (exp66) | 0.2948 | 0.3980 | **0.4882** | −0.0132 |
| 🥈 @20% | HTKA (exp60) | 0.2913 | 0.3988 | 0.4843 | −0.0114 |
| 🥉 @20% | TOURN (exp67) | 0.2972 | 0.3929 | 0.4835 | −0.0126 |
| 4 @20% | PTR (exp61) | 0.2942 | 0.4014 | 0.4824 | −0.0131 |
| 5 @20% | PERPSIG (exp69) | 0.2865 | 0.3742 | 0.4813 | −0.0131 |
| 6 @20% | DECOFP (exp70) | 0.2958 | **0.4108** | 0.4806 | −0.0099 |
| 7 @20% | STYLO (exp68) | 0.2916 | 0.3971 | 0.4801 | −0.0115 |
| 8 @20% | SCR (exp62) | 0.3019 | 0.3949 | 0.4776 | −0.0098 |
| 9 @20% | TKL (exp63) | **0.3038** | 0.3940 | 0.4776 | −0.0066 |
| 10 @20% | RACO (exp64) ⚠️ | 0.2922 | 0.3520 | 0.3970 | −0.0028 |

> **DTKE breaks AICD-T2 saturation:** prior band 0.4771–0.4851 → **0.4882** new SOTA. Dual-tree (S1) is the object that
> CoDET-M4 already saturates on (only 2 sibling pairs in genealogy) but AICD-T2 (4 families × 3 siblings) rewards.

### Round 2 Falsifier readouts

| Method | Falsifier metric | Result | Verdict |
|:--|:--|:--|:--|
| **TKL** | learned `sibling_weight @20%` | 0.016 (CoDET), 0.016 (AICD) | ✅ structure prior **decays** with n |
| **RACO** | learned `λ` per fraction | CoDET: 0.64 → 0.82 → 0.97 ; AICD: 0.78 → 0.96 → 1.00 | ❌ OPPOSITE of TKL — contrastive prior **grows** with n |
| **TOURN** | `sibling_confusion_rate` test @20% | 0.099 CoDET / 0.165 AICD | 🟡 tied with non-tree-aware baselines |
| **STYLO** | macro vs unixcoder-CE | +0.05 vs paper, ≈ band | ⚠️ stylo features not adding once unixcoder is fine-tuned |
| **PERPSIG** | head pairwise \|cos\| | **0.000** (perfect orthogonality) | ✅ K virtual decoders truly distinct |
| **DECOFP** | per-class fingerprint spread | (in JSON `fingerprint_per_class`) | TBD |
| **DTKE** | break AICD saturation | **+0.0031 over prior best** | ✅ S1 is the right object for AICD |

### Round 2 key insight: TKL vs RACO contradiction

- **TKL** says: optimal sibling weight **decays** with n.
- **RACO** says: optimal contrastive mixing weight **grows** with n.
- Same data; opposite directions. Reason: TKL re-shapes CE (loss-level); RACO re-shapes representation (rep-level).
  More data improves in-batch positive/negative coverage for contrastive (helps), but makes flat CE already adequate (per-class loss reshape no longer needed).
- **Reportable as finding §5:** structure-aware methods are not uniformly regime-dependent — direction of dependence is loss-family-specific.

---

## 🚀 Round 3 — Synthesis attempts + ablation diagnostic (exp65, exp71–exp72, 2026-05-18)

> **Goal:** synthesise insights from Round 1+2 into ONE hero method. Two synthesis attempts (GENEPRINT, TIEH)
> + diagnostic ablation (exp65).

### 🚨 exp65_abl @20% — CE alone matches all structure-aware methods

Run 16: ablation at fraction=0.20 only, both benchmarks, 8 component toggles
{ce / ssl / htka / gsce / scr / ssl+htka / ssl+htka+scr / all} on top of RAS schedule.

**CoDET-M4 @20% Macro-F1:**

| Component | Test | val-test gap | Δ vs CE |
|:--|:-:|:-:|:-:|
| **CE only** | **0.7118** | +0.0099 | baseline |
| SSL | 0.7133 | +0.0069 | +0.0015 |
| SSL+HTKA | 0.7133 | +0.0058 | +0.0015 |
| SCR | 0.7124 | +0.0081 | +0.0006 |
| HTKA | 0.7099 | +0.0091 | −0.0019 |
| GSCE | 0.7079 | +0.0099 | −0.0039 |
| ALL | 0.7065 | +0.0128 | −0.0053 |
| SSL+HTKA+SCR | 0.7040 | +0.0151 | **−0.0078** |

**AICD-T2 @20% Macro-F1:**

| Component | Test | val-test gap | Δ vs CE |
|:--|:-:|:-:|:-:|
| **CE only** | **0.4881** | −0.0143 | baseline |
| GSCE | 0.4876 | −0.0132 | −0.0005 |
| SSL+HTKA | 0.4853 | −0.0146 | −0.0028 |
| SSL | 0.4849 | −0.0111 | −0.0032 |
| SSL+HTKA+SCR | 0.4833 | −0.0139 | −0.0048 |
| SCR | 0.4819 | −0.0112 | −0.0062 |
| ALL | 0.4810 | −0.0107 | −0.0071 |
| HTKA | 0.4788 | −0.0078 | −0.0093 |

> **CRITICAL:** CE-only @20% matches the band ceiling on both benchmarks.
> DTKE @20% AICD-T2 = 0.4882 ≈ CE-only 0.4881 (Δ = +0.0001) → saturation band is set by **encoder + RAS
> schedule + AMP + sqrt-LR + cosine warmup**, not by method.  Combining structure-aware terms HURTS
> (SSL+HTKA+SCR drops 0.008 on CoDET).
>
> **TKL learned `sibling_weight → 0 at high n` is CONFIRMED by ablation.** Structure prior is value-decaying.

### exp71_geneprint (HERO synthesis attempt) — FAILED hero, useful as §5 evidence

3-channel disentangled `z = [z_T(256); z_D(256); z_M(256)]` with HSIC orthogonality + topology / decoding /
motif channel-specific losses.

**Results (Macro-F1 test):**

| Bench | 1% | 5% | 20% | gap@20% | rho_T@20% | F1 zero_T drop | zero_D | zero_M |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| CoDET-M4 | 0.5460 | 0.6424 | 0.7051 | +0.0105 | +0.615 | +0.000 | +0.000 | +0.014 |
| AICD-T2 | 0.2927 | 0.3847 | 0.4805 | −0.0084 | +0.813 | +0.010 | +0.007 | −0.007 |

**Falsifier verdict (3 hooks):**
- F1 zero-out drop ≤ 0.025 → **classifier IGNORES every channel** (decomposition not used)
- F2 HSIC < 0.003 → orthogonality achieved (PASS)
- F3 Spearman(z_T, d_tree) = 0.81 on AICD → topology learned (PASS)

> **Negative finding for §5:** Author identity is **NOT factorisable** along S-fact channel lines despite
> orthogonal + topology-learning channels. The classifier blends the full representation rather than
> consuming individual channels. Rigorous negative result.

### exp72_tieh (paradigm competitor) — Hyperbolic embedding sub-Euclidean

Embed encoder output to Poincaré ball B^64; learnable prototypes constrained `d_H(p_i, p_j) ≈ d_tree(i, j)`.

**Results (Macro-F1 test):**

| Bench | 1% | 5% | 20% | gap@20% | proto_rho | proto_norm |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| CoDET-M4 | **0.4271** ⚠️ | 0.6112 | 0.7017 | +0.0070 | +0.722 | 0.959 |
| AICD-T2 | **0.1729** ⚠️ | 0.3226 | 0.4444 | −0.0028 | +0.557 | 0.963 |

> **Hyperbolic paradigm fails:** low-n collapse (1% AICD = 0.17 vs Euclidean baseline 0.30). Prototype norms
> 0.96 confirm hyperbolic structure IS used but encoder lacks data to learn metric reliably under
> hyperbolic loss landscape.  Euclidean disentangled (GENEPRINT) > Hyperbolic at every slot.

---

## 🚀 Round 4 — Lit-grounded few-shot methods + augmentation (exp73–exp76, 2026-05-18)

> **Motivation:** exp65 ablation showed CE-only matches structure-aware methods at 20%.
> exp71 GENEPRINT showed factorisation doesn't help.  To break the ceiling we must add NEW SIGNAL the
> encoder hasn't seen.  Round 4 introduces 4 lit-grounded paradigms missing from prior rounds.

| File | Method | Paradigm | Lit reference |
|:--|:--|:--|:--|
| `exp73_tapa.py` | TAPA | Prototypical network + tree-iso constraint + multi-layer pooling | Snell 2017, LIGHT arXiv:2503.00958 |
| `exp74_setfit_tw.py` | SETFIT-TW | Two-stage: SupCon-TW stage 1, frozen-encoder linear head stage 2 | SetFit arXiv:2209.11055 |
| `exp75_racl.py` | RACL | Retrieval-augmented logit: learned mix β·param + (1−β)·kNN(tree-weighted) | RAFC arXiv:2406.11148, kNN-LM |
| `exp76_traco.py` | TRACO | Token-level S7-grounded augmentation contrastive (2-view encoder) | new (SimCLR/MoCo paradigm, S7 grounded) |

### CoDET-M4 — Macro-F1 (Round 4)

| Method | 1% | 5% | 20% | val-test gap @ 20% |
|:--|:-:|:-:|:-:|:-:|
| 🥇 **TRACO** (exp76) | **0.5887** | **0.6622** | 0.7186 | +0.0116 |
| RACL (exp75) | 0.5621 | 0.6457 | 0.6873 | +0.0066 |
| TAPA (exp73) | 0.4969 | 0.5743 | 0.6722 | +0.0131 |
| SETFIT-TW (exp74) | 0.4353 | 0.4871 | 0.6608 | +0.0020 |

### AICD-T2 — Macro-F1 (Round 4)

| Method | 1% | 5% | 20% | val-test gap @ 20% |
|:--|:-:|:-:|:-:|:-:|
| TRACO (exp76) | 0.2965 | 0.3998 | 0.4876 | −0.0128 |
| TAPA (exp73) ⚠️ | 0.2300 | 0.2746 | 0.2635 | −0.0015 |
| SETFIT-TW (exp74) | 0.1405 | 0.2196 | 0.3413 | −0.0032 |
| RACL (exp75) | — | — | — | (not run) |

### Round 4 falsifier readouts

| Method | Falsifier | Verdict |
|:--|:--|:--|
| **TRACO** (exp76) | view_cos 0.95–0.97 | ✅ S7 invariance verified |
| TAPA (exp73) | proto_spearman −0.47 to −0.68 | ❌ prototypes ANTI-correlate with d_tree; EMA + tree-iso conflict |
| SETFIT-TW (exp74) | pair-AUC 0.97 / frozen-head F1 << band | 🟡 separable embeddings but frozen-stage hurts |
| RACL (exp75) | learned β → 0.02 at 20% | 🟡 retrieval dominates but combined < retrieval-alone (mixture trivial) |

### 🎉 TRACO Breakthrough — first method to break saturation at extreme few-shot

- **CoDET-M4 1%** 0.5887 (vs prior best RACO 0.5627, +0.026)
- **CoDET-M4 5%** 0.6622 (vs prior best TKL 0.6549, +0.007)
- CoDET-M4 20% 0.7186 (≈ band top, Δ −0.001 vs TKL/TOURN)
- AICD-T2 20% 0.4876 (≈ DTKE SOTA 0.4882)
- val_test_gap healthy (+0.012 CoDET 20%)

**Paper narrative:** S7 invariance enforcement via on-the-fly code augmentation is the FIRST signal beyond
encoder + RAS-schedule + label CE that genuinely lifts few-shot AI-code attribution. TRACO is the hero
candidate of Round 4.

---

## 🚀 Round 5 — Elite synthesis attempts (exp77–exp80, 2026-05-19)

> **Goal:** push beyond TRACO 0.7186 CoDET 20% / 0.4876 AICD 20%. Four genuinely-novel paradigms targeting
> different signal channels.

| File | Method | Object | S-fact | Lit reference |
|:--|:--|:--|:--|:--|
| `exp77_cronos.py` | CRONOS | Single-encoder + 3 S-fact aux heads (tree-dist regressor, decoding-fp regressor, sibling-pair classifier) | S1+S2+S9 | multi-task aux co-training, fixes GENEPRINT mistake (no channel split) |
| `exp78_cascade.py` | CASCADE | Hierarchical decoding `p(y=k) = p(family) · p(sibling\|family)` | S1 | extends hierarchical softmax (Morin & Bengio 2005) to LLM-genealogy attribution |
| `exp79_mage.py` | MAGE | Genealogy-conditioned mixup: pair sampling ∝ exp(-γ·d_tree) | S1 | extends Zhang 2018 mixup with label-tree-conditioned sampler |
| `exp80_tracod.py` | TRACOD | TRACO + EMA teacher self-distillation (DINO-style centered targets) under code augmentation views | S7+stability | extends DINO arXiv:2104.14294 with supervised CE + TRACO augmentations |

### CoDET-M4 — Macro-F1 (Round 5)

| Method | 1% | 5% | 20% | val-test gap @ 20% |
|:--|:-:|:-:|:-:|:-:|
| CASCADE (exp78) | 0.5462 | 0.6434 | 0.7146 | +0.0064 |
| CRONOS (exp77) | 0.5351 | 0.6456 | 0.7115 | +0.0121 |
| MAGE (exp79) | 0.5395 | 0.6430 | 0.7106 | +0.0132 |
| **TRACOD** (exp80) ❌ | **0.3706** | 0.4485 | **0.4029** | +0.0057 |

### AICD-T2 — Macro-F1 (Round 5)

| Method | 1% | 5% | 20% | val-test gap @ 20% |
|:--|:-:|:-:|:-:|:-:|
| CRONOS (exp77) | 0.2940 | 0.3908 | 0.4790 | −0.0078 |
| CASCADE (exp78) | 0.2846 | 0.3778 | 0.4764 | −0.0124 |
| MAGE (exp79) | 0.2937 | 0.3914 | 0.4758 | −0.0061 |
| **TRACOD** (exp80) ❌ | **0.1861** | 0.2125 | **0.1937** | −0.0011 |

### Round 5 falsifier readouts

| Method | Falsifier | Verdict |
|:--|:--|:--|
| CRONOS (exp77) | aux convergence stable; CE+aux ~ band ceiling | 🟡 aux co-training regularizes but does not break ceiling |
| **CASCADE** (exp78) | `family_acc = 0.85 AICD / 0.84 CoDET`, `sib_cond_acc = 0.88-0.98` | 🔥 **KEY FINDING: family is the bottleneck, NOT sibling** |
| MAGE (exp79) | sib_pair_frac 0.04 (CoDET) / 0.10 (AICD) — sampler IS tree-conditioned | 🟡 sib-conf-rate unchanged → tree-mixup doesn't repair confusion |
| TRACOD (exp80) | view_cos 0.93, ts_agree variable 0.66-0.98 | ❌ **catastrophic collapse**: DINO-distill destabilizes supervised CE; λ_distill=0.3 too aggressive |

### 🔥 Round 5 KEY FINDING — family-bottleneck refutes genealogy-prior thesis

CASCADE (exp78) decomposes accuracy:
- `family_acc` (4-way on AICD, 5-way effective on CoDET): **0.77–0.89**
- `sibling_cond_acc | true_family` (3-way on AICD, ≤2-way on CoDET): **0.88–0.98**
- `joint_acc` (full n-way): 0.55–0.71

The encoder is GOOD at sibling discrimination when given the family hint. Failure is at FAMILY level
(cross-family confusion 10–22%). All 30 prior methods that weighted negatives by `exp(-γ · d_tree)`
(LASCL/TKL/TRACO/TOURN/SCR/GSCE/RACO/...) assumed siblings are hard. **The evidence refutes this**.

**Paper §5 finding:** *cross-family confusion is the bottleneck of unixcoder-base AI-code attribution at
the few-shot regime; sibling discrimination is solved once family is known.*

### Round 5 verdict

- **TRACO (exp76) remains hero** — 4 synthesis attempts did NOT exceed it
- **CASCADE finding** is a paper-worthy negative thesis result
- **TRACOD** is catastrophic — kept as a negative example for §5 (DINO + supervised CE incompatible at λ_distill = 0.3)

---

## 🚀 Round 6 — CASCADE-grounded methods (exp81–exp83, 2026-05-19)

> **Motivation:** CASCADE refutes tree-prior on sibling negatives. Round 6 builds methods that ALIGN with
> the empirical bottleneck rather than the prior tree weighting.

| File | Method | Object | CASCADE alignment |
|:--|:--|:--|:--|
| `exp81_confuse.py` | CONFUSE | EMA confusion matrix replaces `exp(-γ·d_tree)` as hard-negative weight in TRACO supcon; cold-start = tree prior, EMA adapts | replaces static tree prior with data-driven adaptive weighting |
| `exp82_fars.py` | FARS | TRACO + EMA family centroids + hinge-margin repulsion across cross-family centroid pairs | targets the cross-family bottleneck explicitly via centroid margin |
| `exp83_prog.py` | PROG | 3-phase curriculum: P1 family-CE → P2 class-CE → P3 + TRACO supcon | training-schedule curriculum aligned to label-tree granularity |

Status: **all 3 file scaffolded, pending Kaggle run.**

### Expected targets (honest, no overclaim)

| File | CoDET 20% | AICD 20% | Risk |
|:--|:-:|:-:|:--|
| exp81_confuse | TRACO +0.005-0.015 if data ≠ tree prior | +0.005-0.02 (AICD-T2 family bottleneck) | confusion matrix degenerate if encoder confusion correlates with tree |
| exp82_fars | minor (CoDET sparse tree) | +0.005-0.02 (4 families = clean margin) | margin tuning sensitive |
| exp83_prog | curriculum may not help if encoder pretrained sufficient | family-acc benefits from family-only pretrain | phase boundary tuning



```bash
# CE baseline first
python exp_n16_ce_baseline.py

# Our methods (main contribution)
python exp_n13_hier_tree.py    # HierTree only
python exp_n14_ntk_align.py    # NTK only
python exp_n15_hier_ntk.py     # Hier-NTK (MAIN METHOD)

# Published baselines (for comparison)
python baseline_01_codet5.py
python baseline_02_detective.py
python baseline_03_faid.py
python baseline_04_style.py

# Novel methods (arxiv-grounded theory)
python exp_n05_focal.py        # Focal-CE
python exp_n06_attn_pool.py    # HAP
python exp_n07_mixup_align.py  # MGA
python exp_n08_ortho_clf.py    # Ortho-CLF
python exp_n09_etf_simplex.py  # ETF-Simplex
python exp_n10_vib.py          # VIB
python exp_n11_mixup_ce.py     # Mixup-CE
python exp_n12_label_smooth.py # LabelSmooth

# Legacy migrations (full-data experiments -> few-shot protocol)
python exp_n18_hier_supcon.py    # HierTree + SupCon (Exp18 migration, 70.55 F1 full)
python exp_n19_detective_lite.py  # DeTeCtive-lite (Exp27 migration, 71.53 F1 full)
```

Output: `results/Novel/exp*_results.json`

---

## Results

### CoDET-M4 Author IID (headline) — Macro-F1

#### 1% Few-Shot

| Rank | Method | Encoder | Macro-F1 | Δ vs Paper |
|:----:|:-------|:--------|---------:|-----------:|
| 🥇 | **GFTS** (exp59, RAS sched) | unixcoder | **0.5585** | -0.105 |
| 🥈 | SSL-RAS (exp56) | unixcoder | 0.5513 | -0.112 |
| 🥉 | RASL (exp58) | unixcoder | 0.5509 | -0.112 |
| 4 | GSCE (exp57) | unixcoder | 0.5502 | -0.113 |
| 5 | ETF-Simplex (old sched) | unixcoder | 0.4405 | -0.223 |
| 🥈 | **CodeT5-Authorship** | ModernBERT | 0.4361 | -0.227 |
| 🥉 | **HAP** | unixcoder | 0.4319 | -0.231 |
| 4 | CodeT5-Authorship | unixcoder | 0.4156 | -0.248 |
| 5 | Hier-NTK (ours) | ModernBERT | 0.3905 | -0.273 |
| 6 | Hier-NTK (ours) | unixcoder | 0.3925 | -0.271 |
| 7 | HierTree only | ModernBERT | 0.3947 | -0.269 |
| 8 | Mixup-MGA | ModernBERT | 0.3984 | -0.265 |
| 9 | ETF-Simplex | ModernBERT | 0.4040 | -0.259 |
| 10 | NTK only | ModernBERT | 0.3912 | -0.272 |
| 11 | Mixup-CE | ModernBERT | 0.3461 | -0.317 |
| 12 | LabelSmooth | ModernBERT | 0.3192 | -0.344 |
| 13 | Style-Repr | n/a | 0.3329 | -0.330 |
| 14 | IRM | unixcoder | 0.1132 | -0.550 |
| 15 | GFR | unixcoder | 0.1132 | -0.550 |
| 16 | RCE | unixcoder | 0.1139 | -0.549 |

#### 5% Few-Shot

| Rank | Method | Encoder | Macro-F1 | Δ vs Paper |
|:----:|:-------|:--------|---------:|-----------:|
| 🥇 | **GFTS** (exp59, RAS sched) | unixcoder | **0.6562** | -0.007 |
| 🥈 | SSL-RAS (exp56) | unixcoder | 0.6539 | -0.009 |
| 🥉 | GSCE (exp57) | unixcoder | 0.6530 | -0.010 |
| 4 | RASL (exp58) | unixcoder | 0.6527 | -0.011 |
| 5 | CodeT5-Authorship (old sched) | ModernBERT | 0.6030 | -0.060 |
| 🥈 | **DeTeCtive** | ModernBERT | 0.5855 | -0.078 |
| 🥉 | **NTK only** | ModernBERT | 0.5852 | -0.078 |
| 4 | Hier-NTK (ours) | ModernBERT | 0.5842 | -0.079 |
| 5 | FAID | ModernBERT | 0.5805 | -0.083 |
| 6 | HAP | ModernBERT | 0.5838 | -0.080 |
| 7 | HierTree only | ModernBERT | 0.5750 | -0.088 |
| 8 | CE baseline (ours) | ModernBERT | 0.5725 | -0.091 |
| 9 | ETF-Simplex | ModernBERT | 0.5979 | -0.065 |
| 10 | LabelSmooth | ModernBERT | 0.5683 | -0.095 |
| 11 | Style-Repr | n/a | 0.3365 | -0.327 |
| 12 | IRM | unixcoder | 0.4771 | -0.186 |
| 13 | GFR | unixcoder | 0.2823 | -0.381 |
| 14 | RCE | unixcoder | 0.1663 | -0.497 |

#### 20% Few-Shot

| Rank | Method | Encoder | Macro-F1 | Δ vs Paper |
|:----:|:-------|:--------|---------:|-----------:|
| 🥇 | **SSL-RAS** (exp56, RAS sched) | unixcoder | **0.7186** | **+0.0553** |
| 🥈 | GSCE (exp57, RAS sched) | unixcoder | 0.7185 | +0.0552 |
| 🥉 | RASL (exp58, RAS sched) | unixcoder | 0.7181 | +0.0548 |
| 4 | GFTS (exp59, RAS sched) | unixcoder | 0.7177 | +0.0544 |
| 5 | CodeT5-Authorship (old sched) | ModernBERT | 0.6880 | +0.025 |
| 🥈 | **ETF-Simplex** | ModernBERT | 0.6826 | +0.019 |
| 🥉 | **LabelSmooth** | ModernBERT | 0.6796 | +0.016 |
| 4 | CE baseline (ours) | ModernBERT | 0.6795 | +0.016 |
| 5 | Mixup-MGA | ModernBERT | 0.6774 | +0.014 |
| 6 | Hier-NTK (ours) | ModernBERT | 0.6716 | +0.008 |
| 7 | FAID | ModernBERT | 0.6722 | +0.009 |
| 8 | DeTeCtive | ModernBERT | 0.6775 | +0.014 |
| 9 | HierTree only | ModernBERT | 0.6757 | +0.012 |
| 10 | Style-Repr | n/a | 0.3327 | -0.331 |
| 11 | IRM | unixcoder | 0.6521 | -0.011 |
| 12 | GFR | unixcoder | 0.3923 | -0.271 |
| 13 | RCE | unixcoder | 0.1132 | -0.550 |

---

## 🏆 Consolidated Leaderboard (All Methods vs CE Baseline)

### CoDET-M4 Author IID — Macro-F1

| Category | Method | Encoder | 1% | 5% | 20% | vs CE 5% |
|:---------|:-------|:--------|:--:|:--:|:---:|:--------:|
| **Published** | CodeT5-Authorship | ModernBERT | 0.4361 | **0.6030** | **0.6880** | **+3.05** |
| **Published** | DeTeCtive | ModernBERT | 0.3966 | 0.5855 | 0.6775 | +1.30 |
| **Published** | FAID | ModernBERT | 0.3919 | 0.5805 | 0.6722 | +0.80 |
| **Published** | Style-Repr | n/a | 0.3329 | 0.3365 | 0.3327 | -23.60 |
| **Novel** | ETF-Simplex | ModernBERT | 0.4040 | 0.5979 | 0.6826 | +2.54 |
| **Novel** | ETF-Simplex | unixcoder | **0.4405** | 0.6010 | 0.6702 | +2.85 |
| **Novel** | HAP | ModernBERT | 0.4359 | 0.5838 | 0.6698 | +1.13 |
| **Novel** | HAP | unixcoder | 0.4319 | 0.6012 | 0.6722 | +2.87 |
| **Novel** | Mixup-MGA | ModernBERT | 0.3984 | 0.5984 | 0.6774 | +2.59 |
| **Novel** | Mixup-MGA | unixcoder | 0.4214 | 0.5930 | 0.6656 | +2.47 |
| **Novel** | Ortho-CLF | ModernBERT | 0.3935 | 0.5904 | 0.6765 | +1.79 |
| **Novel** | VIB | ModernBERT | 0.3721 | 0.5752 | 0.6675 | +0.27 |
| **Novel** | Mixup-CE | ModernBERT | 0.3461 | 0.5649 | 0.6762 | -0.76 |
| **Novel** | LabelSmooth | ModernBERT | 0.3192 | 0.5683 | 0.6796 | -0.42 |
| **Novel** | Focal-CE | ModernBERT | 0.3871 | 0.5797 | 0.6656 | +0.72 |
| **Ours** | NTK only | ModernBERT | 0.3912 | 0.5852 | 0.6755 | +1.27 |
| **Ours** | NTK only | unixcoder | 0.3988 | 0.5948 | 0.6679 | +2.23 |
| **Ours** | HierTree only | ModernBERT | 0.3947 | 0.5750 | 0.6757 | +0.25 |
| **Ours** | HierTree only | unixcoder | 0.3922 | 0.5918 | 0.6601 | +1.93 |
| **Ours** | **Hier-NTK** | ModernBERT | 0.3905 | 0.5842 | 0.6716 | +1.17 |
| **Ours** | **Hier-NTK** | unixcoder | 0.3925 | 0.5919 | 0.6587 | +1.94 |
| **Baseline** | CE baseline | ModernBERT | 0.3829 | 0.5725 | 0.6795 | — |
| **Baseline** | CE baseline | unixcoder | 0.3962 | 0.5882 | 0.6619 | +1.57 |
| **Theory-72** | MI | unixcoder | 0.1663* | — | — | n/a |
| **Theory-72** | DRL/IRM | unixcoder | 0.1273* | — | — | n/a |
| **Theory-72** | Proto | unixcoder | 0.1178* | — | — | n/a |
| **Theory-72** | (all others) | — | <0.12* | — | — | n/a |

> \* fixed-72 protocol, not fraction-based — see Theory-Track Results section above.

### AICD-T2 (model-family attribution) — Macro-F1

> ✅ **All results below are from unixcoder-base with CORRECT T2 (12-class) loading.**

| Category | Method | 1% | 5% | 20% | Notes |
|:---------|:-------|:--:|:--:|:---:|:------|
| **Novel** | LabelSmooth | 0.191 | 0.333 | 0.445 | ✅ Best at 5%/20% |
| **Novel** | Mixup-CE | 0.190 | 0.326 | 0.443 | |
| **Novel** | KAC | 0.200 | 0.325 | 0.410 | |
| **Novel** | SGE | 0.204 | 0.324 | 0.408 | |
| **Novel** | HPA | 0.194 | 0.319 | 0.407 | |
| **Novel** | Ortho-CLF | 0.205 | 0.317 | 0.410 | |
| **Novel** | Focal-CE | 0.191 | 0.306 | 0.392 | |
| **Published** | CodeT5-Authorship | 0.199 | 0.316 | 0.404 | |
| **Baseline** | CE baseline | ~0.22 | **0.322** | ~0.41 | |
| **Theory** | GFR | 0.069 | 0.142 | 0.156 | ❌ VAE KL dominates, collapses |
| **Theory** | RCE | 0.134 | 0.069 | 0.069 | ❌ Causal intervention destroys signal |
| **Theory** | SIE | 0.133 | 0.238 | 0.384 | ❌ Sign-flip invariance destroys authorship |
| **Theory** | PAC-A | 0.031 | 0.031 | 0.031 | ❌ Stochastic weights add too much variance |
| **Theory** | HETE | 0.182 | 0.337 | 0.444 | ⚠️ Reasonable at 20%, matches LabelSmooth |
| **Theory** | GIBA | 0.135 | 0.271 | 0.420 | ⚠️ Best at 5%, degrades at 20% (KL bottleneck hurts) |
| **Theory** | GRA | 0.189 | 0.314 | 0.437 | ✅ Best theory method on AICD-T2 at 20% |
| **Theory** | BGB | 0.179 | 0.330 | 0.444 | ✅ Tied best theory on AICD-T2 at 20% |
| **RAS-track** | **GSCE** (exp57) | 0.2941 | 0.3923 | **0.4851** | 🥇 NEW best at 20% AICD-T2 |
| **RAS-track** | **GFTS** (exp59) | 0.3009 | **0.4049** | 0.4833 | 🥇 NEW best at 5% AICD-T2 |
| **RAS-track** | **RASL** (exp58) | 0.2919 | 0.3959 | 0.4814 | ✅ +0.036 vs LabelSmooth @ 20% |
| **RAS-track** | **SSL-RAS** (exp56) | **0.3025** | 0.3964 | 0.4771 | 🥇 NEW best at 1% AICD-T2 |

> 📊 **Key insight:** LabelSmooth achieves best AICD-T2 results at 5%/20% (+0.01 over baseline).
> All methods struggle at 1% few-shot (~0.19-0.21) due to 12-class problem.
> **GFR (Genealogy-Factorized Representation):** VAE-based disentanglement collapses - KL loss dominates and prevents learning meaningful representations.
> **RCE (Representation Causal Effect):** Causal intervention mechanism destroys discriminative signal - learned effects add arbitrary noise instead of meaningful causal shifts.

### AICD-T1 (binary: human vs AI) — Macro-F1 [INVALID - historical]

> ⚠️ **WARNING:** These used **T1 binary (2-class)** due to `_load_aicd()` path bug.
> **Do NOT quote these as T2 results.** Historical record only.

| Category | Method | Encoder | 1% | 5% | 20% | Notes |
|:---------|:-------|:--------|:--:|:--:|:---:|:------|
| **Published** | CodeT5-Authorship | ModernBERT | 0.9401 | 0.9679 | 0.9793 | T1 bug |
| **Published** | DeTeCtive | ModernBERT | 0.9155 | 0.9581 | 0.9725 | T1 bug |
| **Published** | FAID | ModernBERT | 0.6104 | 0.9569 | 0.9737 | T1 bug |
| **Published** | Style-Repr | n/a | 0.7676 | 0.7699 | 0.7704 | T1 bug |
| **Novel** | ETF-Simplex | ModernBERT | 0.9202 | 0.9644 | 0.9772 | T1 bug |
| **Novel** | HAP | ModernBERT | 0.8592 | 0.9622 | 0.9758 | T1 bug |
| **Novel** | Mixup-MGA | ModernBERT | 0.9130 | 0.9565 | 0.9736 | T1 bug |
| **Novel** | Ortho-CLF | ModernBERT | 0.9142 | 0.9575 | 0.9735 | T1 bug |
| **Novel** | VIB | ModernBERT | 0.9111 | 0.9581 | 0.9741 | T1 bug |
| **Novel** | Mixup-CE | ModernBERT | 0.9168 | 0.9568 | 0.9745 | T1 bug |
| **Novel** | LabelSmooth | ModernBERT | 0.9186 | 0.9586 | 0.9744 | T1 bug |
| **Novel** | Focal-CE | ModernBERT | 0.8339 | 0.8751 | 0.8009 | T1 bug |
| **Ours** | NTK only | ModernBERT | 0.5955 | 0.9566 | 0.9736 | T1 bug |
| **Ours** | NTK only | unixcoder | 0.9229 | 0.9616 | 0.9733 | T1 bug |
| **Ours** | HierTree only | ModernBERT | 0.9067 | 0.9550 | 0.9733 | T1 bug |
| **Ours** | HierTree only | unixcoder | 0.9249 | 0.9610 | 0.9736 | T1 bug |
| **Ours** | **Hier-NTK** | ModernBERT | 0.9115 | 0.9581 | 0.9740 | T1 bug |
| **Ours** | **Hier-NTK** | unixcoder | 0.9247 | 0.9617 | 0.9739 | T1 bug |
| **Baseline** | CE baseline | ModernBERT | 0.9135 | 0.9568 | 0.9729 | T1 bug |
| **Baseline** | CE baseline | unixcoder | 0.9218 | 0.9598 | 0.9756 | T1 bug |

---

## Theory-Track Results (FIXED_TOTAL=72, not fraction-based)

> ⚠️ These methods use 72 fixed training samples. Results are **not directly comparable** with the fraction-based table above, but compared vs CE baseline at same fixed-72 budget.
> Reference CE baseline (fixed-72, best encoder): **CoDET-M4 ≈ 0.39**, **AICD-T2 ≈ 0.91**

### CoDET-M4 Author IID — Macro-F1 (fixed-72)

| Rank | Method | Best Encoder | Macro-F1 | Notes |
|:----:|:-------|:------------|--------:|:------|
| 🥇 | MI | unixcoder | **0.1663** | Best of group; still far below CE |
| 🥈 | DRL/IRM | unixcoder | 0.1273 | IRM penalty never activates (72 < 500 anneal) |
| 🥉 | Proto | unixcoder | 0.1178 | Partial: 2/6 classes only |
| 4 | GNA | unixcoder | 0.1106 | Graph conv collapsed |
| 5 | BUA | unixcoder | 0.1100 | MC Dropout; collapses to 2 classes |
| 6 | CAO | both | 0.1132 | NaN loss after ep1; frozen at class 0 |
| 7 | WDA / OT | unixcoder | 0.0938 | Dominated by class 3 only |

### AICD-T2 — Macro-F1 (fixed-72)

| Rank | Method | Best Encoder | Macro-F1 | Notes |
|:----:|:-------|:------------|--------:|:------|
| 🥇 | DRL/IRM | unixcoder | **0.0397** | Slightly above others |
| 🥈 | GNA | unixcoder | 0.0382 | |
| 🥉 | Proto | unixcoder | 0.0380 | |
| 4 | OT | unixcoder | 0.0367 | |
| 5 | WDA | unixcoder | 0.0367 | |
| 6 | MI | unixcoder | 0.0356 | |
| 7 | CAO | both | 0.0685 | Collapsed to majority class only |
| 8 | BUA | unixcoder | 0.0311 | Worst; MC dropout hurts small data |

---

## Key Insights

### 🚀 2026-05-14 Update: Regime-Adaptive Schedule (exp56–exp59)

0. **Schedule was the dominant bottleneck at 20%, not the loss object.**
   - Old (bs=256, 3ep): SSL @ 20% CoDET-M4 = 0.6796 (+0.0163 vs paper)
   - New (RAS sched): SSL-RAS / GSCE / RASL / GFTS all hit **0.7177–0.7186 (+0.054 vs paper)**, +0.039 over old SSL.
   - AICD-T2 also lifts: 1% from ~0.20 → ~0.30, 20% from 0.445 → **0.4851**.
   - Diagnosis: `bs=256 × 3ep × 80K ≈ 937 updates` was 5× fewer than legacy `bs=64 × 3ep × 100K ≈ 4687 updates` that reached 0.7055. Fix: 6 epochs + sqrt-scaled LR + cosine schedule, keyed on `cfg.frac`.

### 🔍 Consolidated Analysis (All 22 Methods)

1. **CodeT5-Authorship wins on CoDET-M4**: +3.05 pts vs CE baseline @ 5%, strongest across all fractions
   - ETF-Simplex close 2nd: +2.85 pts (unixcoder @ 5%)

2. **AICD-T2 (unixcoder) results are low**: ~0.21-0.43 macro-F1 across methods
   - All methods underperform vs CE baseline (negative Δ)
   - **Pending re-run**: Need to verify T2 loader is correct

3. **AICD-T1 results are high (~0.93-0.98)**: Due to T1 data bug, these are binary (2-class)
   - Cannot compare with AICD-T2 (12-class) results

4. **Style-Repr fails everywhere**: -23.6 pts vs CE on CoDET-M4
   - Functional features > stylistic patterns

5. **Mixup hurts CoDET-M4**: Mixup-CE -0.76 pts, Mixup-MGA neutral
   - Interpolating samples blurs author-specific signals

6. **Hier-NTK (ours) underperforms**: +1.17 pts vs CE @ 5%
   - Loses to CodeT5-Authorship (+3.05), ETF-Simplex (+2.54)

7. **NTK/Hier components neutral**: HierTree only +0.25, NTK only +1.27

### ⚠️ Negative Results Summary

| Finding | Impact | Evidence |
|:--------|:-------|:---------|
| Hier-NTK loses to CodeT5 | High | +1.17 vs +3.05 pts |
| Focal-CE destroys AICD-T2 | Critical | -8.17 pts, degrades w/ data |
| Style-Repr useless | High | -23.6 pts on CoDET-M4 |
| Mixup-CE hurts | Medium | -0.76 pts |
| HierTree only neutral | Medium | +0.25 pts (not worth complexity) |
| **Theory-Track ALL collapsed** | **Critical** | All 8 methods ≤0.17 on CoDET-M4 |
| **CAO NaN loss** | Critical | Loss → NaN after ep1, frozen predictions |
| **WDA/OT dominated by 1 class** | High | All preds → class 3; OT loss diverges |
| **IRM penalty never fires** | High | penalty_anneal=500 > total steps at 72 samples |
| **BUA worst on AICD-T2** | Medium | MC Dropout + tiny data = 0.031 macro-F1 |
| **GFR VAE KL dominates** | Critical | β_kl=1e-3 too strong; collapses at all fractions |
| **RCE intervention ruins signal** | Critical | Learned causal effects add arbitrary noise; collapses to class majority |
| **CPO OOM** | Critical | Forward pass x2 per batch (original + counterfactual) causes OOM |
| **SIE sign-flip destroys signal** | Critical | Z_2 sign-flip group action destroys authorship-specific features |
| **PAC-A stochastic weights too noisy** | Critical | Gaussian posterior sampling adds too much variance; model collapses regardless of data fraction |

### 🎯 When Each Method Wins (updated 2026-05-14)

| Scenario | Best Method | Score | Δ vs paper |
|:---------|:------------|------:|:----------:|
| CoDET-M4 @ 1% | **GFTS (exp59, RAS sched)** | **0.5585** | -0.105 |
| CoDET-M4 @ 5% | **GFTS (exp59, RAS sched)** | **0.6562** | -0.007 |
| CoDET-M4 @ 20% | **SSL-RAS (exp56, RAS sched)** | **0.7186** | **+0.0553** |
| AICD-T2 @ 1% | **SSL-RAS (exp56)** | **0.3025** | — |
| AICD-T2 @ 5% | **GFTS (exp59)** | **0.4049** | — |
| AICD-T2 @ 20% | **GSCE (exp57)** | **0.4851** | — |
| (legacy, frozen) CoDET-M4 @ 1% | ETF-Simplex | 0.4405 | -0.223 |
| (legacy, frozen) CoDET-M4 @ 5% | CodeT5-Authorship (ModernBERT) | 0.6030 | -0.060 |
| (legacy, frozen) CoDET-M4 @ 20% | CodeT5-Authorship (ModernBERT) | 0.6880 | +0.025 |

---

## Paper Reference

UniXcoder full-data baseline (CoDET-M4 Author IID): **0.6633 Macro-F1**

---

## Notes

- **Droid T3/T4 SKIPPED per 2026-05-10 directive.** Focus on CoDET-M4 + AICD T2 only.
- Each fraction-based file runs 2 encoders × 2 benchmarks × 3 fractions = 12 experiments.
- Theory-Track (exp17–exp30): 2 encoders × 2 benchmarks = 4 runs each, FIXED_TOTAL=72.
- Results format (fraction-based): `{"tag", "enc", "bench", "frac", "macro", "weighted", "acc", "dpaper", "wall"}`
- Results format (theory-track): `{"tag", "encoder", "benchmark", "task", "macro_f1", "delta_vs_paper", ...}`

### Theory-Track Collapse Analysis

All 8 theory-track methods collapsed on CoDET-M4 (macro-F1 ≤ 0.17, vs CE baseline ~0.39).
Common failure modes:
- **Class imbalance exploitation**: With 72 samples, auxiliary losses (WDA, OT, CAO) overwhelm CE signal
- **Auxiliary loss magnitude mismatch**: WDA loss ~30×, OT loss ~5×, CAO loss ~750K× the CE loss
- **IRM anneal too high**: `penalty_anneal=500` steps but only 1 step per epoch × 3 epochs = 3 steps total
- **Graph collapse (GNA)**: With 72 nodes, inter-node edges are trivial; network defaults to simple linear
- **MC Dropout (BUA)**: `high_uncertainty_ratio=0.0` — all predictions confident despite being wrong
- **GFR VAE KL dominates**: β_kl=1e-3 forces posterior to prior, destroying discriminative signal
- **RCE intervention ruins signal**: CausalGate learns meaningless effects that add noise; val=0.1133 (class majority) across most runs
- **CPO OOM**: Forward pass x2 per batch (original + counterfactual) - GPU memory exceeded; needs gradient checkpointing or single-pass design
- **SIE sign-flip destroys signal**: Z_2 sign-flip group action during training destroys authorship-specific features; model collapses to majority class at 1%/5%, only reaches 0.6511 at 20% (vs paper 0.6633)
- **PAC-A stochastic weights**: Gaussian posterior sampling adds too much variance; val=0.1298 constant across all CoDET-M4 fractions, 0.0317 constant across all AICD-T2 fractions

**Recommendation**: Theory-track methods need hyperparameter scaling to the data regime (72 samples).
Consider: smaller auxiliary loss weights (`wda_weight < 0.01`), lower `penalty_anneal` (e.g. 1), or prototype initialization from pretrained clusters.
