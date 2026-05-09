# CLAUDE.md

> Root entry point for Claude Code. Full repo context is in [docs/CLAUDE.md](docs/CLAUDE.md) — read it before doing anything non-trivial.

@docs/CLAUDE.md

---

## ⏱️ TL;DR — current state (2026-05-09 night, after 31 Kaggle runs)

- **Submission target:** EMNLP 2026 Main, **reach for Oral**. Deadline ~2026-05-26 (~17 days).
- **🏆 BREAKTHROUGH:** **4 of our methods exceed paper UniXcoder full-data 0.6633 at 5% training data** (≈ 1/20 the budget). Top 5 at fraction=0.05 on CoDET-M4 6-class Author IID:

  | Rank | Method | Macro-F1 | Δ vs paper |
  |:-:|:--|:-:|:-:|
  | 🥇 | **FS-Hier-NTK** (combo) | **0.6709** | **+0.0076** |
  | 🥈 | FS-HierTree (Galanti-Poggio family prior) | 0.6682 | +0.0049 |
  | 🥉 | FS-NTKAlign | 0.6652 | +0.0019 |
  | 4 | FS-Focal | 0.6616 | −0.0017 |
  | 5 | FS-Baseline-UniXcoder reimpl | 0.6512 | −0.0121 |

- **Lock the paper main method:** **Hier-NTK** (HierTree family prior + NTK target-kernel alignment). Replaces the earlier "NTKAlign-only" framing.
- **Phase transition story:** F1 jumps from 0.18 (K=32, 192 samples) → 0.28 (K=128) → 0.57 (1%, 5K samples) → **0.67 (5%, 25K samples ≈ paper SOTA)** → 0.71 (Exp_13 lean 20%, 100K samples). The "bend" is between 1% and 5% data.
- **Regime-dependent winner (insight 6+7):** at K=128 / 1% Focal & UniXcoder reimpl WIN; at 5%+ Hier/NTK WIN. Paper §3 must specify "use Hier-NTK at fraction ≥ 1%, drop NTK at K-shot regime".
- **2-bench dispatch:** every novel file accepts `FS_BENCHMARK=codet_m4 | droid_t3 | droid_t4` env var. Cross-bench runs queued.
- **Paper draft:** [Paper/latex/main.tex](Paper/latex/main.tex), 5 pages storytelling. Will be updated to feature Hier-NTK as headline method.
- **Excluded from headline:** AICD-Bench (val→test collapse), Zero-Shot suite (legacy, FDG gap).

## 📐 Two-track structure (`Exp_FewShot/`)

| Track | Folder | Filename | Gate | Purpose |
|:--|:--|:--|:--|:--|
| **Setup-variation** | `Exp_FewShot/testing/` | `exp_fs_*.py` (free) | none | Method × encoder × budget grid; paper-baseline reimpls |
| **Theory-driven** | `Exp_FewShot/novel/` | `exp_nNN_*.py` (sequential) | 5-gate Novelty Filter ([novel/README.md](Exp_FewShot/novel/README.md)) | One new mathematical object per file; oral upside |

- Both tracks: each file is fully self-contained inline; paste into Kaggle cell, run, no `git clone` needed (the file pip-installs deps and uses standard CoDET-M4 / DroidCollection HF datasets).
- Two regimes per file via env vars:
  - `FS_K_SHOT=N` → K-shot mode (1 epoch, no LR schedule)
  - `FS_TRAIN_FRACTION=f` → %-fraction mode (3 epochs, cosine LR + warmup)
- Two benchmarks via env var:
  - `FS_BENCHMARK=codet_m4` (default) | `droid_t3` | `droid_t4`
- JSON output filename encodes benchmark + regime to avoid collision.

## 🧬 Active novel methods (`Exp_FewShot/novel/`)

Each file = one new mathematical object + falsifiable claim + theorem hook.

| File | Method | Open problem | Theory hook |
|:--|:--|:--|:--|
| `exp_n01_sibling_residual.py` | SRD | K-shot codellama↔nxcode collapse | Galanti-Poggio 2025 hierarchical neural collapse |
| `exp_n02_frontdoor_style.py` | FSM | Source-confounding (CF/LC→GH) | Veitch-Wang NeurIPS 2025 front-door criterion |
| `exp_n04_etf_simplex.py` | EFS | Explicit ETF parameterisation | Galanti-Poggio + Papyan-Han-Donoho ETF |
| `exp_n05_mi_floor.py` | MIF | I(Y;X) ceiling for 6-class | Belghazi MINE ICML 2018 + Fano |
| `exp_n06_proximal_sibling.py` | PCS | Sibling pair via proxy | Mastouri-Gretton JMLR 2025 |
| `exp_n07_conformal_mondrian.py` | CMP | Per-class FNR guarantee | Vovk 2005; Romano-Patterson-Candès 2020 |
| `exp_n08_spectral_eigengap.py` | SEA | Phase-transition predictor | Cheeger 1970; Lee-Oveis-Trevisan 2014 |
| `exp_n10_vib.py` | VIB | Data-efficient regulariser | Tishby IB 2000; Alemi VIB ICLR 2017 |

## 💡 Live insights from the 31-run leaderboard (paper-ready)

(Full version + evolution timeline in [Exp_FewShot/tracker.md](Exp_FewShot/tracker.md).)

1. **HierTree family prior is the biggest signal** — alone gives 0.6682 at 5%, beats NTKAlign (0.6652).
2. **Hier+NTK combo wins** — 0.6709, additive contribution.
3. **4 methods cluster ≥ 0.6616 at 5%** — gain comes from prior + ModernBERT, not single trick.
4. **UniXcoder reimpl validates protocol** — 0.6512 at 5% = 98.2% of paper's full-data 0.6633.
5. **ModernBERT > CodeBERT > GraphCodeBERT** — encoder choice matters; ModernBERT is the right backbone.
6. **At 1% data, paper baseline (UniXcoder) wins** — encoder pretrain dominates below 5K samples.
7. **At K=128, Focal loss wins** — class-imbalance-aware loss dominates when batch lacks diverse pairs.
8. **Frozen-encoder ceiling = 0.46** — pretrained features have no generator-specific signal; encoder fine-tune is necessary.
9. **Val-test gap signs are diagnostic, not noise** — negative gap (test > val) indicates under-fitting on tiny val pool, not over-fitting.
10. **Hier+NTK COMBO HURTS at K=128** — sparse kernel target. Specify "use combo at frac ≥ 1%, NTKAlign-only at K-shot".
11. **Phase transition between 1% and 5%** — model needs ~5K samples to unlock generator-specific signal.

## Operational rules for this repo

1. **Execution lives on Kaggle, not locally.** Local machine = code + analyze. Kaggle = run. Use `/kaggle-cell` to emit cells; `/analyze-log` on pasted output. Do NOT run `python Exp_*/exp*.py` locally.
2. **Never mix benchmarks in one training run.** AICD, Droid, CoDET-M4 each get their own model.
3. **Primary metric is benchmark-specific.** Macro-F1 for AICD and CoDET-M4, Weighted-F1 for Droid. Do not substitute.
4. **Always report val-test gap** alongside test metrics.
5. **Experiment IDs are immutable.** Never reuse an `expNN` number. Never rewrite historical tracker rows; always append.
6. **Every experiment file is standalone.** No shared package imports across `Exp_DM/`, `Exp_CodeDet/`, `Exp_Climb/`, `Exp_FewShot/`.
7. **Hardware tiers:** `Exp_Climb/` H100 BF16 batch 64 · `Exp_FewShot/` T4 fp16 batch 16 seq 384 · auto-detect on both.
8. **Don't commit** `logs/`, `results/`, `codet_m4_checkpoints/`, `*.pt`, `*.bin`, LaTeX `*.aux/.log/.pdf` — gitignored for a reason.
9. **Templates are read-only.** Do not edit `Paper/latex/acl.sty` or `Formatting_Instructions_For_NeurIPS_2026/*.sty`.
10. **Headline numbers are locked.** Don't pivot scope or rerun experiments that would change Table 1/2 of the paper without explicit user OK.

## Source-of-truth files

- **Paper draft (THIS IS THE GOAL):** [Paper/latex/main.tex](Paper/latex/main.tex)
- Section outline + 18-day plan: [Paper/outline.md](Paper/outline.md)
- 20%-data climb leaderboard: [Exp_Climb/tracker.md](Exp_Climb/tracker.md)
- CoDET-M4 full-data leaderboard: [Exp_CodeDet/tracker.md](Exp_CodeDet/tracker.md)
- Few-shot K-sweep + decision gate: [Exp_FewShot/tracker.md](Exp_FewShot/tracker.md)
- AICD + Droid champions: [Exp_DM/dm_tracker.md](Exp_DM/dm_tracker.md)
- Paper references: [docs/references/](docs/references/)
- Archived suites + rationale: [legacy/README.md](legacy/README.md)

## Slash commands & agents

Custom tooling lives in [.claude/](.claude/). See [.claude/README.md](.claude/README.md) for what's available.
