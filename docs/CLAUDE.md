# CLAUDE.md — Full Repo Context

> **Read this file first.** One-stop context for any AI assistant working in this repo.
> If something here disagrees with the code, trust the code and update this file.

---

## 1. What this project is

**Goal:** Submit an **AI-generated code detection** paper to **NeurIPS 2026** (oral target).

**Core thesis:** State-of-the-art detectors achieve near-perfect accuracy in-distribution but **collapse under OOD shift** (new languages, domains, generators). We design methods that target this gap via style-content disentanglement, multi-granularity representations, and cross-domain generalization techniques.

**Flagship method name:** `CodeOrigin` — multi-granularity (token + AST + graph) encoder with style-content disentanglement and hierarchical prototypical contrastive heads.

---

## 2. Benchmarks (three in play)

All datasets live on Hugging Face. Links: [references/dataset_links.md](references/dataset_links.md).

| Short name | HF ID | Size | Primary metric | Role |
|---|---|---|---|---|
| **AICD-Bench** | `AICD-bench/AICD-Bench` | ~2.1M (T1) + T2 + T3 | **Macro-F1** | Hard OOD benchmark (val-test gap is the boss fight) |
| **DroidCollection** | `project-droid/DroidCollection` | ~1M+ | **Weighted-F1** (report Macro-F1 as aux) | Adversarial + co-authored + multi-domain |
| **CoDET-M4** | `DaniilOr/CoDET-M4` | ~500K | Macro-F1 | ACL 2025 baseline; cross-language/generator |

**Per-benchmark tasks:**
- AICD: T1 (binary), T2 (12-class family attribution), T3 (4-class fine-grained: human/machine/hybrid/adversarial)
- Droid: T1 (binary), T3 (3-class: human/generated/refined), T4 (variant)
- CoDET-M4: `binary` (human vs machine), `author` (human + generator attribution, 6-class)

**Protocol (non-negotiable):**
- **Never** mix benchmarks in one training run. Train separate model per benchmark.
- Always export full metric pack: `Primary`, `Macro-F1`, `Weighted-F1`, `Macro-Recall`, `Weighted-Recall`, `Accuracy`, `per-class report`.
- Hardware target: **Kaggle H100 80GB, BF16, effective batch 64**.
- Each experiment file is **standalone** and Kaggle-runnable (`python expXX_name.py`).

---

## 3. Repo layout (post-cleanup)

```
ai_code_detection/
├── README.md                          # Public-facing project description
├── experiment/                        # Baseline: exp00_codeorigin.py (the starting point)
│   ├── exp00_codeorigin.py            # Full CodeOrigin implementation (Kaggle single-file)
│   ├── eda_two_benches.py
│   └── exp01..04_*.py                 # Early baselines (simple, stylometric, frozen-DRO, multiview)
├── Exp_DM/                            # "Deep Methods" experiment suite — novel methods on AICD + Droid
│   ├── dm_tracker.md                  # THE tracker for AICD + Droid results. Start reading here.
│   └── exp01..30_*.py                 # 30 standalone experiments (see §4)
├── Exp_CodeDet/                       # CoDET-M4 experiment suite (mirror of Exp_DM methods)
│   ├── tracker.md                     # THE tracker for CoDET-M4 results.
│   ├── README.md                      # Runner-specific notes
│   └── run_codet_m4_exp11..38_*.py    # Parallel runners for CoDET-M4
├── docs/
│   ├── CLAUDE.md                      # This file
│   ├── performance_tracker.md         # Legacy AICD tracker (superseded by Exp_DM/dm_tracker.md)
│   ├── eda_two_benches.json           # EDA output
│   ├── references/                    # Source papers + research notes (read-only)
│   │   ├── paper_AICD.md              # AICD-Bench paper (Orel et al.)
│   │   ├── paper_Droid.md             # DroidCollection paper (Orel et al.)
│   │   ├── paper_CodeDet_M4.md        # CoDET-M4 paper (ACL 2025)
│   │   ├── gemini_deepresearch_round1.md  # 30-method ideation (source of Exp_DM methods)
│   │   ├── dataset_links.md           # HF dataset URLs
│   │   └── dataset_schema_AICD.md     # AICD schema dump
│   └── archive/                       # Completed session artifacts (historical)
│       ├── implementation_plan.md
│       ├── task.md
│       └── walkthrough.md
├── Slide/                             # Proposal deck
│   ├── proposal.tex / proposal.pdf
│   └── script_thuyet_trinh.md
└── Formatting_Instructions_For_NeurIPS_2026/   # NeurIPS 2026 LaTeX template (do not edit)
```

**Single sources of truth:**
- Results → [Exp_DM/dm_tracker.md](../Exp_DM/dm_tracker.md) (AICD + Droid), [Exp_CodeDet/tracker.md](../Exp_CodeDet/tracker.md) (CoDET-M4)
- Baseline code → [experiment/exp00_codeorigin.py](../experiment/exp00_codeorigin.py)
- Methods → individual files in `Exp_DM/` and `Exp_CodeDet/`

---

## 4. Experiment registry (high-level)

All Exp_DM methods build on `exp00_codeorigin.py` and add one novel component. Each has a matching `Exp_CodeDet/run_codet_m4_exp*` runner where applicable.

| Exp | Method | Core innovation | Status |
|---|---|---|---|
| 01 | CausAST | Orthogonal cov penalty between token & AST views + batch-hard triplet | AICD/Droid done |
| 02 | TTA-Evident | Evidential DL (Dirichlet) + test-time MLM adaptation | Pending |
| 03 | AP-NRL | SupCon + code-humanization augmenter + dual-view consistency | AICD/Droid done |
| 04 | BH-SCM | Batch-hard triplet on multi-view + cross-view consistency | AICD/Droid done |
| 05 | OSCP | Frobenius ortho penalty + VILW whitening + SupCon | AICD/Droid done |
| 06 | AST-IRM | IRM across language environments, annealed penalty | AICD/Droid done |
| 07 | DomainMix | Comment/var normalization + embedding mixup | 5-run suite done |
| 08 | MoE-Code | Mixture-of-Experts head (K=4, top-2) + load balance | 5-run suite done |
| 09 | TokenStat | Entropy/burstiness/TTR/Yule-K + neural dual-head gate | 5-run suite done |
| 10 | MetaDomain | Reptile meta-learning, languages-as-tasks | Pending |
| 11 | **SpectralCode** | FFT multi-scale frequency features | **5-run done; current best overall AICD+Droid avg** |
| 12 | WatermarkStat | Watermark-inspired (green-list, n-gram entropy, Zipf, chi-sq) | Pending |
| 13 | SlotCode | Slot attention decomposes code into K structural slots | 5-run suite done |
| 18 | **HierTreeCode** | Hierarchical affinity constraint + spectral-neural backbone | **5-run done; best AICD T2; best CoDET-M4 author** |
| 19 | KANCode | KAN (B-spline learnable activations) replace MLP heads | Pending |
| 20 | HyperCode | Poincaré ball hyperbolic embeddings | Pending |
| 21 | IBCode | Variational Information Bottleneck (anti-shortcut) | Pending |
| 22 | TTLCode | Test-time LoRA via MLM on unlabeled batches | Pending |
| 23 | TopoCode | Topological persistence (Betti H0/H1) from AST filtration | Pending |
| 24 | MambaCode | Selective SSM + MoE routing, O(n) fusion | Pending |
| 25 | EnergyCode | Energy-based OOD with pseudo-OOD noise injection | Pending |
| 26 | WaveCLCode | Discrete Wavelet Transform (Haar) + class-aware bands | Pending |
| 27 | DeTeCtiveCode | exp18 + multi-level SupCon + kNN blend at test | 3-bench unified |
| 28–30 | HardNeg / RetrievalCalib / HierFocus | exp27 variants | Pending |

**CoDET-M4-only experiments:** Exp_CodeDet has extra runners (exp11, 14–38) that adapt these methods to the CoDET-M4 leaderboard. Ranked leaderboard sits in [Exp_CodeDet/tracker.md](../Exp_CodeDet/tracker.md).

Method details (loss formulas, hyperparams, key claims) → see [Exp_DM/dm_tracker.md](../Exp_DM/dm_tracker.md) §Method Details.

---

## 5. Key results & open problems

### Current standing
- **AICD T1 (OOD binary)**: all methods show catastrophic val→test collapse (val ~0.99, test ~0.25–0.31). **No method solves it yet.** This is the single most important bottleneck.
- **AICD T2/T3**: HierTreeCode best on T2 (0.2071); SpectralCode/SlotCode competitive on T3.
- **Droid T3/T4**: stable ~0.85–0.89 weighted-F1; TokenStat edges out others.
- **CoDET-M4 author**: HierTreeCode 70.55 F1 beats paper's UniXcoder (66.33) by **+4.22**. Binary 99.06 vs paper 98.65 (+0.41).

### Baseline reference
| Benchmark | Task | exp00 Val F1 | exp00 Test F1 | Gap |
|---|---|---|---|---|
| AICD | T1 | 0.9954 | 0.2877 | **Severe OOD collapse** |
| Droid | T1 | 0.9693 | 0.9708 | Stable ID |

### The central question
> Why do AICD T1 detectors memorize val but fail on test? The train/val come from the same distribution; test has unseen languages/domains/generators. Methods that target shortcut learning (DomainMix, IRM, IB, orthogonal disentanglement) have all been tried and none close the gap materially.

Any new method proposal should state explicitly how it attacks this collapse.

---

## 6. How to work in this repo

### Running an experiment
```bash
# Edit constants at top of file (TASK, BENCHMARK, RUN_MODE), then:
python Exp_DM/exp11_spectral_code.py
python Exp_CodeDet/run_codet_m4_exp18_hiertree.py
```
- Each file is self-contained. No shared package imports across experiments.
- Kaggle-first: scripts assume H100 BF16. They auto-install `tree-sitter`, `tree-sitter-languages`.
- Results are written to local `logs/`, `results/`, `codet_m4_checkpoints/` (all gitignored).

### Adding a new method (e.g., exp39)
1. Copy the closest existing `Exp_DM/expNN_*.py` as starting skeleton.
2. Change the model class + loss formula only. Keep the training loop, preflight, and metric export identical.
3. If it needs a CoDET-M4 counterpart, mirror the file as `Exp_CodeDet/run_codet_m4_exp39_*.py`.
4. After a run finishes, append results to the two tracker markdowns (see §3).

### Conventions
- Experiment IDs never reused.
- Tracker tables always append; never rewrite historical rows.
- Val-test gap is always reported alongside test metric — this is the paper's headline signal.
- Primary metric is benchmark-specific (Macro-F1 for AICD/CoDET-M4, Weighted-F1 for Droid). Do not substitute.

---

## 7. Paper artifacts

- **Draft paper** (Markdown): [paper_AICD.md](references/paper_AICD.md) is the AICD benchmark paper we cite, NOT our draft. Our own draft lives in the NeurIPS template folder once written.
- **Proposal slides**: [Slide/proposal.tex](../Slide/proposal.tex) + [Slide/proposal.pdf](../Slide/proposal.pdf)
- **Vietnamese presentation script**: [Slide/script_thuyet_trinh.md](../Slide/script_thuyet_trinh.md)
- **NeurIPS 2026 template**: [Formatting_Instructions_For_NeurIPS_2026/](../Formatting_Instructions_For_NeurIPS_2026/) (do not modify .sty)

---

## 8. Things that will waste your time if you don't know

- `docs/performance_tracker.md` is an older tracker — has been superseded by `Exp_DM/dm_tracker.md`. If they conflict, trust `dm_tracker.md`.
- `experiment/` contains early baselines (exp00–exp04). The active experiment suite is `Exp_DM/`, not `experiment/`. Only `exp00_codeorigin.py` is still load-bearing.
- Some `Exp_DM/` files (exp28, 29, 30) are tiny (~1.6KB) — they import from exp27 with different hyperparams. Do not treat their size as a bug.
- `.claude/`, `.cursor/`, `codet_m4_checkpoints/`, `*.pt`, `logs/`, `results/` are all gitignored. Don't commit them.
- Large negative delta on AICD T1 test is expected, not a bug — it's the entire research motivation.

---

## 9. Checklist for any AI opening this repo cold

1. Read this file.
2. Scan [Exp_DM/dm_tracker.md](../Exp_DM/dm_tracker.md) for the latest experiment table and method notes.
3. Scan [Exp_CodeDet/tracker.md](../Exp_CodeDet/tracker.md) for CoDET-M4 leaderboard.
4. Before proposing a new method, check if it (or a variant) already exists in Exp_DM or Exp_CodeDet.
5. Before running anything, confirm the user's target benchmark (AICD vs Droid vs CoDET-M4) — the answer changes the primary metric, the script, and the tracker file.
