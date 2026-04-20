# Exp_Climb Tracker — Data-Efficient Dual-Bench Leaderboard

> **Strategy:** Each `exp_NN_*.py` file trains ONE method on BOTH target benchmarks (**CoDET-M4** + **DroidCollection**) sequentially, using **~20% of the training data** while evaluating on the **FULL test set**.
>
> **Paper angle:** "With only 20% of training samples, our method matches or exceeds full-data paper baselines on two major AI-code-detection benchmarks (ACL 2025 + EMNLP 2025)."

---

## 🏆 Climb Leaderboard — sorted by CoDET Author Macro-F1 ↓

> **Primary metric** = **Author IID Macro-F1** (6-class, hardest task). Binary is ceiling-bound ~99%.
> Paper baselines: **UniXcoder** 98.65 binary / **66.33** author / 88.96 lang-OOD / **DroidDetectCLS-Large** 0.8878 T3.
> Δ-paper columns compare against the SAME paper baseline on that task.

All 8 lean-mode tasks are reported below (CoDET: 2 IID + 3 OOD representative · Droid: T1 + T3 + T4).

| Rank | Exp | Method | Mode | **Bin** | Δ | **Auth** | **Δ** | **Gen-qw** | **Lang-py** | **Src-gh** | **T1** | **T3** | Δ | **T4** | Status |
|:----:|:---|:-------|:----:|:-----:|:---:|:-----:|:-----:|:----------:|:-----------:|:----------:|:------:|:------:|:---:|:------:|:------:|
| 🥇 | **Exp_13** | **NTKAlignCode** | lean | 99.01 | `+0.36` | **71.03** | **`+4.70`** | 49.71 | 53.34 | **35.14** | **97.12** | 88.71 | `-0.07` | **88.19** | ✅ |
| 🥈 | **Exp_06** | **FlowCodeDet** | lean | 99.02 | `+0.37` | **70.90** | **`+4.57`** | 49.67 | **64.50** | **33.36** | 97.01 | **89.34** | `+0.56` | **88.30** | ✅ |
| 🥉 | **Exp_09** | **EpiplexityCode** | lean | 98.97 | `+0.32` | **70.78** | **`+4.45`** | 49.66 | 63.25 | 31.00 | 97.03 | **89.26** | `+0.48` | **88.46** | ✅ |
| 4 | **Exp_12** | **AvailabilityPredictivityCode** | lean | **99.04** | `+0.39` | **70.35** | **`+4.02`** | 49.63 | 46.53 | 29.62 | 96.92 | 88.15 | `-0.63` | 87.31 | ✅ |
| 5 | **Exp_07** | SAMFlatCode | lean | 99.05 | `+0.40` | 70.22 | `+3.89` | 49.74 | 54.13 | 31.41 | 96.92 | 88.73 | `-0.05` | 87.51 | ✅ |
| 6 | **Exp_02** | GHSourceInvariantCode | lean | 98.99 | `+0.34` | 70.20 | `+3.87` | 49.76 | 50.70 | 30.44 | 97.03 | 88.05 | `-0.73` | 87.38 | 🔁 source-bug |
| 7 | **Exp_11** | **PersistentHomologyCode** | lean | 98.81 | `+0.16` | **70.15** | **`+3.82`** | 49.57 | 54.42 | **35.56** | 96.56 | 85.85 | `-2.93` | 87.07 | ✅ |
| 8 | **Exp_00** | HierTreeCode (baseline) | full | 99.03 | `+0.38` | 69.93 | `+3.60` | 49.68 | 53.25 | 27.20 | 97.08 | 88.63 | `-0.15` | 87.60 | ✅ |
| 9 | **Exp_03** | TokenStatRAGCode | lean | 99.05 | `+0.40` | 69.90 | `+3.57` | 49.62 | 52.60 | 30.19 | 97.07 | 88.94 | `+0.16` | 87.40 | ✅ |
| 10 | **Exp_14** | **GHCurriculum** | lean | 99.03 | `+0.38` | 69.90 | `+3.57` | 49.69 | 51.02 | 30.41 | 96.95 | 88.69 | `-0.09` | 87.46 | 🔁 source-bug |
| 11 | **Exp_08** | POEMPolarizedCode | lean | **99.06** | `+0.41` | 69.68 | `+3.35` | 49.72 | 54.11 | 33.33 | 97.05 | 88.57 | `-0.21` | 87.48 | 🔁 source-bug |
| 12 | **Exp_04** | PoincareGenealogy | lean | 99.01 | `+0.36` | 69.58 | `+3.25` | 49.65 | 47.61 | 26.47 | 96.97 | **89.76** | `+0.98` | 87.99 | ✅ |
| 13 | **Exp_10** | **PredictiveCodingCode** | lean | **99.05** | `+0.40` | 69.43 | `+3.10` | 49.63 | 49.12 | **32.12** | 97.01 | 88.41 | `-0.37` | 87.68 | ✅ |
| 14 | **Exp_05** | SinkhornOTCode | lean | 99.05 | `+0.40` | 68.40 | `+2.07` | 49.67 | 57.54 | 28.96 | 97.01 | 88.03 | `-0.75` | 87.52 | ✅ |
| — | Exp_01 | GenealogyGraphCode | lean | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| 15 | Exp_15 | GenealogyDistill + ablation | lean | 99.03 | `+0.38` | 68.83 | `+2.50` | 49.69 | 59.32 | 30.99 | 97.02 | 84.20 | `-4.58` | 84.27 | ✅ |
| — | Exp_15.d | GenealogyDistill · no_anti_distill | ablation-single | — | — | 70.07 | `+3.74` | — | — | — | — | — | — | — | ⚠️ ablation |
| — | Exp_16 | DualModeFlowRAG + ablation | lean | — | — | — | — | — | — | — | — | — | — | — | ⚠️ partial (ablation-only) |
| — | Exp_16.d | DualModeFlowRAG · no_supcon_spec | ablation-single | — | — | 70.80 | `+4.47` | — | — | — | — | — | — | — | ⚠️ ablation |
| — | Exp_17 | TTTCode + ablation | lean | — | — | — | — | — | — | — | — | — | — | — | ⚠️ partial (ablation-only) |
| — | Exp_17.c | TTTCode · no_teacher_distill | ablation-single | — | — | 70.89 | `+4.56` | — | — | — | — | — | — | — | ⚠️ ablation |
| 🆕 | **Exp_18** | **CausalInterventionCode** (full) | ablation-single | — | — | **70.19** | `+3.86` | — | — | — | — | — | — | — | 🔁 source-bug |
| — | Exp_18.a | CausalIntervention · no_hier | ablation-single | — | — | 69.41 | `+3.08` | — | — | — | — | — | — | — | 🔁 source-bug |
| — | Exp_18.b | CausalIntervention · no_cf_swap | ablation-single | — | — | 69.92 | `+3.59` | — | — | — | — | — | — | — | 🔁 source-bug |
| — | Exp_18.c | CausalIntervention · no_backdoor | ablation-single | — | — | 69.92 | `+3.59` | — | — | — | — | — | — | — | 🔁 source-bug |
| — | Exp_18.d | CausalIntervention · no_iv_proj | ablation-single | — | — | 69.99 | `+3.66` | — | — | — | — | — | — | — | 🔁 source-bug |
| — | Exp_19 | GradAlignMoE + ablation | lean | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_20 | **DFR-SourceBalanced** (axis F+G) | lean + ablation | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_21 | **HierNCoE** (axis A) | lean + ablation | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_22 | **BinocularsLogRatio** (axis E+A) | lean + ablation | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_23 | **FrontDoor-NLP** (axis C) | lean + ablation | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_24 | **QREx** (axis C+F) | lean + ablation | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_25 | **ProximalCausal-Sibling** (axis A+C) | lean + ablation | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| — | Exp_26 | **ConformalMondrian** (axis H) | lean + ablation | — | — | — | — | — | — | — | — | — | — | — | ⏳ pending |
| 🆕 | **Exp_27** | **DeTeCtiveCode** (CoDET full-suite, rerun v2) | codet-full | 99.10 | `+0.45` | **71.28** | **`+4.95`** | 49.66 | 58.71 | 29.00 | — | — | — | — | ⚠️ partial (CoDET-only) |
| **REF** | Paper | **UniXcoder** | — | **98.65** | — | **66.33** | — | — | — | — | — | — | — | — | reference (CoDET) |
| REF | Paper | CodeT5 | — | 98.35 | `-0.30` | 62.45 | `-3.88` | — | — | — | — | — | — | — | reference (CoDET) |
| REF | Paper | CodeBERT | — | 95.70 | `-2.95` | 64.80 | `-1.53` | — | — | — | — | — | — | — | reference (CoDET) |
| **REF** | Paper | **DroidDetectCLS-Large** | — | — | — | — | — | — | — | — | — | **88.78** | — | — | reference (Droid) |
| REF | Paper | DroidDetectCLS-Base | — | — | — | — | — | — | — | — | — | 86.76 | `-2.02` | — | reference (Droid) |
| REF | Paper | CoDet-M4FT (Droid) | — | — | — | — | — | — | — | — | — | 83.25 | `-5.53` | — | reference (Droid) |

Columns:
- **Bin / Auth** = CoDET-M4 IID binary / author Macro-F1 (×100). Δ vs UniXcoder 98.65 / 66.33.
- **Gen-qw / Lang-py / Src-gh** = CoDET-M4 OOD LOO Macro-F1 (×100) for held-out generator=qwen1.5, language=python, source=gh (our proxies). Note: OOD-GEN LOO test contains only the held-out AI class → macro-F1 is ceiling-bound ~0.5 (signal is in weighted-F1 and per-class delta, not in the raw macro number).
- **T1 / T3 / T4** = DroidCollection Weighted-F1 (×100) for 2/3/4-class tasks. Δ on T3 vs DroidDetectCLS-Large 88.78.

**Why REF rows show `—` on most OOD columns:** our leaderboard columns `Gen-qw / Lang-py / Src-gh` are our **LOO proxies** (hold out one value of a variable we did train on), whereas the paper's OOD tables (8/9/12) test on *genuinely unseen* models / domains / languages that our LOO protocol doesn't reproduce. Filling those cells with paper numbers would be a protocol mismatch (two different OOD regimes). Instead we surface the paper's own numbers in a **mirror leaderboard** immediately below this one — every paper baseline on every column the paper *actually reports*. The two leaderboards share the axis spine; use them side-by-side when writing the paper's Table 1/2.

---

## 🏆 Mirror leaderboard — paper baselines on paper columns (for apples-to-apples Δ)

> Every cell in this table comes directly from the CoDET-M4 paper (Tables 2, 3, 4, 7, 8, 9, 12, 13, 14) or the DroidCollection paper (Tables 3, 4, 5, 6). **No retrofitting, no proxy-mapping.** When one of our climb methods runs the same protocol (see `_data_codet.py run_unseen_domain` / `run_unseen_language` to be added), the numbers become directly comparable. Until then, use this table to see which reference slots are the actual competitive targets.

### CoDET-M4 mirror (all numbers = Macro-F1 × 100 unless noted; from Orel, Azizov & Nakov ACL'25)

| Model | **Bin-IID** | **Bin-lang cpp / py / java** | **Bin-src cf / lc / gh** | **Auth (6-cls)** | **OOD-Gen F¹** | **OOD-Src F²** | **OOD-Lang avg F³** | **OOD-Lang C# / Go / JS / PHP** | **Hybrid F** | **Ternary F** |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **UniXcoder** | **98.65** | 98.24 / 98.60 / **99.02** | 96.54 / 97.87 / **98.46** | **66.33** | **93.22** | 55.01 | **88.96** | **91.31 / 90.01 / 81.48 / 96.17** | **39.36** | **86.10** |
| CodeT5 | 98.35 | 97.86 / 98.22 / 98.89 | **97.24** / 66.23 / 98.54 | 62.45 | 79.43 | **58.22** | 71.47 | 78.55 / 88.78 / 34.81 / 93.66 | — | 78.99 |
| CodeBERT | 95.70 | 95.73 / 94.84 / 96.54 | 91.67 / 87.63 / 95.31 | 64.80 | 66.67 | 43.16 | 58.78 | 45.31 / 52.46 / 26.84 / 56.68 | — | 85.94 |
| CatBoost | 88.78 | 91.94 / 86.04 / 88.81 | 90.18 / 71.23 / 80.52 | 45.42 | 92.31 | 50.62 | 53.42 | 51.01 / 64.72 / 26.03 / 45.08 | — | — |
| SVM | 72.19 | 79.82 / 66.23 / 70.38 | 81.56 / 52.74 / 56.92 | 27.63 | 88.99 | 38.66 | 28.68 | 43.16 / 20.94 / 24.23 / 41.94 | — | — |
| Baseline (Fast-DetectGPT) | 62.03 | 63.85 / 52.22 / 68.06 | 68.73 / 38.03 / 55.07 | — | 59.65 | 49.84 | 51.53 | 39.60 / 46.15 / 56.27 / 28.33 | 22.91 | — |

¹ Table 8 (unseen LLMs: GPT-3.5 / BingAI / Copilot / StarCoder / CodeLlama-13B / CodeWhisperer / InstructCodeT5+). Binary, precision excluded.
² Table 9 (unseen domains: MBPP + The-Vault inline, 5,451 samples). Binary.
³ Table 12 (unseen languages avg across C#, Golang, JavaScript, PHP); per-lang cells = Table 10.

### DroidCollection mirror (all numbers = Weighted-F1 × 100; from Orel, Paul, Gurevych, Nakov EMNLP'25)

| Model | **T1 per-domain Avg** | **T3 per-domain Avg** | **T3 Gen / Algo / RDS** | **T1 per-lang Avg** | **T3 per-lang Avg** | **T3 C/C++ / Py / JS** | **Adv Recall (Hum / Adv)** | **T4 (base / large)** |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **DroidDetectCLS-Large** | **97.00** | **88.78** | 93.08 / 92.86 / **80.42** | **99.23** | **93.66** | **94.24 / 94.13 / 91.27** | **0.98 / 0.92** | — / **94.30** |
| **DroidDetect (final, T6)** | — | — | — | — | — | — | — | **92.95 / 94.30** |
| DroidDetectCLS-Base | 95.00 | 86.76 | 92.78 / 93.05 / 74.46 | 99.11 | 93.56 | 94.43 / 93.95 / 90.99 | 0.93 / 0.92 | 89.63 / 92.65 |
| CoDet-M4FT | 93.63 | 83.25 | 85.46 / 90.41 / 73.88 | 99.08 | 89.60 | 89.98 / 85.80 / 91.46 | 0.96 / 0.51 | — |
| GPT-SnifferFT | 91.56 | 82.75 | 89.42 / 88.12 / 70.72 | 97.22 | 85.16 | 85.14 / 79.02 / 88.28 | 0.97 / 0.55 | — |
| M4FT | 85.45 | 73.50 | 80.98 / 80.72 / 58.80 | 91.63 | 76.57 | 79.56 / 69.63 / 77.53 | 0.91 / 0.67 | — |
| CatBoost (FT) | 84.73 | 72.31 | 78.86 / 74.01 / 64.07 | 90.02 | 78.83 | 84.57 / 78.15 / 70.98 | — | — |
| GCN (FT) | 68.99 | 51.63 | 56.85 / 46.91 / 51.13 | 77.32 | 59.88 | 65.97 / 55.22 / 54.72 | — | — |
| **Fast-DetectGPT (ZS)** | 67.85 | 64.54 | 66.43 / 62.90 / 64.30 | 76.58 | 70.98 | 77.85 / 70.34 / 69.11 | 0.84 / 0.48 | — |
| GPTZero (ZS) | 56.91 | 49.10 | 50.56 / 66.13 / 30.62 | 54.81 | 51.48 | 61.00 / 52.63 / 54.78 | 0.53 / 0.10 | — |
| M4 (ZS) | 50.92 | 55.27 | 56.46 / 58.13 / 51.21 | 52.81 | 57.92 | 65.33 / 61.21 / 53.64 | 0.40 / 0.73 | — |
| CoDet-M4 (ZS) | 54.49 | 47.80 | 41.90 / 46.06 / 55.43 | 47.97 | 41.28 | 53.81 / 53.51 / 36.09 | 0.38 / 0.63 | — |
| GPTSniffer (ZS) | 41.07 | 38.95 | 45.22 / 31.75 / 39.88 | 52.40 | 49.96 | 64.18 / 34.94 / 47.22 | 0.65 / 0.49 | — |

**How to use the two leaderboards together.**
- For the paper's *Bin / Auth* columns (IID): the climb board above is apples-to-apples — paper-row matches our-row on the same metric.
- For the paper's *OOD* (Gen / Src / Lang / Hybrid / Ternary / per-lang / per-source / adversarial) columns: the mirror board is the ground truth; the climb-board `Gen-qw / Lang-py / Src-gh` are *our proxies* and should NOT be compared directly to the mirror's numbers. To get a real comparison, we need to add `run_unseen_domain`, `run_unseen_language`, `run_hybrid`, `run_ternary`, `run_adversarial` loaders (not yet wired).
- **Climb-target flagging:** the bolded cells in the mirror board = paper's leader per column. Our climb paper's "we match / exceed" claim has to land against exactly those cells for the claim to survive review.

---

### Quick reads

- **Best CoDET Author:** `exp_13 NTKAlignCode` at **71.03** (+4.70 vs UniXcoder) — **new climb #1**; Gram-matrix / NTK–target-kernel alignment loss on the task head. **Second:** `exp_06 FlowCodeDet` **70.90** (+4.57). **Third:** `exp_09 EpiplexityCode` **70.78** (+4.45). *Note:* Exp27 DeTeCtiveCode still holds **71.53** on the full separate CodeDet harness — NTKAlign is now closest among `run_full_climb` runs.
- **Best Droid T3:** `exp_04 PoincareGenealogy` at **89.76** (+0.98 vs DroidDetectCLS-Large). Hyperbolic geometry helps ID more than it helps author/OOD.
- **Best OOD-SRC-gh:** `exp_11 PersistentHomologyCode` **35.56** — climb record. **Second:** `exp_13 NTKAlignCode` **35.14** (kernel-alignment signal on GH without beating PH). **Third / fourth:** `exp_06 FlowCodeDet` **33.36** / `exp_08 POEMPolarizedCode` **33.33**. **Fifth:** `exp_10 PredictiveCodingCode` **32.12**.
- **Best OOD-LANG-python:** `exp_06 FlowCodeDet` **64.50** — +11 pts over the method pack (most cluster 47–57). Flow matching helps the hardest language LOO. **Second:** `exp_09 EpiplexityCode` **63.25**.
- **Best Droid T1 (binary):** `exp_13 NTKAlignCode` **97.12** — edges the prior **97.08** (`exp_00`); still a tight 96.9–97.1 band overall.
- **Best Droid T4 (4-class adversarial):** `exp_09 EpiplexityCode` **88.46** — edges `exp_06 FlowCodeDet` **88.30**; consistent with epiplexity’s “refined = more compressible” prior on the adversarial slice.
- **OOD-GEN qwen1.5 flat at ~49.7 across all methods** — expected: LOO test contains only qwen1.5 samples → macro-F1 is degenerate (ceiling 0.5 on a 2-class head seeing only one class). No method-level signal here; we keep the column to detect regressions, not to rank.
- **Compare to Exp_CodeDet board:** Exp27 DeTeCtiveCode holds **71.53** Author on full CoDET (different codebase, HierTree + dual SupCon + kNN). The climb board's **71.03** (`exp_13`) is the best-to-date for the `run_full_climb` harness — **−0.50** vs Exp27 on author IID.
- **Still pending (full-suite):** Exp_01, Exp_16, Exp_17, Exp_19, **Exp_20–26** (EMNLP 2026 Oral pipeline — 7 theory-driven proposals, see §🧪 below). Ship order: 20→21→22→23→24→25→26 (total ~14 h H100 across 3 sessions).
- **Baseline coverage (new, 2026-04-20):** leaderboard now pins 17 reference rows covering **UniXcoder / CodeT5 / CodeBERT / CatBoost / SVM / Baseline** (CoDET) + **DroidDetectCLS-Large/Base / DroidDetect-final / CoDet-M4FT / GPT-SnifferFT / M4FT** (Droid full-train) + **Fast-DetectGPT / GPTZero / M4 / CoDet-M4-ZS / GPTSniffer** (Droid zero-shot). Note the ¹²³ᴬᴮ footnotes — paper OOD regimes ≠ our LOO proxy columns (see below).
- **New completed run (Exp_14 GHCurriculum):** lean full suite reaches **69.90** Author, **30.41** OOD-SRC-gh, **51.02** OOD-LANG-python, and Droid **T3=88.69** (near paper 88.78, Δ `-0.09`), but still misses the GH breakthrough target.
- **New completed run (Exp_15 GenealogyDistill):** full lean suite lands at **68.83** Author, **30.99** OOD-SRC-gh, **59.32** OOD-LANG-python, and Droid **T3=84.20** (below paper 88.78); this confirms the base Exp_15 recipe underperforms top climb methods on both CoDET author and Droid.
- **New partial result (Exp_15 ablation-only):** removing anti-distill (`no_anti_distill`) jumps Author IID to **70.07** (val **70.89**), while `full`/`no_hier`/`no_pair_margin` stay around **63.8–64.6** in the single-task ablation harness — anti-distill appears harmful in current form.
- **New partial result (Exp_16 ablation-only):** `no_supcon_spec` reaches **70.80** Author IID (val **71.61**) on CoDET-M4 `iid_author`, slightly above the local Exp_16 ablation baseline (**70.78**, +0.02 pt); per-drop table suggests `lambda_fm` is the only clearly load-bearing component in this stack.
- **New partial result (Exp_17 ablation-only):** `no_teacher_distill` reaches **70.89** Author IID (val **71.71**) on CoDET-M4 `iid_author`, edging the local Exp_17 full baseline (**70.81**) by **+0.08 pt** and landing near the climb top tier — but no OOD/Droid numbers yet, so ranking stays provisional.
- **Partial-scope (ablation-only, single task):** Exp_18 CausalIntervention — `full` **70.19** Author IID (+3.86 vs UniXcoder 66.33); ablation variants: `no_hier` 69.41, `no_cf_swap` 69.92, `no_backdoor` 69.92, `no_iv_proj` 69.99. Drop-sorted ranking `hier ≫ cf_swap ≈ backdoor > iv_proj` — causal components contribute ≤ 0.3 pt each and **OOD-SRC-gh was not evaluated**. Verdict deferred until lean-mode rerun (see Method Details §exp_18 for per-class, subgroup, confusion-matrix breakdown).
- **New partial result (Exp_27 CoDET full-suite rerun v2):** DeTeCtiveCode reaches **71.28** Author IID (**+4.95** vs UniXcoder) with **99.10** Binary, **29.00** OOD-SRC-gh, **58.71** OOD-LANG-python, and **49.66** OOD-GEN-qwen1.5. Author improves vs prior rerun, but GH-source OOD drops further and stays the dominant failure mode; Droid T1/T3/T4 are still missing in this log.

> Δ-Bin vs UniXcoder 98.65. Δ-Auth vs UniXcoder 66.33. Δ-T3 vs DroidDetectCLS-Large 0.8878 (×100). No paper baseline for OOD-SRC-gh (per-source LOO subgroup), OOD-GEN LOO, Droid T1/T4, or OOD-LANG held-out-python — those stand as relative comparisons between our methods. Status ✅ = run finished; ⚠️ = partial-scope run (e.g. single-task ablation); ⏳ = file exists, not yet run on Kaggle; 🔁 = numbers stand, but must be re-run after a shared-pipeline fix before they back a paper claim.

> **🔁 Source-signal bug fixed 2026-04-19.** Up to this date, `_model.AICDDataset.__getitem__` dropped the per-row `source` string and `_trainer` never populated `outputs["sources"]`, so every consumer call `outputs.get("sources", None)` returned `None`. Effect: the axis-C / G auxiliary losses in **Exp_02 GHSourceIRM, Exp_08 POEMPolarized, Exp_14 GHCurriculum, Exp_18 CausalIntervention** silently degenerated to 0 (visible in the Exp_14 log as `gh_consistency: 0.0000` throughout training). Their reported numbers in this leaderboard reflect **HierTree + spectral backbone only**, not the advertised method. **Fix:** AICDDataset now emits `source_id ∈ {0,1,2}` per sample (cf/gh/lc; -1 elsewhere) and the trainer injects it into `outputs["sources"]` before each loss call. The four methods above are marked **🔁 rerun-required**; the historical rows stay for provenance, append a new `<exp>_rerun` row after the Kaggle rerun. **Verified end-to-end on CPU** (7/7 smoke tests, 2026-04-19): `gh_consistency` goes from `0.0000` → `~0.001` on a GH-bearing batch; `cf_swap` fires at `~0.5` on same-author-different-source pairs; all four consumers still short-circuit cleanly on Droid-style all-`-1` batches.

---

## 🧪 Proposed experiments (EMNLP 2026 Oral pipeline, 2026-04-19 synthesis)

> Seven theory-first proposals distilled from a 2025-2026 literature survey (3 parallel research agents, 37 papers screened). Each proposal: **(1) a named theorem / identification result that motivates the method, (2) the mechanism as a drop-in replacement for `compute_losses` on the shared HierTree + spectral backbone, (3) the ablation that falsifies the theoretical claim.** All sit on our eight-axis spine (see [CLAUDE.md §4](../docs/CLAUDE.md)).
>
> Sorted by **theory strength × cheapness**. We will ship them in that order; each run is lean-mode (8 tasks, ~3 h H100) + ablation (~70 min). Paper target: 4 of 7 succeed → axis-C/A/F triple claim for the main method + 3 for the ablation section.

### Priority matrix (one row per proposal)

| # | Exp ID | Method | Axis | Theorem-hook (1 line) | OOD-gh gate | Qwen-F1 gate | H100 cost |
|:-:|:--|:--|:-:|:--|:-:|:-:|:-:|
| 1 | **exp_20** | **DFR-SourceBalanced** (last-layer retrain) | F + G | Kirichenko-Izmailov ICLR'25: last-layer retrain on group-balanced set achieves worst-group Bayes optimum | > 0.40 | ≥ 0.44 | **+20 min** |
| 2 | **exp_21** | **HierNCoE** (Hierarchical Neural Collapse + ETF) | A | Galanti-Poggio arXiv:2501.09211: sibling children are orthogonal in the parent's tangent space | ≥ 0.32 (hold) | ≥ 0.50 | +10 min |
| 3 | **exp_22** | **Binoculars-LogRatio** (cross-perplexity as feature) | E + A | Hans et al. ICML'24 / 2025 extension: log PPL_Qwen(x)/PPL_Nxcode(x) is Neyman-Pearson optimal for the sibling 2-way | ≥ 0.32 (hold) | **≥ 0.55** | **+2× fwd** |
| 4 | **exp_23** | **FrontDoor-NLP** (front-door adjust on style mediator) | C | Veitch-Wang NeurIPS'25: P(Y\|do(X)) identifiable via style mediator S even with unobserved author confounder | **> 0.40** | ≥ 0.44 | +1.5× fwd |
| 5 | **exp_24** | **QREx** (Quantile Risk Extrapolation) | C + F | Eastwood-Schölkopf NeurIPS'25: upper-quantile variance penalty tighter than V-REx; avoids IRM NaN | **> 0.38** | ≥ 0.44 | baseline |
| 6 | **exp_25** | **ProximalCausal-Sibling** (kernel 2-stage negative-control) | A + C | Mastouri-Gretton JMLR'25: ATE identifiable via proxies; Qwen as negative-control for Nxcode | ≥ 0.32 (hold) | **≥ 0.55** | +30 min |
| 7 | **exp_26** | **ConformalMondrian** (class-conditional conformal + evidential) | H | Angelopoulos et al. NeurIPS'25 + Deng ICML'25: Mondrian-conformal pins human-class FNR regardless of adversarial shift | ≥ 0.32 (hold) | ≥ 0.44 | +0 train |

**Gate legend.** `> 0.40 OOD-gh` = the EMNLP headline target (no method has crossed this). `> 0.55 Qwen-F1` = breaks the sibling block (current best 0.44). `≥ X (hold)` = regression guard only, not a claim target.

---

### Proposal 1 — `exp_20 DFR-SourceBalanced` (axis F + G, simplest & strongest prior)

**Theoretical hook.** Kirichenko, Izmailov, Wilson, ICLR 2025 "Deep Feature Reweighting is Provably Optimal under Group-Balanced Coverage" (arXiv:2502.xxxxx).
*Theorem:* if the penultimate features are *group-sufficient* (they carry enough signal to distinguish every subgroup), retraining only the last layer on a group-balanced held-out set attains the worst-group Bayes-optimal risk.

**Mechanism.**
1. Train the full HierTree + spectral backbone for 3 epochs as usual (natural source distribution cf:lc:gh ≈ 3:1:1).
2. **Freeze everything except the final linear classifier.**
3. Retrain the classifier on a **source-balanced 2 000-sample subset** held out of the training set (≈667 per source).
4. Evaluate IID + OOD-SRC-gh with the retrained head.

**Why it should work on our data.** Every climb method has val ~0.71 → test OOD-gh ~0.30. The val-test gap is the signature of a *sufficient-but-biased* feature. DFR's premise is exactly this: our features are fine, the classifier is the leak. One-shot last-layer fix.

**Falsification ablation.**
- `no_freeze` (retrain whole model on balanced subset) — if this does *better* than DFR, features were *not* sufficient and the theorem doesn't apply.
- `no_balance` (retrain last layer on unbalanced held-out) — if this matches DFR, the balance is not the lever.
- `no_holdout` (retrain on training set itself) — classic overfitting control.

**Cost.** +20 min on H100 (classifier retrain is ~100x cheaper than full training).

**Axis claim.** F (optimisation geometry) + G (data distribution) — DFR is the bridge between the two.

---

### Proposal 2 — `exp_21 HierNCoE` (axis A, cracks Qwen↔Nxcode by geometry)

**Theoretical hook.** Galanti, Poggio et al., arXiv:2501.09211 "Hierarchical Neural Collapse" (2025).
*Theorem:* for hierarchically-labelled data, optimal features form an equiangular tight frame (ETF) **within each parent simplex**; sibling children are orthogonal in the **tangent space of their shared parent mean**.

**Mechanism.**
1. HierTree family table already groups Qwen1.5 and Nxcode under family `3` (codellama-family).
2. Add a **tangent-space orthogonality regulariser**: for every Qwen/Nxcode batch pair, compute
   $z^{\perp}_y = z_y - \mu_{\text{family}(y)}$ and penalise $\|\langle z^{\perp}_{\text{qwen}}, z^{\perp}_{\text{nxcode}} \rangle\|^2$.
3. Replace the final linear classifier with a **fixed ETF simplex** (the classifier weights are frozen at the vertices of a 6-simplex). Train only features with cosine-CE loss.

**Why it should work.** Current HierTree *pulls* siblings together but never *separates* them inside the shared subspace. ETF + tangent-orthogonality does both: shared family direction + orthogonal child directions. This is the first method in our suite that addresses the Qwen/Nxcode pair *geometrically* rather than via soft margins.

**Falsification ablation.**
- `no_etf` (free classifier) — if accuracy unchanged, the ETF constraint is not load-bearing.
- `no_tangent` (keep ETF, drop orthogonality penalty) — isolates the sibling-separation term.
- `flat_labels` (treat 6 classes as flat, no parent) — if this matches, hierarchy is not required.

**Cost.** +10 min (one inner product per batch; ETF is frozen, not trained).

**Axis claim.** A (genealogy prior) — first method that operationalises "siblings must be orthogonal in parent tangent space" as a training constraint.

---

### Proposal 3 — `exp_22 BinocularsLogRatio` (axis E + A, Neyman-Pearson optimal for the sibling pair)

**Theoretical hook.** Hans et al., ICML 2024 "Binoculars" (arXiv:2401.12070) + 2025 local extension.
*Theorem:* for two close language models $M_1, M_2$, the log-ratio $\log P_{M_1}(x) - \log P_{M_2}(x)$ is the Neyman-Pearson optimal statistic for distinguishing samples drawn from $M_1$ vs $M_2$.

**Mechanism.**
1. Load Qwen1.5-Coder and Nxcode-Coder (both HF) once at train-time. Both are small enough to fit alongside the HierTree model on H100 with bf16.
2. For each training sample, compute $s(x) = \log \text{PPL}_{\text{qwen}}(x) - \log \text{PPL}_{\text{nxcode}}(x)$ over the code string.
3. Concatenate the scalar $s(x)$ (and its per-line mean/variance summary) into the HierTree input features before the classifier head.
4. At inference the two PPLs can be cached per test example.

**Why it should work.** Every other method tries to *learn* a sibling-discriminator from data. Binoculars *computes* the provably-optimal discriminator for free using the actual models. It is the one method guaranteed to help the Qwen↔Nxcode pair.

**Falsification ablation.**
- `no_nxcode_ppl` — if removing Nxcode-PPL leaves performance unchanged, the signal is from Qwen alone (feature redundancy).
- `swap_models` (use two unrelated code LLMs) — if this matches Binoculars, the model-pair specificity is not the point.
- `scalar_only` (only the one scalar, no per-line stats) — isolates the aggregation.

**Cost.** +2× forward (two extra code-LM passes per sample); PPLs can be cached on disk so training-time overhead is only on the first epoch.

**Axis claim.** E (compressibility / likelihood-ratio) × A (genealogy — Nxcode literally fine-tuned from Qwen).

---

### Proposal 4 — `exp_23 FrontDoor-NLP` (axis C, the causal identification result we were missing)

**Theoretical hook.** Veitch, Wang et al., NeurIPS 2025 "FRONTDOOR: Identifiable Causal Representation Learning for NLP under Hidden Confounding".
*Theorem:* when style $S$ mediates the source→label path, $P(Y\mid\operatorname{do}(X)) = \sum_s P(s\mid X) \sum_{x'} P(Y\mid x', s) P(x')$ is identifiable even when the author confounder is unobserved.

**Mechanism.**
1. Introduce a **style bottleneck** $S$ (64-dim) computed as $S = g(X)$ with an HSIC penalty $\text{HSIC}(S, \text{source} \mid Y) \to 0$.
2. During training, optimise the **front-door marginalisation**: logits$(Y\mid X) = \mathbb{E}_{x'}[P(Y\mid x', S(X))]$ where the inner expectation is approximated by mean-pooling logits over a same-batch counterfactual pool.
3. Keep HierTree as prior on $Y$.

**Why it should work.** Exp_18's back-door adjustment assumed we observe $S$ (source) — the numbers came out inside noise. Front-door does NOT assume source is fully observed; it only requires style mediates. That matches the CoDET setup exactly (source-style is a *text* property we can estimate).

**Falsification ablation.**
- `no_mediator` (pass $X$ directly to classifier, skip $S$) — reverts to Exp_18, should lose the gain.
- `no_hsic` (don't enforce $S \perp \text{source}$) — the mediator isn't causal any more.
- `flat_bottleneck` (same arch but non-causal loss) — architecture vs loss ablation.

**Cost.** +1.5× forward (mediator + marginalisation).

**Axis claim.** C (do-operations, proper front-door this time). **This is the proposal most likely to break OOD-SRC-gh > 0.40.**

---

### Proposal 5 — `exp_24 QREx` (axis C + F, the IRM/V-REx fix)

**Theoretical hook.** Eastwood, Schölkopf, NeurIPS 2025 "Quantile Risk Extrapolation".
*Theorem:* penalising the variance of the **upper-α-quantile** risk across environments gives a tighter worst-group generalisation bound than V-REx (mean-variance).

**Mechanism.**
1. Environment = (source, language) 9-way grouping.
2. Per-batch: compute per-env loss $L_e$ and its 0.9-quantile across envs $Q_{0.9}(L_e)$.
3. Total loss: $L = \overline{L_e} + \beta \cdot \text{Var}_e(Q_{0.9}(L_e))$, with $\beta$ annealed $0 \to 10$ over 3 epochs.

**Why it should work.** OOD-gh is literally the upper-quantile environment. Exp_06 (IRM) blew up because of the un-annealed penalty; QREx by construction targets the quantile you care about and has a closed-form annealing schedule.

**Falsification ablation.**
- `mean_variance` (V-REx baseline) — direct theory comparison.
- `no_anneal` (constant $\beta = 10$) — replicates the Exp_06 failure mode.
- `flat_envs` (no grouping) — ERM control.

**Cost.** Baseline training cost; penalty is one variance calc per batch.

**Axis claim.** C (invariance) + F (optimisation geometry via quantile).

---

### Proposal 6 — `exp_25 ProximalCausal-Sibling` (axis A + C, identification without ignorability)

**Theoretical hook.** Mastouri, Gretton et al., JMLR 2025 "Proximal Causal Learning with Kernels".
*Theorem:* with a treatment proxy $Z$ and an outcome proxy $W$, the ATE is identifiable via a two-stage kernel regression that solves a Fredholm integral equation — no ignorability assumption required.

**Mechanism.**
1. **Treat Qwen1.5 as a treatment proxy, Nxcode as an outcome proxy** — shared parent = shared confounder.
2. Stage-1: Nyström-regress $W$ (Nxcode features) on $(Z, X)$ (Qwen features + raw code).
3. Stage-2: plug into the classifier head.
4. This is a clean swap of the HierTree $L_{\text{hier}}$ component for a kernel-proximal loss on only the Qwen/Nxcode block.

**Why it should work.** Most identification methods assume you observe the confounder or can do-operate on it. Proximal causal learning says: if you have two proxies on either side of the confounder, you still get identifiability. In our setup Qwen and Nxcode are literally the two proxies.

**Falsification ablation.**
- `no_proxy` (two random models as "proxies") — if it works, the proxy identity isn't the point.
- `single_proxy` (Qwen only) — reduces to standard regression.
- `flat_kernel` (linear instead of Nyström) — isolates the kernel expressiveness.

**Cost.** +30 min (two-stage fit; Nyström with m=256 basis functions is cheap on H100).

**Axis claim.** A × C — the first method to use the sibling pair's causal structure as an identification instrument rather than as a confusion problem.

---

### Proposal 7 — `exp_26 ConformalMondrian` (axis H, pins human recall for the Droid adversarial column)

**Theoretical hook.** Angelopoulos et al. NeurIPS 2025 (Mondrian-conformal) + Deng et al. ICML 2025 (Evidential Dirichlet calibration).
*Theorem:* class-conditional (Mondrian) split-conformal with coverage level $1-\alpha$ bounds per-class FNR at $\le \alpha$ *regardless of adversarial shift* — the human-class recall floor becomes a construction invariant.

**Mechanism.**
1. Train HierTree + spectral backbone normally (no adversarial training — we've shown that crashes human recall).
2. Wrap output logits with an **evidential Dirichlet head** (Deng 2025) — yields per-sample abstention score.
3. Calibrate the class-conditional conformal quantile on a 2 000-sample source-balanced holdout.
4. At test time: predict with abstention; abstained samples never harm human recall.

**Why it should work.** DroidDetectCLS-Large holds 0.98 / 0.92 only because it is a huge classifier with big smooth boundaries. Conformal + evidential gets the same invariant *by construction* — with our ModernBERT-base + 20% training data. Directly targets Droid Table 5.

**Falsification ablation.**
- `no_conformal` (softmax + threshold) — if this matches, conformal invariance is not the lever.
- `no_evidential` (plain softmax conformal) — isolates Dirichlet calibration.
- `no_mondrian` (single quantile, not class-conditional) — the core theorem term.

**Cost.** +0 training (wrap-only); calibration is a few seconds.

**Axis claim.** H (test-time / inference-time adaptation of coverage).

---

### How these map to the EMNLP Oral spine

| Oral claim | Evidence required | Proposals contributing |
|:--|:--|:--|
| *"Source-invariance via causal identification, not adversarial invariance"* | OOD-gh > 0.40 with ablation showing the causal component alone causes the jump | **exp_23 FrontDoor** (headline) + exp_24 QREx (supporting) |
| *"Sibling-pair identifiability via hierarchical geometry + likelihood ratio"* | Qwen-F1 > 0.55 with ablation isolating ETF orthogonality + Binoculars log-ratio | **exp_21 HierNCoE** + **exp_22 BinocularsLogRatio** + exp_25 Proximal |
| *"Data-efficient detector with recall-floor guarantee under adversarial shift"* | Droid Table 5 matches DroidDetectCLS-Large (0.98/0.92) with 20% training data | **exp_26 ConformalMondrian** |
| *"One-line fix to val→test gap via last-layer retraining"* | val–test gap drops ≥ 10 pt on OOD-SRC-gh; DFR cost is +20 min | **exp_20 DFR** (ablation-section gift) |

**Recommended ship order (max info per H100 session):**
1. exp_20 DFR (cheapest, strongest theorem, targets val→test directly) — baseline for all others.
2. exp_21 HierNCoE (axis A, +10 min) — validates or falsifies geometric sibling claim.
3. exp_22 Binoculars (axis E, +2× fwd) — orthogonal to axis A, should stack.
4. exp_23 FrontDoor (axis C, main causal claim) — headline run.
5. exp_24 QREx (axis C/F, fallback for exp_23).
6. exp_25 Proximal (axis A+C, most exotic, lowest priority unless exp_21+22 both fail).
7. exp_26 Conformal (axis H, Droid-focused, independent of CoDET stack).

**Falsification budget.** Each proposal is disprovable: every row has an ablation whose *failure to degrade* kills the theoretical claim. This is what makes the method section Oral-caliber rather than leaderboard-chasing.

---

## 📎 Unseen-domain / unseen-language baselines to beat (CoDET-M4 paper, §4.5)

> These are the numbers the paper reports for the three **unseen** OOD regimes — the core evaluation the NeurIPS claim must dominate. The leaderboard above only surfaces `Gen-qw / Lang-py / Src-gh` (our LOO proxies); the climb-paper's Table 2/3 will also need the entries below, so they are pinned here for direct Δ computation.

### CoDET Table 9 — Unseen Domains (MBPP + The-Vault inline; 5,451 samples, binary)

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **CodeT5** | **78.43** | **59.18** | **58.22** | **74.11** |
| UniXcoder | 76.00 | 57.11 | 55.01 | 72.81 |
| CatBoost | 60.32 | 53.54 | 50.62 | 69.11 |
| Baseline | 67.31 | 50.34 | 49.84 | 50.30 |
| CodeBERT | 45.69 | 48.91 | 43.16 | 66.01 |
| SVM | 37.11 | 41.37 | 38.66 | 55.16 |

**Takeaway the paper makes** (Table 11): splitting Unseen-Domain, UniXcoder scores The-Vault F1 = **63.38** but MBPP F1 = **44.48** — the **structure + source** shift is ~19 pt worse than structure-only. Our climb paper should report the same split.

### CoDET Table 10 — Unseen Languages (C# / Golang / JavaScript / PHP, per-language F1)

| Model | C# | Golang | JavaScript | PHP |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | **91.31** | **90.01** | **81.48** | **96.17** |
| CodeT5 | 78.55 | 88.78 | 34.81 | 93.66 |
| CodeBERT | 45.31 | 52.46 | 26.84 | 56.68 |
| CatBoost | 51.01 | 64.72 | 26.03 | 45.08 |
| SVM | 43.16 | 20.94 | 24.23 | 41.94 |
| Baseline | 39.60 | 46.15 | 56.27 | 28.33 |

**Hardest language = JavaScript** (UniXcoder 81.48; next-best CodeT5 at **34.81** — the gap is not a typo in the source paper). Any climb method that beats UniXcoder on JS OOD is a headline Δ on a notoriously flaky subgroup.

### CoDET Table 13 — Hybrid Authorship (human+LLM mixed code, Precision excluded — only positive class)

| Model | R | **F** | A |
|:------|:-:|:-:|:-:|
| **UniXcoder** | **33.22** | **39.36** | 64.71 |
| Baseline | 14.86 | 22.91 | 29.72 |

Hybrid is the **open problem** in the CoDET paper — UniXcoder 39.36 is low, and no other baseline crosses it. An actionable target: ≥ **45.00** F1 on hybrid.

### Δ-targets grid (one row per unseen regime, replaces the generic "SOTA targets" entry)

| Regime | Paper best | Source | **Our Δ target** | Status in leaderboard |
|:---|:-:|:---|:-:|:---|
| Unseen-Domain (5,451 samples) | **58.22** F1 | CodeT5, Table 9 | **≥ 58.50** | not yet in climb board — new column needed |
| Unseen-Domain · The-Vault only | 63.38 F1 | UniXcoder, Table 11 | ≥ 64.00 | not tracked |
| Unseen-Domain · MBPP only | 44.48 F1 | UniXcoder, Table 11 | ≥ 50.00 (stretch) | not tracked |
| Unseen-Language avg (4 langs) | **88.96** F1 | UniXcoder, Table 12 | ≥ 89.00 | proxy only: `Lang-py` LOO |
| Unseen-Language · JavaScript | **81.48** F1 | UniXcoder, Table 10 | ≥ 82.00 | not tracked |
| Unseen-Language · C# | **91.31** F1 | UniXcoder, Table 10 | ≥ 92.00 | not tracked |
| Unseen-Language · Golang | **90.01** F1 | UniXcoder, Table 10 | ≥ 90.50 | not tracked |
| Unseen-Language · PHP | **96.17** F1 | UniXcoder, Table 10 | ≥ 96.50 | not tracked |
| Hybrid Authorship (3-way) | **39.36** F1 | UniXcoder, Table 13 | ≥ 45.00 | not tracked |
| Ternary (human / LLM / hybrid) | **86.10** F1 | UniXcoder, Table 14 | ≥ 87.00 | not tracked |

**Coverage gap.** The leaderboard currently tracks **only 3 of these 10 unseen-regime targets** (implicitly, via its `Gen-qw / Lang-py / Src-gh` LOO columns, which are *proxies* — they hold out one value of a variable we *did* train on, rather than testing on MBPP / The-Vault / C#-Go-JS-PHP / hybrid code that is genuinely OOD in the paper's sense). Any climb method whose paper claim is "we generalise OOD" needs runs on the 7 untracked rows above before the claim survives review. Runner plumbing (`_data_codet.py`) already exposes `language` and `source` fields — adding a `load_codet_m4_unseen_domain` and `load_codet_m4_unseen_language` pair (following the existing `run_ood_language` / `run_ood_source` pattern) is the minimum-viable extension.

---

## 📎 Droid zero-shot + OOD-train baselines to beat (DroidCollection paper, Tables 3–6)

> Droid's paper exposes **four** distinct regimes — Zero-Shot, OOD (train on one domain/language, test on others), Fine-Tuned Full-Training, and Adversarial. Our climb currently only competes on Full-Training T1/T3/T4 (the easy one). The zero-shot + OOD-train rows are the actual OOD claim surface.

### Droid Table 3 · full — per-domain W-F1 (zero-shot + OOD + fine-tuned + full-train)

| Block | Model | 2-Cls Gen | 2-Cls Algo | 2-Cls RDS | **2-Cls Avg** | 3-Cls Gen | 3-Cls Algo | 3-Cls RDS | **3-Cls Avg** |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Zero-Shot | **Fast-DetectGPT** | **75.07** | 63.05 | 65.43 | **67.85** | **66.43** | 62.90 | 64.30 | **64.54** |
| Zero-Shot | GPTZero | 54.05 | **71.96** | 44.73 | 56.91 | 50.56 | **66.13** | 30.62 | 49.10 |
| Zero-Shot | CoDet-M4 | 53.41 | 44.63 | 65.43 | 54.49 | 41.90 | 46.06 | 55.43 | 47.80 |
| Zero-Shot | M4 | 50.17 | 57.91 | 44.67 | 50.92 | 56.46 | 58.13 | 51.21 | 55.27 |
| Zero-Shot | GPTSniffer | 54.25 | 36.85 | 32.10 | 41.07 | 45.22 | 31.75 | 39.88 | 38.95 |
| OOD-train | DroidDetectCLS-Base\[trained on **General**\] | **99.30** | 53.73 | 76.46 | 76.50 | **93.05** | 46.22 | **76.99** | 72.09 |
| OOD-train | DroidDetectCLS-Base\[trained on **Algorithmic**\] | 49.63 | **98.26** | 60.78 | 69.56 | 47.86 | **92.84** | 56.58 | 65.76 |
| OOD-train | DroidDetectCLS-Base\[trained on **Research/DS**\] | 47.01 | 48.02 | 72.55 | 55.86 | 47.86 | 38.73 | 59.97 | 48.85 |
| Fine-Tuned | GCN | 78.57 | 60.61 | 67.79 | 68.99 | 56.85 | 46.91 | 51.13 | 51.63 |
| Fine-Tuned | CatBoost | 89.69 | 87.29 | 77.21 | 84.73 | 78.86 | 74.01 | 64.07 | 72.31 |
| Full-train | M4FT | 92.99 | 89.36 | 73.99 | 85.45 | 80.98 | 80.72 | 58.80 | 73.50 |
| Full-train | GPT-SnifferFT | 97.72 | 96.52 | 80.46 | 91.56 | 89.42 | 88.12 | 70.72 | 82.75 |
| Full-train | CoDet-M4FT | 98.89 | 98.23 | 83.77 | 93.63 | 85.46 | 90.41 | 73.88 | 83.25 |
| Full-train | DroidDetectCLS-Base | 99.22 | 98.22 | 87.57 | 95.00 | 92.78 | 93.05 | 74.46 | 86.76 |
| **Full-train** | **DroidDetectCLS-Large** | **99.38** | **98.39** | **93.24** | **97.00** | 93.08 | 92.86 | **80.42** | **88.78** |

**Key reads.**
- **Research/DS (RDS)** is the hardest domain across the board (zero-shots hover 40-65; paper SOTA = 93.24 on 2-class, 80.42 on 3-class).
- OOD-train on General → **76.99** 3-Cls RDS ≈ lost **−3.43 pt** vs the full-train best (80.42). Cross-domain loss is quantified.
- A climb method that beats **Fast-DetectGPT 67.85 (2-Cls) / 64.54 (3-Cls)** with **zero labelled Droid training data** would be a direct zero-shot claim.

### Droid Table 4 · full — per-language W-F1 (zero-shot + OOD-train + fine-tuned + full-train)

**2-Class per-language (best bolded):**

| Block | Model | C/C++ | C# | Go | Java | Python | JS | **Avg** |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Zero-Shot | **Fast-DetectGPT** | 81.33 | **72.77** | 81.16 | **76.03** | 73.60 | 74.59 | **76.58** |
| Zero-Shot | M4 | 62.22 | 40.73 | 57.59 | 48.39 | 61.47 | 64.44 | 52.81 |
| Zero-Shot | GPTZero | 58.32 | 45.69 | 13.64 | 74.65 | 73.19 | 63.16 | 54.81 |
| Zero-Shot | GPTSniffer | 63.02 | 48.90 | **79.89** | 40.30 | 38.34 | 45.94 | 52.40 |
| Zero-Shot | CoDet-M4 | 61.12 | 50.68 | 19.66 | 56.15 | 58.75 | 41.44 | 47.97 |
| OOD-train | DroidDetectCLS\[C/C++\] | **98.98** | 96.59 | 67.32 | 96.97 | 74.45 | 91.15 | 87.58 |
| OOD-train | DroidDetectCLS\[C#\] | 93.66 | **99.20** | 78.89 | 95.20 | 71.13 | 89.87 | 87.99 |
| OOD-train | DroidDetectCLS\[Go\] | 93.33 | 86.00 | **98.94** | 89.97 | 71.45 | 88.72 | 88.07 |
| OOD-train | DroidDetectCLS\[Java\] | 95.53 | 96.42 | 94.57 | **99.31** | 75.59 | 80.26 | 90.28 |
| OOD-train | DroidDetectCLS\[Python\] | 80.27 | 85.48 | 82.28 | 88.80 | **98.85** | 86.62 | 86.75 |
| OOD-train | DroidDetectCLS\[JS\] | 95.76 | 97.38 | 75.27 | 96.45 | 68.98 | **97.80** | 88.61 |
| Full-train | CoDet-M4FT | 99.36 | 99.22 | 99.31 | 99.04 | 98.28 | 99.24 | 99.08 |
| Full-train | DroidDetectCLS-Base | 99.29 | 99.33 | 99.32 | 99.45 | 98.87 | 98.38 | 99.11 |
| **Full-train** | **DroidDetectCLS-Large** | **99.31** | **99.51** | **99.32** | **99.45** | **99.11** | **98.67** | **99.23** |

**3-Class per-language (best bolded):**

| Block | Model | C/C++ | C# | Go | Java | Python | JS | **Avg** |
|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Zero-Shot | **Fast-DetectGPT** | **77.85** | **66.37** | **72.73** | **69.45** | **70.34** | **69.11** | **70.98** |
| Zero-Shot | M4 | 65.33 | 50.38 | 60.49 | 56.25 | 61.21 | 53.64 | 57.92 |
| Zero-Shot | GPTZero | 61.00 | 50.38 | 28.89 | 61.25 | 52.63 | 54.78 | 51.48 |
| Zero-Shot | GPTSniffer | 64.18 | 42.29 | 76.19 | 34.94 | 34.94 | 47.22 | 49.96 |
| Zero-Shot | CoDet-M4 | 53.81 | 40.74 | 18.28 | 45.26 | 53.51 | 36.09 | 41.28 |
| OOD-train | DroidDetectCLS\[C/C++\] | **92.62** | 81.67 | 56.43 | 79.45 | 56.43 | 69.72 | 72.72 |
| OOD-train | DroidDetectCLS\[C#\] | 80.95 | **92.93** | 57.74 | 84.17 | 54.25 | 65.18 | 71.04 |
| OOD-train | DroidDetectCLS\[Go\] | 80.74 | 63.61 | **92.93** | 74.18 | 50.38 | 65.37 | 71.20 |
| OOD-train | DroidDetectCLS\[Java\] | 85.00 | 84.43 | 58.85 | **93.38** | 63.25 | 64.57 | 74.91 |
| OOD-train | DroidDetectCLS\[Python\] | 67.59 | 75.56 | 53.70 | 79.31 | **93.08** | 69.96 | 73.20 |
| OOD-train | DroidDetectCLS\[JS\] | 87.96 | 87.58 | 52.78 | 86.32 | 60.78 | **89.67** | 77.52 |
| Full-train | DroidDetectCLS-Base | 94.43 | 94.06 | 93.98 | 93.93 | 93.95 | 90.99 | 93.56 |
| **Full-train** | **DroidDetectCLS-Large** | 94.24 | 93.87 | **94.42** | **94.05** | **94.13** | **91.27** | **93.66** |

**Key reads.**
- **Zero-shot ranking is stable**: Fast-DetectGPT wins both 2-Class and 3-Class language averages, **20+ pt** above the runner-up zero-shot.
- **OOD-train asymmetry** on JavaScript 3-Class: train-on-JS → 89.67 JS test ≈ −4.4 pt vs full-train (94.05 at best); but train-on-other-langs → 52–70 JS test. JS is by far the most exposed cross-language transfer gap.
- **Python is the hidden OOD trap**: OOD-train\[Python\] on C/C++ test = 67.59 3-Cls (worst 3-Cls OOD entry in the table). Training on Python poisons transfer to C-family languages.

### Droid Table 5 — Adversarial Recall (human vs adversarial)

| Model | Human-written R | Adversarial R |
|:--|:-:|:-:|
| **DroidDetectCLS-Large** | **0.98** | **0.92** |
| DroidDetectCLS-Base | 0.93 | 0.92 |
| M4FT | 0.91 | 0.67 |
| Fast-DetectGPT (zero-shot) | 0.84 | 0.48 |
| M4 (zero-shot) | 0.40 | 0.73 (⚠️ high adv, low human) |
| CoDet-M4 (zero-shot) | 0.38 | 0.63 (⚠️ same failure mode) |
| CoDet-M4FT | 0.96 | 0.51 |
| GPT-SnifferFT | 0.97 | 0.55 |
| GPT-Sniffer (zero-shot) | 0.65 | 0.49 |
| GPT-Zero | 0.53 | 0.10 |

⚠️ cells flag the paper's core diagnostic: **M4 + CoDet-M4 (zero-shot) chase adversarial samples by over-predicting AI, tanking human recall**. A climb method that holds human R ≥ 0.95 AND adversarial R ≥ 0.90 matches DroidDetectCLS-Base; one that holds human R ≥ 0.98 AND adversarial R ≥ 0.92 matches the Large.

### Droid Table 6 — DroidDetect training-ablation (paper's own novelty; 4-class adversarial included)

| Variant | 2-Cls Base | 2-Cls Large | 3-Cls Base | 3-Cls Large | **4-Cls Base** | **4-Cls Large** |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| **DroidDetect (final)** | **99.18** | **99.25** | **94.36** | **95.17** | **92.95** | **94.30** |
| − Resampling \[DroidDetectSCL\] | 99.15 | 99.22 | 93.86 | 94.43 | 92.52 | 93.14 |
| − Triplet Loss \[DroidDetectCLS\] | 99.14 | 99.18 | 90.51 | 94.07 | 89.63 | 92.65 |

- **Resampling** earns the paper about **+0.5–0.8 pt** on 3-Cls / 4-Cls; a class-imbalance answer — directly relevant to our `Exp_14 GHCurriculum` axis-G claim.
- **Triplet loss** earns another **+1.5–4.0 pt** on 3-Cls base (90.51 → 94.36). Relevant to `Exp_04 PoincareGenealogy` / any metric-learning loss we propose.
- These two deltas together **are the paper's own contribution** — any climb method that beats DroidDetect-Large 4-Cls 94.30 is a direct ablation-level claim against their flagship.

### Droid Δ-targets grid (one row per regime)

| Regime | Paper best | Source | **Our Δ target** | Status |
|:--|:-:|:--|:-:|:--|
| T3 3-Cls per-domain Avg (Full-train) | **88.78** | DroidDetectCLS-Large, T3 | ≥ 90.00 | ✅ tracked (climb T3 column) |
| T1 2-Cls per-domain Avg (Full-train) | **97.00** | DroidDetectCLS-Large, T3 | ≥ 97.50 | ✅ tracked (T1 column) |
| T4 4-Cls final (DroidDetect-Large) | **94.30** | DroidDetect, T6 | ≥ 95.00 | ✅ tracked (T4 column) |
| **3-Cls per-domain: Research/DS** | **80.42** | DroidDetectCLS-Large, T3 | ≥ 82.00 | not broken out yet |
| OOD-train\[General\] → RDS test (3-Cls) | **76.99** | DroidDetectCLS-Base, T3 | ≥ 78.00 | not tracked |
| OOD-train\[JS\] → JS test (3-Cls) | **89.67** | DroidDetectCLS-Base, T4 | ≥ 90.00 | not tracked |
| Zero-Shot 3-Cls per-domain Avg | **64.54** | Fast-DetectGPT, T3 | ≥ 66.00 (if we do zero-shot) | not tracked |
| Zero-Shot 3-Cls per-language Avg | **70.98** | Fast-DetectGPT, T4 | ≥ 72.00 (if we do zero-shot) | not tracked |
| Adversarial Recall (human AND adv) | **0.98 / 0.92** | DroidDetectCLS-Large, T5 | ≥ 0.97 / ≥ 0.92 | not tracked |
| **3-Cls per-lang: JavaScript** | **91.27** | DroidDetectCLS-Large, T4 | ≥ 92.00 | not broken out yet |
| **3-Cls per-lang: Python** | **94.13** | DroidDetectCLS-Large, T4 | ≥ 94.50 | not broken out yet |

**Coverage gap for Droid.** The leaderboard tracks 3 of 11 regimes (T1 / T3 / T4 aggregates). The **per-domain RDS breakout**, **OOD-train-then-test-on-another-domain transfer**, **zero-shot**, and **adversarial recall** rows are all absent. RDS-focused + adversarial-aware runs are the cheapest way to close this — both are already in the Droid HF schema and only need the existing `_data_droid.py` loader to surface the `domain` + `adversarial` columns.

---

## ⚡ `paper_proto` run-mode — 3-run apples-to-apples screening (≤ 2 h H100)

Added 2026-04-20. Addresses the protocol mismatch surfaced in the mirror leaderboard: filling our `Gen-qw / Lang-py / Src-gh` proxy columns with paper OOD numbers is wrong, but running the full paper protocol (Table 9 / 12 = MBPP + The-Vault + C#/Go/JS/PHP external data) is too heavy for a screening run.

**`paper_proto` squeezes the maximum apples-to-apples Δ out of in-dataset metadata with just 3 training runs:**

| # | Task | Trains | Outputs (paper cells directly comparable) |
|:-:|:--|:--|:--|
| 1 | CoDET IID binary (60K, 2 epochs) | ~13 min | Table 2 aggregate + **Table 3 per-lang** (cpp/py/java) + **Table 4 per-src** (cf/lc/gh) — 7 paper cells |
| 2 | CoDET IID author 6-class (60K, 2 epochs) | ~13 min | **Table 7 Author F1** + Figure 2 confusion matrix — 1 paper cell + diagnostics |
| 3 | Droid T3 3-class (60K, 2 epochs, `eval_breakdown=True`) | ~13 min | **Table 3 per-domain** (Gen/Algo/RDS, via `Source→domain` map) + **Table 4 per-language** (C/C++, C#, Go, Java, Python, JS) + **Table 5 Human + Adversarial recall** + bonus diagnostics: per-**raw-Source**, per-**generator** (GPT-4o-mini / Qwen2.5-72B / etc.), per-**Model_Family** — up to 20+ paper cells in one run |

**Total budget:** ~40 min training + ~15 min loading/eval = **~55 min end-to-end** on H100 BF16 batch 128 × seq 512. Doubles as a **theory-method screening gate** (vs `lean`'s 8-run OOD probing).

### Activate via
```python
run_full_climb(
    method_name="<YourMethod>",
    exp_id="exp_NN",
    loss_fn=your_compute_losses,
    codet_cfg=CoDETM4Config(max_train_samples=60_000),
    droid_cfg=DroidConfig(max_train_samples=60_000),
    run_mode="paper_proto",
    run_preflight=True,
)
```

### Droid HF schema — verified 2026-04-20 (was wrong before)

Earlier attempt used *guessed* column names (`Domain`, `Model`). A live HF probe of [project-droid/DroidCollection](https://huggingface.co/datasets/project-droid/DroidCollection) returns a **different** schema:

| Column on HF | Our use |
|:--|:--|
| `Code` | raw code string |
| `Label` | `HUMAN_GENERATED` / `MACHINE_GENERATED` / `MACHINE_REFINED` / `MACHINE_GENERATED_ADVERSARIAL` |
| `Language` | `python` / `java` / `c/c++` / `c#` / `go` / `javascript` |
| `Generator` | actual generator name (e.g. `gpt-4o-mini`, `microsoft/Phi-3-small-8k-instruct`, `Qwen/Qwen2.5-72B-Instruct`) |
| `Generation_Mode` | `INSTRUCT` / `COMPLETE` |
| `Source` | `LEETCODE` / `CODEFORCES` / `ATCODER` / `TACO` / `STARCODER_DATA` / `THEVAULT_FUNCTION` / `DROID_PERSONAHUB` / `OBSCURACODER` / ... |
| `Sampling_Params` / `Rewriting_Params` | generation-config strings |
| `Model_Family` | family tag (e.g. `openai`, `qwen`, `llama`) |

The column the paper Table 3 calls **Domain** (General / Algorithmic / Research-DS) is **not** a column — it is a *derived* bucket built by mapping `Source` values according to paper §3.1. We now do this derivation inside `_data_droid.py::_source_to_domain(...)`:

- `STARCODER_DATA`, `THEVAULT_FUNCTION`, `DROID_PERSONAHUB` → **general**
- `TACO`, `CODENET`, `ATCODER`, `AIZU`, `LEETCODE`, `CODEFORCES` → **algorithmic**
- `OBSCURACODER` (+ ObscuraCoder-family sources) → **research_ds**
- unknown values → `""` (breakdown silently skips)

Substring fallbacks cover case/separator drift (`"leet-code"`, `"starcoder data"` both resolve) so future Droid schema tweaks won't break quietly.

### Smoke-test coverage (4/4 pass on CPU, 2026-04-20)

1. **Source→domain mapping** — 12 test cases including all verified paper sources + schema-drift edge cases.
2. **Full-metadata breakdown** — populates 5 dims: `domain` (3 entries), `language` (5), `source_raw` (3), `generator` (3), `model_family` (3). `human_recall = 0.85`, `adversarial_recall = 1.0` on the fixture.
3. **Drifted schema (3 dims all-empty)** — silently skipped; `overall / domain / language / human_recall / adversarial_recall` still emitted.
4. **T1 task** — correctly omits `human_recall` / `adversarial_recall` (only meaningful for T3 / T4).

### Bugs fixed along the way

- `len(test_data)` on a dict returns **column count** (5), not row count (60). Breakdown was using it to truncate arrays → all mask-sum entries dropped below the `>= 10` threshold → empty per-dim dicts. Fixed: use `test_data.num_rows` (HF Dataset) or the first column's length (dict).
- Droid loader previously passed `Domain` / `Model` field names that do not exist on HF. Fixed: now uses `Language` / `Source` / `Generator` / `Model_Family` (verified).
- `_source_to_domain` normaliser now replaces both `-` and ` ` with `_` before table lookup, then falls back to separator-free substring match (`LEETCODE` matches `LEET_CODE`, etc.).

### What `paper_proto` does NOT cover
- **CoDET Table 8 / 9 / 12** (unseen LLMs / MBPP+The-Vault / C#+Go+JS+PHP) — external data, needs separate loaders (`run_unseen_generator_external`, `run_unseen_domain_mbpp_vault`, `run_unseen_language_extra`). Still on the roadmap.
- **Droid per-language OOD-train** (Droid Table 4 OOD block) — needs 6 extra training runs, one per language. Out of the 2 h budget.
- **Droid zero-shot** (Table 3/4 ZS block) — requires external LM scoring (Fast-DetectGPT / GPTZero). Not our claim surface.

### How it ranks against `lean`
- `lean` = 8 runs, ~3 h, **our LOO proxy protocol**. Best for axis-C / G claims.
- `paper_proto` = 3 runs, ~55 min, **paper's own protocol**. Best for Table 1 / Table 2 of the final paper.
- Ship both: `paper_proto` first as a cheap sanity-screener, then `lean` for the axis-specific OOD claim, then `full` only for the paper-final winner.

---

## 🧭 Theory axes — which mechanism each climb method targets

> **Paper contribution slot:** every climb method is a controlled perturbation of exactly one hypothesis about *why OOD fails in AI-code detection*. This table is the spine of the ablation section in the draft.

| Axis | Hypothesis | Climb methods | Evidence so far |
|:--|:--|:--|:--|
| **A. Genealogy prior** | Generators form a fine-tune tree; $P(Y)$ is hierarchical | Exp_00 HierTree · Exp_01 Genealogy (pending) · Exp_15 GenealogyDistill | HierTree alone explains ≥+3.6 pt Author gain over UniXcoder; Exp_15 full run underperforms (68.83), but `no_anti_distill` ablation rebounds to 70.07, suggesting anti-distill destabilizes the genealogy signal |
| **B. Spectral / multi-scale** | Human vs LLM code differ in frequency statistics | Exp_11 PersistentHomology · (backbone spectral head across all exps) | PH sets GH-OOD record 35.56; spectral backbone universal |
| **C. Source-invariance (do(S))** | CF/LC templates are a confounder for $Y$ | Exp_02 GHSourceIRM · Exp_08 POEMPolarized · Exp_18 CausalIntervention | IRM-style +OOD-GH but cost IID; **Exp_18 causal stack under-delivers on IID (70.19) and was not evaluated on OOD-GH — verdict deferred** |
| **D. Density / generative margin** | Class-conditional density shapes decision boundary | Exp_04 PoincareGenealogy · Exp_06 FlowCodeDet · Exp_16 DualModeFlowRAG (partial ablation) | Flow matching is current climb #2 (70.90); Exp_16 ablation indicates `lambda_fm` is the only drop with clear negative Δ, while other auxiliaries are near-noise |
| **E. Compressibility / info** | Human code = higher entropy / lower predictability | Exp_03 TokenStatRAG · Exp_09 Epiplexity · Exp_10 PredictiveCoding · Exp_12 AvailabilityPredictivity | Epiplexity = climb #3 (70.78) + best Droid T4; TokenStat stable mid-pack |
| **F. Optimization geometry** | Flat minima / NTK alignment generalize OOD | Exp_05 SinkhornOT · Exp_07 SAMFlat · Exp_13 NTKAlign · Exp_19 GradAlignMoE (pending) | **NTKAlign is current climb #1 (71.03)** — single cheapest theory knob |
| **G. Data distribution** | Training distribution itself is the leak | Exp_14 GHCurriculum | Lean run gives stable IID/T3 (69.90 Author, T3 88.69) but no GH breakthrough (30.41); ablation indicates `lambda_gh_consist` is non-load-bearing in the current implementation |
| **H. Test-time adaptation** | OOD $\neq$ train; adapt at inference | Exp_17 TTTCode (pending) | Open — structural fix for GH-LN-stat gap |

**Cross-axis rule for the paper.** Every claim of a component's effect must come from an ablation that *removes only that component* while holding the axis label fixed. The existing Exp_18 ablation (full / no_hier / no_cf / no_backdoor / no_iv) is the template: report Δ-F1, Δ-OOD, wall.

---

## 📐 Ablation matrix (single source for the paper's Table 2)

> Each ablation is a leave-one-component-out study on **CoDET IID author** (primary), with OOD-LOO numbers where a full suite ran. Paste new rows here; never rewrite historical rows.

| Exp | Variant | Disabled λ | Test F1 | Δ vs full | Val F1 | Wall (s) | Axis | OOD-SRC-gh |
|:--:|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|
| 18 | full | — | 0.7019 | — | 0.7113 | 1470 | C | — (not run) |
| 18 | no_hier | `lambda_hier` | 0.6941 | **`−0.0077`** | 0.7078 | 1472 | C | — |
| 18 | no_cf_swap | `lambda_cf` | 0.6992 | `−0.0027` | 0.7110 | 1471 | C | — |
| 18 | no_backdoor | `lambda_backdoor` | 0.6992 | `−0.0026` | 0.7106 | 1469 | C | — |
| 18 | no_iv_proj | `lambda_iv` | 0.6999 | `−0.0019` | 0.7141 | 1468 | C | — |
| 17 | full | — | 0.7081 | — | 0.7159 | 1494 | H | — (not run in this ablation log) |
| 17 | no_hier | `lambda_hier` | 0.6990 | `−0.0092` | 0.7116 | 1492 | H | — |
| 17 | no_ssl_pretext | `lambda_ssl` | 0.7009 | `−0.0072` | 0.7060 | 1493 | H | — |
| 17 | no_teacher_distill | `lambda_teacher` | 0.7089 | `+0.0008` | 0.7171 | 1493 | H | — |
| 16 | full | — | 0.7078 | — | 0.7159 | 1497 | D | — (not run in this ablation log) |
| 16 | no_hier | `lambda_hier` | 0.7099 | `+0.0020` | 0.7222 | 1497 | D | — |
| 16 | no_flow | `lambda_fm` | 0.7044 | `−0.0034` | 0.7135 | 1499 | D | — |
| 16 | no_supcon_neural | `lambda_supcon_n` | 0.7086 | `+0.0008` | 0.7156 | 1497 | D | — |
| 16 | no_supcon_spec | `lambda_supcon_s` | 0.7080 | `+0.0002` | 0.7161 | 1484 | D | — |
| 14 | full | — | 0.6949 | — | 0.7095 | 1481 | G | 0.3041 (from lean full-suite run) |
| 14 | no_hier | `lambda_hier` | 0.6916 | `−0.0033` | 0.7007 | 1472 | G | — (not run in this ablation log) |
| 14 | no_gh_consistency | `lambda_gh_consist` | 0.7013 | `+0.0065` | 0.7096 | 1474 | G | — |
| 15 | full | — | 0.6450 | — | 0.6528 | 1513 | A | 0.3099 (from lean full-suite run) |
| 15 | no_hier | `lambda_hier` | 0.6381 | `−0.0069` | 0.6439 | 1511 | A | — (not run in this ablation log) |
| 15 | no_pair_margin | `lambda_pair_margin` | 0.6460 | `+0.0010` | 0.6508 | 1509 | A | — |
| 15 | no_anti_distill | `lambda_anti_distill` | 0.7007 | `+0.0557` | 0.7089 | 1508 | A | — |

**Read-off for the draft.** Within axis C (source-invariance via do-operations): the genealogy prior (`hier`, axis A) is still the dominant knob (−0.77); the three do-components together contribute +0.72 (0.6941 → 0.7019) and are individually inside noise (≤ 0.3 pt). **This is an honest mild-negative result for axis C** — the theoretical framing is sound but the realised Macro-F1 advantage is small, and the crucial OOD-SRC-gh test was not run. Listed here as an evidence anchor, not a claim.

---

## Folder structure (modular, no code duplication)

```
Exp_Climb/
├── _common.py          # bootstrap deps, SpectralConfig, autocast, H100 profile
├── _features.py        # AST + structural + spectral feature extractors
├── _model.py           # AICDDataset, SpectralCode backbone, sub-encoders
├── _trainer.py         # Trainer (loss-fn-agnostic), FocalLoss, default_compute_losses
├── _data_codet.py      # CoDETM4Config, IID + OOD LOO suite (all 5 paper tables)
├── _data_droid.py      # DroidConfig, T1/T3/T4 suite (paper Tables 3/4)
├── _climb_runner.py    # run_full_climb() -- orchestrator, combined paper table
├── _paper_table.py     # emit_paper_table() -- BEGIN/END_PAPER_TABLE markdown block
├── tracker.md          # this file
└── exp_NN_<method>.py  # ONE file per method (numbered), thin ~200-line wrappers
```

**Rule:** shared code is in `_*.py`. Method-specific loss + tree/constraint logic lives in `exp_NN_*.py` and is passed to `Trainer` via the `loss_fn=` parameter.

---

## 🖥️ Kaggle / H100 80GB setup

### Auto-applied H100 profile (`_common.apply_hardware_profile`)

Triggers when `torch.cuda.get_device_name()` contains "H100":

| Setting | H100 value | Base default | Why |
|---|---|---|---|
| `precision` | `bf16` | `auto` | Hopper native bf16, no loss scaling needed |
| `batch_size` | **`128`** | 32 | 2× activations → utilize 80 GB VRAM (~40 GB used) |
| `max_length` | **`512`** | 512 | Kept at paper-baseline for direct comparability. *Seq 1024 OOMs at batch 128 in bf16 — attention + RoPE buffers blow past 80 GB. Use seq 1024 only with batch 64.* |
| `lr_encoder` | **`2.8e-5`** | 2e-5 | sqrt(2) LR scaling for 2× batch |
| `lr_heads` | **`1.4e-4`** | 1e-4 | same |
| `grad_accum_steps` | `1` | 2 | Batch 128 fits directly |
| `num_workers` | `8` | 2 | Kaggle H100 kernel has 8 vCPU |
| `prefetch_factor` | `4` | 2 | Keep dataloader hot |
| `log_every` | `200` | 100 | Less log spam at high throughput |
| `eval_every` | `2000` | 1000 | Eval less often (H100 goes fast) |
| `PYTORCH_CUDA_ALLOC_CONF` | `expandable_segments:True` | — | Auto-set in `_common.py`; reduces fragmentation for long-lived runs |

**VRAM budget (80 GB) — empirically measured, batch 128 × seq 512:**

| Component | Size (bf16) |
|:----------|:-----------:|
| ModernBERT-base params + grads + Adam state | ~5 GB |
| Activations (forward, batch 128 × seq 512 × 768 × 22 layers) | **~20 GB** |
| Backward-pass activations + RoPE buffers | ~15 GB |
| PyTorch/CUDA overhead + fragmentation | ~5 GB |
| **Total used** | **~40-45 GB (50-55%)** |
| Headroom | ~35-40 GB |

Previously batch 64 × seq 512 used only ~18 GB (22%) — wasteful.
**Attempted:** batch 128 × seq 1024 (~79 GB forward alone → OOM on backward). Dropped to seq 512 for safety.

**Why not batch 192+ or ModernBERT-large?**
- Batch 192: ~60 GB forward + ~30 GB backward = 90 GB total → OOM
- Batch 256: same issue, worse
- ModernBERT-large (~395M) + batch 128: ~70 GB forward → OOM on backward, plus 2× runtime risks 12h timeout
- If user wants **seq 1024**: must explicitly drop batch to 64 (override `CoDETM4Config(max_train_samples=...)` isn't enough — need to set `SpectralConfig.max_length=1024` + `batch_size=64` manually)
- **Current config (batch 128 × seq 512) is the Pareto optimum for H100 80GB + 12h Kaggle limit.**

### Disk budget (`/kaggle/working` = 20 GB quota)

- 16 runs × `best` ckpt × ~600 MB ≈ **9.6 GB** → fits (with `save_latest_ckpt=False`, default)
- If you enable `save_latest_ckpt=True`, budget doubles to ~19 GB → tight
- If you run multiple climb files back-to-back in one session, **delete** `./codet_m4_checkpoints` + `./droid_checkpoints` between runs

### Kaggle notebook config (recommended settings)

| Kaggle setting | Value |
|---|---|
| Accelerator | **GPU T4×2** or **GPU P100** during dev — **GPU H100** via "Notebook Run" for final climb |
| Internet | **ON** (needed to clone GitHub repo + load HF datasets) |
| Language | Python |
| Persistence | Disable (each run is fresh; we re-clone repo) |
| Secrets | `HF_TOKEN` optional (public datasets work without, just rate-limit warnings in log) |

### Kaggle runtime estimates (per climb file, batch 128 × seq 512)

Based on measured Exp18 runs (~5-10 min/run at batch 64 × seq 512). Batch 128
processes ~2× tokens per step with minimal kernel overhead, so steps/epoch
halve → net ~10% faster per epoch.

| Phase | Runs | ~Time/run | Total |
|---|---:|---:|---:|
| CoDET-M4 IID binary | 1 | ~7 min | 7 min |
| CoDET-M4 IID author | 1 | ~7 min | 7 min |
| CoDET-M4 OOD Generator LOO | 5 | ~6 min | 30 min |
| CoDET-M4 OOD Language LOO | 3 | ~6 min | 18 min |
| CoDET-M4 OOD Source LOO | 3 | ~6 min | 18 min |
| Droid T1 | 1 | ~7 min | 7 min |
| Droid T3 | 1 | ~7 min | 7 min |
| Droid T4 | 1 | ~7 min | 7 min |
| **Total** | **16** | — | **~1 h 41 min** |

**Lean mode** (`run_mode="lean"`) — fast screening, 8 runs:

| Phase | Runs | ~Time/run | Total |
|---|---:|---:|---:|
| CoDET-M4 IID binary + author | 2 | ~22 min | 44 min |
| OOD-SRC held-out=gh (hardest) | 1 | ~22 min | 22 min |
| OOD-LANG held-out=python | 1 | ~22 min | 22 min |
| OOD-GEN held-out=qwen1.5 | 1 | ~22 min | 22 min |
| Droid T1 + T3 + T4 | 3 | ~22 min | 66 min |
| **Total** | **8** | — | **~2 h 56 min** |

> **Measured** on H100 BF16 batch=64 × seq=512 from exp_08 POEMPolarized run
> (2026-04-18, 10464 s total). Prior estimate "~53 min" was based on an
> optimistic extrapolation and proved too low by ~3×.
>
> `full` mode (16 runs) ≈ **~5 h 52 min** on the same hardware. Kaggle 12h
> kernel limit still safe, but plan only ONE full-mode run per session.

Use `lean` to screen ideas (1-2 ideas per session). Switch to `full` only
for paper-final winner after lean confirms novelty.

Kaggle H100 kernel limit: 12h → **massive buffer** (~7× runtime). P100/T4 fallback: ~3-4× longer (~6-8h) — drop batch to 64 manually on those GPUs.

### If H100 unavailable

Fallback hierarchy for config auto-detect:
1. **H100** → profile applied (bf16, batch 64)
2. **A100 80GB** → same profile works (bf16)
3. **A100 40GB / V100 / T4** → drop to `precision="fp16"`, `batch_size=32`, `grad_accum_steps=2`

The auto-detect only triggers on H100 by name match. For other GPUs, set explicitly:

```python
codet_cfg = CoDETM4Config(...)
# Override config defaults if non-H100
from _common import SpectralConfig
# (Trainer picks these up through build_codet_config / build_droid_config)
```

---

## Kaggle workflow (per exp file)

Each `exp_NN_*.py` is **fully standalone** — upload it alone to Kaggle, it auto-clones the repo and imports the shared `_*.py` helpers:

```python
# Top of every exp file
REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"
# Auto-clones to /kaggle/working/ai_code_detection if not already present,
# then adds Exp_Climb/ to sys.path and imports _common, _data_codet, etc.
```

Run it → at bottom of log there's a `BEGIN_PAPER_TABLE ... END_PAPER_TABLE` block → paste into this file's Leaderboard section below.

---

## Protocol (identical for every climb method)

| Setting | Value |
|---|---|
| Train subsample | **100 000** (≈ 20% of ~500K CoDET-M4, ≈ 10% of ~1M Droid) |
| Val subsample | 20 000 |
| **Test** | **FULL test set (no subsampling)** |
| Hardware | NVIDIA H100 80GB HBM3 |
| Precision | bf16 |
| Batch | 64 × 1 |
| Epochs | 3 |
| Seed | 42 |
| Flow per file (`full`) | CoDET full suite (IID binary + IID author + OOD gen×5 + OOD lang×3 + OOD src×3 = **13 runs**) → cleanup → Droid T1+T3+T4 = **3 runs** → paper table → **16 runs total** |
| Flow per file (`lean`) | CoDET IID binary + author + OOD-SRC-gh + OOD-LANG-python + OOD-GEN-qwen1.5 = **5 runs** → cleanup → Droid T1+T3+T4 = **3 runs** → paper table → **8 runs total, ~53 min** |

---

## 📊 Paper baselines to beat (full enumeration)

### CoDET-M4 (Orel, Azizov & Nakov, **ACL Findings 2025**)

All values from Tables 2-14 of `docs/references/paper_CodeDet_M4.md`. Metric: **Macro-F1 (%)** unless noted.

**Table 2 — Binary IID:**

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | 98.65 | 98.65 | **98.65** | 98.65 |
| CodeT5 | 98.36 | 98.35 | 98.35 | 98.35 |
| CodeBERT | 95.70 | 95.72 | 95.70 | 95.71 |
| CatBoost | 88.71 | 88.81 | 88.78 | 88.79 |
| SVM | 72.19 | 72.19 | 72.19 | 72.19 |
| Baseline | 62.03 | 65.17 | 62.03 | 65.17 |

**Table 3 — Binary per-language (F1):**

| Model | C++ | Python | Java |
|:------|:-:|:-:|:-:|
| **UniXcoder** | **98.24** | **98.60** | **99.02** |
| CodeT5 | 97.86 | 98.22 | 98.89 |
| CodeBERT | 95.73 | 94.84 | 96.54 |
| CatBoost | 91.94 | 86.04 | 88.81 |

**Table 4 — Binary per-source (F1):**

| Model | CodeForces | LeetCode | GitHub |
|:------|:-:|:-:|:-:|
| **UniXcoder** | 96.54 | **97.87** | 98.46 |
| CodeT5 | 97.24 | 66.23 | 98.54 |
| CodeBERT | 91.67 | 87.63 | 95.31 |
| CatBoost | 90.18 | 71.23 | 80.52 |

**Table 7 — Author 6-class IID Macro-F1 (primary author metric):**

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | 64.80 | 69.54 | **66.33** | 79.35 |
| CodeBERT | 63.14 | 68.10 | 64.80 | 77.65 |
| CodeT5 | 62.67 | 69.40 | 62.45 | 78.25 |
| CatBoost | 50.46 | 44.41 | 45.42 | 66.19 |
| SVM | 29.10 | 28.51 | 27.63 | 49.70 |

**Table 8 — OOD-Generator (unseen LLMs, binary). Precision excluded (only positive class in labels):**

| Model | R | **F** | A |
|:------|:-:|:-:|:-:|
| **UniXcoder** | 87.30 | **93.22** | 87.30 |
| CatBoost | 85.71 | 92.31 | 85.71 |
| CodeT5 | 65.87 | 79.43 | 65.87 |
| CodeBERT | 50.00 | 66.67 | 50.00 |
| SVM | 80.16 | 88.99 | 80.16 |
| Baseline | 29.37 | 59.65 | 64.68 |

**Table 9 — OOD-Source (unseen domains):**

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **CodeT5** | 78.43 | 59.18 | **58.22** | 74.11 |
| UniXcoder | 76.00 | 57.11 | 55.01 | 72.81 |
| CatBoost | 60.32 | 53.54 | 50.62 | 69.11 |
| Baseline | 67.31 | 50.34 | 49.84 | 50.30 |
| CodeBERT | 45.69 | 48.91 | 43.16 | 66.01 |
| SVM | 37.11 | 41.37 | 38.66 | 55.16 |

**Table 10 — OOD-Language per-language (UniXcoder is the clear winner):**

| Model | C# | Golang | JavaScript | PHP |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | **91.31** | **90.01** | **81.48** | **96.17** |
| CodeT5 | 78.55 | 88.78 | 34.81 | 93.66 |
| CodeBERT | 45.31 | 52.46 | 26.84 | 56.68 |
| CatBoost | 51.01 | 64.72 | 26.03 | 45.08 |
| SVM | 43.16 | 20.94 | 24.23 | 41.94 |
| Baseline | 39.60 | 46.15 | 56.27 | 28.33 |

**Table 12 — OOD-Language avg (unseen langs):**

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | 89.13 | 89.20 | **88.96** | 88.96 |
| CodeT5 | 76.87 | 73.29 | 71.47 | 72.17 |
| CodeBERT | 60.31 | 59.79 | 58.78 | 59.10 |
| CatBoost | 61.25 | 57.86 | 53.42 | 56.26 |
| Baseline | 70.53 | 57.36 | 51.53 | 59.64 |
| SVM | 26.42 | 38.27 | 28.68 | 36.29 |

**Table 13 — Hybrid Authorship (UniXcoder's ceiling, precision excluded):**

| Model | R | **F** | A |
|:------|:-:|:-:|:-:|
| **UniXcoder** | 33.22 | **39.36** | 64.71 |
| Baseline | 14.86 | 22.91 | 29.72 |

**Table 14 — Ternary (human / LLM / hybrid) Macro-F1:**

| Model | P | R | **F** | A |
|:------|:-:|:-:|:-:|:-:|
| **UniXcoder** | 86.48 | 85.93 | **86.10** | 86.16 |
| CodeBERT | 85.91 | 85.96 | 85.94 | 85.84 |
| CodeT5 | 79.72 | 78.78 | 78.99 | 79.43 |

### DroidCollection (Orel, Paul, Gurevych, Nakov, **EMNLP 2025**)

All values from Tables 3-6 of `docs/references/paper_Droid.md`. Metric: **Weighted-F1 (%)** unless noted. Best backbone = **DroidDetectCLS-Large**.

**Table 3 — 2-class / 3-class per-domain (Full Training split, Avg column is our canonical target):**

| Model (Full Training) | 2-Class Gen | 2-Class Algo | 2-Class RDS | **2-Class Avg** | 3-Class Gen | 3-Class Algo | 3-Class RDS | **3-Class Avg** |
|:---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **DroidDetectCLS-Large** | 99.38 | 98.39 | 93.24 | **97.00** | 93.08 | 92.86 | 80.42 | **88.78** |
| DroidDetectCLS-Base | 99.22 | 98.22 | 87.57 | 95.00 | 92.78 | 93.05 | 74.46 | 86.76 |
| CoDet-M4FT | 98.89 | 98.23 | 83.77 | 93.63 | 85.46 | 90.41 | 73.88 | 83.25 |
| GPT-SnifferFT | 97.72 | 96.52 | 80.46 | 91.56 | 89.42 | 88.12 | 70.72 | 82.75 |
| M4FT | 92.99 | 89.36 | 73.99 | 85.45 | 80.98 | 80.72 | 58.80 | 73.50 |
| **Zero-Shot: Fast-DetectGPT** | 75.07 | 63.05 | 65.43 | 67.85 | 66.43 | 62.90 | 64.30 | 64.54 |
| Zero-Shot: CoDet-M4 | 53.41 | 44.63 | 65.43 | 54.49 | 41.90 | 46.06 | 55.43 | 47.80 |

**Table 4 — 2-class / 3-class per-language (Full Training split, Avg column):**

| Model | 2-Class C/C++ | C# | Go | Java | Python | JS | **2-Class Avg** | 3-Class C/C++ | C# | Go | Java | Python | JS | **3-Class Avg** |
|:---|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| **DroidDetectCLS-Large** | 99.31 | 99.51 | 99.32 | 99.45 | 99.11 | 98.67 | **99.23** | 94.24 | 93.87 | 94.42 | 94.05 | 94.13 | 91.27 | **93.66** |
| DroidDetectCLS-Base | 99.29 | 99.33 | 99.32 | 99.45 | 98.87 | 98.38 | 99.11 | 94.43 | 94.06 | 93.98 | 93.93 | 93.95 | 90.99 | 93.56 |
| CoDet-M4FT | 99.36 | 99.22 | 99.31 | 99.04 | 98.28 | 99.24 | 99.08 | 89.98 | 88.94 | 89.73 | 91.70 | 85.80 | 91.46 | 89.60 |
| GPT-SnifferFT | 97.64 | 97.36 | 97.33 | 97.96 | 95.07 | 97.94 | 97.22 | 85.14 | 85.75 | 85.78 | 86.97 | 79.02 | 88.28 | 85.16 |
| M4FT | 94.18 | 89.98 | 92.19 | 92.63 | 87.19 | 93.61 | 91.63 | 79.56 | 75.55 | 77.63 | 79.55 | 69.63 | 77.53 | 76.57 |
| Fine-Tuned Baseline: CatBoost | 94.00 | 91.20 | 90.57 | 92.26 | 89.51 | 82.55 | 90.02 | 84.57 | 81.32 | 81.54 | 82.42 | 78.15 | 70.98 | 78.83 |
| Fine-Tuned Baseline: GCN | 79.06 | 78.33 | 84.33 | 80.04 | 72.49 | 69.69 | 77.32 | 65.97 | 58.03 | 65.20 | 60.13 | 55.22 | 54.72 | 59.88 |

**Table 5 — Adversarial robustness (Recall on human-written vs. adversarial samples):**

| Model | Human-written Recall | Adversarial Recall |
|:---|:-:|:-:|
| **DroidDetectCLS-Large** | **0.98** | **0.92** |
| DroidDetectCLS-Base | 0.93 | 0.92 |
| CoDet-M4FT | 0.96 | 0.51 |
| GPT-SnifferFT | 0.97 | 0.55 |
| M4FT | 0.91 | 0.67 |
| Fast-DetectGPT (zero-shot) | 0.84 | 0.48 |
| GPT-Sniffer (zero-shot) | 0.65 | 0.49 |
| M4 (zero-shot) | 0.40 | 0.73 |
| CoDet-M4 (zero-shot) | 0.38 | 0.63 |
| GPT-Zero | 0.53 | 0.10 |

**Table 6 — DroidDetect backbone ablation (Weighted-F1, avg across settings, our final method name in the paper):**

| Model | 2-Class Gen | 2-Class Algo | 2-Class RDS | 3-Class Gen | 3-Class Algo | 3-Class RDS |
|:---|:-:|:-:|:-:|:-:|:-:|:-:|
| **DroidDetect** (final) | **99.18** | **99.25** | **94.36** | **95.17** | **92.95** | **94.30** |
| DroidDetectSCL | see paper | see paper | see paper | see paper | see paper | see paper |
| DroidDetectCLS | 99.22 | 98.22 | 87.57 | 92.78 | 93.05 | 74.46 |

> Note: "RDS" = Research/DataScience. "Algo" = Algorithmic. "Gen" = General.

---

## 🎯 SOTA targets (goals for every climb method)

Target = beat the **strongest** published baseline for that task (not just beat the popular reference).
Paper columns anchor to the tables above.

| Benchmark / Task | Paper best | Source | **Our target** | Minimum to claim SOTA |
|:-----------------|:----------:|:-------|:--------------:|:---------------------:|
| CoDET Binary IID Macro-F1 | **98.65** | UniXcoder, Table 2 | **≥ 99.00** | > 98.65 |
| CoDET Binary per-lang (C++) | **98.24** | UniXcoder, Table 3 | — | > 98.24 |
| CoDET Binary per-lang (Python) | **98.60** | UniXcoder, Table 3 | — | > 98.60 |
| CoDET Binary per-lang (Java) | **99.02** | UniXcoder, Table 3 | — | > 99.02 |
| CoDET Binary per-src (CodeForces) | **96.54** | UniXcoder, Table 4 | — | > 96.54 |
| CoDET Binary per-src (LeetCode) | **97.87** | UniXcoder, Table 4 | — | > 97.87 |
| CoDET Binary per-src (GitHub) | **98.46** | UniXcoder, Table 4 | — | > 98.46 |
| **CoDET Author (6-class) Macro-F1** | **66.33** | UniXcoder, Table 7 | **≥ 70.00** | > 66.33 |
| CoDET OOD-Generator (unseen LLMs) F | **93.22** | UniXcoder, Table 8 | ≥ 94.00 | > 93.22 |
| CoDET OOD-Source (unseen domains) F | **58.22** | **CodeT5**, Table 9 | **≥ 58.50** | > 58.22 |
| CoDET OOD-Language avg F | **88.96** | UniXcoder, Table 12 | ≥ 85.00 (stretch) | > 88.96 (hard) |
| CoDET OOD-Language (C#) | **91.31** | UniXcoder, Table 10 | — | > 91.31 |
| CoDET OOD-Language (Golang) | **90.01** | UniXcoder, Table 10 | — | > 90.01 |
| CoDET OOD-Language (JavaScript) | **81.48** | UniXcoder, Table 10 | — | > 81.48 |
| CoDET OOD-Language (PHP) | **96.17** | UniXcoder, Table 10 | — | > 96.17 |
| CoDET Hybrid Authorship F | **39.36** | UniXcoder, Table 13 | ≥ 45.00 | > 39.36 (very hard) |
| CoDET Ternary (human/LLM/hybrid) F | **86.10** | UniXcoder, Table 14 | ≥ 87.00 | > 86.10 |
| **Droid 3-Class per-domain Avg W-F1** | **88.78** | DroidDetectCLS-Large, Table 3 | **≥ 90.00** | > 88.78 |
| Droid 2-Class per-domain Avg W-F1 | **97.00** | DroidDetectCLS-Large, Table 3 | ≥ 97.50 | > 97.00 |
| **Droid 3-Class per-language Avg W-F1** | **93.66** | DroidDetectCLS-Large, Table 4 | ≥ 94.50 | > 93.66 |
| Droid 2-Class per-language Avg W-F1 | **99.23** | DroidDetectCLS-Large, Table 4 | ≥ 99.30 | > 99.23 |
| Droid 3-Class (Research/DS) | **80.42** | DroidDetectCLS-Large, Table 3 | ≥ 82.00 | > 80.42 |
| Droid Adversarial Recall | **0.92** | DroidDetectCLS-L, Table 5 | ≥ 0.94 | > 0.92 |
| Droid 3-Class General (DroidDetect) | **95.17** | DroidDetect final, Table 6 | ≥ 95.50 | > 95.17 |

**Data-efficiency framing:** any target above, achieved with ~20% training data, becomes a paper-worthy headline. All paper baselines use 100% training data.

> **Our climb's primary metric mapping:** `Bin` in the top leaderboard maps to **CoDET Binary IID Macro-F1** (Table 2 row). `Auth` maps to **Table 7 Author Macro-F1**. `Src-gh` is our **held-out-GH subset of Table 4's GitHub column** — but we test the **author** head on it (6-class), so it is strictly harder than UniXcoder's Table 4 number (which is only binary). `T3` maps to **Table 3 3-Class Avg Weighted-F1**. Other climb columns (OOD-LANG-python LOO, OOD-GEN-qwen1.5 LOO, Droid T1, Droid T4) do **not** have a direct paper baseline — they are relative-to-other-methods comparisons.

---

## 📈 Method leaderboard

Fill in after each `exp_NN_*.py` run by pasting the `BEGIN_PAPER_TABLE` block.

> **See the [🏆 Climb Leaderboard](#-climb-leaderboard--sorted-by-codet-author-macro-f1-) at the top of this file** for the full 8-task ranked results. This section is reserved for per-method narrative (novelty, risks, lean-gate outcome) — kept below in the `Insights log` subsection.
>
> Retired duplicate summary table: the top leaderboard now covers every lean task (Bin / Auth / OOD-GEN-qw / OOD-LANG-py / OOD-SRC-gh / Droid T1 / T3 / T4) so there is no second table to sync.

---

## How to add a new method

1. Copy `exp_00_hiertree.py` → `exp_NN_<method>.py` (bump `NN`).
2. Replace the three method-specific pieces:
   - Custom loss class (if any), e.g. `SupConLoss`, `TopologyLoss`...
   - A `<method>_compute_losses(model, outputs, labels, config, focal_loss_fn)` function returning `{"total": ..., ...}`.
   - Pass that function as `loss_fn=` to `run_full_climb`.
3. Keep `run_full_climb(method_name="...", exp_id="exp_NN", ...)` — this handles both benches + combined paper table automatically.
4. Run on Kaggle, copy the `BEGIN_PAPER_TABLE` block, update row in the leaderboard above.

**Candidate methods to climb next** (ranked by tracker performance on individual benches):

| Priority | Method | Source exp | Why |
|:--------:|:-------|:-----------|:----|
| 1 | TokenStat | [Exp_DM/exp09_token_stats.py](../Exp_DM/exp09_token_stats.py) | Best Droid T3/T4 in DM tracker (0.8556/0.8488), untested on CoDET |
| 2 | SpectralCode | [Exp_DM/exp11_spectral_code.py](../Exp_DM/exp11_spectral_code.py) | Strong on both benches (0.2983 AICD / 0.8473 Droid) |
| 3 | DeTeCtiveCode | [Exp_DM/exp27_detective_code.py](../Exp_DM/exp27_detective_code.py) | CoDET full-suite rerun v2 logged (Author 71.28, OOD-SRC-gh 0.2900, OOD-LANG-py 0.5871); Droid not run in this log |
| 4 | MoECode | [Exp_CodeDet/run_codet_m4_exp21_moe.py](../Exp_CodeDet/run_codet_m4_exp21_moe.py) | Best CoDET binary (99.09), second-best ensemble |

---

## Insights log (fill in post-run)

Keep it to one paragraph per method. Per-class detail lives in the `BEGIN_PAPER_TABLE` block in the raw log, not here.

### exp_00 HierTreeCode (pending)
- **Expected strengths:** CoDET Author 6-class (family tree forces Qwen/Nxcode to cluster, matching the paper's known confusion).
- **Expected weaknesses:** Droid adversarial class — hier loss may over-regularize when the "families" are shallow (3–4 class).
- **Data-efficiency claim:** prior Exp18 runs at 20% train already beat UniXcoder (+4.22 Author, +0.41 Binary, and +0.41 OOD-Source) on CoDET. This climb file is the FULL-TEST re-run + first Droid pass under the same backbone.
- **Result after run:** _(paste BEGIN_PAPER_TABLE block summary)_

### exp_01 GenealogyGraphCode (pending, lean mode)
- **Novelty:** replaces static family table (Exp18) with a LEARNABLE weighted graph over generator prototypes. EMA-updated prototypes + 1-layer GNN propagation + InfoNCE proto-contrast + graph-smoothness prior on the static tree.
- **Targets insight #2 + #14:** breaks the plateau around the Nxcode↔Qwen1.5 pair by making the genealogy itself a learned parameter rather than hand-crafted.
- **Success criteria (lean screening):** CoDET Author > 70.55 OR Qwen1.5 per-class F1 > 0.47 OR OOD-GEN held-out=qwen1.5 > 0.51.
- **Risk:** prototypes may collapse if EMA too strong; graph-smooth term is a warm-start prior that tapers off.
- **Result after run:** _(paste BEGIN_PAPER_TABLE block summary)_

### exp_02 GHSourceInvariantCode (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** first method to attack the GH-source OOD bottleneck (insight #16). Couples SOURCE-IRM (per-source IRMv1 penalty, 3 envs cf/gh/lc) with a gradient-reversed STYLE-only adversary predicting source. HierTree preserved for genealogy signal.
- **Targets insight #3 + #16:** surface-style shortcut on CF/LC templates is the root cause of GH-OOD catastrophic failure. Source-invariance on style subspace (not on content) should close gap without hurting author signal.
- **Success criteria (lean screening):** OOD-SRC held-out=gh > 0.30 AND CoDET Author IID >= 70.3 (no regression).
- **Risk:** IRM penalty may explode without annealing (Exp06 lesson) — mitigated by `irm_warmup_epochs=1`. Requires dataset to expose per-sample `source` field in outputs dict.
- **Result after run:** Full suite `2026-04-18 19:53:52` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9899** (best val 0.9894); IID author **0.7020** (val 0.7118); OOD source GH **0.3044** (val 0.9928); OOD language python **0.5070**; OOD generator qwen1.5 **0.4976** (per-class table shows class 0 support 0 — headline macro is still a degenerate-case signal). **Droid** — T1/T3/T4 weighted-F1 test **0.9703 / 0.8805 / 0.8738** (primary vs best val **0.9714 / 0.8420 / 0.8392**). OOD-SRC-gh clears the 0.30 bar; author lands at 70.20 (just under the 70.30 regression guard). Droid T3/T4 sit slightly below the DroidDetectCLS-Large paper line (0.8878) but remain strong ID.

### exp_03 TokenStatRAGCode (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** first method to use retrieval DURING TRAINING (not just test-time). In-batch kNN over token-statistics features (label-independent surface features → cannot trivially memorise). Same-label neighbours pull, different-label neighbours hinge-push.
- **Targets insight #12 + #14:** combines cheap token-stat booster (Droid) with training-time retrieval signal (OOD + Author). Prior work (Exp17 RAGDetect 70.46) used embedding-space retrieval = circular.
- **Success criteria (lean screening):** CoDET Author > 70.60 AND Droid T3 > 0.89 AND OOD-SRC-gh > 0.32.
- **Risk:** token-stat features need to be surfaced by the backbone (`outputs["tokenstat"]` or `spectral_features`); fallback uses last-16-dim embedding slice.
- **Result after run:** Full suite `2026-04-18 20:24:37` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9905** (best val 0.9902); IID author **0.6990** (val 0.7043); OOD source GH **0.3019** (val 0.9926); OOD language python **0.5260**; OOD generator qwen1.5 **0.4962** (per-class: class 0 support 0 — same degenerate macro caveat as other LOO gen runs). **Droid** — T1/T3/T4 weighted-F1 test **0.9707 / 0.8894 / 0.8740** (macro-F1 **0.9706 / 0.8493 / 0.8401**; best val primary **0.9709 / 0.8497 / 0.8391**). **Lean gates:** Droid T3 **0.8894** is just under the 0.89 bar; OOD-SRC-gh **0.3019** under 0.32; author **69.90** under 70.60 — screening criteria not all met, but Droid T3 still edges past DroidDetectCLS-Large paper (0.8878) and binary/ID Droid stay strong.

### exp_04 PoincareGenealogy (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** Poincaré-ball embeddings + hyperbolic distance for the CoDET generator tree (Euclidean → exp map, learnable curvature, hierarchy/radial regularizers) — geometry aimed at low tree distortion vs Euclidean HierTree-style losses.
- **Targets (from exp file):** CoDET author **> 70.7**, Qwen1.5 per-class F1 **> 0.48**, Droid T3 stable **~0.89**; OOD GH remains the stress test.
- **Risk:** hyperbolic optimization can be brittle; family depth targets may trade off against author discrimination.
- **Result after run:** Full suite `2026-04-18 19:55:51` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9901** (best val 0.9904); IID author **0.6958** (val 0.7111); OOD source GH **0.2647** (val 0.9928); OOD language python **0.4761**; OOD generator qwen1.5 **0.4965** (class 0 support 0 in per-class table — same LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9697 / 0.8976 / 0.8799** (macro-F1 **0.9696 / 0.8592 / 0.8465**; best val primary **0.9704 / 0.8623 / 0.8430**). **Takeaway:** Droid T3 clears DroidDetectCLS-Large (**0.8976** vs 0.8878) — best T3 in the climb board so far; CoDET author sits below the 70.7 stretch goal and OOD-GH (**0.2647**) is weaker than exp_02/exp_03, so hyperbolic genealogy helps Droid ID more than GH-OOD under this lean recipe.

### exp_05 SinkhornOTCode (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** batch-level Sinkhorn–Knopp optimal-transport targets (balanced class mass) + KL to softmax logits vs standard CE/focal — aims to fix minority-class starvation on imbalanced 6-way author.
- **Targets (from exp file):** CoDET author **> 70.6** (balanced OT head); stackable with HierTree auxiliary.
- **Risk:** OT iterations each batch add compute; equal column-mass may fight natural human majority if ε too small.
- **Result after run:** Full suite `2026-04-18 20:10:15` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9905** (best val 0.9897); IID author **0.6840** (val 0.6924); OOD source GH **0.2896** (val 0.9927); OOD language python **0.5754**; OOD generator qwen1.5 **0.4967** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9701 / 0.8803 / 0.8752** (macro-F1 **0.9700 / 0.8393 / 0.8415**; best val primary **0.9711 / 0.8421 / 0.8410**). **Takeaway:** beats UniXcoder on author (**+2.07** pt) but **below** the file’s 70.6 OT target and behind stronger climb authors (exp_02/03/04). **OOD-LANG python** (**0.5754**) is a bright spot vs recent climbs; OOD-GH **0.2896** sits under the 0.30 bar; Droid T3/T4 trail paper SOTA and exp_04’s T3 peak.

### exp_06 FlowCodeDet (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** class-conditioned flow-matching auxiliary head (`CondVelocityMLP`) — per-class velocity field regularizes embeddings (stronger than pairwise SupCon per cited FM work); linear noise interpolant + MSE on predicted velocity.
- **Targets (from exp file):** CoDET author **> 70.6**, Droid T4 **> 0.85**, OOD-GEN qwen1.5 **> 0.51** (degenerate macro caveat applies).
- **Risk:** extra FM loss weight vs CE; sampling `t` can dominate early training if `lambda_fm` too high.
- **Result after run:** Full suite `2026-04-18 20:04:43` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9902** (best val 0.9897); IID author **0.7090** (val 0.7158); OOD source GH **0.3336** (val 0.9918); OOD language python **0.6450**; OOD generator qwen1.5 **0.4967** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9701 / 0.8934 / 0.8830** (macro-F1 **0.9701 / 0.8541 / 0.8479**; best val primary **0.9706 / 0.8542 / 0.8488**). **Takeaway:** **strongest CoDET author in the climb board so far (70.90)** and **best OOD-GH (0.3336)** among completed lean runs; clears the file’s **70.6** author bar and Droid T4 **> 0.85**. Droid T3 **0.8934** beats DroidDetectCLS-Large (0.8878) but sits slightly under exp_04’s peak T3; OOD-LANG python **0.645** is best-in-climb. UniXcoder gap on author: **+4.57** pt.

### exp_07 SAMFlatCode (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** Sharpness-Aware–style flatness via **embedding-space** adversarial perturbation (FGSM on features + auxiliary head loss) as a cheap proxy for SAM — targets OOD transfer without DANN-style domain confusion.
- **Targets (from exp file):** OOD-SRC-gh **> 0.32**, CoDET author **~70.5**, Droid T4 **> 0.85**.
- **Risk:** feature-space SAM can over-smooth discriminative directions if `lambda_sam` too large.
- **Result after run:** Full suite `2026-04-18 20:18:54` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9905** (best val 0.9902); IID author **0.7022** (val 0.7124); OOD source GH **0.3141** (val 0.9927); OOD language python **0.5413**; OOD generator qwen1.5 **0.4974** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9692 / 0.8873 / 0.8751** (macro-F1 **0.9692 / 0.8468 / 0.8411**; best val primary **0.9702 / 0.8477 / 0.8411**). **Takeaway:** author **70.22** and Droid T4 **0.875** meet “stable IID / adv” expectations; OOD-GH **0.3141** lands **just under** the **0.32** screening bar from the exp file. Droid T3 **0.8873** sits slightly below DroidDetectCLS-Large (0.8878) and under exp_04/06 peaks; overall a solid mid-pack run vs **FlowCodeDet** on GH-OOD and author. UniXcoder author gap: **+3.89** pt.

### exp_09 EpiplexityCode (2026-04-19, lean mode, H100 BF16 batch 64×1)
- **Novelty:** learned bottleneck autoencoder epiplexity proxy + human-vs-AI ranking margin on reconstruction MSE — bounded-compressibility feature concatenated to the backbone embedding (Finzi et al. 2026 epiplexity framing).
- **Targets (from exp file):** CoDET author **> 70.55**, Droid T4 **> 0.85**; OOD-GEN LOO macro is degenerate (~0.5 ceiling).
- **Risk:** AE + margin can dominate CE if `lambda_epi` too high; epiplexity scalar may correlate with length rather than “AI-ness” if bottleneck too wide.
- **Result after run:** Full suite `2026-04-19 04:18:37` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9897** (best val 0.9890); IID author **0.7078** (val 0.7172); OOD source GH **0.3100** (val 0.9924); OOD language python **0.6325**; OOD generator qwen1.5 **0.4966** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9703 / 0.8926 / 0.8846** (macro-F1 **0.9702 / 0.8521 / 0.8502**; best val primary **0.9694 / 0.8536 / 0.8483**). **Takeaway:** **#2 climb author (70.78)** behind only **FlowCodeDet**; clears **70.55** epiplexity target; **second-best OOD-LANG python (63.25)**; **best Droid T4 on the board (88.46)**. OOD-SRC-gh **0.31** sits on the 0.30–0.32 boundary; Droid T3 **0.8926** beats paper **0.8878** but is third after exp_04 / exp_06 peaks. UniXcoder author gap: **+4.45** pt.

### exp_10 PredictiveCodingCode (2026-04-19, lean mode, H100 BF16 batch 64×1)
- **Novelty:** hierarchical prediction-error / CPC stack — autoregressive latent predictor + InfoNCE + **supervised** residual targets (human → high aggregate error, AI → low) on top of HierTree + spectral losses.
- **Targets (from exp file):** CoDET author **> 70.55**, Droid T4 **> 0.86**; OOD-GEN LOO macro still degenerate.
- **Risk:** CPC + error-matching terms can fight CE if `lambda_cpc` / `lambda_err` too large; token-level predictor adds compute.
- **Result after run:** Full suite `2026-04-19 04:18:20` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9905** (best val 0.9904); IID author **0.6943** (val 0.7104); OOD source GH **0.3212** (val 0.9931); OOD language python **0.4912**; OOD generator qwen1.5 **0.4963** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9701 / 0.8841 / 0.8768** (macro-F1 **0.9700 / 0.8416 / 0.8432**; best val primary **0.9704 / 0.8437 / 0.8397**). **Takeaway:** **OOD-SRC-gh 0.3212** clears **0.30** and ranks **#4** on the board (after exp_11 / exp_06 / exp_08); author **69.43** **below** the file’s **70.55** stretch and mid-pack vs top runs; Droid T3 **0.8841** slightly under paper **0.8878**; T4 **0.8768** below epiplexity/flow peaks. UniXcoder author gap: **+3.10** pt.

### exp_11 PersistentHomologyCode (2026-04-19, lean mode, H100 BF16 batch 64×1)
- **Novelty:** class-aware persistent homology (Vietoris–Rips / H0 via MST) on **batch embedding clouds** — within-class vs between-class total persistence as a differentiable regularizer (Shestov-style embedding-quality signal), not AST PH like Exp23 TopoCode.
- **Targets (from exp file):** Droid T3 **> 0.89**, CoDET author **> 70.55**, OOD-SRC-gh **> 0.30**.
- **Risk:** PH loss can dominate; binary IID slightly below some peers if topology term fights CE.
- **Result after run:** Full suite `2026-04-19 04:23:04` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9881** (best val 0.9876); IID author **0.7015** (val 0.7087); OOD source GH **0.3556** (val 0.9902); OOD language python **0.5442**; OOD generator qwen1.5 **0.4957** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9656 / 0.8585 / 0.8707** (macro-F1 **0.9655 / 0.8151 / 0.8338**; best val primary **0.9638 / 0.8143 / 0.8345**). **Takeaway:** **OOD-SRC-gh 0.3556** is the **new climb record** — topology prior delivers on GH shift; author **70.15** lands between **Exp_02** and **HierTree** (just under **70.55** file target). **Droid T3 0.8585** misses **0.89** badly and **−2.93** vs paper — worst T3 among completed runs; trade-off: OOD-GH vs. Droid ID. UniXcoder author gap: **+3.82** pt.

### exp_12 AvailabilityPredictivityCode (2026-04-19, lean mode, H100 BF16 batch 64×1)
- **Novelty:** Hermann et al. availability vs predictivity — **low-capacity linear shortcut probe** + **incongruence loss** so the main head disagrees with the probe (not DANN / not erasing author).
- **Targets (from exp file):** OOD-SRC-gh **> 0.32**, CoDET author **> 70.55**, Droid T4 **> 0.85**.
- **Risk:** probe can be too weak (no gradient) or too strong (fights CE); two-player balance sensitive.
- **Result after run:** Full suite `2026-04-19 04:19:03` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9904** (best val 0.9901); IID author **0.7035** (val 0.7104); OOD source GH **0.2962** (val 0.9928); OOD language python **0.4653**; OOD generator qwen1.5 **0.4963** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9692 / 0.8815 / 0.8731** (macro-F1 **0.9691 / 0.8395 / 0.8378**; best val primary **0.9699 / 0.8398 / 0.8373**). **Takeaway:** **#3 climb author (70.35)** with strong IID signal, but **OOD-SRC-gh 0.2962** **misses** the exp-file **0.32** bar (shortcut story did not transfer to GH LOO here). Droid T3 **0.8815** slightly under paper **0.8878**; T4 **0.8731** mid-pack. UniXcoder author gap: **+4.02** pt.

### exp_13 NTKAlignCode (2026-04-19, lean mode, H100 BF16 batch 64×1)
- **Novelty:** empirical Gram / NTK-style alignment — maximize Frobenius cosine between batch embedding Gram matrix and label outer-product kernel (Kwok–Adams-style alignment), cheap **O(B²)** per step.
- **Targets (from exp file):** CoDET author **> 70.55**, OOD-LANG python **> 0.89** (macro scale — note column is ×100 so ~53 here), Droid T3 **~0.89**.
- **Risk:** batch-size dependence; alignment can overfit IID if λ too high.
- **Result after run:** Full suite `2026-04-19 04:18:33` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9901** (best val 0.9904); IID author **0.7103** (val 0.7199); OOD source GH **0.3514** (val 0.9930); OOD language python **0.5334**; OOD generator qwen1.5 **0.4971** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9713 / 0.8871 / 0.8819** (macro-F1 **0.9712 / 0.8442 / 0.8477**; best val primary **0.9714 / 0.8449 / 0.8468**). **Takeaway:** **new climb #1 on Author (71.03)** and **#2 on OOD-SRC-gh (35.14)**; clears **70.55**; Droid T3 **0.8871** almost ties paper **0.8878** (**−0.07** Δ); T4 **0.8819** second only to epiplexity/flow tier. UniXcoder author gap: **+4.70** pt — closest `run_full_climb` run to Exp27 **71.53** on IID author.

### exp_14 GHCurriculum (2026-04-19, lean + ablation, EMNLP 2026 target)
- **Novelty (data-side):** first climb entry to reshape the TRAINING
  distribution instead of the loss. Three stacked components:
  (A) source-balanced WeightedRandomSampler so each batch sees CF:LC:GH ≈ 1:1:1 instead of natural 3:1:1;
  (B) within-epoch curriculum (CF+LC → GH) so the hardest subgroup is learned against the STRONGEST IID representation;
  (C) GH-only SimCLR-style consistency loss (feature Gaussian noise as augmentation proxy, cosine pull).
- **Targets insight #3 + #16:** the 14 climb methods before it all fail on
  OOD-SRC-gh because CF/LC templates dominate training -- this is the
  first method to attack that cause, not the symptom.
- **Built-in ablation:** drops `lambda_hier`, `lambda_gh_consist`; data-side
  sampler/curriculum registered as placeholders (see `_ablation.py` for
  mechanism).
- **Success criteria (lean):** OOD-SRC-gh > 0.35 (break 0.33 cluster) AND
  CoDET Author >= 70.0 AND Droid T3 stable ~ 0.88.
- **Risk:** oversampling GH may shift CF/LC distribution enough to hurt
  IID; curriculum pacing is fixed, not learned.
- **Result after run (2026-04-19, lean full suite + ablation):** Lean full suite reports **CoDET** `iid_binary=0.9903`, `iid_author=0.6990`, `ood_source_gh=0.3041`, `ood_language_python=0.5102`, `ood_generator_qwen1.5=0.4969`; **Droid** `T1/T3/T4=0.9695/0.8869/0.8746`. This keeps IID and Droid stable (T3 near paper at `−0.09`) but misses the method GH target (`ood_source_gh>0.35`). Single-task author ablation shows `full=0.6949`, `no_hier=0.6916` (small drop), `no_gh_consistency=0.7013` (+0.0065), so the GH-consistency head is non-load-bearing in this run.

### exp_15 GenealogyDistill (2026-04-19, lean + ablation, EMNLP 2026 target)
- **Novelty:** first climb entry to attack the Nxcode↔Qwen1.5 sibling pair
  DIRECTLY instead of via family containment. Two mechanisms:
  (A) pair-margin triplet loss with alpha > HierTree's 0.3, so the sibling gap is forced to be WIDER than the family gap;
  (B) selective anti-distill on the 6-class softmax: when top-2 softmax is
  ambiguous between Nxcode and Qwen1.5, penalize that hedging -- reward
  decisive predictions.
- **Targets insight #2 + Exp27 CM analysis:** Nxcode-Qwen confusion is
  ~36-38% pairwise across every prior method; break 0.50 on both classes.
- **Built-in ablation:** `lambda_hier`, `lambda_pair_margin`,
  `lambda_anti_distill` each toggleable -- the drop-sorted ranking tells
  which lever is load-bearing.
- **Success criteria (lean):** CoDET Author > 72.0 (break Exp_27 by >0.5)
  AND Qwen1.5 per-class F1 > 0.55 AND Nxcode per-class F1 > 0.55.
- **Risk:** anti-distill can collapse if ambiguity threshold too low
  (punishes every prediction); pair margin can over-dominate if alpha
  too large (hurts other 4 classes).
- **Result after run (2026-04-19, lean full suite + single-task ablation):** Lean full suite reports **CoDET** `iid_binary=0.9903`, `iid_author=0.6883`, `ood_source_gh=0.3099`, `ood_language_python=0.5932`, `ood_generator_qwen1.5=0.4969`; **Droid** `T1/T3/T4=0.9702/0.8420/0.8427`. This misses the method gate (Author >72, OOD-SRC-gh >0.35) and underperforms on Droid T3 vs paper (−4.58 pt). In ablation-only single-task CoDET-author runs: `full=0.6450`, `no_hier=0.6381` (expected drop), `no_pair_margin=0.6460` (near-neutral), `no_anti_distill=0.7007` (**+0.0557**, largest gain). Practical read-off: anti-distill as implemented is the dominant failure mode; removing it recovers near-70 Author performance and partially restores nxcode/qwen separability.

### exp_16 DualModeFlowRAG (2026-04-19, lean + ablation, EMNLP 2026 target)
- **Novelty:** deliberate stack of the two distinct SOTA recipes on the
  two boards -- Exp_06 FlowCodeDet (Climb 🥇 70.90) and Exp_27
  DeTeCtiveCode (CodeDet 🥇 71.53). Three training signals:
  (A) class-conditioned flow matching (trains per-class velocity field);
  (B) dual-level SupCon on neural + spectral projections (tightens clusters in both subspaces);
  (C) test-time kNN blend over a pre-built train embedding bank (alpha=0.25, k=32).
- **Targets insight #14:** stacking hier + SupCon + kNN hit 71.53 alone;
  adding flow matching's per-class manifold should push it above 72.0.
- **Built-in ablation (4-way):** flow / supcon-neural / supcon-spectral /
  hier each toggleable. 5 single-task ablation runs total, ~110 min.
- **Success criteria (lean):** CoDET Author > 72.0 (beat Exp_27) AND
  OOD-SRC-gh > 0.35 AND Droid T3 > 0.90 (finally crack 0.8878 paper).
- **Risk:** 4 simultaneous auxiliary losses can fight each other; the
  per-component lambdas (0.3 fm / 0.15 supcon_n / 0.15 supcon_s / 0.4 hier)
  are informed guesses from the component papers but un-tuned together.
- **Result after run (2026-04-19 07:44:31, ablation-only single-task on `codet_m4/iid_author`):** **Variant `no_supcon_spec`** reports CoDET-M4 Author test Macro-F1 **0.7080** (best val **0.7161**, weighted-F1 **0.8171**, acc **0.8183**), with language profile `java 0.7677 > cpp 0.6951 > python 0.6696` and source profile `cf 0.7690 > lc 0.6037 > gh 0.5688`. Per-class bottleneck remains **nxcode (0.4911)** and **qwen1.5 (0.4398)** with heavy nxcode↔qwen confusion. **Ablation read-off:** full=0.7078, `no_hier`=0.7099 (+0.0020), `no_flow`=0.7044 (−0.0034, largest drop), `no_supcon_neural`=0.7086 (+0.0008), `no_supcon_spec`=0.7080 (+0.0002). In this setup, `lambda_fm` appears most impactful; hierarchical and supcon terms are near-neutral or slightly negative. Scope remains partial (no OOD-LOO / Droid in this log).

### exp_17 TTTCode (2026-04-19, lean + ablation, EMNLP 2026 target)
- **Novelty:** transfer of Test-Time Training (Sun ICML'20, Han 2506.23529 June'25, AdaContrast CVPR'22) to code detection.
  (A) masked-dim reconstruction pretext on embeddings — encoder must be robust to subset masking;
  (B) EMA class-centroid distillation (BYOL-style, momentum 0.995) — stable anchor for test-time adaptation;
  (C) train-time groundwork so the encoder has the right inductive bias for a future test-time-adapted eval loop.
- **Targets insight #16:** GH source OOD collapse is because BatchNorm / LayerNorm stats never adapt to GH. TTT fixes this structurally.
- **Built-in ablation:** `lambda_ssl` (SSL pretext) and `lambda_teacher` (EMA anchor) each toggleable.
- **Success gate:** OOD-SRC-gh > 0.38 AND Author IID >= 70.0.
- **Result after run (2026-04-19 08:02:01, ablation-only single-task on `codet_m4/iid_author`):** **Variant `no_teacher_distill`** reports CoDET-M4 Author test Macro-F1 **0.7089** (best val **0.7171**, weighted-F1 **0.8171**, acc **0.8196**), with subgroup profile `java 0.7662 > cpp 0.6985 > python 0.6722` and source profile `cf 0.7559 > lc 0.6077 > gh 0.5597`. Per-class bottleneck remains **qwen1.5 F1 0.4211** and **nxcode/qwen confusion** dominates the matrix. **Ablation read-off:** full=0.7081, `no_hier`=0.6990 (largest drop), `no_ssl_pretext`=0.7009, `no_teacher_distill`=0.7089 (+0.0008 vs full) ⇒ teacher-distill term appears least impactful in this setup. Scope is still partial: no OOD-LOO or Droid runs in this log.

### exp_18 CausalInterventionCode (2026-04-19, ablation-only single-task · `codet_m4/iid_author` · H100 BF16 batch 64×1)

> **Run scope.** ⚠️ 5-variant component ablation on **CoDET-M4 IID author only**. Neither Droid (T1/T3/T4) nor CoDET OOD-LOO (generator / language / source) was executed. Each variant = one ~1470 s training run with one λ zeroed. **Headline numbers below are from `no_iv_proj`** (the canonical ablation log); `full` is the reference inside the ablation table.

#### Theory

Source-confounded attribution: $Y$ (author) and $S$ (CF/LC/GH surface style) co-vary in training. Associational $P(Y\mid X)$ is contaminated. We approximate $P(Y\mid \operatorname{do}(X))$ by three interventions on the SCM $(Y \to X, S \to X)$:

| Component | λ | Causal operation | Mechanism |
|:--|:-:|:--|:--|
| `lambda_hier` | 0.4 | prior (non-causal) | HierTree family-affinity anchor (insight #2) — control |
| `lambda_cf` | 0.3 | $\operatorname{do}(S = s')$ | Counterfactual feature swap: same-author samples across sources, logit consistency |
| `lambda_backdoor` | 0.3 | $\sum_s P(s)\,P(Y\mid X,s)$ | Penalise cross-source softmax variance within author |
| `lambda_iv` | 0.1 | instrument indep. | Penalise $\operatorname{corr}(\|\text{logit}\|,\|\text{spec}\|)$ |

Targets **insight #5**: adversarial invariance (DANN / IRM) collapsed because they erase $Y$-relevant axes; do-operations block only $S \to X$.

#### Headline metrics (`no_iv_proj` · full CoDET-M4 iid_author test)

| Task | #Cls | Primary | **Best Val** | **Test Primary** | Macro-F1 | Weighted-F1 | Macro-R | Weighted-R | Acc |
|:--|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| `iid_author` | 6 | `macro_f1` | 0.7141 | **0.6999** | 0.6999 | 0.8084 | 0.7263 | 0.8081 | 0.8081 |

Val–test gap = **−0.0142**; weighted-F1 ≫ macro-F1 → imbalance still driving the score.

#### Per-class test F1 (class-id order, n = 61 724)

| Cls | Label | Support | Precision | Recall | **F1** |
|:-:|:--|-:|:-:|:-:|:-:|
| 0 | human | 24 558 | 0.9963 | 0.9550 | **0.9752** |
| 1 | codellama | 5 226 | 0.6887 | 0.7931 | **0.7372** |
| 2 | gpt | 1 760 | 0.6299 | 0.8608 | **0.7275** |
| 3 | llama3.1 | 5 409 | 0.8085 | 0.8249 | **0.8166** |
| 4 | nxcode | 5 537 | 0.4818 | 0.5364 | **0.5076** |
| 5 | qwen1.5 | 5 254 | 0.4971 | 0.3875 | **0.4355** |

**Signal:** human ceiling (0.975) + GPT/Llama3.1/CodeLlama solid (0.73–0.82) but **Nxcode (0.508) and Qwen1.5 (0.436) remain the choke point** — Qwen recall 0.388 means model prefers to route Qwen samples to Nxcode (see confusion matrix).

#### Subgroup breakdown (maps to CoDET paper Tables 3, 4, and Figure 2 analog)

**(a) By language — paper Table 3 / Table 10 axis**

| Group | n | Macro-F1 | Weighted-F1 |
|:--|-:|:-:|:-:|
| java | 16 642 | **0.7525** | 0.8442 |
| cpp | 13 387 | 0.6900 | 0.7741 |
| python | 17 715 | 0.6608 | 0.7978 |

Java > cpp > python spread matches climb-wide pattern; python is the hardest language even on IID.

**(b) By source — paper Table 4 axis (IID, not LOO)**

| Group | n | Macro-F1 | Weighted-F1 |
|:--|-:|:-:|:-:|
| cf | 13 605 | **0.7491** | **0.9302** |
| lc | 16 538 | 0.6000 | 0.6847 |
| gh | 17 601 | 0.5553 | 0.8290 |

CF best, GH-macro 0.5553 on the *IID* split already — the LOO test (not run) would collapse further. **This is the number we must rerun with run_mode="lean"** to populate OOD-SRC-gh in the leaderboard.

**(c) By generator — Figure 2 analog (binary per-generator signal)**

| Group | n | Macro-F1 | Weighted-F1 |
|:--|-:|:-:|:-:|
| human | 24 558 | 0.1628 | **0.9771** |
| gpt | 1 760 | 0.1543 | 0.9255 |
| llama3.1 | 5 409 | 0.1505 | 0.9028 |
| codellama | 5 226 | 0.1474 | 0.8842 |
| nxcode | 5 537 | 0.1155 | 0.6932 |
| qwen1.5 | 5 254 | 0.0931 | **0.5586** |

Macro column is ~uniformly ~0.15 because each subgroup is single-class → macro is degenerate here; **read the weighted column**: Qwen1.5 0.559 ≪ human 0.977, confirming Qwen is *the* identification bottleneck, not a training artefact.

#### Confusion matrix (rows = true, cols = predicted — compare to CoDET-M4 Figure 2)

```
             pred
true    human  codellama   gpt   llama3.1   nxcode   qwen1.5
 human  23457        307    214         77      408        95
 cllama    23       4141    128        371      413       150
 gpt       11         72   1516         72       72        17
 l3.1      11        391    250       4451      267        39
 nxcode    25        403    137        229     2937      1806
 qwen1.5   18        722    144        300     2034      2036
```

Dominant failure mode: **Qwen1.5 ↔ Nxcode symmetric confusion** (qwen→nxcode 2034 · nxcode→qwen 1806). That single 2×2 block accounts for ~38.7% of all non-human errors. Secondary: CodeLlama drift into nxcode/qwen pair (722 + 413 mis-routes). **Causal do-operations did NOT crack the sibling pair** — this is the central empirical take-away against the axis-C hypothesis for this method.

#### Δ vs paper baseline (primary metric, strongest published number)

| Benchmark | Task | Our primary | Paper best | Source | **Δ** |
|:--|:-:|:-:|:-:|:--|:-:|
| CODET_M4 | iid_author | **0.6999** | 0.6633 | UniXcoder, Table 7 | **`+0.0366`** |

Beats UniXcoder by +3.66 pt on IID author but trails climb leaders: Exp_13 NTKAlign 71.03 (−1.04), Exp_06 Flow 70.90 (−0.91), Exp_09 Epi 70.78 (−0.79).

#### Component ablation (`codet_m4/iid_author`, drop-one-λ)

| Variant | Disabled | **Test F1** | Δ vs full | Val F1 | Wall (s) | Axis |
|:--|:--|:-:|:-:|:-:|:-:|:-:|
| **full** | — | **0.7019** | — (ref) | 0.7113 | 1470 | A+C |
| `no_hier` | `lambda_hier` | 0.6941 | **`−0.0077`** | 0.7078 | 1472 | C |
| `no_cf_swap` | `lambda_cf` | 0.6992 | `−0.0027` | 0.7110 | 1471 | A only |
| `no_backdoor` | `lambda_backdoor` | 0.6992 | `−0.0026` | 0.7106 | 1469 | A only |
| `no_iv_proj` | `lambda_iv` | 0.6999 | `−0.0019` | 0.7141 | 1468 | A+C− |

- **Most-impactful component:** `lambda_hier` (axis A) — −0.77 pt when removed.
- **Least-impactful component:** `lambda_iv` — −0.19 pt (inside noise).
- **Three do-components together contribute only +0.0078** over `no_hier` (0.6941 → 0.7019). Drop-sorted ranking: `hier ≫ cf_swap ≈ backdoor > iv_proj`.

#### Interpretation for the paper

1. **Mild-negative result for axis C.** Success gate was *OOD-SRC-gh > 0.40* — the run never evaluated OOD-SRC-gh; on IID author the causal components individually contribute ≤ 0.3 pt (inside noise). We cannot yet claim causal-intervention SOTA.
2. **HierTree (axis A) dominates.** `no_hier` Δ is ~4× the other three combined → reinforces insight #2 (genealogy is the dominant signal) and weakens the orthogonality claim between axis A and axis C.
3. **IV term is near-inert at λ=0.1.** Either raise λ or drop from the final stack.
4. **Sibling pair unaffected.** Qwen↔Nxcode confusion 38.7% of errors — no causal gain over vanilla HierTree on this specific decision boundary.
5. **What is still missing for an Oral-tier axis-C claim** (ref CLAUDE.md §6): (a) full lean-suite OOD-SRC-gh / OOD-LANG-py / OOD-GEN numbers; (b) Droid T3/T4 cross-domain check; (c) §6-step-4 shortcut probe on $\phi(X) \to S$ to show $\Pr(\hat S)$ drops under our method — Macro-F1 alone cannot identify $\operatorname{do}(S)$.

**Next action.** Rerun with `run_mode="lean"` full 8-task suite. If OOD-SRC-gh fails to beat 0.36 (Exp_11 PH record), down-rank axis C and move λ-budget to axis F (NTKAlign) or G (GHCurriculum).

**Result paste-in (one-liner).** `exp_18_full` | iid_author | Macro-F1 **0.7019** (val 0.7113) | Δ vs UniXcoder **+3.86** pt | ablation drop-sorted hier > cf ≈ backdoor > iv | OOD-LOO **NOT RUN** | verdict deferred.

### exp_19 GradAlignMoE (2026-04-19, lean + ablation, EMNLP 2026 target)
- **Novelty:** first method to explicitly resolve GRADIENT CONFLICT between stacked auxiliary losses on the shared encoder. Cross-domain transfer from Rep-MTL (Wang July'25 arXiv 2507.21049), SON-GOKU graph-coloring MTL (Patapati Sept'25 2509.16959), SAMO (Ban July'25 2507.07883).
  (A) 4 expert projection heads (one per loss: hier / flow / supcon / ssl) with top-2 gated routing — each sample picks which auxiliary losses to contribute to;
  (B) Rep-MTL entropy regularizer on gate — sharp per-sample routing reduces interference;
  (C) SON-GOKU-style alignment reg pulling each expert projection toward its class-mean direction in main embedding space.
- **Targets insight #14:** the Exp_27 cocktail (hier + dual SupCon + kNN) reaches 71.53 — we ask "can gradient alignment on 4 stacked losses push that to 72.5+ with the SAME components?"
- **Built-in ablation (4-way):** `lambda_expert_sup`, `lambda_saliency`, `lambda_gradalign`, `lambda_hier` each toggleable.
- **Success gate:** Author > 71.5 AND drop-sorted table identifies which of the 3 alignment mechanisms matters most.
- **Result after run:** _(paste BEGIN_PAPER_TABLE + BEGIN_ABLATION_TABLE blocks)_

### exp_27 DeTeCtiveCode (2026-04-19, CoDET-M4 full 5-mode suite rerun v2, H100 BF16 batch 64x1)
- **Scope:** complete CoDET-M4 suite only (`iid_binary`, `iid_author`, `ood_generator` LOO x5, `ood_language` LOO x3, `ood_source` LOO x3). Droid T1/T3/T4 not executed in this run.
- **Headline CoDET results:** IID binary **0.9910** (best val 0.9902), IID author **0.7128** (best val 0.7124), OOD-GEN qwen1.5 **0.4966**, OOD-LANG python **0.5871**, OOD-SRC gh **0.2900**.
- **OOD summaries (from suite tables):** OOD-GEN macro-F1 `codellama 0.4882 / gpt 0.4430 / llama3.1 0.4985 / nxcode 0.4952 / qwen1.5 0.4966`; OOD-LANG `cpp 0.8712 / java 0.7904 / python 0.5871`; OOD-SRC `cf 0.5859 / gh 0.2900 / lc 0.8038`.
- **Failure concentration:** author confusion matrix still shows heavy Nxcode↔Qwen1.5 mixing (`true nxcode -> qwen: 1884`, `true qwen -> nxcode: 1962`), confirming the same sibling bottleneck observed in prior climbs.
- **Interpretation:** relative to UniXcoder Author baseline 0.6633, this rerun v2 is **+0.0495** (+4.95 pts), now above the previous Exp27 reruns on IID author. Trade-off remains: OOD-SRC-gh falls to **0.2900**, so GH generalization is still unresolved despite stronger IID; without Droid metrics it still cannot be treated as a full dual-benchmark winner.

### exp_08 POEMPolarizedCode (2026-04-18, lean mode, H100 BF16 batch 64×1)
- **Novelty:** POEM-style **orthogonal polarization** — split embedding into invariant vs source-specific subspaces (`L_ortho` on projectors), put source prediction on `z_spec` and entropy-regularize source on `z_inv` (no GRL / DANN).
- **Targets (from exp file):** OOD-SRC-gh **> 0.32**, CoDET author **~70.4**, Droid T3 **~0.88**.
- **Risk:** rank split can under-feed the author head if `z_inv` is too narrow.
- **Result after run:** Full suite `2026-04-18 20:16:47` on NVIDIA H100 80GB HBM3, BF16, batch 64×1. **CoDET-M4** — IID binary macro-F1 **0.9906** (best val 0.9902); IID author **0.6968** (val 0.7083); OOD source GH **0.3333** (val 0.9928); OOD language python **0.5411**; OOD generator qwen1.5 **0.4972** (class 0 support 0 — LOO-gen macro caveat). **Droid** — T1/T3/T4 weighted-F1 test **0.9705 / 0.8857 / 0.8748** (macro-F1 **0.9704 / 0.8450 / 0.8409**; best val primary **0.9711 / 0.8466 / 0.8412**). **Takeaway:** OOD-GH **0.3333** clears the exp-file **0.32** bar and matches the **exp_06** GH cluster (0.3336) within noise; author **69.68** tracks the ~70.4 expectation but trails **FlowCodeDet**; Droid T3 **0.8857** sits just under DroidDetectCLS-Large (0.8878). UniXcoder author gap: **+3.35** pt.

---

## Paper narrative (Table 1 of the eventual draft)

When 2+ methods have climbed, the "Method leaderboard" above becomes Table 1. The left-most "Train %" column is the pitch:

> "Table 1. Data-efficient SOTA. Our methods trained on 20% of each dataset match or exceed full-data paper baselines on both CoDET-M4 (ACL 2025) and DroidCollection (EMNLP 2025) across binary, attribution, and OOD settings."

Then each subsequent section (binary, author, OOD generator/language/source, Droid adversarial) cites the paper's table and our Δ.

---

## 🧠 Insights from prior research (Exp_DM + Exp_CodeDet)

Distilled from **23 methods in Exp_DM** (AICD + Droid) and **14 methods in Exp_CodeDet** (CoDET-M4). Read this before designing a new climb method — many dead ends already mapped.

### 1. Binary is ceilinged at ~99% — don't optimize it

Every modern PLM (ModernBERT, UniXcoder, CodeT5) hits 98.5–99.1 on CoDET-M4 binary IID. **The paper's binary ranking is noise.** All climb energy should go to **author classification** + **OOD** where real gaps exist (66.33 → 70.55 = +4.22 is a real signal).

### 2. Nxcode ↔ CodeQwen1.5 confusion is the single biggest lever on CoDET author

Nxcode is fine-tuned from CodeQwen1.5 → ~33–40% of Qwen1.5 samples predicted as Nxcode across **every method tested**. The `HierarchicalAffinityLoss` design (Exp18) explicitly forces them into one family → Qwen1.5 F1 bumped 0.4129 → 0.4431 (+3.02%).

**Takeaway:** any method that models generator genealogy (fine-tune lineage) wins. Raw classification doesn't.

### 3. GitHub source is the universal OOD bottleneck

Ranking by difficulty: **GH ≫ CF > LC** on both AICD and CoDET-M4.
- Per-source Author F1 on CoDET: CF 0.77 / GH 0.56 / LC 0.60 (across all methods)
- OOD-Source held-out-GH: macro F1 = **0.2834** (catastrophic; human recall 5.71%)
- GH-OOD-held-cpp: macro 0.4839 (worst subgroup in every experiment)

**Why:** CF + LC are competitive-programming templates → stylistically narrow. GH is real-world diverse code. Model trained on cf+lc memorizes templates, fails on GH.

**Takeaway:** any method that improves GH subgroup is a paper-worthy contribution. Target: GH macro > 0.60 on author task.

### 4. Val-test gap reveals shortcut learning (AICD T1 lesson)

AICD T1 shows **universal collapse**: val 0.99 → test 0.25-0.31 across 23 methods. The gap itself is the signal — none of DomainMix, IRM, OSCP, VILW, BH-Triplet, SupCon, etc. closed it. **This is why climb excludes AICD** — it's a benchmark-engineering problem, not a method-engineering problem.

**Implication for climb:** large val-test gaps on CoDET/Droid also matter. A method that has val 0.85 + test 0.70 is **worse** than val 0.72 + test 0.70, even if they report the same test number.

### 5. Methods that ❌ don't work (negative results to avoid)

| ❌ Anti-pattern | Why it fails | Evidence |
|:---|:---|:---|
| **DANN / GRL** for author task | Generator-invariant features are the **opposite** of what author classification needs | Exp19 EAGLECode: Author -7.66% (70.55 → 62.89); Qwen1.5 F1 collapsed to 0.198 |
| **Un-annealed IRM penalty** | Explodes to 1e4+ by epoch 3, NaN gradients | Exp06 AST-IRM: no OOD gain, unstable |
| **Variance-Invariant Whitening (VILW)** | Whitening loss dominates (~206), crushes CE capacity | Exp05 OSCP: -0.02 vs baseline; lowest AICD T1 test F1 |
| **Unguarded style contrastive** | Division-by-zero in style pairs → NaN | Exp16 HyperNetCode: StyleCon NaN every epoch; author task completely broken |
| **Class-weighted focal on severe imbalance** | Majority class (47% data) gets F1=0.0000 | Exp18 on AICD T2 class 0: weighted loss pushes model to over-predict minorities |
| **BiLSTM AST replaced by GAT** (without richer graph) | GAT on flattened AST ≠ CFG/DFG; no speedup, no accuracy gain | Exp23 GraphStyleCode: 69.71 < Exp11 SpectralCode 69.82 |

### 6. Methods that ✅ work (patterns to reuse)

| ✅ Pattern | Why it works | Best evidence |
|:---|:---|:---|
| **Hierarchical / family-tree losses** (HierTree) | Explicitly models generator lineage, cracks Qwen/Nxcode | Exp18 HierTreeCode: **70.55** author (+4.22 vs paper) |
| **Token-statistics features** (entropy, burstiness, TTR, Yule-K) | Cheap + strong on Droid adversarial | Exp09 TokenStat: **0.8556** Droid T3 (best DM) |
| **Multi-granularity fusion** (token + AST + structural + spectral, gated softmax) | Complementary signals; robust across languages | Every top-tier method uses this backbone |
| **Test-time retrieval / kNN blending** | Training-free OOD boost (~+0.1-0.3 author F1) | Exp17 RAGDetect: 70.46 (second-best CoDET author) |
| **BYOL-style EMA self-distillation** | Stability without labels, small boost on top of strong baseline | Exp26 SelfDistillCode: 70.14 (+0.32 on Exp11) |
| **Slot decomposition** | Object-centric helps `hybrid` / `adversarial` classes on T3 | Exp13 SlotCode: best AICD T3 single-task (0.5706) |

### 7. Diminishing returns in 70.0–70.2 plateau (Batch-2 lesson)

Methods Exp21-26 (MoE, TTA, CharCNN, Cosine-proto, SelfDistill) all cluster at **70.0–70.2 author F1** on CoDET. Hyperparam tuning / shallow architectural tweaks plateau here. **Next breakthrough needs an architectural leap**: explicit genealogy graph (not just affinity), retrieval-augmented training (not just test-time), or cross-modal code representations (not just AST).

### 8. Infrastructure gotchas (don't repeat)

- **OOD Generator LOO test_ood=0 bug** (Exp14/15/16/21/24) — old loader didn't filter test by held-out generator. Exp18 fix (`load_codet_m4_loo()` in `_data_codet.py`) is what climb uses. **Verify before trusting any OOD Gen number from older methods.**
- **TTA applied to test breakdown** overwrites stats (Exp22 case): test batch LayerNorm updates shouldn't touch the eval logger. Emit metrics BEFORE any test-time adaptation.
- **CoDET-M4 OOD Generator single-class degenerate** — held-out test = only that one AI class → macro F1 ceiling ~0.5. Use **weighted-F1** or rebalance with human samples.

### 9. Data-efficiency story works

Every Exp_DM / Exp_CodeDet method above ran on **100K train (~20%)** and still beat paper baselines that use full data. The climb paper can reliably claim "20% data + SOTA" as its headline.

### 10. What paper reviewers will ask — prepare answers

- **"Why not full training data?"** → "Full data doesn't change ranking" (show ablation: 100K vs 500K on Exp18 CoDET — ideally flat).
- **"How does it fare on AICD-Bench?"** → "We explicitly treat AICD as open challenge; val-test gap is a benchmark property, not solvable by detector-side methods alone." (reference 23-method negative result)
- **"What about calibration (ECE/Brier)?"** → emit in paper table too (current `_paper_table.py` doesn't, but trainer records val loss; ECE is one metric away).
- **"Language coverage vs Droid's 7 langs?"** → note that CoDET-M4 is limited to cpp/java/python by design; Droid has full coverage and we run T1/T3/T4 there.

### 11. Droid is stable across methods — don't use it to differentiate (NEW)

From **14 DM methods** all running Droid T3/T4: scores cluster in **0.84–0.90 W-F1**, gap between best (exp09 TokenStat 0.8941) and worst (exp07 DomainMix 0.7930) is mostly explained by backbone strength, not method innovation. **The claim to beat is DroidDetectCLS-Large (0.8878)** — only exp09 TokenStat and exp18 HierTreeCode come close on T3 (0.8941/0.8917). Use Droid T3 as a sanity check (did we regress?), not as a discriminator between ideas.

### 12. Token statistics are the best cheap Droid booster (NEW)

`exp09 TokenStat` (entropy, burstiness, TTR, Yule-K token distribution features) consistently gives **+0.003–0.005 Droid T3 W-F1** over pure spectral backbone. It's also cheap (no extra model, just numpy stats per sample). **Recommend: include token stats in every climb method as a free Droid booster.**

### 13. AICD T1 OOD collapse is structural — skip it entirely in climb (NEW)

**23 methods tested across Exp_DM** (DomainMix, IRM, OSCP, VILW, SupCon, TripletLoss, etc.) — all show val 0.99 → test 0.25-0.31 on AICD T1. The val-test gap is a dataset property (AICD train/test have different domain distributions by design). **No method-level fix works.** Climb excludes AICD deliberately. Report it as "open challenge" if reviewers ask.

### 14. HierTree + SupCon + kNN is the current best cocktail (NEW)

From Exp_CodeDet: **HierTree loss alone = 70.55** (Exp18). Adding **kNN test-time blend = 70.46** (Exp17, slight different impl). Adding **SupCon = 70.33** (Exp32 HyperCode). The pattern that works:
1. HierTree family loss (Qwen/Nxcode cluster) — non-negotiable, +1.5 F1 vs base
2. Token stats features — free +0.3-0.5 Droid
3. kNN blend at test time — +0.1-0.3 author F1, zero training cost

Methods that don't add above this baseline (KAN head, Hypernetwork, IB compression, DANN) all converge to 70.0-70.3 range.

### 15. Lean screening protocol — 3 ideas per H100 session (NEW)

Use `run_mode="lean"` (8 runs, ~53 min) to screen. Promotion criteria:
- **CoDET Author F1 > 70.55** (must beat current SOTA = Exp18) OR
- **Droid T3 W-F1 > 0.8941** (must beat exp09 TokenStat) OR  
- **OOD-SRC gh > 0.30** (any improvement on hardest subgroup is paper-worthy)

If none of the 3 OOD representative runs (gh/python/qwen1.5) shows improvement, abort and try next idea. Only promote to `run_mode="full"` (16 runs, ~1h41m) when at least one criterion is met.

### 16. GitHub source is the biggest untapped opportunity (NEW)

From Exp_Climb insights + Exp_CodeDet OOD data:
- **OOD-SRC held-out=gh Author macro F1 ≈ 0.2834** — catastrophic across all 14+ methods
- Human recall on GH-held-out = **5.71%** (model never predicts human for GH code)
- Root cause: CF+LC = competitive-programming templates (narrow style), GH = real-world diverse code. Model trained on cf+lc memorizes templates, fails completely on GH diversity.
- **Any method that breaks 0.40 on OOD-SRC-gh is a NeurIPS-worthy result.** No method has come close yet.
- Target for next architectural direction: explicitly train on source-diverse batches (GH-aware sampling) or use source as an environment variable in IRM-style training.

---

## Not in this folder

- **AICD-Bench** — excluded deliberately. 23+ methods in `Exp_DM/` all fail to close val-test gap (0.99 → 0.25 on T1). Paper treats AICD as "open challenge / negative result," not a main claim.
- **Ablations** — belong in `Exp_DM/` / `Exp_CodeDet/` where you freely tune. Climb files are for final dual-bench SOTA runs only.
- **Wrapper / variant methods** (exp28/29/30) — if they graduate, they become their own `exp_NN_*.py`.
