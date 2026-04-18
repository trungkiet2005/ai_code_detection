# Exp_Climb Tracker — Data-Efficient Dual-Bench Leaderboard

> **Strategy:** Each file trains ONE method on BOTH target benchmarks (**CoDET-M4** + **DroidCollection**) sequentially, using **~20% of the training data** while evaluating on the **full test set**.
>
> **Paper angle:** "With only 20% of training samples, our method matches or exceeds full-data baselines on two major AI-code-detection benchmarks (ACL 2025 + EMNLP 2025)."

---

## Why this folder

Separate from `Exp_DM/` (30+ methods, AICD+Droid) and `Exp_CodeDet/` (25+ methods, CoDET-M4). This folder is the **climb leaderboard** — only top-performing methods get a seat here, each giving us both benches in one file.

**Entry criterion:** a method must already beat paper baseline on at least one CoDET-M4 *or* Droid task in the big tracker. Only then does it graduate to `Exp_Climb/`.

---

## Protocol

| Setting | Value |
|---|---|
| Train samples | **100 000** (~20% of CoDET-M4 ~500K, ~10% of Droid ~1M) |
| Val samples | 20 000 (subsample) |
| **Test samples** | **FULL test set (no subsampling)** |
| Hardware | NVIDIA H100 80GB HBM3 |
| Precision | bf16 |
| Batch | 64 × 1 |
| Epochs | 3 |
| Flow per file | CoDET full suite (IID + OOD) → cleanup → Droid T1/T3/T4 → combined paper table |

### What gets evaluated

**CoDET-M4:**
- IID binary (Table 2, per-language, per-source breakdown)
- IID author 6-class (Table 7, per-generator confusion matrix)
- OOD Generator LOO × 5 (proxy Table 8)
- OOD Language LOO × 3 (proxy Table 10/12)
- OOD Source LOO × 3 (proxy Table 9)

**Droid:**
- T1 binary (human vs AI)
- T3 3-class (human / generated / refined — paper primary)
- T4 4-class (+ adversarial)

= **15 independent runs per method** covering both paper's main tables.

---

## Method leaderboard

| # | Method | File | CoDET Binary | CoDET Author | Droid T3 (W-F1) | Droid T4 (W-F1) | Data % | Status |
|:-:|:-------|:-----|:------------:|:------------:|:---------------:|:---------------:|:------:|:------:|
| 1 | **HierTreeCode** | [climb_hiertree.py](climb_hiertree.py) | TBD | TBD | TBD | TBD | 20% | ⏳ pending run |

**Paper baselines for reference:**

| Benchmark | Metric | Paper best | Ours target |
|:----------|:-------|:----------:|:-----------:|
| CoDET-M4 Binary | Macro-F1 | 98.65 (UniXcoder) | ≥ 99.00 |
| CoDET-M4 Author (6-class) | Macro-F1 | 66.33 (UniXcoder) | ≥ 70.00 |
| Droid T3 (3-class) | Weighted-F1 | 0.8878 (DroidDetectCLS-Large) | ≥ 0.89 |
| CoDET-M4 OOD Source | Macro-F1 avg | 55.01 | ≥ 55.00 |
| CoDET-M4 OOD Language | Macro-F1 avg | 88.96 | ≥ 80.00 (stretch) |

---

## How to add a method

1. Copy `climb_hiertree.py` as a template.
2. Replace the `SpectralCode` model + `HierarchicalAffinityLoss` with your method.
3. Keep the `Trainer`, `load_codet_m4_data`, `load_droid_data`, and `_main__` chaining intact — they're method-agnostic.
4. At the bottom of the run log, the file will auto-emit a **`BEGIN_PAPER_TABLE / END_PAPER_TABLE`** block covering both benches. Copy-paste that into this tracker.
5. Add a row to the leaderboard above.

---

## Insight log (fill in after each run)

_Paste key takeaways from each method run here. Keep it one paragraph per method — save the per-class detail for the PAPER_TABLE logs._

### climb_hiertree.py — HierTreeCode (pending)
- **Expected strengths:** CoDET author 6-class (family tree forces Qwen/Nxcode cluster).
- **Expected weaknesses:** Droid adversarial class if hier loss over-regularizes.
- **Data-efficiency claim:** prior Exp18 run at 20% train already beat UniXcoder (+4.22 author, +0.41 binary). This climb file verifies on FULL test without subsampling.
- **Result after run:** _(fill in)_

---

## Data-efficiency narrative for paper

When all climb methods are done, the `## Method leaderboard` above becomes **Table 1 of the paper's main results section**. The left-most column "Data %" is the pitch:

> "Table 1. HierTreeCode (ours) trained on 20% of training data matches or exceeds full-data paper baselines on both CoDET-M4 (ACL 2025) and DroidCollection (EMNLP 2025)."

Compare the CoDET-M4 paper (Orel et al. ACL 2025) — their baseline UniXcoder **uses 100% of training data** and scores 66.33 Author F1. If we get 70+ on 20%, the headline is data-efficiency + accuracy.

Same story for Droid: DroidDetectCLS-Large uses full training data to get 0.8878 Weighted-F1 on T3. Beating that with 10% training → another win.

---

## Appendix: what is NOT climbed here

- **AICD-Bench:** deliberately excluded. 23+ methods in `Exp_DM/` all failed to close the val-test gap (val 0.99 / test 0.25 on T1). The paper will treat AICD as "open challenge / negative result" rather than a main result. No climb file for AICD.
- **Ablations:** belong in `Exp_DM/` / `Exp_CodeDet/`. Climb files are for final SOTA runs only.
- **Wrapper methods** (exp28/29/30 variants of exp27): if they graduate to climb, they become their own climb file.
