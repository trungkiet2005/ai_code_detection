# AICD_Bench — Master Tracker

> Single consolidated tracker for the entire project. Synthesizes findings from
> **23 Exp_DM methods** (AICD + Droid), **14+ Exp_CodeDet methods** (CoDET-M4),
> and **8 Exp_Climb lean runs** (CoDET + Droid dual-bench, 20% data).
>
> **Champion code copied into this folder:**
> - [exp05_flow_matching_climb.py](exp05_flow_matching_climb.py) — Climb 🥇 Exp_06 FlowCodeDet (CoDET Author 70.90 lean, best OOD)
> - [exp06_detective_codedet.py](exp06_detective_codedet.py) — CoDET 🥇 Exp27 DeTeCtiveCode (Author 71.53 — best overall in repo)
> - [exp07_spectral_code_dm.py](exp07_spectral_code_dm.py) — DM 🥇 exp11 SpectralCode (avg Macro-F1 0.5489 across 5 tasks)
>
> **Source trackers** (raw, in original folders):
> - `../Exp_Climb/tracker.md` — dual-bench lean
> - `../Exp_CodeDet/tracker.md` — CoDET-M4 full
> - `../Exp_DM/dm_tracker.md` — AICD + Droid 5-task suite
>
> Last sync: **2026-04-19**.

---

## 1. Champions per task

| Benchmark / Task | Metric | Champion | Score | Paper SOTA | Δ |
|---|---|---|---:|---:|---:|
| **CoDET-M4 Author IID (full)** | Macro-F1 | **Exp27 DeTeCtiveCode** | **71.53** | UniXcoder 66.33 | **+5.20** |
| CoDET-M4 Author IID (lean 20%) | Macro-F1 | Exp_06 FlowCodeDet | 70.90 | UniXcoder 66.33 | +4.57 |
| CoDET-M4 Binary IID | Macro-F1 | Exp17 / Exp21 / Exp31 | 99.09 | UniXcoder 98.65 | +0.44 |
| CoDET-M4 OOD-LANG python (lean) | Macro-F1 | Exp_06 FlowCodeDet | 64.50 | — | +11 over pack |
| CoDET-M4 OOD-SRC gh (lean) | Macro-F1 | Exp_06 / Exp_08 | 33.36 / 33.33 | — | breaks 0.30 ceiling |
| CoDET-M4 Per-language Java (binary) | F1 | Exp11 SpectralCode | 99.54 | UniXcoder 99.02 | +0.52 |
| CoDET-M4 Per-language C++ (binary) | F1 | Exp16 HyperNetCode | 99.12 | UniXcoder 98.24 | +0.88 |
| CoDET-M4 Per-source LeetCode | F1 | Exp15 GroupDRO | 99.35 | UniXcoder 97.87 | +1.48 |
| **Droid T3 (lean)** | Weighted-F1 | **Exp_04 PoincareGenealogy** | **89.76** | DroidDetectCLS-Large 88.78 | **+0.98** |
| Droid T3 (full DM) | Weighted-F1 | exp09 TokenStat | 89.41 | DroidDetectCLS-Large 88.78 | +0.63 |
| Droid T4 (4-class adversarial) | Weighted-F1 | Exp_06 FlowCodeDet | 88.30 | — | best in pack |
| Droid T1 (binary) | Weighted-F1 | Exp_00 HierTreeCode | 97.08 | — | ceiling (96.9–97.1) |
| **AICD T1 (binary, hardest)** | Macro-F1 | exp07 DomainMix | 0.3088 | — | **NO method solves OOD collapse** |
| AICD T2 (12-class family) | Macro-F1 | exp18 HierTreeCode | 0.2071 | — | class-0 still collapses |
| AICD T3 (4-class) | Macro-F1 | exp13 SlotCode | 0.5706 | — | hybrid class is bottleneck |

**Single best result across the project:** Exp27 DeTeCtiveCode (Author 71.53 on full CoDET-M4) — HierTree backbone + multi-level SupCon + optional kNN test-time blend.

---

## 2. Top-10 method leaderboard (consolidated)

Sorted by **CoDET Author Macro-F1** (the hardest, most discriminative metric).

| Rank | Method | Source tracker | Author F1 | Key innovation |
|:----:|:-------|:--------------:|:---------:|:----------------|
| 1 | Exp27 DeTeCtiveCode | tracker_06 | **71.53** | HierTree + multi-level SupCon + kNN |
| 2 | Exp_06 FlowCodeDet (lean) | tracker_05 | 70.90 | class-conditioned flow matching aux loss |
| 3 | Exp18 HierTreeCode (full) | tracker_06 | 70.55 | hierarchical family-tree affinity loss |
| 4 | Exp17 RAGDetect | tracker_06 | 70.46 | retrieval-augmented training + test-time kNN |
| 5 | Exp32 HyperCode | tracker_06 | 70.33 | Poincaré ball embeddings (NeurIPS'24 geometry) |
| 6 | Exp31 KANCode | tracker_06 | 70.30 | KAN heads (B-spline activations, ICLR'25 Oral) |
| 7 | Exp37 EnergyCode | tracker_06 | 70.26 | energy-margin OOD training |
| 8 | Exp_07 SAMFlatCode (lean) | tracker_05 | 70.22 | sharpness-aware flat minima |
| 9 | Exp_02 GHSourceInvariant (lean) | tracker_05 | 70.20 | source-invariant penalty |
| 10 | Exp20 BiScopeCode / Exp22 TTACode | tracker_06 | 70.20 | bi-scale heads / test-time augmentation |

**Plateau zone (70.0–70.2):** Exp21–26 (MoE, TTA, CharCNN, Cosine-proto, SelfDistill) all converge here. Hyperparam tuning + shallow architectural tweaks have hit a wall.

---

## 3. What WORKS — validated, reusable patterns

| ✅ Pattern | Why it works | Best evidence |
|---|---|---|
| **HierTree family loss** (cluster fine-tune-related LLMs) | Models generator lineage; cracks Qwen↔Nxcode confusion | Exp18: 70.55 author (+1.5 vs base); +3% Qwen F1 |
| **Token-statistics features** (entropy, burstiness, TTR, Yule-K) | Cheap (numpy), language-agnostic; +0.003–0.005 Droid T3 W-F1 | exp09 TokenStat: 89.41 Droid T3 (best DM) |
| **kNN / RAG blending at test time** | Training-free OOD lift, +0.1–0.3 author F1 | Exp17 RAGDetect: 70.46 |
| **Spectral / FFT features on token sequences** | Robust cross-domain transfer signal | exp11 SpectralCode: best AICD avg 0.5489 |
| **Class-conditioned flow matching aux loss** | First climb method to clear 70.6 author + best lean OOD-LANG-py | Exp_06 FlowCodeDet: 70.90 |
| **Multi-level SupCon (neural + spectral heads)** | Combined with HierTree → current SOTA | Exp27 DeTeCtiveCode: 71.53 |
| **Hyperbolic (Poincaré) embeddings on hierarchy** | Best Droid T3 ID stability | Exp_04 PoincareGenealogy: 89.76 |
| **Multi-granularity fusion** (token + AST + structural + spectral, gated) | Complementary signals, robust across languages | Every top-tier method uses this backbone |
| **BYOL-style EMA self-distillation** | Stability without labels; small boost on top of strong baseline | Exp26 SelfDistillCode: 70.14 (+0.32 over Exp11) |
| **Slot decomposition** | Object-centric helps `hybrid` / `adversarial` classes on T3 | exp13 SlotCode: best AICD T3 single-task (0.5706) |

**Current best cocktail (insight #14 from Exp_Climb):**
1. HierTree family loss — non-negotiable, +1.5 F1 vs base
2. Token-stats features — free +0.3–0.5 Droid
3. kNN blend at test time — +0.1–0.3 author F1, zero training cost

---

## 4. What FAILS — do not repeat

| ❌ Anti-pattern | Why it fails | Worst evidence |
|---|---|---|
| **DANN / GRL for author classification** | Generator-invariant features = OPPOSITE of what author classification needs | Exp19 EAGLECode: -7.66% (70.55 → 62.89), Qwen F1 collapsed to 0.198 |
| **Variance-Invariant Whitening (VILW)** | Whitening loss dominates (~206), crushes CE capacity | Exp05 OSCP: -0.02 vs baseline; lowest AICD T1 |
| **Class-weighted focal on severe imbalance** | Majority class (47% of data) gets F1=0.0000 | Exp18 on AICD T2 class 0: model over-predicts minorities |
| **Strict orthogonal cov penalty without warmup** | Penalty drives Cov→0 but kills usable features | Exp01 CausAST: AICD T1 0.2753 (worst DM-AICD) |
| **Un-annealed IRM penalty** | Explodes to 1e4+ by epoch 3, NaN gradients | Exp06 AST-IRM: no OOD gain, unstable |
| **Cross-view consistency without contrast** | Loss collapses to ~0, no learning signal | Exp04 BH-SCM xview→0 |
| **Unguarded style contrastive** | Division-by-zero in style pairs → NaN | Exp16 HyperNetCode: StyleCon NaN every epoch |
| **BiLSTM AST replaced by GAT (without richer graph)** | GAT on flattened AST ≠ CFG/DFG; no gain | Exp23 GraphStyleCode: 69.71 < Exp11 (69.82) |

---

## 5. Dataset insights

### CoDET-M4
- **Binary is ceilinged at ~99%** — every modern PLM hits 98.5–99.1. Don't optimize. The paper's binary ranking is noise.
- **Per-source author difficulty:** GH ≫ CF > LC. CF=0.77 / GH=0.56 / LC=0.60 across all methods. CF+LC = competitive-programming templates (narrow style); GH = real-world diverse code.
- **OOD-Source held-out=GH** is catastrophic: macro F1 ≈ 0.2834, human recall **5.71%**. THE stress test for any new method. **Any method breaking 0.40 on OOD-SRC-gh is NeurIPS-worthy.**
- **OOD-Generator LOO is degenerate by construction:** held-out test contains only ONE class → macro-F1 ceiling ~0.5. Use weighted-F1 or ignore entirely.
- **Nxcode ↔ CodeQwen1.5 confusion** is the single biggest lever: Nxcode is fine-tuned from CodeQwen1.5 → 33–40% of Qwen samples predicted as Nxcode in EVERY method. HierTree forces them into same family → +3% Qwen F1.
- **Languages limited to cpp/java/python by design.** Use Droid for full language coverage.

### Droid
- **Most methods cluster in 0.84–0.90 W-F1** on T3/T4. Backbone strength explains more variance than method innovation.
- **The bar to beat is DroidDetectCLS-Large (0.8878).** Only exp09 TokenStat (0.8941) and exp18 HierTreeCode (0.8917) clear it convincingly on T3.
- **Use Droid as a sanity check** ("did we regress?"), NOT as a discriminator between method ideas.

### AICD-Bench (the unsolved problem)
- **T1 OOD collapse is universal:** val F1 ≈ 0.99 → test F1 ≈ 0.25–0.31 across **23 methods** (DomainMix, IRM, OSCP, VILW, SupCon, TripletLoss, etc.). The val-test gap is a **dataset property** (train/test have different domain distributions by design), not a method-level bug. **No detector-side fix has worked.**
- **T2 class-0 collapse:** 47.7% of test data, F1=0.0000 under class-weighted focal. Macro-F1 (0.2071) ≪ weighted-F1. Fix is rebalancing, not architecture.
- **T3 hybrid class is the bottleneck:** human (0.7350) and adversarial (0.6512) separate well; hybrid sits at 0.3075.
- **Paper position:** climb intentionally excludes AICD; treat as "open challenge / negative result" if reviewers ask.

---

## 6. Pending experiments

### Exp_Climb (lean mode, file exists, not run)
Exp_01 GenealogyGraph · Exp_09 Epiplexity · Exp_10 PredictiveCoding · Exp_11 PersistentHomology · Exp_12 AvailabilityPredictivity · Exp_13 NTKAlign · Exp_14 GHCurriculum · Exp_15 GenealogyDistill · Exp_16 DualModeFlowRAG · Exp_17 TTTCode · Exp_18 CausalIntervention · Exp_19 GradAlignMoE.

EMNLP 2026 targeted (with built-in ablation infrastructure):
- **Exp_15 GenealogyDistill** — anti-distill on Nxcode/Qwen ambiguity. Target: Author > 72.0, Qwen + Nxcode F1 > 0.55.
- **Exp_16 DualModeFlowRAG** — stack Exp_06 (flow) + Exp_27 (HierTree+SupCon+kNN). Target: Author > 72.0 AND OOD-SRC-gh > 0.35 AND Droid T3 > 0.90.
- **Exp_17 TTTCode** — Test-Time Training (masked-dim reconstruction + EMA centroid). Target: OOD-SRC-gh > 0.38.
- **Exp_18 CausalIntervention** — first do-operations method (counterfactual swap + backdoor adjust + IV ortho). Target: OOD-SRC-gh > 0.40 (EMNLP headline).
- **Exp_19 GradAlignMoE** — gradient-conflict resolution on stacked aux losses. Target: Author > 71.5.

### Exp_CodeDet (full data)
- Exp35 TopoCode (only pending entry on full board).

### Exp_DM (5-task suite, files exist, runs pending)
exp02 TTA-Evident (calibration + TTA), exp10 MetaDomain (Reptile), exp12 WatermarkStat, exp19–exp30 (KAN, Hyper, IB, TTL, Topo, Mamba, Energy, WaveCL, DeTeCtive, HardNeg, RetrievalCalib, HierFocus).

---

## 7. Success bars / kill criteria

A new method is worth promoting iff it beats **at least one**:
- **CoDET Author Macro-F1 > 71.53** (current SOTA = Exp27 DeTeCtiveCode)
- **Droid T3 Weighted-F1 > 0.8941** (current SOTA = exp09 TokenStat)
- **CoDET OOD-LANG python Macro-F1 > 64.50** (current SOTA = Exp_06 FlowCodeDet)
- **CoDET OOD-SRC gh Macro-F1 > 33.36** (current SOTA = Exp_06 / Exp_08)
- **AICD T1 test Macro-F1 > 0.31** (would be the first to crack OOD collapse)

**Lean screening protocol** (insight #15): use `run_mode="lean"` (8 runs, ~3h on H100). If none of the 3 OOD representative runs (gh / python / qwen1.5) shows improvement, abort. Promote to `run_mode="full"` (16 runs, ~1h41m) only when at least one criterion is met.

---

## 8. Hardware / runtime profile (Kaggle H100 80GB)

**Auto-applied profile** (`_common.apply_hardware_profile`) when device name contains "H100":

| Setting | H100 value | Why |
|---|---|---|
| precision | bf16 | Hopper native, no loss scaling |
| batch_size | **128** | 2× activations → utilize 80 GB (~40 GB used) |
| max_length | 512 | seq 1024 OOMs at batch 128 in bf16 |
| lr_encoder / lr_heads | 2.8e-5 / 1.4e-4 | sqrt(2) scaling for 2× batch |
| grad_accum_steps | 1 | batch 128 fits directly |
| num_workers / prefetch | 8 / 4 | Kaggle H100 = 8 vCPU |
| eval_every | 2000 | Less eval overhead at high throughput |

**VRAM budget (batch 128 × seq 512, empirical):** model+grads+Adam ~5GB · forward activations ~20GB · backward ~15GB · overhead ~5GB → **~40–45GB used (50–55%)**, headroom ~35GB.

**OOM regimes:** batch 192/256 (90GB+), ModernBERT-large + batch 128 (~70GB forward, OOM on backward), batch 128 × seq 1024 (~79GB forward alone). Current is the Pareto optimum for H100 80GB + 12h Kaggle limit.

**Runtime:**
- `run_full_climb` (16 runs): ~1h 41m, ≤9.6 GB ckpts
- `lean` (8 runs): ~3h 18m (longer per run, fewer runs), ≤5 GB ckpts

**Disk gotcha:** `/kaggle/working` = 20 GB quota. With `save_latest_ckpt=True` budget doubles to ~19 GB → tight. Delete `./codet_m4_checkpoints` + `./droid_checkpoints` between back-to-back runs.

---

## 9. Infrastructure gotchas

- **OOD-Generator LOO `test_ood=0` bug** (Exp14/15/16/21/24, old loader) — didn't filter test by held-out generator. Fixed in `load_codet_m4_loo()` (`_data_codet.py`) which Exp18+ uses. **Verify before trusting any OOD-Gen number from older methods.**
- **TTA applied to test breakdown overwrites stats** (Exp22 case) — test batch LayerNorm updates shouldn't touch the eval logger. **Emit metrics BEFORE any test-time adaptation.**
- **CoDET-M4 OOD-Generator single-class degenerate** — held-out test = only that one AI class → macro-F1 ceiling ~0.5. Use weighted-F1 or rebalance with human samples.

---

## 10. Paper narrative & reviewer Q&A

### Headline pitch
> "With only **20% of the training data**, our methods match or exceed full-data paper baselines on two major AI-code-detection benchmarks (CoDET-M4 ACL 2025 + DroidCollection EMNLP 2025) across binary, attribution, and OOD settings."

### Paper-table mapping

| Paper section | Headline number | From |
|---|---|---|
| Main IID — CoDET Author | **71.53** Macro-F1 (Exp27) | tracker_06 |
| Main IID — Droid 3-class | **89.41** W-F1 (exp09) / **89.76** lean (Exp_04) | tracker_07 / tracker_05 |
| OOD generalization (lean) | **70.90** Author + **64.50** OOD-LANG-py (Exp_06) | tracker_05 |
| Robustness (adversarial) | **88.30** Droid T4 (Exp_06) | tracker_05 |
| Calibration (ECE / Brier) | TBD — exp02 TTA-Evident pending | tracker_07 |
| Efficiency (params/VRAM/time) | H100 batch 128 × seq 512, ~7 min/run | section 8 above |
| Ablations | HierTree / SupCon level / kNN blend toggles | tracker_06 + tracker_07 |

### Anticipated reviewer questions
- **"Why not full training data?"** → "Full data doesn't change ranking" (show ablation: 100K vs 500K on Exp18 CoDET — flat).
- **"How does it fare on AICD-Bench?"** → "We treat AICD as open challenge; val-test gap is a benchmark property (23-method negative result), not solvable by detector-side methods alone."
- **"What about calibration (ECE/Brier)?"** → emit in paper table (current `_paper_table.py` doesn't, but trainer records val loss; ECE is one metric away).
- **"Language coverage vs Droid's 7 langs?"** → CoDET-M4 is limited to cpp/java/python by design; Droid has full coverage (T1/T3/T4 all reported).

### Strategic recommendations
1. **Headline:** "20% data + dual-bench SOTA" — every Exp_DM / Exp_CodeDet method ran on ~100K train and beat full-data paper baselines.
2. **Next breakthrough needs an architectural leap** — explicit genealogy graph (not just affinity), retrieval-augmented *training* (not just test-time), or cross-modal code repr. Hyperparam tuning has plateaued at 70.0–70.2.
3. **GitHub-source OOD is the biggest untapped opportunity.** Explicit GH-aware sampling or source-as-environment IRM-style training is the most promising direction.
4. **AICD-Bench is not winnable from the detector side** — frame it as "open challenge" if you must include it.

---

## 11. Cross-cutting takeaways (one-liners)

1. **Binary is ceilinged at 99%** — don't waste compute on it; all ranking signal is in author + OOD.
2. **Nxcode ↔ Qwen is the single biggest CoDET lever** — any method modeling generator genealogy wins.
3. **GitHub source is the universal OOD bottleneck** — across both AICD and CoDET; templates (CF/LC) memorize, GH diversity breaks every model.
4. **Val-test gap is the shortcut-learning signal** — a method with val 0.85 + test 0.70 is **worse** than val 0.72 + test 0.70 even at the same test number.
5. **DANN / GRL hurt author** — invariance and identity are opposite goals.
6. **Hierarchical / family-tree losses are the strongest novelty pattern** so far.
7. **Droid is stable across methods** — sanity check, not discriminator.
8. **Token statistics are the best cheap Droid booster** — include in every method.
9. **AICD T1 OOD collapse is structural** — skip in climb; report as open challenge.
10. **HierTree + SupCon + kNN is the current best cocktail** (Exp27 = 71.53).
11. **Lean screening (8 runs, ~3h) before promoting to full** — 3 ideas per H100 session.
12. **GitHub source > 0.40 macro F1 = NeurIPS-worthy result** — currently 0.2834 across all 14+ methods.
