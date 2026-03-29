# Performance Tracker

## Experiment Protocol (NeurIPS-oral target)

- Do **not** mix AICD and Droid in one training run.
- Train/evaluate **separate models per benchmark**:
  - `Benchmark A`: AICD-Bench (T1/T2/T3 as defined by benchmark).
  - `Benchmark B`: DroidCollection (T1 or T3 mapping).
- Report per-benchmark metrics independently, plus cross-benchmark transfer only as auxiliary study.

## Baseline Result (AICD T1)

| timestamp | benchmark | task | train | val | test | best_val_f1 | test_f1 |
|---|---|---|---:|---:|---:|---:|---:|
| 2026-03-29 12:24:05 | AICD-Bench | T1 | 100000 | 20000 | 50000 | 0.9948 | 0.2625 |

## EDA Snapshot (2026-03-29)

Source file: `docs/eda_two_benches.json` (API-sampled).

### Dataset scale

- **AICD-Bench**
  - T1: train 500,000 / val 100,000 / test 1,108,207
  - T2: train 502,149 / val 101,176 / test 507,874
  - T3: train 900,000 / val 200,000 / test 1,000,000
- **DroidCollection**
  - train 846,598 / dev 105,824 / test 105,826

### AICD-Bench T1 sample stats

- Sampled rows: train 2,000 / val 2,000 / test 200 (API-rate limited on test window).
- Label ratio (train): class `0`=933, class `1`=1067 (near-balanced).
- Code length (train): mean 805 chars, p50 456, p95 2648.
- LOC (train): mean 34.3, p50 21, p95 104.
- Style signal proxies:
  - comment-like presence: 42.7%
  - tab indentation presence: 43.7%
  - near-duplicate proxy: ~0.05%

### DroidCollection sample stats

- Train quick sample 500:
  - labels: MACHINE_GENERATED 211, MACHINE_REFINED 110, HUMAN_GENERATED 179
  - top languages: Python 182, Java 113, C# 52, C++ 50, JavaScript 43
  - top sources: STARCODER_DATA 109, THEVAULT_FUNCTION 100, TACO 97, DROID_PERSONAHUB 81
- Dev sample 2,000:
  - labels: MACHINE_GENERATED 825, MACHINE_REFINED 435, HUMAN_GENERATED 740
  - generation mode: INSTRUCT 1162, Human_Written 740, COMPLETE 98
  - avg size: 1285 chars, 43.5 lines
  - comment-like presence: 59.9%
- Test sample 1,600:
  - labels: MACHINE_GENERATED 670, MACHINE_REFINED 344, HUMAN_GENERATED 586
  - avg size: 1264 chars, 43.7 lines
  - comment-like presence: 62.5%

## Modeling Insights

1. **Length/domain shift is real**: Droid samples are substantially longer than AICD T1 samples.  
   -> Keep max length 512 but add length-aware batching and report by length buckets.
2. **Human vs machine priors differ per benchmark**: AICD T1 close to balanced, Droid has strong MACHINE_GENERATED + MACHINE_REFINED mass.  
   -> Use benchmark-specific class weights and threshold calibration.
3. **Source heterogeneity in Droid is high** (TACO/TheVault/PersonaHub).  
   -> Include source/domain-aware stratified validation and per-source metrics.
4. **Comments are a strong but risky shortcut** (Droid comment-like ratio is higher).  
   -> Run ablation with comment stripping / docstring masking to avoid leakage.
5. **Tab-indentation behavior differs between benchmarks**.  
   -> Keep structural/style features, but regularize to prevent overfitting to formatting artifacts.

## Next modeling plan

- Train 2 independent tracks:
  - Track A: `benchmark=aicd` (primary AICD leaderboard).
  - Track B: `benchmark=droid` (primary Droid benchmark).
- For each track:
  - standard run + comment-masked ablation
  - length-bucket evaluation
  - confusion matrix on hardest class pairs
