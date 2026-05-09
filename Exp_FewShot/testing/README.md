# Exp_FewShot/testing/ — Setup-Variation Experiments

> **Goal:** explore which setup (method, encoder, training budget, K-shot vs
> %-fraction, hyperparameter) gives the strongest data-efficient story for
> the EMNLP paper. **Not** the place for new theorems — that's `novel/`.

## What lives here

Setup variations on KNOWN methods, paper-baseline reimplementations, and
hyperparameter sweeps. None of these need to clear the Novelty Filter; they
just need to fill cells in the data-efficiency matrix.

## Files (current)

### Inline portfolio (paste-into-cell, no clone needed)

| File | Method | Encoder | Loss |
|:--|:--|:--|:--|
| `exp_fs_inline.py` | dispatcher (FS_METHOD env var) | ModernBERT-base | any |
| `exp_fs_inline_baseline.py` | FS-Baseline-CE | ModernBERT-base | CE |
| `exp_fs_inline_ntkalign.py` | FS-NTKAlign 🏆 | ModernBERT-base | CE + NTK |
| `exp_fs_inline_supcon.py` | FS-SupCon | ModernBERT-base | CE + SupCon |
| `exp_fs_inline_frozen.py` | FS-Frozen-LinearProbe | ModernBERT-base (frozen) | CE |
| `exp_fs_inline_ntk_frozen.py` | FS-NTKAlign-Frozen | ModernBERT-base (frozen) | CE + NTK |
| `exp_fs_inline_supcon_frozen.py` | FS-SupCon-Frozen | ModernBERT-base (frozen) | CE + SupCon |
| `exp_fs_inline_hier.py` | FS-HierTree | ModernBERT-base | CE + family pull/push |
| `exp_fs_inline_focal.py` | FS-Focal | ModernBERT-base | Focal loss |
| `exp_fs_inline_hier_ntk.py` | FS-Hier-NTK | ModernBERT-base | CE + family + NTK |

### Paper baseline reimplementations

| File | Encoder | Paper full-data score |
|:--|:--|:-:|
| `exp_fs_baseline_unixcoder.py` | microsoft/unixcoder-base | 66.33 |
| `exp_fs_baseline_codebert.py` | microsoft/codebert-base | 64.80 |
| `exp_fs_baseline_graphcodebert.py` | microsoft/graphcodebert-base | — |
| `exp_fs_baseline_codet5.py` | Salesforce/codet5-base (T5EncoderModel) | 62.45 |
| `exp_fs_baseline_catboost.py` | handcrafted features + CatBoost (no GPU) | 45.42 |

### Earlier per-method scripts (clone-bootstrap; equivalent to inline above)

| File | Notes |
|:--|:--|
| `exp_fs_00_baseline.py` | imports from `_common_fs.py`; same logic as inline |
| `exp_fs_01_ntkalign.py` | … |
| `exp_fs_02_supcon.py` | … |
| `exp_fs_03_frozen.py` | … |
| `exp_fs_04_ntk_frozen.py` | … |
| `exp_fs_05_supcon_frozen.py` | … |

These run via `git clone + python …` instead of paste-into-cell. Kept for
reproducibility and parity-checking; prefer the inline files for new runs.

## Convention

- **Default sweep:** K=128 + fraction in {0.01, 0.05} = 3 configs (~50 min on T4).
- **Override:** `FS_SWEEP_KS` and `FS_SWEEP_FRACS` env vars.
- **JSON output:** `/kaggle/working/results/<exp_id>_<label>_seed<S>.json`
  (or `./results/...` locally).
- **Append-only.** Never overwrite a JSON; new seed → new file.
- **Always report val-test gap** alongside test (CLAUDE.md mandate).

## When to add a new file here

Pass criterion is loose:
- New encoder swap on existing pipeline → yes.
- New hyperparameter point on an existing method → yes (use `FS_LAMBDA_*` env).
- Combined existing losses → yes (e.g., NTK + SupCon both already exist).

If the idea has a new mathematical object with a theorem behind it →
`novel/`, not here.
