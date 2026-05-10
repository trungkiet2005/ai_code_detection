# Exp_FewShot — K-shot AI-Code Authorship Detection

**Status:** Pivot suite for EMNLP 2026 Main submission (deadline ~2026-05-26).

The 20%-data SOTA already exists in `Exp_Climb/` (NTKAlign 71.03 Author F1).
Few-shot is the higher-impact framing if we can hit ≥ 65 Author F1 at K ≤ 64.

## Files

| File | Purpose |
|:--|:--|
| `_common_fs.py` | bootstrap (Kaggle pip-install), `FSConfig`, T4 hardware profile, autocast |
| `_fewshot_sampler.py` | K-shot stratified sampler + mini-val builder |
| `_data_codet_fs.py` | CoDET-M4 HF loader → label vocab → K-shot subset → DataLoader |
| `_model_fs.py` | ModernBERT classifier + NTK projector + losses (CE / NTK) |
| `_trainer_fs.py` | 1-epoch trainer with adaptive eval + early-stop + full-test eval |
| `exp_fs_00_baseline.py` | Floor: ModernBERT + CE only |
| `exp_fs_01_ntkalign.py` | Method: CE + NTK target-kernel alignment |
| `tracker.md` | Leaderboard + run history (append-only) |

## Hardware

| GPU | VRAM | Batch | Precision | Seq length | Speed |
|:--|:--|:--|:--|:--|:--|
| **RTX 96GB** | 96GB | **128** | **bf16** | **512** | **~6-8× faster than T4** |
| **RTX Pro 6000 Ada** | 48GB | **64** | **bf16** | **512** | **~3-4× faster than T4** |
| T4 | 16GB | 16 | fp16 | 384 | baseline |
| H100 | 80GB | 32-64 | bf16 | 512 | fast |

RTX96GB: `mem_gb >= 90` → bs=128
RTX6000: `"RTX Pro 6000" in gpu or "RTX 6000 Ada" in gpu or mem_gb >= 40` → bs=64

## Run

```bash
# CPU smoke test (~5 min on K=8)
FS_K_SHOT=8 python Exp_FewShot/exp_fs_00_baseline.py
FS_K_SHOT=8 python Exp_FewShot/exp_fs_01_ntkalign.py

# Kaggle T4 — auto-detected, no flags needed
python Exp_FewShot/exp_fs_01_ntkalign.py

# RTX6000 / RTX 96GB — auto-detected
# RTX6000: bs=64, bf16, seq=512 (~3-4× faster than T4)
# RTX 96GB: bs=128, bf16, seq=512 (~6-8× faster than T4)
```

### Offline mode (no-internet, avoids conflict + bias)

All data loaded from Kaggle input paths (no HuggingFace download):

```python
# Models: /kaggle/input/datasets/chiboiz/ai-detection-encoders/models/
# CoDET-M4: /kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet
# DroidCollection: /kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/
tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
model = AutoModel.from_pretrained(path, local_files_only=True)
```

### Merge code strategy (run multiple configs in one cell)

For rapid iteration on RTX6000, chain multiple configs sequentially:

```python
# Run Hier-NTK × 3 encoders × 3 fractions in one cell
import os
os.environ["FS_ENCODER"] = "ModernBERT-base,codebert-base,unixcoder-base"
os.environ["FS_SWEEP_FRACS"] = "0.01,0.05,0.20"
os.environ["FS_SEED"] = "42"
os.environ["FS_BENCHMARK"] = "codet_m4"
# Then run: python Exp_FewShot/testing_chis/run_hier_ntk_portfolio.py
```

Every run emits a `BEGIN_FS_TABLE ... END_FS_TABLE` block; paste into `tracker.md`.

## Env vars (for sweeps)

| Var | Default | Range |
|:--|:-:|:--|
| `FS_K_SHOT` | 32 | {8, 16, 32, 64, 128} |
| `FS_SEED` | 42 | any int |
| `FS_LAMBDA_NTK` | 0.4 | {0.1, 0.2, 0.4, 0.8} |

## What this suite is NOT

- Not a replacement for `Exp_Climb/` — kept as 20% baseline for comparison.
- No AST / tree-sitter / FFT spectral / GNN — too brittle on T4 + minimal benefit
  at K-shot scale.
- No multi-bench dual run — CoDET first, Droid only after Phase B passes.
