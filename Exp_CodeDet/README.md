# CoDET-M4 Dedicated Setup

This folder is a dedicated setup for running experiments on `DaniilOr/CoDET-M4`,
using `EXP11 SpectralCode` as the base model/training pipeline.

## Base

- Base experiment: `Exp_DM/exp11_spectral_code.py` (current best overall in tracker)
- Reused components:
  - model: `SpectralCode`
  - trainer: `Trainer`
  - feature pipeline: token + AST + structural + spectral
  - training policy: focal-style objective, same optimizer/scheduler and H100 profile logic

## Script

- Runner: `Exp_CodeDet/run_codet_m4_spectral.py`

## Supported CoDET tasks

- `binary`: human vs machine
- `author`: human + generator attribution (human=0, generators start from class 1)

## Default run

Edit constants in `run_codet_m4_spectral.py`:

- `RUN_MODE = "single"` or `"full"`
- `TASK = "binary"` (only for single mode)

Then run:

```bash
python Exp_CodeDet/run_codet_m4_spectral.py
```

## Notes

- The script uses the CoDET-M4 `split` column when available.
- If `split` is not available, it falls back to seeded `80/10/10` splitting.
- Subsampling defaults are aligned with current experiment policy:
  - train: 100k
  - val: 20k
  - test: 50k
