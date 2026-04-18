# CodeOrigin

**Domain-Invariant AI-Generated Code Detection via Style-Content Disentanglement and Hierarchical Contrastive Learning**

> Detecting AI-generated code that generalizes across unseen languages, domains, and model families.

## Problem

Current AI-generated code detectors achieve near-perfect accuracy on in-distribution data but **collapse under distribution shift** (new languages, new domains, new generators). On [AICD-Bench](https://huggingface.co/AICD-bench/AICD-Bench), even the best trained model (DeBERTa, 34.13 F1) scores **below random guessing** (45.73 F1) on the out-of-distribution test set.

**Root cause**: Models overfit to surface-level, domain-specific artifacts (e.g., competitive programming templates) instead of learning generalizable authorship signals.

## Method

CodeOrigin disentangles **how** code is written (style) from **what** code does (content), enabling domain-invariant detection.

```
Source Code
    |
    v
[Multi-Granularity Encoder]
    |-- Token-level:  ModernBERT on source tokens
    |-- AST-level:    BiLSTM on AST node type sequences
    |-- Graph-level:  MLP on structural features
    |
    v  (Cross-Attention Fusion)
    |
[Style-Content Disentanglement]
    |-- z_style:   authorship fingerprint (naming, control flow, comments)
    |-- z_content: semantic meaning (algorithm, domain, language)
    |
    |-- MI Minimization (CLUB): ensure z_style independent of z_content
    |-- KL Regularization + Reconstruction
    |
    v
[Task Heads on z_style only]
    |-- T1: Binary classifier + Domain Adversarial Alignment (GRL)
    |-- T2: Prototypical Contrastive Head (learnable family prototypes)
    |-- T3: Focal Loss classifier (handles class imbalance)
```

### Key Contributions

1. **Style-Content Disentanglement** -- First to explicitly separate authorship fingerprints from code semantics for detection, directly addressing OOD collapse
2. **Multi-Granularity Code Representation** -- Token + AST + structural features fused via cross-attention, capturing complementary signals
3. **Hierarchical Prototypical Contrastive Learning** -- Learnable family prototypes for attributing unseen generators to known model families
4. **Unified Multi-Task Framework** -- Single model handles all 3 AICD-Bench tasks with shared representations

## Benchmark: AICD-Bench

| Task | Description | Classes | Train | Test |
|------|-------------|---------|-------|------|
| T1 | Robust Binary Classification (OOD) | 2 | 500K | 1.1M |
| T2 | Model Family Attribution | 12 | 500K | 500K |
| T3 | Fine-Grained Human-Machine Classification | 4 | 900K | 1M |

- 77 generators across 11 families (DeepSeek, Qwen, Llama, GPT-4o, Gemini, ...)
- 9 programming languages (Python, Java, C++, C, Go, PHP, C#, JavaScript, Rust)
- OOD test splits: unseen languages, unseen domains, unseen generators

## Results

| Model | Task 1 | Task 2 | Task 3 |
|-------|--------|--------|--------|
| Random | 45.73 | 5.69 | 20.34 |
| SVM (TF-IDF) | 43.05 | 5.44 | 21.28 |
| CodeBERT | 28.64 | 23.71 | 55.60 |
| ModernBERT | 30.61 | 32.84 | 61.65 |
| DeBERTa | 34.13 | 12.08 | 54.65 |
| Gemini 2.5 (CoT) | 62.31 | 5.54 | 28.14 |
| **CodeOrigin (ours)** | **TBD** | **TBD** | **TBD** |

## Quick Start

### Run on Kaggle

```python
# Cell 1: Install dependencies
!pip install datasets transformers accelerate tree-sitter-languages

# Cell 2: Run
from exp00_codeorigin import main

# Task 1: Robust Binary Classification
main(task="T1")

# Task 2: Model Family Attribution
main(task="T2")

# Task 3: Fine-Grained Classification
main(task="T3")
```

### Train on AICD-Bench (separate benchmark)

```bash
python experiment/exp00_codeorigin.py --benchmark aicd --task T1
```

### Train on DroidCollection (separate benchmark)

```bash
python experiment/exp00_codeorigin.py --benchmark droid --task T1
```

- Each run uses exactly one benchmark (`aicd` or `droid`).
- Train/validation/test are all from the selected benchmark.

### Custom Configuration

```python
from exp00_codeorigin import main, Config

config = Config(
    task="T1",
    epochs=5,
    batch_size=32,
    max_train_samples=100_000,
    max_val_samples=20_000,
    max_test_samples=50_000,
    encoder_name="answerdotai/ModernBERT-base",
    lr_encoder=2e-5,
    lr_heads=1e-4,
    fp16=True,
)
main(task="T1", config=config)
```

## Project Structure

```
ai_code_detection/
├── README.md
├── experiment/                 # Baseline: exp00_codeorigin.py (Kaggle-ready single-file)
├── Exp_DM/                     # Novel methods on AICD + Droid (exp01..30)
│   └── dm_tracker.md           # Results tracker for AICD + Droid
├── Exp_CodeDet/                # Novel methods on CoDET-M4 (exp11..38)
│   └── tracker.md              # Leaderboard for CoDET-M4
├── docs/
│   ├── CLAUDE.md               # Full repo context — start here for any AI assistant
│   ├── performance_tracker.md  # Legacy tracker (superseded by Exp_DM/dm_tracker.md)
│   ├── references/             # Source papers, dataset schemas, research notes
│   └── archive/                # Completed session artifacts
├── Slide/                      # Proposal deck (tex/pdf) + presentation script
└── Formatting_Instructions_For_NeurIPS_2026/   # NeurIPS 2026 LaTeX template
```

**New to the repo?** Read [docs/CLAUDE.md](docs/CLAUDE.md) first.

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- transformers >= 4.36
- datasets
- scikit-learn
- tree-sitter-languages (optional, falls back to regex-based AST extraction)

## Citation

```bibtex
@article{codeorigin2026,
  title={CodeOrigin: Domain-Invariant AI-Generated Code Detection via Style-Content Disentanglement and Hierarchical Contrastive Learning},
  author={},
  year={2026}
}
```

AICD-Bench:
```bibtex
@article{orel2026aicdbench,
  title={AICDBench: A Challenging Benchmark for AI-Generated Code Detection},
  author={Orel, Daniil and Azizov, Dilshod and Paul, Indraneil and Wang, Yuxia and Gurevych, Iryna and Nakov, Preslav},
  journal={arXiv preprint arXiv:2602.02079},
  year={2026}
}
```

## License

MIT
