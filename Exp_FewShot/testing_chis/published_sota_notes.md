# Published Few-Shot / Detection Baselines

This note records published methods worth comparing against our CoDET-M4 author-attribution work.

Primary task in this repo: **CoDET-M4 Author IID, 6-way Macro-F1**. The paper reference is UniXcoder full-data **0.6633** from CoDET-M4 Table 7.

## Reproduce First

| Priority | Method key | Paper | Why it fits | Runner status |
| :-: | :-- | :-- | :-- | :-- |
| 1 | `codet5_authorship` | *I Know Which LLM Wrote Your Code Last Summer* / CodeT5-Authorship | Code-specific LLM authorship attribution. Architecture is directly portable: CodeT5 encoder + first-token MLP classifier. | Implemented in `run_published_sota_portfolio.py` |
| 2 | `detective_supcon` | *DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning* | Published detector built around author/style separation and contrastive learning. Strong conceptual baseline versus our Hier-NTK. | Implemented as CE + SupCon adaptation |
| 3 | `faid_multitask` | *FAID: Fine-grained AI-generated Text Detection using Multi-task Auxiliary and Multi-level Contrastive Learning* | Fine-grained AI text detector with authorship/family modeling and multi-level contrastive learning. | Implemented as class + family multitask adaptation |
| 4 | `style_repr_logreg` | *Few-Shot Detection of Machine-Generated Text using Style Representations* | Explicit few-shot detector that uses style representations and can do model attribution with a handful of examples. | Implemented as code-style features + balanced logistic regression |

Entry files:

- `exp00_sota_codet5_authorship.py`
- `exp01_sota_detective_supcon.py`
- `exp02_sota_faid_multitask.py`
- `exp03_sota_style_repr_logreg.py`

Shared code lives in `run_published_sota_portfolio.py`, so fixes to the loader/trainer only need to be made once.

## Optional / Separate Binary Baselines

| Method | Paper | Fit | Decision |
| :-- | :-- | :-- | :-- |
| ATC | *Zero-Shot Detection of LLM-Generated Code via Approximated Task Conditioning* | Code-specific and SOTA-style zero-shot, but binary human-vs-AI and depends on task conditioning. | Do not compare on 6-way author until prompt reconstruction is solved. |
| Ghostbuster | NAACL 2024 | Strong AI-text detector using weaker LM probabilities and feature search. | Binary detector; possible appendix baseline after mapping CoDET to human-vs-AI. |
| Binoculars | arXiv 2401.12070 | Strong zero-shot text detector using paired LM scores. | Binary detector; possible appendix baseline. |
| DetectGPT / Fast-DetectGPT / DetectLLM / DNA-GPT | 2023 generation-probability detector family | Useful zero-shot baselines, but mostly binary and expensive for full CoDET. | Use only if binary table is added. |

## Source Links

- Few-shot style representations: https://huggingface.co/papers/2401.06712
- DeTeCtive: https://huggingface.co/papers/2410.20964
- FAID: https://aclanthology.org/2026.eacl-long.151/
- CodeT5-Authorship / LLM-AuthorBench: https://huggingface.co/papers/2506.17323
- ATC for generated code: https://huggingface.co/papers/2506.06069
- Ghostbuster: https://aclanthology.org/2024.naacl-long.95/
- Binoculars: https://huggingface.co/papers/2401.12070

## Reporting Rule

Use "published-method adaptation" for the implemented baselines unless the original authors' exact code, dataset, and task are run unchanged. For the EMNLP comparison table, keep columns:

| Method | Published source | CoDET task | Train budget | Encoder/features | Test Macro-F1 | Delta vs 0.6633 |
