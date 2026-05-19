# External SOTA Baselines

This folder contains read-only mirrors of recent AI-code attribution and
detection methods that we are integrating as baselines for the TRACO
EMNLP 2026 paper.

**Important:** every subfolder is an upstream third-party repository
cloned with `--depth 1`. Do not modify code inside these folders. Any
adaptation to our few-shot protocol lives in
`Exp_FewShot/testing_chis/exp9X_*.py` wrappers, not in the upstream tree.

Cloned on 2026-05-19. Last research pass used a general-purpose research
agent (see commit message for the agent prompt). Nine candidates were
investigated; the status of each is below.

---

## Status table

| # | Method | Venue | Repo state | Files | Adaptation plan |
|:-:|:--|:--|:--|:--|:--|
| 1 | **CodeGPTSensor** | TOSEM 2025 | ✅ cloned, full code | `model.py`, `run.py`, utils | `exp90_cgpts.py`: swap binary head to $K$-softmax, keep UniXcoder+InfoNCE backbone |
| 2 | **DeTeCtive (multi-level)** | NeurIPS 2024 | ✅ cloned, full code | `train_classifier.py`, `simclr.py`, `test_knn.py`, `gen_database.py` | `exp91_detmulti.py`: keep multi-level SupCon hierarchy, route to genealogy tree instead of authored hierarchy |
| 3 | **DetectCodeGPT** | ICSE 2025 | ✅ cloned, full code | `main.py`, baselines/{detectGPT, supervised, rank, loss, entropy}.py | `exp92_dcgpt.py`: their supervised baseline already produces logits, adapt to $K$ classes |
| 4 | **CodeT5-JSA / Hidden DNA** | arXiv 2510 (TII) | ⚠️ dataset only | 10MB validation JSONL, no training code | Use as out-of-distribution test set (JavaScript) for $K$-class generalisation |
| 5 | **DCAN / Code Fingerprints** | arXiv 2603 (Sichuan U.) | ⚠️ README only | `README.md` lists `run.py` and `eval.py` but they are absent | Cannot reproduce: code release "coming soon". Re-check before camera-ready |
| 6 | **DroidDetect** | EMNLP 2025 main | ❌ no GitHub | HuggingFace artifacts only (`project-droid/...`) | Detection model on HF; would need a separate `transformers`+HF-hub loader. Skip for now |
| 7 | **LLM-Tournament** | arXiv 2501.08165 | ❌ no public code | none | Cannot reproduce |
| 8 | **DNB (Nested Bigrams)** | arXiv 2502.15740 | ❌ no public code | none | Cannot reproduce |
| 9 | **CodeVision** | arXiv 2501.03288 | ❌ no public code | none | Cannot reproduce |

Three actionable cloned repos: **CodeGPTSensor**, **DeTeCtive
(multi-level)**, **DetectCodeGPT**.

---

## Per-repo adaptation notes

### 1. CodeGPTSensor (TOSEM 2025) — `codegptsensor/`

- Upstream: <https://github.com/doriscullen/CodeGPTSensor>
- Architecture: UniXcoder + projection head + InfoNCE on positive/negative
  pairs (human vs.\ AI binary).
- Adaptation: replace the binary head with a $K$-softmax classifier head,
  keep the InfoNCE on pairs of \emph{same-class} positives, swap the
  hard-negative pool to \emph{all other classes}. This is essentially
  flat supervised contrastive on the same encoder we use.
- Wrapper file: `Exp_FewShot/testing_chis/exp90_cgpts.py` (planned).
- Why it matters: **architecturally matched twin** of TRACO.
  Difference is only the tree weighting on the negatives. Isolates the
  contribution of our tree-weighting term cleanly.

### 2. DeTeCtive multi-level (NeurIPS 2024) — `detective_multi/`

- Upstream: <https://github.com/heyongxin233/DeTeCtive>
- Architecture: multi-level contrastive (author, family, register) on
  pre-trained text encoder, with kNN retrieval at inference.
- Adaptation: replace text encoder with UniXcoder; route the "family"
  level to our genealogy tree (CodeLlama family etc.); keep kNN
  inference as the original paper specifies.
- Wrapper file: `Exp_FewShot/testing_chis/exp91_detmulti.py` (planned).
- Why it matters: closest published competitor to TRACO's
  tree-weighting concept. If the multi-level hierarchy already captures
  our gain, TRACO's contribution is reduced to view augmentation; if it
  does not, TRACO's tree-distance weighting wins on its own merits.

### 3. DetectCodeGPT (ICSE 2025) — `detectcodegpt/`

- Upstream: <https://github.com/YerbaPage/DetectCodeGPT>
- Architecture: whitespace-perturbation + DetectGPT-style log-prob
  curvature score; the `supervised.py` baseline trains a head on
  encoder embeddings.
- Adaptation: use only the `baselines/supervised.py` path, swap the
  encoder to UniXcoder, swap the head to $K$-class.
- Wrapper file: `Exp_FewShot/testing_chis/exp92_dcgpt.py` (planned).
- Why it matters: A-tier SE-venue stylistic-perturbation baseline.
  Covers a different paradigm (code-aware perturbations + log-prob
  score) than our contrastive recipe, broadening the
  family-coverage of the baseline section.

---

## Reproduction protocol

Every wrapper follows the same five rules so the numbers are directly
comparable to TRACO's:

1. Encoder: `unixcoder-base`, `local_files_only=True`.
2. Few-shot fractions: $\alpha \in \{0.01, 0.05, 0.20\}$, stratified
   per class.
3. Regime-adaptive schedule: $1\% \to 10$ ep / lr $3{\times}10^{-5}$ /
   warm $0.20$; $5\% \to 6$ / $3$e-5 / $0.15$; $20\% \to 6$ / $4$e-5 / $0.10$.
4. AMP bfloat16, batch size auto-scaled from 256 to 32.
5. Output a single JSON with `val_macro`, `macro` (test), `val_test_gap`,
   and `dpaper` per slot. Six slots per file (3 fractions × 2 benchmarks).

Expected compute per wrapper: ~6 GPU-hours total across the six slots
(one $20\%$ run dominates at ~2h on H100).

---

## Excluded methods and why

These were considered but excluded:

- **Zero-shot text detectors** (Ghostbuster, Binoculars, Fast-DetectGPT):
  binary detection only, no code-specific multi-class adaptation
  released.
- **CodeT5-Authorship, FAID, Style-Repr, GPTSniffer, Whodunit**:
  already retrained under our protocol; numbers are in Table 1 of the
  paper.
- **LASCL**: general hierarchical SupCon for text classification, not
  code attribution. Already discussed in related work; conceptual
  ancestor of our tree weighting.

---

## License notes

Each cloned subfolder retains its original upstream license. Verify
license compatibility before redistribution. We do not modify upstream
code; all adaptations live outside this folder.
