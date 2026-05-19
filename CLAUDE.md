---
description:
alwaysApply: true
---

# CLAUDE.md — Repo Briefing (EMNLP 2026 Main, Aiming Oral A*)

> Read this file first. If it disagrees with the code, trust the code and fix this file.

---

## 0. North Star

**Goal:** EMNLP 2026 Main, long paper, 8 pages. **Target: ORAL.**

What "oral A*" means for this project:
- A clear scientific contribution a reviewer will remember a week later.
- A surprising empirical finding the field has not seen.
- An architectural or methodological move that opens a new direction, not closes one.
- Numbers that are state-of-the-art at the regimes that matter, defended by ablation that the reviewer cannot wave away.

Everything else is in service of this. If a rule in this file gets in the way of an oral A* swing, the rule is wrong and we change the rule, not the swing.

**Deadline:** ~2026-05-26. Time is the binding constraint, not method conservatism.

---

## 1. Operating philosophy (replaces previous "frozen protocol")

Three principles, in order of priority:

### 1.1 Novelty first, polish second
A new method that scores 0.65 with a novel architectural object beats an old method that scores 0.72 with a parameter tweak. Reviewers reward ideas, not Macro-F1 alone. We will spend effort on **what is being modelled**, not on which scheduler decays better.

### 1.2 Architecture is a design space, not a constant
The encoder, the projector, the head, the augmentation, the schedule, the loss family — all of these are tunable. There are no permanent commitments. If a candidate hero method needs ModernBERT instead of UniXcoder, or a graph-neural head instead of a linear classifier, or a multi-task encoder with three heads, run it. The historical "unixcoder-base only" rule is now a default starting point, not a constraint.

### 1.3 Impact > rigour, but rigour is the moat
A high-impact claim with shaky evidence is worse than a polished but boring claim. The deal: we swing big on the contribution, then back it with the rigour the field expects (component ablation, controlled comparison, negative results that constrain the design space, val--test gap audit). Big swings without rigour get desk-rejected. Rigour without a big swing gets short-paper'd.

---

## 2. What changes vs the previous CLAUDE.md

| Old rule | New rule |
|:--|:--|
| Hero LOCKED to TRACO (exp76) | **Hero UNLOCKED.** TRACO is the current best, not the final hero. Any new method that beats TRACO on the composite leaderboard with a falsifier-passing novelty claim displaces it. |
| Frozen protocol: unixcoder-base only, fraction `[0.01, 0.05, 0.20]`, RAS schedule | **Default protocol, not frozen.** New methods can swap the encoder (GraphCodeBERT, CodeT5+, StarCoder2, DeepSeek-Coder-1.3B-Base), expand fractions (add 0.50 and full), or pick a new schedule. Justify the change in the theory block; do not silently drift. |
| Single-method §3, baselines §4, ablation §5 | **Single dominant contribution still preferred** for oral clarity, but the contribution can be a **system** (encoder + loss + inference recipe) rather than a single loss term. |
| Every exp must exploit ≥ 1 of S1–S10 | **S-facts are inspiration, not gatekeepers.** A method that introduces a genuinely new architectural object for code attribution is acceptable even if the S-fact map is post-hoc. Generic ML re-applications are still rejected. |
| "DO NOT" decision rules | Reduced to **three hard rules** (see §6). |

The 12-rule template (think before coding, simplicity first, surgical changes, fail loud, etc.) is retained verbatim in §7.

---

## 3. Where we are right now

### 3.1 Empirical state (composite leaderboard)

| Rank | Method | Composite Macro-F1 | Per-slot SOTA |
|:-:|:--|:-:|:--|
| 1 | TRACO (exp76) | 0.5256 | CoDET 1%, CoDET 5% |
| 2 | CARGO (exp84) | 0.5251 | AICD 1% |
| 3 | SCR (exp62) | 0.5174 | -- |
| 4 | TKL (exp63) | 0.5167 | CoDET 20% (tied) |
| 5 | DECOFP (exp70) | 0.5155 | AICD 5% |

Top 8 methods within 0.018 of each other on the composite. The cluster suggests the **encoder + RAS schedule** dominates at $20\%$ data; method-level wins live at extreme few-shot.

### 3.2 Big unexplored directions (priority for oral swings)

Each of these is worth a separate exp file. None has been tried in the active slate. All would, if they work, justify an oral pitch by themselves.

1. **Encoder pivot: pretrained code LLM as encoder.** Replace UniXcoder-base (125M) with a frozen 1B-2B code LLM (DeepSeek-Coder-1.3B, StarCoder2-3B with adapter, or a Qwen2.5-Coder-1.5B). Mean-pool the final hidden, train only a small projector + classifier. If this beats fine-tuned UniXcoder, the entire encoder choice in the field has been wrong.

2. **Graph-of-thoughts encoder.** Run a small AST/DFG GNN on top of UniXcoder outputs, with edge types encoding the family-tree (S1) prior. Genuinely new architectural object: a **dual-graph** encoder (token-graph $\cap$ AST-graph) for attribution.

3. **Retrieval-augmented attribution.** Index the training set with FAISS; at inference, mix the parametric classifier logits with a kNN vote over retrieved samples, with the retrieval weight learned. Test-time adaptation without finetuning.

4. **Multi-task encoder with auxiliary genealogy decoder.** Train a single encoder with three heads: classification, tree-distance regression, decoding-temperature regression. Use only the classification head at test, but the auxiliary tasks shape the representation.

5. **Prompt-conditioned attribution.** Most CoDET-M4 samples share prompts across generators. Encode the prompt explicitly; predict author conditional on the (prompt, code) pair. Tests whether style is the actual signal or whether it is just "this prompt produces consistent outputs from generator X".

6. **Differentiable family-tree learning.** Replace the hard-coded $\mathcal{T}$ with a learnable tree (parameterised as a Gumbel-softmax over edges or a continuous ultrametric matrix) that recovers a genealogy from data, not from release notes. Massively reduces the practical objection that we are using insider information.

7. **Active-learning few-shot.** Train a smaller initial model, score every unlabeled sample by predicted-entropy, label only the top-K by oracle, repeat. Pushes the $1\%$ regime from random to informed sampling.

8. **Reasoning-trace attribution.** Use a small reasoning model (Qwen-3-Reason or DeepSeek-R1-Distill-1.5B) to produce a chain-of-thought explanation of "who wrote this code" then attribute. Tests whether explicit reasoning closes the sibling-confusion gap.

Pick one, scope it for ten days, and run. If it fails, pivot. The bar is not "this is novel"; the bar is "this is novel and **could lift the composite by $+0.03$ or more**".

### 3.3 Hero method status

- **Currently:** TRACO (exp76).
- **Unlocked:** any new method that (i) beats TRACO on the composite by $\ge 0.01$ and (ii) passes its own falsifier, becomes the new hero candidate.
- **To displace TRACO**: the new method must produce a paper-section-level reframe of the contribution, not just a number bump. The paper is currently written around TRACO + tree-weighting + view augmentation; switching hero requires rewriting §3 and §6 around the new method.

---

## 4. Research question (current framing)

How should a few-shot code-authorship model exploit the fact that the label space is not flat?

The answer set is bigger than just "tree-weighted contrastive". The current paper says it is tree-weighted contrastive; an oral pitch can broaden it to "we expose what the label-space hierarchy implies for representation learning, and we show three different architectural ways to consume it: a loss term (TRACO), an encoder bias (graph head), and an inference rule (retrieval/kNN)".

---

## 5. Repo layout

```
Exp_FewShot/testing_chis/      ← ACTIVE
  tracker.md                   ← live leaderboard
  exp{60..89}.py               ← method round 2 + 3 + CARGO family
  exp{90..92}.py               ← external-baseline adapters (CGPTS, DETMULTI, DCGPT)
  external_baselines/          ← read-only third-party clones
  legacy/                      ← read-only old methods
Paper/latex/main.tex           ← current draft, written around TRACO
Paper/outline.md               ← section-level plan
Exp_Climb/, Exp_CodeDet/, Exp_DM/   ← frozen, do not touch
```

New experiments live in `Exp_FewShot/testing_chis/`. Pick the next free `expNN` ID (current free range starts at exp93).

---

## 6. Three hard rules (everything else is negotiable)

1. **Never reuse an `expNN` ID.** History is append-only.
2. **Always report `val_macro`, `test_macro`, and the `val_test_gap`** for every run. A method without a val--test gap report is not finished.
3. **Never mix benchmarks in a single Macro-F1 cell.** CoDET-M4 → Macro-F1, AICD-T2 → Macro-F1, Droid → Weighted-F1 (per the repo hook). Each benchmark stays in its own column.

Everything else, including encoder choice, schedule, augmentation pool, loss family, hero method, S-fact mapping, even the paper's framing, is up for renegotiation.

---

## 7. The 12-rule template (retained)

Apply to every task unless explicitly overridden. Bias: caution over speed on non-trivial work.

1. **Think before coding** — state assumptions, ask if uncertain.
2. **Simplicity first** — no abstractions for single-use code.
3. **Surgical changes** — touch only what you must.
4. **Goal-driven execution** — define success, loop until verified.
5. **Use the model only for judgment calls** — if code can answer, code answers.
6. **Token budgets are not advisory** — per-task 4k, per-session 30k.
7. **Surface conflicts, do not average them** — pick one, flag the other.
8. **Read before you write** — read callers and shared utilities first.
9. **Tests verify intent** — encode why behavior matters.
10. **Checkpoint after every significant step** — summarise done / verified / left.
11. **Match codebase conventions** — conformance over taste.
12. **Fail loud** — "completed" is wrong if anything was skipped silently.

**Project addendum:** every new `expNN_*.py` file MUST open with `# expNN — Method name` and include the 6-line theory block (NAME / ARXIV_ID / ONE-LINE CLAIM / EQUATION / WHY NOT BEFORE / FALSIFIER) before any imports.

---

## 8. How to add a new experiment

1. **Pick the next free `expNN` ID** (currently exp93+).
2. **Write the 6-line theory block.** State the new mathematical object or architectural component in equation form. State what would falsify your claim.
3. **Decide whether you need the existing protocol or a new one.** If new (different encoder, different fractions, different schedule), justify it in one paragraph at the top.
4. **Implement.** Reuse plumbing from the closest existing exp (TRACO at `exp76_traco.py` is the canonical template) but feel free to swap any component.
5. **Always report val + test + val--test gap.** Output a single `{expNN_method}_results.json` with the full eval pack.
6. **Update `tracker.md`** with the row.

Theory block example:
```
# exp93 — METHOD_NAME
# NAME       : ...
# REFERENCE  : ... (arXiv id or "new")
# CLAIM      : one sentence
# EQUATION   : L = ...
# WHY NEW    : what existing method does NOT do this
# FALSIFIER  : what metric, on what test set, would invalidate the claim
```

---

## 9. References

- Paper draft: [Paper/latex/main.tex](Paper/latex/main.tex)
- Section outline: [Paper/outline.md](Paper/outline.md)
- Active tracker: [Exp_FewShot/testing_chis/tracker.md](Exp_FewShot/testing_chis/tracker.md)
- External baselines: [Exp_FewShot/external_baselines/README.md](Exp_FewShot/external_baselines/README.md)
- Benchmarks:
  - CoDET-M4 (Orel et al., ACL Findings 2025)
  - AICD Bench Task 2 (Orel et al., EACL 2026)
  - Droid (Orel et al., EMNLP 2025)

Recent literature worth reading for the new directions in §3.2:
- LASCL — Label Hierarchy SupCon (arXiv:2402.00232)
- HCAL — Hierarchy-Consistent Adaptive Loss (arXiv:2508.13452)
- Tournament Code Attribution (arXiv:2501.08165)
- LLM-AuthorBench Stylometry (arXiv:2506.17323)
- CodeT5-JSA / Hidden DNA structural patterns (arXiv:2510.10493)
- Code Fingerprints / DCAN disentangled attribution (arXiv:2603.04212)

---

## 10. One closing note

If you are reading this and thinking "should I run a small parameter sweep on TRACO to squeeze another $0.005$ Macro-F1?", the answer is no. Spend that time on one of the eight directions in §3.2. The paper has its hero number already; the paper is missing its **second act**.
