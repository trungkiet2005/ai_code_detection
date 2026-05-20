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

## 1. Operating philosophy — NOVELTY-FIRST, NO-COMPROMISE

Three principles, in order of priority:

### 1.1 The "WOW Test" is the only test

A method passes if a reviewer reads §3 and **physically reacts** — raises an eyebrow, takes a screenshot, sends it to a colleague. If a reviewer can summarise the method in one boring sentence ("they add a tree-weighted SupCon loss"), the paper is dead on arrival regardless of the number.

The bar:
- A method that scores 0.62 with a **new architectural object the field has not seen** beats a method that scores 0.72 by tweaking $\lambda$.
- A method that reads as "yet another contrastive loss" is auto-rejected at the planning stage, even before we run it.
- If your method's one-line claim could plausibly have been published in 2022, it is not novel enough for EMNLP 2026 oral.

Heuristic for "WOW":
- Does it introduce a new computational primitive (router, memory, slot, expert, agent, world-model rollout)?
- Does it move information in a direction no prior code-attribution paper has moved it (test-time gradient, learned retrieval, agentic deliberation, distillation from a tool-using teacher)?
- Does it make a counter-intuitive claim (e.g. "the encoder is the wrong place to inject genealogy — the *router* is")?
- Does the title sound like a 2026 paper, not a 2023 paper?

If "yes" to ≥ 2 of these, plan the experiment. If "no" to all four, kill it.

### 1.2 Architecture is a design space, not a constant

Encoder, projector, router, head, augmentation pool, schedule, loss family, inference-time computation — **everything** is tunable. There are no permanent commitments. UniXcoder-base is a default starting point, not a constraint. If a candidate hero needs a frozen 7B code LLM, a Mixture-of-Experts router, a slot-attention head, or a multi-agent inference loop — run it.

Specifically allowed (and encouraged):
- Swap encoder to any open code LLM ≤ 7B.
- Add inference-time computation (kNN retrieval, MCMC, agent deliberation, beam-of-classifiers).
- Add learnable components beyond a classifier head (routers, slots, memory banks, hypernetworks, prompt-tunable prefixes).
- Train with anything: distillation from a strong teacher, RL-from-AI-feedback, contrastive + reconstruction + diffusion-denoising — whatever the WOW Test rewards.

### 1.3 Rigour is the moat, not the cage

Big swings without rigour get desk-rejected. Rigour without a big swing gets short-paper'd. So:
1. Swing big in §3 (the new object).
2. Then defend it in §5 with ablation that strips the object down to its parts and shows each part earns its keep.
3. Then ship a negative result in §6 that **constrains** the design space and shows the reviewer you understand your own method's failure mode.

The val--test gap audit is non-negotiable (see §6, hard rule 2). Everything else in "rigour" is in service of making the WOW claim land.

### 1.4 Anti-patterns — methods we will NOT plan

Reject at the planning stage:
- "Add another contrastive loss term." — TRACO/CARGO/SCR already saturate this direction.
- "Change the schedule / lr / warmup / dropout." — That is a hyperparameter ablation, not a contribution.
- "Use a slightly bigger encoder with the same head." — That is an engineering report, not a paper.
- "Combine method A + method B linearly." — Linear combinations are a §5 ablation, not a §3 hero.
- "Augment with synonyms / token dropout / paraphrasing." — Surface augmentation has been done to death.
- "Distill from GPT-4." — Not novel; reviewers will ask why GPT-4 itself isn't the baseline.
- **NEW 2026-05-20**: "Yet another family/sibling-tree trick." TRACO's tree-weighting is the spotlight; piling more "sibling-only" heuristics on top reads to reviewers like *over-fitting the headline finding*. New methods must work on AUTHORSHIP in general, not just on the 1-sibling-pair CoDET-M4 setup. A method whose pitch reduces to "we separate Qwen from Nxcode harder" is a §5 ablation, not §3.

Reject reflexively. If you find yourself proposing one of these, go re-read §3.2 and pick a hot trend instead.

### 1.5 Storytelling > engineering — added 2026-05-20

A method that reads as **engineering** ("we combine X with Y with Z and tune all five hyperparameters") is auto-§5. A method that reads as **a single declarative claim about the world** ("identity is the bass frequency of code") is §3 material. The bar at planning time:

- Can you summarise the method in ONE quotable sentence that does not mention any hyperparameter, training detail, or component name?
- Does that sentence make a CLAIM ABOUT THE PROBLEM (e.g. "style lives at three scales" / "identity is a fixed point of multi-invariance"), not about the method (e.g. "we add a head with X loss")?
- Could a reviewer paraphrase the contribution to a colleague without owning a copy of the paper?

If yes → write it up. If no → keep refining the claim until it lands.

This rule was added because Round 8 / Round 9 plans were drifting toward "framework-of-components" descriptions that read as engineering rather than as scientific claims. The 3 picks that survive that filter for Round 9 are CHORALE, PRISM, RESONANCE — each is one sentence ("identity is multi-scale" / "we design the geometry" / "style is the bass").

---

## 2. What changes vs the previous CLAUDE.md

| Old rule | New rule |
|:--|:--|
| Hero LOCKED to TRACO (exp76) | **Hero UNLOCKED.** TRACO is the floor, not the ceiling. Any new method that beats TRACO on the composite AND passes the WOW Test displaces it. |
| Frozen protocol: unixcoder-base only, fraction `[0.01, 0.05, 0.20]`, RAS schedule | **Default protocol, not frozen.** Swap encoder freely (code LLMs up to 7B), expand fractions if needed, redesign the schedule if the method demands it. |
| Single-method §3, baselines §4, ablation §5 | **Single dominant contribution** still preferred for oral clarity, but the contribution must be a **system or framework**, not a loss term. Names like "FRAMEWORK", "ARCHITECTURE", "PARADIGM" beat names like "LOSS". |
| Every exp must exploit ≥ 1 of S1–S10 | **S-facts are inspiration, not gatekeepers.** A method that introduces a genuinely new architectural primitive is acceptable even if the S-fact map is post-hoc. Generic ML re-applications are still rejected. |
| "Simplicity first" applied everywhere | **Simplicity first applies to PLUMBING only.** The hero method should be *conceptually* simple to state but *architecturally* novel. "A new object" is the goal; "a clean implementation of the new object" is the deliverable. |
| "DO NOT" decision rules | Reduced to **three hard rules** (see §6). |

The 12-rule template (think before coding, surgical changes, fail loud, etc.) is retained verbatim in §7 — but rule 2 ("simplicity first") explicitly does NOT apply to the hero method's *conceptual surface*.

---

## 3. Where we are right now (updated 2026-05-20)

### 3.1 Empirical state (composite leaderboard, current top 6)

| Rank | Method | Composite | Per-slot SOTA | Status |
|:-:|:--|:-:|:--|:--|
| 🥇 | **METATRACO** (exp100) | **0.5408** | CoDET 1%, AICD 1% | Hero candidate (decision pending) |
| 🥈 | DCGPT (exp92) | 0.5313 | CoDET 20% | inspired-by adaptation |
| 🥉 | PROG (exp83) | 0.5296 | AICD 5% | curriculum |
| 🥉 | RAINFER (exp94) | 0.5296 | AICD 20% | retrieval |
| 5 | CARMIX (exp86) | 0.5295 | CoDET 5% | dual-aug |
| 6 | CARBO (exp85) | 0.5285 | -- | compositional |
| 9 | TRACO (exp76) | 0.5256 | -- | placeholder hero in paper |

**Diagnosis (revised):** METATRACO (MAML wrapper around TRACO supcon) is the
new composite #1, crossing the +0.010 unlock threshold by +0.0152. Saturation
band at $20\%$ persists across top 8 methods; META's win comes entirely from
the extreme few-shot regime (+0.033 on CoDET 1%, +0.070 on AICD 1%). **The
path forward is NOT a sibling/family trick** — five faithful external
baselines (LLMSniffer, DA-MTL, FAID, LUAR, CodeGPTSensor) all underperform
TRACO by ≥ 5pp, and Round 8 DUALGRAPH (a sibling-graph idea) also failed.
The next hero must work on AUTHORSHIP IN GENERAL, not on a specific
benchmark's family layout.

### 3.1.b External baselines (faithful K-class reproductions, 2026-05-20)

| Method | Composite | Δ vs TRACO | Reading |
|:--|--:|--:|:--|
| ext_codegptsensor | 0.4775 | −0.048 | strongest faithful baseline |
| ext_luar          | 0.4398 | −0.086 | frozen prototype-NN lags |
| ext_damtl         | 0.4255 | −0.100 | dual-head MTL underwhelms |
| ext_faid          | 0.4006 | −0.125 | multi-level SupCon collapsed |
| ext_llmsniffer    | 0.1491 | −0.377 | cls.detach + 1e-6 lr → near-random |

Every faithful published baseline loses to TRACO at matched protocol. The
paper's Table 1 favours us decisively at every slot.

### 3.2 Hot directions — novelty × tractability, low-compute prioritised

We organise candidates into **two tracks**. **Track 1 is the default.** Track 2 only if Track 1's WOW Test is satisfied AND compute allows.

**Compute tags** used throughout:
- 🟢 **Low**: trains in ≤ 1× TRACO budget (≈ 12 hr on RTX Pro 6000). Same encoder, same fractions, +1 small module.
- 🟡 **Medium**: 1×–2.5× TRACO. New encoder OR new training objective OR small inference-time overhead.
- 🔴 **High**: > 2.5× TRACO. Frozen 1B+ code LLM, multi-agent inference loop, generative author model.

---

#### Track 1 — LOW-COMPUTE, HIGH-VALUE (default, pick first)

These are the directions where the WOW comes from **what is being modelled**, not from throwing more compute at it. Each fits inside one Kaggle session, each is its own paper if it lands.

##### 1.A. DENOISE — Diffusion-style code denoising pre-train 🟢 ★ priority
Pre-train the encoder with a code-denoising objective on the **training data itself** (no external corpus), then fine-tune for attribution. The noise function is structured and **style-preserving**: token-level corruption + AST-level perturbation (variable rename, control-flow jitter, paren canonicalisation) — all of which preserve author *intent* but break surface form. The encoder must reconstruct the original from the corruption, which forces it to learn **style-invariant semantic features AND style-discriminative residuals** simultaneously, because style is exactly what the noise *cannot* corrupt away.

Three concrete variants (start with v1, escalate if needed):
- **v1 — Discrete denoising (cheap, 🟢)**: BART-style. Mask K% of tokens + apply 1 random AST transform per sample. Train encoder + tiny decoder head to reconstruct original tokens. Re-use CARGO's structural-rewrite library — no new code needed for the noise function. 1 epoch warmup → standard TRACO fine-tune.
- **v2 — Score-based embedding diffusion (medium, 🟢-🟡)**: Add Gaussian noise to the *embedding* of each token at variance $\sigma_t$. Train a small "denoiser" MLP that predicts the clean embedding. Like a tiny continuous diffusion model living in encoder-output space. Final attribution head reads from the *score* (= noise prediction) at $t=0$.
- **v3 — Conditional style-preserving denoising (the WOW version, 🟡)**: Corrupt only *non-style* attributes (mask variable names but keep their *count*; rename methods but keep their *casing convention*; jitter whitespace but keep *indent depth*). The encoder must recover the corrupted attribute conditional on author-style features it already has. This is the version that gets the "we are the first to do this" claim.

- Name candidate: **DENOISE**, **STYLEDIFFUSE**, **CRAFT**, **DRIP** (Denoising Representation for Identity Profiling).
- New object: a code-specific structured-noise pretext task that **co-trains** style invariance and style discriminability.
- WOW hook: "Code attribution is a denoising problem in disguise: every plausible style-preserving perturbation reveals what the author's signal *isn't*, and what's left over is identity."
- Falsifier: if v1 matches MLM-pretrained at matched compute, the structural noise contributes nothing → drop to v2. If v3 matches v1, the style-preservation constraint is fake.
- Compute: v1 = +20–30% over TRACO; v2 = +30–40%; v3 = +40–50%. All comfortably one Kaggle session.

##### 1.B. RAGAR — Retrieval-augmented attribution with learned fusion 🟢
FAISS-index the training set. At inference: (1) encode query, (2) retrieve top-K neighbours, (3) a learned *fusion router* decides how much weight to give the parametric softmax vs the kNN vote vs a third "abstain" head. End-to-end trained with a margin loss that rewards correctly calibrated decisions.
- New object: the fusion router that arbitrates between parametric and non-parametric prediction.
- WOW hook: "Code attribution is partly memorisation and partly generalisation — we learn the split, per query."
- Falsifier: if the fusion router collapses to "always use parametric" (or "always use kNN"), the fusion is fake.
- Compute: same as TRACO + ~1 min FAISS index build at startup.

##### 1.C. DUALGRAPH — Token-graph ∩ AST-graph dual encoder 🟢
Run a small GNN on top of UniXcoder outputs, with **two** graph views per sample: a token-co-occurrence graph and an AST/DFG graph. Edge types in the AST graph encode the family-tree (S1) prior. Final representation is a learned fusion of the two graph readouts.
- New object: dual-graph encoding of code for attribution — first paper to do this.
- WOW hook: "Style lives on edges, not on tokens — and there are two graph topologies that disagree about which edges matter."
- Falsifier: if removing either graph view costs < 0.005 Macro-F1, the dual-graph claim collapses.
- Compute: +5–10% over TRACO (GNN on ~256 nodes is cheap).

##### 1.D. TREELEARN — Differentiable family-tree learning 🟢
Replace the hard-coded $\mathcal{T}$ (CARGO/TRACO genealogy) with a learnable tree, parameterised as a continuous ultrametric matrix or a Gumbel-softmax over edges. The tree is **discovered from data**, not from release notes — eliminating the reviewer objection that we use insider information.
- New object: a learnable ultrametric / Gumbel-tree co-optimised with the attribution loss.
- WOW hook: "We learn the genealogy of generators end-to-end. The discovered tree may not match the release-note tree — and that is the finding."
- Falsifier: if the learned tree converges to a star (no hierarchy) or to a chain, the prior is useless.
- Compute: same as TRACO + ~0.1% (just K² extra parameters where K = n_cls).

##### 1.E. AUXGENE — Multi-task encoder with auxiliary genealogy decoder 🟢
Train a single encoder with three heads: (i) classification, (ii) tree-distance regression on label pairs in the batch, (iii) decoding-temperature regression (predict the LLM's sampling temperature from the code). Use only head (i) at test time; (ii) and (iii) only shape the representation.
- New object: decoding-temperature as an auxiliary supervised signal — never used in code-attribution before.
- WOW hook: "We let the encoder *see* the sampling temperature during training, and it learns a more disentangled representation even when temperature is unobserved at test."
- Falsifier: if removing the temperature head gives the same accuracy, the auxiliary task is decorative.
- Compute: +5% over TRACO (two extra linear heads).

##### 1.F. PROMPTCOND — Prompt-conditioned attribution 🟢
Most CoDET-M4 samples share prompts across generators. Encode the *prompt* explicitly with the encoder; predict author conditional on the **(prompt, code)** pair. This directly tests whether style is the real signal, or whether attribution has been exploiting "this prompt produces consistent outputs from generator X".
- New object: explicit prompt-conditional attribution. Possibly with a *contrastive* loss that pulls (prompt, code) closer to (prompt, paraphrase-by-same-author) and pushes from (prompt, paraphrase-by-different-author).
- WOW hook: "We ask whether attribution is style-detection or prompt-leakage detection — and the answer reframes the entire field's benchmark."
- Falsifier: if prompt-conditioning hurts or matches prompt-free attribution, then style is the signal (which is fine — that becomes the paper's claim).
- Compute: +30–50% over TRACO (longer sequences = prompt + code).

##### 1.G. ACTIVE — Active-learning few-shot oracle 🟢
Train a smaller initial model on 0.5%, score every unlabeled sample by predicted-entropy or BALD, label only the top-K via oracle, repeat for R rounds. Pushes the 1% regime from random sampling to **informed** sampling.
- New object: an active-learning loop for the 1%/5%/20% few-shot regimes specifically — these regimes have not been treated as active-learning targets before in code-attribution.
- WOW hook: "The 1% regime is not a sample-efficiency problem — it is a *sample-selection* problem. We show the right 1% beats random 5%."
- Falsifier: if random 5% beats active 1%, active learning has not closed the regime gap.
- Compute: ~1.5× TRACO (multi-round training, but each round is small).

---

#### Track 2 — HIGH-WOW, HIGH-COMPUTE (backup; only if Track 1 stalls)

These need a 1B+ code LLM, multi-agent inference, or generative author models. WOW factor is higher but compute is too. Plan them only if a Track 1 method has been tried and failed to land.

##### 2.A. MAA — Multi-Agent Attribution (jury / tribunal) 🔴
Treat attribution as a **deliberation**, not classification. Spawn 3–5 small "specialist" agents (one per author family or per stylistic axis), each with its own prompt, LoRA, and inference budget. They argue, then a judge agent aggregates. Built on a 1B-2B code LLM with cheap LoRAs.
- WOW hook: "We replace softmax classification with a structured multi-agent debate."
- Falsifier: if a single-agent baseline at matched parameter count beats the jury, deliberation contributes nothing.

##### 2.B. MoE-Style — Mixture-of-Experts router over K style experts 🟡
A learnable router over K stylistic experts (each a small adapter on top of UniXcoder). The router is conditioned on a learned *style code* (per-sample latent), not on the code surface. Final attribution is a sparse mixture over expert logits.
- WOW hook: "Authorship is not a label — it is a router policy over latent style experts."
- Falsifier: if uniform mixture (no routing) matches the learned router, the router is useless.
- Compute: 🟡 — fits Track 1 if we keep K small (4–6 experts, each just a LoRA).

##### 2.C. SAD — Slot-Attention Code Decomposition 🟡
Slot attention (Locatello et al. 2020) applied to code *tokens*, decomposing each sample into K interpretable slots ("loop discipline", "naming style", "import pattern", etc.). Attribute from slot vectors. Unsupervised slots + supervised classifier.
- WOW hook: "We decompose code into interpretable style slots no prior code-attribution paper has surfaced."
- Falsifier: if slots collapse to CLS-token copies (slot-CLS cos > 0.95), no real decomposition.
- Compute: 🟡 — slot iter count is cheap; the win is qualitative (interpretability visualisations are reviewer gold).

##### 2.D. TTT-Sib — Test-time training via synthetic siblings 🔴
At test time, use a small code LLM to generate K paraphrases of the query, encode them, and update a tiny prediction head on the fly via consistency loss. Attribution becomes manifold-aware.
- WOW hook: "The model continues training at inference, on synthetic neighbours of the query, with no extra labels."
- Falsifier: if removing the TTT update changes nothing, it is decorative.

##### 2.E. HYPER — Frozen code LLM + hypernetwork head 🔴
Freeze DeepSeek-Coder-1.3B / StarCoder2-3B / Qwen2.5-Coder-1.5B. Train a hypernetwork that emits a per-author classifier from a small support set. At test time, synthesise a fresh classifier for the current author set.
- WOW hook: "We never train a classifier for a specific author — we *generate* one."
- Falsifier: if a trained-from-scratch linear probe at matched data beats the hypernet, dead weight.

##### 2.F. REASON — Reasoning-trace attribution 🔴
Use a small reasoning model (DeepSeek-R1-Distill-1.5B, Qwen3-Reason-1.7B, Phi-4-Reason) to produce a chain-of-thought attribution argument for each sample. Final prediction parsed from the argument, not from a softmax.
- WOW hook: "We replace softmax classification with explicit step-by-step authorship reasoning."
- Falsifier: if reasoning matches a CE-trained classifier within ±0.005, reasoning is decorative.

##### 2.G. ENCODERPIVOT — Pretrained code LLM as frozen encoder 🟡
Replace UniXcoder-base (125M) with a **frozen** 1B-2B code LLM (DeepSeek-Coder-1.3B, StarCoder2-3B, Qwen2.5-Coder-1.5B). Mean-pool final hidden, train only a small projector + classifier. If this beats fine-tuned UniXcoder, the field's encoder choice has been wrong.
- WOW hook: "Authorship attribution is solved by *not* fine-tuning — the right frozen encoder beats every fine-tuned baseline."
- Falsifier: if fine-tuned UniXcoder matches frozen 1B LLM at matched data, freezing is just an engineering convenience, not a finding.
- Compute: 🟡 — frozen forward only, no encoder backprop (so memory is OK).

##### 2.H. SIM — Generative author simulator (Bayesian attribution) 🔴
Train a small generative model $p(\text{code} \mid a, \text{prompt})$ per author. Attribution = Bayesian inference over priors.
- WOW hook: "We invert the problem — learn a generator per author and do Bayesian attribution."
- Falsifier: if discriminative baseline matches at matched parameters, generative direction has no marginal value.

---

**Selection guide (read before picking expNN):**

| Budget | Target | Recommended hero |
|:--|:--|:--|
| **≤ 1× TRACO** | safe contribution | DENOISE-v1 (1.A.v1), RAGAR (1.B), DUALGRAPH (1.C), TREELEARN (1.D), AUXGENE (1.E) |
| **1–1.5× TRACO** | oral-tier WOW | **DENOISE-v3 (1.A.v3) — TOP PICK**, ACTIVE (1.G), PROMPTCOND (1.F) |
| **1.5–2.5× TRACO** | high WOW | MoE-Style (2.B), SAD (2.C), ENCODERPIVOT-frozen-only (2.G) |
| **> 2.5× TRACO** | gamble for best paper | MAA (2.A), HYPER (2.E), SIM (2.H), TTT-Sib (2.D), REASON (2.F) |

**Default recommendation: start with DENOISE-v3 (1.A.v3).** It scores highest on WOW × tractability:
- Genuinely new pretext task no code-attribution paper has used.
- Reuses existing CARGO/TRACO plumbing (structural-rewrite library is already written).
- Low compute (≤ 1.5× TRACO).
- Clear story: "style is *what cannot be denoised away*" — quotable.
- Has falsifier that strips down to MLM, so the ablation writes itself.

### 3.3 Hero method status (updated 2026-05-20)

- **TRACO (exp76) — paper §3 placeholder.** Still in the paper's headline
  table; will stay there unless a Round 9 framework lands hard enough to
  reframe the narrative.
- **METATRACO (exp100) — composite #1 (0.5408), exceeds +0.010 unlock.**
  Decision pending whether to promote to §3 hero. Adds a MAML outer loop
  over TRACO; wins extreme few-shot, loses 20% slot by ~0.01 to TRACO.
  Already integrated into Paper Table 1 as `\method+\textsc{Meta}`.
- **Round 9 picks (planned 2026-05-20)**: CHORALE (exp117 done), PRISM
  (exp118 done), RESONANCE (exp114 in progress). These are the 3 hero
  candidates for the next round, picked for GENERAL impact (no family/
  sibling crutch) and CLEAN STORYTELLING (one declarative claim each).
  See §3.4.
- **Bar to claim hero displacement:** (i) beat METATRACO composite by
  ≥ +0.005, (ii) pass WOW Test §1.1 AND §1.5 (one-sentence claim test),
  (iii) pass own falsifier, (iv) the story it tells must be GENERAL
  enough to not feel benchmark-specific (per §1.4).

### 3.4 Round 9 hero candidates — the 3 picks

Three frameworks, three declarative claims, each general:

| Exp | One-sentence claim | Compute | Status |
|:--|:--|:--|:--|
| **exp117 CHORALE** | "Author identity is multi-scale: three voices, each cleansed of its own confound." | 🟡 | code done, run pending |
| **exp118 PRISM** | "We don't learn class embeddings — we DESIGN them and let the encoder refract code onto our prism." | 🟢 | code done, run pending |
| **exp114 RESONANCE** | "Author style is the bass frequency of code rhythm. The treble is decoder noise." | 🟢 | code in progress |

The 5 single-primitive ideas (ECHO, REWIND, CHORUS, MIRROR, LIGHTHOUSE)
and the heavier multi-component frameworks (FOREST, DECIPHER, MIRRORED,
ECHO+, HORIZON) documented in `Round_9_proposals.md` are §5 ablation
candidates, not §3 heroes.

---

## 4. Research question (reframed — 2026-05-20)

**Old framing (TRACO paper):** how should a few-shot code-authorship model
exploit the fact that the label space is not flat? *(Family-tree-centric;
reads as benchmark-specific.)*

**Newer framing (was 'what is the right primitive'):** too broad, too
list-like.

**Newest framing (the one we pitch in §1 if Round 9 lands):**
*What is the right representation for code authorship?* The field has
made three assumptions: (i) one pooled vector per code snippet, (ii) a
learned K-class classifier, (iii) token-position as the right axis to
read a code sequence. The three Round 9 picks reject one assumption each:

- **CHORALE** rejects single-scale pooling → identity is multi-scale.
- **PRISM** rejects learned classifier weights → geometry is hand-designed.
- **RESONANCE** rejects token-position → style lives in the frequency domain.

A reviewer can name the contribution as "they questioned three assumptions
the field never questioned." That is an oral pitch. The previous "tree-
weighting" pitch was a loss-engineering pitch; this one is a representation
pitch — much harder to summarise as a trick.

This reframing is what makes the paper an oral, not a long paper.

---

## 5. Repo layout

```
Exp_FewShot/testing_chis/      ← ACTIVE
  tracker.md                   ← live leaderboard (48 result JSONs)
  Round_9_proposals.md         ← 20 detailed proposals; CHORALE/PRISM/RESONANCE are picks
  exp{60..101}.py              ← Round 2-7 (TRACO, CARGO family, METATRACO, etc.)
  exp{102..106}.py             ← Round 8 (DENOISE/DUALGRAPH/ACTIVE/ULTRATREE)
  exp{107..126}.py             ← Round 9 picks + backup (Round 9 picks: 117, 118, 114)
  ext_*.py                     ← faithful external-baseline reproductions (7 files; 5 done)
  external_baselines/          ← read-only third-party clones
  legacy/                      ← read-only old methods
Paper/latex/main.tex           ← updated 2026-05-20 with ext_* baselines + METATRACO row
Paper/outline.md               ← section-level plan
Exp_Climb/, Exp_CodeDet/, Exp_DM/   ← frozen, do not touch
```

New experiments live in `Exp_FewShot/testing_chis/`. **Next free expNN ID
window: exp127+** (exp102–126 reserved or used; exp98/99 used by
uncertainty track).

---

## 6. Three hard rules (everything else is negotiable)

1. **Never reuse an `expNN` ID.** History is append-only.
2. **Always report `val_macro`, `test_macro`, and the `val_test_gap`** for every run. A method without a val--test gap report is not finished.
3. **Never mix benchmarks in a single Macro-F1 cell.** CoDET-M4 → Macro-F1, AICD-T2 → Macro-F1, Droid → Weighted-F1 (per the repo hook). Each benchmark stays in its own column.

Everything else, including encoder choice, schedule, augmentation pool, loss family, hero method, S-fact mapping, even the paper's framing, is up for renegotiation.

---

## 7. The 12-rule template (retained, with one carve-out)

Apply to every task unless explicitly overridden. Bias: caution over speed on non-trivial work.

1. **Think before coding** — state assumptions, ask if uncertain.
2. **Simplicity first** — **APPLIES TO PLUMBING ONLY.** The hero method's *conceptual surface* should be novel (see §1.1); its *implementation* should be as simple as the novelty allows.
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

**Project addendum:** every new `expNN_*.py` file MUST open with `# expNN — Method name` and include the 7-line theory block (NAME / ARXIV_ID / ONE-LINE CLAIM / EQUATION / WHY NOT BEFORE / WOW HOOK / FALSIFIER) before any imports. The WOW HOOK line is new (added 2026-05-20) and is the sentence we would put in the paper's abstract.

---

## 8. How to add a new experiment

1. **Pick the next free `expNN` ID** (currently exp93+).
2. **Write the 7-line theory block** including the WOW HOOK. If you cannot write a WOW HOOK that doesn't sound like 2023, the experiment fails §1.1 and should not be planned.
3. **Decide whether you need the existing protocol or a new one.** If new (different encoder, different fractions, different schedule), justify it in one paragraph at the top.
4. **Implement.** Reuse plumbing from the closest existing exp but build the hero object from scratch — do not bolt a new loss onto an old model.
5. **Always report val + test + val--test gap.** Output a single `{expNN_method}_results.json` with the full eval pack (overall, per_class, per_language, per_source, confusion_matrix, sibling_confusion_rate, cross_family_confusion_rate, val_history).
6. **Update `tracker.md`** with the row.

Theory block example:
```
# exp93 — METHOD_NAME
# NAME       : ...
# REFERENCE  : ... (arXiv id or "new")
# CLAIM      : one sentence
# EQUATION   : L = ...   (or: the new architectural primitive)
# WHY NEW    : what existing method does NOT do this
# WOW HOOK   : the one sentence we want a reviewer to remember a week later
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

Literature for §3.2 hot directions (read before planning):
- **MoE / Routing**: Switch Transformer (arXiv:2101.03961), Mixtral-of-Experts (arXiv:2401.04088), Soft MoE (arXiv:2308.00951), AdaMix (arXiv:2205.12410)
- **Slot Attention**: Locatello et al. NeurIPS 2020 (arXiv:2006.15055), SLATE (arXiv:2110.11405), Slot-DINO (arXiv:2401.02835)
- **Multi-Agent / Debate**: AutoGen (arXiv:2308.08155), Society of Minds debate (arXiv:2305.14325), Multi-Agent Debate (arXiv:2305.19118), AgentBench (arXiv:2308.03688)
- **Test-Time Training**: TTT Sun et al. (arXiv:1909.13231), TTT-MAE (arXiv:2209.07522), MEMO (arXiv:2110.09506)
- **Hypernetworks for few-shot**: HyperNet (Ha et al. arXiv:1609.09106), HyperTransformer (arXiv:2201.04182), CoCa-Hypernet (arXiv:2403.01209)
- **Retrieval-augmented classification**: kNN-CLIP (arXiv:2304.09872), RA-CLS (arXiv:2210.02928), REINA (arXiv:2105.06409)
- **Code-attribution prior work**:
  - LASCL — Label Hierarchy SupCon (arXiv:2402.00232)
  - HCAL — Hierarchy-Consistent Adaptive Loss (arXiv:2508.13452)
  - Tournament Code Attribution (arXiv:2501.08165)
  - LLM-AuthorBench Stylometry (arXiv:2506.17323)
  - CodeT5-JSA / Hidden DNA structural patterns (arXiv:2510.10493)
  - Code Fingerprints / DCAN disentangled attribution (arXiv:2603.04212)
- **Reasoning models (small)**: DeepSeek-R1-Distill (arXiv:2501.12948), Qwen3-Reason, Phi-4-Reason
- **Code LLMs for use as frozen encoder**: DeepSeek-Coder-V2 (arXiv:2406.11931), StarCoder2 (arXiv:2402.19173), Qwen2.5-Coder (arXiv:2409.12186)

---

## 10. One closing note (updated 2026-05-20)

If you are reading this and thinking "should I run a small parameter sweep
on TRACO to squeeze another 0.005 Macro-F1?", **stop**. The paper already
has METATRACO (composite #1, 0.5408) over TRACO (0.5256). The paper is
missing its **second act** — a representation-level reframing.

If you are reading this and thinking "should I propose `TRACO + sibling-
distance-square + extra-tree-weight`?", **also stop**. §1.4 explicitly
rejects more family/sibling tricks. Family-tree-weighted SupCon is the
spotlight; piling more sibling-only heuristics reads as benchmark-overfitting.

If you are reading this and thinking "should I combine X+Y+Z into a
multi-component framework with five hyperparameters?", **stop and read
§1.5**. The contribution must read as a single declarative claim about
the problem (e.g. "style is multi-scale"), not as a list of components.

**Default action 2026-05-20**: the 3 Round 9 hero picks (CHORALE, PRISM,
RESONANCE) are documented in `Round_9_proposals.md` and partially
implemented:
- `exp117_chorale.py` (done) — multi-scale + adversarial confound
- `exp118_prism.py` (done) — designed geometry + spectral filter
- `exp114_resonance.py` (in progress) — pure DCT spectral story

Run all three on Kaggle as a controlled bake-off. Whichever wins composite
becomes the new §3 hero; the other two become §5 ablation evidence for the
three-assumptions-rejected narrative (see §4).

The bar for what we put in §3 of the paper is **(§1.1 WOW Test) AND
(§1.5 storytelling test) AND (general impact, not benchmark-specific)**.
Three filters in series. If you cannot write a one-sentence claim that
passes all three, do not start coding.
