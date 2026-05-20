# Round 9 — Ten oral-tier framework proposals (exp107–exp116)

> Generated 2026-05-20. Ten experiments designed to clear the EMNLP 2026 oral
> bar: each introduces a new computational primitive, combines ≥ 2 components,
> and ships a falsifier the reviewer cannot wave away. Every proposal pulls
> from a documented insight in `tracker.md`.

---

## Cross-cutting design constraint

Every Round 9 exp must satisfy ALL of:

1. **A new computational primitive** the field has not used for code attribution.
2. **A title that reads like 2026, not 2023** — name the framework, not the loss.
3. **A falsifier with a measurable threshold** (no "we expect it to help").
4. **A reusable plumbing dependency** (TRACO supcon, CARGO library, METATRACO outer-loop, FAISS, frozen LLM) — no greenfield rebuild.
5. **A "wow hook" sentence** that lands in the abstract of the would-be standalone paper.

---

## Tracker insights that drove these proposals

| Insight (where in tracker.md) | What it implies |
|---|---|
| Sibling confusion = 50%+ of off-diagonal errors | Sibling-aware structure is the right target |
| 20% saturation at composite ~0.71 across top 8 methods | Encoder/schedule, not loss, dominates at high data → high-data wins need ARCHITECTURE not LOSS |
| CARGO ⊥ TRACO (CARMIX wins) | Surface and structural augmentations are complementary, not redundant |
| CARGO v1 (30% fire) > v2 (100% fire) | Moderate noise > saturating noise; gradient signal needs ambiguity |
| RACO/TKL contradiction | Loss-level and rep-level priors have OPPOSITE dependence on n — direction of dependence is loss-family-specific |
| METATRACO wins 1% by +0.07 AICD | Meta-learning specifically wins extreme few-shot |
| TRACOD EMA collapsed | Self-distillation via teacher EMA fails — needs DIFFERENT signal |
| SETFIT-TW dropped 15 pts | Frozen encoder fails — encoder MUST adapt |
| DUALVIEW failed (code↔comment) | Cross-modal pairing degrades attribution — code↔code only |
| DCGPT tightest val-test gap (+0.0025) | Consistency regularization works; KL on perturbed views is non-trivial |
| LLMSniffer collapsed | `cls.detach() + lr=1e-6` design is incompatible with K-class |

---

## The ten proposals

Each proposal includes the 7-line theory block (NAME / REFERENCE / CLAIM /
EQUATION / WHY NEW / WOW HOOK / FALSIFIER), an architecture sketch, and a
compute tag (🟢 low / 🟡 medium / 🔴 high).

---

### exp107 — **ECHO** — Self-reinforcing pseudo-labels via ensemble disagreement 🟢

```
# exp107 — ECHO
# NAME       : ECHO (Ensemble disagreement as pseudo-label oracle)
# REFERENCE  : new; co-training (Blum & Mitchell 1998), confident-tri-training
#              (Tarvainen 2017), but using DISAGREEMENT as positive signal
#              instead of agreement.
# CLAIM      : The training set is too small at 1%, but the unlabeled
#              pool (the rest of the training data, without their labels)
#              is large. Train K=5 TRACO models with different augmentation
#              seeds. On the unlabeled pool, the K models DISAGREE. We
#              pseudo-label the disagreement samples with the highest-
#              confidence vote, ADD them to the labeled pool, and retrain.
#              The pseudo-labels are systematically harder than random
#              samples and target the encoder's failure mode.
# EQUATION   : Given K TRACO models {phi_k}, unlabeled x:
#                p_k = softmax(phi_k(x)); v(x) = entropy(mean_k(p_k))
#                accept if v(x) > tau_disagree AND max_k argmax(p_k)
#                appears in >= K/2 models
#                add (x, mode(predictions)) to labeled pool, retrain final
# WHY NEW    : TRACOD (exp80) used EMA teacher and collapsed. ECHO uses
#              disagreement, the OPPOSITE signal: samples where the ensemble
#              fights are precisely the samples that contain the most
#              information for the next training round.
# WOW HOOK   : "The model's disagreement IS the labeling oracle. We mine
#              the ensemble's own confusion and feed it back as
#              training signal."
# FALSIFIER  : (F1) If ECHO's pseudo-labeled pool size < 5% of unlabeled
#              pool at frac=0.01, disagreement signal is too weak → falsified.
#              (F2) If ECHO composite ≤ best single TRACO model, the
#              pseudo-labeling adds nothing.
```

**Architecture**: 5 TRACO encoders trained with different `random.seed(0..4)` for augmentation. After initial training, run all 5 on unlabeled pool, compute predictive entropy and vote consensus. Threshold both, add to labeled pool with consensus label. Retrain ONE final TRACO model on the augmented pool. Reports: pseudo-label pool size, accuracy of pseudo-labels (since we have ground truth), F1 lift over best single model.

**Reuses**: exp76 TRACO architecture verbatim, just multi-seed launcher + pseudo-label injection.

---

### exp108 — **GRAVITY** — Fixed-point attention over retrieval set 🟡

```
# exp108 — GRAVITY
# NAME       : GRAVITY (Author embedding as fixed point of retrieval-attention)
# REFERENCE  : new; combines DEQ-style fixed-point inference (Bai 2019)
#              with retrieval-augmented attribution (kNN-CLIP, RA-CLS).
# CLAIM      : The author identity of a query code is NOT a vector
#              the encoder must output in one shot. It is a FIXED POINT
#              of an iterative operator that (i) retrieves K nearest
#              training samples, (ii) attends over them weighted by
#              their labels, and (iii) refines the query embedding.
#              The fixed point IS the identity prediction.
# EQUATION   : z_0 = phi(query)
#              for t = 1..T:
#                N_t = top-K(z_{t-1}, train_emb)        # retrieval
#                a_t = softmax(z_{t-1} @ N_t.T / tau)   # attention
#                z_t = LayerNorm(z_{t-1} + a_t @ N_t)   # refine
#              y_hat = argmax_c sum_{i in N_T : y_i == c} a_T[i]
#              Trained end-to-end with TRACO supcon on z_T (fixed-point unrolled).
# WHY NEW    : RAINFER (exp94) used kNN at inference but with a single
#              retrieval step. GRAVITY iterates retrieval, letting the
#              query EMBEDDING DRIFT toward the cluster center of its
#              likely author. The drift is the prediction.
# WOW HOOK   : "Authorship is not a label assigned by softmax. It is
#              a fixed point of attention over the data manifold —
#              the query embedding drifts until it lands on its author."
# FALSIFIER  : (F1) If ||z_T - z_{T-1}|| > 0.1 at T=10, no fixed point
#              → falsified. (F2) If accuracy at T=1 = accuracy at T=10,
#              the iteration is decorative.
```

**Architecture**: UniXcoder encoder + FAISS index of training embeddings + iterative attention block (3-layer transformer cell, weight-tied). Test-time inference does T=5 iterations. Training: unroll T=3 steps and apply TRACO supcon on z_T. Reports: convergence curve ||z_t - z_{t-1}|| vs t, accuracy at each t.

**Reuses**: TRACO encoder, RAINFER's FAISS code.

---

### exp109 — **SCHISM** — Adversarial co-training of sibling-merger vs sibling-splitter 🟡

```
# exp109 — SCHISM
# NAME       : SCHISM (Adversarial co-training: sibling-splitter vs sibling-merger)
# REFERENCE  : new; adversarial discriminator framework (GAN, ACoL 2018),
#              re-purposed for hierarchical attribution.
# CLAIM      : The signal that distinguishes siblings (Qwen vs Nxcode)
#              is by definition WEAK — they share a fine-tune family.
#              If we train TWO encoders adversarially, where encoder A
#              pushes siblings APART while encoder B pulls them
#              TOGETHER, the COMPONENT THAT SURVIVES the subtraction
#              z_A - z_B is exactly the sibling-distinctive signal.
# EQUATION   : z_A = phi_A(x), z_B = phi_B(x)
#              L_A = CE + lambda * TRACO_supcon(z_A) with sibling neg weight 10x
#              L_B = CE + lambda * 1 - cos(z_B^i, z_B^j) for sibling pairs
#              Classifier: clf(z_A - z_B)
#              Two-step training: alternate A-step, B-step like GAN.
# WHY NEW    : Existing methods use ONE encoder + ONE loss. SCHISM uses
#              TWO encoders + adversarial training, where one acts as
#              the "what NOT to learn" anchor. The subtraction is the
#              sibling signal isolated from generic stylometry.
# WOW HOOK   : "We train a sibling-merger that we then subtract.
#              What survives is identity."
# FALSIFIER  : (F1) ||z_A - z_B|| on sibling pairs > ||z_A - z_B|| on
#              cross-family pairs by factor 2x (interpretable residual).
#              (F2) Composite > TRACO; if not, the adversary is decorative.
```

**Architecture**: Two UniXcoder encoders (A, B), each with its own projector + small classifier head. Training alternates: 1 step train A with TRACO supcon (sibling neg w=10), 1 step train B with anti-supcon (force sibling cos→1). Final classifier reads z_A - z_B (or [z_A; z_B] concat with learned fusion). Reports: per-pair ||z_A - z_B|| histogram, sibling vs cross-family ratio.

**Reuses**: TRACO encoder × 2, CARGO library.

---

### exp110 — **REWIND** — Backwards distillation with attention transfer 🟢

```
# exp110 — REWIND
# NAME       : REWIND (Backwards distillation: 20% teacher → 1% student via attention)
# REFERENCE  : new; combines Hinton 2015 knowledge distillation with
#              attention transfer (Zagoruyko 2017) and few-shot setting.
# CLAIM      : The 1% slot is a sample-efficiency problem and the 20%
#              slot has already been won (saturation at 0.71). Train a
#              teacher on 20%, then distill to a student trained on 1%
#              ONLY, transferring not just soft labels but PER-LAYER
#              ATTENTION PATTERNS. The student inherits "where to look"
#              from the teacher's 20% experience without needing the data.
# EQUATION   : Phase 1: teacher = TRACO on 20% data.
#              Phase 2: student = TRACO on 1% data + 3 extra loss terms:
#                L_soft = KL(student logits/T, teacher logits/T)
#                L_attn = sum_layers ||A_student^l - A_teacher^l||^2 (attention maps)
#                L_repr = 1 - cos(z_student, z_teacher)              (embedding)
#              Student is initialized from teacher weights but trained on 1%.
# WHY NEW    : TRACOD (exp80) used EMA momentum and collapsed. REWIND
#              uses a STATIC teacher trained on more data, with attention
#              transfer that no prior code-attribution method has tried.
# WOW HOOK   : "Train backwards. The teacher knows what 20% looks like;
#              the 1% student inherits not weights but VIEWPOINT."
# FALSIFIER  : (F1) Student F1 @ 1% > TRACO @ 1% by ≥ 0.02; if no
#              gap, attention transfer adds nothing. (F2) Attention map
#              distance ||A_s - A_t|| drops below 0.1 during training;
#              if it stays at 1.0, attention transfer is fake.
```

**Architecture**: Two-phase training in a single script. Phase 1: TRACO on 20% data → save full state including per-layer attention. Phase 2: TRACO on 1% data with the three extra losses against frozen teacher. Apply attention transfer on all 12 UniXcoder layers. Reports: per-phase Macro-F1, per-layer attention distance curves.

**Reuses**: TRACO architecture + supcon, METATRACO's two-phase training pattern.

---

### exp111 — **CHORUS** — Adversarial disentanglement of author ⊥ decoding temperature 🟢

```
# exp111 — CHORUS
# NAME       : CHORUS (Author identity as residual of code after subtracting temperature)
# REFERENCE  : new; gradient reversal (Ganin 2015) repurposed for
#              orthogonalizing two known confounds in code generation.
# CLAIM      : CoDET-M4 metadata includes per-sample decoding temperature.
#              That temperature is a CONFOUND: same author at T=0.2 vs
#              T=0.9 looks different; cross-author at the same T looks
#              similar in surface stats. Train the encoder to be
#              SIMULTANEOUSLY good at predicting author AND BAD at
#              predicting temperature, via gradient reversal. The
#              residual representation captures only the temperature-
#              invariant identity signal.
# EQUATION   : z = phi(x)
#              L_main = CE(W_author @ z, y_author)
#              L_aux  = CE(W_temp @ GradReverse(z), y_temp)   # we WANT this high
#              L = L_main + lambda * L_aux
#              At test, use only W_author.
# WHY NEW    : DECOFP (exp70) predicted temperature as auxiliary. CHORUS
#              instead UN-predicts it — explicitly removes temperature
#              from the representation. Inverts the sign of the signal.
# WOW HOOK   : "Author identity is what is LEFT OVER after we remove
#              decoding temperature. We disentangle by negative gradient."
# FALSIFIER  : (F1) After training, a frozen-z linear probe should NOT
#              be able to predict temperature (probe Macro-F1 < 0.30 vs
#              chance 1/T_buckets). (F2) Composite > TRACO at the slots
#              where temperature variance is highest.
```

**Architecture**: UniXcoder + projector + TWO heads (author, temperature). Temperature head reads z through `GradReverse` layer. Standard TRACO supcon on z for author identity. Reports: post-training probe accuracy on z for temperature (should be low), confusion matrix vs T-bucket.

**Reuses**: TRACO; need temperature metadata extraction (already in CoDET-M4 schema).

---

### exp112 — **SNAPSHOT** — K representative examples replace the softmax 🟢

```
# exp112 — SNAPSHOT
# NAME       : SNAPSHOT (Per-author signatures, not weights)
# REFERENCE  : new; combines prototypical networks (Snell 2017) with
#              data-attribution / example-selection theory (TracIn 2020).
# CLAIM      : Standard K-way classifier learns K weight vectors W_c.
#              We instead pick ONE labeled training example per author
#              as the "signature." Classification = cosine to each
#              signature. The TRICK: signatures are CHOSEN by gradient
#              descent on the validation set, not by random selection
#              or class mean. Best example per author is the one whose
#              embedding maximizes downstream val Macro-F1 as a kNN
#              center.
# EQUATION   : For each author c, signature s_c ∈ {labelled examples of c}
#              Selection: s_c = argmin_{x in train_c} [
#                val_loss(query → argmax_c' cos(phi(query), phi(s_c'))) ]
#              evaluated via gradient over discrete-set selection
#              (Gumbel-softmax or greedy with margin).
#              Encoder phi trained with TRACO + signature-margin loss.
# WHY NEW    : Prototype networks use CLASS MEAN. TAPA (exp73) used
#              prototypes and collapsed. SNAPSHOT uses TRAINED SELECTION
#              of a single example — the model gets to pick its own
#              ambassador per class.
# WOW HOOK   : "We replace the K-way softmax with K labeled examples.
#              The classifier IS the dataset, selected for maximum margin."
# FALSIFIER  : (F1) Selected signatures have val-kNN F1 > class-mean
#              prototype F1 by ≥ 0.02. (F2) Random-signature baseline
#              within 0.005 of SNAPSHOT → selection adds nothing.
```

**Architecture**: TRACO encoder. After training, do a discrete optimization (Gumbel-softmax soft selection + temperature annealing, or beam search) to pick best example per author. At test time, classify by argmax cosine to signature embeddings. Reports: signature identity per author, robustness to signature swap.

**Reuses**: TRACO encoder; simple post-hoc selection routine.

---

### exp113 — **FLUX** — Normalizing flow from human-style to AI-style 🔴

```
# exp113 — FLUX
# NAME       : FLUX (Per-author generative flow; attribution by inverse trajectory)
# REFERENCE  : new; normalizing flows (Rezende 2015, RealNVP 2017,
#              Glow 2018) applied to discrete-code embedding space.
# CLAIM      : Train K invertible flows {f_c}, one per author, where
#              f_c maps a fixed "human-style" base distribution to the
#              data distribution of author c. At inference, given query
#              q, compute log p_c(q) = log p_base(f_c^{-1}(q)) +
#              log|det J_{f_c^{-1}}(q)| for each c. Predict argmax_c.
#              This is BAYESIAN attribution with a generative flow as
#              the per-author density model.
# EQUATION   : f_c: R^d -> R^d, invertible
#              p_c(z) = p_base(f_c^{-1}(z)) * |det J_{f_c^{-1}}(z)|
#              y_hat = argmax_c log p_c(phi(query))
#              Train: max_phi,f_c sum_{x ∈ class c} log p_c(phi(x))
# WHY NEW    : Existing methods are discriminative. FLUX is generative
#              + invertible — the first attribution method to learn a
#              full density per author and do likelihood-ratio test.
# WOW HOOK   : "We learn the trajectory from human-style to each AI's
#              style. Identity is the trajectory that, run backwards,
#              brings the query closest to human."
# FALSIFIER  : (F1) Flow Jacobian should be well-conditioned (sigma_min /
#              sigma_max > 0.01) for inversion stability. (F2) Composite
#              > TRACO at AICD 1% (where generative direction has most
#              marginal value vs discriminative).
```

**Architecture**: UniXcoder + projector → 256-dim embedding. K small flows (3-coupling-layer RealNVP per author, ~50K params each). Train phi jointly with all flows. Inference: run all K inverse flows + Jacobian determinants on query. Reports: per-author log-likelihood histograms.

**Reuses**: UniXcoder encoder; new flow module (~200 lines).

---

### exp114 — **RESONANCE** — Spectral decomposition of code embeddings 🟢

```
# exp114 — RESONANCE
# NAME       : RESONANCE (Author style is the DC component of code rhythm)
# REFERENCE  : new; DCT-based signal decomposition repurposed for
#              authorship at the embedding-sequence level.
# CLAIM      : Token-position embedding sequence h_1...h_L (after
#              UniXcoder) is a 1D signal. Apply DCT along the position
#              axis. AUTHOR STYLE is hypothesized to be a LOW-FREQUENCY
#              signature (consistent over the whole sample), and LLM-
#              SPECIFIC NOISE is HIGH-FREQUENCY (per-token sampling
#              variance). Classifier reads only the first K DCT
#              coefficients per channel, summed across channels.
# EQUATION   : H = [h_1; ...; h_L] ∈ R^{L x d}
#              DCT[H][k] = sqrt(2/L) sum_n h_n cos(pi(2n+1)k / 2L)
#              z = flatten(DCT[H][:K])    # only first K freq components
#              logits = W z
# WHY NEW    : No prior code-attribution paper analyzes the FREQUENCY
#              CONTENT of code embeddings. We bring the signal-processing
#              perspective to the table: author = bass frequencies,
#              decoding noise = treble.
# WOW HOOK   : "Author style is the DC component of code rhythm.
#              The high-frequency noise is what generators leak
#              through, but identity is in the bass."
# FALSIFIER  : (F1) F1 at K=8 coefficients ≈ F1 at K=512 (all of them)
#              → spectral hypothesis holds. (F2) F1 at K=8 << F1 at
#              K=512 → high-frequency carries identity, hypothesis fails.
```

**Architecture**: UniXcoder → per-token hidden states → DCT along token dim (no learnable params) → keep first K=8/16/32/64 coefficients → linear classifier. Optional: TRACO supcon on top of the spectral representation. Reports: F1 vs K curve, per-frequency component importance.

**Reuses**: TRACO. DCT is one PyTorch line (`torch.fft.dct` or manual cosine matrix multiply).

---

### exp115 — **LIGHTHOUSE** — Fixed ETF geometry + sibling repulsion bonus 🟢

```
# exp115 — LIGHTHOUSE
# NAME       : LIGHTHOUSE (Designed class geometry; encoder projects onto it)
# REFERENCE  : new; equiangular tight frame theory (Papyan 2020 neural
#              collapse) + sibling boost from this paper's tree structure.
# CLAIM      : Standard classifier W ∈ R^{K x d} is learned. We FIX
#              W to be an Equiangular Tight Frame (ETF: row vectors
#              maximally spread on the unit sphere), PLUS we add an
#              extra REPULSION between SIBLINGS so sibling vectors are
#              even further apart than they would be in a vanilla ETF.
#              The encoder learns to PROJECT onto this hand-designed
#              geometry. No classifier weights to learn.
# EQUATION   : W is fixed: W = ETF(K, d) + delta * SiblingRepulse(adj)
#              where SiblingRepulse pushes pairs (i, j) in adj further
#              by adding extra component in some direction.
#              Encoder learns phi such that <phi(x), W_y> is maximized.
# WHY NEW    : ETF-Simplex (exp_n09) used vanilla ETF and underperformed
#              at low data. LIGHTHOUSE adds the genealogy prior into the
#              ETF construction itself — sibling separation is HARD-CODED
#              into the target geometry, not learned.
# WOW HOOK   : "We don't learn the class embeddings — we DESIGN them.
#              Class vectors are an ETF; siblings get a bonus push.
#              The encoder learns to PROJECT onto our geometry."
# FALSIFIER  : (F1) cos(phi(x), W_y) > cos(phi(x), W_y') for y != y'
#              on test (geometry is matched). (F2) Composite > TRACO,
#              else the hand-designed geometry adds no value.
```

**Architecture**: Construct K×d matrix as ETF + sibling-repulsion correction. Set as `nn.Parameter(W, requires_grad=False)`. Train encoder + projector with CE + TRACO supcon, using cosine similarity to the fixed W rows as logits. Reports: cosine matrix of trained encoder vs W (should be near-diagonal).

**Reuses**: TRACO supcon, GENE_ADJ for sibling identification.

---

### exp116 — **MIRROR** — Reverse-token contrastive: code-order invariance 🟢

```
# exp116 — MIRROR
# NAME       : MIRROR (Token-reverse positive pairs: position-invariant style)
# REFERENCE  : new; combines view-augmentation (TRACO) with the strong
#              prior that author STYLE should be invariant to the order
#              we read the tokens in.
# CLAIM      : Surface features (token sequence order) carry no
#              authorship information — author style is captured by
#              statistics over the code, not the linear order. If the
#              encoder learns this invariance, it will be robust to
#              decoder ordering noise. Train with positive pairs =
#              (forward(x), reverse(x)), in addition to TRACO's
#              augmentation positives.
# EQUATION   : x_rev = tokens_of(x)[::-1]   # token-level reversal
#              z = phi(x), z_rev = phi(x_rev)
#              L_invariance = 1 - cos(z, z_rev)
#              L = L_CE + lambda_TRACO * L_TW + lambda_inv * L_invariance
# WHY NEW    : ContraCode (Jain 2020) used AST-renaming positives.
#              MIRROR uses TOKEN-ORDER REVERSAL — a stronger and more
#              surprising invariance claim. If true, it implies
#              authorship is a bag-of-features property, not a sequential
#              one — a publishable insight on its own.
# WOW HOOK   : "We teach the encoder that code style is order-invariant.
#              Forward and reversed code are the same author. If true,
#              authorship is bag-of-features; if false, the order
#              matters and we learn what survives reversal."
# FALSIFIER  : (F1) cos(z, z_rev) > 0.85 on test, AND composite >=
#              TRACO → invariance learned without harm. (F2) cos < 0.4
#              → encoder rejects the invariance, meaning order DOES
#              carry style. Either outcome is publishable.
```

**Architecture**: Standard TRACO model. FSDS adds reversed-token view as a third forward pass. New loss term penalizes 1 - cos(z, z_rev). Reports: per-class cos(z, z_rev) distribution, F1 vs lambda_inv ablation.

**Reuses**: TRACO architecture verbatim; just one extra forward + loss term.

---

## Selection guide — what to run first

Ranked by (WOW × tractability):

| Rank | Exp | Compute | Why first |
|:--|:--|:--|:--|
| 1 | **exp111 CHORUS** | 🟢 | Cheapest, most "wow" — gradient reversal for confound removal, applicable to CoDET-M4 immediately |
| 2 | **exp114 RESONANCE** | 🟢 | One PyTorch line for DCT; falsifier is a clean ablation curve |
| 3 | **exp116 MIRROR** | 🟢 | Token-reverse view is trivial; either outcome (cos high or low) is publishable |
| 4 | **exp110 REWIND** | 🟢 | Reuses TRACO + METATRACO plumbing; cheap; targets the 1% slot directly |
| 5 | **exp107 ECHO** | 🟢 | Multi-seed launch + post-hoc pseudo-label; no new architecture |
| 6 | **exp115 LIGHTHOUSE** | 🟢 | Tiny code change (fix W matrix), strong story |
| 7 | **exp112 SNAPSHOT** | 🟢 | Discrete selection is the only new piece; rest is TRACO |
| 8 | **exp108 GRAVITY** | 🟡 | FAISS + iterative attention — needs more engineering |
| 9 | **exp109 SCHISM** | 🟡 | Two encoders + adversarial alternation — non-trivial |
| 10 | **exp113 FLUX** | 🔴 | Per-author flows is the highest-WOW but also highest-compute |

**Recommended first batch (3 days)**: exp111 + exp114 + exp116. Each is 🟢, none requires new infrastructure, and collectively they cover three distinct novelty axes (disentanglement, spectral, invariance).

**Stretch second batch (10 days)**: exp110 + exp107 + exp115. Adds distillation, ensemble, and geometry axes.

**Best-paper gambits**: exp113 FLUX (generative attribution) or exp109 SCHISM (adversarial co-training). One full Kaggle session each.

---

## Storytelling threads for the paper

Several proposals share a thematic spine that can be braided into a single
oral pitch:

1. **"What is the right computational primitive for code attribution?"**
   exp107/108/109 each propose a different primitive (ensemble, fixed-point,
   adversarial). The paper section can list them as alternative answers and
   benchmark them head-to-head.

2. **"Identity is the residual"** — exp111/116/114 all frame authorship as
   what is LEFT OVER after subtracting (temperature / order / high
   frequency). A single oral slide with three subtraction equations is
   memorable.

3. **"Hand-designed beats learned for the K-class head"** — exp112/115
   both propose fixing the K class prototypes (one from data, one from
   theory) and only learning the encoder's projection. The contrast against
   the standard learned-W baseline is a single graph.

Pick one of the three threads, run the three corresponding exps, and the
oral talk writes itself.

---

## Risk register

- **All "low-compute" proposals share a hidden risk**: the saturation band
  at 20% means a NEW method that scores ~0.71 is indistinguishable from
  noise relative to existing top-8 methods. To break out, target 1% or 5%
  slots where the gap is real.
- **METATRACO already wins 1% by +0.07 on AICD-T2**: Round 9 proposals
  that don't beat that are weaker than METATRACO and become §5 ablations
  rather than §3 heroes. Aim for ≥ +0.02 over METATRACO at 1% to claim
  hero displacement.
- **ext_llmsniffer-style architectural collapse risk**: exp109 SCHISM
  (two encoders + adversarial) and exp113 FLUX (per-author generative)
  are exactly the kind of design that can collapse to random. Falsifiers
  (F1, F2) are mandatory pre-flight checks.

---

## Implementation order

The proposals above are PLANS, not yet runnable Python. Once user picks
the first batch:

1. Copy `exp84_cargo.py` as template
2. Re-use plumbing (data loading, eval_pack, GENE_ADJ, FSDS)
3. Implement ONLY the new architectural primitive (typically 50-200 lines)
4. Add the falsifier reporting block to the result dict
5. Update tracker.md Round 9 row

Average implementation effort per exp: 200-300 new lines on top of TRACO
template = 600-800 line files (in line with existing exp10x files).

---
---

# Round 9 v2 — MEGA frameworks (multi-component, multi-insight)

> The single-primitive proposals (exp107–exp116) above are each one
> reviewer-eyebrow. The proposals below are each a **whole paper section**.
> Every MEGA framework combines ≥ 3 existing tracker wins with one new
> architectural object. The "wow" is in the SYNTHESIS, not any single piece.
> These read as named systems (PRISM, CHORALE, NEUROCHAIN) rather than
> as loss tweaks — they are intended to be the kind of method a reviewer
> remembers a year later.

Each framework lists: (1) the new architectural object, (2) the existing
components it integrates, (3) the storytelling spine, (4) why no prior
work has assembled this combination, (5) falsifier panel (≥ 3 hooks).

---

## exp117 — **CHORALE** — Three-scale disentangled fusion with adversarial confound removal 🟡

> Insight cocktail: CHORUS adversarial disentanglement + multi-scale tokens
> + TRACO supcon + sibling-conditioned gating.

```
# exp117 — CHORALE
# NAME       : CHORALE (Choir of three scales × adversarial confound subtraction)
# CLAIM      : Author identity lives at THREE SCALES simultaneously — token-bag
#              (vocabulary signature), sequence (CLS attention pattern), and
#              structure (AST node-type histogram). Existing methods collapse
#              all three into one pooled vector and lose the per-scale signal.
#              CHORALE keeps THREE parallel scale-specific encoders, each with
#              its own classifier head AND its own adversarial confound remover
#              (gradient reversal against decoding-temperature). A sibling-
#              conditioned gating network learns which scale dominates per
#              sibling-pair: e.g. "siblings A–B distinguished mainly at the
#              vocabulary scale" is something the gate can learn.
# COMPONENTS :
#   1. Three encoders share UniXcoder body, three different pool layers:
#      (a) mean-pool of token embeddings (bag scale)
#      (b) CLS hidden state (sequence scale)
#      (c) AST node-type histogram → small MLP (structure scale)
#   2. Each scale gets gradient-reversal head against temperature (CHORUS).
#   3. TRACO tree-weighted SupCon on the FUSED 3-scale representation.
#   4. Sibling-gated mixer: g(y_i, y_j) → soft weight over three scales,
#      learned only on sibling pairs identified by GENE_ADJ.
# WOW HOOK   : "We replace pooled embeddings with a CHORALE of three
#              scale-specialists, each cleansed of its own confound. The
#              gate learns which voice carries identity for each sibling
#              pair — and the answer is different for each pair."
# FALSIFIER  :
#   (F1) Per-scale ablation: removing any of 3 scales must cost ≥ 0.005 Macro-F1.
#   (F2) Sibling-gate weights must not collapse to (1,0,0) — three scales
#        must each get > 0.10 mass on at least one sibling pair.
#   (F3) Adversarial probe Macro-F1 on temperature < 0.30 (chance level).
#   (F4) Composite > METATRACO at AICD 1% (the largest sibling-pair-dense
#        slot) by ≥ +0.01.
```

**Why no one has done this**: Multi-scale code encoders exist (GraphCodeBERT)
but only as a single fused representation. No prior code-attribution paper
keeps three separate scale-specialist encoders + per-scale adversarial
disentanglement + a sibling-conditioned gating mixer. Each piece exists
elsewhere; their combination is the contribution.

**Compute**: 🟡 — 3× encoder forward but UniXcoder body is shared, so memory
cost is ~1.5× TRACO. Estimated +60% over TRACO single-session.

---

## exp118 — **PRISM** — Designed geometry + spectral filtering + sibling repulsion 🟢

> Insight cocktail: LIGHTHOUSE (fixed ETF geometry) + RESONANCE (DCT spectral)
> + sibling repulsion + TRACO encoder.

```
# exp118 — PRISM
# NAME       : PRISM (Projection-onto-Refracted-Identity-Spectrum Manifold)
# CLAIM      : The standard learned-W K-class classifier is the wrong choice
#              for AI-code attribution. The right choice is a HAND-DESIGNED
#              geometric target (an Equiangular Tight Frame with sibling-
#              repulsion bonus) combined with SPECTRAL pre-filtering of the
#              encoder output. The encoder learns to REFRACT the input
#              embedding onto the fixed prism: high-frequency content is
#              filtered out (decoding noise), the residual is projected onto
#              the prism, and the closest face wins.
# COMPONENTS :
#   1. UniXcoder encoder → per-token hidden states H ∈ R^{L×d}.
#   2. DCT along position dim → keep first K=16 coefficients.
#   3. Flatten, project to embedding z ∈ R^{256}.
#   4. Fixed prism W = ETF(K_cls, 256) + δ · SiblingRepulse(adj)  (NO grad).
#   5. logits = cos(z, W); classify by argmax.
#   6. Loss: CE(logits, y) + λ · TRACO tree-weighted SupCon (using fixed W as proxies).
#   7. CARGO structural augmentation on input (preserves intent through DCT).
# WOW HOOK   : "We replace the classifier with a PRISM — a hand-designed
#              geometric target that already encodes sibling repulsion. The
#              encoder just learns to refract code onto the prism after
#              filtering out the high-frequency decoder noise."
# FALSIFIER  :
#   (F1) Encoder cos(z, W_y) > cos(z, W_y') for y ≠ y' on test (geometry matched).
#   (F2) DCT-truncated (K=16) Macro-F1 ≈ full-spectrum (K=L) Macro-F1
#        ± 0.005 → spectral hypothesis confirmed.
#   (F3) Replacing fixed W with learnable W loses ≥ 0.01 at 1% slot → design
#        is doing real work.
#   (F4) Composite > TRACO by ≥ +0.005.
```

**Why no one has done this**: ETF-Simplex (exp_n09) used vanilla ETF and
failed at low data; nobody added the spectral filter OR sibling-repulsion
bonus. The combination of "hand-designed geometry + spectral filter +
encoder learns projection" is structurally new for code attribution.

**Compute**: 🟢 — DCT and ETF are both parameter-free. Same as TRACO.

---

## exp119 — **FOREST** — Per-family experts × meta-learning × differentiable tree 🟡

> Insight cocktail: SPECEXPERT (per-family heads) + METATRACO (MAML) +
> TREELEARN (learnable tree) + TRACO.

```
# exp119 — FOREST
# NAME       : FOREST (Family-Of-Recursive-Experts via Self-Trained tree)
# CLAIM      : The "right" classifier for K-class authorship is not one
#              K-way softmax. It is a recursive TREE of specialists:
#              (1) a root family-classifier predicts which of F families
#              the code belongs to; (2) a per-family expert (small head)
#              predicts which of the siblings within that family wrote it.
#              We jointly LEARN the tree structure (TREELEARN ultrametric)
#              while META-LEARNING the per-family experts (MAML outer loop
#              where each task is one family's sibling discrimination).
#              The result is a forest of experts where each expert is
#              meta-tuned for the family it serves.
# COMPONENTS :
#   1. Shared UniXcoder + projector → embedding z ∈ R^256.
#   2. Differentiable ultrametric D_learn (TREELEARN exp106).
#   3. Family-root classifier on z (predicts F ≤ K families inferred from D_learn).
#   4. F per-family specialist heads, each meta-learned via MAML inner loop on
#      few-shot sibling-discrimination episodes within that family.
#   5. Outer loop: meta-gradient update of (encoder, D_learn, family-root, specialists).
#   6. CARGO augmentation in the inner loop.
# WOW HOOK   : "Attribution as descent down a tree the model GREW itself.
#              The forest discovers its own taxonomy through meta-learning,
#              and each leaf-expert is a few-shot specialist for its family's
#              sibling boundary."
# FALSIFIER  :
#   (F1) Learned D_learn must have a non-trivial family structure: cluster
#        membership Rand-index with hand-coded tree > 0.5.
#   (F2) Family-root accuracy on test > 0.85 (must reliably route to the
#        right specialist).
#   (F3) Removing per-family experts (just root) loses ≥ 0.02 at AICD 1%.
#   (F4) Removing meta-learning (vanilla joint training) loses ≥ 0.02 at AICD 1%.
#   (F5) Composite > METATRACO + 0.005 on AICD-T2 (family-rich benchmark).
```

**Why no one has done this**: SPECEXPERT (exp98) tried per-family heads but
used hand-coded family map and joint training. METATRACO uses MAML but on
flat K-way. FOREST is the FIRST to combine (a) learn-the-tree + (b)
per-family experts + (c) meta-learn the experts. Three orthogonal
components, each a paper on its own.

**Compute**: 🟡 — MAML doubles compute; F=4 families × small heads is cheap.
Estimated ~2× TRACO budget.

---

## exp120 — **DECIPHER** — Adversarial co-training + retrieval fixed-point + tree weighting 🟡

> Insight cocktail: SCHISM (adversarial dual encoder) + GRAVITY (fixed-point
> attention) + TRACO + CHORUS.

```
# exp120 — DECIPHER
# NAME       : DECIPHER (Dual-Encoder Contrastive Inference, PHrased-as-fixed-point Estimator-Refined)
# CLAIM      : Sibling discrimination requires WEAKER signal than cross-family
#              discrimination. Standard contrastive treats them uniformly. We
#              train TWO encoders adversarially: encoder A pushes siblings APART
#              (sibling-splitter), encoder B PULLS them together (sibling-merger,
#              the adversary). The classifier reads z_A − z_B. At inference,
#              we iteratively retrieve nearest neighbours in the (z_A − z_B)
#              space and refine the query by fixed-point attention. THE RESIDUAL
#              after retrieval refinement IS the prediction. Both encoders also
#              get gradient-reversal heads against decoding temperature.
# COMPONENTS :
#   1. Two UniXcoder bodies (A, B), each with its own projector.
#   2. Adversarial alternation: 1 step train A with TRACO supcon (siblings push
#      apart 10×), 1 step train B with anti-supcon (siblings pull together).
#   3. Classifier on z_A − z_B (the residual that survives subtraction).
#   4. Temperature gradient-reversal on each of z_A, z_B (CHORUS).
#   5. At inference: FAISS index on (z_A − z_B) training pool. Iterate
#      retrieval-refinement T=5 steps until fixed point.
#   6. CARGO augmentation for both encoders.
# WOW HOOK   : "We train a sibling-merger that we then subtract; what survives
#              is identity. At inference, the query embedding drifts through
#              its own neighbourhood until it lands on its author. Three
#              architectural insights compose into a single decoder."
# FALSIFIER  :
#   (F1) ||z_A − z_B|| on sibling pairs > ||z_A − z_B|| on cross-family
#        pairs by 2× (interpretable residual).
#   (F2) Fixed-point convergence: ||z_T − z_{T-1}|| < 0.05 at T=5 on > 80%
#        of test samples.
#   (F3) Removing the adversarial step (only A, no B) loses ≥ 0.01 at AICD 1%.
#   (F4) Removing the retrieval iteration (T=1) loses ≥ 0.005.
#   (F5) Composite > METATRACO by ≥ +0.005.
```

**Why no one has done this**: SCHISM is novel by itself; GRAVITY is novel by
itself; combining them in a single training+inference loop has not been done.
The DECIPHER architecture is the first to make adversarial co-training serve
a RETRIEVAL-augmented inference, with both modes referencing the same
sibling-aware subtracted representation.

**Compute**: 🟡 — Two encoders + FAISS index + iterative inference. Estimated
~2.5× TRACO budget.

---

## exp121 — **NEUROCHAIN** — Tree-walking reasoning with TRACO-grounded chains 🔴

> Insight cocktail: REASON (LLM chain-of-thought) + tree-structured walk +
> TRACO embeddings as reasoning anchors.

```
# exp121 — NEUROCHAIN
# NAME       : NEUROCHAIN (Neural Chain reasoning over family tree, grounded in TRACO embeddings)
# CLAIM      : The strongest possible attribution model is one that EXPLAINS
#              its prediction as a descent down the family tree. A small
#              reasoning LLM (Qwen3-Reason-1.7B, DeepSeek-R1-Distill-1.5B)
#              walks the tree from root: (1) "this is human or AI?" (family-root),
#              (2) "if AI, which family?" (mid-level node), (3) "which sibling
#              of that family?" (leaf). At each step, the LLM is CONDITIONED on
#              TRACO embedding similarity to the candidate node prototypes,
#              so its reasoning is grounded in our trained encoder's distance
#              estimates. The output is both a label AND a verbalised
#              tree-walk justification.
# COMPONENTS :
#   1. TRACO encoder (frozen after pretraining) → query embedding z.
#   2. Per-node prototype embeddings (mean of training samples).
#   3. Small reasoning LLM (e.g. DeepSeek-R1-Distill-1.5B), prompted with:
#        "Given a code snippet with embedding-similarities [s_1, ..., s_K]
#         to candidate authors, walk the family tree to identify the author.
#         Explain at each step which candidates you rule out."
#   4. Final prediction parsed from the LLM's last step.
#   5. Optional fine-tuning of LLM on (TRACO_sim, true_walk_path) pairs.
# WOW HOOK   : "Attribution as a tree walk. The model verbalises its descent —
#              'I rule out human because indentation style; rule out CodeLlama
#              family because operator spacing; within Qwen family, the
#              variable-name length suggests Nxcode' — and the verbal trace
#              IS the explanation that downstream auditors need."
# FALSIFIER  :
#   (F1) LLM-walk Macro-F1 > TRACO-only Macro-F1 by ≥ +0.01 at AICD 1%
#        (otherwise reasoning is decorative).
#   (F2) LLM walks are not trivial: > 60% of test samples involve > 1
#        elimination step (not just root → leaf in one shot).
#   (F3) Walk-grounded confidence > pure-LLM confidence calibration: ECE on
#        walks vs zero-shot LLM lower by ≥ 0.02.
#   (F4) Removing TRACO grounding (LLM walks without similarity input)
#        loses ≥ 0.05 → grounding is essential.
```

**Why no one has done this**: REASON family (DeepSeek-R1, Qwen3-Reason) is
2025-era; no code-attribution paper grounds an LLM walk in a TRACO-style
encoder distance vector AND uses the family tree as the walk topology.
The combination is fundamentally new and produces an interpretable artefact
(the walk) that downstream auditors care about.

**Compute**: 🔴 — 1.5B-3B LLM inference per test sample. Likely 5× TRACO
test-time. Training can stay cheap if the LLM is frozen.

---

## exp122 — **MIRRORED** — Multi-invariance intersection: reverse, denoise, augment 🟢

> Insight cocktail: MIRROR (reverse-token invariance) + DENOISE-v3 (style-
> preserving conditional denoising) + CARGO augmentation + TRACO supcon.

```
# exp122 — MIRRORED
# NAME       : MIRRORED (Multi-Invariance Residual Representation On Encoder-Decoder)
# CLAIM      : Author identity is defined as what SURVIVES the intersection of
#              multiple structural invariances simultaneously: token-order
#              reversal (surface-permutation-invariant), style-preserving
#              denoising (non-style attributes corrupted), and CARGO structural
#              augmentation (control-flow rewrites). The encoder is trained
#              to make ALL THREE views (reversed, denoised, augmented) plus
#              the original four points collapse to the same cluster per
#              author. Identity = fixed point of multi-invariance.
# COMPONENTS :
#   1. UniXcoder encoder; for each batch, produce FOUR views:
#      (a) original x
#      (b) x_rev = reversed token order (MIRROR)
#      (c) x_dn  = style-preserving denoised (DENOISE-v3 corruption)
#      (d) x_aug = CARGO structural augmentation
#   2. Quadrupled batch [z; z_rev; z_dn; z_aug] (4B rows).
#   3. TRACO tree-weighted SupCon on the 4B batch — positives are all four
#      views of same-author samples; negatives are tree-weighted.
#   4. Auxiliary invariance losses:
#      L_rev   = 1 - cos(z, z_rev)
#      L_dn    = 1 - cos(z, z_dn)
#      L_aug   = 1 - cos(z, z_aug)
#      L_cross = 1 - cos(z_rev, z_dn)   (pairwise invariance across views)
#   5. Total: L_CE + λ_TW · L_TRACO + λ_inv · (L_rev + L_dn + L_aug + L_cross)
# WOW HOOK   : "Identity is the fixed point of multiple invariances. We define
#              an author as the unique embedding that survives reversal,
#              denoising, AND structural rewriting simultaneously. Four
#              views, one identity."
# FALSIFIER  :
#   (F1) All four pairwise cos > 0.85 on test (invariances achieved).
#   (F2) Per-invariance ablation: each of the four loss terms must
#        contribute ≥ 0.003 Macro-F1 at 1% slot.
#   (F3) The invariance intersection ≠ trivial: random pair cos << 0.85.
#   (F4) Composite > TRACO + CARGO single-view by ≥ +0.005.
```

**Why no one has done this**: Multi-view contrastive exists (SimCLR, MoCo,
ContraCode) but always with ≤ 2 view types. MIRRORED is the first to use
FOUR DIFFERENT invariance families simultaneously, and the first to argue
that identity should be defined as the intersection of multiple invariances
rather than learned from a single one.

**Compute**: 🟡 — 4× encoder forward. Halve bs to 64. Estimated ~2× TRACO.

---

## exp123 — **COVENANT** — Adversarial co-evolution: detector vs paraphraser 🔴

> Insight cocktail: Adversarial training + LLM paraphraser + TRACO + CHORUS
> + open-set rejection.

```
# exp123 — COVENANT
# NAME       : COVENANT (CO-evolution VENture for Attribution under Adversarial Style Transfer)
# CLAIM      : The strongest test of attribution is whether it survives
#              adversarial paraphrasing. We co-evolve TWO models in a GAN-like
#              loop: (1) the DETECTOR (our TRACO + CHORUS combo); (2) a
#              PARAPHRASER (a small code LLM, e.g. Qwen2.5-Coder-1.5B with a
#              LoRA) trained to DISGUISE the author. The paraphraser rewrites
#              code to confuse the detector; the detector adapts to see
#              through paraphrases. Final detector is robust to adversarial
#              style transfer — which is the threat model that matters for
#              compliance auditing (a sophisticated user might run their LLM
#              output through another LLM to hide its origin).
# COMPONENTS :
#   1. Detector: TRACO encoder + classifier + CHORUS gradient reversal.
#   2. Paraphraser: frozen Qwen2.5-Coder-1.5B + LoRA, prompted "rewrite this
#      code to preserve function but disguise the author."
#   3. Detector trained on (original, label) AND (paraphrased, label).
#   4. Paraphraser LoRA trained adversarially to maximize detector confusion
#      while preserving execution semantics (verified by AST distance bound).
#   5. Outer loop: alternate detector-step and paraphraser-step.
#   6. Open-set rejection: if max_softmax < τ_def, predict "unknown."
# WOW HOOK   : "We don't just train a detector. We train a detector that
#              has SEEN ITS OWN ADVERSARY. The paraphraser is its predator;
#              the detector evolves to see through it. The final model is
#              calibrated against the threat model that matters."
# FALSIFIER  :
#   (F1) Paraphraser must achieve > 30% reduction in detector accuracy on
#        unadapted detector (paraphrases are genuinely confusing).
#   (F2) Adapted detector vs paraphraser at equilibrium: detector accuracy
#        > 0.50 on AICD-T2 12-class (vs random 0.083).
#   (F3) Open-set rejection: at 5%-coverage cost, error rate < 0.10
#        (the detector knows when to abstain).
#   (F4) Composite > METATRACO on the *paraphrased* test set by ≥ +0.05
#        (adapted detector wins against attack).
```

**Why no one has done this**: GAN-based detection exists for images. No
prior code-attribution paper uses an LLM PARAPHRASER as the adversary and
co-evolves both. The threat-model framing ("auditor vs sophisticated user")
is publishable on its own.

**Compute**: 🔴 — Paraphraser LoRA training is expensive; full COVENANT
likely 5-8× TRACO. The biggest gamble in this list.

---

## exp124 — **AUTHORSPACE** — Continuous author manifold with flow-based density 🔴

> Insight cocktail: FLUX (per-author flows) + LIGHTHOUSE (designed geometry)
> + open-set rejection.

```
# exp124 — AUTHORSPACE
# NAME       : AUTHORSPACE (Author identity as a learnable continuous manifold)
# CLAIM      : Authors are not discrete classes — they are LOCATIONS on a
#              learned continuous manifold. New authors (released next month)
#              should appear as new POINTS, not as a softmax retraining.
#              We learn an author-embedding manifold via a SHARED flow body
#              with per-author parameter heads (low-rank), and attribute by
#              -log p_a(code) for each known a. New authors are added by
#              estimating their location from K labelled examples (no
#              retraining).
# COMPONENTS :
#   1. UniXcoder body → per-token hidden states.
#   2. SHARED normalizing flow body (3-coupling-layer RealNVP).
#   3. Per-author LOW-RANK CORRECTION (rank-8) on top of the shared flow.
#   4. Author-correction VECTORS arranged on a unit hypersphere via ETF
#      (LIGHTHOUSE-style designed geometry) — siblings repulsed extra.
#   5. Test-time: compute log p_a(query) = log p_base(f_shared^-1(query) - δ_a)
#      + |det J| for each candidate a; argmax.
#   6. Open-set: max_a log p_a(q) < threshold → predict "unknown."
#   7. NEW author at deployment: K-shot estimation of δ_{a_new} = mean
#      embedding of K examples projected onto the LIGHTHOUSE ETF.
# WOW HOOK   : "Authors are LOCATIONS, not classes. The K-way softmax dies
#              the moment a new generator appears; AUTHORSPACE absorbs new
#              authors as new points on its learned manifold. Deployment-
#              ready attribution that grows with the field."
# FALSIFIER  :
#   (F1) Held-out-author open-set rejection AUC > 0.85 (genuine new
#        generators are correctly flagged).
#   (F2) K-shot new-author estimation: 8-shot adapted accuracy > 0.50 on
#        held-out generators from known families (without retraining).
#   (F3) Composite > METATRACO on closed-set evaluation by ≥ +0.005
#        (the manifold doesn't lose closed-set performance).
#   (F4) Per-author log-likelihood ratios are well-separated: ROC-AUC
#        between own-author vs other-author > 0.90.
```

**Why no one has done this**: Flow-based attribution exists in NLP
(text-watermark detection) but not for code authorship. Combining a
SHARED flow body + per-author low-rank corrections on a DESIGNED ETF
geometry, with K-shot new-author absorption, is the first
"deployment-ready" attribution framework in the literature.

**Compute**: 🔴 — Flow training is expensive; K-author low-rank heads add
~5% params. Estimated ~3× TRACO.

---

## exp125 — **ECHO+** — Ensemble disagreement × MAML × self-pseudolabeling 🟡

> Insight cocktail: ECHO (ensemble disagreement) + METATRACO (MAML) +
> CARGO + sample-selection oracle.

```
# exp125 — ECHO+
# NAME       : ECHO+ (Ensemble Co-Reinforcing Operators with MAML inner loop)
# CLAIM      : ECHO mines ensemble disagreement; METATRACO meta-learns the
#              representation. We combine them: K=5 TRACO+MAML models train
#              on disjoint cross-validation folds. On each fold's holdout,
#              the OTHER 4 models vote. Disagreement-rich holdout samples
#              get pseudo-labelled (majority vote) and added to the training
#              pool for the NEXT cross-validation round. This is k-fold
#              co-training with meta-learned representations. After 3 rounds,
#              we have a self-distilled ensemble where each model has been
#              trained on data the others labelled — without any oracle.
# COMPONENTS :
#   1. K=5 TRACO+MAML models (each is METATRACO-style MAML).
#   2. CARGO structural augmentation in each inner loop.
#   3. K-fold split of training data; each model trains on K-1 folds.
#   4. On the held-out fold for each model, run the other K-1 models, vote.
#   5. Confidence-weighted pseudo-labels added to the next round's pool.
#   6. Three rounds; final ensemble average for prediction.
# WOW HOOK   : "Self-labelling becomes meta-learning: each model in the
#              ensemble meta-trains on data the others labelled. Disagreement
#              is the labelling oracle; agreement is convergence. We never
#              ask for an external label."
# FALSIFIER  :
#   (F1) Pseudo-label accuracy at round 3 > 0.85 on labelled holdout
#        (the ensemble is good at self-labelling).
#   (F2) Each round's composite > previous round's by ≥ +0.003 (real
#        self-improvement).
#   (F3) Ensemble disagreement entropy decreases monotonically across rounds.
#   (F4) Composite > METATRACO + 0.005 (the ensemble + co-training adds).
```

**Why no one has done this**: ECHO alone is k-fold pseudo-labelling; MAML
alone is meta-learning. ECHO+ COMBINES them — k-fold co-training where
each model is itself a meta-learner — which neither paper proposes.

**Compute**: 🟡 — K=5 models, but each smaller than full METATRACO due to
fold split. Estimated ~3× METATRACO budget (~6× TRACO).

---

## exp126 — **HORIZON** — Tiered confidence cascade with open-set + reasoning escalation 🔴

> Insight cocktail: TRIAGE (cascade) + REASON (LLM escalation) + open-set
> rejection + TRACO.

```
# exp126 — HORIZON
# NAME       : HORIZON (Hierarchical Open-Recurrent Inference, Zonal optimal-Notation)
# CLAIM      : Production attribution is a SERVING problem, not an offline
#              eval problem. We design a three-tier cascade:
#              (1) tiny student (DistillBERT or 6-layer UniXcoder, ~30M params)
#                  answers 80% of queries with high confidence;
#              (2) full TRACO+CHORUS UniXcoder answers 15% (medium confidence);
#              (3) Reasoning LLM (DeepSeek-R1-Distill-1.5B) handles 4% (low
#                  confidence) with chain-of-thought tree walk;
#              (4) open-set rejection on 1% (the model knows "I don't know").
#              Each tier is calibrated; we measure cost-vs-accuracy tradeoff.
# COMPONENTS :
#   1. Tier 1: small student distilled from TRACO via offline distillation.
#   2. Tier 2: TRACO+CHORUS (already exists).
#   3. Tier 3: small reasoning LLM with TRACO-grounded prompts (NEUROCHAIN).
#   4. Routing gate: predictive entropy thresholds.
#   5. Open-set: max softmax < threshold → reject as "unknown generator."
# WOW HOOK   : "Attribution as triage. Most code is easy and the small model
#              suffices; some code is hard and the big model gets called;
#              a rare slice is genuinely ambiguous and the model says 'I don't
#              know.' This is production-ready calibration, not just a number
#              on a leaderboard."
# FALSIFIER  :
#   (F1) Tier-1 alone accuracy > 0.90 on its routed slice (small model
#        is genuinely good on easy samples).
#   (F2) Tier-2 accuracy on its routed slice > Tier-1 accuracy on the
#        same slice by ≥ 0.05 (medium tier earns its compute).
#   (F3) Open-set rejection AUC > 0.90 on held-out generators.
#   (F4) End-to-end cost (weighted by tier traffic) < 30% of Tier-2-only
#        cost, with composite within 0.005 of Tier-2-only.
```

**Why no one has done this**: Tiered model serving exists in industry but
no academic code-attribution paper formalises it. HORIZON is the first to
combine (a) hierarchical cascade, (b) open-set rejection, (c) reasoning
LLM escalation, and (d) cost-aware composite metric for code attribution.

**Compute**: 🔴 — Full system is expensive to train (3 tiers) but cheap to
serve. Estimated ~4× TRACO training cost.

---

## Storytelling spine for the paper (with MEGA frameworks)

If two or three of these MEGA frameworks land, the paper's §3 can pivot
from "TRACO is the hero" to a stronger narrative:

> **"Code attribution is not a single-loss problem. It is a system-design
> problem. We propose three new system architectures — CHORALE
> (multi-scale fusion), PRISM (designed geometry), and FOREST (per-family
> meta-experts) — each of which composes existing wins into a structure
> that the field has not assembled before. Across both benchmarks, the
> best of our three frameworks beats every published baseline at every
> few-shot slot by margins exceeding the saturation band."**

That is an oral talk. The single new primitive ideas (exp107–exp116) are
§5 ablations — components inside the larger frameworks. The MEGA
frameworks are §3 candidates.

---

## Compute prioritisation (revised)

| Priority | Exp | Compute | Why this rank |
|:--|:--|:--|:--|
| **TOP** | **exp118 PRISM** | 🟢 | Cheapest MEGA; combines ETF + DCT + sibling repulsion; if any framework lands at 🟢 cost, it lands here first |
| **TOP** | **exp122 MIRRORED** | 🟡 | Multi-invariance intersection; 4 views but reuses CARGO and DENOISE plumbing |
| 3 | exp117 CHORALE | 🟡 | Three-scale + adversarial; +60% over TRACO |
| 4 | exp119 FOREST | 🟡 | Meta-learning forest; 2× TRACO |
| 5 | exp125 ECHO+ | 🟡 | k-fold meta-co-training; 6× TRACO |
| 6 | exp120 DECIPHER | 🟡 | Adversarial dual + retrieval fixed-point; 2.5× TRACO |
| 7 | exp121 NEUROCHAIN | 🔴 | LLM tree-walk; 5× at inference |
| 8 | exp126 HORIZON | 🔴 | Three-tier serving; 4× training |
| 9 | exp123 COVENANT | 🔴 | Adversarial paraphraser; 5-8× TRACO |
| 10 | exp124 AUTHORSPACE | 🔴 | Flow manifold; 3× TRACO |

**Recommended first batch (5 days)**: exp118 PRISM. Cheapest MEGA, novel
multi-component combo, all 🟢 plumbing reuse, clear falsifier panel. If
it lands above METATRACO composite, the paper rewrites around PRISM as
the §3 hero.

**Recommended second batch (10 days)**: exp122 MIRRORED + exp117 CHORALE.
Both 🟡, both extending TRACO in orthogonal directions; if either lands,
combined ablations write the §5 component table by themselves.

**Best-paper gambit (3 weeks)**: exp123 COVENANT. The adversarial
paraphraser framing is unique and addresses the "what if the user runs
their LLM output through another LLM" attack — a question every paper
review is going to ask.

---

## Final note on novelty

Every framework above either:

1. **Combines components that have NEVER been combined**, even if each
   piece exists elsewhere (the synthesis is novel).
2. **Introduces an entirely new computational primitive** (designed
   prism, multi-invariance intersection, co-evolution loop).
3. **Reframes the problem** at a level the field has not articulated
   (attribution as triage, attribution as manifold location, attribution
   as paraphraser-detector co-evolution).

A reviewer reading the abstract for any of CHORALE / PRISM / FOREST /
DECIPHER / NEUROCHAIN / COVENANT / AUTHORSPACE will recognise the name
as a *system*, not a loss tweak. That is the bar for EMNLP 2026 oral.
