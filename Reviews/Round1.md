Tree-Weighted Contrastive Learning for Few-Shot Authorship Attribution of LLM-Generated Code
EMNLP
Submitted: May 19, 2026
Contents
Summary
Strengths
Weaknesses
Detailed Comments
Questions
Overall Assessment
Summary
The paper proposes TRACO, a simple supervised contrastive learning objective for few-shot model-level authorship attribution of LLM-generated code. It injects the known fine-tuning genealogy of generator models directly into the loss by exponentially re-weighting contrastive negatives by their tree distance and enforces invariance to decoding noise via lightweight code augmentations. Across CoDET-M4 (6-way) and AICD-T2 (12-way family attribution), TRACO reports new state-of-the-art Macro-F1 under strict class-stratified few-shot settings, with especially large gains at 1%, and presents a thorough bake-off and ablations indicating that the tree-weighting is the main driver of improvements.

Strengths
Technical novelty and innovation
The paper introduces a principled and minimal way to incorporate known hierarchical label structure (generator family tree) into supervised contrastive learning via distance-weighted negatives.
The “decoding-noise” view-augmentation rationale is intuitive for code generation and integrates cleanly with the doubled-batch SupCon setup.
The method adds just one scalar hyperparameter (γ) and otherwise stays close to standard SupCon + CE, which is elegant and easy to adopt.
Experimental rigor and validation
Evaluations on two public code-authorship benchmarks under few-shot regimes with class-stratified sampling and consistent training protocols.
Component ablations isolating the contributions of tree-weighting and view augmentation; analysis of sibling-pair error reductions aligns with the method’s stated goal.
A large “bake-off” across 20 method variants, including multiple negative results, demonstrating where architectural complexity fails to help under the studied conditions.
Clear reporting of validation-test gaps, compute budgets, and some seed variance statistics; release plans for code and result artifacts.
Clarity of presentation
The paper is well-structured and readable, with strong motivation grounded in an error analysis (sibling confusions dominate).
Method specification is concise; loss and training pseudocode are easy to reproduce.
Limitations are explicitly discussed, including the 20% “saturation regime” and source-domain confounding.
Significance of contributions
Addresses a practically important bottleneck: distinguishing closely related generator models in low-label regimes.
Offers a broadly applicable recipe (hierarchy-aware SupCon) that is not domain-specific to code and could transfer to other hierarchical-label attribution problems.
The empirical takeaway—that hierarchical side-information already available in release notes is the right inductive prior—has methodological value for the community.
Weaknesses
Technical limitations or concerns
Sensitivity to errors in the family tree is unmeasured; real deployments often involve incomplete/ambiguous genealogy (closed-source models), and small tree mistakes could harm performance.
The choice to assign special-case distances (e.g., HUMAN vs AI = 3, other cross-family pairs = 4) is somewhat ad hoc and may bias optimization; there is no sensitivity or justification for these particular defaults.
The decoding-noise augmentations (token dropout, identifier renaming, whitespace jitter, comment strip) may not faithfully emulate temperature-induced sampling variation and could occasionally alter semantics; no formal semantic validation or compilation checks are reported.
Experimental gaps or methodological issues
No open-set or unseen-model evaluation (e.g., new generator families or held-out generators within families), which is central for real-world attribution as models evolve.
Out-of-domain robustness: while a source-domain breakdown is provided post hoc, there is no held-out-domain evaluation protocol (train on Codeforces, test on GitHub, etc.) to systematically quantify domain shift.
Baseline coverage is strong in number but mixed in strength: comparisons lack some high-performing modern encoders (e.g., ModernBERT) under identical protocols; hierarchical cross-entropy and other tree losses are mentioned but only partly documented in appendices.
Reported AICD-T2 results show unusually large Macro vs. Weighted F1 discrepancies; this raises questions about class imbalance handling and whether Macro-F1 improvements translate into uniform per-class gains.
Clarity or presentation issues
Some reported negative baselines (e.g., hyperbolic prototypes, SetFit variants) are described briefly; more detail on tuning and fairness controls would help readers interpret “collapse.”
The paper mixes strong rhetorical claims (“only ingredient that matters”) with nuanced results (e.g., diminishing gains at 20%), which could be toned down.
Missing related work or comparisons
Limited discussion of recent work on model-level attribution in code/text with hierarchical or geometry-aware methods and multimodal cues; while hyperbolic approaches are tested here, comparisons to stronger, recent baselines in the same spirit (e.g., modern encoders or multi-view fusion) are thin.
No direct comparison to open-set attribution or instance-retrieval style paradigms from adjacent domains (e.g., image attribution) that might be adapted for code.
Detailed Comments
Technical soundness evaluation
The loss formulation is straightforward and technically sound: tree-distance–weighted SupCon combined with standard CE on the original view ensures cross-family separability is not neglected while emphasizing sibling separations in the contrastive objective.
The weighting schedule W_ij is sensible, but the special handling of “human-vs-AI” and “cross-family” distances introduces inductive biases that should be justified or ablated; small perturbations of d_T or γ might strongly influence training in low-data regimes.
The augmentations are plausible for style invariance but may not strictly preserve semantics; at minimum, selective compile checks or execution-based equivalence on a subset would strengthen claims of semantic preservation.
Experimental evaluation assessment
The few-shot class-stratified protocol and consistency across methods are welcome. Reporting Macro-F1 as primary metric is appropriate; showing val–test gaps and seed variance is good practice.
The targeted sibling-confusion analysis convincingly shows improvements occur where intended.
The 20-method bake-off usefully contextualizes the marginal value of different losses at higher data budgets and the claim that the encoder eventually dominates.
Gaps to address: (i) no unseen-generator/family evaluation that would test the method’s promise for fast adaptation to new models; (ii) no systematic domain shift evaluation; (iii) limited hyperparameter sensitivity (γ, τ, λ), and no robustness to tree noise; (iv) limited discussion of compute/data costs vs. baselines beyond a rough GPU-hours summary.
Comparison with related work (using the summaries provided)
Relative to recent detectors and benchmarks (e.g., AICD Bench), the paper focuses on family-aware attribution and reports gains under few-shot conditions; this is complementary to large-scale detector training (e.g., DroidCollection/DroidDetect) that emphasizes robustness to diverse obfuscations and domains.
Compared to hierarchical/geometry-aware attribution (e.g., hyperbolic embeddings, GoCoMA-like approaches), this paper argues that simple Euclidean SupCon with tree re-weighting beats or matches more complex geometry in low-data code attribution, aligning with their negative hyperbolic results.
Recent work on LLM-driven attribution for human-written code (tournament prompting, one-shot Bayesian attribution) targets different settings (human authors, large author pools, zero/few-shot without fine-tuning). TRACO is a supervised encoder-based approach optimized for LLM model attribution—complementary but not directly substitutable.
The presented comparisons are directionally fair, but inclusion of stronger encoders and open-/few-shot baselines relevant to AICD’s Task 2 would better situate the gains.
Discussion of broader impact and significance
Practically, a lightweight modification that exploits public genealogy can be impactful in compliance auditing, dataset provenance, and academic integrity when labels per new model are scarce.
Risks include overconfidence in closed-set attribution, sensitivity to mislabeled or incomplete family trees, and potential misuse in adversarial settings without calibration. The paper acknowledges calibration risk and recommends multi-signal pipelines, which is appropriate.
The observation that above ~20% data the loss becomes less critical is a valuable methodological insight for practitioners allocating effort between data collection and loss engineering.
Questions for Authors
How sensitive is TRACO to noise or misspecification in the family tree? Please report experiments with randomized edge perturbations or distance noise and show performance vs. noise level.
Can you provide a hyperparameter sensitivity study for γ, τ, and λ? In particular, how does γ interact with class count and tree depth, and what ranges are robust across both benchmarks?
The AICD-T2 results exhibit large Macro vs. Weighted F1 gaps. Are classes heavily imbalanced in your splits, and do per-class F1s show improvements across the board? Please include per-class breakdowns.
How often do your augmentations change code semantics in practice? Can you report a small-scale compile/run check (per language) to quantify semantic preservation rates?
You assign HUMAN–AI distance = 3 and cross-family = 4 on CoDET-M4. Why is human “closer” to each AI family than families are to each other? Please justify or ablate these design choices.
How does TRACO perform under open-set conditions (e.g., unseen generators at test time with few labeled examples, or unseen families)? A small-scale leave-one-generator/family-out evaluation would strengthen the deployment relevance.
Could you add stronger encoder baselines (e.g., ModernBERT) under the same protocol, and/or show that TRACO’s relative gains persist with a stronger backbone?
Have you tried a learned distance-bucket weighting (instead of fixed exp(-γ d_T)) or hierarchical cross-entropy as a direct baseline with the same encoder to test whether contrastive vs. cross-entropy differences are material?
Overall Assessment
This paper presents a simple, well-motivated, and empirically effective approach to a timely problem: few-shot model-level attribution of LLM-generated code. The key insight—use the readily available generator family tree to reweight contrastive negatives—translates into consistent gains, especially in the extreme few-shot regime where practitioners most need help. The work is clearly written, includes thoughtful ablations and a broad bake-off, and the sibling-confusion analysis directly validates the method’s design. The main concerns center on external validity: lack of open-set/unseen-generator evaluations, limited robustness studies to tree noise and domain shift, and somewhat ad hoc distance choices (e.g., human vs. AI) without sensitivity analysis. Baseline coverage, while extensive in count, would be stronger with modern encoders and a few alternative hierarchical objectives under identical training protocols. Overall, I view the contribution as a solid, pragmatic advance with a favorable simplicity-to-gain ratio. Addressing the above questions—especially tree-noise sensitivity, open-set generalization, and hyperparameter/ distance-choice ablations—would significantly strengthen the paper. I lean toward acceptance given the clarity, practicality, and consistent improvements in the most challenging data regimes.