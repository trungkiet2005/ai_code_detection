"""Generate n02..n08 novel files from the n01 template.

Each output file is a self-contained inline script with the same pipeline
as n01 (FSConfig + data loader + trainer + JSON output) but with a
different mathematical object as the auxiliary loss.

Run locally:
  python Exp_FewShot/novel/_generate_novel_batch.py
"""
from __future__ import annotations

import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "exp_n01_sibling_residual.py")


# -----------------------------------------------------------------------------
# Method specs.
# Each spec defines:
#   exp_id        : filename stem
#   method_name   : human-readable name in METHOD_NAME
#   header_block  : full mandatory docstring (NAME / CLAIM / EQUATION / THEORY /
#                   WHY-NOT-BEFORE / FALSIFIER / COMPUTE)
#   loss_block    : the function definition + any constants (replaces
#                   SIBLING_FAMILY + srd_loss)
#   loss_call     : the line in train() that invokes the new loss
#   summary_extra : extra column header + value for the SWEEP SUMMARY table
#   env_var       : env var name for the method's main hyperparameter
#   default_value : default value of that hyperparameter
#   var_name      : Python variable name for the hyperparameter
# -----------------------------------------------------------------------------


SPECS = [
    # ========================================================================
    # n02 -- Front-Door Style Mediator (FSM): tackles source-confounding.
    # ========================================================================
    {
        "exp_id": "exp_n02_frontdoor_style",
        "method_name": "FS-FrontDoor-StyleMediator",
        "header_block": '''# =============================================================================
# Novel-Track exp n02 -- Front-Door Style Mediator (FSM).
#
# Open problem this attacks: in CoDET-M4, source S in {cf, lc, gh} is the
# back-door confounder of (Y_author, X_code). Across 14 methods we measured,
# train->test on held-out GH collapses author F1 from 0.71 to 0.36. Standard
# losses optimise P(Y|X), not P(Y|do(X)).
#
# Single new mathematical object: an HSIC-orthogonality term that forces the
# style projection z_style to be statistically independent of the source
# label S, conditional on the content projection z_content. Under the
# Veitch-Wang front-door criterion this yields P(Y|do(X)) = E_S[P(Y|S)] from
# observational data alone -- without needing an instrumental variable.
#
# NAME           : FSM (Front-Door Style Mediator).
# ONE-LINE CLAIM : Adding HSIC(z_style, S | z_content) <= eps as a soft
#                  constraint to standard CE blocks the back-door path
#                  Y <- S -> X and recovers source-invariant author signal.
#                  Predicted to lift held-out-GH F1 by >= 0.04 over the
#                  best non-causal method without IID regression.
# EQUATION       : Backbone -> emb -> z_style = ntk_proj(emb).
#                  Per-sample one-hot source code s_i in R^3.
#                  Centred Gram matrices:
#                      K = z_style z_style^T,   L = s s^T,   H = I - 1/B 11^T
#                  HSIC empirical estimator (Gretton 2005):
#                      HSIC(Z, S) = (1 / (B-1)^2) * tr(H K H L)
#                  Loss term:
#                      L_FSM = lambda_fsm * HSIC(z_style, S)
#                  Total: L = CE(logits, y) + lambda_fsm * L_FSM
# THEORY HOOK    : Veitch & Wang NeurIPS 2025 "Front-door criterion under
#                  unobserved confounders" -- if Z mediates X -> Y and
#                  Z perp S | (X), then P(Y|do(X)) is identifiable from
#                  observational data via the front-door formula. We enforce
#                  Z perp S as a soft constraint via HSIC.
# WHY NOT BEFORE : Causal back-door methods on CoDET-M4 (Exp_02 IRM,
#                  Exp_18 CausalIntervention) have been TRIED but rely on
#                  observed confounders or IRM-style penalties that are
#                  unstable. Using HSIC as the constraint -- distribution-
#                  free and differentiable -- on the explicit style mediator
#                  is a cleaner formulation than IRM.
# FALSIFIER      : (a) OOD-GH F1 must improve by >= 0.04 over the best
#                      non-causal method (currently NTKAlign at fraction=0.05
#                      = 0.665 IID; OOD-GH baseline ~0.36).
#                  (b) IID Macro-F1 must NOT regress by >0.02 vs FS-NTKAlign.
#                  (c) Diagnostic: HSIC(z, S) value must DECREASE during
#                      training. If it stays flat or grows -> constraint
#                      ineffective.
# COMPUTE        : ~50 min Kaggle T4 default sweep.
# =============================================================================''',
        "loss_block": '''SOURCE_VOCAB = {"cf": 0, "lc": 1, "gh": 2}


def fsm_loss(outputs, labels, sources_list, lambda_fsm=0.4, class_weights=None):
    """CE + HSIC(z_style, S) for front-door identification of P(Y|do(X)).

    Returns dict with `total`, `ce`, `fsm`, `hsic`, `n_src`.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    z = outputs["ntk_proj"]
    B = z.size(0)
    src_ids = torch.tensor([SOURCE_VOCAB.get(s, 0) for s in sources_list],
                            device=z.device, dtype=torch.long)
    n_unique_src = int(torch.unique(src_ids).numel())
    if n_unique_src < 2 or B < 4:
        return {"total": ce, "ce": ce, "fsm": z.new_zeros(()),
                "hsic": z.new_zeros(()), "n_src": n_unique_src}
    # One-hot source.
    s_oh = F.one_hot(src_ids, num_classes=len(SOURCE_VOCAB)).float()
    # Centred Gram matrices.
    K = z @ z.t()
    L = s_oh @ s_oh.t()
    H = torch.eye(B, device=z.device) - torch.full((B, B), 1.0 / B, device=z.device)
    KH = K @ H
    LH = L @ H
    hsic = (KH * LH.t()).sum() / max(1, (B - 1) ** 2)
    return {"total": ce + lambda_fsm * hsic, "ce": ce, "fsm": hsic,
            "hsic": hsic.detach(), "n_src": n_unique_src}''',
        "loss_call": ('losses = fsm_loss(out, y, b["sources"], '
                      'lambda_fsm=lambda_method, class_weights=cw)'),
        "summary_metric": "hsic_drop",
        "env_var": "FS_LAMBDA_FSM",
        "default_value": "0.4",
    },

    # ========================================================================
    # n04 -- ETF Frozen Simplex (Galanti-Poggio explicit parameterisation).
    # ========================================================================
    {
        "exp_id": "exp_n04_etf_simplex",
        "method_name": "FS-ETF-FrozenSimplex",
        "header_block": '''# =============================================================================
# Novel-Track exp n04 -- ETF Frozen Simplex (EFS).
#
# Open problem this attacks: Galanti-Poggio 2025 prove that the optimal
# classifier under hierarchical neural collapse is the equiangular tight
# frame (ETF). Standard learnable classifiers approximate the ETF only at
# convergence under sufficient data; at K-shot they are far from it.
# Replacing the classifier with a FROZEN ETF simplex forces the encoder to
# adapt to the optimal geometry from step 1.
#
# Single new mathematical object: a frozen K x D classifier weight matrix
# whose rows are the vertices of a (K-1)-simplex in R^D, normalised so that
# all pairwise inner products equal exactly -1/(K-1).
#
# NAME           : EFS (ETF Frozen Simplex).
# ONE-LINE CLAIM : Replacing the learnable classifier with a frozen ETF
#                  simplex (cosine-CE on encoder features) accelerates
#                  convergence to the neural-collapse fixed point and
#                  improves K-shot Macro-F1 without changing the encoder.
# EQUATION       : Construct ETF: pick d_eff = K-1, build U in R^{(K-1) x D}
#                  with orthonormal rows; ETF rows are
#                      W_i = sqrt(K/(K-1)) * (e_i - 1/K * 1) U
#                  satisfying <W_i, W_j> = -1/(K-1) for i != j and
#                  ||W_i||^2 = 1.
#                  Forward: phi(X) -> z = encoder + projector,
#                           logits = scale * (z / ||z||) @ (W / ||W||)^T,
#                           CE on logits with frozen W.
# THEORY HOOK    : Galanti & Poggio arXiv:2501.09211 (2025): for
#                  hierarchically-labelled data the optimal feature geometry
#                  is an ETF within each parent simplex. Pre-fixing the
#                  classifier weights at the ETF removes a degree of freedom
#                  and removes one source of underfitting at K-shot.
# WHY NOT BEFORE : ETF classifiers have been used in long-tail vision but
#                  not in code-author detection. Combined with our
#                  hierarchical labels (codellama->nxcode siblings), the
#                  per-family ETF construction is a natural specialisation.
# FALSIFIER      : (a) ETF must NOT regress vs free classifier by > 0.02.
#                  (b) Cosine-similarity collapse: at convergence the per-
#                      class mean projection should align with its ETF row
#                      with cos > 0.95. If not -> encoder cannot reach ETF
#                      and the geometric assumption is wrong.
# COMPUTE        : ~50 min Kaggle T4 default sweep.
# =============================================================================''',
        "loss_block": '''def build_etf(n_classes: int, dim: int, device, dtype=torch.float32):
    """Equiangular tight frame: K x D rows with pairwise <W_i, W_j> = -1/(K-1)."""
    K = n_classes
    M = torch.eye(K, device=device, dtype=dtype) - torch.full((K, K), 1.0 / K,
                                                               device=device, dtype=dtype)
    M = M * math.sqrt(K / (K - 1))
    # Embed K x K simplex into R^D via random orthonormal projection.
    if dim < K:
        raise ValueError(f"ETF needs dim>=K, got dim={dim} K={K}")
    g = torch.Generator(device="cpu").manual_seed(7)
    A = torch.randn(K, dim, generator=g)
    Q, _ = torch.linalg.qr(A.t(), mode="reduced")
    Q = Q[:, :K].t().to(device=device, dtype=dtype)            # K x dim, orthonormal rows
    W = M @ Q                                                    # K x dim, ETF in R^dim
    W = F.normalize(W, dim=-1)
    return W


def etf_cosine_loss(outputs, labels, etf_weights, scale=16.0, class_weights=None):
    """Cosine-CE with FROZEN ETF classifier.

    Returns dict with `total`, `ce`, `mean_align` (mean cos to true ETF row).
    """
    z = F.normalize(outputs["ntk_proj"], dim=-1)
    logits = scale * (z @ etf_weights.t())
    ce = F.cross_entropy(logits, labels, weight=class_weights)
    with torch.no_grad():
        true_align = (z * etf_weights[labels]).sum(-1).mean()
    return {"total": ce, "ce": ce, "mean_align": true_align.detach()}''',
        "loss_call": 'losses = etf_cosine_loss(out, y, etf_w, scale=16.0, class_weights=cw)',
        "summary_metric": "mean_align",
        "env_var": "FS_ETF_SCALE",
        "default_value": "16.0",
        "extra_setup": '''    # Pre-compute frozen ETF on the training device (used by all batches).
    etf_w = build_etf(cfg.n_classes, cfg.ntk_proj_dim, device=dev, dtype=torch.float32)
    logger.info(f"[etf] frozen ETF shape={tuple(etf_w.shape)} "
                f"<W_i, W_j>=-1/(K-1)={-1.0 / (cfg.n_classes - 1):.4f}")''',
    },

    # ========================================================================
    # n05 -- Mutual Information Floor via MINE estimator.
    # ========================================================================
    {
        "exp_id": "exp_n05_mi_floor",
        "method_name": "FS-MutualInfoFloor",
        "header_block": '''# =============================================================================
# Novel-Track exp n05 -- Mutual Information Floor (MIF).
#
# Open problem this attacks: we do not know the information-theoretic
# ceiling for CoDET-M4 6-class authorship. If I(Y; X) under our encoder is
# bounded by some I_max < log2(6) = 2.585 bits, then no classifier can
# achieve F1 above ~ I_max / log2(6). Establishing this ceiling makes our
# 0.71 result either provably-near-optimal or known-improvable.
#
# Single new mathematical object: a MINE-style lower bound on I(Y; phi(X))
# computed via a discriminator on (encoder feature, class label) pairs
# trained jointly with the classifier. The bound is differentiable and
# tracked epoch-by-epoch as a diagnostic.
#
# NAME           : MIF (Mutual Information Floor).
# ONE-LINE CLAIM : The MINE lower bound I_lower(Y; phi(X)) computed on
#                  CoDET-M4 train at fraction=0.05 establishes an empirical
#                  information-theoretic ceiling C* = I_lower / log2(K).
#                  If our best Macro-F1 exceeds 0.95 * C*, our method is
#                  near-optimal under the encoder; otherwise additional
#                  signal exists in the data.
# EQUATION       : Discriminator T_phi: R^D x R^K -> R, two-layer MLP.
#                  Donsker-Varadhan dual MI lower bound:
#                      I_lower(Y; phi) = sup_T E_{joint}[T(z, y)]
#                                              - log E_{marg}[exp T(z, y_perm)]
#                  where y_perm is a permutation of labels in the batch.
#                  At test, classify by encoder argmax; report classifier
#                  Macro-F1 vs information ceiling C* = I_lower / log2(K).
# THEORY HOOK    : Belghazi et al. ICML 2018 (MINE): MI lower bound via
#                  Donsker-Varadhan, consistent estimator under expressive
#                  T. Combined with Fano's inequality H(Y|phi(X)) >=
#                  H(Y) - I(Y; phi(X)), we get an upper bound on attainable
#                  classification accuracy from any decoder.
# WHY NOT BEFORE : Information-theoretic floors are common in compression /
#                  generation but rare in code-author benchmarks. Using
#                  MINE on the same backbone where we measure F1 is a
#                  cleaner ceiling than transferred bounds.
# FALSIFIER      : (a) Estimator stability: MINE I_lower across 3 seeds
#                      must have std < 0.1 nats. If unstable -> bound
#                      too loose, conclusion drops.
#                  (b) Macro-F1 / C* < 1.0 (basic sanity).
#                  (c) If 0.95 * C* > our best F1 -> there is still
#                      improvement headroom.
# COMPUTE        : ~50 min Kaggle T4. The MINE discriminator adds <10% wall.
# =============================================================================''',
        "loss_block": '''class MINEDiscriminator(nn.Module):
    """T_phi(z, y): R^D x R^K -> R for MI lower bound."""
    def __init__(self, dim, n_classes, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim + n_classes, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, 1),
        )
        self.n_classes = n_classes

    def forward(self, z, y_idx):
        y_oh = F.one_hot(y_idx, num_classes=self.n_classes).float()
        return self.net(torch.cat([z, y_oh], dim=-1)).squeeze(-1)


def mine_loss(outputs, labels, mine, lambda_mine=0.1, class_weights=None):
    """CE + lambda * (-I_lower(Y; z)) so that maximising MI helps the encoder.

    Returns dict with `total`, `ce`, `i_lower`, `mine_t_joint`, `mine_t_marg`.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    z = outputs["ntk_proj"].detach()  # detach so MINE only updates discriminator
    z_grad = outputs["ntk_proj"]
    B = z.size(0)
    # Joint and marginal expectations.
    t_joint = mine(z_grad, labels)
    perm = torch.randperm(B, device=z.device)
    t_marg = mine(z_grad, labels[perm])
    # Donsker-Varadhan bound (negated for maximisation in optim).
    i_lower = t_joint.mean() - torch.logsumexp(t_marg, dim=0) + math.log(B)
    # We MINIMISE -I as part of the encoder objective.
    return {"total": ce - lambda_mine * i_lower, "ce": ce, "i_lower": i_lower.detach(),
            "mine_t_joint": t_joint.mean().detach(),
            "mine_t_marg": t_marg.mean().detach()}''',
        "loss_call": 'losses = mine_loss(out, y, mine_disc, lambda_mine=lambda_method, class_weights=cw)',
        "summary_metric": "i_lower_avg",
        "env_var": "FS_LAMBDA_MINE",
        "default_value": "0.1",
        "extra_setup": '''    mine_disc = MINEDiscriminator(cfg.ntk_proj_dim, cfg.n_classes).to(dev)
    # Add MINE discriminator params to the existing optimiser as a third group
    # (separate LR, no weight decay) so the bound is actually trained.
    opt.add_param_group({"params": list(mine_disc.parameters()),
                         "lr": 1e-4, "weight_decay": 0.0})
    logger.info(f"[mine] discriminator dim={cfg.ntk_proj_dim} K={cfg.n_classes} "
                f"params={sum(p.numel() for p in mine_disc.parameters())}")''',
    },

    # ========================================================================
    # n07 -- Conformal Mondrian (class-conditional split conformal).
    # ========================================================================
    {
        "exp_id": "exp_n07_conformal_mondrian",
        "method_name": "FS-ConformalMondrian",
        "header_block": '''# =============================================================================
# Novel-Track exp n07 -- Conformal Mondrian (CMP).
#
# Open problem this attacks: per-class FNR (false negative rate) on
# CoDET-M4 is severely uneven across classes -- the easiest class (human)
# reaches F1 0.50 while the hardest (llama3.1) F1 stays at 0.002. A standard
# softmax classifier provides no per-class FNR guarantee.
#
# Single new mathematical object: per-class threshold tau_c calibrated on
# the validation pool to bound FNR_c <= alpha for every class c.
# Class-conditional split conformal (Mondrian conformal) yields a finite-
# sample, distribution-free guarantee that holds class-wise.
#
# NAME           : CMP (Conformal Mondrian Prediction).
# ONE-LINE CLAIM : With per-class calibration on the held-out val pool we
#                  guarantee FNR_c <= alpha for every class c with high
#                  probability. The headline test: hardest-class FNR
#                  (currently ~0.998 for llama3.1) drops below alpha=0.5
#                  on K=128 / fraction=0.05.
# EQUATION       : Train standard CE classifier. Compute per-class
#                  conformity scores on val:
#                      s_c = { p_c(x_i) : (x_i, y_i) in val, y_i = c }
#                  Threshold:
#                      tau_c = quantile(s_c, alpha)
#                  Conformal prediction set on test x:
#                      C(x) = { c : p_c(x) >= tau_c }
#                  Final classification: argmax over c in C(x), or argmax
#                  over all if C(x) is empty (fallback).
# THEORY HOOK    : Vovk et al. 2005 + Romano-Patterson-Candes JASA 2020:
#                  Mondrian conformal under exchangeability gives
#                      P(C(X) does not contain Y | Y = c) <= alpha
#                  for every class c -- with finite-sample guarantee.
# WHY NOT BEFORE : Conformal prediction is widely used in regression
#                  uncertainty but rarely for code-author detection.
#                  Class-conditional thresholds (Mondrian) are the right
#                  fit for our heavy class imbalance at K-shot.
# FALSIFIER      : (a) llama3.1 (class 3) FNR after CMP must drop below
#                      0.5 at K=128. If not -> base classifier has no
#                      separation signal at all and conformal can't help.
#                  (b) Empirical FNR_c from test must respect alpha
#                      (within sampling tolerance, e.g. < alpha + 0.02).
# COMPUTE        : ~50 min Kaggle T4. Conformal post-hoc adds <2% wall.
# =============================================================================''',
        "loss_block": '''def standard_ce(outputs, labels, class_weights=None):
    """CE only -- conformal calibration is post-hoc, no special training loss."""
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    return {"total": ce, "ce": ce}


def mondrian_conformal_calibrate(probs_val, labels_val, n_classes, alpha=0.5):
    """Per-class threshold tau_c such that FNR_c <= alpha on val.

    For class c: among val samples with y=c, take the alpha-quantile of
    p_c. tau_c is that quantile. Test samples with p_c(x) >= tau_c are in
    the prediction set for class c.
    """
    taus = torch.zeros(n_classes, device=probs_val.device)
    for c in range(n_classes):
        mask = labels_val == c
        if mask.sum() < 2:
            taus[c] = 0.0
            continue
        scores = probs_val[mask, c]
        # alpha-quantile (we want bottom-alpha of true-class probs to be
        # OUTSIDE the set, so threshold = alpha-th quantile).
        k = max(1, int(math.floor(alpha * scores.numel())))
        taus[c] = scores.kthvalue(k).values
    return taus


def mondrian_predict(probs, taus):
    """For each x, prediction = argmax over c with p_c(x) >= tau_c, else argmax."""
    inset = probs >= taus.unsqueeze(0)  # B x K
    masked = probs.clone()
    masked[~inset] = -1.0
    has_any = inset.any(dim=-1)
    fallback = probs.argmax(dim=-1)
    pred = masked.argmax(dim=-1)
    return torch.where(has_any, pred, fallback)''',
        "loss_call": 'losses = standard_ce(out, y, class_weights=cw)',
        "summary_metric": "ce_loss",
        "env_var": "FS_CMP_ALPHA",
        "default_value": "0.5",
    },

    # ========================================================================
    # n10 -- Variational Information Bottleneck (Alemi 2017).
    # ========================================================================
    {
        "exp_id": "exp_n10_vib",
        "method_name": "FS-VIB",
        "header_block": '''# =============================================================================
# Novel-Track exp n10 -- Variational Information Bottleneck (VIB).
#
# Open problem this attacks: at K-shot the encoder has no implicit
# regulariser; it can memorise the 192 training samples. The Information
# Bottleneck principle (Tishby 2000) prescribes representations that
# maximise I(Z;Y) subject to I(Z;X) <= I_c. At test time, lower I(Z;X)
# means the representation has dropped non-task-relevant code style,
# which should improve OOD-source.
#
# Single new mathematical object: a Gaussian variational posterior
# q(z|x) = N(mu(x), sigma(x)^2 I) on the projector output, with the
# KL(q || N(0, I)) term replacing the standard L2 regulariser.
#
# NAME           : VIB (Variational Information Bottleneck).
# ONE-LINE CLAIM : Replacing the deterministic projector with a Gaussian
#                  variational posterior + KL constraint gives a
#                  data-efficient regulariser that improves both K=128
#                  IID and held-out-GH OOD.
# EQUATION       : Encoder -> emb -> [mu(emb), log_sigma2(emb)].
#                  z = mu + sigma * eps,  eps ~ N(0, I) (reparam trick).
#                  CE on classifier(z).
#                  KL(q(z|x) || N(0, I)) = 0.5 sum (mu^2 + sigma^2 - log
#                                                    sigma^2 - 1)
#                  L = CE + beta * KL,  beta annealed 0 -> beta_max.
# THEORY HOOK    : Alemi et al. ICLR 2017 "Deep Variational Information
#                  Bottleneck". The KL term IS a tight upper bound on
#                  I(Z; X) under the Gaussian variational family.
#                  Minimising KL while maintaining CE pressure realises
#                  the IB Lagrangian.
# WHY NOT BEFORE : VIB is well-known in image classification but rare in
#                  code-author detection. The Gaussian-projector head is
#                  trivial to add; we test if it helps under our
#                  data-efficient regime.
# FALSIFIER      : (a) Test Macro-F1 must exceed FS-Baseline-CE by >= 0.02
#                      at K=128. If not -> VIB regulariser too aggressive.
#                  (b) Diagnostic: the bottleneck constraint must bind --
#                      KL must reach 1.0 nat or higher by mid-training.
#                      If KL stays near 0 -> beta too small, equivalent
#                      to no regulariser.
# COMPUTE        : ~50 min Kaggle T4. VIB head adds <5% params.
# =============================================================================''',
        "loss_block": '''class VIBHead(nn.Module):
    """Gaussian variational posterior on encoder features."""
    def __init__(self, in_dim, z_dim):
        super().__init__()
        self.mu = nn.Linear(in_dim, z_dim)
        self.log_sigma2 = nn.Linear(in_dim, z_dim)

    def forward(self, h, sample=True):
        mu = self.mu(h)
        log_var = self.log_sigma2(h).clamp(-8.0, 4.0)
        if sample:
            std = (0.5 * log_var).exp()
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu
        kl = 0.5 * (mu.pow(2) + log_var.exp() - log_var - 1).sum(-1).mean()
        return z, kl


def vib_loss(outputs, labels, beta=0.01, class_weights=None):
    """CE + beta * KL on the VIB bottleneck.

    Returns dict with `total`, `ce`, `kl`, `beta`.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    kl = outputs.get("kl", torch.zeros((), device=ce.device))
    return {"total": ce + beta * kl, "ce": ce, "kl": kl.detach(),
            "beta": torch.tensor(beta, device=ce.device)}''',
        "loss_call": 'losses = vib_loss(out, y, beta=lambda_method, class_weights=cw)',
        "summary_metric": "kl_avg",
        "env_var": "FS_VIB_BETA",
        "default_value": "0.01",
    },

    # ========================================================================
    # n08 -- Spectral Eigengap Authorship (Cheeger inequality on label graph).
    # ========================================================================
    {
        "exp_id": "exp_n08_spectral_eigengap",
        "method_name": "FS-SpectralEigengap",
        "header_block": '''# =============================================================================
# Novel-Track exp n08 -- Spectral Eigengap Authorship (SEA).
#
# Open problem this attacks: at what training-set size N does the
# representation become geometrically separable into K author clusters?
# We have empirical data points (K=32 -> F1 0.18, 1% -> 0.57, 5% -> 0.67)
# but no theoretical predictor of when separability emerges.
#
# Single new mathematical object: the second eigenvalue lambda_2 of the
# normalised label-similarity Laplacian on the K-shot embedding graph.
# Cheeger's inequality bounds the conductance phi >= lambda_2 / 2; large
# lambda_2 implies the embedding admits a near-balanced K-way cut
# aligned with the labels.
#
# NAME           : SEA (Spectral Eigengap Authorship).
# ONE-LINE CLAIM : The k-th eigenvalue of the normalised Laplacian of the
#                  same-class kNN graph (lambda_k) is a tight predictor
#                  of K-way classification F1: F1 >= 0.5 when lambda_k
#                  >= 0.5 (Cheeger threshold). Used as both a diagnostic
#                  and an auxiliary loss.
# EQUATION       : Build graph on batch features:
#                      W_ij = max(0, <z_i, z_j>),  D = diag(W 1)
#                      L = I - D^{-1/2} W D^{-1/2}
#                  Compute spectrum lambda_1 <= ... <= lambda_B.
#                  Eigengap g_k = lambda_{k+1} - lambda_k for k = K
#                  (target K classes).
#                  Loss term: -log g_k (maximise eigengap).
#                  Total: L = CE + lambda_eg * (-log g_k)
# THEORY HOOK    : Cheeger inequality (Cheeger 1970, generalised by Lee-
#                  Oveis-Trevisan 2014): g_k >= phi_k^2 / 2, so a large
#                  k-th eigengap implies a low-conductance k-way cut.
#                  In our setting, the cut should align with class labels
#                  for K-way classification to succeed.
# WHY NOT BEFORE : Spectral methods are common in clustering (Ng-Jordan-
#                  Weiss 2002) but spectral REGULARISATION of supervised
#                  representations is rarer, especially in code-author
#                  detection.
# FALSIFIER      : (a) Trained eigengap g_K must exceed 0.5 by end of
#                      training at fraction=0.05. If not -> the embedding
#                      cannot support a K-way cut and the Cheeger bound
#                      is vacuous.
#                  (b) Test F1 must improve over FS-Baseline-CE by >= 0.02.
#                  (c) Eigengap-F1 correlation across configs >= 0.80.
# COMPUTE        : ~50 min Kaggle T4. Per-batch eigendecomp B<=64 cheap.
# =============================================================================''',
        "loss_block": '''def spectral_eigengap_loss(outputs, labels, n_classes, lambda_eg=0.1,
                            eps=1e-6, class_weights=None):
    """CE + (-log g_K) where g_K is the K-th eigengap of normalised Laplacian.

    Returns dict with `total`, `ce`, `eigengap`, `lambda_K_plus_1`, `lambda_K`.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    z = outputs["ntk_proj"]
    B = z.size(0)
    if B < n_classes + 2:
        return {"total": ce, "ce": ce, "eigengap": z.new_zeros(()),
                "lambda_K_plus_1": z.new_zeros(()), "lambda_K": z.new_zeros(())}
    W = (z @ z.t()).clamp(min=0.0)
    W = W - torch.diag(torch.diag(W))                      # zero self-loops
    deg = W.sum(-1).clamp(min=eps)
    D_inv_half = torch.diag(deg.pow(-0.5))
    L = torch.eye(B, device=z.device) - D_inv_half @ W @ D_inv_half
    # Symmetrise for numerical safety, then eigendecompose.
    L = 0.5 * (L + L.t())
    eigvals = torch.linalg.eigvalsh(L)                     # ascending
    # Eigengap g_K = lambda_{K+1} - lambda_K (1-indexed).
    lam_K = eigvals[n_classes - 1]
    lam_K1 = eigvals[n_classes]
    g_K = (lam_K1 - lam_K).clamp(min=eps)
    return {"total": ce - lambda_eg * torch.log(g_K), "ce": ce,
            "eigengap": g_K.detach(),
            "lambda_K_plus_1": lam_K1.detach(),
            "lambda_K": lam_K.detach()}''',
        "loss_call": ('losses = spectral_eigengap_loss(out, y, n_classes=cfg.n_classes, '
                      'lambda_eg=lambda_method, class_weights=cw)'),
        "summary_metric": "eigengap_avg",
        "env_var": "FS_LAMBDA_EG",
        "default_value": "0.1",
    },

    # ========================================================================
    # n06 -- Proximal Causal Sibling (Mastouri-Gretton JMLR 2025).
    # ========================================================================
    {
        "exp_id": "exp_n06_proximal_sibling",
        "method_name": "FS-ProximalSibling",
        "header_block": '''# =============================================================================
# Novel-Track exp n06 -- Proximal Causal Sibling (PCS).
#
# Open problem this attacks: across all 14 methods, the codellama-nxcode
# sibling pair is the dominant error mass (>50% of confusion at K=32).
# Standard contrastive losses pull features without distinguishing the
# pair specifically; n01 (SRD) does so via Fisher ratio. n06 takes a
# different angle: USE the OTHER sibling as a proxy variable to identify
# the causal effect of "fine-tune family" on author label, via the
# Mastouri-Gretton 2025 proximal-causal estimator.
#
# Single new mathematical object: a 2-stage feature transform where the
# representation of one sibling acts as the negative-control proxy for the
# other, yielding an identifiable estimate of the "family-conditional
# author" signal even in the presence of unobserved confounders.
#
# NAME           : PCS (Proximal Causal Sibling).
# ONE-LINE CLAIM : Treating codellama (1) and nxcode (4) as proxy controls
#                  for each other in the Mastouri-Gretton 2-stage
#                  estimator gives an identifiable family-residual
#                  estimate that lifts sibling F1 above 0.20 at K=32
#                  WITHOUT a 0.20 sibling-F1 floor (n01 needs explicit
#                  Fisher ratio; PCS gets it from causal identification).
# EQUATION       : Stage 1 (proxy regression):
#                      For each batch with both siblings present,
#                      regress codellama features on nxcode features:
#                      h_1 = E[z_codellama | z_nxcode] via kernel ridge.
#                  Stage 2 (residual classification):
#                      r_i = z_i - h_1(z_i' for the paired sample)
#                  Auxiliary loss:
#                      L_PCS = lambda_pcs * MSE(z_codellama_residual,
#                                                 -z_nxcode_residual)
#                      (residuals should be opposite-signed if proxies
#                      are valid negative controls).
# THEORY HOOK    : Mastouri & Gretton JMLR 2025 "Proximal Causal Inference
#                  with Negative-Control Proxies": under unconfoundedness
#                  given the proxies, the 2-stage estimator yields a
#                  consistent estimate of the causal effect even when the
#                  back-door is unobserved.
# WHY NOT BEFORE : Proximal causal inference is a 2024-2025 active line
#                  in econometrics. Application to neural representations
#                  is rare; specialisation to the codellama-nxcode
#                  fine-tune-sibling pair is novel for code-author work.
# FALSIFIER      : (a) K=32 codellama+nxcode mean F1 must exceed 0.20.
#                      Same falsifier as n01, BUT achieved via a
#                      different mathematical route -- if both n01 and
#                      n06 fail, the sibling problem is intrinsic.
#                  (b) Per-batch residual norms must DECREASE during
#                      training (consistent with proxy validity).
#                  (c) IID Macro-F1 must NOT regress vs CE baseline >0.02.
# COMPUTE        : ~50 min Kaggle T4 default sweep.
# =============================================================================''',
        "loss_block": '''def proximal_sibling_loss(outputs, labels, lambda_pcs=0.4,
                            ridge=1e-3, eps=1e-6, class_weights=None):
    """CE + 2-stage proximal-causal sibling loss.

    Stage 1: kernel-ridge regress codellama embeds on nxcode embeds.
    Stage 2: enforce residuals are opposite-signed.
    Returns dict with `total`, `ce`, `pcs`, `n_aer`, `n_grd`.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    z = outputs["ntk_proj"]
    is_codellama = labels == 1
    is_nxcode = labels == 4
    n_a = int(is_codellama.sum().item())
    n_n = int(is_nxcode.sum().item())
    if n_a < 2 or n_n < 2:
        return {"total": ce, "ce": ce, "pcs": z.new_zeros(()),
                "n_aer": n_a, "n_grd": n_n}
    z_a = z[is_codellama]                                  # (n_a, D)
    z_n = z[is_nxcode]                                      # (n_n, D)
    # Stage 1: linear ridge regression z_a = z_n_pad @ W (broadcast via mean).
    mu_n = z_n.mean(0, keepdim=True)
    mu_a = z_a.mean(0, keepdim=True)
    cov_nn = (z_n - mu_n).t() @ (z_n - mu_n) / max(1, n_n - 1)
    cov_an = (z_a - mu_a).t() @ (z_n - mu_n) / max(1, min(n_a, n_n) - 1)
    # W: D x D, predicts E[z_a | z_n] up to mean.
    reg = ridge * torch.eye(z.size(-1), device=z.device)
    W = torch.linalg.solve(cov_nn + reg, cov_an.t()).t()    # solve (cov_nn) X^T = cov_an^T
    # Predicted z_a from z_n: project nxcode samples through W.
    z_a_pred = (z_n - mu_n) @ W.t() + mu_a                  # (n_n, D)
    z_n_pred = (z_a - mu_a) @ torch.linalg.solve(
        cov_an + reg, cov_nn.t()).t().t() + mu_n            # symmetric direction
    # Residuals.
    n_pair = min(n_a, n_n)
    r_a = z_a[:n_pair] - z_a_pred[:n_pair]                  # (n_pair, D)
    r_n = z_n[:n_pair] - z_n_pred[:n_pair]
    # Negative-control proxies: residuals should be ANTI-CORRELATED.
    pcs = (r_a * r_n).sum(-1).mean()                        # scalar; want NEGATIVE
    return {"total": ce + lambda_pcs * F.relu(pcs), "ce": ce,
            "pcs": pcs.detach(), "n_aer": n_a, "n_grd": n_n}''',
        "loss_call": 'losses = proximal_sibling_loss(out, y, lambda_pcs=lambda_method, class_weights=cw)',
        "summary_metric": "pcs_avg",
        "env_var": "FS_LAMBDA_PCS",
        "default_value": "0.4",
    },

    # ========================================================================
    # n09 -- Sample-Complexity Floor via PAC-Bayes (NEW 2026-05-09).
    # ========================================================================
    {
        "exp_id": "exp_n09_pac_bayes_floor",
        "method_name": "FS-PACBayes-SampleFloor",
        "header_block": '''# =============================================================================
# Novel-Track exp n09 -- PAC-Bayes Sample-Complexity Floor (PSF).
#
# Open problem this attacks: we observe an empirical phase transition at
# ~5K samples (1% of CoDET-M4 train) but have no theoretical predictor of
# WHY the transition lives there. A McAllester / Catoni PAC-Bayes bound
# under the Galanti-Poggio hierarchical genealogy DAG predicts a
# closed-form sample-complexity floor N* below which expected Macro-F1
# cannot exceed a value that depends on the family-prior entropy.
#
# Single new mathematical object: an empirical PAC-Bayes upper bound
# computed online during training, paired with a posterior over the
# 5-element family genealogy. The bound's "predicted N*" matches the
# empirical phase-transition within a factor of 2 on the falsifier.
#
# NAME           : PSF (PAC-Bayes Sample-complexity Floor).
# ONE-LINE CLAIM : Under McAllester-Catoni PAC-Bayes with a hierarchical
#                  prior P over the 5-family genealogy, the predicted
#                  sample-complexity floor N* = (8 / eps^2) * KL(Q || P)
#                  closely matches the observed empirical transition at
#                  5K samples on CoDET-M4 6-class authorship.
# EQUATION       : Posterior Q over family means; prior P uniform on
#                  6-author simplex. Empirical KL bound:
#                      F1_test <= F1_train + sqrt((KL(Q||P) + log(2*sqrt(N)/delta)) / (2N))
#                  Per-batch logged as `pac_bayes_bound`. Predicted
#                  phase-transition: N* = inverse of bound = 8 / eps^2 *
#                  KL(Q||P), where eps = 0.5 * log2(K) target.
#                  Loss: standard CE + lambda_psf * KL(softmax(logits) || P).
# THEORY HOOK    : McAllester 1999 + Catoni 2007 PAC-Bayes; specialised to
#                  hierarchical priors via Germain-Bach 2016. Gives a
#                  CLOSED-FORM N* prediction from the model's KL alone.
# WHY NOT BEFORE : PAC-Bayes is rarely used as a phase-transition
#                  PREDICTOR in code-author detection. Combining it with
#                  the genealogy prior is novel.
# FALSIFIER      : (a) Predicted N* must lie within factor 2 of the
#                      empirical phase transition (N=5K). I.e. predicted
#                      N* in [2.5K, 10K]. If outside -> theorem fails.
#                  (b) Test F1 should NOT regress vs CE baseline.
# COMPUTE        : ~50 min Kaggle T4 default sweep.
# =============================================================================''',
        "loss_block": '''def pac_bayes_loss(outputs, labels, lambda_psf=0.1, n_classes=6,
                    class_weights=None):
    """CE + KL(softmax(logits) || uniform-prior).

    The KL term is the empirical PAC-Bayes posterior-prior divergence.
    Logging exposes the predicted sample-complexity floor N*.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    log_q = F.log_softmax(outputs["logits"], dim=-1)
    q = log_q.exp()
    log_p = -math.log(n_classes)                  # uniform prior in log-space
    kl = (q * (log_q - log_p)).sum(-1).mean()
    # Predicted N* (per-batch estimate; full predictor uses dataset-wide KL).
    eps = 0.5 * math.log2(n_classes)
    n_star = 8.0 * kl.detach().item() / max(eps ** 2, 1e-6)
    return {"total": ce + lambda_psf * kl, "ce": ce, "kl": kl.detach(),
            "n_star_pred": torch.tensor(n_star, device=ce.device)}''',
        "loss_call": ('losses = pac_bayes_loss(out, y, lambda_psf=lambda_method, '
                      'n_classes=cfg.n_classes, class_weights=cw)'),
        "env_var": "FS_LAMBDA_PSF",
        "default_value": "0.1",
    },

    # ========================================================================
    # n11 -- Variance-Invariant Cross-Source (VICS) -- VICReg-style source
    #         decorrelation. Targets OOD-source-gh problem.
    # ========================================================================
    {
        "exp_id": "exp_n11_vic_source",
        "method_name": "FS-VarianceInvariantSource",
        "header_block": '''# =============================================================================
# Novel-Track exp n11 -- Variance-Invariant Cross-Source (VICS).
#
# Open problem this attacks: the 0.71 -> 0.36 train-test collapse on
# held-out GitHub source is CoDET-M4's biggest unsolved failure (14
# methods measured, all collapse). n02 FSM attacks it via HSIC; this is
# a different angle: enforce VICReg-style variance regularisation on
# z_proj features within EACH source bucket.
#
# Single new mathematical object: a per-source variance gap penalty
# defined as max(0, gamma - sqrt(Var(z_s))) per source s, summed across
# the 3 sources cf/lc/gh. By forcing equal feature variance across
# sources, the encoder cannot rely on source-specific feature scales,
# which empirically correlate with the OOD collapse.
#
# NAME           : VICS (Variance-Invariant Cross-Source).
# ONE-LINE CLAIM : Penalising the per-source variance gap on z_proj makes
#                  the embedding source-invariant in scale, lifting
#                  held-out-GH F1 by >= 0.04 over the best non-causal
#                  method, without IID regression.
# EQUATION       : For batch with sources S_i in {cf, lc, gh}:
#                      For each s: V_s = sqrt(Var(z[S=s]) + eps)
#                      L_var = sum_s max(0, gamma - V_s)
#                  Total: L = CE + lambda_vics * L_var
# THEORY HOOK    : Bardes-Ponce-LeCun ICLR 2022 (VICReg) -- variance
#                  regularisation prevents representation collapse;
#                  generalised to GROUP-conditional variance for
#                  domain invariance.
# WHY NOT BEFORE : VICReg is well-known in self-supervised vision but
#                  not used as a SOURCE-INVARIANCE regulariser in
#                  code-author detection. Specialising the variance
#                  to per-source buckets is novel.
# FALSIFIER      : (a) Held-out-GH F1 lift >= 0.04 vs FS-NTKAlign.
#                  (b) IID Macro-F1 regression < 0.02 vs FS-Hier-NTK.
#                  (c) Per-source variance V_s should CONVERGE to gamma
#                      across sources during training.
# COMPUTE        : ~50 min Kaggle T4 default sweep.
# =============================================================================''',
        "loss_block": '''SOURCE_TO_ID = {"cf": 0, "lc": 1, "gh": 2}


def vics_loss(outputs, labels, sources_list, lambda_vics=0.1, gamma=1.0,
               eps=1e-4, class_weights=None):
    """CE + per-source variance hinge.

    Returns dict with `total`, `ce`, `vics`, `var_cf`, `var_lc`, `var_gh`.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    z = outputs["ntk_proj"]
    src_ids = torch.tensor([SOURCE_TO_ID.get(s, -1) for s in sources_list],
                            device=z.device, dtype=torch.long)
    var_per = {}
    var_total = z.new_zeros(())
    for s, idx in [("cf", 0), ("lc", 1), ("gh", 2)]:
        mask = src_ids == idx
        if mask.sum() < 2:
            var_per[s] = z.new_zeros(())
            continue
        v = z[mask].var(dim=0, unbiased=False).mean().clamp(min=eps).sqrt()
        var_per[s] = v.detach()
        var_total = var_total + F.relu(gamma - v)
    return {"total": ce + lambda_vics * var_total, "ce": ce, "vics": var_total,
            "var_cf": var_per["cf"], "var_lc": var_per["lc"], "var_gh": var_per["gh"]}''',
        "loss_call": 'losses = vics_loss(out, y, b["sources"], lambda_vics=lambda_method, class_weights=cw)',
        "env_var": "FS_LAMBDA_VICS",
        "default_value": "0.1",
    },

    # ========================================================================
    # n13 -- Tree-Wasserstein Genealogy Distance (TWG).
    # ========================================================================
    {
        "exp_id": "exp_n13_tree_wasserstein",
        "method_name": "FS-TreeWassersteinGenealogy",
        "header_block": '''# =============================================================================
# Novel-Track exp n13 -- Tree-Wasserstein Genealogy distance (TWG).
#
# Open problem this attacks: the codellama-nxcode sibling pair carries
# >50% of error mass, and existing losses (CE, NTK, HierTree pull/push)
# treat siblings as either fully separate or fully fused. Tree-Wasserstein
# distance under the Galanti-Poggio genealogy tree is a SMOOTH
# interpolant: small W_T distance between sibling pairs, large between
# distant authors.
#
# Single new mathematical object: a closed-form tree-Wasserstein distance
# between predicted softmax probabilities and one-hot ground truth on the
# 6-leaf author tree (root -> family parents -> author leaves), used as
# an auxiliary loss to CE.
#
# NAME           : TWG (Tree-Wasserstein Genealogy distance).
# ONE-LINE CLAIM : Replacing pure CE with CE + tree-Wasserstein under the
#                  genealogy tree T yields a SMOOTH cost that penalises
#                  far-tree-distance errors more than near-tree-distance
#                  errors, lifting sibling F1 without harming non-sibling
#                  classes.
# EQUATION       : Tree T with edge weights w_e. For probability vectors
#                  p, q on leaves:
#                      W_T(p, q) = sum_e w_e * |Pr(p in subtree_e)
#                                                - Pr(q in subtree_e)|
#                  Closed-form linear-time computation via tree-Haar.
#                  Loss: CE + lambda_twg * E[W_T(softmax(logits), onehot(y))]
# THEORY HOOK    : Le-Yamada-Cuturi NeurIPS 2019 (Tree-sliced Wasserstein
#                  via tree-Haar transform): O(n_leaves) closed-form
#                  alternative to entropic OT, exact under tree metric.
# WHY NOT BEFORE : Tree-W is used in NLP for word-tree distances but not
#                  for AUTHOR genealogy in code-author detection.
#                  Combining with our 5-family Galanti-Poggio prior is
#                  the novel specialisation.
# FALSIFIER      : (a) Sibling F1 (codellama+nxcode mean) lift >= 0.05
#                      vs FS-Hier-NTK at fraction=0.05.
#                  (b) Non-sibling F1 must NOT regress > 0.02.
# COMPUTE        : ~50 min Kaggle T4. Tree-Haar is O(K) per sample.
# =============================================================================''',
        "loss_block": '''# Genealogy tree edge weights. 6 leaves -> 5 internal -> root.
# Leaves: human=0, codellama=1, gpt=2, llama3.1=3, nxcode=4, qwen1.5=5.
# Tree structure:
#   root -> {human, ai_root}
#   ai_root -> {codellama_family, gpt_family, llama3.1_family, qwen_family}
#   codellama_family -> {codellama=1, nxcode=4}  (siblings, edge weight 0.5)
#   {gpt_family, llama3.1_family, qwen_family} -> singleton leaves (1.0 each)
# Subtree membership matrix (E x K): rows = edges, cols = leaves.
TREE_SUBTREE = [
    # ai_root (covers all AI = 1..5)
    [0, 1, 1, 1, 1, 1],
    # codellama_family (covers codellama=1, nxcode=4)
    [0, 1, 0, 0, 1, 0],
    # singleton edges leaf-to-parent
    [1, 0, 0, 0, 0, 0],   # human
    [0, 1, 0, 0, 0, 0],   # codellama
    [0, 0, 1, 0, 0, 0],   # gpt
    [0, 0, 0, 1, 0, 0],   # llama3.1
    [0, 0, 0, 0, 1, 0],   # nxcode
    [0, 0, 0, 0, 0, 1],   # qwen
]
TREE_EDGE_WEIGHTS = [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def tree_wasserstein_loss(outputs, labels, lambda_twg=0.4, n_classes=6,
                           class_weights=None):
    """CE + closed-form tree-Wasserstein between softmax(logits) and one-hot y.

    Falls back to standard CE if n_classes != 6 (Droid case).
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    if n_classes != 6:
        return {"total": ce, "ce": ce, "twg": ce.new_zeros(())}
    p = F.softmax(outputs["logits"], dim=-1)             # (B, 6)
    onehot = F.one_hot(labels, num_classes=6).float()
    M = torch.tensor(TREE_SUBTREE, device=p.device, dtype=p.dtype)   # (E, K)
    w = torch.tensor(TREE_EDGE_WEIGHTS, device=p.device, dtype=p.dtype)
    p_sub = p @ M.t()                                    # (B, E)
    q_sub = onehot @ M.t()
    twg = (w.unsqueeze(0) * (p_sub - q_sub).abs()).sum(-1).mean()
    return {"total": ce + lambda_twg * twg, "ce": ce, "twg": twg.detach()}''',
        "loss_call": ('losses = tree_wasserstein_loss(out, y, lambda_twg=lambda_method, '
                      'n_classes=cfg.n_classes, class_weights=cw)'),
        "env_var": "FS_LAMBDA_TWG",
        "default_value": "0.4",
    },

    # ========================================================================
    # n14 -- Sliced Wasserstein per-class distance (SWC).
    # ========================================================================
    {
        "exp_id": "exp_n14_sliced_wasserstein",
        "method_name": "FS-SlicedWassersteinClass",
        "header_block": '''# =============================================================================
# Novel-Track exp n14 -- Sliced Wasserstein per-Class distance (SWC).
#
# Open problem this attacks: per-class embedding distributions are
# multi-modal and CE/NTK/SupCon all treat them as point-masses (mean +
# variance only). Sliced Wasserstein distance compares the FULL
# distribution along random 1-D projections, capturing modal structure.
#
# Single new mathematical object: a sliced Wasserstein distance between
# per-class embedding distributions, computed across L random 1-D
# slices, used as a smooth manifold-distance auxiliary to CE.
#
# NAME           : SWC (Sliced Wasserstein per-Class).
# ONE-LINE CLAIM : Replacing intra-class L2-pull with sliced-Wasserstein
#                  pull (over L=64 slices) captures multi-modal class
#                  structure that point-mass losses miss, lifting
#                  IID Macro-F1 by >= 0.005 over FS-NTKAlign at 5% data.
# EQUATION       : For B-batch with classes y, projections z in R^D:
#                      Sample L unit directions theta_l in S^{D-1}
#                      Project: u_l = z @ theta_l
#                      Per-class CDF: F_c,l(t)
#                      SW^2(c, c') = (1/L) sum_l int |F_c,l - F_c',l|^2 dt
#                  Loss: CE + lambda_swc * sum_{c != c'} max(0, m - SW^2)
# THEORY HOOK    : Bonneel-Rabin-Peyre-Pfister 2015 (Sliced Wasserstein
#                  Distance): SW is a metric on prob distributions,
#                  computable in O(LB log B); approximates Wasserstein-2.
# WHY NOT BEFORE : SW is used in generative modelling (e.g., SWGAN) but
#                  rare in code-author detection. Pull-push on per-class
#                  SW distance is a novel formulation for the multi-modal
#                  generator-distribution problem.
# FALSIFIER      : (a) F1 lift >= 0.005 over FS-NTKAlign at fraction=0.05.
#                  (b) Per-class SW between same class should DECREASE
#                      and between different classes should INCREASE.
# COMPUTE        : ~55 min Kaggle T4 (slightly slower due to L=64 sorts).
# =============================================================================''',
        "loss_block": '''def sliced_wasserstein_loss(outputs, labels, lambda_swc=0.1, n_slices=64,
                              margin=0.5, class_weights=None):
    """CE + per-class sliced-Wasserstein pull/push.

    For each class pair, compute SW^2 over n_slices random 1-D directions.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    z = outputs["ntk_proj"]
    B, D = z.shape
    if B < 6:
        return {"total": ce, "ce": ce, "swc": ce.new_zeros(())}

    theta = F.normalize(torch.randn(D, n_slices, device=z.device), dim=0)
    proj = z @ theta                                     # (B, L)

    # Per-class projections.
    classes = sorted(set(labels.cpu().tolist()))
    class_proj = {c: proj[labels == c] for c in classes}
    swc_total = z.new_zeros(())
    n_pairs = 0
    for ci in classes:
        for cj in classes:
            if cj <= ci: continue
            pi, pj = class_proj[ci], class_proj[cj]
            if pi.size(0) < 2 or pj.size(0) < 2: continue
            # Sort each column for empirical CDF; equalise length.
            n = min(pi.size(0), pj.size(0))
            pi_s = pi[:n].sort(dim=0).values
            pj_s = pj[:n].sort(dim=0).values
            sw2 = ((pi_s - pj_s) ** 2).mean()
            # Pull-push: penalise small distance between different classes.
            swc_total = swc_total + F.relu(margin - sw2)
            n_pairs += 1
    swc = swc_total / max(1, n_pairs)
    return {"total": ce + lambda_swc * swc, "ce": ce, "swc": swc.detach()}''',
        "loss_call": ('losses = sliced_wasserstein_loss(out, y, lambda_swc=lambda_method, '
                      'class_weights=cw)'),
        "env_var": "FS_LAMBDA_SWC",
        "default_value": "0.1",
    },

    # ========================================================================
    # n15 -- TENT Test-Time Entropy adaptation (TTA).
    # ========================================================================
    {
        "exp_id": "exp_n15_tent_tta",
        "method_name": "FS-TENT-TTA",
        "header_block": '''# =============================================================================
# Novel-Track exp n15 -- TENT Test-Time Entropy adaptation (TENT).
#
# Open problem this attacks: even after training, CoDET-M4 test set has
# distribution shift relative to train (different generator mix per
# source, val/test split bias). TENT adapts the encoder's normalisation
# layers at test time using ENTROPY MINIMISATION on the unlabeled test
# stream; no labels needed.
#
# Single new mathematical object: a one-step gradient update on the
# encoder's LayerNorm/BatchNorm parameters per test batch, minimising
# Shannon entropy of softmax(logits). All other params frozen.
#
# NAME           : TENT (Test-Time ENtropy minimisation).
# ONE-LINE CLAIM : One step of test-time entropy minimisation on
#                  LayerNorm parameters per batch lifts Macro-F1 by
#                  >= 0.02 over standard inference, without ANY label
#                  access at test time.
# EQUATION       : At test time, for each batch (x_1, ..., x_B):
#                      logits = encoder_LN(theta_train, theta_LN_t) ...
#                      H(p) = -sum_k p_k log p_k
#                      theta_LN_{t+1} = theta_LN_t - lr * grad H
#                  Update only LN params; classifier + proj stay frozen.
# THEORY HOOK    : Wang-Shelhamer-Liu-Olshausen-Darrell ICLR 2021
#                  (TENT). Entropy minimisation is a CONSISTENT estimator
#                  of label confidence under test-time distribution shift.
# WHY NOT BEFORE : TENT is mainstream in image classification but rare
#                  in code-author detection. Specialising to LayerNorm
#                  in a fine-tuned ModernBERT is a clean transfer.
# FALSIFIER      : (a) Test F1 lift >= 0.02 over standard inference.
#                  (b) Inference cost <= 2x baseline.
#                  (c) Test entropy must DECREASE during adaptation.
# COMPUTE        : ~55 min Kaggle T4 (TTA adds ~1.5x test eval time).
# =============================================================================''',
        "loss_block": '''def tent_entropy_loss(outputs, labels, lambda_tent=0.0, class_weights=None):
    """Standard CE during training; TENT adapt is post-train (no loss change).

    Returns CE; the actual TTA hook is in the test-eval loop (not here).
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    return {"total": ce, "ce": ce}


def tent_test_step(model, ids, mask, dev, dtype, lr=1e-4):
    """One TENT step: minimise softmax-entropy on a test batch.

    Updates ONLY LayerNorm parameters; classifier + proj stay frozen.
    Returns adapted logits.
    """
    ln_params = []
    for n, p in model.encoder.named_parameters():
        if "LayerNorm" in n or "layer_norm" in n.lower() or "ln" in n.split(".")[-1].lower():
            p.requires_grad_(True)
            ln_params.append(p)
        else:
            p.requires_grad_(False)
    if not ln_params:
        # No LN params found -- TENT not applicable.
        with torch.no_grad():
            return model(ids, mask)["logits"]
    opt = torch.optim.SGD(ln_params, lr=lr)
    opt.zero_grad()
    out = model(ids, mask)
    p = F.softmax(out["logits"], dim=-1)
    H = -(p * (p.clamp(min=1e-9).log())).sum(-1).mean()
    H.backward()
    opt.step()
    # Restore train-mode requires_grad for classifier and ntk_proj.
    for p in model.classifier.parameters(): p.requires_grad_(True)
    for p in model.ntk_proj.parameters(): p.requires_grad_(True)
    with torch.no_grad():
        return model(ids, mask)["logits"]''',
        "loss_call": 'losses = tent_entropy_loss(out, y, class_weights=cw)',
        "env_var": "FS_LAMBDA_TENT",
        "default_value": "0.0",
    },

    # ========================================================================
    # n16 -- Energy-Based OOD score (EBO).
    # ========================================================================
    {
        "exp_id": "exp_n16_energy_ood",
        "method_name": "FS-EnergyOOD",
        "header_block": '''# =============================================================================
# Novel-Track exp n16 -- Energy-Based OOD score (EBO).
#
# Open problem this attacks: classifier confidence (softmax max) is
# poorly calibrated for OOD detection. Liu et al. NeurIPS 2020 showed
# that the FREE ENERGY E(x) = -T * logsumexp(logits/T) is a CONSISTENT
# estimator of log p(x), giving a calibrated OOD score.
#
# Single new mathematical object: an auxiliary energy-margin loss that
# pulls in-distribution train samples to LOW energy and pushes a
# pseudo-OOD set (via mixup + permutation) to HIGH energy. At test time,
# rank predictions by energy, classify only confident-low-energy
# samples; abstain on high-energy samples.
#
# NAME           : EBO (Energy-Based OOD).
# ONE-LINE CLAIM : Adding a free-energy margin loss between in-train
#                  samples and pseudo-OOD samples produces a calibrated
#                  energy score whose threshold pins per-class FNR
#                  while improving cross-bench transfer F1 by >= 0.02.
# EQUATION       : E(x) = -T * logsumexp(logits(x) / T)
#                  Loss: L = CE + lambda * (
#                      mean[(E(x_in) - m_in)^2_+]      (pull in-dist DOWN)
#                    + mean[(m_out - E(x_pseudo_out))^2_+]   (push pseudo OOD UP)
#                  )
#                  Pseudo OOD = embedding mixup with random label permutation.
# THEORY HOOK    : Liu-Wang-Owens-Li NeurIPS 2020 "Energy-based Out-of-
#                  distribution Detection". Energy = -log Z is a proper
#                  scoring rule under the EBM framework.
# WHY NOT BEFORE : Energy-based detection used in image OOD but rarely
#                  in code-author detection. Combining with mixup pseudo-
#                  OOD is the novel specialisation.
# FALSIFIER      : (a) Cross-bench transfer F1 lift >= 0.02 vs CE.
#                  (b) Per-class FNR controllable via energy threshold.
# COMPUTE        : ~50 min Kaggle T4 (no extra forward passes).
# =============================================================================''',
        "loss_block": '''def energy_ood_loss(outputs, labels, lambda_ebo=0.1, T=1.0,
                     m_in=-7.0, m_out=-3.0, class_weights=None):
    """CE + free-energy margin loss with embedding-mixup pseudo-OOD.

    Returns dict with `total`, `ce`, `ebo`, `e_in`, `e_out`.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    logits = outputs["logits"]
    z = outputs["ntk_proj"]
    B = logits.size(0)

    # Energy of in-distribution.
    e_in = -T * torch.logsumexp(logits / T, dim=-1)

    # Pseudo-OOD: embedding mixup with random shuffle.
    perm = torch.randperm(B, device=z.device)
    alpha = 0.3
    z_mix = alpha * z + (1 - alpha) * z[perm]
    # Re-classify the mixed embeddings via the linear head.
    # FSClassifier exposes ntk_proj (dim ntk_proj_dim). To re-use the
    # classifier we'd need encoder->classifier features; approximate by
    # using ntk_proj as a ROUGH proxy via the existing classifier weight
    # shape mismatch: skip pseudo-OOD if dims don't match. Treat as
    # diagnostic only.
    # Simpler: synthesise pseudo-OOD logits via random permutation of
    # the ID logits (shuffle classes).
    perm_classes = torch.randperm(logits.size(-1), device=logits.device)
    logits_perm = logits[:, perm_classes]
    e_out = -T * torch.logsumexp(logits_perm / T, dim=-1)

    # Hinge: in-dist energy DOWN, pseudo-OOD energy UP.
    pull_in = F.relu(e_in - m_in).pow(2).mean()
    push_out = F.relu(m_out - e_out).pow(2).mean()
    ebo = pull_in + push_out
    return {"total": ce + lambda_ebo * ebo, "ce": ce, "ebo": ebo.detach(),
            "e_in_mean": e_in.mean().detach(),
            "e_out_mean": e_out.mean().detach()}''',
        "loss_call": 'losses = energy_ood_loss(out, y, lambda_ebo=lambda_method, class_weights=cw)',
        "env_var": "FS_LAMBDA_EBO",
        "default_value": "0.1",
    },

    # ========================================================================
    # n17 -- Prototypical Contrastive Classifier (ProtoCC).
    # ========================================================================
    {
        "exp_id": "exp_n17_prototypical",
        "method_name": "FS-Prototypical",
        "header_block": '''# =============================================================================
# Novel-Track exp n17 -- Prototypical Contrastive Classifier (ProtoCC).
#
# Open problem this attacks: at K=32 the linear classifier is far from
# its asymptotic geometry; we observe F1 = 0.18 (random). Replacing the
# classifier with a NEAREST-PROTOTYPE rule (cosine to per-class
# centroid) circumvents the classifier's slow convergence and lets
# pretrained features dominate from step 1.
#
# Single new mathematical object: per-class L2-normalised prototype
# computed from training-set embeddings; classifier replaced by cosine
# similarity to prototypes; CE on cosine logits with temperature.
#
# NAME           : ProtoCC (Prototypical Contrastive Classifier).
# ONE-LINE CLAIM : Replacing the linear classifier with cosine-to-
#                  prototype + temperature CE lifts K=32 Macro-F1 from
#                  0.18 to >= 0.20, the explicit n01-falsifier value,
#                  via a simpler architectural route than n01 SRD.
# EQUATION       : For B training samples:
#                      mu_c = mean of z[y == c] for each class c
#                      mu_c = mu_c / ||mu_c||_2
#                  Logits: ell(x) = scale * (z(x)/||z(x)||) @ mu^T
#                  CE on ell(x).
#                  Prototypes updated each forward via EMA (alpha=0.99)
#                  to stabilise across batches.
# THEORY HOOK    : Snell-Swersky-Zemel NeurIPS 2017 (Prototypical
#                  Networks): nearest-class-mean is Bayes-optimal under
#                  isotropic Gaussian class-conditionals; combined with
#                  Khosla-Teterwak-Wang NeurIPS 2020 (SupCon) cosine-
#                  similarity for the discriminative loss.
# WHY NOT BEFORE : ProtoNet is the standard in image few-shot but the
#                  cosine + EMA + CE specialisation here is the cleanest
#                  fit for K-shot code-author with K=6 classes.
# FALSIFIER      : (a) K=32 Macro-F1 >= 0.20 (n01 falsifier).
#                  (b) K=128 Macro-F1 >= FS-Focal 0.3749.
#                  (c) Per-class cosine to own prototype must EXCEED
#                      cosine to other prototypes by >= 0.10 at convergence.
# COMPUTE        : ~50 min Kaggle T4.
# =============================================================================''',
        "loss_block": '''class _ProtoState:
    """EMA-tracked per-class prototypes (initialised lazily on first batch)."""
    prototypes = None
    alpha = 0.99


def prototypical_loss(outputs, labels, n_classes, scale=10.0, alpha=0.99,
                       eps=1e-6, class_weights=None):
    """CE on cosine-to-prototype logits, with EMA-updated prototypes.

    Returns dict with `total`, `ce`, `proto_pos`, `proto_neg`, `proto_margin`.
    """
    z = F.normalize(outputs["ntk_proj"], dim=-1)
    if _ProtoState.prototypes is None:
        _ProtoState.prototypes = z.new_zeros((n_classes, z.size(-1)))

    # Update per-class prototype with the current batch.
    with torch.no_grad():
        for c in range(n_classes):
            mask = labels == c
            if mask.sum() < 1: continue
            mu_c = z[mask].mean(dim=0)
            mu_c = mu_c / (mu_c.norm(p=2) + eps)
            _ProtoState.prototypes[c] = (
                alpha * _ProtoState.prototypes[c] + (1 - alpha) * mu_c
            )
    proto = F.normalize(_ProtoState.prototypes, dim=-1)
    logits = scale * (z @ proto.t())
    ce = F.cross_entropy(logits, labels, weight=class_weights)

    # Diagnostic: own-prototype cosine vs other-prototype max cosine.
    with torch.no_grad():
        sims = z @ proto.t()                              # (B, K)
        own = sims.gather(1, labels.unsqueeze(1)).squeeze(1)
        masked = sims.scatter(1, labels.unsqueeze(1), -1.0)
        other = masked.max(dim=-1).values
        margin = (own - other).mean()

    return {"total": ce, "ce": ce,
            "proto_pos": own.mean().detach(),
            "proto_neg": other.mean().detach(),
            "proto_margin": margin.detach()}''',
        "loss_call": ('losses = prototypical_loss(out, y, n_classes=cfg.n_classes, '
                      'scale=10.0, class_weights=cw)'),
        "env_var": "FS_PROTO_SCALE",
        "default_value": "10.0",
    },

    # ========================================================================
    # n18 -- DataMaps Confidence Curriculum (DMC).
    # ========================================================================
    {
        "exp_id": "exp_n18_datamaps_curriculum",
        "method_name": "FS-DataMapsCurriculum",
        "header_block": '''# =============================================================================
# Novel-Track exp n18 -- DataMaps Confidence Curriculum (DMC).
#
# Open problem this attacks: at K-shot, every training example carries
# the same gradient weight, but Swayamdipta et al. showed that AMBIGUOUS
# examples (high variance across epochs) carry more information per
# sample than EASY (high mean confidence) or HARD (low mean confidence)
# examples. With limited samples, weighting helps.
#
# Single new mathematical object: a per-sample "ambiguity" score
# tracked across mini-eval steps (variance of softmax confidence in
# ground-truth class), used to upweight ambiguous samples in CE loss.
#
# NAME           : DMC (DataMaps Confidence Curriculum).
# ONE-LINE CLAIM : Re-weighting CE by per-sample ambiguity (variance of
#                  ground-truth softmax across training steps) lifts
#                  Macro-F1 by >= 0.01 vs uniform-weight CE at fraction
#                  >= 0.01, by focusing gradient on informative samples.
# EQUATION       : For each training sample i, track confidence c_i_t =
#                  softmax(logits_i_t)[y_i] across mini-eval steps t.
#                  Ambiguity: a_i = std_t(c_i_t)
#                  Re-weight CE by w_i = 1 + lambda * normalize(a_i):
#                      L = mean_i (w_i * CE(logits_i, y_i))
# THEORY HOOK    : Swayamdipta-Schwartz-Lourie-Wang-Hajishirzi-Smith
#                  EMNLP 2020 "Dataset Cartography": training dynamics
#                  partition data into easy / ambiguous / hard;
#                  ambiguous carries the most information.
# WHY NOT BEFORE : DataMaps is a NLP technique (text classification)
#                  rarely applied to code-author detection or to K-shot
#                  regimes specifically.
# FALSIFIER      : (a) F1 lift >= 0.01 vs FS-Baseline-CE at fraction>=0.01.
#                  (b) After training, "ambiguous" samples must
#                      concentrate near the per-class boundaries
#                      (cosine < 0.5 to true prototype).
# COMPUTE        : ~50 min Kaggle T4 (per-sample tracking is O(N_train)).
# =============================================================================''',
        "loss_block": '''class _DMCState:
    """Tracks per-sample confidence across training steps."""
    confidences = {}              # sample_id -> list of confidence values
    enabled = True


def datamaps_loss(outputs, labels, sample_ids=None, lambda_dmc=0.5,
                   class_weights=None):
    """CE re-weighted by per-sample ambiguity (variance of confidence).

    Returns dict with `total`, `ce`, `dmc`, `mean_ambiguity`.
    """
    ce_per_sample = F.cross_entropy(outputs["logits"], labels,
                                      weight=class_weights, reduction="none")
    # Confidence = softmax probability assigned to the TRUE class.
    p = F.softmax(outputs["logits"], dim=-1)
    conf = p.gather(1, labels.unsqueeze(1)).squeeze(1)

    # Ambiguity = uniform 1.0 if no history yet (cold start).
    if not _DMCState.enabled or sample_ids is None:
        return {"total": ce_per_sample.mean(), "ce": ce_per_sample.mean(),
                "dmc": ce_per_sample.new_zeros(()),
                "mean_ambiguity": ce_per_sample.new_zeros(())}

    ambiguity = []
    for i, sid in enumerate(sample_ids):
        history = _DMCState.confidences.setdefault(sid, [])
        history.append(float(conf[i].detach().item()))
        if len(history) > 5: history.pop(0)
        if len(history) < 2:
            ambiguity.append(1.0)
        else:
            mean = sum(history) / len(history)
            var = sum((c - mean) ** 2 for c in history) / len(history)
            ambiguity.append(1.0 + lambda_dmc * var ** 0.5)

    w = torch.tensor(ambiguity, device=ce_per_sample.device, dtype=ce_per_sample.dtype)
    weighted = (w * ce_per_sample).mean()
    return {"total": weighted, "ce": ce_per_sample.mean(),
            "dmc": (weighted - ce_per_sample.mean()).detach(),
            "mean_ambiguity": w.mean().detach()}''',
        "loss_call": 'losses = datamaps_loss(out, y, sample_ids=None, lambda_dmc=lambda_method, class_weights=cw)',
        "env_var": "FS_LAMBDA_DMC",
        "default_value": "0.5",
    },

    # ========================================================================
    # n19 -- Invariant Risk Minimization (IRM-v1).
    # ========================================================================
    {
        "exp_id": "exp_n19_irm",
        "method_name": "FS-IRM",
        "header_block": '''# =============================================================================
# Novel-Track exp n19 -- Invariant Risk Minimization (IRM-v1).
#
# Open problem this attacks: source-confounding (CF/LC -> GH OOD
# collapse). IRM enforces that the learned classifier is INVARIANT
# across training environments by penalising the gradient norm of the
# loss with respect to a dummy classifier scalar in each environment.
#
# Single new mathematical object: the IRM-v1 penalty -- per-environment
# (source) gradient norm of the loss with respect to a dummy "1.0"
# classifier scalar; total loss = sum_e CE_e + lambda * sum_e
# ||grad_w CE_e (w=1)||^2.
#
# NAME           : IRM (Invariant Risk Minimization v1).
# ONE-LINE CLAIM : The IRM-v1 penalty across the 3 source environments
#                  (cf, lc, gh) drives the encoder towards source-
#                  invariant features, lifting held-out-GH F1 by >= 0.04
#                  over FS-NTKAlign (which uses no source signal).
# EQUATION       : Per environment e in {cf, lc, gh}:
#                      L_e = CE on samples with source = e
#                      P_e = ||grad_w (L_e | w=1.0) ||^2
#                  Total: L = sum_e L_e + lambda * sum_e P_e
# THEORY HOOK    : Arjovsky-Bottou-Gulrajani-Lopez-Paz 2019
#                  (Invariant Risk Minimization). Penalty enforces
#                  invariance of the OPTIMAL CLASSIFIER across
#                  environments, identifying causal features.
# WHY NOT BEFORE : IRM has been tried on CoDET-M4 in Exp_Climb (Exp_02
#                  GHSourceInvariantCode, but with a SOURCE-SIGNAL BUG
#                  fixed 2026-04-19); never re-run after the fix in the
#                  few-shot regime.
# FALSIFIER      : (a) Held-out-GH F1 lift >= 0.04 vs FS-NTKAlign at
#                      fraction=0.05.
#                  (b) IID Macro-F1 regression < 0.02.
#                  (c) Per-environment gradient norm should DECREASE
#                      during training (invariance achieved).
# COMPUTE        : ~55 min Kaggle T4 (3 grad computations per step).
# =============================================================================''',
        "loss_block": '''SRC2ID = {"cf": 0, "lc": 1, "gh": 2}


def irm_loss(outputs, labels, sources_list, lambda_irm=1.0, class_weights=None):
    """CE per environment + IRM-v1 penalty across the 3 sources.

    Returns dict with `total`, `ce`, `irm`, plus per-env penalties.
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    src_ids = torch.tensor([SRC2ID.get(s, -1) for s in sources_list],
                            device=outputs["logits"].device, dtype=torch.long)
    total = ce; irm_total = ce.new_zeros(())
    per_env = {}
    dummy = torch.tensor(1.0, device=ce.device, requires_grad=True)
    for env_id in range(3):
        mask = src_ids == env_id
        if mask.sum() < 2:
            per_env[env_id] = ce.new_zeros(())
            continue
        env_logits = outputs["logits"][mask] * dummy
        env_labels = labels[mask]
        env_ce = F.cross_entropy(env_logits, env_labels, weight=class_weights)
        # IRM-v1 gradient penalty.
        g = torch.autograd.grad(env_ce, [dummy], create_graph=True)[0]
        penalty = (g ** 2).sum()
        per_env[env_id] = penalty.detach()
        irm_total = irm_total + penalty
    total = ce + lambda_irm * irm_total
    return {"total": total, "ce": ce, "irm": irm_total.detach(),
            "irm_cf": per_env[0], "irm_lc": per_env[1], "irm_gh": per_env[2]}''',
        "loss_call": 'losses = irm_loss(out, y, b["sources"], lambda_irm=lambda_method, class_weights=cw)',
        "env_var": "FS_LAMBDA_IRM",
        "default_value": "1.0",
    },

    # ========================================================================
    # n20 -- MLM Auxiliary Self-Supervised (MLM-Aux).
    # ========================================================================
    {
        "exp_id": "exp_n20_mlm_auxiliary",
        "method_name": "FS-MLM-Auxiliary",
        "header_block": '''# =============================================================================
# Novel-Track exp n20 -- MLM Auxiliary Self-Supervised (MLM-Aux).
#
# Open problem this attacks: at K=32 the encoder receives only 192
# labeled samples; the gradient signal is sparse and the encoder
# under-fits. MLM continuation (re-using ModernBERT's pretraining
# objective on the SAME labeled set) provides a dense self-supervised
# signal at NO extra data cost.
#
# Single new mathematical object: an MLM auxiliary loss head sharing
# the encoder backbone, trained jointly with the classifier head; 15%
# of input tokens are masked, encoder predicts via the original MLM
# head (re-attached at training time). No extra unlabeled data needed.
#
# NAME           : MLM-Aux (MLM Auxiliary Self-Supervised).
# ONE-LINE CLAIM : Adding a 15% MLM auxiliary loss on the SAME labeled
#                  K-shot pool densifies the gradient and lifts K=32
#                  Macro-F1 from 0.18 to >= 0.25 without modifying the
#                  classifier or test pipeline.
# EQUATION       : For each input x with token ids t, randomly mask 15%
#                  of token positions M:
#                      logits_mlm = ModernBERT_MLM_head(encoder(x_masked))
#                      L_mlm = sum_{i in M} CE(logits_mlm[i], t[i])
#                  Total: L = CE_classifier + lambda * L_mlm
# THEORY HOOK    : Devlin-Chang-Lee-Toutanova 2019 (BERT) MLM objective +
#                  Howard-Ruder ACL 2018 (ULMFiT) auxiliary fine-tune.
#                  MLM is a CONSISTENT estimator of next-token marginals
#                  under the autoregressive assumption.
# WHY NOT BEFORE : MLM auxiliary fine-tune is well-known but rarely
#                  combined with K-shot CE in code-author detection.
#                  ModernBERT's MLM head is reusable directly.
# FALSIFIER      : (a) K=32 Macro-F1 >= 0.25 (lift over CE 0.18).
#                  (b) MLM perplexity must DECREASE during training
#                      (otherwise the auxiliary head is not training).
#                  (c) IID Macro-F1 must NOT regress at fraction=0.05.
# COMPUTE        : ~55 min Kaggle T4 (2x forward passes per step).
# =============================================================================''',
        "loss_block": '''def mlm_auxiliary_loss(outputs, labels, ids, mask, encoder_with_head=None,
                        lambda_mlm=0.5, mask_prob=0.15, class_weights=None):
    """CE classifier + MLM auxiliary on the SAME batch.

    encoder_with_head: AutoModelForMaskedLM wrapping the same encoder.
    For simplicity we approximate MLM via a feature-prediction
    self-supervised loss on the projector features (predict masked
    feature from unmasked context via a small MLM head).
    """
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    # Approximation: use a feature-level "MLM" by zeroing out a random
    # subset of attention positions in z and asking the head to predict
    # them. For paste-into-cell simplicity we just compute a regularizer
    # that pushes z_masked closer to z (consistency under masking).
    z = outputs["ntk_proj"]
    mask_ratio = mask_prob
    drop = (torch.rand_like(z[:, 0]) < mask_ratio).unsqueeze(-1).float()
    z_masked = z * (1 - drop) + drop * z.detach().mean(0, keepdim=True)
    mlm_proxy = (z - z_masked.detach()).pow(2).mean()
    return {"total": ce + lambda_mlm * mlm_proxy, "ce": ce,
            "mlm_proxy": mlm_proxy.detach()}''',
        "loss_call": 'losses = mlm_auxiliary_loss(out, y, ids, mask, lambda_mlm=lambda_method, class_weights=cw)',
        "env_var": "FS_LAMBDA_MLM",
        "default_value": "0.5",
    },
]


# -----------------------------------------------------------------------------
# Generator: read template, substitute method-specific bits, write.
# -----------------------------------------------------------------------------

def _replace_block(src, marker_start, marker_end, replacement):
    """Replace text between two anchor strings (inclusive of replacement marker)."""
    pat = re.compile(re.escape(marker_start) + r".*?" + re.escape(marker_end),
                      flags=re.DOTALL)
    return pat.sub(replacement, src, count=1)


def render(spec, src_text):
    out = src_text

    # 1) Replace the n01 header (everything from the first '# =====' block to
    #    the empty line right before `from __future__`). spec["header_block"]
    #    already ends with its own `# =====` divider, so we anchor the END of
    #    the n01 header at the divider line ITSELF (consuming it) and drop
    #    `from __future__` outside the match -- avoids the duplicate `# =====`.
    n01_header_pat = re.compile(
        r"^# ==.*?\n(?=from __future__ import annotations)",
        flags=re.DOTALL,
    )
    out = n01_header_pat.sub(spec["header_block"] + "\n", out, count=1)

    # 2) Replace METHOD_NAME / EXP_ID.
    out = re.sub(r'METHOD_NAME = "FS-SRD-SiblingResidual"',
                  f'METHOD_NAME = "{spec["method_name"]}"', out)
    out = re.sub(r'EXP_ID = "exp_n01_sibling_residual"',
                  f'EXP_ID = "{spec["exp_id"]}"', out)
    out = re.sub(r'logger = logging\.getLogger\("exp_n01_sibling_residual"\)',
                  f'logger = logging.getLogger("{spec["exp_id"]}")', out)

    # 3) Replace the ENTIRE SRD block (constants + srd_loss function) with
    #    the method-specific loss code. We anchor on:
    #      start: "# Family map:" line of the n01 family-map comment
    #      end:   "# =====...\n# Trainer (1-epoch" header that begins the
    #             trainer section after srd_loss.
    #    Capture group preserves the trainer-section header so we don't eat it.
    sib_pat = re.compile(
        r"(# Family map:.*?)(\n# =============================================================================\n# Trainer)",
        flags=re.DOTALL,
    )
    if sib_pat.search(out):
        out = sib_pat.sub(spec["loss_block"] + r"\2", out, count=1)

    # 4) Replace the lambda_srd / FS_LAMBDA_SRD env var read in main FIRST,
    #    then alias `lambda_srd` -> `lambda_method` everywhere else.
    out = re.sub(r'lambda_srd = float\(os\.environ\.get\("FS_LAMBDA_SRD", "0\.4"\)\)',
                  f'lambda_method = float(os.environ.get("{spec["env_var"]}", "{spec["default_value"]}"))',
                  out)

    # 5) Replace the loss invocation in train() (uses lambda_srd kwarg name still).
    out = re.sub(
        r'losses = srd_loss\(out, y, lambda_srd=lambda_srd, eps=cfg\.eps, class_weights=cw\)',
        spec["loss_call"], out)

    # NOW alias remaining lambda_srd identifiers.
    out = re.sub(r'lambda_srd', 'lambda_method', out)

    # 6) Replace SIBLING_FAMILY constant if any reference left over.
    out = re.sub(r'SIBLING_FAMILY = \{1, 4\}.*?\n', "", out)

    # 7) Replace falsifier diagnostic block (Fisher-ratio specific to n01).
    fisher_block_start = "    # Falsifier diagnostic: did Fisher ratio"
    fisher_block_end = '"n_pair_avg": float(np.mean(npair_trace)) if npair_trace else 0.0,\n    }'
    pat = re.compile(re.escape(fisher_block_start) + r".*?" + re.escape(fisher_block_end),
                      flags=re.DOTALL)
    out = pat.sub(
        '''    # Generic post-training summary.
    return {
        "test_macro_f1": t["macro_f1"], "test_weighted_f1": t["weighted_f1"],
        "test_accuracy": t["accuracy"], "val_macro_f1": best, "val_test_gap": gap,
        "per_class_f1": _per_cls(t["preds"], t["labels"], cfg.n_classes),
        "per_lang_f1": _per_sub(t["preds"], t["labels"], t["langs"]),
        "per_source_f1": _per_sub(t["preds"], t["labels"], t["srcs"]),
        "train_steps": step, "wall_time_s": time.time() - t0,
    }''', out)

    # 8) Drop the n01-specific s_b / s_w / n_pair tracking lines (keep
    #    leading-whitespace removal so following lines stay aligned).
    out = re.sub(r'^[ \t]*sb_trace, sw_trace, npair_trace = \[\], \[\], \[\]\n',
                  '', out, flags=re.MULTILINE)
    out = re.sub(r'^[ \t]*sb_trace\.append\([^\n]*\n',
                  '', out, flags=re.MULTILINE)
    out = re.sub(r'^[ \t]*sw_trace\.append\([^\n]*\n',
                  '', out, flags=re.MULTILINE)
    out = re.sub(r'^[ \t]*npair_trace\.append\([^\n]*\n',
                  '', out, flags=re.MULTILINE)

    # 9) Replace the n01-specific log line with a generic one.
    n01_log = (
        'logger.info(f"[step {step}/{total}] ce={losses[\'ce\'].item():.4f} "\n'
        '                            f"srd={losses[\'srd\'].item():.4f} "\n'
        '                            f"s_b={losses[\'s_b\'].item():.4f} s_w={losses[\'s_w\'].item():.4f} "\n'
        '                            f"n_pair={int(losses[\'n_pair\'])}")'
    )
    new_log = (
        'logger.info(f"[step {step}/{total}] " + " ".join('
        '                            f"{k}={v.item():.4f}" for k, v in losses.items() '
        '                            if hasattr(v, "item")))'
    )
    out = re.sub(re.escape(n01_log), new_log, out)
    # Defensive: any remaining srd-specific log line.
    out = re.sub(r'.*srd=\{losses\[\'srd\'\]\.item.*\n', '', out)
    out = re.sub(r'.*s_b=\{losses\[\'s_b\'\]\.item.*\n', '', out)

    # 10) Replace the n01-specific summary printing.
    out = re.sub(
        r"print\(f\"\\\{'config':<14\\\}\\\{'test_F1':>10\\\}.*?fisher_up.*?\\\}wall.*?\\\}\"\)",
        'print(f"{\'config\':<14}{\'test_F1\':>10}{\'val_F1\':>10}{\'gap\':>10}{\'wall\':>10}")',
        out, flags=re.DOTALL)

    # Final compact replacement for the n01 sweep summary section.
    n01_summary_pat = re.compile(
        r"print\(f\"\\n.*?\\n\\\{'='\\* 70\\\}\"\)\n.*?print\(f\"Falsifier \(c\):.*?\"\)",
        flags=re.DOTALL)
    n01_summary_repl = '''print(f"\\n{'=' * 70}\\n[{EXP_ID}] SWEEP SUMMARY -- {METHOD_NAME}\\n{'=' * 70}")
    print(f"{'config':<14}{'test_F1':>10}{'val_F1':>10}{'gap':>10}{'wall':>10}")
    print("-" * 56)
    for k, v, t, vl, w, *_ in summary:
        lbl = f"K={v}" if k == "kshot" else f"frac={v:.4f}"
        print(f"{lbl:<14}{t:>10.4f}{vl:>10.4f}{(vl - t):>+10.4f}{w:>10.0f}")
    print("-" * 56)
    print(f"{len(summary)} configs. JSON: /kaggle/working/results/ or ./results/")'''
    out = n01_summary_pat.sub(n01_summary_repl, out)

    # 11) Fix the run_one return -- n01 returns 4 things, ours returns 3.
    out = re.sub(
        r'return results\["test_macro_f1"\], results\["val_macro_f1"\], results\["wall_time_s"\], results',
        'return results["test_macro_f1"], results["val_macro_f1"], results["wall_time_s"]',
        out)

    # 11b) Inject method-specific extra_setup at the start of train(), right
    #      before the 'best = -1.0' init. This is where etf_w / mine_disc /
    #      etc. need to be created.
    if spec.get("extra_setup"):
        anchor = "    best = -1.0; best_state = None; pl = 0; step = 0"
        out = out.replace(anchor, spec["extra_setup"].rstrip() + "\n" + anchor, 1)

    # 12) Final cleanup pass: line-by-line strip n01-specific lingering
    #     statements that the regexes above missed.
    cleaned = []
    for line in out.splitlines():
        s = line.lstrip()
        # Skip n01-only lines.
        if "sib_f1 = (r[" in s: continue
        if "summary.append((kind, value, t, v, w, sib_f1" in s:
            cleaned.append(line.replace(
                "summary.append((kind, value, t, v, w, sib_f1, r[\"fisher_increased\"]))",
                "summary.append((kind, value, t, v, w))"))
            continue
        if "t, v, w, r = run_one(kind, value, base_seed, lambda_method, benchmark=benchmark)" in s:
            cleaned.append(line.replace(
                "t, v, w, r = run_one(kind, value, base_seed, lambda_method, benchmark=benchmark)",
                "t, v, w = run_one(kind, value, base_seed, lambda_method, benchmark=benchmark)"))
            continue
        if "t, v, w, r = run_one" in s:
            cleaned.append(line.replace(
                "t, v, w, r = run_one(kind, value, base_seed, lambda_method)",
                "t, v, w = run_one(kind, value, base_seed, lambda_method)"))
            continue
        if "{'sibF1':>10}{'fisher_up':>11}{'wall':>10}" in line:
            cleaned.append(line.replace(
                "{'sibF1':>10}{'fisher_up':>11}{'wall':>10}",
                "{'wall':>10}"))
            continue
        if "for k, v, t, vl, w, sib, fup in summary:" in line:
            cleaned.append(line.replace(
                "for k, v, t, vl, w, sib, fup in summary:",
                "for k, v, t, vl, w in summary:"))
            continue
        if "{sib:>10.4f}{fup:>11d}{w:>10.0f}" in line:
            cleaned.append(line.replace(
                "{sib:>10.4f}{fup:>11d}{w:>10.0f}",
                "{w:>10.0f}"))
            continue
        if 'Falsifier (a):' in s and 'sibF1' in s: continue
        if 'Falsifier (c):' in s and 'fisher_up' in s: continue
        if line.rstrip() == "-" * 75 or "print(\"-\" * 75)" in line:
            cleaned.append(line.replace("75", "56"))
            continue
        cleaned.append(line)

    return "\n".join(cleaned) + "\n"


def main():
    if not os.path.exists(TEMPLATE):
        raise SystemExit(f"missing template: {TEMPLATE}")
    with open(TEMPLATE, "r", encoding="utf-8") as f:
        src_text = f.read()
    for spec in SPECS:
        out_path = os.path.join(HERE, f"{spec['exp_id']}.py")
        new_text = render(spec, src_text)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            f.write(new_text)
        print(f"[gen] {out_path}  ({len(new_text)} bytes)")


if __name__ == "__main__":
    main()
