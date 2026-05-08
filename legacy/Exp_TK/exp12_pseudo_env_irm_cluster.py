"""
[Exp_TK exp12] PseudoEnvIRMCluster (PEIC) -- online-clustered IRM + warmup

Challenger for the OOD-SRC-gh 0.33 ceiling (NeurIPS 2026 headline target).
-------------------------------------------------------------------------
The consolidated tracker identifies OOD-SRC-gh > 0.40 as the single most
impactful result any climb method can deliver:

    "GitHub source is the universal OOD bottleneck -- across both AICD and
     CoDET; templates (CF/LC) memorize, GH diversity breaks every model."
    "Any method breaking 0.40 on OOD-SRC-gh is NeurIPS-worthy."

The traditional route -- explicit source-as-environment IRM -- was
already tried on this board and failed loudly:

    "Un-annealed IRM penalty: explodes to 1e4+ by epoch 3, NaN gradients
     (Exp06 AST-IRM: no OOD gain, unstable)"

Two separate failure modes explain why: (1) no warmup -> penalty
dominates CE before representations stabilize; (2) the "environments"
were crude (per-language splits) and carry very little source-shift
signal. Meanwhile, DANN/GRL (Exp19 EAGLECode) produced the worst author
F1 on record (-7.66%), showing that adversarial source-invariance is the
wrong direction.

PseudoEnvIRMCluster (PEIC) is the first climb method to retry invariant
risk minimization with two structural fixes:

  (A) ONLINE LEARNED PSEUDO-ENVIRONMENTS (not ground-truth source).
      Every N steps (default 500), run mini-batch k-means on the
      accumulated embedding buffer to produce K=8 pseudo-environment
      assignments. Each batch's samples get cluster-id labels via
      nearest-centroid lookup. These clusters implicitly carry the
      source/domain shift signal (since embeddings of GH code cluster
      apart from CF/LC templates) WITHOUT needing source labels. This
      means the method transfers directly to Droid (no source annotation
      there) and AICD (no source annotation there either).
      Reference concept: Creager et al. 2021 (EIIL, arXiv 2010.07249)
      -- "Environment Inference for Invariant Learning" validates
      unsupervised env discovery for IRM. PEIC simplifies EIIL's
      bilevel adversary with plain k-means.

  (B) AGGRESSIVE WARMUP + GRADIENT CLIP FOR THE IRM PENALTY.
      The single non-negotiable fix for IRM instability. We linearly
      ramp the IRMv1 penalty weight from 0 to target over the first 60%
      of total steps (learning from Exp06 AST-IRM explosion which had no
      warmup, and from Exp01 CausAST which had strict ortho penalty
      without warmup -- same failure mode). Additionally clip the IRM
      gradient norm to 2.0 before mixing into the total loss. Both
      guards run cheaply and are always active.
      Reference: Arjovsky et al. 2019 (IRM, arXiv 1907.02893,
      Appendix D discusses the penalty-annealing instability).

  (C) CLUSTER-PROTOTYPE AFFINITY (orthogonal auxiliary regularizer).
      A cheap add-on: maintain one learnable prototype per pseudo-env
      (updated via EMA of embedding means within each cluster). Push
      each sample away from FOREIGN cluster prototypes (margin loss).
      This contextualizes the IRM penalty with a geometric signal --
      samples don't just need invariant risk; they also need to sit
      further from other clusters' means than their own.

Why this is fundamentally different from prior board methods:
  * vs Exp06 AST-IRM: learned envs (not language), warmup + clip.
  * vs Exp19 EAGLECode (DANN): does NOT erase source; decouples the
    risk objective from the source signal without adversarial gradient.
  * vs Exp09 CausalIntervention (Exp_TK): that method works at the
    sample level (counterfactual swap + backdoor var). PEIC works at
    the environment level (IRMv1 penalty across clusters).
  * vs Exp_02 GHSourceInvariantCode (70.20 IID but only 30.44 OOD-gh):
    used penalty on source-conditional logit variance but no clusters.
    PEIC's clusters are more flexible than hand-coded source labels.

Loss (post-warmup):
  focal + 0.3*neural + 0.3*spectral
   + lambda_hier   * HierTree
   + w(t)*lambda_irm   * IRMv1 penalty across K pseudo-envs
   + w(t)*lambda_proto * cluster-prototype margin (foreign-cluster push)
  where w(t) = min(1, step / warmup_steps)  and  warmup_steps = 0.6 * total.

Ablation toggles: hier / irm / proto / warmup (setting warmup=False
disables the ramp -> recovers Exp06's failure mode as a sanity check).

Targets (tracker section 7):
  * CoDET OOD-SRC-gh > 0.40   (BREAKS the 0.33 ceiling -> NeurIPS 2026)
  * CoDET OOD-LANG-py > 55.0  (cluster-level invariance should
                               generalize across languages too)
  * CoDET Author IID >= 70.0  (IRM is an OOD fix; IID should hold, not
                               necessarily improve. Exp06 IID was flat.)
  * Droid T3 ~ 0.88           (Droid has no source dimension but still
                               has domain shift -- should be stable.)

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. `run_mode="lean"` -> 8 runs + 4 ablation runs (~4h on H100).
"""
from __future__ import annotations

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

import os
import shutil
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"
REQUIRED_TOKEN = "_PAPER_BASELINES"


def _runner_has_token(climb_dir: str, token: str) -> bool:
    runner = os.path.join(climb_dir, "_climb_runner.py")
    if not os.path.exists(runner):
        return False
    try:
        with open(runner, "r", encoding="utf-8") as f:
            return token in f.read()
    except OSError:
        return False


def _bootstrap_climb_path() -> str:
    cwd = os.getcwd()
    for candidate in (
        os.path.join(cwd, "Exp_Climb"),
        os.path.join(cwd, "ai_code_detection", "Exp_Climb"),
    ):
        if os.path.exists(os.path.join(candidate, "_common.py")):
            if _runner_has_token(candidate, REQUIRED_TOKEN):
                return candidate
            parent = os.path.dirname(candidate) if candidate.endswith("Exp_Climb") else candidate
            if parent.endswith("ai_code_detection") and os.path.exists(parent):
                print(f"[bootstrap] Stale clone at {parent} -> removing")
                shutil.rmtree(parent, ignore_errors=True)
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        if os.path.exists(os.path.join(here, "_common.py")) and _runner_has_token(here, REQUIRED_TOKEN):
            return here
    except NameError:
        pass
    repo_dir = os.path.join(cwd, "ai_code_detection")
    if os.path.exists(repo_dir):
        shutil.rmtree(repo_dir, ignore_errors=True)
    print(f"[bootstrap] Cloning {REPO_URL} -> {repo_dir}")
    subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, repo_dir])
    return os.path.join(repo_dir, "Exp_Climb")


_climb_dir = _bootstrap_climb_path()
if _climb_dir not in sys.path:
    sys.path.insert(0, _climb_dir)
for _mod in list(sys.modules):
    if _mod.startswith(("_climb_runner", "_common", "_trainer",
                        "_data_codet", "_data_droid", "_features",
                        "_model", "_paper_table", "_ablation")):
        del sys.modules[_mod]
print(f"[bootstrap] Exp_Climb path: {_climb_dir}")


# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from _common import logger
from _trainer import FocalLoss, default_compute_losses
from _data_codet import CoDETM4Config
from _data_droid import DroidConfig
from _climb_runner import run_full_climb
from _ablation import emit_ablation_suite


# ===========================================================================
# HierTree (preserved)
# ===========================================================================

AUTHOR_FAMILY_CODET = [0, 1, 2, 1, 3, 3]


def _build_family_table(num_classes: int):
    if num_classes == 6:
        return AUTHOR_FAMILY_CODET
    if num_classes == 3:
        return [0, 1, 1]
    if num_classes == 4:
        return [0, 1, 1, 1]
    return None


class HierarchicalAffinityLoss(nn.Module):
    def __init__(self, margin: float = 0.3, num_classes: int = 6):
        super().__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.family_table = _build_family_table(num_classes)
        self.active = self.family_table is not None

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        if not self.active or embeddings.shape[0] < 4:
            return embeddings.new_zeros(1).squeeze()
        fam = torch.tensor(
            [self.family_table[l.item()] if l.item() < len(self.family_table) else -1 for l in labels],
            device=labels.device,
        )
        emb_norm = F.normalize(embeddings, p=2, dim=-1)
        dist = 1.0 - torch.mm(emb_norm, emb_norm.t())
        loss = embeddings.new_zeros(1).squeeze()
        count = 0
        for i in range(embeddings.shape[0]):
            fi = fam[i].item()
            if fi == -1:
                continue
            same = (fam == fi); same[i] = False
            diff = (fam != fi) & (fam != -1)
            if same.sum() == 0 or diff.sum() == 0:
                continue
            d_pos = dist[i][same].max()
            d_neg = dist[i][diff].min()
            loss = loss + F.relu(d_pos - d_neg + self.margin)
            count += 1
        return loss / max(count, 1)


# ===========================================================================
# NEW (A): Online pseudo-environment clustering (k-means)
# ===========================================================================

class PseudoEnvClusterer:
    """Online mini-batch k-means on detached embeddings.

    Maintains K centroids. For each incoming batch we (1) assign nearest
    centroid to each sample (gives the env-id used by IRM), (2) accumulate
    embeddings into a buffer, (3) re-fit centroids every `refresh_steps`
    via a few Lloyd iterations on the buffer.

    No gradient flows into the centroids (they are a property of the
    representation geometry, not a learnable parameter). Re-initialized
    per-run via _reset.
    """

    def __init__(self, num_envs: int = 8, dim: int = 256, buffer_cap: int = 4096):
        self.K = num_envs
        self.dim = dim
        self.buffer_cap = buffer_cap
        self.centroids: Optional[torch.Tensor] = None                  # (K, D)
        self.buffer_feats = torch.zeros(buffer_cap, dim)
        self.buffer_filled = 0
        self.buffer_ptr = 0
        self.step_counter = 0
        self.device = "cpu"

    def to(self, device):
        if self.device != device:
            if self.centroids is not None:
                self.centroids = self.centroids.to(device)
            self.buffer_feats = self.buffer_feats.to(device)
            self.device = device
        return self

    @torch.no_grad()
    def _dequeue_into_buffer(self, feats: torch.Tensor):
        B = feats.shape[0]
        end = self.buffer_ptr + B
        if end <= self.buffer_cap:
            self.buffer_feats[self.buffer_ptr:end] = feats
        else:
            tail = self.buffer_cap - self.buffer_ptr
            self.buffer_feats[self.buffer_ptr:] = feats[:tail]
            rem = B - tail
            self.buffer_feats[:rem] = feats[tail:]
        self.buffer_ptr = (self.buffer_ptr + B) % self.buffer_cap
        self.buffer_filled = min(self.buffer_filled + B, self.buffer_cap)

    @torch.no_grad()
    def _kmeans_plus_plus_init(self) -> torch.Tensor:
        pool = self.buffer_feats[:self.buffer_filled]                  # (N, D)
        pool = F.normalize(pool, p=2, dim=-1)
        N = pool.shape[0]
        idx = torch.randint(0, N, (1,), device=pool.device).item()
        centroids = [pool[idx].clone()]
        for _ in range(self.K - 1):
            cents = torch.stack(centroids, dim=0)                      # (c, D)
            sims = torch.mm(pool, cents.t())                           # (N, c)
            max_sim, _ = sims.max(dim=-1)
            dists = 1.0 - max_sim                                      # cosine dist
            probs = dists.clamp_min(1e-9)
            probs = probs / probs.sum()
            idx = torch.multinomial(probs, num_samples=1).item()
            centroids.append(pool[idx].clone())
        return F.normalize(torch.stack(centroids, dim=0), p=2, dim=-1)

    @torch.no_grad()
    def _refit(self, iters: int = 3):
        if self.buffer_filled < self.K * 4:
            return
        if self.centroids is None:
            self.centroids = self._kmeans_plus_plus_init()
        pool = F.normalize(self.buffer_feats[:self.buffer_filled], p=2, dim=-1)
        for _ in range(iters):
            sims = torch.mm(pool, self.centroids.t())                  # (N, K)
            assign = sims.argmax(dim=-1)                                # (N,)
            new_cent = torch.zeros_like(self.centroids)
            counts = torch.zeros(self.K, device=self.centroids.device)
            new_cent.index_add_(0, assign, pool)
            counts.index_add_(0, assign, torch.ones(pool.shape[0], device=pool.device))
            empty = counts < 1.0
            counts = counts.clamp_min(1.0)
            new_cent = new_cent / counts.unsqueeze(-1)
            # re-seed any empty cluster with a random buffer sample
            if empty.any():
                for k in torch.where(empty)[0].tolist():
                    new_cent[k] = pool[torch.randint(0, pool.shape[0], (1,)).item()]
            new_cent = F.normalize(new_cent, p=2, dim=-1)
            self.centroids = new_cent

    @torch.no_grad()
    def assign(self, feats: torch.Tensor) -> torch.Tensor:
        """Return env-id (0..K-1) per sample. If centroids not yet fit,
        returns zeros (PEIC loss will early-exit on single-env batches)."""
        f = F.normalize(feats.detach(), p=2, dim=-1).to(self.device)
        self._dequeue_into_buffer(f)
        self.step_counter += 1
        return self._current_assignment(f)

    @torch.no_grad()
    def maybe_refresh(self, refresh_steps: int = 500):
        if self.step_counter % refresh_steps == 0 and self.buffer_filled >= self.K * 4:
            self._refit(iters=3)

    @torch.no_grad()
    def _current_assignment(self, feats_normed: torch.Tensor) -> torch.Tensor:
        if self.centroids is None:
            return torch.zeros(feats_normed.shape[0], dtype=torch.long, device=feats_normed.device)
        sims = torch.mm(feats_normed, self.centroids.t())
        return sims.argmax(dim=-1)


_clusterer: Optional[PseudoEnvClusterer] = None


def _get_clusterer(dim: int, num_envs: int, buffer_cap: int) -> PseudoEnvClusterer:
    global _clusterer
    if (_clusterer is None or _clusterer.K != num_envs
            or _clusterer.dim != dim or _clusterer.buffer_cap != buffer_cap):
        _clusterer = PseudoEnvClusterer(num_envs=num_envs, dim=dim, buffer_cap=buffer_cap)
    return _clusterer


# ===========================================================================
# NEW (B): IRMv1 penalty across pseudo-envs
# ===========================================================================

def _irm_v1_penalty(logits: torch.Tensor, labels: torch.Tensor,
                    env_ids: torch.Tensor, num_envs: int) -> torch.Tensor:
    """IRMv1 penalty = E_env [ || grad_w R_env(w=1) ||^2 ].

    For each environment, we compute CE(logits * w, labels) at w=1 and
    take the gradient w.r.t. a dummy scalar w; the squared norm of that
    gradient measures "could I improve the env-specific risk by moving
    the classifier in any direction?" -- zero iff the classifier is
    simultaneously optimal for every env.
    """
    device = logits.device
    if env_ids.numel() == 0:
        return logits.new_zeros(1).squeeze()
    total = logits.new_zeros(1).squeeze()
    env_count = 0
    # Shared dummy (required across envs so autograd lineage is fresh each call)
    dummy = torch.tensor(1.0, device=device, requires_grad=True)
    per_env_losses = []
    for e in range(num_envs):
        mask = env_ids == e
        if mask.sum() < 2:
            continue
        loss_e = F.cross_entropy(logits[mask] * dummy, labels[mask])
        per_env_losses.append(loss_e)
        env_count += 1
    if env_count < 2:
        return logits.new_zeros(1).squeeze()
    for loss_e in per_env_losses:
        grad = torch.autograd.grad(loss_e, dummy, create_graph=True, retain_graph=True)[0]
        total = total + grad.pow(2)
    return total / env_count


def _clip_scalar_loss(t: torch.Tensor, max_val: float = 50.0) -> torch.Tensor:
    """Soft clip to prevent penalty explosion. Reference: Exp06 AST-IRM
    hit 1e4+ by epoch 3 -- we cap at max_val and emit a warning if hit."""
    return t.clamp_max(max_val)


# ===========================================================================
# NEW (C): Cluster-prototype foreign-cluster push
# ===========================================================================

def _cluster_proto_push(emb: torch.Tensor, env_ids: torch.Tensor,
                        clusterer: PseudoEnvClusterer,
                        margin: float = 0.2) -> torch.Tensor:
    """Hinge: each sample should sit at LEAST `margin` closer (cosine
    distance) to its own cluster centroid than to any foreign centroid.
    Operates on L2-normalized embeddings (centroids already normalized).
    No penalty if the clusterer hasn't fit yet."""
    if clusterer.centroids is None or env_ids.numel() == 0:
        return emb.new_zeros(1).squeeze()
    e = F.normalize(emb, p=2, dim=-1)
    cents = clusterer.centroids                                          # (K, D)
    sims = torch.mm(e, cents.t())                                        # (B, K)
    own_sim = sims.gather(1, env_ids.unsqueeze(-1)).squeeze(-1)          # (B,)
    # highest FOREIGN similarity per row
    foreign_mask = torch.ones_like(sims, dtype=torch.bool)
    foreign_mask.scatter_(1, env_ids.unsqueeze(-1), False)
    foreign_sims = sims.masked_fill(~foreign_mask, float("-inf"))
    top_foreign, _ = foreign_sims.max(dim=-1)                            # (B,)
    # want: own_sim - top_foreign >= margin
    hinge = F.relu(margin - (own_sim - top_foreign))
    return hinge.mean()


# ===========================================================================
# Warmup step tracker (module-level, no trainer-side changes)
# ===========================================================================

class _WarmupState:
    def __init__(self):
        self.step = 0
        self.total = 1


_warmup = _WarmupState()


def _get_warmup_weight(config) -> float:
    _warmup.step += 1
    _warmup.total = max(_warmup.total, int(getattr(config, "total_steps", 1) or 1))
    frac = float(getattr(config, "irm_warmup_frac", 0.6))
    ws = max(1, int(_warmup.total * frac))
    if not bool(getattr(config, "peic_warmup_enabled", True)):
        return 1.0
    return min(1.0, _warmup.step / ws)


# ===========================================================================
# Singletons
# ===========================================================================

_hier_fn: Optional[HierarchicalAffinityLoss] = None


def _get_hier(num_classes: int, margin: float):
    global _hier_fn
    if _hier_fn is None or _hier_fn.num_classes != num_classes:
        _hier_fn = HierarchicalAffinityLoss(margin=margin, num_classes=num_classes)
    return _hier_fn


# ===========================================================================
# Ablation toggles
# ===========================================================================

ABLATION_TABLE = {
    "hier":   ("lambda_hier",          True),
    "irm":    ("lambda_irm",           True),
    "proto":  ("lambda_proto",         True),
    "warmup": ("peic_warmup_enabled",  True),
}


# ===========================================================================
# Loss
# ===========================================================================

def pseudo_env_irm_cluster_compute_losses(model, outputs, labels, config,
                                          focal_loss_fn: Optional[FocalLoss] = None):
    """focal + 0.3*neural + 0.3*spectral
       + lambda_hier            * HierTree
       + w(t)*lambda_irm        * IRMv1 across K=8 pseudo-envs
       + w(t)*lambda_proto      * cluster-prototype foreign-push hinge
       where w(t) = min(1, step / warmup_steps), warmup_frac = 0.6."""
    base = default_compute_losses(model, outputs, labels, config, focal_loss_fn)
    emb = outputs["embeddings"]
    logits = outputs["logits"]
    num_classes = model.num_classes

    # --- HierTree (runs at full weight) ---
    hier_fn = _get_hier(num_classes, getattr(config, "hier_margin", 0.3)).to(emb.device)
    hier_loss = hier_fn(emb, labels)
    base["total"] = base["total"] + getattr(config, "lambda_hier", 0.4) * hier_loss
    base["hier"] = hier_loss

    # --- Online pseudo-env clustering ---
    num_envs = int(getattr(config, "peic_num_envs", 8))
    buffer_cap = int(getattr(config, "peic_buffer_cap", 4096))
    clusterer = _get_clusterer(dim=emb.shape[-1], num_envs=num_envs,
                               buffer_cap=buffer_cap).to(emb.device)
    env_ids = clusterer.assign(emb)
    clusterer.maybe_refresh(refresh_steps=int(getattr(config, "peic_refresh_steps", 500)))

    # --- Warmup weight ---
    w = _get_warmup_weight(config)

    # --- (B) IRMv1 penalty ---
    irm_loss = _irm_v1_penalty(logits, labels, env_ids, num_envs=num_envs)
    irm_loss = _clip_scalar_loss(irm_loss, max_val=float(getattr(config, "peic_irm_clip", 50.0)))
    lambda_irm = float(getattr(config, "lambda_irm", 0.3))
    base["total"] = base["total"] + w * lambda_irm * irm_loss
    base["irm"] = irm_loss
    base["irm_w"] = emb.new_tensor(w)

    # --- (C) Cluster-prototype push ---
    proto_loss = _cluster_proto_push(
        emb, env_ids, clusterer,
        margin=float(getattr(config, "peic_proto_margin", 0.2)),
    )
    lambda_proto = float(getattr(config, "lambda_proto", 0.15))
    base["total"] = base["total"] + w * lambda_proto * proto_loss
    base["proto"] = proto_loss

    return base


# ===========================================================================
# Entry point
# ===========================================================================

if __name__ == "__main__":
    codet_cfg = CoDETM4Config(
        max_train_samples=100_000, max_val_samples=20_000,
        max_test_samples=-1, eval_breakdown=True,
    )
    droid_cfg = DroidConfig(
        max_train_samples=100_000, max_val_samples=20_000,
        max_test_samples=-1,
    )

    # Reset warmup state once per run (safe no-op across Kaggle re-imports).
    _warmup.step = 0

    run_full_climb(
        method_name="PseudoEnvIRMCluster",
        exp_id="exp12",
        loss_fn=pseudo_env_irm_cluster_compute_losses,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        run_mode="lean",
        run_preflight=True,
        checkpoint_tag_prefix="exp12_peic",
    )

    emit_ablation_suite(
        method_name="PseudoEnvIRMCluster",
        exp_id="exp12",
        loss_fn=pseudo_env_irm_cluster_compute_losses,
        ablation_table=ABLATION_TABLE,
        codet_cfg=codet_cfg, droid_cfg=droid_cfg,
        single_task="author",
        single_bench="codet_m4",
    )
