"""
Shared ablation helper for Exp_Climb methods.

Goal: every lean-mode run produces an explicit component-contribution table
inside the BEGIN_PAPER_TABLE block, so reviewers can see which part of a
stacked loss actually moved the needle (instead of guessing from the
cocktail of lambdas).

Contract
--------
A method that wants ablation reporting must:
  1. At the TOP of its exp file, define an `ABLATION_TABLE` dict that maps
     component name -> (config_flag, default_on: bool). Example:

        ABLATION_TABLE = {
            "hier":     ("lambda_hier", True),
            "flow":     ("lambda_fm",   True),
            "knn":      ("use_rag",     True),
        }

  2. Inside the compute_losses function, read each flag via
     `getattr(config, flag, default)` so disabling a component is
     equivalent to zeroing its lambda / setting the bool False.

  3. At the END of __main__, call:
        emit_ablation_suite(
            method_name="FlowCodeDet",
            exp_id="exp_06",
            loss_fn=flow_compute_losses,
            ablation_table=ABLATION_TABLE,
            codet_cfg=codet_cfg, droid_cfg=droid_cfg,
            single_task="author",            # ablation target task
            single_bench="codet_m4",
        )

The helper runs the full set with every component ON (the "baseline of the
method"), then one run per component with that component OFF. For each
run it re-uses the climb harness at `run_mode="single"` (CoDET IID author
only — cheapest signal that tracks the primary metric).

Output is a compact markdown block appended to the paper table section.
"""

from __future__ import annotations

import copy
import gc
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from _common import logger


@dataclass
class AblationResult:
    name: str                # "full", "no_hier", "no_flow", ...
    disabled: Optional[str]  # None for the full baseline
    test_f1: float
    best_val_f1: float
    wall_time_s: float


def _patch_config(exp_cfg, flag: str, disable: bool):
    """Temporarily zero a lambda / flip a bool flag. Returns the old value."""
    old = getattr(exp_cfg, flag, None)
    if disable:
        if isinstance(old, bool):
            setattr(exp_cfg, flag, False)
        else:
            # numeric lambda -> 0 disables the term
            setattr(exp_cfg, flag, 0.0)
    return old


def _restore_config(exp_cfg, flag: str, value: Any):
    setattr(exp_cfg, flag, value)


def emit_ablation_suite(
    method_name: str,
    exp_id: str,
    loss_fn: Callable,
    ablation_table: Dict[str, Tuple[str, bool]],
    codet_cfg,
    droid_cfg,
    single_task: str = "author",
    single_bench: str = "codet_m4",
    checkpoint_tag_prefix: Optional[str] = None,
) -> List[AblationResult]:
    """Run a leave-one-out ablation on a single target task and log a table.

    Rationale: a full-lean ablation would cost 8 runs × (1 + #components)
    which is wasteful. We use the single cheapest run that tracks the
    primary metric (CoDET IID author) -- ~22 min on H100 -- and repeat
    it per dropped component. For 3 components: 4 × 22 = ~88 min.
    """
    from _climb_runner import run_full_climb

    if checkpoint_tag_prefix is None:
        checkpoint_tag_prefix = f"{exp_id}_ablation"

    # Each ablation run uses a fresh copy of configs + a small monkey-patch
    # on the module-level lambda values the method reads via getattr(config, flag).
    # The `codet_cfg` / `droid_cfg` dataclasses themselves are data-config only;
    # the loss-fn-side lambdas live on the per-task SpectralConfig built inside
    # the runner. We intercept via a wrapping loss_fn that reads our override.

    runs: List[AblationResult] = []

    def _run_one(name: str, disabled_flag: Optional[str]) -> AblationResult:
        logger.info("\n" + "#" * 72)
        logger.info(f"# [Ablation] {method_name} -- variant={name} | disabled={disabled_flag}")
        logger.info("#" * 72)

        # Wrap loss_fn to override the disabled flag on `config` at call time.
        if disabled_flag is None:
            effective_loss = loss_fn
        else:
            def effective_loss(model, outputs, labels, config, focal_loss_fn=None,
                               _flag=disabled_flag, _base=loss_fn):
                # Zero the disabled component's lambda (or flip its bool flag)
                old = getattr(config, _flag, None)
                if isinstance(old, bool):
                    setattr(config, _flag, False)
                else:
                    setattr(config, _flag, 0.0)
                try:
                    return _base(model, outputs, labels, config, focal_loss_fn)
                finally:
                    if old is not None:
                        setattr(config, _flag, old)

        t0 = time.time()
        results = run_full_climb(
            method_name=f"{method_name}-{name}",
            exp_id=f"{exp_id}_{name}",
            loss_fn=effective_loss,
            codet_cfg=codet_cfg,
            droid_cfg=droid_cfg,
            run_mode="single",
            single_task=single_task,
            run_preflight=False,
            checkpoint_tag_prefix=f"{checkpoint_tag_prefix}_{name}",
        )
        dt = time.time() - t0

        codet_res = (results or {}).get("codet_m4") or {}
        key = f"iid_{single_task}"
        stats = codet_res.get(key, {}) if isinstance(codet_res, dict) else {}
        test_f1 = float(stats.get("test_f1", 0.0))
        val_f1 = float(stats.get("best_val_f1", 0.0))
        logger.info(
            f"[Ablation] variant={name} | test_f1={test_f1:.4f} | "
            f"val_f1={val_f1:.4f} | wall={dt:.0f}s"
        )
        # Free memory before the next variant
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return AblationResult(
            name=name, disabled=disabled_flag,
            test_f1=test_f1, best_val_f1=val_f1, wall_time_s=dt,
        )

    # 1) Full (all components on) -- baseline
    runs.append(_run_one("full", None))

    # 2) Leave-one-out: drop each component in turn
    for comp, (flag, default_on) in ablation_table.items():
        if not default_on:
            continue  # component is already off by default; skip
        runs.append(_run_one(f"no_{comp}", flag))

    # 3) Emit ablation markdown table
    full_run = runs[0]
    table_lines = [
        "",
        "=" * 78,
        f"BEGIN_ABLATION_TABLE ({method_name} / {exp_id}, target={single_bench}/{single_task})",
        "=" * 78,
        "",
        "### Component ablation (single-task CoDET author, drop one component per row)",
        "",
        f"| Variant | Disabled | Test F1 | Δ vs. full | Val F1 | Wall (s) |",
        f"|:--------|:---------|:-------:|:----------:|:------:|:--------:|",
    ]
    for r in runs:
        delta = r.test_f1 - full_run.test_f1
        delta_str = f"`{'+' if delta >= 0 else ''}{delta:.4f}`" if r.disabled else "— (baseline)"
        disabled_str = f"`{r.disabled}`" if r.disabled else "—"
        table_lines.append(
            f"| {r.name} | {disabled_str} | "
            f"{r.test_f1:.4f} | {delta_str} | {r.best_val_f1:.4f} | {r.wall_time_s:.0f} |"
        )

    # Ranking: largest drop == most-important component
    drop_sorted = sorted(
        [r for r in runs if r.disabled is not None],
        key=lambda r: (r.test_f1 - full_run.test_f1),   # most negative first
    )
    if drop_sorted:
        table_lines.append("")
        table_lines.append(
            f"**Most-impactful component:** `{drop_sorted[0].disabled}` "
            f"(removing it drops test F1 by {full_run.test_f1 - drop_sorted[0].test_f1:+.4f})."
        )
        table_lines.append(
            f"**Least-impactful component:** `{drop_sorted[-1].disabled}` "
            f"(removing it drops test F1 by {full_run.test_f1 - drop_sorted[-1].test_f1:+.4f})."
        )
    table_lines.append("")
    table_lines.append("=" * 78)
    table_lines.append(f"END_ABLATION_TABLE")
    table_lines.append("=" * 78)

    for line in table_lines:
        logger.info(line)

    return runs
