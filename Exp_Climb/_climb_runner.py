"""
Orchestrator — one-call entry point that runs CoDET-M4 full suite +
DroidCollection T1/T3/T4 sequentially, then emits ONE combined paper-table
block (BEGIN_PAPER_TABLE / END_PAPER_TABLE) covering both benches.

Usage in exp_NN_*.py:

    from _climb_runner import run_full_climb
    run_full_climb(
        method_name="HierTreeCode",
        exp_id="exp_00",
        loss_fn=hiertree_compute_losses,   # method-specific loss
        codet_cfg=CoDETM4Config(...),
        droid_cfg=DroidConfig(...),
        run_mode="full",                   # full | codet_only | droid_only | codet_iid
    )
"""
from __future__ import annotations

import gc
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from _common import PreflightError, logger
from _data_codet import (
    CoDETM4Config,
    FULL_RUN_PLAN,
    run_codet_suite,
    run_iid as run_codet_iid,
)
from _data_droid import DroidConfig, run_droid_suite


# ---------------------------------------------------------------------------
# Combined paper-table emitter (CoDET + Droid)
# ---------------------------------------------------------------------------

def emit_combined_paper_table(
    codet_results: Optional[Dict[str, Any]],
    droid_results: Optional[Dict[str, Any]],
    method_name: str,
    exp_id: str,
):
    """Flatten CoDET + Droid results into a single run_plan and emit one paper table."""
    try:
        from _paper_table import emit_paper_table
    except ImportError:
        logger.warning("[_paper_table] helper not found; skipping paper-ready table")
        return

    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    flat_run_plan: List[Tuple[str, str]] = []
    flat_results: Dict[str, Dict] = {}

    if codet_results:
        for key, stats in codet_results.items():
            if not isinstance(stats, dict):
                continue
            if key.startswith("iid_") and "test_f1" in stats:
                flat_run_plan.append(("codet_m4", key))
                flat_results[f"codet_m4_{key}"] = stats
            elif key.startswith("ood_"):
                mode = key.split("_", 1)[1]
                for held, h_stats in stats.items():
                    if isinstance(h_stats, dict) and "test_f1" in h_stats:
                        task_id = f"ood_{mode}_{held}"
                        flat_run_plan.append(("codet_m4", task_id))
                        flat_results[f"codet_m4_{task_id}"] = h_stats

    if droid_results:
        for key, stats in droid_results.items():
            if not isinstance(stats, dict) or "error" in stats:
                continue
            task = key.replace("droid_", "")
            flat_run_plan.append(("droid", task))
            flat_results[f"droid_{task}"] = stats

    if not flat_run_plan:
        logger.warning("No results to emit in paper table")
        return

    emit_paper_table(
        method_name=method_name,
        exp_id=exp_id,
        run_plan=flat_run_plan,
        results=flat_results,
        timestamp=ts,
        logger=logger,
    )


# ---------------------------------------------------------------------------
# Full climb orchestrator
# ---------------------------------------------------------------------------

RUN_MODES = ("full", "codet_only", "droid_only", "codet_iid", "single")


def run_full_climb(
    method_name: str,
    exp_id: str,
    loss_fn: Optional[Callable] = None,
    codet_cfg: Optional[CoDETM4Config] = None,
    droid_cfg: Optional[DroidConfig] = None,
    run_mode: str = "full",
    single_task: str = "binary",
    run_preflight: bool = True,
    checkpoint_tag_prefix: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the standard climb: CoDET-M4 suite -> cleanup -> Droid suite -> combined table.

    Args:
        method_name: Human-readable method name (e.g. "HierTreeCode")
        exp_id: Numbered ID (e.g. "exp_00")
        loss_fn: Method-specific loss function (see _trainer.LossFn). None = default (focal+neural+spectral).
        codet_cfg / droid_cfg: benchmark-specific configs. Defaults use CLIMB protocol (20% train / FULL test).
        run_mode: "full" (default) | "codet_only" | "droid_only" | "codet_iid" | "single"
    """
    if run_mode not in RUN_MODES:
        raise ValueError(f"Unknown run_mode: {run_mode}. Pick from {RUN_MODES}.")

    if codet_cfg is None:
        codet_cfg = CoDETM4Config()
    if droid_cfg is None:
        droid_cfg = DroidConfig()
    if checkpoint_tag_prefix is None:
        checkpoint_tag_prefix = exp_id

    codet_results: Optional[Dict[str, Any]] = None
    droid_results: Optional[Dict[str, Any]] = None

    try:
        if run_mode == "full":
            logger.info("\n" + "#" * 72)
            logger.info(f"# [{method_name}] PHASE 1/2: CoDET-M4 full suite")
            logger.info("#" * 72)
            codet_results = run_codet_suite(
                FULL_RUN_PLAN, codet_cfg, loss_fn=loss_fn,
                run_preflight=run_preflight, checkpoint_tag_prefix=checkpoint_tag_prefix,
            )

            # Free VRAM + Python-side tensors between benches
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            logger.info("\n" + "#" * 72)
            logger.info(f"# [{method_name}] PHASE 2/2: DroidCollection T1 + T3 + T4")
            logger.info("#" * 72)
            droid_results = run_droid_suite(
                droid_cfg, loss_fn=loss_fn, checkpoint_tag_prefix=checkpoint_tag_prefix,
            )

        elif run_mode == "codet_only":
            codet_results = run_codet_suite(
                FULL_RUN_PLAN, codet_cfg, loss_fn=loss_fn,
                run_preflight=run_preflight, checkpoint_tag_prefix=checkpoint_tag_prefix,
            )

        elif run_mode == "droid_only":
            droid_results = run_droid_suite(
                droid_cfg, loss_fn=loss_fn, checkpoint_tag_prefix=checkpoint_tag_prefix,
            )

        elif run_mode == "codet_iid":
            iid_plan = [e for e in FULL_RUN_PLAN if e[0] == "iid"]
            codet_results = run_codet_suite(
                iid_plan, codet_cfg, loss_fn=loss_fn,
                run_preflight=run_preflight, checkpoint_tag_prefix=checkpoint_tag_prefix,
            )

        elif run_mode == "single":
            r = run_codet_iid(
                single_task, codet_cfg, loss_fn=loss_fn,
                run_preflight=run_preflight, checkpoint_tag_prefix=checkpoint_tag_prefix,
            )
            codet_results = {f"iid_{single_task}": r}

        # Emit ONE combined paper table
        emit_combined_paper_table(
            codet_results=codet_results,
            droid_results=droid_results,
            method_name=method_name,
            exp_id=exp_id,
        )

    except PreflightError as e:
        logger.error(f"PRE-FLIGHT FAILED: {e}")
        raise

    return {"codet_m4": codet_results, "droid": droid_results}
