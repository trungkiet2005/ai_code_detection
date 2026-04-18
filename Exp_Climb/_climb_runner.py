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

from _common import PreflightError, get_gpu_name, logger
from _data_codet import (
    CoDETM4Config,
    FULL_RUN_PLAN,
    preflight as preflight_codet,
    run_codet_suite,
    run_iid as run_codet_iid,
)
from _data_droid import DroidConfig, preflight_droid, run_droid_suite
from _features import ast_parser_available


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
# Environment + full-suite preflight (fail FAST before committing compute)
# ---------------------------------------------------------------------------

def preflight_env() -> Dict[str, Any]:
    """Check CUDA, tree-sitter, torch version. Raises PreflightError on failure."""
    import torch as _torch
    import transformers as _tf

    logger.info("\n" + "=" * 70)
    logger.info("PREFLIGHT ENV")
    logger.info("=" * 70)

    gpu = get_gpu_name()
    cuda_available = _torch.cuda.is_available()
    report = {
        "gpu": gpu,
        "cuda_available": cuda_available,
        "torch_version": _torch.__version__,
        "transformers_version": _tf.__version__,
        "tree_sitter_available": ast_parser_available(),
    }
    logger.info(
        f"  GPU={gpu} | CUDA={cuda_available} | "
        f"torch={_torch.__version__} | transformers={_tf.__version__} | "
        f"tree-sitter={'OK' if report['tree_sitter_available'] else 'MISSING'}"
    )
    if not cuda_available:
        logger.warning("  No CUDA device detected -- will run on CPU (extremely slow)")
    if not report["tree_sitter_available"]:
        logger.warning(
            "  tree-sitter fallback active -- AST features use regex heuristic (lower quality)"
        )
    return report


def _print_full_run_plan(
    run_mode: str,
    codet_plan: List[Tuple[str, str]],
    droid_tasks: List[str],
    codet_cfg: CoDETM4Config,
    droid_cfg: DroidConfig,
    method_name: str,
):
    """Print the full list of planned runs + data sizes before committing compute."""
    logger.info("\n" + "#" * 72)
    logger.info(f"# FULL RUN PLAN ({method_name}) -- mode={run_mode}")
    logger.info("#" * 72)

    logger.info(f"\nCoDET-M4 | train={codet_cfg.max_train_samples:,} val={codet_cfg.max_val_samples:,} "
                f"test={'FULL' if codet_cfg.max_test_samples <= 0 else f'{codet_cfg.max_test_samples:,}'}")
    ood_counts = {
        ("ood", "generator"): 5,
        ("ood", "language"): 3,
        ("ood", "source"): 3,
    }
    codet_runs = 0
    for i, (mode, task) in enumerate(codet_plan):
        extra = f" ({ood_counts[(mode, task)]} held-outs)" if (mode, task) in ood_counts else ""
        logger.info(f"  [C{i+1}] {mode}/{task}{extra}")
        codet_runs += ood_counts.get((mode, task), 1)

    logger.info(f"\nDroid | train={droid_cfg.max_train_samples:,} val={droid_cfg.max_val_samples:,} "
                f"test={'FULL' if droid_cfg.max_test_samples <= 0 else f'{droid_cfg.max_test_samples:,}'}")
    for i, task in enumerate(droid_tasks):
        logger.info(f"  [D{i+1}] {task}")

    total = codet_runs + len(droid_tasks)
    logger.info(f"\nTOTAL TRAIN-EVAL CYCLES: {total} "
                f"(CoDET={codet_runs} + Droid={len(droid_tasks)}) -- ~{total * 25} min on H100")
    logger.info("#" * 72 + "\n")


def preflight_full_climb(
    codet_cfg: CoDETM4Config,
    droid_cfg: DroidConfig,
    run_mode: str,
    method_name: str,
    droid_tasks: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """One-shot preflight for the entire climb.

    Runs env check + CoDET probe + Droid probe (per needed bench). Aborts early
    (PreflightError) before any training so you don't lose 7 hours to find out
    Droid auth was broken.
    """
    if droid_tasks is None:
        droid_tasks = ["T1", "T3", "T4"]

    report: Dict[str, Any] = {"env": preflight_env()}

    need_codet = run_mode in ("full", "codet_only", "codet_iid", "single")
    need_droid = run_mode in ("full", "droid_only")

    _print_full_run_plan(
        run_mode=run_mode,
        codet_plan=FULL_RUN_PLAN if run_mode in ("full", "codet_only") else [e for e in FULL_RUN_PLAN if e[0] == "iid"],
        droid_tasks=droid_tasks if need_droid else [],
        codet_cfg=codet_cfg,
        droid_cfg=droid_cfg,
        method_name=method_name,
    )

    if need_codet:
        # Build a temp exp_cfg just for preflight tokenizer/AST check
        from _common import apply_hardware_profile, SpectralConfig
        exp_cfg = apply_hardware_profile(SpectralConfig(
            task=codet_cfg.task, benchmark="codet_m4",
            save_dir=codet_cfg.save_root,
            require_tree_sitter=False,  # don't hard-fail preflight on regex fallback
        ))
        report["codet"] = preflight_codet(codet_cfg, exp_cfg)

    if need_droid:
        report["droid"] = preflight_droid(droid_cfg, tasks=droid_tasks)

    logger.info("\n" + "=" * 70)
    logger.info(f"PREFLIGHT FULL CLIMB ({method_name}) -- ALL CHECKS PASS")
    logger.info("=" * 70 + "\n")
    return report


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

    # ── One-shot full-climb preflight (env + CoDET + Droid probes + run plan).
    # Fails FAST before any training -- saves hours on misconfig.
    if run_preflight:
        preflight_full_climb(
            codet_cfg=codet_cfg,
            droid_cfg=droid_cfg,
            run_mode=run_mode,
            method_name=method_name,
        )

    # Inner suites should NOT re-run preflight (we already ran it once above).
    inner_preflight = False

    codet_results: Optional[Dict[str, Any]] = None
    droid_results: Optional[Dict[str, Any]] = None

    try:
        if run_mode == "full":
            logger.info("\n" + "#" * 72)
            logger.info(f"# [{method_name}] PHASE 1/2: CoDET-M4 full suite")
            logger.info("#" * 72)
            codet_results = run_codet_suite(
                FULL_RUN_PLAN, codet_cfg, loss_fn=loss_fn,
                run_preflight=inner_preflight, checkpoint_tag_prefix=checkpoint_tag_prefix,
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
                run_preflight=inner_preflight, checkpoint_tag_prefix=checkpoint_tag_prefix,
            )

        elif run_mode == "droid_only":
            droid_results = run_droid_suite(
                droid_cfg, loss_fn=loss_fn, checkpoint_tag_prefix=checkpoint_tag_prefix,
            )

        elif run_mode == "codet_iid":
            iid_plan = [e for e in FULL_RUN_PLAN if e[0] == "iid"]
            codet_results = run_codet_suite(
                iid_plan, codet_cfg, loss_fn=loss_fn,
                run_preflight=inner_preflight, checkpoint_tag_prefix=checkpoint_tag_prefix,
            )

        elif run_mode == "single":
            r = run_codet_iid(
                single_task, codet_cfg, loss_fn=loss_fn,
                run_preflight=inner_preflight, checkpoint_tag_prefix=checkpoint_tag_prefix,
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
