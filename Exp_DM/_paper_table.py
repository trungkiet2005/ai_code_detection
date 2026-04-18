"""
Paper-ready table emitter — shared across all DM experiments.

Usage at end of `run_benchmark_suite()`:

    from _paper_table import emit_paper_table
    emit_paper_table(
        method_name="HierTreeCode",
        exp_id="exp18",
        run_plan=run_plan,
        results=results,
        timestamp=ts,
        logger=logger,
    )

Results dict format (per run_key):
    {
        "num_classes": int,
        "paper_primary_metric": "macro_f1" | "weighted_f1",
        "best_val_f1": float,
        "test_f1": float,            # primary
        "test_macro_f1": float,
        "test_weighted_f1": float,
        "test_macro_recall": float,
        "test_weighted_recall": float,
        "test_accuracy": float,
        "test_per_class": dict,      # sklearn classification_report(output_dict=True)
    }

Output is bounded by BEGIN_PAPER_TABLE / END_PAPER_TABLE markers for easy extraction.
"""

from typing import Dict, List, Tuple, Optional


def _fmt(v, nd: int = 4) -> str:
    try:
        return f"{float(v):.{nd}f}"
    except Exception:
        return "-"


def emit_paper_table(
    method_name: str,
    exp_id: str,
    run_plan: List[Tuple[str, str]],
    results: Dict[str, Dict],
    timestamp: str,
    logger=None,
    gpu_name: str = "NVIDIA H100 80GB HBM3",
    precision: str = "bf16",
    batch_info: str = "64x1",
):
    """Emit a self-contained markdown block ready for paper / tracker copy-paste."""
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)

    lines: List[str] = []
    lines.append("")
    lines.append("=" * 78)
    lines.append("BEGIN_PAPER_TABLE")
    lines.append("=" * 78)
    lines.append(f"## {method_name} ({exp_id.upper()}) -- Full Suite Results")
    lines.append(
        f"**Timestamp:** `{timestamp}` | **GPU:** `{gpu_name}` | "
        f"**Precision:** `{precision}` | **Batch:** `{batch_info}`"
    )
    lines.append("")

    # 1) Headline Metrics
    lines.append("### Headline Metrics")
    lines.append("")
    lines.append("| Benchmark | Task | # Classes | Primary | Best Val | Test Primary | Macro-F1 | Weighted-F1 | Macro-R | Weighted-R | Accuracy |")
    lines.append("|:----------|:----:|:---------:|:-------:|:--------:|:------------:|:--------:|:-----------:|:-------:|:----------:|:--------:|")
    for (bench, task), stats in zip(run_plan, results.values()):
        lines.append(
            f"| {bench.upper()} | {task} | {stats.get('num_classes', '-')} | "
            f"`{stats.get('paper_primary_metric', 'macro_f1')}` | "
            f"{_fmt(stats.get('best_val_f1'))} | {_fmt(stats.get('test_f1'))} | "
            f"{_fmt(stats.get('test_macro_f1'))} | {_fmt(stats.get('test_weighted_f1'))} | "
            f"{_fmt(stats.get('test_macro_recall'))} | {_fmt(stats.get('test_weighted_recall'))} | "
            f"{_fmt(stats.get('test_accuracy'))} |"
        )
    lines.append("")

    # 2) Per-class Test F1
    lines.append("### Per-Class Test F1 (per run)")
    lines.append("")
    for (bench, task), stats in zip(run_plan, results.values()):
        per_class = stats.get("test_per_class", {}) or {}
        class_rows = {k: v for k, v in per_class.items() if isinstance(k, str) and k.isdigit()}
        if not class_rows:
            continue
        lines.append(f"#### {bench.upper()} {task} ({stats.get('num_classes', '?')}-class)")
        lines.append("")
        lines.append("| Class | Support | Precision | Recall | F1 |")
        lines.append("|:------|--------:|:---------:|:------:|:--:|")
        sorted_items = sorted(
            class_rows.items(),
            key=lambda kv: kv[1].get("f1-score", 0.0),
            reverse=True,
        )
        for cls, m in sorted_items:
            lines.append(
                f"| {cls} | {int(m.get('support', 0))} | "
                f"{_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | "
                f"**{_fmt(m.get('f1-score'))}** |"
            )
        lines.append("")

    # 3) Tracker rows
    lines.append("### Tracker Rows (paste into the matching tracker file)")
    lines.append("")
    lines.append("> `Exp_DM/dm_tracker.md` for AICD/Droid runs -- `Exp_CodeDet/tracker.md` for CoDET-M4 runs.")
    lines.append("")

    def _group(prefix: str):
        return [
            (task, stats)
            for (bench, task), stats in zip(run_plan, results.values())
            if bench.lower() == prefix
        ]

    aicd_runs = _group("aicd")
    if aicd_runs:
        tasks = "/".join(t for t, _ in aicd_runs)
        val_str = " / ".join(_fmt(s.get("best_val_f1")) for _, s in aicd_runs)
        macro_str = " / ".join(_fmt(s.get("test_macro_f1")) for _, s in aicd_runs)
        weighted_str = " / ".join(_fmt(s.get("test_weighted_f1")) for _, s in aicd_runs)
        lines.append("**AICD-Bench row (Primary = Macro-F1):**")
        lines.append("```")
        lines.append(
            f"| {exp_id} | {method_name} | {tasks} | 3 | "
            f"{val_str} | {macro_str} | {weighted_str} | - | - | <notes> |"
        )
        lines.append("```")
        lines.append("")

    droid_runs = _group("droid")
    if droid_runs:
        tasks = "/".join(t for t, _ in droid_runs)
        val_str = " / ".join(_fmt(s.get("best_val_f1")) for _, s in droid_runs)
        weighted_str = " / ".join(_fmt(s.get("test_weighted_f1")) for _, s in droid_runs)
        macro_str = " / ".join(_fmt(s.get("test_macro_f1")) for _, s in droid_runs)
        lines.append("**DroidCollection row (Primary = Weighted-F1):**")
        lines.append("```")
        lines.append(
            f"| {exp_id} | {method_name} | {tasks} | 3 | "
            f"{val_str} | {weighted_str} | {macro_str} | <notes> |"
        )
        lines.append("```")
        lines.append("")

    codet_runs = [
        (task, stats)
        for (bench, task), stats in zip(run_plan, results.values())
        if bench.lower() in ("codet", "codet_m4", "codet-m4")
    ]
    if codet_runs:
        tasks = "/".join(t for t, _ in codet_runs)
        val_str = " / ".join(_fmt(s.get("best_val_f1")) for _, s in codet_runs)
        macro_str = " / ".join(_fmt(s.get("test_macro_f1")) for _, s in codet_runs)
        weighted_str = " / ".join(_fmt(s.get("test_weighted_f1")) for _, s in codet_runs)
        lines.append("**CoDET-M4 row (Primary = Macro-F1):**")
        lines.append("```")
        lines.append(
            f"| {exp_id} | {method_name} | {tasks} | 3 | "
            f"{val_str} | {macro_str} | {weighted_str} | <notes> |"
        )
        lines.append("```")
        lines.append("")

    # 4) Method Details stub
    lines.append("### Method Details Block (paste into `## Method Details` section)")
    lines.append("```markdown")
    lines.append(f"### {exp_id} - {method_name}")
    lines.append("")
    lines.append(f"**Run timestamp:** `{timestamp}`")
    lines.append("")
    lines.append(
        f"- Preflight + full suite completed on {gpu_name} {precision.upper()}: "
        + ", ".join(f"`{b}_{t}`" for b, t in run_plan)
    )
    if aicd_runs:
        aicd_test = ", ".join(f"`{t}={_fmt(s.get('test_macro_f1'))}`" for t, s in aicd_runs)
        lines.append(f"- AICD final test Macro-F1: {aicd_test}")
    if droid_runs:
        droid_test = ", ".join(f"`{t}={_fmt(s.get('test_macro_f1'))}`" for t, s in droid_runs)
        lines.append(f"- Droid final test Macro-F1: {droid_test}")
    if codet_runs:
        codet_test = ", ".join(f"`{t}={_fmt(s.get('test_macro_f1'))}`" for t, s in codet_runs)
        lines.append(f"- CoDET-M4 final test Macro-F1: {codet_test}")
    val_pairs = ", ".join(
        f"`{b.upper()}_{t}={_fmt(s.get('best_val_f1'))}`"
        for (b, t), s in zip(run_plan, results.values())
    )
    lines.append(f"- Best Val primary: {val_pairs}")
    lines.append("- Overall: <one-line takeaway>")
    lines.append("```")
    lines.append("")

    lines.append("=" * 78)
    lines.append("END_PAPER_TABLE")
    lines.append("=" * 78)

    for line in lines:
        logger.info(line)
