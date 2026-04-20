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


# ---------------------------------------------------------------------------
# Paper baselines to beat (strongest published primary-metric row per task).
#
# CoDET-M4 -- Orel, Azizov & Nakov (ACL Findings 2025). Macro-F1 primary.
#   - Table 2  binary IID             UniXcoder 98.65
#   - Table 7  author 6-class         UniXcoder 66.33
#   - Table 8  OOD generator (unseen) UniXcoder 93.22
#   - Table 9  OOD source  (unseen)   CodeT5    58.22   <-- NOT UniXcoder
#   - Table 12 OOD language (unseen)  UniXcoder 88.96
#
# DroidCollection -- Orel, Paul, Gurevych, Nakov (EMNLP 2025). Weighted-F1.
#   - Table 3  3-class per-domain Avg       DroidDetectCLS-Large 88.78
#   - Table 4  3-class per-language Avg     DroidDetectCLS-Large 93.66
#   - T1 (2-class) / T4 (4-class)           <-- not in either paper; flagged "—"
#
# Values are on the same scale as the trainer's test_f1 output (0-1 decimal,
# NOT percent). Each entry = (paper_value, source_string).
# ---------------------------------------------------------------------------
_PAPER_BASELINES: Dict[Tuple[str, str], Tuple[float, str]] = {
    # CoDET-M4 IID
    ("codet_m4", "iid_binary"):              (0.9865, "UniXcoder, Table 2"),
    ("codet_m4", "iid_author"):              (0.6633, "UniXcoder, Table 7"),
    # CoDET-M4 OOD Generator LOO -- paper headline is one averaged number,
    # not per-generator; listed here as the best avg any held-out can reach.
    ("codet_m4", "ood_generator_codellama"): (0.9322, "UniXcoder, Table 8 avg"),
    ("codet_m4", "ood_generator_gpt"):       (0.9322, "UniXcoder, Table 8 avg"),
    ("codet_m4", "ood_generator_llama3.1"):  (0.9322, "UniXcoder, Table 8 avg"),
    ("codet_m4", "ood_generator_nxcode"):    (0.9322, "UniXcoder, Table 8 avg"),
    ("codet_m4", "ood_generator_qwen1.5"):   (0.9322, "UniXcoder, Table 8 avg"),
    # CoDET-M4 OOD Language LOO -- Table 12 gives an avg; per-lang is Table 10.
    ("codet_m4", "ood_language_cpp"):        (0.9824, "UniXcoder, Table 3 (C++)"),
    ("codet_m4", "ood_language_java"):       (0.9902, "UniXcoder, Table 3 (Java)"),
    ("codet_m4", "ood_language_python"):     (0.9860, "UniXcoder, Table 3 (Python)"),
    # CoDET-M4 OOD Source LOO -- Table 4 per-source numbers (binary). Best row
    # per source: UniXcoder owns lc/gh, CodeT5 owns cf. For the author-level
    # subgroup we point to the same rows as soft references.
    ("codet_m4", "ood_source_cf"):           (0.9724, "CodeT5, Table 4 (CodeForces)"),
    ("codet_m4", "ood_source_lc"):           (0.9787, "UniXcoder, Table 4 (LeetCode)"),
    ("codet_m4", "ood_source_gh"):           (0.9854, "CodeT5, Table 4 (GitHub)"),
    # DroidCollection
    ("droid", "T1"):                         (None,   "—"),            # 2-class not in paper headline
    ("droid", "T3"):                         (0.8878, "DroidDetectCLS-Large, Table 3 Avg"),
    ("droid", "T4"):                         (None,   "— (first to report)"),
}


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

    # 2.5) CoDET-M4 breakdowns — per-language / per-source / per-generator
    # Maps directly to paper Tables 3/4/10/11 for binary and Figure 2 + per-gen
    # analysis for author. Prints ONLY when the trainer populated `breakdown`.
    codet_breakdown_runs = [
        (bench, task, stats)
        for (bench, task), stats in zip(run_plan, results.values())
        if bench.lower() in ("codet", "codet_m4", "codet-m4") and stats.get("breakdown")
    ]
    if codet_breakdown_runs:
        lines.append("### CoDET-M4 Subgroup Breakdown (paper Tables 3/4 + Author subgroups)")
        lines.append("")
        for bench, task, stats in codet_breakdown_runs:
            bd = stats.get("breakdown") or {}
            overall = bd.get("overall") or {}
            lines.append(
                f"#### {bench.upper()} {task} — overall macro={_fmt(overall.get('macro_f1'))} "
                f"weighted={_fmt(overall.get('weighted_f1'))}"
            )
            lines.append("")
            for dim_name, dim_label in (
                ("language", "language (paper Table 3 / Table 10 for OOD)"),
                ("source",   "source   (paper Table 4)"),
                ("generator","generator (author-only signal, Figure 2 analog)"),
            ):
                dim_res = bd.get(dim_name) or {}
                if not dim_res:
                    continue
                lines.append(f"**By {dim_label}:**")
                lines.append("")
                lines.append("| Group | n | Macro-F1 | Weighted-F1 |")
                lines.append("|:------|---:|:--------:|:-----------:|")
                for key, rec in sorted(dim_res.items(),
                                       key=lambda kv: kv[1].get("macro_f1", 0.0),
                                       reverse=True):
                    lines.append(
                        f"| {key} | {int(rec.get('n', 0))} | "
                        f"{_fmt(rec.get('macro_f1'))} | {_fmt(rec.get('weighted_f1'))} |"
                    )
                lines.append("")
            # Confusion matrix (author 6-class only) -- matches paper Figure 2
            cm = bd.get("confusion_matrix")
            if cm:
                lines.append("**Confusion matrix (rows=true, cols=pred) — compare to paper Figure 2:**")
                lines.append("")
                lines.append("```")
                for row in cm:
                    lines.append("  " + " ".join(f"{int(v):>6d}" for v in row))
                lines.append("```")
                lines.append("")

    # 2.6) Paper-anchor comparison — one-row-per-task delta vs the strongest
    # published baseline on that exact task. Sourced from `_PAPER_BASELINES`
    # below; mirrors the "SOTA targets" table in tracker.md.
    lines.append("### Δ vs. Paper Baselines (primary metric, strongest published number)")
    lines.append("")
    lines.append("> Paper OOD-LOO numbers are **binary detection** (human vs LLM). "
                 "Our lean-mode OOD-LOO runs are also binary (`force task=binary` in "
                 "`_run_single_loo`), BUT the held-out class is a single AI generator "
                 "(e.g. qwen1.5) so the test set is class-imbalanced (100% of one class). "
                 "Macro-F1 is ceiling-bound ~0.5 in that regime — use Weighted-F1 + "
                 "per-class Recall for OOD-Gen delta. Source/Language LOO tests are "
                 "more balanced and the macro delta is directly comparable.")
    lines.append("")
    lines.append("| Benchmark | Task | Our primary | Paper best | Source | Δ |")
    lines.append("|:----------|:----:|:-----------:|:----------:|:-------|:-:|")
    for (bench, task), stats in zip(run_plan, results.values()):
        ours = stats.get("test_f1")
        ref = _PAPER_BASELINES.get((bench.lower(), task))
        if ref is None:
            ref_val, ref_src = (None, "— (no paper baseline)")
        else:
            ref_val, ref_src = ref
        if ours is not None and ref_val is not None:
            delta = float(ours) - float(ref_val)
            delta_str = f"`{'+' if delta >= 0 else ''}{delta:.4f}`"
        else:
            delta_str = "—"
        ref_disp = _fmt(ref_val) if ref_val is not None else "—"
        lines.append(
            f"| {bench.upper()} | {task} | {_fmt(ours)} | "
            f"{ref_disp} | {ref_src} | {delta_str} |"
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
        # Droid paper-primary = Weighted-F1 (CLAUDE.md §5; Droid paper Tables 3–5
        # column header "weighted F1-score"). Macro-F1 shown as secondary.
        droid_test = ", ".join(f"`{t}={_fmt(s.get('test_weighted_f1'))}`" for t, s in droid_runs)
        lines.append(f"- Droid final test Weighted-F1 (paper-primary): {droid_test}")
        droid_macro = ", ".join(f"`{t}={_fmt(s.get('test_macro_f1'))}`" for t, s in droid_runs)
        lines.append(f"- Droid final test Macro-F1 (secondary): {droid_macro}")
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
