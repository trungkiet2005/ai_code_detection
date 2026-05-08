"""Zero-shot run harness.

API contract for each exp_zs_NN_*.py:

    from _zs_runner import run_zs_suite
    from _common import ZSConfig

    def score_fn(codes: list[str], cfg: ZSConfig) -> np.ndarray:
        # Return a per-sample score where HIGHER = more AI-like.
        ...

    run_zs_suite(
        method_name="MyZSMethod",
        exp_id="exp_zs_NN",
        score_fn=score_fn,
        cfg=ZSConfig(benchmark="droid_T3"),
    )

The harness:
  1. Loads dev + test (no train).
  2. Scores every sample via score_fn.
  3. Calibrates τ on dev to pin human recall at target (default 0.95).
  4. Reports test Macro-F1 + Weighted-F1 + human recall + adversarial recall
     + per-domain + per-language breakdown.
  5. Emits a paper-table-ready BEGIN_ZS_PAPER_TABLE markdown block.
"""
from __future__ import annotations

import time
from datetime import datetime
from typing import Callable, Dict, List

import numpy as np
from sklearn.metrics import f1_score, recall_score

from _common import ZSConfig, calibrate_threshold_at_human_recall, logger, set_seed, apply_hardware_profile
from _zs_loaders import load_zs


# Default benchmark set for oral-level ZS claims: Droid T3 (3-cls, headline
# baseline Fast-DetectGPT 64.54) + CoDET binary (cross-dataset transfer check).
# Droid T1 is optional -- binary ceiling-bound on both benches, little story.
ORAL_BENCHMARKS = ("droid_T3", "codet_binary")
FULL_BENCHMARKS = ("droid_T3", "droid_T1", "codet_binary")


def _binarize(labels: np.ndarray, human_label: int = 0) -> np.ndarray:
    return (labels != human_label).astype(int)


def _breakdown(
    preds: np.ndarray,
    labels: np.ndarray,
    rows,
    dim: str,
    min_n: int = 10,
) -> Dict[str, Dict[str, float]]:
    """Per-group Macro / Weighted F1 on a categorical dim.

    `rows` is a HF Dataset (iterable of dicts) or a list of dicts."""
    try:
        vals = rows[dim]
    except (KeyError, TypeError):
        return {}
    # HF Dataset returns a list; guarantee list-typed.
    vals = list(vals)
    out: Dict[str, Dict[str, float]] = {}
    unique = sorted(set(v for v in vals if v))
    for v in unique:
        mask = np.array([x == v for x in vals], dtype=bool)
        if mask.sum() < min_n:
            continue
        lbl_bin = _binarize(labels[mask])
        pr = preds[mask]
        macro = float(f1_score(lbl_bin, pr, average="macro", zero_division=0))
        weighted = float(f1_score(lbl_bin, pr, average="weighted", zero_division=0))
        out[str(v)] = {"n": int(mask.sum()), "macro_f1": macro, "weighted_f1": weighted}
    return out


def run_zs_suite(
    method_name: str,
    exp_id: str,
    score_fn: Callable[[List[str], ZSConfig], np.ndarray],
    cfg: ZSConfig,
) -> Dict[str, object]:
    """Run a zero-shot exp end-to-end. Returns a dict with keys:
    test_macro_f1, test_weighted_f1, test_human_recall, test_adversarial_recall,
    tau, dev_metrics, breakdown_*.
    """
    cfg = apply_hardware_profile(cfg)
    set_seed(cfg.seed)

    logger.info("\n" + "=" * 72)
    logger.info(f"ZERO-SHOT RUN | method={method_name} | exp={exp_id} | bench={cfg.benchmark}")
    logger.info("=" * 72)

    t0 = time.time()
    dev, test = load_zs(cfg.benchmark, cfg)

    # ---- Score dev ---------------------------------------------------------
    logger.info(f"[ZS] Scoring dev (n={len(dev)}) ...")
    dev_codes = list(dev["code"])
    dev_scores = score_fn(dev_codes, cfg)
    assert len(dev_scores) == len(dev), f"score_fn returned {len(dev_scores)} for {len(dev)} dev samples"
    dev_scores = np.asarray(dev_scores, dtype=np.float64)
    dev_labels = np.asarray(dev["label"], dtype=np.int64)

    tau, dev_metrics = calibrate_threshold_at_human_recall(
        dev_scores, dev_labels, cfg.human_recall_target, human_label=0
    )
    logger.info(f"[ZS] tau = {tau:.4f} | dev human recall = {dev_metrics['human_recall']:.4f}")

    # ---- Score test --------------------------------------------------------
    logger.info(f"[ZS] Scoring test (n={len(test)}) ...")
    test_codes = list(test["code"])
    test_scores = np.asarray(score_fn(test_codes, cfg), dtype=np.float64)
    test_labels = np.asarray(test["label"], dtype=np.int64)

    # Apply threshold to get binary predictions (0=human, 1=AI)
    test_preds = (test_scores >= tau).astype(int)

    # ---- Metrics -----------------------------------------------------------
    labels_bin = _binarize(test_labels)
    macro = float(f1_score(labels_bin, test_preds, average="macro", zero_division=0))
    weighted = float(f1_score(labels_bin, test_preds, average="weighted", zero_division=0))
    human_r = float(recall_score(labels_bin, test_preds, pos_label=0, zero_division=0))
    ai_r = float(recall_score(labels_bin, test_preds, pos_label=1, zero_division=0))

    # Adversarial recall: among rows where is_adversarial=1, did we predict AI?
    is_adv = np.asarray(test["is_adversarial"], dtype=np.int64)
    adv_r = float(test_preds[is_adv == 1].mean()) if is_adv.sum() > 0 else float("nan")

    logger.info("\n" + "=" * 60)
    logger.info("ZERO-SHOT TEST METRICS")
    logger.info("=" * 60)
    logger.info(f"  Macro-F1:            {macro:.4f}")
    logger.info(f"  Weighted-F1:         {weighted:.4f}")
    logger.info(f"  Human recall:        {human_r:.4f}  (target was {cfg.human_recall_target:.2f})")
    logger.info(f"  AI recall:           {ai_r:.4f}")
    logger.info(f"  Adversarial recall:  {adv_r:.4f}  (n_adv={int(is_adv.sum())})")

    # ---- Breakdown ---------------------------------------------------------
    bk: Dict[str, object] = {}
    for dim in ("domain", "language", "source_raw", "generator", "model_family"):
        bk[dim] = _breakdown(test_preds, test_labels, test, dim)
        if bk[dim]:
            logger.info(f"\n  Breakdown by {dim}:")
            for k, v in bk[dim].items():
                logger.info(f"    {k:>20s}: n={v['n']:>5d}  macro={v['macro_f1']:.4f}  weighted={v['weighted_f1']:.4f}")

    wall = time.time() - t0

    # ---- Paper-table emit --------------------------------------------------
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info("\n" + "=" * 72)
    logger.info("BEGIN_ZS_PAPER_TABLE")
    logger.info("=" * 72)
    logger.info(f"## {method_name} ({exp_id}) -- Zero-Shot Results")
    logger.info(f"**Timestamp:** `{ts}` | **Benchmark:** `{cfg.benchmark}` | **Wall:** `{wall:.0f}s`")
    logger.info("")
    logger.info("### Headline")
    logger.info("| Benchmark | Macro-F1 | Weighted-F1 | Human R | AI R | Adv R | tau |")
    logger.info("|:--|:-:|:-:|:-:|:-:|:-:|:-:|")
    logger.info(
        f"| {cfg.benchmark} | {macro:.4f} | {weighted:.4f} | "
        f"{human_r:.4f} | {ai_r:.4f} | {adv_r:.4f} | {tau:.4f} |"
    )
    logger.info("")
    logger.info("### Delta vs Droid Table 3/4/5 zero-shot baselines")
    # Paper Droid (EMNLP'25) Tables 3–5 report **Weighted-F1** for Droid T1/T3 —
    # per repo rule CLAUDE.md: Droid -> Weighted-F1. Report ours-vs-paper on the
    # same metric. CoDET follows Macro-F1 protocol (CLAUDE.md) — its deltas stay
    # macro below.
    logger.info("| Baseline | Bench | Paper metric | Paper F1 | Ours (same metric) | Delta |")
    logger.info("|:--|:-:|:-:|:-:|:-:|:-:|")
    if cfg.benchmark == "droid_T3":
        ours = weighted * 100  # Droid -> Weighted-F1
        logger.info(f"| Fast-DetectGPT (ZS) | T3 | W-F1 | 64.54 | {ours:.2f} | `{(ours - 64.54):+.2f}` |")
        logger.info(f"| M4 (ZS)             | T3 | W-F1 | 55.27 | {ours:.2f} | `{(ours - 55.27):+.2f}` |")
        logger.info(f"| GPTZero             | T3 | W-F1 | 49.10 | {ours:.2f} | `{(ours - 49.10):+.2f}` |")
        logger.info(f"| CoDet-M4 (ZS)       | T3 | W-F1 | 47.80 | {ours:.2f} | `{(ours - 47.80):+.2f}` |")
        logger.info(f"| GPTSniffer (ZS)     | T3 | W-F1 | 38.95 | {ours:.2f} | `{(ours - 38.95):+.2f}` |")
    elif cfg.benchmark == "droid_T1":
        ours = weighted * 100  # Droid -> Weighted-F1
        logger.info(f"| Fast-DetectGPT (ZS) | T1 | W-F1 | 67.85 | {ours:.2f} | `{(ours - 67.85):+.2f}` |")
        logger.info(f"| GPTZero             | T1 | W-F1 | 56.91 | {ours:.2f} | `{(ours - 56.91):+.2f}` |")
        logger.info(f"| M4 (ZS)             | T1 | W-F1 | 50.92 | {ours:.2f} | `{(ours - 50.92):+.2f}` |")
        logger.info(f"| CoDet-M4 (ZS)       | T1 | W-F1 | 54.49 | {ours:.2f} | `{(ours - 54.49):+.2f}` |")
        logger.info(f"| GPTSniffer (ZS)     | T1 | W-F1 | 41.07 | {ours:.2f} | `{(ours - 41.07):+.2f}` |")
    elif cfg.benchmark == "codet_binary":
        ours = macro * 100  # CoDET -> Macro-F1 (CLAUDE.md)
        logger.info(f"| Fast-DetectGPT (ZS) | CoDET bin | Macro-F1 | 62.03 | {ours:.2f} | `{(ours - 62.03):+.2f}` |")
    logger.info("")
    logger.info("=" * 72)
    logger.info("END_ZS_PAPER_TABLE")
    logger.info("=" * 72)

    return {
        "method": method_name,
        "exp_id": exp_id,
        "benchmark": cfg.benchmark,
        "tau": tau,
        "dev_metrics": dev_metrics,
        "test_macro_f1": macro,
        "test_weighted_f1": weighted,
        "test_human_recall": human_r,
        "test_ai_recall": ai_r,
        "test_adversarial_recall": adv_r,
        "breakdown": bk,
        "wall_time_s": wall,
    }


# -----------------------------------------------------------------------------
# Dual-benchmark driver
# -----------------------------------------------------------------------------

def run_zs_oral(
    method_name: str,
    exp_id: str,
    score_fn: Callable[[List[str], ZSConfig], np.ndarray],
    cfg: ZSConfig,
    benchmarks=ORAL_BENCHMARKS,
) -> Dict[str, Dict]:
    """Run one zero-shot exp across multiple benchmarks and emit a combined
    oral-level paper table. Default runs Droid T3 + CoDET binary.

    Each benchmark uses a fresh copy of `cfg` with cfg.benchmark overridden.
    """
    import dataclasses as _dc
    results: Dict[str, Dict] = {}

    for bench in benchmarks:
        logger.info("\n" + "#" * 78)
        logger.info(f"# [ZS-ORAL] {method_name} -- benchmark={bench}")
        logger.info("#" * 78)
        per_cfg = _dc.replace(cfg, benchmark=bench)
        try:
            results[bench] = run_zs_suite(
                method_name=method_name,
                exp_id=exp_id,
                score_fn=score_fn,
                cfg=per_cfg,
            )
        except Exception as e:
            logger.error(f"[ZS-ORAL] {bench} failed: {e}")
            results[bench] = {"error": str(e)}

    # Combined oral-level table
    logger.info("\n" + "=" * 78)
    logger.info("BEGIN_ZS_ORAL_TABLE")
    logger.info("=" * 78)
    logger.info(f"## {method_name} ({exp_id}) -- Dual-Benchmark Zero-Shot Summary")
    logger.info("")
    logger.info("| Benchmark | Macro-F1 | Weighted-F1 | Human R | Adv R | tau | Wall |")
    logger.info("|:--|:-:|:-:|:-:|:-:|:-:|:-:|")
    for bench, r in results.items():
        if "error" in r:
            logger.info(f"| {bench} | ERROR: {r['error'][:40]} | - | - | - | - | - |")
            continue
        logger.info(
            f"| {bench} | {r['test_macro_f1']:.4f} | {r['test_weighted_f1']:.4f} | "
            f"{r['test_human_recall']:.4f} | {r.get('test_adversarial_recall', float('nan')):.4f} | "
            f"{r['tau']:.4f} | {r['wall_time_s']:.0f}s |"
        )
    logger.info("")
    logger.info("### Oral-claim checks")
    logger.info("> Primary metric per CLAUDE.md: Droid -> **Weighted-F1**, CoDET -> **Macro-F1**.")
    # Claim 1: Droid T3 Weighted-F1 > Fast-DetectGPT 64.54 (paper Table 3 Avg row)
    if "droid_T3" in results and "test_weighted_f1" in results["droid_T3"]:
        d = results["droid_T3"]["test_weighted_f1"] * 100
        delta = d - 64.54
        verdict = "PASS" if delta > 0 else "FAIL"
        logger.info(f"- Beat Fast-DetectGPT T3 (W-F1 64.54): **{d:.2f}** ({delta:+.2f}) -- **{verdict}**")
    # Claim 2: Human recall >= 0.95 on both benches
    for bench, r in results.items():
        if "test_human_recall" in r:
            hr = r["test_human_recall"]
            verdict = "PASS" if hr >= 0.95 else "FAIL"
            logger.info(f"- Human recall >= 0.95 on {bench}: **{hr:.4f}** -- **{verdict}**")
    # Claim 3: Cross-benchmark transfer — use each bench's PRIMARY metric
    # (Droid W-F1 vs CoDET Macro-F1). Mixing metrics is intentional: each is
    # the paper-standard primary for that benchmark (CLAUDE.md §5).
    if "droid_T3" in results and "codet_binary" in results:
        r1 = results["droid_T3"].get("test_weighted_f1")   # Droid primary
        r2 = results["codet_binary"].get("test_macro_f1")  # CoDET primary
        if r1 is not None and r2 is not None:
            gap = abs(r1 - r2) * 100
            verdict = "PASS" if gap < 10 else "FAIL"
            logger.info(f"- Cross-benchmark stability (|Droid W-F1 - CoDET Macro-F1| < 10 pt): **{gap:.2f}** -- **{verdict}**")

    logger.info("")
    logger.info("=" * 78)
    logger.info("END_ZS_ORAL_TABLE")
    logger.info("=" * 78)

    # Persist per-exp result to JSON so `aggregate_results.py` can rebuild
    # the full cross-exp leaderboard even if each exp was launched separately
    # (user runs files individually, not via a single runner script).
    try:
        import json as _json
        import os as _os

        # Find a writeable results dir; prefer repo-local Exp_ZeroShot/results/
        candidates = []
        try:
            here = _os.path.dirname(_os.path.abspath(__file__))
            candidates.append(_os.path.join(here, "results"))
        except NameError:
            pass
        candidates.append(_os.path.join(_os.getcwd(), "Exp_ZeroShot", "results"))
        candidates.append(_os.path.join(_os.getcwd(), "ai_code_detection", "Exp_ZeroShot", "results"))

        out_dir = None
        for c in candidates:
            try:
                _os.makedirs(c, exist_ok=True)
                out_dir = c
                break
            except OSError:
                continue
        if out_dir is not None:
            payload = {"method": method_name, "exp_id": exp_id, "results": {}}
            for bench, r in results.items():
                if not isinstance(r, dict) or "error" in r:
                    payload["results"][bench] = {"error": r.get("error", "unknown") if isinstance(r, dict) else "no_dict"}
                    continue
                payload["results"][bench] = {
                    "test_macro_f1": r.get("test_macro_f1"),
                    "test_weighted_f1": r.get("test_weighted_f1"),
                    "test_human_recall": r.get("test_human_recall"),
                    "test_ai_recall": r.get("test_ai_recall"),
                    "test_adversarial_recall": r.get("test_adversarial_recall"),
                    "tau": r.get("tau"),
                    "wall_time_s": r.get("wall_time_s"),
                }
            out_path = _os.path.join(out_dir, f"{exp_id}.json")
            with open(out_path, "w") as _f:
                _json.dump(payload, _f, indent=2)
            logger.info(f"[persist] Saved {out_path}")
    except Exception as _e:
        logger.warning(f"[persist] Failed to save result JSON: {_e}")

    return results
