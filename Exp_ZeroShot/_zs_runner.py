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
    logger.info("| Baseline | Bench | Macro-F1 | Delta (ours-paper) |")
    logger.info("|:--|:-:|:-:|:-:|")
    # Paper Table 3 zero-shot Avg rows (3-class)
    if cfg.benchmark == "droid_T3":
        logger.info(f"| Fast-DetectGPT (ZS) | T3 | 64.54 | `{(macro*100 - 64.54):+.2f}` |")
        logger.info(f"| M4 (ZS)             | T3 | 55.27 | `{(macro*100 - 55.27):+.2f}` |")
        logger.info(f"| GPTZero             | T3 | 49.10 | `{(macro*100 - 49.10):+.2f}` |")
        logger.info(f"| CoDet-M4 (ZS)       | T3 | 47.80 | `{(macro*100 - 47.80):+.2f}` |")
        logger.info(f"| GPTSniffer (ZS)     | T3 | 38.95 | `{(macro*100 - 38.95):+.2f}` |")
    elif cfg.benchmark == "droid_T1":
        logger.info(f"| Fast-DetectGPT (ZS) | T1 | 67.85 | `{(macro*100 - 67.85):+.2f}` |")
        logger.info(f"| GPTZero             | T1 | 56.91 | `{(macro*100 - 56.91):+.2f}` |")
        logger.info(f"| M4 (ZS)             | T1 | 50.92 | `{(macro*100 - 50.92):+.2f}` |")
        logger.info(f"| CoDet-M4 (ZS)       | T1 | 54.49 | `{(macro*100 - 54.49):+.2f}` |")
        logger.info(f"| GPTSniffer (ZS)     | T1 | 41.07 | `{(macro*100 - 41.07):+.2f}` |")
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
