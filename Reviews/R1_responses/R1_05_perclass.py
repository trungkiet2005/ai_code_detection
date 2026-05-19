# R1_05_perclass — Per-class F1 + class-imbalance diagnostic (W7/Q3)
# =============================================================================
# Reviewer Q3: "The AICD-T2 results exhibit large Macro vs. Weighted F1 gaps.
# Are classes heavily imbalanced in your splits, and do per-class F1s show
# improvements across the board?"
#
# Diagnostic-only file. Loads exp76_traco_results.json and exp65_abl
# (CE-only baseline) results, dumps per-class F1/precision/recall/support
# breakdown for AICD-T2 at all three fractions, computes class-support
# distribution, and reports the difference between TRACO and CE-only
# per-class to show whether the gain is concentrated on one class or
# spread across many.
# =============================================================================
from __future__ import annotations
import os, sys, json
from collections import defaultdict
import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
RES_DIR = os.path.join(REPO_ROOT, "Exp_FewShot", "testing_chis", "results")
OUT_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT_DIR, exist_ok=True)


def load_method(name):
    path = os.path.join(RES_DIR, f"{name}_results.json")
    if not os.path.exists(path):
        print(f"WARN: {path} missing")
        return []
    with open(path) as f: return json.load(f)


def support_from_cm(cm):
    """Class support = row sum of confusion matrix (true labels)."""
    cm = np.array(cm)
    return cm.sum(axis=1).tolist()


def per_class_table(method_name, results, bench):
    out = {}
    for r in results:
        if r.get("bench") != bench: continue
        frac = r["frac"]
        tm = r.get("test_metrics", {})
        per_class = tm.get("per_class", {})
        cm = tm.get("confusion_matrix", [])
        support = support_from_cm(cm) if cm else []
        f1_list = per_class.get("f1", [])
        p_list = per_class.get("precision", [])
        r_list = per_class.get("recall", [])
        out[round(frac, 2)] = {
            "macro_f1": tm.get("overall", {}).get("macro_f1"),
            "weighted_f1": tm.get("overall", {}).get("weighted_f1"),
            "f1": f1_list, "precision": p_list, "recall": r_list,
            "support": support,
        }
    return out


def main():
    traco = load_method("exp76_traco")
    abl = load_method("exp65_abl")        # CE-only baseline at 20% only

    report = {"meta": {"reviewer_concern": "W7 / Q3", "benchmark": "AICD-T2"}}

    # TRACO per-class breakdown at all three fractions
    traco_pc = per_class_table("TRACO", traco, "aicd_t2")
    report["traco_aicd_per_class"] = traco_pc

    # Class-imbalance summary
    if traco_pc:
        # Use the 20pct support (full test set) as the canonical class-size
        # distribution since test is shared across fractions.
        support = traco_pc.get(0.2, {}).get("support", [])
        if support:
            tot = sum(support)
            report["class_support"] = {
                "raw": support,
                "fraction": [round(s / max(1, tot), 4) for s in support],
                "max_min_ratio": float(max(support) / max(1, min(support))) if min(support) > 0 else None,
                "gini": float(_gini(support)),
            }

    # If we have CE-only AICD-T2 at 20pct, compute per-class lift
    ce_aicd = [r for r in abl if r.get("bench") == "aicd_t2" and abs(r.get("frac", -1) - 0.20) < 1e-6]
    if ce_aicd and traco_pc.get(0.2):
        # Take CE-only row tagged as "CE_only" if available; else first row
        ce_row = None
        for r in ce_aicd:
            tag = r.get("tag", "").lower()
            if "ce_only" in tag or "ce-only" in tag or "ce only" in tag:
                ce_row = r; break
        if ce_row is None: ce_row = ce_aicd[0]
        ce_pc = ce_row.get("test_metrics", {}).get("per_class", {}).get("f1", [])
        traco_pc20 = traco_pc[0.2].get("f1", [])
        if ce_pc and traco_pc20 and len(ce_pc) == len(traco_pc20):
            lift = [round(t - c, 4) for t, c in zip(traco_pc20, ce_pc)]
            report["per_class_lift_traco_vs_ce_20pct"] = {
                "ce_only": ce_pc, "traco": traco_pc20, "lift": lift,
                "n_classes_positive_lift": int(sum(1 for v in lift if v > 0)),
                "n_classes_negative_lift": int(sum(1 for v in lift if v < 0)),
                "mean_lift": float(np.mean(lift)), "max_lift": float(max(lift)),
                "min_lift": float(min(lift)),
            }

    # Macro vs Weighted F1 gap summary
    report["macro_vs_weighted_gap"] = {}
    for frac, d in traco_pc.items():
        m = d.get("macro_f1"); w = d.get("weighted_f1")
        if m is not None and w is not None:
            report["macro_vs_weighted_gap"][str(frac)] = {
                "macro": round(m, 4), "weighted": round(w, 4),
                "gap_w_minus_m": round(w - m, 4),
            }

    # Write
    path = os.path.join(OUT_DIR, "R1_05_perclass.json")
    with open(path, "w") as f: json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
    print(f"\nWrote {path}")


def _gini(xs):
    """Class-support Gini coefficient. 0 = uniform, 1 = one class dominates."""
    xs = sorted(xs)
    n = len(xs); s = sum(xs)
    if n == 0 or s == 0: return 0.0
    cum = 0.0
    for i, x in enumerate(xs, start=1):
        cum += i * x
    return (2 * cum / (n * s)) - (n + 1) / n


if __name__ == "__main__":
    main()
