"""
aggregate_fs_results.py -- read every results/*.json and print a leaderboard.

Use after a Kaggle sweep (or after downloading the JSONs):
  python Exp_FewShot/aggregate_fs_results.py                           # ./results
  python Exp_FewShot/aggregate_fs_results.py /kaggle/working/results   # Kaggle
  python Exp_FewShot/aggregate_fs_results.py --csv  results            # also write summary.csv

Output:
  - A leaderboard sorted by test_macro_f1 DESC.
  - One row per (method, regime) pair; if multiple seeds exist for a cell,
    the row shows mean ± std and seed_count.
  - A `summary.json` and (optional) `summary.csv` next to the input dir.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import defaultdict


def _load_jsons(root: str):
    if not os.path.isdir(root):
        print(f"[aggregate] no such dir: {root}", file=sys.stderr)
        return []
    out = []
    for name in sorted(os.listdir(root)):
        if not name.endswith(".json") or name.startswith("summary"):
            continue
        path = os.path.join(root, name)
        try:
            with open(path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            out.append((path, payload))
        except (OSError, json.JSONDecodeError) as e:
            print(f"[aggregate] skip {path}: {e}", file=sys.stderr)
    return out


def _key(payload):
    """Group key = (method, regime_label)."""
    return payload["method"], payload["regime"]


def _stats(values):
    n = len(values)
    if n == 0:
        return None, None, 0
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, 1
    var = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(var), n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", nargs="?", default="./results",
                        help="directory containing exp_fs_*.json files")
    parser.add_argument("--csv", action="store_true",
                        help="also write summary.csv")
    args = parser.parse_args()

    runs = _load_jsons(args.dir)
    if not runs:
        print(f"[aggregate] no JSONs found in {args.dir}", file=sys.stderr)
        sys.exit(1)

    # Group: (method, regime) -> list of (test_f1, val_f1, gap, file)
    cells = defaultdict(list)
    for path, p in runs:
        try:
            test_f1 = float(p["results"]["test_macro_f1"])
            val_f1 = float(p["results"].get("val_macro_f1", 0.0))
            gap = float(p["results"].get("val_test_gap", 0.0))
        except (KeyError, TypeError, ValueError):
            print(f"[aggregate] skip (missing fields): {path}", file=sys.stderr)
            continue
        cells[_key(p)].append({"test_f1": test_f1, "val_f1": val_f1,
                                "gap": gap, "seed": p["config"]["fs_seed"],
                                "file": os.path.basename(path)})

    # Build summary rows.
    rows = []
    for (method, regime), entries in cells.items():
        test_vals = [e["test_f1"] for e in entries]
        mean_t, std_t, n = _stats(test_vals)
        rows.append({
            "method": method,
            "regime": regime,
            "n_seeds": n,
            "test_macro_f1_mean": mean_t,
            "test_macro_f1_std": std_t,
            "test_macro_f1_min": min(test_vals),
            "test_macro_f1_max": max(test_vals),
            "val_macro_f1_mean": _stats([e["val_f1"] for e in entries])[0],
            "val_test_gap_mean": _stats([e["gap"] for e in entries])[0],
            "files": [e["file"] for e in entries],
        })
    rows.sort(key=lambda r: r["test_macro_f1_mean"], reverse=True)

    # Pretty print.
    width = max((len(r["method"]) for r in rows), default=10) + 2
    print(f"\n{'method':<{width}}{'regime':<14}{'test':>10}{'± std':>10}"
          f"{'val':>10}{'gap':>10}{'n_seeds':>10}")
    print("-" * (width + 64))
    for r in rows:
        print(
            f"{r['method']:<{width}}{r['regime']:<14}"
            f"{r['test_macro_f1_mean']:>10.4f}"
            f"{r['test_macro_f1_std']:>10.4f}"
            f"{r['val_macro_f1_mean']:>10.4f}"
            f"{r['val_test_gap_mean']:>+10.4f}"
            f"{r['n_seeds']:>10d}"
        )
    print("-" * (width + 64))
    print(f"{len(runs)} runs across {len(cells)} cells")

    # Write summary.json (always) and summary.csv (--csv).
    summary_path = os.path.join(args.dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"rows": rows, "n_runs": len(runs)}, f, indent=2)
    print(f"[aggregate] wrote {summary_path}")

    if args.csv:
        csv_path = os.path.join(args.dir, "summary.csv")
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write("method,regime,n_seeds,test_macro_f1_mean,test_macro_f1_std,"
                    "test_macro_f1_min,test_macro_f1_max,val_macro_f1_mean,"
                    "val_test_gap_mean\n")
            for r in rows:
                f.write(
                    f"{r['method']},{r['regime']},{r['n_seeds']},"
                    f"{r['test_macro_f1_mean']:.4f},{r['test_macro_f1_std']:.4f},"
                    f"{r['test_macro_f1_min']:.4f},{r['test_macro_f1_max']:.4f},"
                    f"{r['val_macro_f1_mean']:.4f},{r['val_test_gap_mean']:+.4f}\n"
                )
        print(f"[aggregate] wrote {csv_path}")


if __name__ == "__main__":
    main()
