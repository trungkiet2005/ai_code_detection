"""Emit §1 leaderboard markdown from results/summary.json (maint helper)."""
from __future__ import annotations

import json
from pathlib import Path

BASE = 0.6633
ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "summary.json"


def track(files: list[str]) -> str:
    fn = files[0] if files else ""
    return "novel" if fn.startswith("exp_n") else "test"


def md_table(title: str, regime_key: str, skip_stub: bool = False) -> str:
    rows = [r for r in json.loads(SUMMARY.read_text(encoding="utf-8"))["rows"]
            if r["regime"] == regime_key]
    if skip_stub:
        rows = [r for r in rows if not any("exp_fs_99" in f for f in r["files"])]
    rows.sort(key=lambda x: -x["test_macro_f1_mean"])
    lines = [
        f"### {title}",
        "",
        "| Rank | Trk | Method | Test | d vs paper | Val | Gap | Canonical JSON |",
        "|:-:|:-:|:-:|-:|:-:|:-:|:-:|:--|",
    ]
    for i, r in enumerate(rows, 1):
        t = track(r["files"])
        test = r["test_macro_f1_mean"]
        d = test - BASE
        val = r["val_macro_f1_mean"]
        gap = r["val_test_gap_mean"]
        fn = r["files"][0]
        lines.append(
            f"| {i} | {t} | {r['method']} | {test:.4f} | {d:+.4f} | "
            f"{val:.4f} | {gap:+.4f} | `{fn}` |"
        )
    lines.append("")
    return "\n".join(lines)


def section1_markdown() -> str:
    intro = """## 1. Leaderboards

**Unified source:** `results/summary.json` from:

```bash
python Exp_FewShot/aggregate_fs_results.py results
```

**Column Trk:** `test` = `Exp_FewShot/testing/` (`exp_fs_*`); `novel` = `Exp_FewShot/novel/` (`exp_n*`). **d vs paper** = test Macro-F1 minus UniXcoder full-data **0.6633**.

**Coverage:** **104** JSON files ingested -> **87** `(method, regime)` cells in `summary.json`. Duplicate filenames `*__dup*.json` repeat the same metrics (ignore as extra seeds). **`exp_fs_00`** baseline CE (**0.1836**) appears only in Section 6 run history -- **no** `exp_fs_00_*.json` in `results/` yet. **`exp_fs_99_K128_seed42.json`** is an incomplete stub (`method=Test`) -- **excluded** from the `K=128` table below; delete or fix it for clean aggregates.

No JSON yet for scripts not listed under Canonical JSON (example: `exp_n20_*` if never run).

"""
    parts = [
        intro,
        md_table("Regime `frac=0.05` (~25K train, 3 epochs)", "frac=0.0500"),
        md_table("Regime `frac=0.01` (~5K train)", "frac=0.0100"),
        md_table("Regime `K=128` (~768 train)", "K=128", skip_stub=True),
        md_table("Regime `K=32` (~192 train)", "K=32"),
    ]
    return "\n".join(parts)


def main() -> None:
    print(section1_markdown())


if __name__ == "__main__":
    main()
