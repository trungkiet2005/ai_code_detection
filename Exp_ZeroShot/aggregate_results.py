"""[aggregate_results] One-stop aggregator for all exp_zs_NN results.

USAGE (Kaggle H100 or local, after running individual exps):
  !python ai_code_detection/Exp_ZeroShot/aggregate_results.py

WHAT IT DOES:
  1. Scans Exp_ZeroShot/results/*.json (one per exp, written by _zs_runner).
  2. Emits a SINGLE combined leaderboard sorted by Droid T3 Weighted-F1
     (paper-primary per CLAUDE.md §5).
  3. Also prints:
     - CoDET binary Macro-F1 leaderboard (CoDET paper-primary)
     - Stability ranking |Droid W-F1 - CoDET Macro-F1|
     - Delta-vs-FDG-ours (own reproduction baseline) for both benches
     - Delta-vs-paper for Droid T3 (Fast-DetectGPT 64.54 W-F1)
     - Oral pass-gate check (3 claims)
  4. Emits a tracker-ready markdown table block that can be copy-pasted
     directly into Exp_ZeroShot/tracker.md.

WHY THIS EXISTS:
  User runs each exp_zs_NN_*.py individually (not via runners), so each
  Kaggle cell prints only its own log. This script scans all persisted
  result JSONs and rebuilds the full cross-exp leaderboard.

METRIC POLICY (CLAUDE.md §5, Droid paper Tables 3-5, CoDET paper Tables 2/7):
  - Droid T3        -> PRIMARY = Weighted-F1 (paper "weighted F1-score")
  - CoDET binary    -> PRIMARY = Macro-F1    (paper col "F" = Macro-F1)
  - Never compare Droid W-F1 against CoDET Macro-F1 directly as "the same
    metric"; each is its own benchmark's primary.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Paper baselines (authoritative quotes from 3 papers)
# ---------------------------------------------------------------------------

# Droid (Orel et al. EMNLP'25) Table 3 Avg row, 3-class, Weighted-F1.
DROID_T3_PAPER_WF1 = {
    "Fast-DetectGPT": 64.54,
    "M4 (ZS)":        55.27,
    "GPTZero":        49.10,
    "CoDet-M4 (ZS)":  47.80,
    "GPTSniffer":     38.95,
    "DroidDetectCLS-Base": 86.76,   # Full-training ceiling (Table 3 Avg)
    "DroidDetectCLS-Large": 88.78,  # Full-training ceiling (Table 3 Avg)
}

# CoDET-M4 (Orel et al. ACL'25). Macro-F1 ("F" column in Table 2).
# Paper's binary FDG ZS baseline.
CODET_BINARY_PAPER_MACROF1 = {
    "Fast-DetectGPT (paper)": 62.03,
}

# ---------------------------------------------------------------------------
# Result discovery
# ---------------------------------------------------------------------------

def _find_results_dir() -> Optional[Path]:
    """Locate Exp_ZeroShot/results/ regardless of CWD."""
    cwd = Path.cwd()
    candidates = [
        cwd / "Exp_ZeroShot" / "results",
        cwd / "ai_code_detection" / "Exp_ZeroShot" / "results",
    ]
    try:
        here = Path(__file__).resolve().parent
        candidates.insert(0, here / "results")
    except NameError:
        pass
    for c in candidates:
        if c.exists() and c.is_dir():
            return c
    return None


def _load_all_results(results_dir: Path) -> List[Dict]:
    """Load every <exp_id>.json under results_dir."""
    payloads = []
    for jf in sorted(results_dir.glob("exp_zs_*.json")):
        try:
            with open(jf) as f:
                payload = json.load(f)
            payloads.append(payload)
        except Exception as e:
            print(f"[skip] Failed to parse {jf}: {e}")
    return payloads


# ---------------------------------------------------------------------------
# Metric extraction + formatting
# ---------------------------------------------------------------------------

def _extract_primary(payload: Dict, bench: str, metric: str) -> Optional[float]:
    """Return the bench's primary metric (float 0–1) or None if error/missing."""
    br = payload.get("results", {}).get(bench)
    if not isinstance(br, dict) or "error" in br:
        return None
    v = br.get(metric)
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _fmt_f1(x: Optional[float]) -> str:
    return f"{x * 100:.2f}" if x is not None else "—"


def _fmt_delta(ours: Optional[float], paper: float) -> str:
    if ours is None:
        return "—"
    return f"{(ours * 100 - paper):+.2f}"


# ---------------------------------------------------------------------------
# Main aggregation
# ---------------------------------------------------------------------------

def main() -> int:
    results_dir = _find_results_dir()
    if results_dir is None:
        print("ERROR: Could not find Exp_ZeroShot/results/ directory.")
        print("  Looked under cwd, __file__/results, ai_code_detection/Exp_ZeroShot/results.")
        print("  Run at least one exp_zs_NN_*.py first to populate results.")
        return 1

    payloads = _load_all_results(results_dir)
    if not payloads:
        print(f"ERROR: No result JSONs found in {results_dir}")
        return 1

    print("\n" + "=" * 100)
    print(f"ZERO-SHOT AGGREGATE LEADERBOARD — {len(payloads)} exps discovered in {results_dir}")
    print("=" * 100)

    # Build flat rows
    rows = []
    for p in payloads:
        exp_id = p.get("exp_id", "?")
        method = p.get("method", "?")
        droid_wf1 = _extract_primary(p, "droid_T3", "test_weighted_f1")
        droid_mf1 = _extract_primary(p, "droid_T3", "test_macro_f1")
        codet_mf1 = _extract_primary(p, "codet_binary", "test_macro_f1")
        codet_wf1 = _extract_primary(p, "codet_binary", "test_weighted_f1")
        droid_hr = _extract_primary(p, "droid_T3", "test_human_recall")
        codet_hr = _extract_primary(p, "codet_binary", "test_human_recall")
        droid_adv = _extract_primary(p, "droid_T3", "test_adversarial_recall")
        droid_wall = _extract_primary(p, "droid_T3", "wall_time_s")
        codet_wall = _extract_primary(p, "codet_binary", "wall_time_s")

        stability = None
        if droid_wf1 is not None and codet_mf1 is not None:
            stability = abs(droid_wf1 - codet_mf1) * 100

        rows.append({
            "exp_id": exp_id,
            "method": method,
            "droid_wf1": droid_wf1,
            "droid_mf1": droid_mf1,
            "codet_mf1": codet_mf1,
            "codet_wf1": codet_wf1,
            "droid_hr": droid_hr,
            "codet_hr": codet_hr,
            "droid_adv": droid_adv,
            "stability": stability,
            "wall_s": (droid_wall or 0) + (codet_wall or 0),
        })

    # ------ Table 1: Droid T3 Weighted-F1 leaderboard (paper-primary) ------
    print("\n### TABLE 1 — Droid T3 Weighted-F1 leaderboard (paper-primary per CLAUDE.md §5)\n")
    print(f"{'Rank':<5}| {'Exp':<14}| {'Method':<32}| {'W-F1':<7}| {'Macro':<7}| "
          f"{'Δ-FDG-ours':<11}| {'Δ-paper':<9}| {'HR-D':<7}| {'Adv-R':<7}")
    print("-" * 110)
    fdg_row = next((r for r in rows if r["exp_id"] == "exp_zs_02"), None)
    fdg_own_wf1 = fdg_row["droid_wf1"] if fdg_row else None

    sorted_droid = sorted(rows, key=lambda r: (r["droid_wf1"] is None, -(r["droid_wf1"] or 0)))
    for i, r in enumerate(sorted_droid, 1):
        if r["droid_wf1"] is None:
            continue
        delta_own = f"{(r['droid_wf1']*100 - (fdg_own_wf1 or 0)*100):+.2f}" if fdg_own_wf1 else "—"
        delta_paper = _fmt_delta(r["droid_wf1"], DROID_T3_PAPER_WF1["Fast-DetectGPT"])
        print(f"{i:<5}| {r['exp_id']:<14}| {r['method'][:32]:<32}| "
              f"{_fmt_f1(r['droid_wf1']):<7}| {_fmt_f1(r['droid_mf1']):<7}| "
              f"{delta_own:<11}| {delta_paper:<9}| "
              f"{(f'{r[\"droid_hr\"]:.3f}' if r['droid_hr'] is not None else '—'):<7}| "
              f"{(f'{r[\"droid_adv\"]:.3f}' if r['droid_adv'] is not None else '—'):<7}")

    # ------ Table 2: CoDET binary Macro-F1 leaderboard (paper-primary) ------
    print("\n### TABLE 2 — CoDET binary Macro-F1 leaderboard (paper-primary per CLAUDE.md §5)\n")
    print(f"{'Rank':<5}| {'Exp':<14}| {'Method':<32}| {'Macro':<7}| {'W-F1':<7}| "
          f"{'Δ-FDG-paper':<12}| {'HR-C':<7}")
    print("-" * 95)
    fdg_paper_codet = CODET_BINARY_PAPER_MACROF1["Fast-DetectGPT (paper)"]
    sorted_codet = sorted(rows, key=lambda r: (r["codet_mf1"] is None, -(r["codet_mf1"] or 0)))
    for i, r in enumerate(sorted_codet, 1):
        if r["codet_mf1"] is None:
            continue
        delta_paper = _fmt_delta(r["codet_mf1"], fdg_paper_codet)
        print(f"{i:<5}| {r['exp_id']:<14}| {r['method'][:32]:<32}| "
              f"{_fmt_f1(r['codet_mf1']):<7}| {_fmt_f1(r['codet_wf1']):<7}| "
              f"{delta_paper:<12}| "
              f"{(f'{r[\"codet_hr\"]:.3f}' if r['codet_hr'] is not None else '—'):<7}")

    # ------ Table 3: Cross-bench stability ------
    print("\n### TABLE 3 — Cross-benchmark stability (|Droid W-F1 − CoDET Macro-F1|, lower is better)\n")
    print(f"{'Rank':<5}| {'Exp':<14}| {'Method':<32}| {'Gap (pt)':<10}| {'Droid W-F1':<11}| {'CoDET Macro':<11}")
    print("-" * 90)
    stable = [r for r in rows if r["stability"] is not None]
    stable.sort(key=lambda r: r["stability"])
    for i, r in enumerate(stable, 1):
        print(f"{i:<5}| {r['exp_id']:<14}| {r['method'][:32]:<32}| "
              f"{r['stability']:<10.2f}| {_fmt_f1(r['droid_wf1']):<11}| {_fmt_f1(r['codet_mf1']):<11}")

    # ------ Oral pass-gate checks ------
    print("\n### TABLE 4 — Oral pass-gate checks (3 claims)\n")
    print(f"{'Exp':<14}| {'Claim1: W-F1>64.54':<22}| {'Claim2a: HR-D≥0.95':<22}| "
          f"{'Claim2b: HR-C≥0.95':<22}| {'Claim3: gap<10pt':<20}| {'Pass All'}")
    print("-" * 115)
    for r in rows:
        if r["droid_wf1"] is None and r["codet_mf1"] is None:
            continue
        c1 = r["droid_wf1"] is not None and r["droid_wf1"] * 100 > 64.54
        c2a = r["droid_hr"] is not None and r["droid_hr"] >= 0.95
        c2b = r["codet_hr"] is not None and r["codet_hr"] >= 0.95
        c3 = r["stability"] is not None and r["stability"] < 10
        all_pass = c1 and c2a and c2b and c3
        def _v(b): return "✓" if b else "✗"
        v1 = f"{_fmt_f1(r['droid_wf1'])} {_v(c1)}"
        v2a = f"{r['droid_hr']:.3f} {_v(c2a)}" if r['droid_hr'] is not None else "— —"
        v2b = f"{r['codet_hr']:.3f} {_v(c2b)}" if r['codet_hr'] is not None else "— —"
        v3 = f"{r['stability']:.2f} {_v(c3)}" if r['stability'] is not None else "— —"
        print(f"{r['exp_id']:<14}| {v1:<22}| {v2a:<22}| {v2b:<22}| {v3:<20}| {_v(all_pass)}")

    # ------ Tracker-ready markdown block ------
    print("\n" + "=" * 100)
    print("TRACKER-READY MARKDOWN (copy-paste into Exp_ZeroShot/tracker.md):")
    print("=" * 100)
    print("```markdown")
    print("| Rank | Exp | Method | **Droid T3 W-F1** | Droid Macro | **CoDET Macro-F1** | Human R (D/C) | Adv R (D) | Status |")
    print("|:-:|:--|:--|:-:|:-:|:-:|:-:|:-:|:-:|")
    for i, r in enumerate(sorted_droid, 1):
        if r["droid_wf1"] is None and r["codet_mf1"] is None:
            continue
        hr_pair = (f"{r['droid_hr']:.3f}" if r['droid_hr'] is not None else "—") + " / " + \
                  (f"{r['codet_hr']:.3f}" if r['codet_hr'] is not None else "—")
        adv = f"{r['droid_adv']:.3f}" if r['droid_adv'] is not None else "—"
        delta_own = ""
        if fdg_own_wf1 is not None and r['droid_wf1'] is not None:
            d = r['droid_wf1']*100 - fdg_own_wf1*100
            delta_own = f"Δ-FDG-ours **{d:+.2f}**"
        codet_mf1_s = _fmt_f1(r['codet_mf1'])
        print(f"| {i} | {r['exp_id']} | {r['method']} | "
              f"**{_fmt_f1(r['droid_wf1'])}** | {_fmt_f1(r['droid_mf1'])} | "
              f"**{codet_mf1_s}** | {hr_pair} | {adv} | {delta_own} |")
    print("```")

    # ------ JSON dump ------
    out_json = results_dir.parent / "aggregate_leaderboard.json"
    with open(out_json, "w") as f:
        json.dump({"rows": rows, "papers": {"droid_t3_wf1": DROID_T3_PAPER_WF1,
                                            "codet_binary_mf1": CODET_BINARY_PAPER_MACROF1}}, f, indent=2)
    print(f"\n[OK] Aggregate JSON saved: {out_json}")
    print(f"[OK] Processed {len(rows)} exp results.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
