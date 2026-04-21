"""
[run_zs_21_to_26] Launch 6 more SOTA-paper-driven detectors in parallel on Kaggle H100 80GB.

USAGE (Kaggle H100 80GB):
  !python Exp_ZeroShot/run_zs_21_to_26.py

Optimized for H100 80GB: 3 concurrent exps (safe VRAM), total wall ~50-60min.
6 new detectors: task-conditioning, contrastive, KL-divergence, entropy-watermark,
syntactic-predictability, code-acrostic.
Each exp auto-detects H100 and uses batch_size=128, bf16, pin_memory=True.
"""
from __future__ import annotations

import os
import subprocess
import sys
import json
import time
import psutil
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

# 6 SOTA-paper-driven experiments, ordered by estimated VRAM cost (low->high)
EXP_FILES = [
    ("exp_zs_25_syntactic_predictability.py", "25_SyntacticPred", 6),      # ~6 min, no GPU
    ("exp_zs_24_entropy_watermark.py", "24_EntropyWatermark", 9),         # ~9 min, light GPU
    ("exp_zs_26_code_acrostic.py", "26_CodeAcrostic", 7),                 # ~7 min, light GPU
    ("exp_zs_21_task_conditioning.py", "21_TaskConditioning", 12),        # ~12 min, medium GPU
    ("exp_zs_23_kl_divergence.py", "23_KLDivergence", 10),                # ~10 min, medium GPU
    ("exp_zs_22_contrastive_hard_negatives.py", "22_ContrastiveHN", 13),  # ~13 min, medium GPU
]


def get_gpu_memory_gb() -> float:
    """Detect available GPU VRAM in GB."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    except Exception:
        pass
    return 0.0


def run_exp(exp_file: str, exp_id: str) -> dict:
    """Run a single experiment file and return metadata."""
    print(f"\n{'='*75}")
    print(f"[LAUNCH] {exp_id:20s} | {exp_file}")
    print(f"{'='*75}")

    start = time.time()
    mem_start = psutil.virtual_memory().percent
    try:
        result = subprocess.run(
            [sys.executable, f"ai_code_detection/Exp_ZeroShot/{exp_file}"],
            cwd=os.getcwd(),
            capture_output=False,
            text=True,
            timeout=1800,  # 30 min max per exp
        )
        elapsed = time.time() - start
        mem_peak = psutil.virtual_memory().percent
        status = "✓ PASS" if result.returncode == 0 else f"✗ FAIL (code {result.returncode})"
        return {
            "exp_id": exp_id,
            "file": exp_file,
            "status": status,
            "elapsed_sec": elapsed,
            "elapsed_min": f"{elapsed/60:.1f}",
            "mem_used_pct": f"{mem_peak - mem_start:.1f}%",
        }
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return {
            "exp_id": exp_id,
            "file": exp_file,
            "status": "✗ TIMEOUT (>30m)",
            "elapsed_sec": elapsed,
            "elapsed_min": f"{elapsed/60:.1f}",
            "mem_used_pct": "—",
        }
    except Exception as e:
        return {
            "exp_id": exp_id,
            "file": exp_file,
            "status": f"✗ ERROR: {str(e)[:40]}",
            "elapsed_sec": 0,
            "elapsed_min": "—",
            "mem_used_pct": "—",
        }


if __name__ == "__main__":
    print("\n" + "="*75)
    print("ZERO-SHOT SOTA-PAPER-DRIVEN EXPERIMENTS RUNNER (exp_zs_21..26)")
    print("="*75)

    # Clone repo if not already here
    if not os.path.exists("Exp_ZeroShot"):
        print("[SETUP] Cloning ai_code_detection repo...")
        subprocess.check_call(["git", "clone", "--depth=1",
                              "https://github.com/trungkiet2005/ai_code_detection.git"])

    gpu_vram = get_gpu_memory_gb()
    print(f"[HW] Detected GPU VRAM: {gpu_vram:.1f} GB")
    print(f"[SUITE] 6 SOTA-driven detectors: task-conditioning, contrastive, KL-div,")
    print(f"        entropy-watermark, syntactic-pred, code-acrostic")
    print(f"[MODE] H100 80GB -> Sequential (max_workers=1) — workers=3 caused 4/6 TIMEOUT in rerun 2 (2026-04-21)")
    print("="*75)

    results = []

    # Parallel execution: max 3 concurrent exps
    # Sequential: workers=3 caused 4/6 TIMEOUT in rerun 2 (2026-04-21)
    # when 4 processes (24,26,22,23) contended for VRAM on 106k-sample test set.
    # Even exps that PASSED in run 1 (24@7.2m, 26@7.7m) TIMEOUT >30m in run 2.
    with ThreadPoolExecutor(max_workers=1) as executor:
        futures = {}
        for exp_file, exp_id, est_min in EXP_FILES:
            future = executor.submit(run_exp, exp_file, exp_id)
            futures[future] = (exp_id, est_min)

        # Collect results as they complete
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            exp_id, est_min = futures[future]
            print(f"\n[DONE] {exp_id:20s} ({result['elapsed_min']:>5s}m, {result['mem_used_pct']:>8s})")

    # Sort results by exp_id for consistent output
    results.sort(key=lambda r: int(r["exp_id"].split("_")[0]))

    # Summary table
    print("\n" + "="*75)
    print("SUMMARY TABLE (H100 Parallel Run)")
    print("="*75)
    print(f"{'Exp ID':<25} | {'Status':<20} | {'Wall':<8} | {'Mem Delta':<8}")
    print("-"*75)
    for r in results:
        print(f"{r['exp_id']:<25} | {r['status']:<20} | {r['elapsed_min']:>6s}m | {r['mem_used_pct']:>6s}")

    total_sec = sum(r["elapsed_sec"] for r in results)
    wall_sec = max(r["elapsed_sec"] for r in results)
    print("-"*75)
    print(f"{'TOTAL (sum)':<25} | {'':<20} | {total_sec/60:>6.1f}m | —")
    print(f"{'WALL (max parallel)':<25} | {'':<20} | {wall_sec/60:>6.1f}m | —")
    print("="*75)

    # Save results to JSON for downstream analysis
    out_json = "ai_code_detection/Exp_ZeroShot/run_zs_21_to_26_results.json" \
               if os.path.exists("ai_code_detection/Exp_ZeroShot") \
               else "Exp_ZeroShot/run_zs_21_to_26_results.json"
    try:
        os.makedirs(os.path.dirname(out_json), exist_ok=True)
        with open(out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n[OK] Results saved to {out_json}")
    except Exception as e:
        print(f"\n[WARN] Could not save runner JSON: {e}")

    # Auto-aggregate: scan Exp_ZeroShot/results/*.json and print combined leaderboard.
    print("\n" + "="*75)
    print("[AGGREGATE] Scanning all per-exp result JSONs for combined leaderboard...")
    print("="*75)
    agg_script_candidates = [
        "ai_code_detection/Exp_ZeroShot/aggregate_results.py",
        "Exp_ZeroShot/aggregate_results.py",
    ]
    agg_script = next((p for p in agg_script_candidates if os.path.exists(p)), None)
    if agg_script is None:
        print("[AGGREGATE] aggregate_results.py not found — skipping combined table.")
    else:
        try:
            subprocess.run([sys.executable, agg_script], check=False)
        except Exception as e:
            print(f"[AGGREGATE] Failed to run aggregator: {e}")
