"""
[run_zs_11_to_16] Launch all 6 novel zero-shot methods in parallel on Kaggle H100 80GB.

USAGE (Kaggle H100 80GB):
  !python Exp_ZeroShot/run_zs_11_to_16.py

Optimized for H100 80GB: 3 concurrent exps (safe VRAM), total wall ~50-60min.
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

# 6 novel experiments, ordered by estimated VRAM cost (low→high)
# This ordering allows safe parallel execution
EXP_FILES = [
    ("exp_zs_16_ksd_scope.py", "16_KSDScope", 5),               # ~2 min, no GPU
    ("exp_zs_14_martingale_curvature.py", "14_Martingale", 10), # ~10 min, light GPU
    ("exp_zs_12_attention_criticality.py", "12_AttCrit", 12),   # ~12 min, light GPU
    ("exp_zs_11_path_signature.py", "11_PathSig", 15),          # ~15 min, medium GPU
    ("exp_zs_15_bures_quantum.py", "15_Bures", 11),             # ~11 min, medium GPU (sqrtm)
    ("exp_zs_13_sinkhorn_ot.py", "13_SinkhornOT", 18),          # ~18 min, heavy GPU (k^2 Sinkhorn)
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
            timeout=1800,  # 30 min max per exp (safe for H100)
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
    print("ZERO-SHOT NOVEL EXPERIMENTS RUNNER (exp_zs_11..16)")
    print("="*75)

    # Clone repo if not already here
    if not os.path.exists("Exp_ZeroShot"):
        print("[SETUP] Cloning ai_code_detection repo...")
        subprocess.check_call(["git", "clone", "--depth=1",
                              "https://github.com/trungkiet2005/ai_code_detection.git"])

    gpu_vram = get_gpu_memory_gb()
    print(f"[HW] Detected GPU VRAM: {gpu_vram:.1f} GB")
    print(f"[MODE] H100 80GB → Parallel mode: max_workers=3 (safe scheduling)")
    print("="*75)

    results = []

    # Sequential execution: max_workers=1 to avoid VRAM contention on heavy ops
    # (PathSig rank_proxy 12 GB, Sinkhorn cost matrix 12 GB, Bures density 25 GB,
    # AttCrit attention 6 GB). Earlier run with workers=3 OOMed 4/6 methods.
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
    print(f"{'Exp ID':<25} | {'Status':<20} | {'Wall':<8} | {'Mem Δ':<8}")
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
    with open("Exp_ZeroShot/run_zs_11_to_16_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to Exp_ZeroShot/run_zs_11_to_16_results.json")
    print(f"[READY] Next: paste console output above into tracker update")

