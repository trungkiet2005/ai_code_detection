"""
[run_zs_27_to_30] Launch 4 A*-level oral-grade detectors on Kaggle H100 80GB.

USAGE (Kaggle H100 80GB):
  !python Exp_ZeroShot/run_zs_27_to_30.py

Optimized for H100 80GB: 2 concurrent exps (safe VRAM), total wall ~40-50min.
4 breakthrough detectors: FrontDoor-NLP, Contrastive-Twin, Token-Entropy-Forks,
Semantic-Resilience. All target oral-level theoretical novelty.
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

# 4 A*-level oral-grade experiments, ordered by estimated VRAM cost (low->high)
EXP_FILES = [
    ("exp_zs_29_token_entropy_forks.py", "29_TokenEntForks", 10),      # ~10 min, light GPU
    ("exp_zs_30_semantic_resilience.py", "30_SemanticResilient", 8),   # ~8 min, light GPU
    ("exp_zs_28_contrastive_twin.py", "28_ContrastiveTwin", 12),       # ~12 min, medium GPU
    ("exp_zs_27_frontdoor_nlp.py", "27_FrontDoor", 15),                # ~15 min, medium GPU (HSIC)
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
    print("A*-LEVEL ORAL-GRADE DETECTORS RUNNER (exp_zs_27..30)")
    print("="*75)

    # Clone repo if not already here
    if not os.path.exists("Exp_ZeroShot"):
        print("[SETUP] Cloning ai_code_detection repo...")
        subprocess.check_call(["git", "clone", "--depth=1",
                              "https://github.com/trungkiet2005/ai_code_detection.git"])

    gpu_vram = get_gpu_memory_gb()
    print(f"[HW] Detected GPU VRAM: {gpu_vram:.1f} GB")
    print(f"[SUITE] 4 A*-level detectors:")
    print(f"        - FrontDoor-NLP (causal identification, NeurIPS theorem)")
    print(f"        - Contrastive-Twin (pair-divergence signal, AISec)")
    print(f"        - Token-Entropy-Forks (decision-point semantics, ACL)")
    print(f"        - Semantic-Resilience (robustness meta-signal, arXiv)")
    print(f"[MODE] H100 80GB -> Parallel mode: max_workers=2 (safe VRAM scheduling)")
    print("="*75)

    results = []

    # Parallel execution: max 2 concurrent exps
    with ThreadPoolExecutor(max_workers=2) as executor:
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
    with open("Exp_ZeroShot/run_zs_27_to_30_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[OK] Results saved to Exp_ZeroShot/run_zs_27_to_30_results.json")
    print(f"[READY] 30-detector suite complete. Paste console output into tracker update.")
