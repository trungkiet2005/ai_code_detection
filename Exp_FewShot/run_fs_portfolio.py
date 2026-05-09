"""
Portfolio runner — fire every (method, setup) combo and dump logs.

Use this on Kaggle T4 instead of running each exp_fs_*.py one by one.
Sweeps:
  - 4 methods x 4 K values x 3 fractions = up to 28 runs (some redundant).
  - Hits the K=32 baseline numbers we already have plus the unknown cells.
  - Writes per-run BEGIN_FS_TABLE blocks to logs/<method>_<setup>.log.

Default sweep is conservative (K in {32, 128} + fraction in {0.01, 0.05}).
Override via env vars to expand.

  python Exp_FewShot/run_fs_portfolio.py                   # default sweep
  FS_METHODS=baseline,ntkalign python Exp_FewShot/run_fs_portfolio.py
  FS_KS=8,16,32,64,128 FS_FRACTIONS=0.01,0.05,0.1 python ... .py

Log layout:
  logs/exp_fs_<method_idx>_<setup_label>.log
  e.g. logs/exp_fs_01_K128.log, logs/exp_fs_02_frac0.05.log
"""
from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import time


# Bootstrap (auto-clone). Identical pattern to the per-exp scripts.
GITHUB_REPO = "https://github.com/trungkiet2005/ai_code_detection.git"
CLONE_DIR = "ai_code_detection"


def _setup_paths():
    here_candidates = []
    if "__file__" in globals():
        here_candidates.append(os.path.abspath(os.path.dirname(__file__)))
    base_dirs = [".", os.getcwd(), "/kaggle/working", f"/kaggle/working/{CLONE_DIR}"]
    candidates = here_candidates + base_dirs + [os.path.join(b, "Exp_FewShot") for b in base_dirs]
    for c in candidates:
        if c and os.path.exists(os.path.join(c, "_common_fs.py")):
            return c
    target = os.path.join(os.getcwd(), CLONE_DIR)
    if not os.path.isdir(target):
        print(f"[fs-bootstrap] cloning {GITHUB_REPO} -> {target}")
        subprocess.check_call(["git", "clone", "--depth", "1", GITHUB_REPO, target])
    return os.path.join(target, "Exp_FewShot")


FS_DIR = _setup_paths()


METHOD_TO_SCRIPT = {
    "baseline":      "testing/exp_fs_00_baseline.py",
    "ntkalign":      "testing/exp_fs_01_ntkalign.py",
    "supcon":        "testing/exp_fs_02_supcon.py",
    "frozen":        "testing/exp_fs_03_frozen.py",
    "ntk_frozen":    "testing/exp_fs_04_ntk_frozen.py",
    "supcon_frozen": "testing/exp_fs_05_supcon_frozen.py",
}


def _parse_csv(env_key: str, default: str):
    raw = os.environ.get(env_key, default)
    return [x.strip() for x in raw.split(",") if x.strip()]


def main():
    methods = _parse_csv(
        "FS_METHODS",
        "baseline,ntkalign,supcon,frozen,ntk_frozen,supcon_frozen",
    )
    ks      = [int(x) for x in _parse_csv("FS_KS", "32,128")]
    fracs   = [float(x) for x in _parse_csv("FS_FRACTIONS", "0.01,0.05")]
    seed    = int(os.environ.get("FS_SEED", "42"))

    pathlib.Path("logs").mkdir(exist_ok=True)
    plan = []
    for m in methods:
        if m not in METHOD_TO_SCRIPT:
            print(f"[skip] unknown method '{m}' (valid: {list(METHOD_TO_SCRIPT)})")
            continue
        for k in ks:
            plan.append((m, "kshot", k))
        for f in fracs:
            plan.append((m, "fraction", f))

    print(f"[portfolio] {len(plan)} runs queued: methods={methods} Ks={ks} fracs={fracs} seed={seed}")
    print(f"[portfolio] FS_DIR={FS_DIR}")

    t0 = time.time()
    for i, (method, regime, value) in enumerate(plan, 1):
        script = METHOD_TO_SCRIPT[method]
        env = os.environ.copy()
        env["FS_SEED"] = str(seed)
        if regime == "kshot":
            env["FS_K_SHOT"] = str(value)
            env["FS_TRAIN_FRACTION"] = "0.0"
            label = f"K{value}"
        else:
            env["FS_K_SHOT"] = "0"
            env["FS_TRAIN_FRACTION"] = str(value)
            label = f"frac{value:.4f}".rstrip("0").rstrip(".")
        log_path = f"logs/{script.replace('.py','')}_{label}.log"
        cmd = [sys.executable, os.path.join(FS_DIR, script)]
        print(f"\n[{i}/{len(plan)}] {method} {regime}={value} -> {log_path}")
        with open(log_path, "w", encoding="utf-8") as f:
            ret = subprocess.call(cmd, env=env, stdout=f, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        print(f"[{i}/{len(plan)}] exit={ret} elapsed_so_far={elapsed/60:.1f} min")

    print(f"\n[portfolio] all done in {(time.time()-t0)/60:.1f} min")
    print("[portfolio] grep 'BEGIN_FS_TABLE' logs/*.log to extract leaderboard rows")


if __name__ == "__main__":
    main()
