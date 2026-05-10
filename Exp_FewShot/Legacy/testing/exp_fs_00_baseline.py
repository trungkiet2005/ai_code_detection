"""
exp_fs_00 -- Few-shot baseline: ModernBERT + linear head + cross-entropy.

Floor for the few-shot suite. CE only. main() runs a SWEEP across both
K-shot and %-fraction configurations in one go and writes one JSON per
configuration to /kaggle/working/results/.

KAGGLE-READY: paste this entire file into a Kaggle cell, or
  !python exp_fs_00_baseline.py
The bootstrap auto-clones the repo from GitHub if the support modules
(_common_fs, _data_codet_fs, ...) aren't present locally.

Default sweep: K in {32, 128} + fraction in {0.01, 0.05} = 4 configs
(~30 min on T4). Override:
  FS_SWEEP_KS="8,16,32,64,128"     # extend or shrink K-shot points
  FS_SWEEP_FRACS="0.01,0.05,0.1"   # extend or shrink fraction points
  FS_SWEEP_KS=""                   # skip K-shot regime entirely
  FS_SWEEP_FRACS=""                # skip fraction regime entirely
  FS_SEED=42                       # reproducibility
"""
from __future__ import annotations

import os
import subprocess
import sys


# ---------------------------------------------------------------------------
# Bootstrap path resolution (Kaggle-friendly, auto-clones from GitHub if absent)
# ---------------------------------------------------------------------------

GITHUB_REPO = "https://github.com/trungkiet2005/ai_code_detection.git"
CLONE_DIR = "ai_code_detection"


def _setup_paths():
    """Find or clone the Exp_FewShot/ folder, add it to sys.path.

    Search order (Kaggle-friendly):
      1) the directory of this script if __file__ is defined
      2) cwd, /kaggle/working, /kaggle/working/<clone>
      3) those + /Exp_FewShot suffix
    If none of those contain _common_fs.py, clone the GitHub repo into
    ./ai_code_detection and use its Exp_FewShot/.

    Robust to %run / paste-into-cell (no __file__) and re-imports
    (drops stale cached modules so a fresh clone replaces partial state).
    """
    here_candidates = []
    if "__file__" in globals():
        here_candidates.append(os.path.abspath(os.path.dirname(__file__)))

    base_dirs = [".", os.getcwd(), "/kaggle/working", f"/kaggle/working/{CLONE_DIR}"]
    candidates = here_candidates + base_dirs + [os.path.join(b, "Exp_FewShot") for b in base_dirs]

    def _attach(path):
        if path not in sys.path:
            sys.path.insert(0, path)
        # drop stale cached modules so a fresh clone is picked up
        for mod in list(sys.modules):
            if mod.startswith(("_common_fs", "_fewshot_sampler", "_data_codet_fs",
                                "_model_fs", "_trainer_fs")):
                del sys.modules[mod]
        return path

    for c in candidates:
        if c and os.path.exists(os.path.join(c, "_common_fs.py")):
            return _attach(c)

    # Not found locally -- clone from GitHub (Kaggle has internet).
    target = os.path.join(os.getcwd(), CLONE_DIR)
    if not os.path.isdir(target):
        print(f"[fs-bootstrap] cloning {GITHUB_REPO} -> {target}")
        subprocess.check_call(["git", "clone", "--depth", "1", GITHUB_REPO, target])
    fewshot_dir = os.path.join(target, "Exp_FewShot")
    if not os.path.exists(os.path.join(fewshot_dir, "_common_fs.py")):
        raise RuntimeError(
            f"Cloned {target} but Exp_FewShot/_common_fs.py is missing. "
            "Repo layout may have changed."
        )
    return _attach(fewshot_dir)


_setup_paths()

from _common_fs import (FSConfig, apply_hardware_profile, cleanup_after_run,
                         emit_paper_table, logger, parse_sweep_configs,
                         print_sweep_summary)
from _data_codet_fs import build_codet_fs_loaders
from _model_fs import FSClassifier, cross_entropy_loss
from _trainer_fs import train_fewshot


METHOD_NAME = "FS-Baseline-CE"
EXP_ID = "exp_fs_00"


def run_one(kind: str, value, base_seed: int):
    cfg = FSConfig(
        benchmark="codet_m4",
        task="author",
        k_shot=value if kind == "kshot" else 0,
        train_fraction=value if kind == "fraction" else 0.0,
        fs_seed=base_seed,
        encoder_name="answerdotai/ModernBERT-base",
    )
    cfg = apply_hardware_profile(cfg)
    logger.info(f"\n{'=' * 60}\n[{EXP_ID}] config={kind}={value}\n{'=' * 60}")

    bundle = build_codet_fs_loaders(cfg)
    model = FSClassifier(cfg)

    def loss_fn(outputs, labels, class_weights=None):
        return cross_entropy_loss(outputs, labels, class_weights=class_weights)

    results = train_fewshot(cfg, model,
                            bundle.train_loader, bundle.val_loader, bundle.test_loader,
                            loss_fn=loss_fn, method_name=METHOD_NAME)

    emit_paper_table(METHOD_NAME, EXP_ID, cfg, {
        "test_macro_f1":    results.test_macro_f1,
        "test_weighted_f1": results.test_weighted_f1,
        "test_accuracy":    results.test_accuracy,
        "val_macro_f1":     results.val_macro_f1,
        "val_test_gap":     results.val_macro_f1 - results.test_macro_f1,
        "train_steps":      results.train_steps,
        "wall_time_s":      f"{results.wall_time_s:.1f}",
        "per_class_f1":     results.per_class_f1,
        "per_lang_f1":      results.per_lang_f1,
        "per_source_f1":    results.per_source_f1,
        "train_per_class":  bundle.train_per_class,
    })

    return results.test_macro_f1, results.val_macro_f1, results.wall_time_s


def main():
    base_seed = int(os.environ.get("FS_SEED", "42"))
    configs = parse_sweep_configs()
    logger.info(f"[{EXP_ID}] sweep plan: {configs}  seed={base_seed}")

    summary = []
    for kind, value in configs:
        test_f1, val_f1, wall_s = run_one(kind, value, base_seed)
        summary.append((kind, value, test_f1, val_f1, wall_s))
        cleanup_after_run()

    print_sweep_summary(summary, EXP_ID, METHOD_NAME)


if __name__ == "__main__":
    main()
