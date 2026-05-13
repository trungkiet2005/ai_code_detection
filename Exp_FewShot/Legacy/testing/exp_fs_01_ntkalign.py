"""
exp_fs_01 -- Few-shot NTK Alignment.

NTK target-kernel alignment loss (Jacot 2018 + Cristianini 2001) on top of
the cross-entropy baseline.

  L = CE(logits, y) + lambda_ntk * || H K H - H Y H ||_F^2 / B^2

  K_ij = <z_i, z_j>            (z = ntk_proj L2-normalized)
  Y_ij = 1[y_i == y_j]
  H    = centering matrix

Motivation: at K-shot, the model has too few labeled pairs to learn class
boundaries from CE alone. NTK alignment adds a kernel-level objective that
pulls same-class projections together and pushes others apart, even when
each class has only K=8..128 training examples.

KAGGLE-READY: this single file auto-clones the repo from GitHub if the
support modules aren't present locally. Just `!python exp_fs_01_ntkalign.py`
in any Kaggle cell — no setup needed.

Run on Kaggle T4 (auto-detected). Set FS_K_SHOT and FS_LAMBDA_NTK to sweep.

  python Exp_FewShot/exp_fs_01_ntkalign.py
  FS_K_SHOT=64 FS_LAMBDA_NTK=0.4 python Exp_FewShot/exp_fs_01_ntkalign.py

  # Standalone Kaggle cell:
  !FS_K_SHOT=32 python exp_fs_01_ntkalign.py
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

    Search local paths first; if not found, git-clone from GitHub.
    Robust to %run / paste-into-cell (no __file__) and re-imports.
    """
    here_candidates = []
    if "__file__" in globals():
        here_candidates.append(os.path.abspath(os.path.dirname(os.path.realpath(__file__))))

    base_dirs = [".", os.getcwd(), "/kaggle/working", f"/kaggle/working/{CLONE_DIR}"]
    candidates = here_candidates + base_dirs + [os.path.join(b, "Exp_FewShot") for b in base_dirs]

    def _attach(path):
        if path not in sys.path:
            sys.path.insert(0, path)
        for mod in list(sys.modules):
            if mod.startswith(("_common_fs", "_fewshot_sampler", "_data_codet_fs",
                                "_model_fs", "_trainer_fs")):
                del sys.modules[mod]
        return path

    for c in candidates:
        if c and os.path.exists(os.path.join(c, "_common_fs.py")):
            return _attach(c)

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
from _model_fs import FSClassifier, ntk_alignment_loss
from _trainer_fs import train_fewshot


METHOD_NAME = "FS-NTKAlign"
EXP_ID = "exp_fs_01"


def run_one(kind: str, value, base_seed: int, lambda_ntk: float):
    cfg = FSConfig(
        benchmark="codet_m4",
        task="author",
        k_shot=value if kind == "kshot" else 0,
        train_fraction=value if kind == "fraction" else 0.0,
        fs_seed=base_seed,
        lambda_ntk=lambda_ntk,
        encoder_name="answerdotai/ModernBERT-base",
    )
    cfg = apply_hardware_profile(cfg)
    logger.info(f"\n{'=' * 60}\n[{EXP_ID}] config={kind}={value} lambda_ntk={lambda_ntk}\n{'=' * 60}")

    bundle = build_codet_fs_loaders(cfg)
    model = FSClassifier(cfg)

    def loss_fn(outputs, labels, class_weights=None):
        return ntk_alignment_loss(outputs, labels, lambda_ntk=cfg.lambda_ntk,
                                  class_weights=class_weights)

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
        "lambda_ntk":       cfg.lambda_ntk,
        "per_class_f1":     results.per_class_f1,
        "per_lang_f1":      results.per_lang_f1,
        "per_source_f1":    results.per_source_f1,
        "train_per_class":  bundle.train_per_class,
    })

    return results.test_macro_f1, results.val_macro_f1, results.wall_time_s


def main():
    base_seed = int(os.environ.get("FS_SEED", "42"))
    lambda_ntk = float(os.environ.get("FS_LAMBDA_NTK", "0.4"))
    configs = parse_sweep_configs()
    logger.info(f"[{EXP_ID}] sweep plan: {configs}  seed={base_seed} lambda_ntk={lambda_ntk}")

    summary = []
    for kind, value in configs:
        test_f1, val_f1, wall_s = run_one(kind, value, base_seed, lambda_ntk)
        summary.append((kind, value, test_f1, val_f1, wall_s))
        cleanup_after_run()

    print_sweep_summary(summary, EXP_ID, METHOD_NAME)


if __name__ == "__main__":
    main()
