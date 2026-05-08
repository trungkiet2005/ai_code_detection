"""
exp_fs_00 -- Few-shot baseline: ModernBERT + linear head + cross-entropy.

Floor for the few-shot suite. If our novel methods can't beat this baseline
at K=64 / 128, the entire pivot fails. CE-only.

KAGGLE-READY: this single file auto-clones the repo from GitHub if the
support modules (_common_fs, _data_codet_fs, ...) aren't present locally.
Just `!python exp_fs_00_baseline.py` in any Kaggle cell — no setup needed.

Run on Kaggle T4 (auto-detected). Set FS_K_SHOT env var to override K.

  python Exp_FewShot/exp_fs_00_baseline.py
  FS_K_SHOT=128 python Exp_FewShot/exp_fs_00_baseline.py

  # Standalone Kaggle cell (just upload THIS file or paste it in):
  !python exp_fs_00_baseline.py
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

from _common_fs import FSConfig, apply_hardware_profile, emit_paper_table, logger
from _data_codet_fs import build_codet_fs_loaders
from _model_fs import FSClassifier, cross_entropy_loss
from _trainer_fs import train_fewshot


METHOD_NAME = "FS-Baseline-CE"
EXP_ID = "exp_fs_00"


def main():
    cfg = FSConfig(
        benchmark="codet_m4",
        task="author",
        k_shot=int(os.environ.get("FS_K_SHOT", "32")),
        fs_seed=int(os.environ.get("FS_SEED", "42")),
        encoder_name="answerdotai/ModernBERT-base",
    )
    cfg = apply_hardware_profile(cfg)
    logger.info(f"[{EXP_ID}] starting K={cfg.k_shot} on {cfg.benchmark}/{cfg.task}")

    bundle = build_codet_fs_loaders(cfg)
    model = FSClassifier(cfg)

    def loss_fn(outputs, labels, class_weights=None):
        return cross_entropy_loss(outputs, labels, class_weights=class_weights)

    results = train_fewshot(
        cfg=cfg,
        model=model,
        train_loader=bundle.train_loader,
        val_loader=bundle.val_loader,
        test_loader=bundle.test_loader,
        loss_fn=loss_fn,
        method_name=METHOD_NAME,
    )

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


if __name__ == "__main__":
    main()
