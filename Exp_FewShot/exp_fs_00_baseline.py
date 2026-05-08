"""
exp_fs_00 -- Few-shot baseline: ModernBERT + linear head + cross-entropy.

Floor for the few-shot suite. If our novel methods can't beat this baseline
at K=64 / 128, the entire pivot fails. CE-only.

Run on Kaggle T4 (auto-detected). Set FS_K_SHOT env var to override K.

  python Exp_FewShot/exp_fs_00_baseline.py
  FS_K_SHOT=128 python Exp_FewShot/exp_fs_00_baseline.py
"""
from __future__ import annotations

import os
import sys


# ---------------------------------------------------------------------------
# Bootstrap path resolution (Kaggle-friendly)
# ---------------------------------------------------------------------------

def _setup_paths():
    here = os.path.abspath(os.path.dirname(__file__) if "__file__" in globals() else os.getcwd())
    candidates = [here] + [
        os.path.join(c, "Exp_FewShot")
        for c in (".", os.getcwd(), "/kaggle/working", "/kaggle/working/ai_code_detection")
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "_common_fs.py")):
            if c not in sys.path:
                sys.path.insert(0, c)
            return c
    raise RuntimeError(f"Could not find _common_fs.py in {candidates}")


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
