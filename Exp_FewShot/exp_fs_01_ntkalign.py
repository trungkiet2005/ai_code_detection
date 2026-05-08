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

Run on Kaggle T4 (auto-detected). Set FS_K_SHOT and FS_LAMBDA_NTK to sweep.

  python Exp_FewShot/exp_fs_01_ntkalign.py
  FS_K_SHOT=64 FS_LAMBDA_NTK=0.4 python Exp_FewShot/exp_fs_01_ntkalign.py
"""
from __future__ import annotations

import os
import sys


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
from _model_fs import FSClassifier, ntk_alignment_loss
from _trainer_fs import train_fewshot


METHOD_NAME = "FS-NTKAlign"
EXP_ID = "exp_fs_01"


def main():
    cfg = FSConfig(
        benchmark="codet_m4",
        task="author",
        k_shot=int(os.environ.get("FS_K_SHOT", "32")),
        fs_seed=int(os.environ.get("FS_SEED", "42")),
        lambda_ntk=float(os.environ.get("FS_LAMBDA_NTK", "0.4")),
        encoder_name="answerdotai/ModernBERT-base",
    )
    cfg = apply_hardware_profile(cfg)
    logger.info(
        f"[{EXP_ID}] starting K={cfg.k_shot} lambda_ntk={cfg.lambda_ntk} "
        f"on {cfg.benchmark}/{cfg.task}"
    )

    bundle = build_codet_fs_loaders(cfg)
    model = FSClassifier(cfg)

    def loss_fn(outputs, labels, class_weights=None):
        return ntk_alignment_loss(
            outputs, labels,
            lambda_ntk=cfg.lambda_ntk,
            class_weights=class_weights,
        )

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
        "lambda_ntk":       cfg.lambda_ntk,
        "per_class_f1":     results.per_class_f1,
        "per_lang_f1":      results.per_lang_f1,
        "per_source_f1":    results.per_source_f1,
        "train_per_class":  bundle.train_per_class,
    })


if __name__ == "__main__":
    main()
