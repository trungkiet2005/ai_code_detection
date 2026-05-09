"""
exp_fs_05 -- SupCon + Frozen Encoder.

Combines SupCon (Khosla 2020) with frozen encoder. The 4 corner cases:
   exp_fs_00 (CE)         : free encoder, no structure loss
   exp_fs_01 (NTKAlign)   : free encoder, NTK structure
   exp_fs_02 (SupCon)     : free encoder, SupCon structure
   exp_fs_03 (Frozen)     : frozen encoder, no structure loss
   exp_fs_04 (NTK+Frozen) : frozen encoder, NTK structure
   exp_fs_05 (SupCon+Frozen) : frozen encoder, SupCon structure  <-- this file

The SupCon loss is per-anchor softmax-based, so it should degrade more
gracefully than NTK at small batches even when encoder is frozen.

KAGGLE-READY: auto-clones repo on first run.
"""
from __future__ import annotations

import os
import subprocess
import sys


GITHUB_REPO = "https://github.com/trungkiet2005/ai_code_detection.git"
CLONE_DIR = "ai_code_detection"


def _setup_paths():
    here_candidates = []
    if "__file__" in globals():
        here_candidates.append(os.path.abspath(os.path.dirname(__file__)))
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
    return _attach(os.path.join(target, "Exp_FewShot"))


_setup_paths()

from _common_fs import FSConfig, apply_hardware_profile, emit_paper_table, logger
from _data_codet_fs import build_codet_fs_loaders
from _model_fs import FSClassifier, supcon_loss
from _trainer_fs import train_fewshot


METHOD_NAME = "FS-SupCon-Frozen"
EXP_ID = "exp_fs_05"


def main():
    cfg = FSConfig(
        benchmark="codet_m4",
        task="author",
        k_shot=int(os.environ.get("FS_K_SHOT", "32")),
        train_fraction=float(os.environ.get("FS_TRAIN_FRACTION", "0.0")),
        fs_seed=int(os.environ.get("FS_SEED", "42")),
        encoder_name="answerdotai/ModernBERT-base",
        lr_encoder=0.0,
        lr_heads=float(os.environ.get("FS_LR_HEADS", "1e-3")),
    )
    cfg = apply_hardware_profile(cfg)
    lambda_supcon = float(os.environ.get("FS_LAMBDA_SUPCON", "0.4"))
    temperature = float(os.environ.get("FS_TEMP", "0.07"))
    logger.info(
        f"[{EXP_ID}] K={cfg.k_shot} frac={cfg.train_fraction} "
        f"FROZEN encoder, lr_heads={cfg.lr_heads}, "
        f"lambda_supcon={lambda_supcon}, T={temperature}"
    )

    bundle = build_codet_fs_loaders(cfg)
    model = FSClassifier(cfg)

    n_frozen = 0
    for p in model.encoder.parameters():
        p.requires_grad = False
        n_frozen += p.numel()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"[{EXP_ID}] frozen {n_frozen/1e6:.1f}M params, "
        f"trainable {n_trainable/1e6:.2f}M (head + projector only)"
    )

    def loss_fn(outputs, labels, class_weights=None):
        return supcon_loss(outputs, labels, lambda_supcon=lambda_supcon,
                           temperature=temperature, class_weights=class_weights)

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
        "lambda_supcon":    lambda_supcon,
        "temperature":      temperature,
        "frozen_params_M":  f"{n_frozen/1e6:.1f}",
        "trainable_params_M": f"{n_trainable/1e6:.2f}",
        "per_class_f1":     results.per_class_f1,
        "per_lang_f1":      results.per_lang_f1,
        "per_source_f1":    results.per_source_f1,
        "train_per_class":  bundle.train_per_class,
    })


if __name__ == "__main__":
    main()
