"""
exp_fs_04 -- NTKAlign + Frozen Encoder.

Combines two ideas: a structure-aware loss (NTK target-kernel alignment)
PLUS frozen encoder (no catastrophic forgetting). The hypothesis from the
K=32 results:
  - Baseline K=32 = 0.18
  - NTKAlign K=32 = 0.12 (worse, encoder destroyed by full fine-tune)
  - Frozen K=32 = ?    (linear probe baseline)
  - Frozen+NTK K=32 = ? (this file)
If Frozen+NTK > Frozen, NTK helps when encoder is stable.
If Frozen+NTK < Frozen, NTK is structurally bad at low data even with
frozen features (kernel target Y too sparse at bs=16).

KAGGLE-READY: auto-clones repo on first run.
Set FS_K_SHOT or FS_TRAIN_FRACTION; rest auto-tunes.
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

from _common_fs import (FSConfig, apply_hardware_profile, cleanup_after_run,
                         emit_paper_table, logger, parse_sweep_configs,
                         print_sweep_summary)
from _data_codet_fs import build_codet_fs_loaders
from _model_fs import FSClassifier, ntk_alignment_loss
from _trainer_fs import train_fewshot


METHOD_NAME = "FS-NTKAlign-Frozen"
EXP_ID = "exp_fs_04"


def run_one(kind, value, base_seed, lr_heads, lambda_ntk):
    cfg = FSConfig(
        benchmark="codet_m4", task="author",
        k_shot=value if kind == "kshot" else 0,
        train_fraction=value if kind == "fraction" else 0.0,
        fs_seed=base_seed,
        encoder_name="answerdotai/ModernBERT-base",
        lr_encoder=0.0, lr_heads=lr_heads, lambda_ntk=lambda_ntk,
    )
    cfg = apply_hardware_profile(cfg)
    logger.info(f"\n{'=' * 60}\n[{EXP_ID}] {kind}={value} FROZEN+NTK lambda={lambda_ntk}\n{'=' * 60}")

    bundle = build_codet_fs_loaders(cfg)
    model = FSClassifier(cfg)

    n_frozen = 0
    for p in model.encoder.parameters():
        p.requires_grad = False
        n_frozen += p.numel()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"[{EXP_ID}] frozen {n_frozen/1e6:.1f}M, trainable {n_trainable/1e6:.2f}M")

    def loss_fn(outputs, labels, class_weights=None):
        return ntk_alignment_loss(outputs, labels, lambda_ntk=cfg.lambda_ntk,
                                  class_weights=class_weights)

    results = train_fewshot(cfg, model,
                            bundle.train_loader, bundle.val_loader, bundle.test_loader,
                            loss_fn=loss_fn, method_name=METHOD_NAME)

    emit_paper_table(METHOD_NAME, EXP_ID, cfg, {
        "test_macro_f1": results.test_macro_f1, "test_weighted_f1": results.test_weighted_f1,
        "test_accuracy": results.test_accuracy, "val_macro_f1": results.val_macro_f1,
        "val_test_gap": results.val_macro_f1 - results.test_macro_f1,
        "train_steps": results.train_steps, "wall_time_s": f"{results.wall_time_s:.1f}",
        "lambda_ntk": cfg.lambda_ntk,
        "frozen_params_M": f"{n_frozen/1e6:.1f}",
        "trainable_params_M": f"{n_trainable/1e6:.2f}",
        "per_class_f1": results.per_class_f1, "per_lang_f1": results.per_lang_f1,
        "per_source_f1": results.per_source_f1, "train_per_class": bundle.train_per_class,
    })
    return results.test_macro_f1, results.val_macro_f1, results.wall_time_s


def main():
    base_seed = int(os.environ.get("FS_SEED", "42"))
    lr_heads = float(os.environ.get("FS_LR_HEADS", "1e-3"))
    lambda_ntk = float(os.environ.get("FS_LAMBDA_NTK", "0.4"))
    configs = parse_sweep_configs()
    logger.info(f"[{EXP_ID}] sweep: {configs}  seed={base_seed}")
    summary = []
    for kind, value in configs:
        test_f1, val_f1, wall = run_one(kind, value, base_seed, lr_heads, lambda_ntk)
        summary.append((kind, value, test_f1, val_f1, wall))
        cleanup_after_run()
    print_sweep_summary(summary, EXP_ID, METHOD_NAME)


if __name__ == "__main__":
    main()
