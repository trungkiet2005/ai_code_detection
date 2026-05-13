"""
exp_fs_02 -- Supervised Contrastive (SupCon, Khosla NeurIPS 2020).

Hypothesis: NTK alignment uses a Frobenius distance that vanishes when the
target kernel Y is sparse (few same-class pairs per batch). SupCon uses a
softmax-over-similarities formulation that is per-anchor normalised and so
more robust to small batches at K-shot.

  L = CE + lambda_supcon * SupCon(z, y, tau)
  SupCon = -1/B sum_i 1/|P(i)| sum_{p in P(i)} log exp(z_i z_p / T) / Z_i

KAGGLE-READY: auto-clones the repo on first run if support modules absent.
Set FS_K_SHOT (K-shot mode) OR FS_TRAIN_FRACTION (phase-transition mode).

  python Exp_FewShot/exp_fs_02_supcon.py                       # K=32 default
  FS_K_SHOT=128 python Exp_FewShot/exp_fs_02_supcon.py
  FS_TRAIN_FRACTION=0.05 python Exp_FewShot/exp_fs_02_supcon.py
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
    return _attach(os.path.join(target, "Exp_FewShot"))


_setup_paths()

from _common_fs import (FSConfig, apply_hardware_profile, cleanup_after_run,
                         emit_paper_table, logger, parse_sweep_configs,
                         print_sweep_summary)
from _data_codet_fs import build_codet_fs_loaders
from _model_fs import FSClassifier, supcon_loss
from _trainer_fs import train_fewshot


METHOD_NAME = "FS-SupCon"
EXP_ID = "exp_fs_02"


def run_one(kind, value, base_seed, lambda_supcon, temperature):
    cfg = FSConfig(
        benchmark="codet_m4", task="author",
        k_shot=value if kind == "kshot" else 0,
        train_fraction=value if kind == "fraction" else 0.0,
        fs_seed=base_seed,
        encoder_name="answerdotai/ModernBERT-base",
    )
    cfg = apply_hardware_profile(cfg)
    logger.info(f"\n{'=' * 60}\n[{EXP_ID}] {kind}={value} lambda={lambda_supcon} T={temperature}\n{'=' * 60}")

    bundle = build_codet_fs_loaders(cfg)
    model = FSClassifier(cfg)

    def loss_fn(outputs, labels, class_weights=None):
        return supcon_loss(outputs, labels, lambda_supcon=lambda_supcon,
                           temperature=temperature, class_weights=class_weights)

    results = train_fewshot(cfg, model,
                            bundle.train_loader, bundle.val_loader, bundle.test_loader,
                            loss_fn=loss_fn, method_name=METHOD_NAME)

    emit_paper_table(METHOD_NAME, EXP_ID, cfg, {
        "test_macro_f1": results.test_macro_f1, "test_weighted_f1": results.test_weighted_f1,
        "test_accuracy": results.test_accuracy, "val_macro_f1": results.val_macro_f1,
        "val_test_gap": results.val_macro_f1 - results.test_macro_f1,
        "train_steps": results.train_steps, "wall_time_s": f"{results.wall_time_s:.1f}",
        "lambda_supcon": lambda_supcon, "temperature": temperature,
        "per_class_f1": results.per_class_f1, "per_lang_f1": results.per_lang_f1,
        "per_source_f1": results.per_source_f1, "train_per_class": bundle.train_per_class,
    })
    return results.test_macro_f1, results.val_macro_f1, results.wall_time_s


def main():
    base_seed = int(os.environ.get("FS_SEED", "42"))
    lambda_supcon = float(os.environ.get("FS_LAMBDA_SUPCON", "0.4"))
    temperature = float(os.environ.get("FS_TEMP", "0.07"))
    configs = parse_sweep_configs()
    logger.info(f"[{EXP_ID}] sweep: {configs}  seed={base_seed}")
    summary = []
    for kind, value in configs:
        test_f1, val_f1, wall = run_one(kind, value, base_seed, lambda_supcon, temperature)
        summary.append((kind, value, test_f1, val_f1, wall))
        cleanup_after_run()
    print_sweep_summary(summary, EXP_ID, METHOD_NAME)


if __name__ == "__main__":
    main()
