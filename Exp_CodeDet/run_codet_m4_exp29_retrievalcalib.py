"""
[CoDET-M4] Exp29 RetrievalCalibCode  (Kaggle-standalone wrapper)

Variant of Exp27 DeTeCtiveCode:
- moderate contrastive pressure
- stronger retrieval blending for test-time calibration under shift

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. Bootstrap auto-clones the repo and imports
     Exp_CodeDet/run_codet_m4_exp27_detective.py as the `base` module.
"""

# ---------------------------------------------------------------------------
# Bootstrap: clone the repo if base module isn't importable
# ---------------------------------------------------------------------------

import importlib
import os
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"
BASE_MODULE = "run_codet_m4_exp27_detective"


def _bootstrap_base_module_path() -> str:
    cwd = os.getcwd()
    for candidate in (
        os.path.join(cwd, "Exp_CodeDet"),
        os.path.join(cwd, "ai_code_detection", "Exp_CodeDet"),
    ):
        if os.path.exists(os.path.join(candidate, f"{BASE_MODULE}.py")):
            return candidate
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        if os.path.exists(os.path.join(here, f"{BASE_MODULE}.py")):
            return here
    except NameError:
        pass
    repo_dir = os.path.join(cwd, "ai_code_detection")
    if not os.path.exists(repo_dir):
        print(f"[bootstrap] Cloning {REPO_URL} -> {repo_dir}")
        subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, repo_dir])
    return os.path.join(repo_dir, "Exp_CodeDet")


_base_dir = _bootstrap_base_module_path()
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)
print(f"[bootstrap] Exp_CodeDet path: {_base_dir}")

if BASE_MODULE in sys.modules:
    del sys.modules[BASE_MODULE]

base = importlib.import_module(BASE_MODULE)


def build_cfg():
    return base.CoDETM4Config(
        max_train_samples=100_000,
        max_val_samples=20_000,
        max_test_samples=50_000,
        eval_breakdown=True,
    )


def apply_variant(over_cfg):
    over_cfg.lambda_supcon = 0.08
    over_cfg.temperature = 0.07
    over_cfg.use_rag = True
    over_cfg.rag_k = 48
    over_cfg.rag_alpha = 0.35
    over_cfg.rag_bank_subsample = 80_000
    return over_cfg


if __name__ == "__main__":
    _orig_builder = base.build_detective_config

    def _patched_builder(task_tag: str, save_root: str):
        cfg = _orig_builder(task_tag, save_root)
        cfg.save_dir = cfg.save_dir.replace("codet_m4_checkpoints", "codet_m4_exp29_checkpoints")
        return apply_variant(cfg)

    base.build_detective_config = _patched_builder

    RUN_MODE = "full"
    RUN_PREFLIGHT_CHECK = True
    SINGLE_TASK = "binary"
    codet_cfg = build_cfg()

    if RUN_MODE == "full":
        base.run_suite(base.FULL_RUN_PLAN, codet_cfg, run_preflight=RUN_PREFLIGHT_CHECK)
    elif RUN_MODE == "iid_only":
        iid_plan = [e for e in base.FULL_RUN_PLAN if e[0] == "iid"]
        base.run_suite(iid_plan, codet_cfg, run_preflight=RUN_PREFLIGHT_CHECK)
    elif RUN_MODE == "ood_only":
        ood_plan = [e for e in base.FULL_RUN_PLAN if e[0] == "ood"]
        base.run_suite(ood_plan, codet_cfg, run_preflight=RUN_PREFLIGHT_CHECK)
    elif RUN_MODE == "single":
        base.run_iid(SINGLE_TASK, codet_cfg, run_preflight=RUN_PREFLIGHT_CHECK)
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")
