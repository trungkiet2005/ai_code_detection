"""
[CoDET-M4] Exp28 HardNegCode  (Kaggle-standalone wrapper)

Variant of Exp27 DeTeCtiveCode:
  - stronger supervised contrastive pressure (lambda_supcon=0.20, tau=0.05,
    contrast_dim=192)
  - NO retrieval blending (use_rag=False) to isolate the representation-learning
    effect of harder SupCon.

Kaggle workflow:
  1. Upload ONLY this file.
  2. Run. Bootstrap auto-clones the repo and imports
     Exp_CodeDet/run_codet_m4_exp27_detective.py as the `base` module.
"""

# ---------------------------------------------------------------------------
# Bootstrap: clone the repo if run_codet_m4_exp27_detective.py isn't importable
# ---------------------------------------------------------------------------

import importlib
import os
import subprocess
import sys

REPO_URL = "https://github.com/trungkiet2005/ai_code_detection.git"
BASE_MODULE = "run_codet_m4_exp27_detective"


def _exp_codedet_dir_from(candidate: str) -> str:
    """Return candidate if it contains the base module .py file, else ''."""
    if os.path.exists(os.path.join(candidate, f"{BASE_MODULE}.py")):
        return candidate
    return ""


def _bootstrap_base_module_path() -> str:
    """Find or clone the repo so run_codet_m4_exp27_detective.py is importable."""
    cwd = os.getcwd()

    # Case A: already run from inside the repo layout
    for candidate in (
        os.path.join(cwd, "Exp_CodeDet"),
        os.path.join(cwd, "ai_code_detection", "Exp_CodeDet"),
    ):
        p = _exp_codedet_dir_from(candidate)
        if p:
            return p

    # Case B: this script lives inside Exp_CodeDet/ (when __file__ is defined)
    try:
        here = os.path.dirname(os.path.abspath(__file__))  # noqa: F821
        if os.path.exists(os.path.join(here, f"{BASE_MODULE}.py")):
            return here
    except NameError:
        pass  # notebook upload -- no __file__

    # Case C: clone fresh
    repo_dir = os.path.join(cwd, "ai_code_detection")
    if not os.path.exists(repo_dir):
        print(f"[bootstrap] Cloning {REPO_URL} -> {repo_dir}")
        subprocess.check_call(["git", "clone", "--depth=1", REPO_URL, repo_dir])
    return os.path.join(repo_dir, "Exp_CodeDet")


_base_dir = _bootstrap_base_module_path()
if _base_dir not in sys.path:
    sys.path.insert(0, _base_dir)
print(f"[bootstrap] Exp_CodeDet path: {_base_dir}")

# Evict any stale cached import from a previous session
if BASE_MODULE in sys.modules:
    del sys.modules[BASE_MODULE]

base = importlib.import_module(BASE_MODULE)


# ---------------------------------------------------------------------------
# HardNegCode variant
# ---------------------------------------------------------------------------

def build_cfg():
    return base.CoDETM4Config(
        max_train_samples=100_000,
        max_val_samples=20_000,
        max_test_samples=50_000,
        eval_breakdown=True,
    )


def apply_variant(over_cfg):
    over_cfg.lambda_supcon = 0.20
    over_cfg.temperature = 0.05
    over_cfg.contrast_dim = 192
    over_cfg.use_rag = False
    return over_cfg


if __name__ == "__main__":
    # Monkey-patch the config builder used by base.run_iid / base.run_suite.
    _orig_builder = base.build_detective_config

    def _patched_builder(task_tag: str, save_root: str):
        cfg = _orig_builder(task_tag, save_root)
        cfg.save_dir = cfg.save_dir.replace(
            "codet_m4_checkpoints", "codet_m4_exp28_checkpoints"
        )
        return apply_variant(cfg)

    base.build_detective_config = _patched_builder

    RUN_MODE = "full"          # full | iid_only | ood_only | single
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
