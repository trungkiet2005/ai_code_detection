"""
[CoDET-M4] Exp29 RetrievalCalibCode

Variant of Exp27 DeTeCtiveCode:
- moderate contrastive pressure
- stronger retrieval blending for test-time calibration under shift
"""

import run_codet_m4_exp27_detective as base


def build_cfg() -> base.CoDETM4Config:
    return base.CoDETM4Config(
        max_train_samples=100_000,
        max_val_samples=20_000,
        max_test_samples=50_000,
        eval_breakdown=True,
    )


def apply_variant(over_cfg: base.SpectralConfig) -> base.SpectralConfig:
    over_cfg.lambda_supcon = 0.08
    over_cfg.temperature = 0.07
    over_cfg.use_rag = True
    over_cfg.rag_k = 48
    over_cfg.rag_alpha = 0.35
    over_cfg.rag_bank_subsample = 80_000
    return over_cfg


if __name__ == "__main__":
    _orig_builder = base.build_detective_config

    def _patched_builder(task_tag: str, save_root: str) -> base.SpectralConfig:
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
