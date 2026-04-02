"""
[CoDET-M4] Exp28 HardNegCode

Variant of Exp27 DeTeCtiveCode:
- stronger supervised contrastive pressure
- no retrieval blending (isolate representation learning effect)
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
    over_cfg.lambda_supcon = 0.20
    over_cfg.temperature = 0.05
    over_cfg.contrast_dim = 192
    over_cfg.use_rag = False
    return over_cfg


if __name__ == "__main__":
    # Monkey-patch config builder used by base run_iid/run_suite.
    _orig_builder = base.build_detective_config

    def _patched_builder(task_tag: str, save_root: str) -> base.SpectralConfig:
        cfg = _orig_builder(task_tag, save_root)
        cfg.save_dir = cfg.save_dir.replace("codet_m4_checkpoints", "codet_m4_exp28_checkpoints")
        return apply_variant(cfg)

    base.build_detective_config = _patched_builder

    RUN_MODE = "full"  # full | iid_only | ood_only | single
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
