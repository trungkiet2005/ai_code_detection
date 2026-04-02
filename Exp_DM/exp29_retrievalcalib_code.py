"""
[EXP29] RetrievalCalibCode (DM)

Wrapper over EXP27 DeTeCtiveCode:
- lighter SupCon
- stronger kNN blend for OOD calibration
"""

import exp27_detective_code as base


def apply_variant(cfg: base.Config) -> base.Config:
    cfg.lambda_supcon = 0.08
    cfg.temperature = 0.07
    cfg.use_rag = True
    cfg.rag_k = 48
    cfg.rag_alpha = 0.35
    cfg.rag_bank_subsample = 80_000
    cfg.save_dir = "./exp29_retrievalcalib_checkpoints"
    return cfg


if __name__ == "__main__":
    RUN_MODE = "full"
    RUN_PREFLIGHT_CHECK = True
    TASK = "T1"
    BENCHMARK = "aicd"

    full_plan = [
        ("aicd", "T1"),
        ("aicd", "T2"),
        ("aicd", "T3"),
        ("droid", "T3"),
        ("droid", "T4"),
    ]

    cfg = apply_variant(base.Config(
        epochs=3,
        batch_size=32,
        grad_accum_steps=2,
        precision="auto",
        auto_h100_profile=True,
        num_workers=4,
        prefetch_factor=2,
        max_train_samples=100_000,
        max_val_samples=20_000,
        max_test_samples=50_000,
        require_tree_sitter=True,
    ))
    cfg = base.apply_hardware_profile(cfg)

    if RUN_MODE == "full":
        if RUN_PREFLIGHT_CHECK:
            base.preflight_benchmark_suite(full_plan, base_config=cfg)
        base.run_benchmark_suite(full_plan, base_config=cfg)
    elif RUN_MODE == "single":
        cfg.task = TASK
        cfg.benchmark = BENCHMARK
        if RUN_PREFLIGHT_CHECK:
            base.preflight_single_benchmark(cfg)
        base.main(task=TASK, config=cfg)
    else:
        raise ValueError(f"Unknown RUN_MODE: {RUN_MODE}")
