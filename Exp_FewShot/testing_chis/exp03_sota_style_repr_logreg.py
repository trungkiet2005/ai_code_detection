"""
exp03_sota_style_repr_logreg.py

Single-method runner for few-shot style-representation baseline.

This is the lightweight published-method adaptation:
code stylometry features + balanced logistic regression.

Default:
  FS_SWEEP_FRACS=0.01,0.05,0.20
  FS_BENCHMARK=codet_m4
  FS_SEED=42
"""
from __future__ import annotations

import json
import math
import os
from datetime import datetime

from run_published_sota_portfolio import (
    FSConfig,
    PAPER_BASELINE,
    PAPER_REGISTRY,
    _parse_fracs,
    apply_hardware_profile,
    cleanup,
    run_style_repr_logreg,
    set_seed,
)


METHOD = "style_repr_logreg"
METHOD_NAME = "Published-StyleRepr-LogReg"
EXP_PREFIX = "exp03_sota_style_repr_logreg"


def main():
    fractions = _parse_fracs()
    benchmark = os.environ.get("FS_BENCHMARK", "codet_m4")
    seed = int(os.environ.get("FS_SEED", "42"))
    set_seed(seed)

    rows = []
    for i, frac in enumerate(fractions, 1):
        cfg = FSConfig(
            benchmark=benchmark,
            task="author",
            k_shot=0,
            train_fraction=frac,
            fs_seed=seed,
            encoder_name="style_repr",
        )
        cfg = apply_hardware_profile(cfg)
        try:
            test_f1, val_f1, wall = run_style_repr_logreg(cfg, f"pub_style_repr_{i}")
            rows.append(("style_repr", frac, test_f1, val_f1, val_f1 - test_f1, wall))
        except Exception as exc:
            print(f"[{METHOD_NAME}] FAILED frac={frac}: {exc}")
            rows.append(("style_repr", frac, float("nan"), float("nan"), float("nan"), 0.0))
            cleanup()

    print(f"\n{METHOD_NAME}")
    print(f"{'Encoder':<24} {'Frac':>6} {'Test':>10} {'dPaper':>10} {'Val':>10} {'Gap':>10} {'Wall':>8}")
    print("-" * 90)
    for encoder, frac, test, val, gap, wall in rows:
        dp = test - PAPER_BASELINE if not math.isnan(test) else float("nan")
        print(f"{encoder:<24} {frac:>6.2%} {test:>10.4f} {dp:>+10.4f} {val:>10.4f} {gap:>+10.4f} {wall:>8.0f}")

    os.makedirs("./results", exist_ok=True)
    out_path = os.path.join("./results", f"{EXP_PREFIX}_seed{seed}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "method": METHOD_NAME,
                "method_key": METHOD,
                "benchmark": benchmark,
                "seed": seed,
                "paper_baseline": PAPER_BASELINE,
                "paper": PAPER_REGISTRY[METHOD],
                "results": [
                    {
                        "encoder": e,
                        "frac": fr,
                        "test_macro_f1": t,
                        "val_macro_f1": v,
                        "gap": g,
                        "wall_s": w,
                    }
                    for e, fr, t, v, g, w in rows
                ],
            },
            f,
            indent=2,
        )
    print(f"Aggregate saved: {out_path}")


if __name__ == "__main__":
    main()

