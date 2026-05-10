"""
exp02_sota_faid_multitask.py

Single-method runner for FAID-style multitask + multi-level contrastive baseline.

Default:
  FS_ENCODER=ModernBERT-base,codebert-base,unixcoder-base
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
    PAPER_BASELINE,
    PAPER_REGISTRY,
    _parse_csv_env,
    _parse_fracs,
    cleanup,
    run_deep_method,
    set_seed,
)


METHOD = "faid_multitask"
METHOD_NAME = "Published-FAID-MultiTask"
EXP_PREFIX = "exp02_sota_faid_multitask"


def main():
    encoders = _parse_csv_env("FS_ENCODER", "ModernBERT-base,codebert-base,unixcoder-base")
    fractions = _parse_fracs()
    benchmark = os.environ.get("FS_BENCHMARK", "codet_m4")
    seed = int(os.environ.get("FS_SEED", "42"))
    set_seed(seed)

    rows = []
    for i, encoder in enumerate(encoders, 1):
        for j, frac in enumerate(fractions, 1):
            exp_num = (i - 1) * len(fractions) + j
            try:
                test_f1, val_f1, wall = run_deep_method(METHOD, encoder, frac, exp_num, seed, benchmark)
                rows.append((encoder, frac, test_f1, val_f1, val_f1 - test_f1, wall))
            except Exception as exc:
                print(f"[{METHOD_NAME}] FAILED encoder={encoder} frac={frac}: {exc}")
                rows.append((encoder, frac, float("nan"), float("nan"), float("nan"), 0.0))
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

