# R1_02_hpsens — Hyperparameter sensitivity (Q2)
# =============================================================================
# Reviewer: "Hyperparameter sensitivity study for γ, τ, and λ."
# Grid sweep at 5pct CoDET-M4 (most stable slot, 6h budget).
# =============================================================================
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from _traco_lib import Cfg, run_traco, torch

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

# Smaller-than-Cartesian sweep: vary one knob at a time around the paper's
# default (1.0, 0.10, 0.5) to bound sensitivity within ~30 runs.
DEFAULTS = {"gamma": 1.0, "tau": 0.10, "lambda_aug": 0.5}
SWEEPS = {
    "gamma":      [0.25, 0.5, 1.0, 2.0, 4.0],
    "tau":        [0.05, 0.10, 0.20, 0.50],
    "lambda_aug": [0.1, 0.3, 0.5, 1.0],
}


def main():
    rows = []
    bench, task, n_cls = "codet_m4", "author", 6
    frac = 0.05
    for axis, vals in SWEEPS.items():
        for v in vals:
            kwargs = {**DEFAULTS, axis: v}
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls, seed=42,
                      gamma=kwargs["gamma"], tau=kwargs["tau"], lambda_aug=kwargs["lambda_aug"])
            tag = f"R1_02_hpsens_{axis}={v}_codet5pct"
            print(f"\n=== {tag} ===")
            t0 = time.time()
            try:
                r = run_traco(cfg)
                r.update({"tag": tag, "axis": axis, "value": v,
                          "gamma": kwargs["gamma"], "tau": kwargs["tau"], "lambda_aug": kwargs["lambda_aug"],
                          "wall": round(time.time()-t0, 1)})
                r.pop("preds", None); r.pop("labels", None); r.pop("probs", None); r.pop("sources", None)
                rows.append(r)
                print(f"[{tag}] val={r['val_macro']:.4f} test={r['macro']:.4f} "
                      f"gap={r['val_test_gap']:+.4f}")
            except Exception as e:
                print(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    out_path = os.path.join(OUT, "R1_02_hpsens.json")
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
