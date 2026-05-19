# R1_04_modernbert — TRACO with ModernBERT-base encoder (W6/Q7)
# =============================================================================
# Reviewer: "Stronger encoder baselines (e.g., ModernBERT) under the same
# protocol, and/or show that TRACO's relative gains persist with a stronger
# backbone."
# All 6 slots, identical protocol, only swap the encoder.
# =============================================================================
import os, sys, json, time
sys.path.insert(0, os.path.dirname(__file__))
from _traco_lib import Cfg, run_traco, torch

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def main():
    rows = []
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls, seed=42,
                       enc="ModernBERT-base")
            tag = f"R1_04_modernbert_{bench}_f{frac}"
            print(f"\n=== {tag} ===")
            t0 = time.time()
            try:
                r = run_traco(cfg)
                r.update({"tag": tag, "encoder": "ModernBERT-base",
                          "bench": bench, "frac": frac,
                          "wall": round(time.time()-t0, 1)})
                r.pop("preds", None); r.pop("labels", None); r.pop("probs", None); r.pop("sources", None)
                rows.append(r)
                print(f"[{tag}] val={r['val_macro']:.4f} test={r['macro']:.4f} "
                      f"gap={r['val_test_gap']:+.4f} t={r['wall']:.0f}s")
            except Exception as e:
                print(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    out_path = os.path.join(OUT, "R1_04_modernbert.json")
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
