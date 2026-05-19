# R1_06_treedist — HUMAN-vs-AI default distance ablation (W2/Q5)
# =============================================================================
# Reviewer: "You assign HUMAN-AI distance = 3 and cross-family = 4 ... Why
# is human 'closer' to each AI family than families are to each other?
# Please justify or ablate these design choices."
# We sweep d_HUMAN in {1, 2, 3, 4, 5} and report Macro-F1. d=3 was the
# paper's hand-picked default; we want to show it sits within a flat
# plateau or, equivalently, identify the best empirical value.
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
    hum_dists = [1.0, 2.0, 3.0, 4.0, 5.0]
    for bench, task, n_cls in bms:
        for frac in fracs:
            for hd in hum_dists:
                cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls, seed=42,
                          hum_dist=hd)
                tag = f"R1_06_treedist_hum={hd:.1f}_{bench}_f{frac}"
                print(f"\n=== {tag} ===")
                t0 = time.time()
                try:
                    r = run_traco(cfg)
                    r.update({"tag": tag, "bench": bench, "frac": frac,
                              "hum_dist": hd, "wall": round(time.time()-t0, 1)})
                    r.pop("preds", None); r.pop("labels", None); r.pop("probs", None); r.pop("sources", None)
                    rows.append(r)
                    print(f"[{tag}] val={r['val_macro']:.4f} test={r['macro']:.4f} "
                          f"gap={r['val_test_gap']:+.4f}")
                except Exception as e:
                    print(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    out_path = os.path.join(OUT, "R1_06_treedist.json")
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
