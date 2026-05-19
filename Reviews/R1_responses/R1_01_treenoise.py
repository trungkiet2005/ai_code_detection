# R1_01_treenoise — Tree-noise sensitivity (W1/Q1)
# =============================================================================
# Reviewer: "Sensitivity to errors in the family tree is unmeasured."
# Action: perturb the family-tree adjacency at 5 noise levels and report
# Macro-F1 vs noise. Two slots only (1pct CoDET-M4 + 1pct AICD-T2) since
# these are the most prior-sensitive.
# =============================================================================
import os, sys, json, copy, random, time
sys.path.insert(0, os.path.dirname(__file__))
from _traco_lib import (Cfg, run_traco, GENE_ADJ_CODET, GENE_ADJ_AICD,
                         build_dist, torch, np)

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def perturb_tree(adj, n_cls, noise_level: float, seed: int):
    """Edge-flip perturbation. With probability `noise_level` each
    (i,j) pair either drops its edge or adds a random one. Returns a
    new adjacency dict that still has integer-distance semantics."""
    rng = random.Random(seed)
    new_adj = {i: set(adj.get(i, [])) for i in range(n_cls)}
    for i in range(n_cls):
        for j in range(i+1, n_cls):
            if rng.random() < noise_level:
                if j in new_adj[i]:
                    new_adj[i].discard(j); new_adj[j].discard(i)
                else:
                    new_adj[i].add(j); new_adj[j].add(i)
    return {i: sorted(list(v)) for i, v in new_adj.items()}


def main():
    noise_levels = [0.0, 0.10, 0.25, 0.50, 1.00]
    benches = [("codet_m4", "author", 6, GENE_ADJ_CODET),
               ("aicd_t2",  "t2",     12, GENE_ADJ_AICD)]
    fracs = [0.01]      # most prior-sensitive
    rows = []
    for bench, task, n_cls, adj in benches:
        for frac in fracs:
            for nl in noise_levels:
                # Use a fixed perturbation seed independent of training seed.
                pert_adj = perturb_tree(adj, n_cls, nl, seed=1000)
                pert_dist = build_dist(n_cls, pert_adj, hum_dist=3.0)
                cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls, seed=42)
                tag = f"R1_01_noise={nl:.2f}_{bench}_f{frac}"
                print(f"\n=== {tag} ===")
                t0 = time.time()
                try:
                    r = run_traco(cfg, dist_override=pert_dist)
                    r.update({"tag": tag, "bench": bench, "frac": frac,
                              "noise_level": nl, "wall": round(time.time()-t0, 1)})
                    r.pop("preds", None); r.pop("labels", None); r.pop("probs", None); r.pop("sources", None)
                    rows.append(r)
                    print(f"[{tag}] val={r['val_macro']:.4f} test={r['macro']:.4f} "
                          f"gap={r['val_test_gap']:+.4f} t={r['wall']:.0f}s")
                except Exception as e:
                    print(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
    out_path = os.path.join(OUT, "R1_01_treenoise.json")
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
