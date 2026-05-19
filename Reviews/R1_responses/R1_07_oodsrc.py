# R1_07_oodsrc — Held-out-domain (CF train, GH/LC test) (W5)
# =============================================================================
# Reviewer: "Held-out-domain evaluation protocol (train on Codeforces, test
# on GitHub, etc.) to systematically quantify domain shift."
# CoDET-M4 only (its `source` field encodes cf/gh/lc).
# Train on cf-only at 20pct stratified-per-class, test on the gh and lc
# slices of the standard test split. Compare TRACO vs CE-only (gamma=0).
# =============================================================================
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _traco_lib import (Cfg, run_traco, GENE_ADJ_CODET, build_dist, torch,
                         _load_codet, _conv_codet, _vocab, FSDS, eval_macro,
                         AutoTokenizer, KAGGLE_MODELS, os as _os, set_seed,
                         _hw, adaptive_schedule, TRACOModel, supcon_tw_loss,
                         GradScaler, get_cosine_schedule_with_warmup,
                         DataLoader, train_traco, F)
from sklearn.metrics import f1_score, accuracy_score

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)


def main():
    rows = []
    bench, task, n_cls = "codet_m4", "author", 6
    fracs = [0.20]  # only the largest budget; held-out-domain is the focus
    train_src = "cf"
    eval_srcs = ["cf", "gh", "lc"]   # cf is in-domain control; gh/lc are out
    for frac in fracs:
        for use_tree in [True, False]:    # TRACO (True) vs CE-only (False)
            method = "TRACO" if use_tree else "CE-only"
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls, seed=42)
            if not use_tree: cfg.lambda_aug = 0.0  # disable contrastive term
            set_seed(cfg.seed); cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
            dist = build_dist(n_cls, GENE_ADJ_CODET, hum_dist=3.0).to(cfg.device)
            tr_raw, vl_raw, ts_raw = _load_codet()
            vocab = _vocab(tr_raw)
            tr_data = _conv_codet(tr_raw, "author", vocab)
            vl_data = _conv_codet(vl_raw, "author", vocab)
            ts_data = _conv_codet(ts_raw, "author", vocab)
            # Restrict train to source == cf
            tr_data_cf = tr_data.filter(lambda x: x.get("source") == train_src)
            print(f"\n=== R1_07 {method} train={train_src} frac={frac} "
                  f"train_size={len(tr_data_cf)} ===")
            tok = AutoTokenizer.from_pretrained(_os.path.join(KAGGLE_MODELS, cfg.enc),
                                                local_files_only=True)
            tr_ds = FSDS(tr_data_cf, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
            vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
            total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
            lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
            tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
            vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
            model = TRACOModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
            enc_ids = {id(p) for p in model.encoder.parameters()}
            head = [p for p in model.parameters() if id(p) not in enc_ids]
            opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                                      {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
            sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
            scaler = GradScaler()
            best_val, best_state = 0.0, None
            t0 = time.time()
            for ep in range(cfg.epochs):
                loss, ce, sc = train_traco(model, tr_dl, opt, sch, scaler, cfg, dist)
                vm = eval_macro(model, vl_dl, cfg)
                v = vm["macro_f1"]
                if v > best_val:
                    best_val = v
                    best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
                print(f"  [ep{ep+1}] loss={loss:.4f} val_macro={v:.4f}")
            if best_state is not None: model.load_state_dict(best_state)
            # Test on each source slice
            per_src = {}
            for src in eval_srcs:
                ts_src = ts_data.filter(lambda x: x.get("source") == src)
                if len(ts_src) == 0:
                    per_src[src] = {"n": 0}
                    continue
                ts_ds = FSDS(ts_src, tok, cfg.seq, frac=1.0, seed=cfg.seed+10, do_aug=False)
                ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
                tm = eval_macro(model, ts_dl, cfg)
                per_src[src] = {"n": int(len(ts_src)), "macro_f1": tm["macro_f1"],
                                "weighted_f1": tm["weighted_f1"], "accuracy": tm["accuracy"]}
                print(f"  test src={src} n={per_src[src]['n']} "
                      f"macro={per_src[src]['macro_f1']:.4f}")
            rows.append({"tag": f"R1_07_oodsrc_{method}_train={train_src}_f{frac}",
                         "method": method, "train_source": train_src, "frac": frac,
                         "val_macro": best_val, "per_source_test": per_src,
                         "wall": round(time.time()-t0, 1)})
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    out_path = os.path.join(OUT, "R1_07_oodsrc.json")
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
