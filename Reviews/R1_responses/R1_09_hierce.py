# R1_09_hierce — Hierarchical cross-entropy baseline (Q8)
# =============================================================================
# Reviewer: "Have you tried a learned distance-bucket weighting or
# hierarchical cross-entropy as a direct baseline with the same encoder?"
# Two-stage softmax: predict family first, then class within family. We
# implement as a SINGLE forward with two heads sharing the encoder.
# Loss = CE(family) + CE(class | predicted family) using teacher-forced
# family during training.
# =============================================================================
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _traco_lib import (Cfg, GENE_ADJ_CODET, GENE_ADJ_AICD, build_dist, torch, F, nn,
                         _load_codet, _conv_codet, _vocab, FSDS, _hw,
                         adaptive_schedule, set_seed, AutoTokenizer,
                         AutoModel, KAGGLE_MODELS, GradScaler,
                         get_cosine_schedule_with_warmup, DataLoader,
                         _load_aicd, _conv_aicd, PAPER_BASELINE, accuracy_score,
                         f1_score)
from tqdm import tqdm

OUT = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(OUT, exist_ok=True)

FAM_OF_CODET = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 4}    # 5 families
FAM_OF_AICD  = {i: i // 3 for i in range(12)}           # 4 families


class HierCEModel(nn.Module):
    def __init__(self, enc_name, n_cls, n_fam, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name),
                                                  local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, emb_dim))
        self.head_fam = nn.Linear(emb_dim, n_fam)
        self.head_cls = nn.Linear(emb_dim, n_cls)
        self.emb_dim, self.n_cls, self.n_fam = emb_dim, n_cls, n_fam

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return self.head_fam(z), self.head_cls(z)


def main():
    rows = []
    bms = [("codet_m4", "author", 6, FAM_OF_CODET),
           ("aicd_t2",  "t2",     12, FAM_OF_AICD)]
    fracs = [0.01, 0.05, 0.20]
    for bench, task, n_cls, fam_of in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls, seed=42)
            set_seed(cfg.seed); cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
            n_fam = max(fam_of.values()) + 1
            if bench == "codet_m4":
                tr_raw, vl_raw, ts_raw = _load_codet()
                vocab = _vocab(tr_raw)
                tr_data = _conv_codet(tr_raw, "author", vocab)
                vl_data = _conv_codet(vl_raw, "author", vocab)
                ts_data = _conv_codet(ts_raw, "author", vocab)
            else:
                tr_raw, vl_raw, ts_raw = _load_aicd("t2")
                tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
            tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc),
                                                local_files_only=True)
            tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=False)
            vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
            ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
            total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
            lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
            tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
            vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
            ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
            model = HierCEModel(cfg.enc, cfg.n_cls, n_fam, cfg.emb_dim).to(cfg.device)
            enc_ids = {id(p) for p in model.encoder.parameters()}
            head = [p for p in model.parameters() if id(p) not in enc_ids]
            opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                                      {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
            sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
            scaler = GradScaler()
            best_val, best_state = 0.0, None
            t0 = time.time()
            tag = f"R1_09_hierce_{bench}_f{frac}"
            print(f"\n=== {tag} ===")
            for ep in range(cfg.epochs):
                model.train(); tot, lf_s, lc_s = 0.0, 0.0, 0.0
                for b in tqdm(tr_dl, desc=f"Train ep{ep+1}"):
                    ids = b["ids0"].to(cfg.device); m = b["mask0"].to(cfg.device)
                    labs = b["label"].to(cfg.device)
                    fams = torch.tensor([fam_of.get(int(y), int(y)) for y in labs.tolist()],
                                         device=cfg.device)
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                                        enabled=(cfg.device == "cuda")):
                        lg_f, lg_c = model(ids, m)
                        l_f = F.cross_entropy(lg_f, fams)
                        l_c = F.cross_entropy(lg_c, labs)
                        loss = l_f + l_c
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
                    tot += loss.item(); lf_s += l_f.item(); lc_s += l_c.item()
                # Eval
                model.eval(); preds, labels = [], []
                with torch.no_grad():
                    for b in vl_dl:
                        ids = b["ids0"].to(cfg.device); m = b["mask0"].to(cfg.device)
                        _, lg_c = model(ids, m)
                        preds.extend(lg_c.argmax(-1).cpu().tolist())
                        labels.extend(b["label"].tolist() if torch.is_tensor(b["label"]) else list(b["label"]))
                v = float(f1_score(labels, preds, average="macro", zero_division=0))
                print(f"  [ep{ep+1}] loss={tot/len(tr_dl):.4f} fam={lf_s/len(tr_dl):.4f} "
                      f"cls={lc_s/len(tr_dl):.4f} val_macro={v:.4f}")
                if v > best_val:
                    best_val = v; best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
            if best_state is not None: model.load_state_dict(best_state)
            # Test
            model.eval(); preds_t, labels_t = [], []
            with torch.no_grad():
                for b in ts_dl:
                    ids = b["ids0"].to(cfg.device); m = b["mask0"].to(cfg.device)
                    _, lg_c = model(ids, m)
                    preds_t.extend(lg_c.argmax(-1).cpu().tolist())
                    labels_t.extend(b["label"].tolist() if torch.is_tensor(b["label"]) else list(b["label"]))
            test_macro = float(f1_score(labels_t, preds_t, average="macro", zero_division=0))
            test_weighted = float(f1_score(labels_t, preds_t, average="weighted", zero_division=0))
            test_acc = float(accuracy_score(labels_t, preds_t))
            gap = best_val - test_macro
            print(f"[{tag}] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f}")
            rows.append({"tag": tag, "method": "HierCE", "bench": bench, "frac": frac,
                         "val_macro": best_val, "macro": test_macro,
                         "weighted": test_weighted, "acc": test_acc,
                         "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
                         "wall": round(time.time()-t0, 1)})
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    out_path = os.path.join(OUT, "R1_09_hierce.json")
    with open(out_path, "w") as f: json.dump(rows, f, indent=2)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
