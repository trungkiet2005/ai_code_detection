# exp91_detmulti — DeTeCtive multi-level adapter: hierarchical SupCon on genealogy
# =============================================================================
# NAME           : DETMULTI (DeTeCtive multi-level, code attribution adaptation)
# UPSTREAM       : Anonymous, "DeTeCtive: Detecting AI-generated Text via
#                  Multi-Level Contrastive Learning", NeurIPS 2024
#                  Repo: github.com/heyongxin233/DeTeCtive
# ROLE           : Closest published competitor to TRACO's tree-weighting idea.
#                  Upstream uses two-level SupCon (author + writing-style)
#                  on text. We adapt: level-1 = author class, level-2 =
#                  family group from the fine-tune genealogy. Both levels
#                  use FLAT SupCon (no smooth tree distance), only DISCRETE
#                  level masks. If this matches TRACO, then smooth tree
#                  distance is unnecessary and discrete two-level grouping
#                  already captures the gain. If TRACO wins, the smooth
#                  exponential weighting is the operative ingredient.
# EQUATION       : L = L_ce + a * SupCon_author + b * SupCon_family
#                  Each SupCon is the standard Khosla et al. 2020 form with
#                  positives = same-level samples, negatives = remainder.
#                  Inference is the same end-to-end classifier (we skip the
#                  kNN-retrieval step from the upstream paper for protocol
#                  parity with our other adapters).
# FALSIFIER      : Delta vs TRACO at extreme few-shot.
# GPU TUNING     : Same 2-view doubled-batch pattern as exp90; bs auto-scales.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from dataclasses import dataclass, field
from typing import Dict, List
import re as _re

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
for _p in ("numpy", "torch", "datasets", "transformers", "scikit-learn", "tqdm"):
    _ensure(_p)

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, confusion_matrix)
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp91_detmulti")

PAPER_BASELINE = 0.6633


# ---- Genealogy: author -> family group --------------------------------------
# CoDET-M4 (6 classes): 0=Human, 1=CLa(sibling), 2=Lla, 3=Nxc(sibling), 4=GPT, 5=Qwn
# Families: {Human:0}, {Llama-line: 1=CLa,3=Nxc}, {Llama-3.1: 2=Lla}, {OpenAI: 4=GPT}, {Qwen: 5=Qwn}
FAMILY_OF_CODET = {0: 0, 1: 1, 2: 2, 3: 1, 4: 3, 5: 4}
# AICD-T2 (12 classes): 4 triplet families {0,1,2}, {3,4,5}, {6,7,8}, {9,10,11}
FAMILY_OF_AICD = {i: (i // 3) for i in range(12)}


# ---- Augmentation pool ------------------------------------------------------

_RESERVED = {"if","else","elif","for","while","do","return","def","function",
    "class","struct","interface","enum","import","from","include","public",
    "private","void","new","this","extends","implements","try","catch","except",
    "finally","with","in","of","is","not","and","or","as","True","False","None",
    "null","true","false","self","int","float","double","char","long","bool",
    "string","var","let","const"}

def aug_token_dropout(code, rng, p=0.1):
    out = []
    for t in _re.split(r"(\s+|[^\w\s])", code):
        if t.strip() and t.strip() not in _RESERVED and not t.isspace():
            if rng.random() < p: out.append(" "); continue
        out.append(t)
    return "".join(out)

def aug_id_rename(code, rng, max_n=8):
    ids = [i for i in set(_re.findall(r"\b[a-zA-Z_]\w{2,}\b", code))
           if i not in _RESERVED and not i[0].isdigit()]
    if not ids: return code
    chosen = rng.sample(ids, min(max_n, len(ids)))
    new = code
    for k, orig in enumerate(chosen):
        new = _re.sub(rf"\b{_re.escape(orig)}\b", f"v{k}", new)
    return new

def aug_ws_jitter(code, rng, p=0.15):
    out = []
    for c in code:
        out.append(c)
        if c in "+-*/%=<>,;" and rng.random() < p: out.append(" ")
    return "".join(out)

def aug_comment_strip(code, rng):
    new = _re.sub(r"/\*[\s\S]*?\*/", "", code)
    new = _re.sub(r"//[^\n]*", "", new)
    new = _re.sub(r"#[^\n]*", "", new)
    return new

_AUG = [("tok_drop", aug_token_dropout), ("id_rename", aug_id_rename),
        ("ws_jit", aug_ws_jitter), ("com_strip", aug_comment_strip)]

def augment(code, rng):
    name, fn = _AUG[rng.randrange(len(_AUG))]
    try: return fn(code, rng), name
    except Exception: return code, "noop"


# ---- Model + losses ---------------------------------------------------------

class DETMULTIModel(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, emb_dim))
        self.clf = nn.Linear(emb_dim, n_cls)
        self.emb_dim, self.n_cls = emb_dim, n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        z = self.proj(sem)
        return F.normalize(z, dim=-1), self.clf(z)


def supcon_level(z_d, lvl_d, tau=0.1):
    """Flat supervised contrastive at a SINGLE granularity level (Khosla 2020).
    Positives = same level-label; negatives = everything else with unit weight.
    Used twice in DeTeCtive: once at author level, once at family level."""
    N = z_d.size(0)
    if N < 2: return z_d.sum() * 0.0
    sim = (z_d @ z_d.t()) / tau
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()
    eye = torch.eye(N, device=z_d.device, dtype=torch.bool)
    pos = (lvl_d.unsqueeze(0) == lvl_d.unsqueeze(1)).float().masked_fill(eye, 0.0)
    not_self = (~eye).float()
    exp_s = (torch.exp(sim) * not_self).clamp(min=1e-12)
    num = (exp_s * pos).sum(-1).clamp(min=1e-12)
    den = exp_s.sum(-1).clamp(min=1e-12)
    has = (pos.sum(-1) > 0).float()
    return (-(torch.log(num) - torch.log(den)) * has).sum() / has.sum().clamp(min=1.0)


# ---- Plumbing ---------------------------------------------------------------

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_auth: float = 0.35; lambda_fam: float = 0.35; tau: float = 0.1
    emb_dim: int = 256; device: str = "cuda"

def adaptive_schedule(c):
    if c.frac <= 0.02: c.epochs, c.lr_enc, c.warmup = 10, 3e-5, 0.20
    elif c.frac <= 0.10: c.epochs, c.lr_enc, c.warmup = 6, 3e-5, 0.15
    else: c.epochs, c.lr_enc, c.warmup = 6, 4e-5, 0.10
    return c

def _hw(c):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40: c.bs, c.seq = 128, 512
        elif mem >= 20: c.bs, c.seq = 96, 448
        elif mem >= 10: c.bs, c.seq = 64, 384
        else: c.bs, c.seq = 32, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={c.bs} seq={c.seq} (2-view, 2-level SupCon)")
    return c

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _is_human(t): return str(t or "").strip().lower() in {"human","human_written","human-generated"}

def _vocab(tr):
    names = {str(r.get("model", "") or "").strip() for r in tr
             if not _is_human(r.get("target", "")) and r.get("model", "")}
    return {n: i + 1 for i, n in enumerate(sorted(names))}

def _conv_codet(s, t, vocab):
    def row(r):
        code = next((r.get(f, "") for f in ("cleaned_code", "code")
                     if isinstance(r.get(f, ""), str) and r.get(f, "").strip()), "")
        lbl = (0 if _is_human(r.get("target", ""))
               else (1 if t == "binary"
                     else vocab.get(str(r.get("model", "") or "").strip(), -1)))
        return {"code": code, "label": lbl,
                "language": str(r.get("language", "")).strip().lower(),
                "source": str(r.get("source", "")).strip().lower()}
    return s.map(row, remove_columns=s.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _conv_aicd(s):
    def row(r):
        return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1)),
                "language": str(r.get("language", "")).strip().lower(), "source": ""}
    return s.map(row, remove_columns=s.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)

def _load_codet():
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]

def _load_aicd(task):
    tn = {"t1":"T1","t2":"T2","t3":"T3"}.get(task.lower())
    if tn is None: raise ValueError(f"[aicd] bad '{task}'")
    p = os.path.join(KAGGLE_AICD, tn)
    if not os.path.isdir(p): raise FileNotFoundError(f"[aicd] STRICT: {tn} not found")
    pf = sorted(glob.glob(os.path.join(p, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: no parquet")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0: return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


class FSDS_AUG(TD):
    def __init__(self, data, tok, seq, frac=1.0, seed=42, do_aug=True):
        self.data = data; self.tok = tok; self.seq = seq; self.do_aug = do_aug; self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]
        e0 = self.tok(code, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        ids0, m0 = e0["input_ids"].squeeze(0), e0["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            ca, _ = augment(code, rng)
            e1 = self.tok(ca, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
            ids1, m1 = e1["input_ids"].squeeze(0), e1["attention_mask"].squeeze(0)
        else:
            ids1, m1 = ids0, m0
        return {"ids0": ids0, "mask0": m0, "ids1": ids1, "mask1": m1,
                "label": r["label"],
                "language": r.get("language", "") or "", "source": r.get("source", "") or ""}


@torch.no_grad()
def eval_pack(model, loader, cfg):
    model.eval(); preds, labels, langs, sources = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device); labs = b["label"]
        _, lg = model.encode(ids0, m0)
        preds.extend(lg.argmax(-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        lang_b = b.get("language", []); src_b = b.get("source", [])
        langs.extend(list(lang_b) if not isinstance(lang_b, list) else lang_b)
        sources.extend(list(src_b) if not isinstance(src_b, list) else src_b)
    preds = np.array(preds); labels = np.array(labels); n = cfg.n_cls
    ov = {"accuracy": float(accuracy_score(labels, preds)),
          "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
          "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
          "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
          "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
          "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0))}
    cm = confusion_matrix(labels, preds, labels=list(range(n)))
    per_lang, per_src = {}, {}
    if any(l for l in langs):
        la = np.array(langs)
        for L in sorted(set(langs)):
            if not L: continue
            sel = (la == L)
            if sel.sum() < 2: continue
            per_lang[L] = {"n": int(sel.sum()),
                "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0))}
    if any(s for s in sources):
        sa = np.array(sources)
        for S in sorted(set(sources)):
            if not S: continue
            sel = (sa == S)
            if sel.sum() < 2: continue
            per_src[S] = {"n": int(sel.sum()),
                "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0))}
    return {"overall": ov, "confusion_matrix": cm.tolist(),
            "per_language": per_lang, "per_source": per_src,
            "n_samples": int(len(labels))}


def train_epoch(model, loader, opt, sch, scaler, cfg, family_map):
    model.train(); tot, ce_s, sa_s, sf_s = 0.0, 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); m1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        # family labels from author labels
        fams = torch.tensor([family_map.get(int(y), int(y)) for y in labs.tolist()],
                             device=cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, lg = model.encode(ids0, m0)
            z1, _ = model.encode(ids1, m1)
            zd = torch.cat([z0, z1], 0)
            yd_a = torch.cat([labs, labs], 0)
            yd_f = torch.cat([fams, fams], 0)
            loss_ce = F.cross_entropy(lg, labs)
            loss_a = supcon_level(zd, yd_a, tau=cfg.tau)
            loss_f = supcon_level(zd, yd_f, tau=cfg.tau)
            loss = loss_ce + cfg.lambda_auth * loss_a + cfg.lambda_fam * loss_f
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sa_s += loss_a.item(); sf_s += loss_f.item()
    n = len(loader)
    return tot/n, ce_s/n, sa_s/n, sf_s/n


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    family_map = FAMILY_OF_CODET if cfg.benchmark == "codet_m4" else FAMILY_OF_AICD
    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw)
        tr_data = _conv_codet(tr_raw, "author", vocab)
        vl_data = _conv_codet(vl_raw, "author", vocab)
        ts_data = _conv_codet(ts_raw, "author", vocab)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd("t2")
        tr_data = _conv_aicd(tr_raw); vl_data = _conv_aicd(vl_raw); ts_data = _conv_aicd(ts_raw)
    tok = AutoTokenizer.from_pretrained(os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True)
    tr_ds = FSDS_AUG(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS_AUG(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS_AUG(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} wu={cfg.warmup} "
                f"la={cfg.lambda_auth} lf={cfg.lambda_fam}")
    lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = DETMULTIModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh = 0.0, None, []
    for ep in range(cfg.epochs):
        loss, ce, sa, sf = train_epoch(model, tr_dl, opt, sch, scaler, cfg, family_map)
        vm = eval_pack(model, vl_dl, cfg)
        v = vm["overall"]["macro_f1"]; vh.append(v)
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce={ce:.4f} sc_auth={sa:.4f} sc_fam={sf:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    tm = eval_pack(model, ts_dl, cfg)
    test = tm["overall"]["macro_f1"]; gap = best_val - test
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f}")
    return {"tag": tag, "method": "DETMULTI", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_auth": cfg.lambda_auth, "lambda_fam": cfg.lambda_fam, "tau": cfg.tau,
            "val_macro": best_val, "macro": test,
            "weighted": tm["overall"]["weighted_f1"], "acc": tm["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "test_metrics": tm, "val_history": vh,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp91_detmulti_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time()-t0, 1); results.append(r)
                logger.info(f"[{tag}] test={r['macro']:.4f} ({r['dpaper']:+.4f}) gap={r['val_test_gap']:+.4f} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp91_detmulti_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + "=" * 110)
    print(f"{'Bench':<12} {'Frac':>6} {'Val':>8} {'Test':>8} {'Gap':>8} {'dPaper':>9} {'Wall':>8}")
    print("-" * 110)
    for r in results:
        print(f"{r['bench']:<12} {r['frac']:>6.0%} {r['val_macro']:>8.4f} {r['macro']:>8.4f} "
              f"{r['val_test_gap']:>+8.4f} {r['dpaper']:>+9.4f} {r['wall']:>8.0f}s")
    print("=" * 110)


if __name__ == "__main__":
    main()
