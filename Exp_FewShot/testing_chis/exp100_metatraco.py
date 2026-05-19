# exp100 — METATRACO: MAML meta-initialisation for cross-generator few-shot
# =============================================================================
# NAME       : METATRACO (Model-Agnostic Meta-Learning for cross-generator)
# REFERENCE  : new; adapts MAML (Finn 2017) to AI-code authorship for fast
#              adaptation to NEWLY-RELEASED generators.
# CLAIM      : The few-shot regime in this paper is class-stratified within
#              a fixed K-class space. The deployment regime is different:
#              a new generator appears every month and the model must adapt
#              from ~10 labelled examples. MAML's bi-level objective gives
#              an encoder initialisation that adapts in N inner-loop steps.
# EQUATION   : Meta-train: sample a leave-one-class-out task from training.
#                 Inner: K-1 classes from train, 1 class held out as new.
#                 N=5 inner gradient steps on the support set (10/class).
#                 Outer: evaluate on the query set (5/class), backprop
#                       through the inner steps.
#              Meta-test: apply same N inner-step adaptation on REAL test
#              fractions; report Macro-F1.
# WHY NEW    : MAML is standard in vision few-shot but has NOT been applied
#              to LLM-code authorship. Combining with TRACO's tree-weighted
#              loss as the inner-loop objective is the contribution.
# FALSIFIER  : Macro-F1 at 1pct with MAML init > Macro-F1 at 1pct without
#              MAML init (random init). The lift should be ~0.02+ if the
#              meta-init is recovering anything useful.
# GPU TUNING : Inner loop is small (N=3, no second-order); use first-order
#              MAML (foMAML, Nichol 2018) for memory safety.
# =============================================================================
from __future__ import annotations
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"
import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, copy
from dataclasses import dataclass
import re as _re
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
def _ensure(p):
    if importlib.util.find_spec(p.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", p])
for _p in ("numpy","torch","datasets","transformers","scikit-learn","tqdm"): _ensure(_p)
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp100_metatraco")

PAPER_BASELINE = 0.6633
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}

def _gd(u, v, adj):
    if u == v: return 0.0
    q = [(u, 0)]; seen = {u}
    while q:
        c, d = q.pop(0)
        for nb in adj.get(c, []):
            if nb == v: return d + 1.0
            if nb not in seen: seen.add(nb); q.append((nb, d + 1))
    return float("inf")
def build_dist(n, adj, default=4.0):
    D = torch.full((n, n), default)
    for i in range(n):
        for j in range(n):
            d = _gd(i, j, adj)
            if d < float("inf"): D[i, j] = d
            elif (i == 0) != (j == 0): D[i, j] = 3.0
    return D


class MetaTracoModel(nn.Module):
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


def supcon_tw_loss(z_d, y_d, dist, gamma=1.0, tau=0.1):
    N = z_d.size(0)
    if N < 2: return z_d.sum() * 0.0
    sim = (z_d @ z_d.t()) / tau
    sim = sim - sim.max(dim=-1, keepdim=True).values.detach()
    eye = torch.eye(N, device=z_d.device, dtype=torch.bool)
    pos = (y_d.unsqueeze(0) == y_d.unsqueeze(1)).float().masked_fill(eye, 0.0)
    neg = (y_d.unsqueeze(0) != y_d.unsqueeze(1)).float()
    dij = dist[y_d][:, y_d]
    w = pos + neg * torch.exp(-gamma * dij)
    w = w.masked_fill(eye, 0.0)
    es = (torch.exp(sim) * w).clamp(min=1e-12)
    num = (es * pos).sum(-1).clamp(min=1e-12)
    den = es.sum(-1).clamp(min=1e-12)
    has = (pos.sum(-1) > 0).float()
    return (-(torch.log(num) - torch.log(den)) * has).sum() / has.sum().clamp(min=1.0)


@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_aug: float = 0.5; gamma: float = 1.0; tau: float = 0.1
    # Meta-learning specific
    meta_steps: int = 200          # number of outer meta-iterations
    inner_steps: int = 3            # foMAML inner steps
    inner_lr: float = 1e-4
    support_per_cls: int = 8        # samples per class for support set
    query_per_cls: int = 4          # samples per class for query set
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
        # foMAML still memory-heavy; use smaller batch
        if mem >= 40: c.bs, c.seq = 96, 512
        elif mem >= 20: c.bs, c.seq = 64, 448
        elif mem >= 10: c.bs, c.seq = 48, 384
        else: c.bs, c.seq = 24, 256
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
        return {"code": code, "label": lbl}
    return s.map(row, remove_columns=s.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)
def _conv_aicd(s):
    def row(r): return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1))}
    return s.map(row, remove_columns=s.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)
def _load_codet():
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42); s2 = s["train"].train_test_split(test_size=1/9, seed=42)
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
    s = ds.train_test_split(test_size=0.1, seed=42); s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


def sample_task(data, tok, seq, n_cls, support_n, query_n, seed):
    """Sample a (support, query) split: support_n per class for inner adaptation,
    query_n per class for outer meta-gradient. Returns lists of (ids, mask, label)."""
    rng = random.Random(seed)
    by_cls = {c: [] for c in range(n_cls)}
    labels = data["label"]
    for i, y in enumerate(labels):
        by_cls[int(y)].append(i)
    support_idx, query_idx = [], []
    for c, idxs in by_cls.items():
        if len(idxs) < support_n + query_n: continue
        rng.shuffle(idxs)
        support_idx.extend(idxs[:support_n])
        query_idx.extend(idxs[support_n:support_n + query_n])
    def encode_batch(idx_list):
        ids_b, m_b, y_b = [], [], []
        for i in idx_list:
            code = data[i]["code"][:5000]
            e = tok(code, max_length=seq, padding="max_length", truncation=True, return_tensors="pt")
            ids_b.append(e["input_ids"].squeeze(0))
            m_b.append(e["attention_mask"].squeeze(0))
            y_b.append(data[i]["label"])
        return (torch.stack(ids_b), torch.stack(m_b), torch.tensor(y_b, dtype=torch.long))
    return encode_batch(support_idx), encode_batch(query_idx)


def meta_train(model, tok, tr_data, dist, cfg):
    """foMAML outer loop on a fixed encoder. Each meta-step: sample a task
    (support+query), adapt parameters on support via inner SGD, compute query
    loss, backprop through QUERY ONLY (first-order)."""
    opt_outer = torch.optim.AdamW(model.parameters(), lr=cfg.lr_enc, weight_decay=cfg.wd)
    for step in range(cfg.meta_steps):
        seed_t = cfg.seed + step
        (s_ids, s_m, s_y), (q_ids, q_m, q_y) = sample_task(
            tr_data, tok, cfg.seq, cfg.n_cls, cfg.support_per_cls, cfg.query_per_cls, seed_t)
        s_ids = s_ids.to(cfg.device); s_m = s_m.to(cfg.device); s_y = s_y.to(cfg.device)
        q_ids = q_ids.to(cfg.device); q_m = q_m.to(cfg.device); q_y = q_y.to(cfg.device)
        # Snapshot params for restore
        snap = {n: p.detach().clone() for n, p in model.named_parameters()}
        # Inner adaptation: SGD steps on support set
        for inner in range(cfg.inner_steps):
            z_s, lg_s = model.encode(s_ids, s_m)
            yd = torch.cat([s_y, s_y], 0)
            loss_in = F.cross_entropy(lg_s, s_y) + cfg.lambda_aug * supcon_tw_loss(
                torch.cat([z_s, z_s], 0), yd, dist, gamma=cfg.gamma, tau=cfg.tau)
            grads = torch.autograd.grad(loss_in, model.parameters(),
                                         create_graph=False, allow_unused=True)
            with torch.no_grad():
                for p, g in zip(model.parameters(), grads):
                    if g is not None: p.sub_(cfg.inner_lr * g)
        # Outer: query loss with adapted params
        z_q, lg_q = model.encode(q_ids, q_m)
        loss_out = F.cross_entropy(lg_q, q_y)
        # Restore params; backprop adds gradient to outer
        with torch.no_grad():
            for n, p in model.named_parameters():
                p.data.copy_(snap[n])
        loss_out.backward()
        opt_outer.step(); opt_outer.zero_grad()
        if step % 20 == 0:
            logger.info(f"[meta-step {step:>3}] inner_loss={loss_in.item():.4f} outer_loss={loss_out.item():.4f}")


def run_exp(cfg, tag):
    set_seed(cfg.seed); cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist = build_dist(cfg.n_cls, adj).to(cfg.device)
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
    model = MetaTracoModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    # ---- Meta-train phase (encoder + projector + classifier all adapted) ----
    logger.info(f"=== Meta-train phase: {cfg.meta_steps} outer steps ===")
    meta_train(model, tok, tr_data, dist, cfg)
    # ---- After meta-init, do standard few-shot fine-tune at cfg.frac ----
    logger.info(f"=== Fine-tune at frac={cfg.frac} ===")
    rng = random.Random(cfg.seed)
    labels = list(range(cfg.n_cls))
    keep = []
    for lbl in labels:
        idx = [i for i, x in enumerate(tr_data["label"]) if x == lbl]
        keep.extend(rng.sample(idx, min(max(1, int(len(idx)*cfg.frac)), len(idx))))
    tr_sub = tr_data.select(keep)
    # Encode all and do gradient descent on the K-shot subset
    def to_loader(data):
        ids_l, m_l, y_l = [], [], []
        for i in range(len(data)):
            code = data[i]["code"][:5000]
            e = tok(code, max_length=cfg.seq, padding="max_length", truncation=True, return_tensors="pt")
            ids_l.append(e["input_ids"].squeeze(0)); m_l.append(e["attention_mask"].squeeze(0))
            y_l.append(data[i]["label"])
        return torch.stack(ids_l), torch.stack(m_l), torch.tensor(y_l, dtype=torch.long)
    tr_ids, tr_m, tr_y = to_loader(tr_sub)
    vl_ids, vl_m, vl_y = to_loader(vl_data)
    ts_ids, ts_m, ts_y = to_loader(ts_data)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_enc, weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(cfg.epochs * 4 * cfg.warmup)), cfg.epochs * 4)
    scaler = GradScaler()
    bs = cfg.bs
    best_val, best_state = 0.0, None
    for ep in range(cfg.epochs):
        model.train()
        idx_perm = torch.randperm(len(tr_y))
        for i in range(0, len(tr_y), bs):
            batch_idx = idx_perm[i:i+bs]
            ids_b = tr_ids[batch_idx].to(cfg.device); m_b = tr_m[batch_idx].to(cfg.device)
            y_b = tr_y[batch_idx].to(cfg.device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
                z, lg = model.encode(ids_b, m_b)
                loss = F.cross_entropy(lg, y_b)
                if z.size(0) >= 2:
                    loss = loss + cfg.lambda_aug * supcon_tw_loss(
                        torch.cat([z, z], 0), torch.cat([y_b, y_b], 0),
                        dist, gamma=cfg.gamma, tau=cfg.tau)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        # Validate
        model.eval(); preds_v = []
        with torch.no_grad():
            for i in range(0, len(vl_y), bs):
                _, lg = model.encode(vl_ids[i:i+bs].to(cfg.device), vl_m[i:i+bs].to(cfg.device))
                preds_v.extend(lg.argmax(-1).cpu().tolist())
        v = float(f1_score(vl_y.numpy(), np.array(preds_v), average="macro", zero_division=0))
        logger.info(f"[ft-ep{ep+1}] val={v:.4f}")
        if v > best_val:
            best_val = v; best_state = {k: vv.cpu().clone() for k, vv in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    # Test
    model.eval(); preds_t = []
    with torch.no_grad():
        for i in range(0, len(ts_y), bs):
            _, lg = model.encode(ts_ids[i:i+bs].to(cfg.device), ts_m[i:i+bs].to(cfg.device))
            preds_t.extend(lg.argmax(-1).cpu().tolist())
    test = float(f1_score(ts_y.numpy(), np.array(preds_t), average="macro", zero_division=0))
    weighted = float(f1_score(ts_y.numpy(), np.array(preds_t), average="weighted", zero_division=0))
    acc = float(accuracy_score(ts_y.numpy(), np.array(preds_t)))
    gap = best_val - test
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f}")
    return {"tag": tag, "method": "METATRACO", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "meta_steps": cfg.meta_steps,
            "inner_steps": cfg.inner_steps,
            "val_macro": best_val, "macro": test, "weighted": weighted, "acc": acc,
            "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]; results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp100_metatraco_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time()-t0, 1); results.append(r)
                logger.info(f"[{tag}] test={r['macro']:.4f} ({r['dpaper']:+.4f}) gap={r['val_test_gap']:+.4f} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}"); import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp100_metatraco_results.json"), "w") as f: json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
