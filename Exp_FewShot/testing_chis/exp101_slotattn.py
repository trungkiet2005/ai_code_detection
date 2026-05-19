# exp101 — SLOTATTN: Slot Attention object-centric representation for code
# =============================================================================
# NAME       : SLOTATTN (Slot Attention for code authorship)
# REFERENCE  : new; first application of Slot Attention (Locatello et al.,
#              NeurIPS 2020, arXiv:2006.15055) to code attribution. Slot
#              Attention was developed for object-centric representation
#              learning on images; we transfer it to TEXT/code tokens.
# CLAIM      : Author identity in code is COMPOSITIONAL: each generator's
#              fingerprint mixes control-flow patterns, identifier style,
#              comment habits, and surface-token preferences. A SINGLE
#              pooled vector flattens these factors. Slot Attention
#              decomposes the token sequence into K=8 learnable "slots"
#              that COMPETE to bind to subsets of tokens, producing an
#              object-centric representation. The classifier reads the
#              K slot vectors (or their pool); the contrastive head reads
#              the per-slot embeddings.
# EQUATION   : 1. Encoder gives token features F in R^{L x h}.
#              2. Slot Attention iterates T=3 times:
#                    slots <- LayerNorm(slots)
#                    attn = softmax(F K^T Q / sqrt(d), dim=slot)
#                    attn = attn / attn.sum(token-dim)
#                    updates = attn^T V
#                    slots <- GRU(updates, slots) + MLP(LN(updates))
#              3. Pool slots: z = mean_k slot_k  (for classification).
#              4. Contrastive head: stack all K slot vectors as the
#                 representation; the doubled batch is 2*B*K vectors.
# WHY NEW    : Slot Attention has been used in vision (object discovery),
#              video segmentation, and recently for sentence-level
#              compositionality (Chang 2023). No prior work applies it
#              to AI-code attribution. The bridge -- treating each LLM's
#              style as a set of compositional "fingerprint slots" -- is
#              novel to our knowledge.
# FALSIFIER  : (i) Average attention entropy across K slots > log(K)/2:
#                  slots actually specialise on different token subsets,
#                  not collapse to identical attention.
#              (ii) Per-slot ablation drops Macro-F1 by at least 0.005 at
#                  1pct: every slot carries non-trivial information.
#              (iii) Macro-F1 > TRACO at extreme few-shot.
# GPU TUNING : K=8 slots, slot_dim=128, T=3 iterations. Extra params: ~150K.
#              Memory: K slots adds ~K x B x slot_dim = small for K=8.
# =============================================================================
from __future__ import annotations
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"
import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
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
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup
from tqdm import tqdm
from torch.cuda.amp import GradScaler
warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp101_slotattn")

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

# Augmentation pool (TRACO)
_RESERVED = {"if","else","elif","for","while","do","return","def","function","class",
    "struct","interface","enum","import","from","include","public","private","void",
    "new","this","extends","implements","try","catch","except","finally","with","in",
    "of","is","not","and","or","as","True","False","None","null","true","false","self",
    "int","float","double","char","long","bool","string","var","let","const"}
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
    chosen = rng.sample(ids, min(max_n, len(ids))); new = code
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
_AUG = [aug_token_dropout, aug_id_rename, aug_ws_jitter, aug_comment_strip]
def augment(code, rng):
    try: return _AUG[rng.randrange(len(_AUG))](code, rng)
    except Exception: return code


# ===== Slot Attention module (Locatello 2020, adapted for text) =====

class SlotAttention(nn.Module):
    """Iterative slot attention. Takes token features F (B, L, d) and a
    starting slot distribution; produces slot features S (B, K, slot_dim).

    Each iteration:
        slots = LayerNorm(slots)
        q = W_q slots; k, v = W_k F, W_v F
        attn(i,j) = softmax_j(q_i . k_j / sqrt(d_attn))
        attn(i,j) = attn(i,j) / sum_i attn(i,j)        # normalise over slots
        u = attn^T v
        slots = GRU(u, prev_slots) + MLP(LayerNorm(u))
    """
    def __init__(self, d_in, slot_dim, K=8, T=3):
        super().__init__()
        self.K, self.T, self.slot_dim = K, T, slot_dim
        self.norm_in = nn.LayerNorm(d_in)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_pre_mlp = nn.LayerNorm(slot_dim)
        self.q = nn.Linear(slot_dim, slot_dim, bias=False)
        self.k = nn.Linear(d_in, slot_dim, bias=False)
        self.v = nn.Linear(d_in, slot_dim, bias=False)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(nn.Linear(slot_dim, 2 * slot_dim), nn.GELU(),
                                  nn.Linear(2 * slot_dim, slot_dim))
        # Learnable slot init distribution (mu, log-sigma)
        self.slot_mu = nn.Parameter(torch.randn(1, K, slot_dim) * 0.1)
        self.slot_logsigma = nn.Parameter(torch.zeros(1, K, slot_dim) - 1.0)
        self.scale = slot_dim ** -0.5

    def forward(self, F_in, mask=None):
        """F_in: B, L, d_in;  mask: B, L (1 for valid, 0 for pad)."""
        B, L, _ = F_in.shape
        F_in = self.norm_in(F_in)
        # Init slots: gaussian centred at learnable mu, scaled by exp(log-sigma)
        sigma = self.slot_logsigma.exp()
        slots = self.slot_mu + sigma * torch.randn(B, self.K, self.slot_dim, device=F_in.device)
        K_proj = self.k(F_in)                                    # B, L, d
        V_proj = self.v(F_in)                                    # B, L, d
        attn_record = None
        for t in range(self.T):
            slots_prev = slots
            slots = self.norm_slots(slots)
            Q_proj = self.q(slots)                               # B, K, d
            dots = torch.einsum("bld,bkd->blk", K_proj, Q_proj) * self.scale  # B, L, K
            if mask is not None:
                dots = dots.masked_fill(~mask.bool().unsqueeze(-1), -1e9)
            # Softmax over slots (each token "votes" for a slot)
            attn = F.softmax(dots, dim=-1)                       # B, L, K
            # Normalise over tokens (per slot)
            denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
            attn_norm = attn / denom                              # B, L, K
            # Updates: K slots aggregate from tokens
            updates = torch.einsum("blk,bld->bkd", attn_norm, V_proj)
            # GRU update + MLP
            slots_flat = updates.reshape(-1, self.slot_dim)
            prev_flat = slots_prev.reshape(-1, self.slot_dim)
            new_slots = self.gru(slots_flat, prev_flat).reshape(B, self.K, self.slot_dim)
            slots = new_slots + self.mlp(self.norm_pre_mlp(new_slots))
            attn_record = attn
        return slots, attn_record   # B, K, slot_dim; B, L, K


class SlotAttnModel(nn.Module):
    def __init__(self, enc_name, n_cls, K_slots=8, slot_dim=128, slot_T=3, emb_dim=256):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.slot_attn = SlotAttention(d_in=h, slot_dim=slot_dim, K=K_slots, T=slot_T)
        self.proj = nn.Sequential(nn.Linear(slot_dim, emb_dim), nn.GELU(),
                                   nn.Dropout(0.1), nn.Linear(emb_dim, emb_dim))
        # Classifier: pool over K slots then linear
        self.clf = nn.Linear(emb_dim, n_cls)
        self.K_slots, self.slot_dim, self.emb_dim, self.n_cls = K_slots, slot_dim, emb_dim, n_cls

    def encode(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        F_tok = out.last_hidden_state                            # B, L, h
        slots, attn = self.slot_attn(F_tok, mask)                # B, K, slot_dim
        slot_feats = self.proj(slots)                            # B, K, emb_dim
        slot_feats = F.normalize(slot_feats, dim=-1)
        # Classification: mean-pool over slots
        pooled = slot_feats.mean(dim=1)                          # B, emb_dim
        pooled = F.normalize(pooled, dim=-1)
        logits = self.clf(pooled)
        return pooled, slot_feats, logits, attn


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
    K_slots: int = 8; slot_dim: int = 128; slot_T: int = 3
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
        # Slot attention adds modest overhead; 2-view contrastive doubled batch.
        if mem >= 40: c.bs, c.seq = 96, 512
        elif mem >= 20: c.bs, c.seq = 72, 448
        elif mem >= 10: c.bs, c.seq = 48, 384
        else: c.bs, c.seq = 24, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={c.bs} seq={c.seq} K={c.K_slots} T={c.slot_T}")
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


class FSDS(TD):
    def __init__(self, data, tok, seq, frac=1.0, seed=42, do_aug=True):
        self.data = data; self.tok = tok; self.seq = seq; self.do_aug = do_aug; self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed); labels = list(range(max(self.data["label"]) + 1)); keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]
        e0 = self.tok(code, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        ids0, m0 = e0["input_ids"].squeeze(0), e0["attention_mask"].squeeze(0)
        if self.do_aug:
            rng = random.Random(self.seed + i * 7919 + int(time.time_ns() % 1_000_000))
            ca = augment(code, rng)
            e1 = self.tok(ca, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
            ids1, m1 = e1["input_ids"].squeeze(0), e1["attention_mask"].squeeze(0)
        else:
            ids1, m1 = ids0, m0
        return {"ids0": ids0, "mask0": m0, "ids1": ids1, "mask1": m1, "label": r["label"]}


@torch.no_grad()
def eval_pack(model, loader, cfg, collect_attn=False):
    model.eval(); preds, labels = [], []
    slot_entropies = []
    for b in tqdm(loader, desc="Eval"):
        ids = b["ids0"].to(cfg.device); m = b["mask0"].to(cfg.device); labs = b["label"]
        _, slot_feats, lg, attn = model.encode(ids, m)
        preds.extend(lg.argmax(-1).cpu().tolist())
        labels.extend(labs.tolist() if torch.is_tensor(labs) else list(labs))
        if collect_attn and attn is not None:
            # Per-slot attention entropy over tokens
            # attn: B, L, K -> normalise per slot over L, compute entropy
            denom = attn.sum(dim=1, keepdim=True).clamp(min=1e-8)
            attn_per_slot = (attn / denom)             # B, L, K
            ent = -(attn_per_slot * torch.log(attn_per_slot.clamp(min=1e-12))).sum(dim=1)  # B, K
            slot_entropies.append(ent.cpu().float())
    preds = np.array(preds); labels = np.array(labels)
    macro = float(f1_score(labels, preds, average="macro", zero_division=0))
    weighted = float(f1_score(labels, preds, average="weighted", zero_division=0))
    acc = float(accuracy_score(labels, preds))
    out = {"macro_f1": macro, "weighted_f1": weighted, "accuracy": acc}
    if slot_entropies:
        ents = torch.cat(slot_entropies, dim=0).numpy()
        out["per_slot_entropy_mean"] = ents.mean(axis=0).tolist()
        out["per_slot_entropy_std"] = ents.std(axis=0).tolist()
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg, dist):
    model.train(); tot, ce_s, sc_s = 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids0 = b["ids0"].to(cfg.device); m0 = b["mask0"].to(cfg.device)
        ids1 = b["ids1"].to(cfg.device); m1 = b["mask1"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            z0, slot0, lg, _ = model.encode(ids0, m0)            # z0: B, d
            z1, slot1, _, _ = model.encode(ids1, m1)
            # Contrastive head: stack all K slots per sample.
            B, K, D = slot0.shape
            slot_d = torch.cat([slot0.reshape(B*K, D), slot1.reshape(B*K, D)], dim=0)
            y_slot = torch.cat([labs.repeat_interleave(K), labs.repeat_interleave(K)], dim=0)
            loss_ce = F.cross_entropy(lg, labs)
            loss_sc = supcon_tw_loss(slot_d, y_slot, dist, gamma=cfg.gamma, tau=cfg.tau)
            loss = loss_ce + cfg.lambda_aug * loss_sc
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_s += loss_ce.item(); sc_s += loss_sc.item()
    n = len(loader)
    return tot/n, ce_s/n, sc_s/n


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
    tr_ds = FSDS(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed, do_aug=True)
    vl_ds = FSDS(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+1, do_aug=False)
    ts_ds = FSDS(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed+2, do_aug=False)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = SlotAttnModel(cfg.enc, cfg.n_cls, cfg.K_slots, cfg.slot_dim, cfg.slot_T, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh = 0.0, None, []
    for ep in range(cfg.epochs):
        loss, ce, sc = train_epoch(model, tr_dl, opt, sch, scaler, cfg, dist)
        vm = eval_pack(model, vl_dl, cfg)
        v = vm["macro_f1"]; vh.append(v)
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce={ce:.4f} sc={sc:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    tm = eval_pack(model, ts_dl, cfg, collect_attn=True)
    test = tm["macro_f1"]; gap = best_val - test
    ent_mean = tm.get("per_slot_entropy_mean", [])
    log_K = float(np.log(cfg.K_slots)) if cfg.K_slots > 1 else 1.0
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f} "
                f"per_slot_entropy_mean={[round(e, 2) for e in ent_mean]} (log K={log_K:.2f})")
    return {"tag": tag, "method": "SLOTATTN", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "K_slots": cfg.K_slots, "slot_dim": cfg.slot_dim, "slot_T": cfg.slot_T,
            "lambda_aug": cfg.lambda_aug, "gamma": cfg.gamma, "tau": cfg.tau,
            "val_macro": best_val, "macro": test, "weighted": tm["weighted_f1"],
            "acc": tm["accuracy"], "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "per_slot_entropy_mean": ent_mean,
            "per_slot_entropy_std": tm.get("per_slot_entropy_std", []),
            "log_K": log_K,
            "val_history": vh, "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]; results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp101_slotattn_unixcoder-base_{bench}_f{frac}"
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
    with open(os.path.join(out, "exp101_slotattn_results.json"), "w") as f: json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
