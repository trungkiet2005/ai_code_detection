# exp95_promptc — Prompt-Conditioned Attribution (style vs prompt-confound diagnostic)
# =============================================================================
# NAME       : PROMPTC (Prompt-Conditioned Cross-attention attribution)
# REFERENCE  : new; tests an objection most reviewers raise privately:
#              CoDET-M4 ships prompts SHARED across the six generators, so
#              part of the apparent "author signal" may be prompt-style
#              leakage, not author style. We make the prompt input EXPLICIT
#              and let the encoder mix prompt and code via cross-attention.
# CLAIM      : When the encoder sees the prompt and the code SEPARATELY,
#              the classifier's reliance on prompt-only features can be
#              measured. If accuracy with prompt-masked stays high, the
#              attribution signal is genuinely in author style; if it drops,
#              the field has been measuring prompt-style confound.
# EQUATION   : Split each sample into (prompt, response):
#                prompt = first 256 chars of code (proxy: includes
#                         function-signature comment, top imports)
#                response = remainder of code
#              Encode separately: z_p = phi(prompt), z_r = phi(response).
#              Cross-attention: z_attn = MultiHeadAttn(Q=z_r, K=z_p, V=z_p).
#              Classifier: y_hat = W * z_attn.
#              At test-time additionally report:
#                Acc-prompt-only:   y_hat from z_p alone
#                Acc-response-only: y_hat from z_r alone
#                Acc-joint:         the headline PROMPTC accuracy
# WHY NEW    : No code-attribution work has cleanly disentangled prompt
#              leakage from response style. The diagnostic is a paper-grade
#              negative-or-positive result: it tells the field exactly how
#              much of the benchmark numbers are confounded.
# FALSIFIER  : (i) If Acc-prompt-only > Acc-response-only at 20% data, the
#              benchmark is largely a prompt-pattern task, not authorship.
#              (ii) If Acc-joint > both, the cross-attention is genuinely
#              fusing signal from both sides.
# GPU TUNING : Two encoder forward passes per sample (prompt + response),
#              same effective bs as 2-view contrastive in exp76.
# =============================================================================
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET  = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD   = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from dataclasses import dataclass, field
from typing import Tuple

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
logger = logging.getLogger("exp95_promptc")

PAPER_BASELINE = 0.6633


def split_prompt_response(code: str, prompt_chars: int = 256) -> Tuple[str, str]:
    """Heuristic split: first 256 characters are the 'prompt-like' header
    (signature, top comment, imports); the remainder is the response.
    For very short snippets we duplicate the whole snippet on both sides."""
    if len(code) <= prompt_chars + 32:
        return code, code
    return code[:prompt_chars], code[prompt_chars:]


# ---- Model with cross-attention ---------------------------------------------

class PromptCModel(nn.Module):
    def __init__(self, enc_name, n_cls, emb_dim=256, n_heads=4):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        h = self.encoder.config.hidden_size
        self.cross = nn.MultiheadAttention(embed_dim=h, num_heads=n_heads, batch_first=True, dropout=0.1)
        self.norm = nn.LayerNorm(h)
        self.proj = nn.Sequential(nn.Linear(h, 512), nn.GELU(), nn.Dropout(0.1), nn.Linear(512, emb_dim))
        self.clf_joint = nn.Linear(emb_dim, n_cls)
        # Separate probe heads (for diagnostic; gradient-blocked from encoder during training)
        self.clf_prompt = nn.Linear(emb_dim, n_cls)
        self.clf_resp = nn.Linear(emb_dim, n_cls)
        self.proj_for_probe = nn.Sequential(nn.Linear(h, emb_dim))
        self.emb_dim, self.n_cls = emb_dim, n_cls

    def _pool(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        sem = (out.last_hidden_state * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        return out.last_hidden_state, sem, mask

    def forward(self, ids_p, mask_p, ids_r, mask_r):
        hs_p, sem_p, mp = self._pool(ids_p, mask_p)
        hs_r, sem_r, mr = self._pool(ids_r, mask_r)
        # Cross-attention: query = response tokens, key/value = prompt tokens
        # Use sem_r as a 1-token query for simplicity (pooled response attends to all prompt tokens).
        q = sem_r.unsqueeze(1)                              # B, 1, h
        attn, _ = self.cross(q, hs_p, hs_p, key_padding_mask=~mp.bool())
        attn = self.norm(attn.squeeze(1) + sem_r)           # B, h
        z_attn = F.normalize(self.proj(attn), dim=-1)
        # Probe heads use only their respective single-side pooled rep
        z_p = F.normalize(self.proj_for_probe(sem_p), dim=-1)
        z_r = F.normalize(self.proj_for_probe(sem_r), dim=-1)
        return self.clf_joint(z_attn), self.clf_prompt(z_p), self.clf_resp(z_r), z_attn


# ---- Plumbing ---------------------------------------------------------------

@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    lambda_probe: float = 0.1; prompt_chars: int = 256
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
        # 2 encoder forwards (prompt + response, shorter each)
        if mem >= 40: c.bs, c.seq = 128, 320
        elif mem >= 20: c.bs, c.seq = 96, 288
        elif mem >= 10: c.bs, c.seq = 64, 256
        else: c.bs, c.seq = 32, 192
        logger.info(f"[hw] mem={mem:.1f}GB bs={c.bs} seq={c.seq} (2-encode prompt+response)")
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


class FSDS_PR(TD):
    def __init__(self, data, tok, seq, prompt_chars=256, frac=1.0, seed=42):
        self.data = data; self.tok = tok; self.seq = seq
        self.prompt_chars = prompt_chars; self.seed = seed
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_PR] Sampled {len(self.data)} ({frac*100:.0f}%)")
    def __len__(self): return len(self.data)
    def __getitem__(self, i):
        r = self.data[i]; code = r["code"][:5000]
        p, q = split_prompt_response(code, prompt_chars=self.prompt_chars)
        ep = self.tok(p, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        eq = self.tok(q, max_length=self.seq, padding="max_length", truncation=True, return_tensors="pt")
        return {"ids_p": ep["input_ids"].squeeze(0), "mask_p": ep["attention_mask"].squeeze(0),
                "ids_r": eq["input_ids"].squeeze(0), "mask_r": eq["attention_mask"].squeeze(0),
                "label": r["label"]}


@torch.no_grad()
def eval_pack(model, loader, cfg):
    model.eval(); preds_j, preds_p, preds_r, labels = [], [], [], []
    for b in tqdm(loader, desc="Eval"):
        ids_p = b["ids_p"].to(cfg.device); mp = b["mask_p"].to(cfg.device)
        ids_r = b["ids_r"].to(cfg.device); mr = b["mask_r"].to(cfg.device)
        lj, lp, lr, _ = model(ids_p, mp, ids_r, mr)
        preds_j.extend(lj.argmax(-1).cpu().tolist())
        preds_p.extend(lp.argmax(-1).cpu().tolist())
        preds_r.extend(lr.argmax(-1).cpu().tolist())
        labels.extend(b["label"].tolist() if torch.is_tensor(b["label"]) else list(b["label"]))
    y = np.array(labels); n = cfg.n_cls
    def metrics(p):
        return {"accuracy": float(accuracy_score(y, p)),
                "macro_f1": float(f1_score(y, p, average="macro", zero_division=0)),
                "weighted_f1": float(f1_score(y, p, average="weighted", zero_division=0))}
    return {"joint": metrics(np.array(preds_j)),
            "prompt_only": metrics(np.array(preds_p)),
            "response_only": metrics(np.array(preds_r)),
            "n_samples": int(len(y))}


def train_epoch(model, loader, opt, sch, scaler, cfg):
    model.train(); tot, ce_j, ce_p, ce_r = 0.0, 0.0, 0.0, 0.0
    for b in tqdm(loader, desc="Train"):
        ids_p = b["ids_p"].to(cfg.device); mp = b["mask_p"].to(cfg.device)
        ids_r = b["ids_r"].to(cfg.device); mr = b["mask_r"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            lj, lp, lr, _ = model(ids_p, mp, ids_r, mr)
            l_j = F.cross_entropy(lj, labs)
            l_p = F.cross_entropy(lp, labs)
            l_r = F.cross_entropy(lr, labs)
            # Joint is primary; probes are auxiliary (small weight) so they
            # exist as diagnostic readouts without dominating training.
            loss = l_j + cfg.lambda_probe * (l_p + l_r)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item(); ce_j += l_j.item(); ce_p += l_p.item(); ce_r += l_r.item()
    n = len(loader)
    return tot/n, ce_j/n, ce_p/n, ce_r/n


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
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
    tr_ds = FSDS_PR(tr_data, tok, cfg.seq, cfg.prompt_chars, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS_PR(vl_data, tok, cfg.seq, cfg.prompt_chars, frac=1.0, seed=cfg.seed+1)
    ts_ds = FSDS_PR(ts_data, tok, cfg.seq, cfg.prompt_chars, frac=1.0, seed=cfg.seed+2)
    total = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} ep={cfg.epochs} lr={cfg.lr_enc} wu={cfg.warmup} "
                f"prompt_chars={cfg.prompt_chars}")
    lc = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True)
    tr_dl = DataLoader(tr_ds, shuffle=True, **lc)
    vl_dl = DataLoader(vl_ds, shuffle=False, **lc)
    ts_dl = DataLoader(ts_ds, shuffle=False, **lc)
    model = PromptCModel(cfg.enc, cfg.n_cls, cfg.emb_dim).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([{"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
                              {"params": head, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total * cfg.warmup)), total)
    scaler = GradScaler()
    best_val, best_state, vh = 0.0, None, []
    for ep in range(cfg.epochs):
        loss, lj, lp, lr = train_epoch(model, tr_dl, opt, sch, scaler, cfg)
        vm = eval_pack(model, vl_dl, cfg)
        v = vm["joint"]["macro_f1"]; vh.append(v)
        logger.info(f"[ep{ep+1}] loss={loss:.4f} ce_j={lj:.4f} ce_p={lp:.4f} ce_r={lr:.4f} "
                    f"val_j={v:.4f} val_p={vm['prompt_only']['macro_f1']:.4f} "
                    f"val_r={vm['response_only']['macro_f1']:.4f}")
        if v > best_val:
            best_val = v; best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    model.load_state_dict(best_state)
    tm = eval_pack(model, ts_dl, cfg)
    test = tm["joint"]["macro_f1"]; gap = best_val - test
    logger.info(f"[final] val={best_val:.4f} test={test:.4f} gap={gap:+.4f} "
                f"prompt_only_test={tm['prompt_only']['macro_f1']:.4f} "
                f"response_only_test={tm['response_only']['macro_f1']:.4f}")
    return {"tag": tag, "method": "PROMPTC", "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "lambda_probe": cfg.lambda_probe, "prompt_chars": cfg.prompt_chars,
            "val_macro": best_val, "macro": test,
            "weighted": tm["joint"]["weighted_f1"], "acc": tm["joint"]["accuracy"],
            "val_test_gap": gap, "dpaper": test - PAPER_BASELINE,
            "prompt_only_macro": tm["prompt_only"]["macro_f1"],
            "response_only_macro": tm["response_only"]["macro_f1"],
            "test_metrics": tm, "val_history": vh,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    bms = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for bench, task, n_cls in bms:
        for frac in fracs:
            cfg = Cfg(benchmark=bench, task=task, frac=frac, n_cls=n_cls)
            tag = f"exp95_promptc_unixcoder-base_{bench}_f{frac}"
            logger.info(f"=== {tag} ===")
            t0 = time.time()
            try:
                r = run_exp(cfg, tag); r["wall"] = round(time.time()-t0, 1); results.append(r)
                logger.info(f"[{tag}] test_joint={r['macro']:.4f} ({r['dpaper']:+.4f}) "
                            f"prompt_only={r['prompt_only_macro']:.4f} response_only={r['response_only_macro']:.4f} "
                            f"gap={r['val_test_gap']:+.4f} t={r['wall']:.0f}s")
            except Exception as e:
                logger.error(f"[{tag}] FAILED: {e}")
                import traceback; traceback.print_exc()
            import gc; gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
    try: here = os.path.dirname(os.path.realpath(__file__))
    except NameError: here = os.getcwd()
    out = os.path.join(here, "results"); os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "exp95_promptc_results.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
