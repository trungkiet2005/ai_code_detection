"""
================================================================================
Theory-Track exp -- Graph Neural Attribution (GNA):
Modeling generator relationships as a graph for message-passing attribution.

ARXIV_ID      : NeurIPS 2017 GCN (1611.07308); ICLR 2019 Graph Networks (1806.01261)
NAME          : Graph Neural Attribution (GNA)
ONE-LINE CLAIM: Explicitly modeling generator family relationships as a graph
                and propagating embeddings through message-passing improves attribution.
EQUATION      : h_v^{(l+1)} = σ(W^{(l)} · AGG({h_u^{(l)} : u ∈ N(v) ∪ {v}}))
                Node features: generator embeddings; Edge weights: family proximity.
PROPERTY      : Graph structure injects prior knowledge about generator relationships,
                reducing sample complexity for family-level attribution.
WHY NOT BEFORE: Standard classifiers treat all classes as independent. GNA leverages
                the genealogical structure as an explicit inductive bias.
FALSIFIER     : If GNA does NOT improve family-group accuracy more than individual
                class accuracy, the graph structure is not helping.
================================================================================

exp26_graph.py — Graph neural attribution for few-shot AI-code attribution.
Protocol: fraction-based (1% / 5% / 20%), unixcoder-base only.
"""

# === KAGGLE PATHS ===
KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"
KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_DROID = "/kaggle/input/datasets/chiboiz/droid-collection/DroidCollection/data"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"

from __future__ import annotations
import os, sys, time, json, random, subprocess, importlib.util, warnings, glob
from collections import defaultdict
from dataclasses import dataclass

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("scikit-learn"); _ensure("tqdm")

import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import Dataset as TD, DataLoader
from transformers import AutoModel, AutoTokenizer
from tqdm import tqdm

try:
    from torch.amp import autocast as _ac, GradScaler
except ImportError:
    from torch.cuda.amp import autocast as _ac, GradScaler

def _autocast_ctx(dev: torch.device):
    enabled = (dev.type == "cuda")
    try:
        return _ac(device_type=dev.type, enabled=enabled)
    except TypeError:
        return _ac(enabled=enabled)

warnings.filterwarnings("ignore")
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", stream=sys.stdout)
logger = logging.getLogger("exp26")

PAPER_BASELINE = 0.6633

# Generator family graph structure
GENERATOR_GRAPH = {
    0: [],  # human - isolated
    1: [2, 3],  # gpt-3.5 related to gpt-4
    2: [1, 3],  # gpt-4 related to gpt-3.5
    3: [1, 2],  # gpt-4o related to gpt family
    4: [5, 6, 7],  # llama-3 related to llama-3.1, codellama, nxcode
    5: [4, 6, 7],  # llama-3.1 related to llama-3, codellama, nxcode
    6: [4, 5, 7],  # codellama related to llama family
    7: [4, 5, 6],  # nxcode related to llama family
    8: [9, 10],  # qwen2 related to qwen2.5
    9: [8, 10],  # qwen2.5 related to qwen family
    10: [8, 9],  # qwen1.5 related to qwen family
}

@dataclass
class Cfg:
    benchmark: str = "codet_m4"
    task: str = "author"
    enc: str = "unixcoder-base"
    frac: float = 0.05
    n_cls: int = 6
    seed: int = 42
    bs: int = 256
    seq: int = 512
    epochs: int = 3
    lr_enc: float = 2e-5
    lr_head: float = 1e-4
    wd: float = 0.01
    warmup: float = 0.1
    device: str = "cuda"
    # GNA specific
    graph_hidden: int = 128  # Graph convolution hidden dim
    num_layers: int = 2  # Number of GNN layers

    def __post_init__(self):
        if self.benchmark == "codet_m4":
            self.n_cls = 6 if self.task == "author" else 2
        elif self.benchmark == "aicd_t2":
            self.n_cls = 12; self.task = "t2"
        elif self.benchmark in ("droid_t3",):
            self.n_cls = 3; self.task = "t3"
        elif self.benchmark == "droid_t4":
            self.n_cls = 4; self.task = "t4"

def _hw(cfg: Cfg) -> Cfg:
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        if mem >= 40: cfg.bs, cfg.seq = 256, 512
        elif mem >= 10: cfg.bs, cfg.seq = 128, 384
        else: cfg.bs, cfg.seq = 64, 256
    return cfg

def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)

def _is_human(t):
    return str(t or "").strip().lower() in {"human","human_written","human-generated"}

def _vocab(train):
    names = {str(r.get("model","") or "").strip() for r in train
             if not _is_human(r.get("target","")) and r.get("model","")}
    return {n:i+1 for i,n in enumerate(sorted(names))}

def _conv_codet(split, task, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code","code"):
            v = r.get(f,"")
            if isinstance(v,str) and v.strip(): code = v; break
        if task == "binary": label = 0 if _is_human(r.get("target","")) else 1
        else:
            if _is_human(r.get("target","")): label = 0
            else: label = vocab.get(str(r.get("model","") or "").strip(), -1)
        return {"code":code,"label":label,"lang":str(r.get("language","")).strip().lower()}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"]>=0 and len(x["code"].strip())>0)

def _conv_droid(split, task):
    lm = {"HUMAN_GENERATED":0,"HUMAN":0,"MACHINE_GENERATED":1,"AI_GENERATED":1,
          "MACHINE_REFINED":2,"REFINED":2,"ADVERSARIAL":3,"ADVERSARIALLY_HUMANISED":3}
    def row(r):
        code = str(r.get("code","")).strip()
        raw = r.get("label",-1)
        label = lm.get(str(raw).strip().upper(), int(raw) if isinstance(raw,int) else -1)
        if task == "t3": label = 1 if label == 3 else label
        return {"code":code,"label":label,"lang":str(r.get("language","")).strip().lower()}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"]>=0 and len(x["code"].strip())>0)

def _conv_aicd(split):
    def row(r):
        code = str(r.get("code","") or r.get("text","") or "").strip()
        label = int(r.get("label", -1))
        return {"code":code,"label":label}
    out = split.map(row, remove_columns=split.column_names)
    return out.filter(lambda x: x["label"]>=0 and len(x["code"].strip())>0)

def _load_codet():
    """Load CoDET-M4 from Kaggle path with proper train/val/test split."""
    ds = load_dataset("parquet", data_files=KAGGLE_CODET, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split","")).lower()=="train")
        vl = ds.filter(lambda x: str(x.get("split","")).lower() in {"val","validation","dev"})
        ts = ds.filter(lambda x: str(x.get("split","")).lower()=="test")
        return tr, vl, ts
    else:
        s = ds.train_test_split(test_size=0.1, seed=42)
        s2 = s["train"].train_test_split(test_size=1/9, seed=42)
        return s2["train"], s2["test"], s["test"]

def _load_droid():
    """Load DroidCollection with proper shard handling."""
    train_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "test-*.parquet")))
    dev_files = sorted(glob.glob(os.path.join(KAGGLE_DROID, "dev-*.parquet")))

    if train_files:
        logger.info(f"[droid] Loading from local: {len(train_files)} train shards, {len(test_files)} test shards, {len(dev_files)} dev shards")
        ds_train = load_dataset("parquet", data_files=train_files, split="train")
        
        if dev_files:
            ds_dev = load_dataset("parquet", data_files=dev_files, split="train")
            ds_test = load_dataset("parquet", data_files=test_files, split="train") if test_files else None
            return ds_train, ds_dev, ds_test
        elif test_files:
            ds_test = load_dataset("parquet", data_files=test_files, split="train")
            s = ds_train.train_test_split(test_size=0.1, seed=42)
            return s["train"], s["test"], ds_test
        else:
            s = ds_train.train_test_split(test_size=0.2, seed=42)
            return s["train"], s["test"], s["test"]
    else:
        logger.warning("[droid] Kaggle path not found, falling back to HuggingFace...")
        tr = load_dataset("project-droid/DroidCollection", split="train")
        vl = load_dataset("project-droid/DroidCollection", split="dev")
        ts = load_dataset("project-droid/DroidCollection", split="test")
        return tr, vl, ts

def _load_aicd(task):
    """Load AICD-Bench from Kaggle local path or HuggingFace.

    Kaggle path structure:
      /kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench/
        ├── T1/ (binary: human vs AI, 2 classes)
        ├── T2/ (model family attribution, 12 classes)
        └── T3/ (fine-grained detection, 4 classes)

    Always tries the CORRECT task directory first (T2 for aicd_t2, etc.).
    Falls back to other directories only if the target dir is missing.
    """
    task_map = {"t1": "T1", "t2": "T2", "t3": "T3"}
    task_name = task_map.get(task.lower(), "T2")

    local_base = KAGGLE_AICD
    # Try correct task dir first, then remaining dirs as fallback
    dirs_to_try = [task_name] + [d for d in ["T1", "T2", "T3"] if d != task_name]

    for t in dirs_to_try:
        task_path = os.path.join(local_base, t)
        if os.path.isdir(task_path):
            parquet_files = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
            if parquet_files:
                logger.info(f"[aicd] Loading {t} from local: {task_path} ({len(parquet_files)} files)")
                ds = load_dataset("parquet", data_files=parquet_files, split="train")
                # Honour built-in split column if present
                if "split" in ds.column_names:
                    try:
                        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
                        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
                        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
                        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0:
                            return tr, vl, ts
                    except Exception:
                        pass
                # Manual 80/10/10 split
                s = ds.train_test_split(test_size=0.1, seed=42)
                s2 = s["train"].train_test_split(test_size=1/9, seed=42)
                return s2["train"], s2["test"], s["test"]

    # Flat fallback: scan entire AICD base dir
    parquet_files = sorted(glob.glob(os.path.join(local_base, "**", "*.parquet"), recursive=True))
    if parquet_files:
        logger.info(f"[aicd] Loading parquet files from base: {local_base}")
        ds = load_dataset("parquet", data_files=parquet_files, split="train")
        s = ds.train_test_split(test_size=0.1, seed=42)
        s2 = s["train"].train_test_split(test_size=1/9, seed=42)
        return s2["train"], s2["test"], s["test"]

    # Last resort: HuggingFace (requires internet)
    logger.warning(f"[aicd] Local path not found for {task_name}, trying HuggingFace (requires internet)")
    return (load_dataset("AICD-bench/AICD-Bench", name=task_name, split=s) for s in ["train", "validation", "test"])


def _preflight_check():
    logger.info("="*60)
    logger.info("[PREFLIGHT] Checking dataset availability...")
    all_ok = True
    for name, loader in [
        ("CoDET-M4", lambda: _load_codet()),
        ("DroidCollection", lambda: _load_droid()),
        ("AICD", lambda: _load_aicd("t2")),
    ]:
        try:
            d = loader()
            logger.info(f"  {name}: train={len(d[0])}, val={len(d[1])}, test={len(d[2])}")
        except Exception as e:
            logger.error(f"  {name}: FAILED - {e}")
            all_ok = False
    if all_ok:
        logger.info("[PREFLIGHT] All datasets loaded successfully!")
    else:
        logger.error("[PREFLIGHT] Dataset validation FAILED!")
        raise RuntimeError("Dataset validation failed")
    logger.info("="*60)

class FSDS(TD):
    def __init__(self, hf, tok, max_len):
        self.hf = hf; self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.hf)
    def __getitem__(self, i):
        r = self.hf[i]
        enc = self.tok(r["code"], max_length=self.max_len, truncation=True, padding="max_length", return_tensors="pt")
        return {"ids":enc["input_ids"].squeeze(0),"mask":enc["attention_mask"].squeeze(0),"label":int(r["label"])}

def collate(b):
    return {"ids":torch.stack([x["ids"] for x in b]),"mask":torch.stack([x["mask"] for x in b]),
            "labels":torch.tensor([x["label"] for x in b], dtype=torch.long)}

def build_dls(cfg: Cfg):
    import glob
    set_seed(cfg.seed)
    enc_path = os.path.join(KAGGLE_MODELS, cfg.enc)
    tok = AutoTokenizer.from_pretrained(enc_path, local_files_only=True)

    if cfg.benchmark == "codet_m4":
        tr_raw, vl_raw, ts_raw = _load_codet()
        vocab = _vocab(tr_raw) if cfg.task == "author" else {}
        tr_d = _conv_codet(tr_raw, cfg.task, vocab)
        vl_d = _conv_codet(vl_raw, cfg.task, vocab)
        ts_d = _conv_codet(ts_raw, cfg.task, vocab)
    elif cfg.benchmark.startswith("droid"):
        tr_raw, vl_raw, ts_raw = _load_droid()
        tr_d = _conv_droid(tr_raw, cfg.task)
        vl_d = _conv_droid(vl_raw, cfg.task)
        ts_d = _conv_droid(ts_raw, cfg.task)
    else:
        tr_raw, vl_raw, ts_raw = _load_aicd(cfg.task)
        tr_d = _conv_aicd(tr_raw); vl_d = _conv_aicd(vl_raw); ts_d = _conv_aicd(ts_raw)

    by_cls = defaultdict(list)
    for i, lab in enumerate(tr_d["label"]): by_cls[int(lab)].append(i)
    rng = random.Random(cfg.seed)
    chosen = []
    for cls in range(cfg.n_cls):
        pool = by_cls.get(cls, [])
        n = max(1, int(round(len(pool) * cfg.frac))) if pool else 0
        chosen.extend(rng.sample(pool, min(n, len(pool))) if pool else [])
    logger.info(f"[data] {cfg.enc} | {cfg.benchmark} | frac={cfg.frac} | n_train={len(chosen)}")
    rng.shuffle(chosen)
    tr_d = tr_d.select(chosen)

    def ld(ds, shuf):
        return DataLoader(FSDS(ds, tok, cfg.seq), batch_size=cfg.bs, shuffle=shuf, num_workers=4, collate_fn=collate, pin_memory=True)
    return ld(tr_d, True), ld(vl_d, False), ld(ts_d, False)


class GraphConvLayer(nn.Module):
    """Single graph convolution layer."""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x, adj):
        # x: [n_cls, in_dim], adj: [n_cls, n_cls]
        out = torch.mm(adj, x)  # Aggregate neighbors
        out = self.linear(out)
        return F.relu(out)


class GNANet(nn.Module):
    """Encoder with graph neural network for generator relationships."""
    def __init__(self, cfg: Cfg):
        super().__init__()
        self.cfg = cfg
        self.n_cls = cfg.n_cls

        # Build adjacency matrix from graph
        self.adj = self._build_adj_matrix()

        self.encoder = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, cfg.enc), local_files_only=True
        )
        hidden = self.encoder.config.hidden_size

        # Node embeddings for each class
        self.node_embeddings = nn.Parameter(torch.randn(cfg.n_cls, cfg.graph_hidden))

        # GNN layers
        self.gnn_layers = nn.ModuleList([
            GraphConvLayer(cfg.graph_hidden, cfg.graph_hidden)
            for _ in range(cfg.num_layers)
        ])

        # Combine encoder and graph
        self.head = nn.Sequential(
            nn.Linear(hidden + cfg.graph_hidden, hidden // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden // 2, cfg.n_cls),
        )

    def _build_adj_matrix(self):
        adj = torch.zeros(self.n_cls, self.n_cls)
        for node, neighbors in GENERATOR_GRAPH.items():
            if node < self.n_cls:
                for neighbor in neighbors:
                    if neighbor < self.n_cls:
                        adj[node, neighbor] = 1.0
                        adj[neighbor, node] = 1.0
        # Self-loop
        adj.fill_diagonal_(1)
        # Normalize
        deg = adj.sum(dim=1, keepdim=True)
        adj = adj / (deg + 1e-8)
        return adj

    def forward(self, ids, mask):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        pooled = out.last_hidden_state[:, 0]

        # Propagate through GNN
        node_features = self.node_embeddings
        adj = self.adj.to(node_features.device)
        for layer in self.gnn_layers:
            node_features = layer(node_features, adj)

        # Use class-0 node embedding as prototype (or mean)
        graph_feat = node_features[0].unsqueeze(0).expand(pooled.size(0), -1)

        # Combine encoder and graph features
        combined = torch.cat([pooled, graph_feat], dim=1)
        logits = self.head(combined)
        return {"logits": logits, "node_features": node_features}


def train(cfg: Cfg, tr_dl, vl_dl, ts_dl):
    dev = torch.device(cfg.device)
    model = GNANet(cfg).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr_enc, weight_decay=cfg.wd)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=cfg.lr_enc,
        steps_per_epoch=len(tr_dl), epochs=cfg.epochs, pct_start=cfg.warmup)
    scaler = GradScaler(enabled=(dev.type == "cuda"))

    best_val, best_state = 0, None
    train_history, val_history = [], []

    for ep in range(cfg.epochs):
        model.train()
        pbar = tqdm(tr_dl, desc=f"Epoch {ep+1}/{cfg.epochs}", leave=False)
        ep_loss = []

        for b in pbar:
            ids, mask, labs = b["ids"].to(dev), b["mask"].to(dev), b["labels"].to(dev)

            with _autocast_ctx(dev):
                logits = model(ids, mask)["logits"]
                loss = F.cross_entropy(logits, labs)

            ep_loss.append(loss.item())

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); opt.zero_grad()
            sched.step()
            pbar.set_postfix({"loss": f"{loss.item():.3f}"})

        tr_met = eval_m(model, tr_dl, dev)
        vr = eval_m(model, vl_dl, dev)
        train_history.append({
            "epoch": ep + 1, "loss": round(np.mean(ep_loss), 6),
            "macro_f1": round(tr_met["macro"], 6),
        })
        val_history.append({
            "epoch": ep + 1, "macro_f1": round(vr["macro"], 6),
            "weighted_f1": round(vr["weighted"], 6), "accuracy": round(vr["acc"], 6),
        })
        if vr["macro"] > best_val:
            best_val = vr["macro"]
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    final_test = eval_m(model, ts_dl, dev)
    final_test["train_history"] = train_history
    final_test["val_history"] = val_history
    return final_test


@torch.no_grad()
def eval_m(model, loader, dev):
    model.eval()
    ps, ls = [], []
    for b in loader:
        with _autocast_ctx(dev): logits = model(b["ids"].to(dev), b["mask"].to(dev))["logits"]
        ps.extend(logits.argmax(1).cpu().tolist())
        ls.extend(b["labels"].tolist())
    from sklearn.metrics import f1_score as f1s, precision_score, recall_score, confusion_matrix
    from collections import Counter
    return {
        "macro": f1s(ls, ps, average="macro", zero_division=0),
        "weighted": f1s(ls, ps, average="weighted", zero_division=0),
        "acc": accuracy_score(ls, ps),
        "per_class_f1": f1s(ls, ps, average=None, zero_division=0).tolist(),
        "per_class_precision": precision_score(ls, ps, average=None, zero_division=0).tolist(),
        "per_class_recall": recall_score(ls, ps, average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(ls, ps).tolist(),
        "pred_distribution": dict(Counter(ps)),
        "label_distribution": dict(Counter(ls)),
    }


def main():
    logger.info("[PREFLIGHT] Running dataset validation...")
    _preflight_check()

    encoders = ["unixcoder-base"]
    benchmarks = [("codet_m4","author"), ("aicd_t2","t2")]
    fracs = [0.01, 0.05, 0.20]

    results = []
    for enc in encoders:
        for bench, task in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac)
                cfg = _hw(cfg)
                if torch.cuda.is_available(): torch.cuda.reset_peak_memory_stats()
                tag = f"exp26_gna_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                tr_dl, vl_dl, ts_dl = build_dls(cfg)
                res = train(cfg, tr_dl, vl_dl, ts_dl)
                elapsed = time.time() - t0

                row = {
                    "tag": tag, "enc": enc, "bench": bench, "frac": frac,
                    "macro": round(res["macro"], 6), "weighted": round(res["weighted"], 6),
                    "acc": round(res["acc"], 6),
                    "dpaper": round(res["macro"] - PAPER_BASELINE, 6),
                    "graph_hidden": cfg.graph_hidden, "num_layers": cfg.num_layers,
                    "per_class_f1": [round(x, 6) for x in res["per_class_f1"]],
                    "confusion_matrix": res["confusion_matrix"],
                    "train_history": res["train_history"],
                    "val_history": res["val_history"],
                    "wall": round(elapsed, 1),
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                }
                results.append(row)
                logger.info(f"[{tag}] MacroF1={res['macro']:.4f} ({res['macro']-PAPER_BASELINE:+.4f} vs paper) time={elapsed:.0f}s")
                del tr_dl, vl_dl, ts_dl
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    os.makedirs("results", exist_ok=True)
    with open("results/exp26_gna_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "="*100)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Macro-F1':>10} {'dPaper':>10} {'Weighted':>10} {'Wall':>8}")
    print("-"*100)
    for r in results:
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['macro']:>10.4f} {r['dpaper']:>+10.4f} {r['weighted']:>10.4f} {r['wall']:>8.0f}s")
    print("="*100)
    print("\n[OK] GNA experiments complete!")

if __name__ == "__main__":
    main()
