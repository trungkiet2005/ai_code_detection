# exp104 — DUALGRAPH
# NAME       : DUALGRAPH (Dual-Graph Encoder for Code Authorship: Token-Cooc x AST-DFG)
# REFERENCE  : new (inspired by GraphCodeBERT arXiv:2009.08366, SynCoBERT arXiv:2108.04556,
#              GIN arXiv:1810.00826)
# CLAIM      : Code style lives on EDGES, not on tokens. Two graph topologies — token
#              co-occurrence within a sliding window AND AST/DFG parent-child — disagree
#              about which edges are stylistically meaningful. A dual-GNN that reads
#              from both topologies on top of UniXcoder hidden states learns a style
#              representation strictly richer than either graph alone.
# EQUATION   : H_tok = UniXcoder(x)                # (B, L, d)
#              G_cooc = window-graph on H_tok (k=3 token window)
#              G_ast  = AST parent-child graph (Python: via ast.walk; fallback: token-bigram)
#              h_cooc = GIN(H_tok, G_cooc)         # (B, d)
#              h_ast  = GIN(H_tok, G_ast)          # (B, d)
#              z      = LayerNorm(W_f [h_cooc ; h_ast ; mean(H_tok)])
#              logits = clf(z)
# WHY NEW    : No prior AI code attribution paper uses BOTH a token-graph AND a
#              syntactic graph readout simultaneously. SynCoBERT used data-flow only.
#              GraphCodeBERT injects DFG into the encoder, not as a separate readout.
# WOW HOOK   : "Style lives on edges, not on tokens — and there are two graph topologies
#              that disagree about which edges matter."
# FALSIFIER  : (F1) If composite(token_graph_only) ~ composite(ast_graph_only) ~
#              composite(dual) (delta < 0.005), the dual claim collapses. (F2) If learned
#              fusion weights -> (1, 0) or (0, 1), one graph is redundant.
from __future__ import annotations

KAGGLE_MODELS = "/kaggle/input/datasets/chiboiz/ai-detection-encoders/models"

import os, sys, time, json, random, subprocess, importlib.util, warnings, glob, math
import ast as _ast
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

def _ensure(pkg):
    if importlib.util.find_spec(pkg.split(".")[0]) is None:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

_ensure("numpy"); _ensure("torch"); _ensure("datasets")
_ensure("transformers"); _ensure("scikit-learn"); _ensure("tqdm")

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
logger = logging.getLogger("exp104_dualgraph")

PAPER_BASELINE = 0.6633
GENE_ADJ_CODET = {0: [], 1: [3], 2: [], 3: [1], 4: [], 5: []}
GENE_ADJ_AICD = {i: [(i // 3) * 3 + j for j in range(3) if (i // 3) * 3 + j != i] for i in range(12)}


def _gene_distance(u, v, adj):
    if u == v: return 0.0
    queue = [(u, 0)]; visited = {u}
    while queue:
        curr, d = queue.pop(0)
        for nb in adj.get(curr, []):
            if nb == v: return d + 1.0
            if nb not in visited:
                visited.add(nb); queue.append((nb, d + 1))
    return float("inf")


def build_distance_matrix(n_cls, adj, default_dist=4.0):
    D = torch.full((n_cls, n_cls), default_dist)
    for i in range(n_cls):
        for j in range(n_cls):
            d = _gene_distance(i, j, adj)
            if d < float("inf"): D[i, j] = d
            elif (i == 0) != (j == 0): D[i, j] = 3.0
    return D


def build_sibling_mask(n_cls, adj):
    M = torch.zeros(n_cls, n_cls)
    for i in range(n_cls):
        for j in adj.get(i, []): M[i, j] = 1.0
    return M


# =============================================================================
# AST edge extraction
# =============================================================================

def _ast_parent_child_lines(code: str) -> List[Tuple[int, int]]:
    """Return list of (parent_line, child_line) pairs from a Python AST.
    Falls back to empty list on parse failure."""
    try:
        tree = _ast.parse(code)
    except (SyntaxError, ValueError):
        return []
    edges = []
    for node in _ast.walk(tree):
        if not hasattr(node, "lineno"):
            continue
        for child in _ast.iter_child_nodes(node):
            if hasattr(child, "lineno"):
                edges.append((int(getattr(node, "lineno", 1)),
                              int(getattr(child, "lineno", 1))))
    return edges


def _line_offsets(code: str) -> List[int]:
    """Cumulative character offset at the start of each line (0-indexed line)."""
    offs = [0]
    for i, ch in enumerate(code):
        if ch == "\n":
            offs.append(i + 1)
    return offs


def build_ast_edge_index(code: str, tok, seq_len: int, pad_id: int) -> List[Tuple[int, int]]:
    """Map AST parent-child (line-level) pairs to token-position pairs.

    Strategy:
      1. Try AST parse; collect (parent_line, child_line) pairs.
      2. Re-tokenize and recover a line -> token-position mapping by tracking
         where each newline lands. For each line, the representative token
         position is the first non-pad token on/after that line.
      3. If AST parse fails or yields 0 edges, fall back to a token-bigram chain
         (token i -> token i+1) on the non-pad portion.

    Returns: list of (src_tok_idx, dst_tok_idx) edges (both directions are
    implicitly handled by the GIN aggregation -- we add reverse + self below).
    """
    edges_line = _ast_parent_child_lines(code)
    pad_token = tok.pad_token_id

    if edges_line:
        # Tokenise the code carefully to map lines to positions
        lines = code.split("\n")
        # For each line, compute its representative token position by tokenizing
        # the code up through that line and counting non-pad tokens.
        # Faster approximate: tokenize line-by-line and sum lengths.
        line_token_pos = []
        running = 3  # account for [CLS, <encoder_only>, SEP] prefix in _tokenize
        for li, ln in enumerate(lines):
            line_token_pos.append(min(running, seq_len - 1))
            ln_clean = " ".join(ln.split())
            if ln_clean:
                running += len(tok.tokenize(ln_clean))
            if running >= seq_len - 1:
                # Remaining lines all map to last valid position
                for _ in range(li + 1, len(lines)):
                    line_token_pos.append(seq_len - 1)
                break
        # Pad in case break didn't hit
        while len(line_token_pos) < len(lines):
            line_token_pos.append(min(seq_len - 1, running))

        out = []
        for (pl, cl) in edges_line:
            pl_i = max(0, min(pl - 1, len(line_token_pos) - 1))
            cl_i = max(0, min(cl - 1, len(line_token_pos) - 1))
            ps = line_token_pos[pl_i]
            cs = line_token_pos[cl_i]
            if ps != cs and ps < seq_len and cs < seq_len:
                out.append((ps, cs))
        if out:
            return out

    # Fallback: token-bigram chain
    return []  # signal -> caller will inject bigram chain at forward time


# =============================================================================
# Model
# =============================================================================

class GINLayer(nn.Module):
    """Hand-rolled GIN layer using scatter_add via index_add_.
    h_new[i] = MLP( (1+eps) * h[i] + sum_{j in N(i)} h[j] )
    """
    def __init__(self, dim):
        super().__init__()
        self.eps = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(),
            nn.Linear(dim, dim))
        self.ln = nn.LayerNorm(dim)

    def forward(self, h, edge_src, edge_dst):
        # h: (N_total, d)
        # edge_src, edge_dst: (E,) into h
        agg = torch.zeros_like(h)
        if edge_src.numel() > 0:
            # For each edge (s -> d), add h[s] to agg[d]
            agg.index_add_(0, edge_dst, h.index_select(0, edge_src))
        out = (1.0 + self.eps) * h + agg
        out = self.mlp(out)
        return self.ln(out + h)  # residual


class DUALGRAPHModel(nn.Module):
    def __init__(self, enc_name, n_cls, cooc_window=3, n_gnn_layers=2):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(
            os.path.join(KAGGLE_MODELS, enc_name), local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.cooc_window = cooc_window
        self.gnn_cooc = nn.ModuleList([GINLayer(hidden) for _ in range(n_gnn_layers)])
        self.gnn_ast = nn.ModuleList([GINLayer(hidden) for _ in range(n_gnn_layers)])
        # Fusion: learnable softmax over 3 readout streams
        self.fuse_logits = nn.Parameter(torch.zeros(3))
        self.fuse_proj = nn.Linear(3 * hidden, hidden)
        self.fuse_ln = nn.LayerNorm(hidden)
        self.clf = nn.Linear(hidden, n_cls)
        self.n_cls = n_cls
        self.hidden = hidden
        self._override_fusion = None  # for ablation

    def set_fusion_override(self, mode):
        """mode in {None, 'cooc_only', 'ast_only', 'mean_only'}."""
        self._override_fusion = mode

    @staticmethod
    def _build_cooc_edges(mask, window, device):
        """For each sample in the batch, build window edges (i, j) with |i-j|<=window
        within the non-pad region. Returns (edge_src_global, edge_dst_global) into
        the flattened (B*L) node tensor."""
        B, L = mask.shape
        srcs, dsts = [], []
        for b in range(B):
            n_valid = int(mask[b].sum().item())
            if n_valid <= 1:
                continue
            base = b * L
            for i in range(n_valid):
                lo = max(0, i - window)
                hi = min(n_valid, i + window + 1)
                for j in range(lo, hi):
                    if j != i:
                        srcs.append(base + j)
                        dsts.append(base + i)
        if not srcs:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device))
        return (torch.tensor(srcs, dtype=torch.long, device=device),
                torch.tensor(dsts, dtype=torch.long, device=device))

    @staticmethod
    def _build_bigram_edges(mask, device):
        """Fallback AST: token i <-> token i+1 within non-pad region."""
        B, L = mask.shape
        srcs, dsts = [], []
        for b in range(B):
            n_valid = int(mask[b].sum().item())
            if n_valid <= 1:
                continue
            base = b * L
            for i in range(n_valid - 1):
                srcs.append(base + i); dsts.append(base + i + 1)
                srcs.append(base + i + 1); dsts.append(base + i)
        if not srcs:
            return (torch.empty(0, dtype=torch.long, device=device),
                    torch.empty(0, dtype=torch.long, device=device))
        return (torch.tensor(srcs, dtype=torch.long, device=device),
                torch.tensor(dsts, dtype=torch.long, device=device))

    @staticmethod
    def _stitch_ast_edges(ast_edges_list, mask, device):
        """ast_edges_list: list of B items, each a list of (i, j) edge tuples.
        Offset to global indexing into the flattened (B*L) tensor."""
        B, L = mask.shape
        srcs, dsts = [], []
        any_edges = False
        for b in range(B):
            base = b * L
            elist = ast_edges_list[b]
            if elist:
                any_edges = True
                for (i, j) in elist:
                    if 0 <= i < L and 0 <= j < L:
                        srcs.append(base + i); dsts.append(base + j)
                        # reverse direction so GIN aggregates from both sides
                        srcs.append(base + j); dsts.append(base + i)
        if not any_edges:
            return None  # signal caller to use bigram fallback
        return (torch.tensor(srcs, dtype=torch.long, device=device),
                torch.tensor(dsts, dtype=torch.long, device=device))

    def forward(self, ids, mask, ast_edges_list):
        out = self.encoder(input_ids=ids, attention_mask=mask)
        H = out.last_hidden_state  # (B, L, d)
        B, L, d = H.shape
        device = H.device
        H_flat = H.reshape(B * L, d)

        # ----- co-occurrence graph -----
        c_src, c_dst = self._build_cooc_edges(mask, self.cooc_window, device)
        h_c = H_flat
        for layer in self.gnn_cooc:
            h_c = layer(h_c, c_src, c_dst)
        h_c = h_c.view(B, L, d)

        # ----- AST graph -----
        stitched = self._stitch_ast_edges(ast_edges_list, mask, device)
        if stitched is None:
            a_src, a_dst = self._build_bigram_edges(mask, device)
        else:
            a_src, a_dst = stitched
        h_a = H_flat
        for layer in self.gnn_ast:
            h_a = layer(h_a, a_src, a_dst)
        h_a = h_a.view(B, L, d)

        # ----- masked mean readout -----
        m = mask.unsqueeze(-1).float()
        denom = m.sum(1).clamp(min=1)
        r_cooc = (h_c * m).sum(1) / denom
        r_ast = (h_a * m).sum(1) / denom
        r_mean = (H * m).sum(1) / denom

        # ----- fusion -----
        if self._override_fusion == "cooc_only":
            w = torch.tensor([1.0, 0.0, 0.0], device=device)
        elif self._override_fusion == "ast_only":
            w = torch.tensor([0.0, 1.0, 0.0], device=device)
        elif self._override_fusion == "mean_only":
            w = torch.tensor([0.0, 0.0, 1.0], device=device)
        else:
            w = F.softmax(self.fuse_logits, dim=-1)
        # weight each stream then concat
        cat = torch.cat([w[0] * r_cooc, w[1] * r_ast, w[2] * r_mean], dim=-1)
        z = self.fuse_ln(self.fuse_proj(cat))
        logits = self.clf(z)
        return logits, z, w.detach()


# =============================================================================
# Plumbing
# =============================================================================

KAGGLE_CODET = "/kaggle/input/datasets/chiboiz/codetm4/dataset_without_comments.parquet"
KAGGLE_AICD = "/kaggle/input/datasets/chiboiz/ai-code-detection/AICD-Bench"


@dataclass
class Cfg:
    benchmark: str = "codet_m4"; task: str = "author"; enc: str = "unixcoder-base"
    frac: float = 0.20; n_cls: int = 6; seed: int = 42
    bs: int = 256; seq: int = 512; epochs: int = 3
    lr_enc: float = 2e-5; lr_head: float = 1e-4
    warmup: float = 0.1; wd: float = 0.01
    cooc_window: int = 3; n_gnn_layers: int = 2
    device: str = "cuda"; gene_adj: dict = field(default_factory=dict)


def adaptive_schedule(cfg):
    f = cfg.frac
    if f <= 0.02: cfg.epochs, cfg.lr_enc, cfg.warmup = 10, 3e-5, 0.20
    elif f <= 0.10: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 3e-5, 0.15
    else: cfg.epochs, cfg.lr_enc, cfg.warmup = 6, 4e-5, 0.10
    return cfg


def _hw(cfg):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        mem = torch.cuda.get_device_properties(0).total_memory / 1e9
        # Slightly smaller bs because the GNN doubles memory on top of encoder
        if mem >= 40: cfg.bs, cfg.seq = 96, 384
        elif mem >= 10: cfg.bs, cfg.seq = 48, 320
        else: cfg.bs, cfg.seq = 24, 256
        logger.info(f"[hw] mem={mem:.1f}GB bs={cfg.bs} seq={cfg.seq}")
    return cfg


def set_seed(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(s)


def _is_human(t):
    return str(t or "").strip().lower() in {"human", "human_written", "human-generated"}


def _vocab(train):
    names = {str(r.get("model", "") or "").strip() for r in train
             if not _is_human(r.get("target", "")) and r.get("model", "")}
    return {n: i + 1 for i, n in enumerate(sorted(names))}


def _conv_codet(split, task, vocab):
    def row(r):
        code = ""
        for f in ("cleaned_code", "code"):
            v = r.get(f, "")
            if isinstance(v, str) and v.strip(): code = v; break
        if task == "binary":
            label = 0 if _is_human(r.get("target", "")) else 1
        else:
            label = 0 if _is_human(r.get("target", "")) else vocab.get(str(r.get("model", "") or "").strip(), -1)
        return {"code": code, "label": label,
                "language": str(r.get("language", "")).strip().lower(),
                "source": str(r.get("source", "")).strip().lower()}
    return split.map(row, remove_columns=split.column_names).filter(
        lambda x: x["label"] >= 0 and len(x["code"].strip()) > 0)


def _conv_aicd(split):
    def row(r):
        return {"code": str(r.get("code", "")).strip(), "label": int(r.get("label", -1)),
                "language": str(r.get("language", "")).strip().lower(), "source": ""}
    return split.map(row, remove_columns=split.column_names).filter(
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
    task_name = {"t1": "T1", "t2": "T2", "t3": "T3"}.get(task.lower())
    if task_name is None: raise ValueError(f"[aicd] Unknown task '{task}'")
    task_path = os.path.join(KAGGLE_AICD, task_name)
    if not os.path.isdir(task_path): raise FileNotFoundError(f"[aicd] STRICT: {task_name} not found")
    pf = sorted(glob.glob(os.path.join(task_path, "**", "*.parquet"), recursive=True))
    if not pf: raise FileNotFoundError(f"[aicd] STRICT: No parquet files")
    ds = load_dataset("parquet", data_files=pf, split="train")
    if "split" in ds.column_names:
        tr = ds.filter(lambda x: str(x.get("split", "")).lower() == "train")
        vl = ds.filter(lambda x: str(x.get("split", "")).lower() in {"val", "validation", "dev"})
        ts = ds.filter(lambda x: str(x.get("split", "")).lower() == "test")
        if len(tr) > 0 and len(vl) > 0 and len(ts) > 0: return tr, vl, ts
    s = ds.train_test_split(test_size=0.1, seed=42)
    s2 = s["train"].train_test_split(test_size=1/9, seed=42)
    return s2["train"], s2["test"], s["test"]


def _tokenize(code, tokenizer, max_len):
    """UniXcoder tokenisation: CLS <encoder_only> SEP ... SEP."""
    toks = tokenizer.tokenize(" ".join(code.split()))[:max_len - 4]
    toks = [tokenizer.cls_token, "<encoder_only>", tokenizer.sep_token] + toks + [tokenizer.sep_token]
    ids = tokenizer.convert_tokens_to_ids(toks)
    ids += [tokenizer.pad_token_id] * (max_len - len(ids))
    return ids[:max_len]


class FSDS_DUAL(TD):
    """Returns input_ids, attention_mask, label, language, source, ast_edges, ast_n_nodes."""
    def __init__(self, data, tok, seq_len, frac=1.0, seed=42):
        self.data = data; self.tok = tok; self.seq_len = seq_len
        self.pad_id = tok.pad_token_id
        if frac < 1.0:
            rng = random.Random(seed)
            labels = list(range(max(self.data["label"]) + 1))
            keep = []
            for lbl in labels:
                idx = [i for i, x in enumerate(self.data["label"]) if x == lbl]
                keep.extend(rng.sample(idx, min(max(1, int(len(idx)*frac)), len(idx))))
            self.data = self.data.select(keep)
            logger.info(f"[FSDS_DUAL] Sampled {len(self.data)} samples ({frac*100:.0f}%)")

    def __len__(self): return len(self.data)

    def __getitem__(self, i):
        r = self.data[i]
        code = r["code"][:5000]
        lang = r.get("language", "") or ""
        ids = _tokenize(code, self.tok, self.seq_len)
        ids_t = torch.tensor(ids, dtype=torch.long)
        mask_t = (ids_t != self.pad_id).long()
        # Build AST edges only for python-looking code (else empty -> bigram fallback)
        lang_l = lang.lower()
        use_ast = lang_l in {"python", "py"} or (lang_l == "" and "def " in code[:200])
        if use_ast:
            ast_edges = build_ast_edge_index(code, self.tok, self.seq_len, self.pad_id)
        else:
            ast_edges = []
        return {"input_ids": ids_t, "attention_mask": mask_t,
                "label": r["label"], "ast_edges": ast_edges,
                "ast_n_nodes": int(mask_t.sum().item()),
                "language": lang, "source": r.get("source", "") or ""}


def collate_dual(batch):
    out = {
        "input_ids": torch.stack([b["input_ids"] for b in batch], dim=0),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch], dim=0),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
        "ast_edges": [b["ast_edges"] for b in batch],
        "language": [b["language"] for b in batch],
        "source": [b["source"] for b in batch],
    }
    return out


@torch.no_grad()
def eval_pack(model, loader, cfg, sib_mask_np, dist_mat_cpu,
              fusion_override=None, collect_fusion=False):
    model.eval()
    if fusion_override is not None:
        model.set_fusion_override(fusion_override)
    else:
        model.set_fusion_override(None)

    preds, labels, langs, sources = [], [], [], []
    fusion_w_sum = torch.zeros(3); n_batches = 0
    for b in tqdm(loader, desc=f"Eval[{fusion_override or 'dual'}]"):
        ids = b["input_ids"].to(cfg.device); mask = b["attention_mask"].to(cfg.device)
        labs = b["label"]
        logits, _z, w = model(ids, mask, b["ast_edges"])
        preds.extend(logits.argmax(dim=-1).cpu().tolist())
        labels.extend(labs.tolist())
        if collect_fusion:
            fusion_w_sum += w.detach().cpu()
            n_batches += 1
        langs.extend(b.get("language", [""] * len(labs)))
        sources.extend(b.get("source", [""] * len(labs)))

    preds = np.array(preds); labels = np.array(labels); n_cls = cfg.n_cls
    overall = {"accuracy": float(accuracy_score(labels, preds)),
               "macro_f1": float(f1_score(labels, preds, average="macro", zero_division=0)),
               "weighted_f1": float(f1_score(labels, preds, average="weighted", zero_division=0)),
               "micro_f1": float(f1_score(labels, preds, average="micro", zero_division=0)),
               "macro_precision": float(precision_score(labels, preds, average="macro", zero_division=0)),
               "macro_recall": float(recall_score(labels, preds, average="macro", zero_division=0))}
    per_class = {"f1": f1_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
                 "precision": precision_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist(),
                 "recall": recall_score(labels, preds, average=None, zero_division=0, labels=list(range(n_cls))).tolist()}
    cm = confusion_matrix(labels, preds, labels=list(range(n_cls)))
    off_diag = int(cm.sum() - cm.trace())
    sib_conf = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls) if i != j and sib_mask_np[i, j] > 0))
    sib_rate = sib_conf / max(off_diag, 1)
    cross = int(sum(cm[i, j] for i in range(n_cls) for j in range(n_cls) if i != j and dist_mat_cpu[i, j] >= 3.0))
    cross_rate = cross / max(off_diag, 1)
    per_lang, per_src = {}, {}
    if any(l for l in langs):
        la = np.array(langs)
        for L in sorted(set(langs)):
            if not L: continue
            sel = (la == L)
            if sel.sum() < 2: continue
            per_lang[L] = {"n": int(sel.sum()),
                           "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0)),
                           "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                           "accuracy": float(accuracy_score(labels[sel], preds[sel]))}
    if any(s for s in sources):
        sa = np.array(sources)
        for S in sorted(set(sources)):
            if not S: continue
            sel = (sa == S)
            if sel.sum() < 2: continue
            per_src[S] = {"n": int(sel.sum()),
                          "macro_f1": float(f1_score(labels[sel], preds[sel], average="macro", zero_division=0)),
                          "weighted_f1": float(f1_score(labels[sel], preds[sel], average="weighted", zero_division=0)),
                          "accuracy": float(accuracy_score(labels[sel], preds[sel]))}
    out = {"overall": overall, "per_class": per_class, "per_language": per_lang, "per_source": per_src,
           "confusion_matrix": cm.tolist(), "sibling_confusion_rate": float(sib_rate),
           "cross_family_confusion_rate": float(cross_rate),
           "off_diag_total": off_diag, "n_samples": int(len(labels))}
    if collect_fusion and n_batches > 0:
        fw = (fusion_w_sum / n_batches).tolist()
        out["fusion_weights_mean"] = fw
    return out


def train_epoch(model, loader, opt, sch, scaler, cfg):
    model.train(); tot = 0.0
    for b in tqdm(loader, desc="Train"):
        ids = b["input_ids"].to(cfg.device); mask = b["attention_mask"].to(cfg.device)
        labs = b["label"].to(cfg.device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=(cfg.device == "cuda")):
            logits, _z, _w = model(ids, mask, b["ast_edges"])
            loss = F.cross_entropy(logits, labs)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); opt.zero_grad(); sch.step()
        tot += loss.item()
    return tot / max(1, len(loader))


def run_exp(cfg, tag):
    set_seed(cfg.seed)
    cfg = _hw(cfg); cfg = adaptive_schedule(cfg)
    cfg.gene_adj = GENE_ADJ_CODET if cfg.benchmark == "codet_m4" else GENE_ADJ_AICD
    dist_mat_t = build_distance_matrix(cfg.n_cls, cfg.gene_adj)
    dist_mat_cpu = dist_mat_t.numpy()
    sib_mask_np = build_sibling_mask(cfg.n_cls, cfg.gene_adj).numpy()

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
    tr_ds = FSDS_DUAL(tr_data, tok, cfg.seq, frac=cfg.frac, seed=cfg.seed)
    vl_ds = FSDS_DUAL(vl_data, tok, cfg.seq, frac=1.0, seed=cfg.seed + 1)
    ts_ds = FSDS_DUAL(ts_data, tok, cfg.seq, frac=1.0, seed=cfg.seed + 2)

    loader_cfg = dict(batch_size=cfg.bs, num_workers=4, pin_memory=True, collate_fn=collate_dual)
    tr_dl = DataLoader(tr_ds, shuffle=True, **loader_cfg)
    vl_dl = DataLoader(vl_ds, shuffle=False, **loader_cfg)
    ts_dl = DataLoader(ts_ds, shuffle=False, **loader_cfg)

    total_steps = max(1, len(tr_ds) // cfg.bs) * cfg.epochs
    logger.info(f"[sched] frac={cfg.frac} epochs={cfg.epochs} lr_enc={cfg.lr_enc} "
                f"warmup={cfg.warmup} cooc_window={cfg.cooc_window} n_gnn={cfg.n_gnn_layers}")
    model = DUALGRAPHModel(cfg.enc, cfg.n_cls, cooc_window=cfg.cooc_window,
                           n_gnn_layers=cfg.n_gnn_layers).to(cfg.device)
    enc_ids = {id(p) for p in model.encoder.parameters()}
    head_params = [p for p in model.parameters() if id(p) not in enc_ids]
    opt = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()), "lr": cfg.lr_enc},
        {"params": head_params, "lr": cfg.lr_head}], weight_decay=cfg.wd)
    sch = get_cosine_schedule_with_warmup(opt, max(1, int(total_steps * cfg.warmup)), total_steps)
    scaler = GradScaler()

    best_val, best_state, val_hist = 0.0, None, []
    for epoch in range(cfg.epochs):
        loss = train_epoch(model, tr_dl, opt, sch, scaler, cfg)
        val_met = eval_pack(model, vl_dl, cfg, sib_mask_np, dist_mat_cpu)
        v = val_met["overall"]["macro_f1"]; val_hist.append(v)
        logger.info(f"[epoch {epoch+1}] loss={loss:.4f} val={v:.4f}")
        if v > best_val:
            best_val = v
            best_state = {k: v_.cpu().clone() for k, v_ in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)

    # ----- Test: full dual + ablations -----
    ts_met = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu,
                       fusion_override=None, collect_fusion=True)
    test_macro = ts_met["overall"]["macro_f1"]
    gap = best_val - test_macro
    fusion_w = ts_met.get("fusion_weights_mean", [0.0, 0.0, 0.0])

    # Ablations: force single-stream fusion
    abl_cooc = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu,
                         fusion_override="cooc_only")
    abl_ast = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu,
                        fusion_override="ast_only")
    abl_mean = eval_pack(model, ts_dl, cfg, sib_mask_np, dist_mat_cpu,
                         fusion_override="mean_only")
    # restore default
    model.set_fusion_override(None)

    logger.info(f"[final] val={best_val:.4f} test={test_macro:.4f} gap={gap:+.4f} "
                f"fusion_w=[{fusion_w[0]:.3f},{fusion_w[1]:.3f},{fusion_w[2]:.3f}]")
    logger.info(f"[ablation] cooc_only={abl_cooc['overall']['macro_f1']:.4f} "
                f"ast_only={abl_ast['overall']['macro_f1']:.4f} "
                f"mean_only={abl_mean['overall']['macro_f1']:.4f}")

    return {"tag": tag, "method": "DUALGRAPH",
            "upstream": "new (token-cooc x AST-DFG dual-GIN on UniXcoder)",
            "note": "GIN over two graph topologies; learnable softmax fusion over 3 streams",
            "enc": cfg.enc, "bench": cfg.benchmark,
            "frac": cfg.frac, "epochs": cfg.epochs, "lr_enc": cfg.lr_enc,
            "cooc_window": cfg.cooc_window, "n_gnn_layers": cfg.n_gnn_layers,
            "val_macro": best_val, "macro": test_macro,
            "weighted": ts_met["overall"]["weighted_f1"], "acc": ts_met["overall"]["accuracy"],
            "val_test_gap": gap, "dpaper": test_macro - PAPER_BASELINE,
            "fusion_weights_mean": [float(x) for x in fusion_w],
            "ablation_token_only_macro": float(abl_cooc["overall"]["macro_f1"]),
            "ablation_ast_only_macro": float(abl_ast["overall"]["macro_f1"]),
            "ablation_mean_only_macro": float(abl_mean["overall"]["macro_f1"]),
            "test_metrics": ts_met, "val_history": val_hist,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}


def main():
    encoders = ["unixcoder-base"]
    benchmarks = [("codet_m4", "author", 6), ("aicd_t2", "t2", 12)]
    fracs = [0.01, 0.05, 0.20]
    results = []
    for enc in encoders:
        for bench, task, n_cls in benchmarks:
            for frac in fracs:
                cfg = Cfg(benchmark=bench, task=task, enc=enc, frac=frac, n_cls=n_cls)
                tag = f"exp104_dualgraph_{enc}_{bench}_f{frac}"
                logger.info(f"=== {tag} ===")
                t0 = time.time()
                try:
                    res = run_exp(cfg, tag)
                    res["wall"] = round(time.time() - t0, 1)
                    results.append(res)
                    logger.info(f"[{tag}] test={res['macro']:.4f} ({res['dpaper']:+.4f}) "
                                f"gap={res['val_test_gap']:+.4f} "
                                f"abl=(cooc={res['ablation_token_only_macro']:.3f}, "
                                f"ast={res['ablation_ast_only_macro']:.3f}, "
                                f"mean={res['ablation_mean_only_macro']:.3f}) "
                                f"time={res['wall']:.0f}s")
                except Exception as e:
                    logger.error(f"[{tag}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                import gc; gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()

    try: _here = os.path.dirname(os.path.realpath(__file__))
    except NameError: _here = os.getcwd()
    out_dir = os.path.join(_here, "results"); os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "exp104_dualgraph_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 160)
    print(f"{'Encoder':<22} {'Benchmark':<12} {'Frac':>6} {'Ep':>4} {'Val-F1':>8} {'Test-F1':>8} "
          f"{'Gap':>8} {'dPaper':>9} {'AblCooc':>9} {'AblAst':>9} {'AblMean':>9} "
          f"{'FuseW':>22} {'Wall':>8}")
    print("-" * 160)
    for r in results:
        fw = r['fusion_weights_mean']
        fw_str = f"[{fw[0]:.2f},{fw[1]:.2f},{fw[2]:.2f}]"
        print(f"{r['enc']:<22} {r['bench']:<12} {r['frac']:>6.0%} {r['epochs']:>4d} "
              f"{r['val_macro']:>8.4f} {r['macro']:>8.4f} {r['val_test_gap']:>+8.4f} "
              f"{r['dpaper']:>+9.4f} {r['ablation_token_only_macro']:>9.4f} "
              f"{r['ablation_ast_only_macro']:>9.4f} {r['ablation_mean_only_macro']:>9.4f} "
              f"{fw_str:>22} {r['wall']:>8.0f}s")
    print("=" * 160)


if __name__ == "__main__":
    main()
