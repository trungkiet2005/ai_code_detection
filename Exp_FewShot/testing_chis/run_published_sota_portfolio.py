"""
run_published_sota_portfolio.py

Published-method reproduction/adaptation runner for CoDET-M4 Author IID.

This file intentionally reuses the Kaggle/offline loader, hardware profile, and
JSON output style from `run_hier_ntk_portfolio.py`.

Implemented paper families:
  - codet5_authorship: CodeT5-Authorship style encoder-only CodeT5 + 2-layer head.
  - detective_supcon: DeTeCtive-style CE + supervised contrastive learning.
  - faid_multitask: FAID/DeTeCtive-style class + family auxiliary + multi-level SupCon.
  - style_repr_logreg: few-shot style-representation baseline with code stylometry features.

These are faithful CoDET-M4 adaptations, not original-benchmark reruns.
"""
from __future__ import annotations

import json
import math
import os
import random
import time
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel

from run_hier_ntk_portfolio import (
    FSConfig,
    HIER_FAMILY,
    apply_hardware_profile,
    autocast,
    build_loaders,
    build_minival_indices,
    cleanup,
    emit_json,
    fraction_stratified_indices,
    kshot_stratified_indices,
    logger,
    make_grad_scaler,
    set_seed,
    _build_author_vocab,
    _class_weights,
    _convert_codet,
    _evaluate,
    _load_codet_splits,
    _per_class,
    _per_subgroup,
)


PAPER_BASELINE = 0.6633

PAPER_REGISTRY = {
    "codet5_authorship": {
        "paper": "I Know Which LLM Wrote Your Code Last Summer (AISec 2025 / arXiv 2506.17323)",
        "source": "CodeT5-Authorship: encoder-only CodeT5, first-token embedding, 2-layer GELU/dropout head.",
    },
    "detective_supcon": {
        "paper": "DeTeCtive: Detecting AI-generated Text via Multi-Level Contrastive Learning (arXiv 2410.20964)",
        "source": "Adapted to code attribution as CE + supervised contrastive style separation.",
    },
    "faid_multitask": {
        "paper": "FAID: Fine-grained AI-generated Text Detection (EACL 2026)",
        "source": "Adapted as class CE + family auxiliary CE + class/family supervised contrastive loss.",
    },
    "style_repr_logreg": {
        "paper": "Few-Shot Detection of Machine-Generated Text using Style Representations (arXiv 2401.06712)",
        "source": "Adapted from human-author style representations to code stylometry features + logistic regression.",
    },
}


def _family_labels(labels: torch.Tensor) -> torch.Tensor:
    vals = [HIER_FAMILY.get(int(y), int(y)) for y in labels.detach().cpu().tolist()]
    return torch.tensor(vals, dtype=torch.long, device=labels.device)


def _supcon(z: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised contrastive loss with anchors lacking positives skipped."""
    if z.size(0) <= 1:
        return torch.zeros((), device=z.device)
    z = F.normalize(z, dim=-1)
    logits = (z @ z.t()) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    eye = torch.eye(z.size(0), dtype=torch.bool, device=z.device)
    logits_mask = (~eye).float()
    pos = (labels.unsqueeze(0) == labels.unsqueeze(1)) & ~eye
    pos_f = pos.float()
    denom = (torch.exp(logits) * logits_mask).sum(dim=1, keepdim=True).clamp(min=1e-12)
    log_prob = logits - torch.log(denom)
    pos_count = pos_f.sum(dim=1)
    valid = pos_count > 0
    if not valid.any():
        return torch.zeros((), device=z.device)
    mean_log_prob_pos = (pos_f * log_prob).sum(dim=1)[valid] / pos_count[valid].clamp(min=1.0)
    return -mean_log_prob_pos.mean()


class CodeT5AuthorshipClassifier(nn.Module):
    """CodeT5-Authorship style classifier: first token -> GELU/dropout MLP."""

    def __init__(self, cfg: FSConfig):
        super().__init__()
        self.cfg = cfg
        encoder_path = cfg.get_encoder_path()
        logger.info(f"[CodeT5-Authorship] loading model from: {encoder_path}")
        base = AutoModel.from_pretrained(encoder_path, local_files_only=True)
        self.encoder = base.encoder if hasattr(base, "encoder") else base
        hidden = getattr(base.config, "hidden_size", None) or getattr(base.config, "d_model", None)
        if hidden is None:
            raise ValueError("Cannot infer hidden size for CodeT5AuthorshipClassifier")
        self.head = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, cfg.n_classes),
        )
        self.ntk_proj = nn.Sequential(
            nn.Linear(hidden, cfg.ntk_proj_dim),
            nn.GELU(),
            nn.Linear(cfg.ntk_proj_dim, cfg.ntk_proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        emb = out.last_hidden_state[:, 0]
        return {
            "logits": self.head(emb),
            "embedding": emb,
            "ntk_proj": F.normalize(self.ntk_proj(emb), dim=-1),
        }

    def param_groups(self):
        enc = list(self.encoder.parameters())
        head = list(self.head.parameters()) + list(self.ntk_proj.parameters())
        return [
            {"params": enc, "lr": self.cfg.lr_encoder, "weight_decay": self.cfg.weight_decay},
            {"params": head, "lr": self.cfg.lr_heads, "weight_decay": self.cfg.weight_decay},
        ]


class MultiTaskStyleClassifier(nn.Module):
    """Shared encoder with class head, family head, and projection head."""

    def __init__(self, cfg: FSConfig):
        super().__init__()
        self.cfg = cfg
        encoder_path = cfg.get_encoder_path()
        logger.info(f"[FAID/DeTeCtive] loading model from: {encoder_path}")
        self.encoder = AutoModel.from_pretrained(encoder_path, local_files_only=True)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden, cfg.n_classes)
        self.family_classifier = nn.Linear(hidden, len(set(HIER_FAMILY.values())))
        self.ntk_proj = nn.Sequential(
            nn.Linear(hidden, cfg.ntk_proj_dim),
            nn.GELU(),
            nn.Linear(cfg.ntk_proj_dim, cfg.ntk_proj_dim),
        )

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        mask = attention_mask.unsqueeze(-1).float()
        emb = (out.last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        dropped = self.dropout(emb)
        return {
            "logits": self.classifier(dropped),
            "family_logits": self.family_classifier(dropped),
            "embedding": emb,
            "ntk_proj": F.normalize(self.ntk_proj(emb), dim=-1),
        }

    def param_groups(self):
        enc = list(self.encoder.parameters())
        heads = (
            list(self.classifier.parameters())
            + list(self.family_classifier.parameters())
            + list(self.ntk_proj.parameters())
        )
        return [
            {"params": enc, "lr": self.cfg.lr_encoder, "weight_decay": self.cfg.weight_decay},
            {"params": heads, "lr": self.cfg.lr_heads, "weight_decay": self.cfg.weight_decay},
        ]


def loss_ce(outputs, labels, cfg, class_weights=None):
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    return {"total": ce, "ce": ce}


def loss_detective_supcon(outputs, labels, cfg, class_weights=None):
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    sup = _supcon(outputs["ntk_proj"], labels, temperature=float(os.environ.get("FS_SUPCON_TEMP", "0.07")))
    total = ce + cfg.lambda_ntk * sup
    return {"total": total, "ce": ce, "class_supcon": sup}


def loss_faid_multitask(outputs, labels, cfg, class_weights=None):
    fam = _family_labels(labels)
    ce = F.cross_entropy(outputs["logits"], labels, weight=class_weights)
    fam_ce = F.cross_entropy(outputs["family_logits"], fam)
    class_sup = _supcon(outputs["ntk_proj"], labels, temperature=0.07)
    fam_sup = _supcon(outputs["ntk_proj"], fam, temperature=0.07)
    total = ce + 0.5 * fam_ce + cfg.lambda_ntk * class_sup + cfg.lambda_hier * fam_sup
    return {
        "total": total,
        "ce": ce,
        "family_ce": fam_ce,
        "class_supcon": class_sup,
        "family_supcon": fam_sup,
    }


@dataclass
class TrainResult:
    test_macro_f1: float
    test_weighted_f1: float
    test_accuracy: float
    val_macro_f1: float
    per_class_f1: Dict
    per_lang_f1: Dict
    per_source_f1: Dict
    train_steps: int
    wall_time_s: float


def train_with_loss(
    cfg: FSConfig,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    loss_fn: Callable,
    method_name: str,
) -> TrainResult:
    device = torch.device(cfg.device)
    model = model.to(device)
    class_weights = _class_weights(train_loader, cfg.n_classes).to(device) if cfg.use_class_weights else None
    if class_weights is not None:
        logger.info(f"[trainer] class_weights={class_weights.tolist()}")

    optimizer = torch.optim.AdamW(model.param_groups())
    scaler = make_grad_scaler(enabled=(cfg.precision == "fp16" and device.type == "cuda"))
    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16

    total_steps = max(1, len(train_loader) * cfg.epochs)
    eval_every = max(5, total_steps // 8)
    scheduler = None
    if cfg.warmup_ratio > 0 and total_steps > 20:
        try:
            from transformers import get_cosine_schedule_with_warmup

            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=max(1, int(total_steps * cfg.warmup_ratio)),
                num_training_steps=total_steps,
            )
        except ImportError:
            scheduler = None

    logger.info(f"[trainer] {method_name} steps={total_steps} eval_every={eval_every} epochs={cfg.epochs}")
    best_val = -1.0
    best_state = None
    plateau = 0
    step = 0
    t0 = time.time()

    for _ in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            step += 1
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                enabled=device.type == "cuda",
                dtype=amp_dtype,
            ):
                out = model(ids, mask)
                losses = loss_fn(out, labels, cfg, class_weights=class_weights)
                loss = losses["total"]

            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                optimizer.step()
            if scheduler is not None:
                scheduler.step()
            if step % 10 == 0:
                logger.info(
                    f"[step {step}/{total_steps}] "
                    + " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                )
            if step % eval_every == 0 or step == total_steps:
                val = _evaluate(model, val_loader, cfg, device)
                logger.info(f"[val @ step {step}] macro_f1={val['macro_f1']:.4f} acc={val['accuracy']:.4f}")
                if val["macro_f1"] > best_val + 1e-4:
                    best_val = val["macro_f1"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    plateau = 0
                else:
                    plateau += 1
                    if plateau >= cfg.early_stop_patience:
                        logger.info(f"[early-stop] plateau={plateau}; stop")
                        break
                model.train()
        else:
            continue
        break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"[trainer] restored best val={best_val:.4f}")

    logger.info("[trainer] FULL test eval...")
    test = _evaluate(model, test_loader, cfg, device)
    logger.info(
        f"[trainer] FINAL test_macro_f1={test['macro_f1']:.4f} "
        f"weighted={test['weighted_f1']:.4f} acc={test['accuracy']:.4f} "
        f"val={best_val:.4f} gap={best_val - test['macro_f1']:+.4f}"
    )
    return TrainResult(
        test["macro_f1"],
        test["weighted_f1"],
        test["accuracy"],
        best_val,
        _per_class(test["preds"], test["labels"], cfg.n_classes),
        _per_subgroup(test["preds"], test["labels"], test["languages"]),
        _per_subgroup(test["preds"], test["labels"], test["sources"]),
        step,
        time.time() - t0,
    )


def _code_style_features(code: str) -> Dict[str, float]:
    lines = code.splitlines() or [code]
    stripped = [ln.strip() for ln in lines]
    tokens = code.replace("\n", " ").split()
    n_chars = max(1, len(code))
    n_lines = max(1, len(lines))
    n_tokens = max(1, len(tokens))
    keywords = [
        "for",
        "while",
        "if",
        "else",
        "return",
        "class",
        "public",
        "static",
        "def",
        "import",
        "include",
        "try",
        "except",
    ]
    feats = {
        "chars": float(len(code)),
        "lines": float(n_lines),
        "tokens": float(n_tokens),
        "avg_line_len": float(sum(len(x) for x in lines) / n_lines),
        "avg_token_len": float(sum(len(x) for x in tokens) / n_tokens),
        "blank_ratio": float(sum(1 for x in stripped if not x) / n_lines),
        "indent_mean": float(sum(len(ln) - len(ln.lstrip(" ")) for ln in lines) / n_lines),
        "brace_rate": code.count("{") / n_chars,
        "paren_rate": code.count("(") / n_chars,
        "semicolon_rate": code.count(";") / n_chars,
        "comment_rate": (code.count("//") + code.count("#")) / n_chars,
        "underscore_rate": code.count("_") / n_chars,
        "digit_rate": sum(ch.isdigit() for ch in code) / n_chars,
    }
    low = " " + code.lower() + " "
    for kw in keywords:
        feats[f"kw_{kw}"] = low.count(f" {kw} ") / n_tokens
    return feats


def _load_style_splits(cfg: FSConfig):
    train_raw, val_raw, test_raw = _load_codet_splits(cfg.seed)
    vocab = _build_author_vocab(train_raw) if cfg.task == "author" else {}
    train_ds = _convert_codet(train_raw, cfg.task, vocab)
    val_ds = _convert_codet(val_raw, cfg.task, vocab)
    test_ds = _convert_codet(test_raw, cfg.task, vocab)

    labels = list(train_ds["label"])
    if cfg.k_shot > 0 and cfg.train_fraction <= 0:
        idxs, counts = kshot_stratified_indices(labels, cfg.k_shot, cfg.n_classes, cfg.fs_seed)
    else:
        idxs, counts = fraction_stratified_indices(labels, cfg.train_fraction, cfg.n_classes, cfg.fs_seed)
    train_ds = train_ds.select(idxs)
    val_idxs = build_minival_indices(list(val_ds["label"]), cfg.val_size_per_class, cfg.n_classes, cfg.fs_seed + 1000)
    val_ds = val_ds.select(val_idxs)
    if cfg.test_max_samples > 0:
        test_ds = test_ds.select(range(min(cfg.test_max_samples, len(test_ds))))
    return train_ds, val_ds, test_ds, counts, vocab


def run_style_repr_logreg(cfg: FSConfig, exp_id: str) -> Tuple[float, float, float]:
    t0 = time.time()
    train_ds, val_ds, test_ds, counts, vocab = _load_style_splits(cfg)
    x_train = [_code_style_features(row["code"]) for row in train_ds]
    y_train = list(train_ds["label"])
    x_val = [_code_style_features(row["code"]) for row in val_ds]
    y_val = list(val_ds["label"])
    x_test = [_code_style_features(row["code"]) for row in test_ds]
    y_test = list(test_ds["label"])

    clf = make_pipeline(
        DictVectorizer(sparse=False),
        StandardScaler(),
        LogisticRegression(max_iter=2000, class_weight="balanced", random_state=cfg.seed, n_jobs=-1),
    )
    clf.fit(x_train, y_train)
    val_pred = clf.predict(x_val)
    test_pred = clf.predict(x_test)
    val_macro = float(f1_score(y_val, val_pred, average="macro", zero_division=0))
    test_macro = float(f1_score(y_test, test_pred, average="macro", zero_division=0))
    test_weighted = float(f1_score(y_test, test_pred, average="weighted", zero_division=0))
    test_acc = float(accuracy_score(y_test, test_pred))
    rep = classification_report(y_test, test_pred, output_dict=True, zero_division=0)
    payload = {
        "test_macro_f1": test_macro,
        "test_weighted_f1": test_weighted,
        "test_accuracy": test_acc,
        "val_macro_f1": val_macro,
        "val_test_gap": val_macro - test_macro,
        "train_steps": 0,
        "wall_time_s": f"{time.time() - t0:.1f}",
        "paper": PAPER_REGISTRY["style_repr_logreg"]["paper"],
        "adaptation": PAPER_REGISTRY["style_repr_logreg"]["source"],
        "train_per_class": counts,
        "author_vocab": vocab,
        "per_class_f1": {f"class_{i}": float(rep.get(str(i), {}).get("f1-score", 0.0)) for i in range(cfg.n_classes)},
    }
    emit_json("Published-StyleRepr-LogReg", exp_id, cfg, payload)
    return test_macro, val_macro, time.time() - t0


def _parse_csv_env(name: str, default: str) -> List[str]:
    raw = os.environ.get(name, default).strip()
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_fracs() -> List[float]:
    return [float(x) for x in _parse_csv_env("FS_SWEEP_FRACS", "0.01,0.05,0.20")]


def run_deep_method(method: str, encoder: str, frac: float, exp_num: int, base_seed: int, benchmark: str):
    cfg = FSConfig(
        benchmark=benchmark,
        task="author",
        k_shot=0,
        train_fraction=frac,
        fs_seed=base_seed,
        encoder_name=encoder,
        lambda_ntk=float(os.environ.get("FS_LAMBDA_NTK", "0.4")),
        lambda_hier=float(os.environ.get("FS_LAMBDA_HIER", "0.4")),
        lr_encoder=float(os.environ.get("FS_LR_ENCODER", "2e-5")),
        lr_heads=float(os.environ.get("FS_LR_HEADS", "1e-4")),
    )
    cfg = apply_hardware_profile(cfg)
    train_l, val_l, test_l, counts, vocab = build_loaders(cfg)

    if method == "codet5_authorship":
        model = CodeT5AuthorshipClassifier(cfg)
        loss_fn = loss_ce
        method_name = "Published-CodeT5-Authorship"
    elif method == "detective_supcon":
        model = MultiTaskStyleClassifier(cfg)
        loss_fn = loss_detective_supcon
        method_name = "Published-DeTeCtive-SupCon"
    elif method == "faid_multitask":
        model = MultiTaskStyleClassifier(cfg)
        loss_fn = loss_faid_multitask
        method_name = "Published-FAID-MultiTask"
    else:
        raise ValueError(f"Unsupported deep method: {method}")

    result = train_with_loss(cfg, model, train_l, val_l, test_l, loss_fn, method_name=method_name)
    payload = {
        "test_macro_f1": result.test_macro_f1,
        "test_weighted_f1": result.test_weighted_f1,
        "test_accuracy": result.test_accuracy,
        "val_macro_f1": result.val_macro_f1,
        "val_test_gap": result.val_macro_f1 - result.test_macro_f1,
        "train_steps": result.train_steps,
        "wall_time_s": f"{result.wall_time_s:.1f}",
        "paper": PAPER_REGISTRY[method]["paper"],
        "adaptation": PAPER_REGISTRY[method]["source"],
        "train_per_class": counts,
        "author_vocab": vocab,
        "per_class_f1": result.per_class_f1,
        "per_lang_f1": result.per_lang_f1,
        "per_source_f1": result.per_source_f1,
    }
    emit_json(method_name, f"pub_{method}_{exp_num}", cfg, payload)
    cleanup()
    return result.test_macro_f1, result.val_macro_f1, result.wall_time_s


def main():
    methods = _parse_csv_env(
        "PUBLISHED_METHODS",
        "codet5_authorship,detective_supcon,faid_multitask,style_repr_logreg",
    )
    encoders = _parse_csv_env("FS_ENCODER", "ModernBERT-base,codebert-base,unixcoder-base")
    fractions = _parse_fracs()
    benchmark = os.environ.get("FS_BENCHMARK", "codet_m4")
    base_seed = int(os.environ.get("FS_SEED", "42"))
    set_seed(base_seed)

    print("=" * 80)
    print("Published SOTA reproduction/adaptation portfolio")
    print(f"methods={methods}")
    print(f"encoders={encoders}")
    print(f"fractions={fractions}")
    print(f"benchmark={benchmark} seed={base_seed}")
    print("=" * 80)

    rows = []
    exp_num = 0
    for method in methods:
        if method == "style_repr_logreg":
            for frac in fractions:
                exp_num += 1
                cfg = FSConfig(
                    benchmark=benchmark,
                    task="author",
                    k_shot=0,
                    train_fraction=frac,
                    fs_seed=base_seed,
                    encoder_name="style_repr",
                )
                cfg = apply_hardware_profile(cfg)
                test_f1, val_f1, wall = run_style_repr_logreg(cfg, f"pub_style_repr_{exp_num}")
                rows.append((method, "style_repr", frac, test_f1, val_f1, val_f1 - test_f1, wall))
                cleanup()
            continue

        method_encoders = encoders
        if method == "codet5_authorship":
            method_encoders = _parse_csv_env("CODET5_ENCODERS", "codet5-base")
        for encoder in method_encoders:
            for frac in fractions:
                exp_num += 1
                try:
                    test_f1, val_f1, wall = run_deep_method(method, encoder, frac, exp_num, base_seed, benchmark)
                    rows.append((method, encoder, frac, test_f1, val_f1, val_f1 - test_f1, wall))
                except Exception as exc:
                    logger.exception(f"[{method}] failed encoder={encoder} frac={frac}: {exc}")
                    rows.append((method, encoder, frac, float("nan"), float("nan"), float("nan"), 0.0))
                    cleanup()

    print("\n" + "=" * 110)
    print(f"{'Method':<24} {'Encoder':<24} {'Frac':>6} {'Test':>10} {'dPaper':>10} {'Val':>10} {'Gap':>10} {'Wall':>8}")
    print("-" * 110)
    for method, encoder, frac, test, val, gap, wall in rows:
        dp = test - PAPER_BASELINE if not math.isnan(test) else float("nan")
        print(f"{method:<24} {encoder:<24} {frac:>6.2%} {test:>10.4f} {dp:>+10.4f} {val:>10.4f} {gap:>+10.4f} {wall:>8.0f}")
    print("-" * 110)

    os.makedirs("./results", exist_ok=True)
    agg_path = os.path.join("./results", f"published_sota_portfolio_seed{base_seed}.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "benchmark": benchmark,
                "seed": base_seed,
                "paper_baseline": PAPER_BASELINE,
                "registry": PAPER_REGISTRY,
                "results": [
                    {
                        "method": m,
                        "encoder": e,
                        "frac": fr,
                        "test_macro_f1": t,
                        "val_macro_f1": v,
                        "gap": g,
                        "wall_s": w,
                    }
                    for m, e, fr, t, v, g, w in rows
                ],
            },
            f,
            indent=2,
        )
    print(f"Aggregate saved: {agg_path}")


if __name__ == "__main__":
    main()

