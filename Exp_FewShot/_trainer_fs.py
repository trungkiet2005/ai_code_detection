"""
Few-shot trainer: 1 epoch, no LR scheduler, early-stop on val plateau.

Design choices for K-shot:
  - 1 epoch over the K*N training samples (K=8..128, N=2..6) is enough.
    With K=128 N=6 = 768 samples / bs=16 = 48 steps. Multiple epochs overfit fast.
  - No warmup / cosine schedule — too few steps for them to matter.
  - Constant LR; early-stop monitors val Macro-F1 with patience.
  - Eval frequency adaptive: every floor(steps/8) steps, min 5.

Test eval at the end is on the FULL test split (paper-comparable).
"""
from __future__ import annotations

import math
import os
import time
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    recall_score,
)

from _common_fs import FSConfig, autocast, logger, make_grad_scaler


@dataclass
class FSResults:
    test_macro_f1: float
    test_weighted_f1: float
    test_accuracy: float
    val_macro_f1: float
    val_macro_f1_curve: List[float]
    per_class_f1: Dict[str, float]
    per_lang_f1: Dict[str, float]
    per_source_f1: Dict[str, float]
    train_steps: int
    wall_time_s: float


def _compute_class_weights(loader, n_classes: int) -> torch.Tensor:
    counts = np.zeros(n_classes, dtype=np.float64)
    for batch in loader:
        for lab in batch["labels"].tolist():
            counts[lab] += 1
    counts = np.maximum(counts, 1.0)
    weights = 1.0 / counts
    weights = weights / weights.sum() * n_classes
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def _evaluate(model, loader, cfg: FSConfig, device) -> Dict:
    model.eval()
    all_preds, all_labels, all_langs, all_srcs = [], [], [], []
    for batch in loader:
        ids = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = batch["labels"]
        amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16
        with autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                      enabled=device.type == "cuda", dtype=amp_dtype):
            outputs = model(ids, mask)
        preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds.tolist())
        all_labels.extend(labels.tolist())
        all_langs.extend(batch.get("languages", [""] * len(preds)))
        all_srcs.extend(batch.get("sources", [""] * len(preds)))
    macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    weighted = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    acc = accuracy_score(all_labels, all_preds)
    return {
        "macro_f1": float(macro),
        "weighted_f1": float(weighted),
        "accuracy": float(acc),
        "preds": all_preds,
        "labels": all_labels,
        "languages": all_langs,
        "sources": all_srcs,
    }


def _per_subgroup_f1(preds, labels, groups) -> Dict[str, float]:
    out = {}
    for g in sorted(set(groups)):
        if not g:
            continue
        idx = [i for i, x in enumerate(groups) if x == g]
        if len(idx) < 5:
            continue
        sub_preds = [preds[i] for i in idx]
        sub_labels = [labels[i] for i in idx]
        out[g] = float(f1_score(sub_labels, sub_preds, average="macro", zero_division=0))
    return out


def _per_class_f1(preds, labels, n_classes) -> Dict[str, float]:
    rep = classification_report(labels, preds, output_dict=True, zero_division=0)
    return {f"class_{i}": float(rep.get(str(i), {}).get("f1-score", 0.0)) for i in range(n_classes)}


def train_fewshot(
    cfg: FSConfig,
    model: nn.Module,
    train_loader,
    val_loader,
    test_loader,
    loss_fn: Callable,
    method_name: str = "FewShot",
) -> FSResults:
    """Run one few-shot training pass and return final test metrics."""
    device = torch.device(cfg.device)
    model = model.to(device)

    # Class weights from the K-shot train set (all K per class so usually uniform).
    class_weights = None
    if cfg.use_class_weights:
        class_weights = _compute_class_weights(train_loader, cfg.n_classes).to(device)
        logger.info(f"[trainer] class_weights={class_weights.tolist()}")

    optimizer = torch.optim.AdamW(model.param_groups())
    scaler = make_grad_scaler(enabled=(cfg.precision == "fp16" and device.type == "cuda"))
    amp_dtype = torch.float16 if cfg.precision == "fp16" else torch.bfloat16

    total_steps = max(1, len(train_loader) * cfg.epochs)
    eval_every = max(5, total_steps // 8)

    # Optional LR schedule for fraction mode (multi-epoch). K-shot keeps constant LR.
    scheduler = None
    if cfg.warmup_ratio > 0 and total_steps > 20:
        warmup_steps = max(1, int(total_steps * cfg.warmup_ratio))
        try:
            from transformers import get_cosine_schedule_with_warmup
            scheduler = get_cosine_schedule_with_warmup(
                optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
            )
            logger.info(f"[trainer] LR schedule: cosine, warmup={warmup_steps} of {total_steps}")
        except ImportError:
            scheduler = None

    logger.info(
        f"[trainer] method={method_name} total_steps={total_steps} "
        f"eval_every={eval_every} epochs={cfg.epochs} "
        f"lr_enc={cfg.lr_encoder} lr_head={cfg.lr_heads}"
    )

    best_val_f1 = -1.0
    best_state = None
    val_curve: List[float] = []
    plateau_count = 0
    step = 0
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        for batch in train_loader:
            step += 1
            ids = batch["input_ids"].to(device, non_blocking=True)
            mask = batch["attention_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                          enabled=device.type == "cuda", dtype=amp_dtype):
                outputs = model(ids, mask)
                losses = loss_fn(outputs, labels, class_weights=class_weights)
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

            if step % cfg.log_every == 0:
                comp = " ".join(f"{k}={v.item():.4f}" for k, v in losses.items())
                logger.info(f"[step {step}/{total_steps}] {comp}")

            if step % eval_every == 0 or step == total_steps:
                val = _evaluate(model, val_loader, cfg, device)
                val_curve.append(val["macro_f1"])
                logger.info(
                    f"[val @ step {step}] macro_f1={val['macro_f1']:.4f} "
                    f"weighted={val['weighted_f1']:.4f} acc={val['accuracy']:.4f}"
                )
                if val["macro_f1"] > best_val_f1 + 1e-4:
                    best_val_f1 = val["macro_f1"]
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                    plateau_count = 0
                else:
                    plateau_count += 1
                    if plateau_count >= cfg.early_stop_patience:
                        logger.info(f"[early-stop] no improve for {plateau_count} evals; stop.")
                        break
                model.train()
        else:
            continue
        break

    if best_state is not None:
        model.load_state_dict(best_state)
        logger.info(f"[trainer] restored best val={best_val_f1:.4f}")

    # FULL test eval
    logger.info(f"[trainer] running TEST on full split...")
    test = _evaluate(model, test_loader, cfg, device)

    # val-test gap (CLAUDE.md mandate)
    gap = best_val_f1 - test["macro_f1"]
    logger.info(
        f"[trainer] FINAL test_macro_f1={test['macro_f1']:.4f} "
        f"weighted={test['weighted_f1']:.4f} acc={test['accuracy']:.4f} "
        f"val_macro={best_val_f1:.4f} val-test_gap={gap:+.4f}"
    )

    return FSResults(
        test_macro_f1=test["macro_f1"],
        test_weighted_f1=test["weighted_f1"],
        test_accuracy=test["accuracy"],
        val_macro_f1=best_val_f1,
        val_macro_f1_curve=val_curve,
        per_class_f1=_per_class_f1(test["preds"], test["labels"], cfg.n_classes),
        per_lang_f1=_per_subgroup_f1(test["preds"], test["labels"], test["languages"]),
        per_source_f1=_per_subgroup_f1(test["preds"], test["labels"], test["sources"]),
        train_steps=step,
        wall_time_s=time.time() - t0,
    )
