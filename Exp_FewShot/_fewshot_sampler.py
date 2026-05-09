"""
K-shot stratified sampler for few-shot AI-code detection.

Given a HuggingFace `Dataset` with integer `label` field, returns a SUBSET
of size <= K * n_classes, with EXACTLY K examples per class (or fewer if a
class has < K available — caller is warned).

Sampling is deterministic given (k_shot, seed). The chosen indices are
also returned so they can be logged for reproducibility — every paper run
must report (k_shot, seed, n_per_class) in the tracker row.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

from datasets import Dataset


def kshot_stratified_indices(
    labels: List[int],
    k_shot: int,
    n_classes: int,
    seed: int = 42,
) -> Tuple[List[int], Dict[int, int]]:
    """Return (indices, per_class_counts) for a K-shot subset.

    Args:
        labels: integer labels of size N (one per row).
        k_shot: examples per class to sample.
        n_classes: expected number of distinct classes.
        seed: RNG seed for reproducibility.

    Returns:
        indices:           list of length <= k_shot * n_classes
        per_class_counts:  {class_id -> sampled count} for the leaderboard row.

    Raises:
        ValueError on inconsistent input.
    """
    if k_shot <= 0:
        raise ValueError(f"k_shot must be positive, got {k_shot}")
    if n_classes <= 0:
        raise ValueError(f"n_classes must be positive, got {n_classes}")

    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        if lab < 0:
            continue
        by_class[int(lab)].append(idx)

    rng = random.Random(seed)
    chosen: List[int] = []
    counts: Dict[int, int] = {}
    for cls in range(n_classes):
        pool = by_class.get(cls, [])
        n_take = min(k_shot, len(pool))
        if n_take < k_shot:
            print(
                f"[kshot] WARNING class {cls} has only {len(pool)} samples "
                f"(< K={k_shot}); using all available."
            )
        if n_take > 0:
            sampled = rng.sample(pool, n_take)
            chosen.extend(sampled)
        counts[cls] = n_take

    rng.shuffle(chosen)
    return chosen, counts


def kshot_stratified_subset(
    dataset: Dataset,
    k_shot: int,
    n_classes: int,
    seed: int = 42,
    label_field: str = "label",
) -> Tuple[Dataset, Dict[int, int]]:
    """K-shot sample from a HF Dataset, returning (subset, per_class_counts)."""
    labels = list(dataset[label_field])
    indices, counts = kshot_stratified_indices(labels, k_shot, n_classes, seed)
    subset = dataset.select(indices) if indices else dataset.select([])
    return subset, counts


def fraction_stratified_subset(
    dataset: Dataset,
    fraction: float,
    n_classes: int,
    seed: int = 42,
    label_field: str = "label",
) -> Tuple[Dataset, Dict[int, int]]:
    """Take `fraction` of the dataset, stratified by label.

    Used for the phase-transition study: we want a clean "X% of full train"
    sweep that scales each class proportionally (preserves class imbalance,
    unlike K-shot which forces per-class equality).

    Args:
        dataset:   HF Dataset with `label_field`
        fraction:  0 < fraction <= 1.0
        n_classes: expected class count (for logging only; we sample whatever exists)
        seed:      RNG seed for reproducibility

    Returns:
        subset, {class_id -> sampled_count}
    """
    if not (0.0 < fraction <= 1.0):
        raise ValueError(f"fraction must be in (0, 1], got {fraction}")

    labels = list(dataset[label_field])
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        if lab >= 0:
            by_class[int(lab)].append(idx)

    rng = random.Random(seed)
    chosen: List[int] = []
    counts: Dict[int, int] = {}
    for cls in range(n_classes):
        pool = by_class.get(cls, [])
        n_take = max(1, int(round(len(pool) * fraction))) if pool else 0
        if n_take > 0:
            chosen.extend(rng.sample(pool, n_take))
        counts[cls] = n_take

    rng.shuffle(chosen)
    subset = dataset.select(chosen) if chosen else dataset.select([])
    return subset, counts


def report_fraction_distribution(counts: Dict[int, int], fraction: float) -> str:
    """Format fraction-mode counts for logging."""
    parts = [f"class{cls}={n}" for cls, n in sorted(counts.items())]
    total = sum(counts.values())
    return f"{' '.join(parts)} (total={total}, fraction={fraction:.4f})"


def report_kshot_distribution(counts: Dict[int, int], k_shot: int) -> str:
    """Format K-shot counts for logging:  class0=32 class1=32 ... (total=192)."""
    parts = [f"class{cls}={n}" for cls, n in sorted(counts.items())]
    total = sum(counts.values())
    return " ".join(parts) + f" (total={total}, target={k_shot * len(counts)})"


def build_minival_indices(
    labels: List[int],
    n_per_class: int,
    n_classes: int,
    seed: int = 1234,
    exclude: List[int] = None,
) -> List[int]:
    """Held-out mini-val from validation pool (separate seed from train K-shot).

    Used to avoid touching the FULL test split for early-stopping / model selection.
    """
    exclude_set = set(exclude or [])
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, lab in enumerate(labels):
        if idx in exclude_set or lab < 0:
            continue
        by_class[int(lab)].append(idx)

    rng = random.Random(seed)
    chosen: List[int] = []
    for cls in range(n_classes):
        pool = by_class.get(cls, [])
        n_take = min(n_per_class, len(pool))
        if n_take > 0:
            chosen.extend(rng.sample(pool, n_take))
    rng.shuffle(chosen)
    return chosen


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("[smoke] K-shot sampler self-test")
    fake_labels = [i % 6 for i in range(2000)]
    for k in (8, 16, 32, 64, 128):
        idxs, counts = kshot_stratified_indices(fake_labels, k_shot=k, n_classes=6, seed=42)
        assert len(idxs) == k * 6, f"K={k}: got {len(idxs)}, expected {k*6}"
        assert all(c == k for c in counts.values()), f"K={k}: per-class={counts}"
        print(f"  K={k:>3} -> total={len(idxs)}  per-class={counts}")
    print("[smoke] PASSED")
