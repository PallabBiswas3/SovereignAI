from __future__ import annotations

from collections import Counter, defaultdict
from random import Random
from typing import Sequence, TypeVar


T = TypeVar("T")


def _label(item: object) -> str:
    label = getattr(item, "ground_truth_label", None)
    return str(getattr(label, "value", label) or "unlabeled")


def sample_records(
    records: Sequence[T],
    max_records: int | None,
    *,
    method: str = "stratified_random",
    seed: int = 42,
) -> list[T]:
    """Select a deterministic evaluation sample without ordered-prefix bias.

    RAGTruth records are grouped by source and generator, so ``records[:N]``
    can contain many near-neighbour examples. Stratified random sampling keeps
    the available response-label prevalence while shuffling those groups.
    """
    items = list(records)
    if max_records is None or max_records >= len(items):
        return items
    if max_records <= 0:
        return []

    rng = Random(seed)
    if method == "sequential":
        return items[:max_records]
    if method == "random":
        rng.shuffle(items)
        return items[:max_records]
    if method != "stratified_random":
        raise ValueError(
            "sampling method must be one of: sequential, random, stratified_random"
        )

    buckets: dict[str, list[T]] = defaultdict(list)
    for item in items:
        buckets[_label(item)].append(item)
    for bucket in buckets.values():
        rng.shuffle(bucket)

    total = len(items)
    exact = {
        label: max_records * len(bucket) / total
        for label, bucket in buckets.items()
    }
    quotas = {label: int(value) for label, value in exact.items()}
    remaining = max_records - sum(quotas.values())
    allocation_order = sorted(
        buckets,
        key=lambda label: (exact[label] - quotas[label], len(buckets[label]), label),
        reverse=True,
    )
    while remaining:
        allocated = False
        for label in allocation_order:
            if quotas[label] < len(buckets[label]):
                quotas[label] += 1
                remaining -= 1
                allocated = True
                if remaining == 0:
                    break
        if not allocated:
            break

    selected = [
        item
        for label, bucket in buckets.items()
        for item in bucket[: quotas[label]]
    ]
    rng.shuffle(selected)
    return selected


def sampling_summary(records: Sequence[object]) -> dict[str, object]:
    return {
        "label_counts": dict(Counter(_label(record) for record in records)),
        "unique_source_ids": len(
            {
                getattr(record, "source_id", None)
                for record in records
                if getattr(record, "source_id", None) is not None
            }
        ),
    }
