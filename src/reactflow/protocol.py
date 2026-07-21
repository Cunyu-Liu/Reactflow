"""Protocol-safe labels and deterministic sample selection for C0."""

from __future__ import annotations

import hashlib
from typing import Callable, Sequence, Tuple, TypeVar


MMSEQS_COMPONENT_TEST = "mmseqs_component_test"
MMSEQS_COMPONENT_HOLDOUT = "mmseqs_component_holdout"
LEGACY_TIER_ALIASES = {
    "in_clan": MMSEQS_COMPONENT_TEST,
    "novel_clan": MMSEQS_COMPONENT_HOLDOUT,
}


def normalize_tier_label(label: str) -> str:
    """Return an evidence-faithful tier label while accepting old artifacts."""

    cleaned = str(label).strip()
    if not cleaned:
        raise ValueError("tier label must be non-empty")
    return LEGACY_TIER_ALIASES.get(cleaned, cleaned)


def stable_sample_key(source_id: object, sequence: str, *, seed: int = 20260718) -> str:
    """Return the fixed C0 SHA-256 sampling key."""

    payload = f"{source_id or ''}|{str(sequence).upper()}|{int(seed)}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


T = TypeVar("T")


def stable_subset(
    items: Sequence[T],
    count: int,
    *,
    source_id: Callable[[T], object],
    sequence: Callable[[T], str],
    seed: int = 20260718,
) -> Tuple[T, ...]:
    """Select a deterministic hash-ordered subset without using target data."""

    if count < 0:
        raise ValueError("count must be non-negative")
    ordered = sorted(
        items,
        key=lambda item: stable_sample_key(source_id(item), sequence(item), seed=seed),
    )
    return tuple(ordered[:count])
