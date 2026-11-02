"""Compatibility API backed by the repository's one split manifest contract.

New callers should use DevelopmentSplitManifest and SplitUnit directly. The
compact SplitManifest constructor delegates boundary checks and checksums to
evaluation.split_manifest. It has no independent schema or leakage validator.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Mapping, Sequence

import numpy as np

from ..evaluation.split_manifest import DevelopmentSplitManifest, SplitUnit

__all__ = ["LeakageError", "SplitManifest", "group_split"]


class LeakageError(ValueError):
    """Development/validation/evaluation units violate their declared boundary."""


def _time_text(value) -> str:
    if isinstance(value, (int, float, np.number)):
        if not np.isfinite(value):
            raise LeakageError("interval endpoints must be finite")
        return datetime.fromtimestamp(float(value), timezone.utc).isoformat()
    return str(value)


@dataclass(frozen=True, eq=False)
class SplitManifest:
    boundary: str
    dev_ids: tuple[str, ...]
    eval_ids: tuple[str, ...]
    asset_of: Mapping[str, str] | None = None
    intervals: Mapping[str, tuple[float, float]] | None = None

    def to_manifest(self) -> DevelopmentSplitManifest:
        boundaries = {"event": "unit", "asset": "group", "temporal": "temporal"}
        if self.boundary not in boundaries:
            raise LeakageError(f"unknown boundary {self.boundary!r}")

        def unit(identifier):
            asset = (self.asset_of or {}).get(identifier)
            interval = (self.intervals or {}).get(identifier)
            if interval is not None and len(interval) != 2:
                raise LeakageError("intervals must contain start and end")
            return SplitUnit(
                unit_id=identifier, group_ids=(asset,) if asset else (),
                start_time=_time_text(interval[0]) if interval is not None else None,
                end_time=_time_text(interval[1]) if interval is not None else None)

        manifest = DevelopmentSplitManifest(
            dataset_id="standalone-unspecified", development_pool_id="development",
            evaluation_pool_id="evaluation", leakage_boundary=boundaries[self.boundary],
            unit_kind="event", group_kind=None if self.boundary == "event" else "asset",
            development_units=tuple(unit(i) for i in sorted(self.dev_ids)),
            evaluation_units=tuple(unit(i) for i in sorted(self.eval_ids)))
        return canonical_manifest(manifest)

    def validate(self) -> SplitManifest:
        self.to_manifest()
        return self

    def checksum(self) -> str:
        return self.to_manifest().sha256()


def canonical_manifest(manifest: SplitManifest | DevelopmentSplitManifest) -> DevelopmentSplitManifest:
    if isinstance(manifest, SplitManifest):
        return manifest.to_manifest()
    if not isinstance(manifest, DevelopmentSplitManifest):
        raise TypeError("manifest must use the development split manifest contract")
    try:
        manifest.validate()
    except ValueError as exc:
        raise LeakageError(str(exc)) from exc
    return manifest


def group_split(ids: Sequence[str], groups: Mapping[str, str] | None = None,
                fraction: float = 0.25, seed: int = 0) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split development IDs while keeping every physical group together."""
    ids = list(ids)
    if len(ids) != len(set(ids)):
        raise LeakageError("duplicate ids within the development pool")
    if not 0 < fraction < 1:
        raise ValueError("fraction must be in (0, 1)")
    if groups is not None and any(not groups.get(i) for i in ids):
        raise LeakageError("every development ID must have a group identity")
    group_of = {i: groups[i] if groups is not None else i for i in ids}
    unique = sorted(set(group_of.values()))
    np.random.default_rng(seed).shuffle(unique)
    count = min(len(unique) - 1, max(1, round(fraction * len(unique)))) if len(unique) > 1 else 0
    held = set(unique[:count])
    return (tuple(i for i in ids if group_of[i] not in held),
            tuple(i for i in ids if group_of[i] in held))
