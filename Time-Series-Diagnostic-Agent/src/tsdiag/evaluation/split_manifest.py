from __future__ import annotations

"""Leakage-safe development/evaluation partition manifests.

The manifest records *why* two partitions are considered disjoint.  Domain
code supplies the correct physical unit (event, asset, specimen, engine, and
so on); this module enforces the selected boundary and freezes the decision in
a checksummed JSON artifact.
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping


SPLIT_MANIFEST_SCHEMA = "tsdiag.development-split"
SPLIT_MANIFEST_VERSION = "1.0"
LeakageBoundary = Literal["unit", "group", "temporal"]


def _timestamp(value: str | None, *, field_name: str) -> datetime | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    except ValueError as exc:
        raise ValueError(f"invalid {field_name} timestamp: {value!r}") from exc


@dataclass(frozen=True)
class SplitUnit:
    """One indivisible evaluation unit and its leakage-relevant identity."""

    unit_id: str
    group_ids: tuple[str, ...] = ()
    start_time: str | None = None
    end_time: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "SplitUnit":
        return cls(
            unit_id=str(value["unit_id"]),
            group_ids=tuple(str(item) for item in value.get("group_ids", ())),
            start_time=value.get("start_time"),
            end_time=value.get("end_time"),
            metadata=dict(value.get("metadata") or {}),
        )


@dataclass(frozen=True)
class DevelopmentSplitManifest:
    dataset_id: str
    development_pool_id: str
    evaluation_pool_id: str
    leakage_boundary: LeakageBoundary
    unit_kind: str
    group_kind: str | None
    development_units: tuple[SplitUnit, ...]
    evaluation_units: tuple[SplitUnit, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.leakage_boundary not in {"unit", "group", "temporal"}:
            raise ValueError(f"unsupported leakage boundary: {self.leakage_boundary}")
        for name, value in (
            ("dataset_id", self.dataset_id),
            ("development_pool_id", self.development_pool_id),
            ("evaluation_pool_id", self.evaluation_pool_id),
            ("unit_kind", self.unit_kind),
        ):
            if not str(value).strip():
                raise ValueError(f"{name} is required")
        if not self.development_units:
            raise ValueError("development partition must contain at least one unit")
        if not self.evaluation_units:
            raise ValueError("evaluation partition must contain at least one unit")

        development_ids = [unit.unit_id for unit in self.development_units]
        evaluation_ids = [unit.unit_id for unit in self.evaluation_units]
        if len(development_ids) != len(set(development_ids)):
            raise ValueError("development unit IDs must be unique")
        if len(evaluation_ids) != len(set(evaluation_ids)):
            raise ValueError("evaluation unit IDs must be unique")
        overlap = sorted(set(development_ids) & set(evaluation_ids))
        if overlap:
            raise ValueError(f"development units overlap frozen evaluation units: {overlap}")

        if self.leakage_boundary == "unit":
            return
        if not self.group_kind:
            raise ValueError("group_kind is required for group or temporal boundaries")
        all_units = self.development_units + self.evaluation_units
        missing_groups = [unit.unit_id for unit in all_units if not unit.group_ids]
        if missing_groups:
            raise ValueError(
                f"{self.group_kind} identity is missing for units: {missing_groups}"
            )

        development_groups = {
            group for unit in self.development_units for group in unit.group_ids
        }
        evaluation_groups = {
            group for unit in self.evaluation_units for group in unit.group_ids
        }
        shared_groups = sorted(development_groups & evaluation_groups)
        if self.leakage_boundary == "group":
            if shared_groups:
                raise ValueError(
                    f"development and frozen evaluation share {self.group_kind} groups: "
                    f"{shared_groups}"
                )
            return

        # A temporal boundary deliberately permits the same physical group in
        # both partitions, but only when every development interval ends before
        # every evaluation interval for that group begins.
        intervals: dict[tuple[str, str], list[tuple[datetime, datetime, str]]] = {}
        for partition_name, units in (
            ("development", self.development_units),
            ("evaluation", self.evaluation_units),
        ):
            for unit in units:
                start = _timestamp(unit.start_time, field_name="start_time")
                end = _timestamp(unit.end_time, field_name="end_time")
                if start is None or end is None:
                    raise ValueError(
                        f"temporal boundary requires start/end timestamps for unit {unit.unit_id}"
                    )
                if end < start:
                    raise ValueError(f"unit {unit.unit_id} ends before it starts")
                for group in unit.group_ids:
                    intervals.setdefault((partition_name, group), []).append(
                        (start, end, unit.unit_id)
                    )
        violations = []
        for group in shared_groups:
            development_end = max(
                end for _, end, _ in intervals[("development", group)]
            )
            evaluation_start = min(
                start for start, _, _ in intervals[("evaluation", group)]
            )
            if development_end >= evaluation_start:
                violations.append(group)
        if violations:
            raise ValueError(
                "development is not strictly earlier than frozen evaluation for "
                f"{self.group_kind} groups: {violations}"
            )

    def to_dict(self, *, include_checksum: bool = True) -> dict[str, Any]:
        self.validate()
        payload = {
            "schema": SPLIT_MANIFEST_SCHEMA,
            "schema_version": SPLIT_MANIFEST_VERSION,
            "dataset_id": self.dataset_id,
            "development_pool_id": self.development_pool_id,
            "evaluation_pool_id": self.evaluation_pool_id,
            "leakage_boundary": self.leakage_boundary,
            "unit_kind": self.unit_kind,
            "group_kind": self.group_kind,
            "development_units": [asdict(unit) for unit in self.development_units],
            "evaluation_units": [asdict(unit) for unit in self.evaluation_units],
            "metadata": self.metadata,
        }
        if include_checksum:
            payload["manifest_sha256"] = self.sha256()
        return payload

    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(include_checksum=False),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def assert_development_units(self, unit_ids: Iterable[str | int]) -> None:
        expected = {unit.unit_id for unit in self.development_units}
        observed = {str(unit_id) for unit_id in unit_ids}
        if observed != expected:
            raise ValueError(
                "calibration inputs do not match the development split manifest: "
                f"missing={sorted(expected - observed)}, unexpected={sorted(observed - expected)}"
            )

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "DevelopmentSplitManifest":
        if payload.get("schema") != SPLIT_MANIFEST_SCHEMA:
            raise ValueError("unsupported development split manifest schema")
        if payload.get("schema_version") != SPLIT_MANIFEST_VERSION:
            raise ValueError("unsupported development split manifest version")
        manifest = cls(
            dataset_id=str(payload["dataset_id"]),
            development_pool_id=str(payload["development_pool_id"]),
            evaluation_pool_id=str(payload["evaluation_pool_id"]),
            leakage_boundary=str(payload["leakage_boundary"]),  # type: ignore[arg-type]
            unit_kind=str(payload["unit_kind"]),
            group_kind=payload.get("group_kind"),
            development_units=tuple(
                SplitUnit.from_mapping(unit) for unit in payload["development_units"]
            ),
            evaluation_units=tuple(
                SplitUnit.from_mapping(unit) for unit in payload["evaluation_units"]
            ),
            metadata=dict(payload.get("metadata") or {}),
        )
        manifest.validate()
        expected = payload.get("manifest_sha256")
        if expected is not None and str(expected) != manifest.sha256():
            raise ValueError("development split manifest checksum mismatch")
        return manifest


def save_split_manifest(
    manifest: DevelopmentSplitManifest,
    path: str | Path,
) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest.to_dict(), indent=2), encoding="utf-8")
    return target


def load_split_manifest(path: str | Path) -> DevelopmentSplitManifest:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return DevelopmentSplitManifest.from_dict(payload)
