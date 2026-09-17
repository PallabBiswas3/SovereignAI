from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np
from scipy.io import loadmat

from ..tools.bearing_hybrid import bearing_fault_frequencies


PADERBORN_SAMPLE_RATE_HZ = 64_000.0
PADERBORN_BEARING_LABELS = {
    **{f"K{i:03d}": "healthy" for i in range(1, 7)},
    **{code: "outer_race" for code in ("KA01", "KA03", "KA04", "KA05", "KA06", "KA07", "KA08", "KA09", "KA15", "KA16", "KA22", "KA30")},
    **{code: "inner_race" for code in ("KI01", "KI03", "KI04", "KI05", "KI07", "KI08", "KI14", "KI16", "KI17", "KI18", "KI21")},
    "KB23": "combined_bearing",
    "KB24": "combined_bearing",
    "KB27": "combined_bearing",
}


@dataclass(frozen=True)
class PaderbornRecord:
    path: Path
    bearing_id: str
    label: str
    rpm: float
    torque_nm: float
    radial_force_n: float
    run: int


_NAME = re.compile(
    r"N(?P<speed>\d+)_M(?P<torque>\d+)_F(?P<force>\d+)_(?P<bearing>K[A-Z]?\d{2,3})_(?P<run>\d+)$",
    re.IGNORECASE,
)


def parse_paderborn_filename(path: str | Path) -> PaderbornRecord:
    source = Path(path)
    match = _NAME.match(source.stem)
    if match is None:
        raise ValueError(f"unrecognized Paderborn filename: {source.name}")
    bearing = match.group("bearing").upper()
    if bearing not in PADERBORN_BEARING_LABELS:
        raise ValueError(f"bearing {bearing} has no reviewed label mapping")
    return PaderbornRecord(
        path=source,
        bearing_id=bearing,
        label=PADERBORN_BEARING_LABELS[bearing],
        rpm=float(match.group("speed")) * 100.0,
        torque_nm=float(match.group("torque")) / 10.0,
        radial_force_n=float(match.group("force")) * 100.0,
        run=int(match.group("run")),
    )


def discover_paderborn_records(root: str | Path, *, include_combined: bool = False) -> list[PaderbornRecord]:
    records: list[PaderbornRecord] = []
    for path in sorted(Path(root).rglob("*.mat")):
        try:
            record = parse_paderborn_filename(path)
        except ValueError:
            continue
        if record.label == "combined_bearing" and not include_combined:
            continue
        records.append(record)
    return records


def paderborn_specimen_split(records: list[PaderbornRecord], test_fraction: float = 0.3) -> tuple[list[PaderbornRecord], list[PaderbornRecord]]:
    """Deterministically split complete bearing specimens, never windows/files."""
    by_label: dict[str, list[str]] = {}
    for record in records:
        by_label.setdefault(record.label, []).append(record.bearing_id)
    test_ids: set[str] = set()
    for ids in by_label.values():
        unique = sorted(set(ids))
        count = max(1, int(round(len(unique) * float(test_fraction)))) if len(unique) > 1 else 0
        test_ids.update(unique[-count:] if count else [])
    return (
        [record for record in records if record.bearing_id not in test_ids],
        [record for record in records if record.bearing_id in test_ids],
    )


def _walk(value: Any, prefix: str = "") -> Iterator[tuple[str, np.ndarray]]:
    if isinstance(value, dict):
        name = value.get("Name") or value.get("name") or value.get("signal_name")
        data = value.get("Data") if "Data" in value else value.get("data")
        if name is not None and data is not None:
            arr = np.asarray(data).squeeze()
            if arr.ndim == 1 and arr.size > 32 and np.issubdtype(arr.dtype, np.number):
                yield str(name), arr.astype(float, copy=False)
        for key, item in value.items():
            yield from _walk(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from _walk(item, f"{prefix}.{index}")
    elif isinstance(value, np.ndarray):
        if value.dtype == object or value.dtype.names:
            for item in value.ravel():
                yield from _walk(item, prefix)
        else:
            arr = np.asarray(value).squeeze()
            if arr.ndim == 1 and arr.size > 32 and np.issubdtype(arr.dtype, np.number):
                yield prefix, arr.astype(float, copy=False)


def _select(signals: list[tuple[str, np.ndarray]], terms: tuple[str, ...], *, required: bool = True) -> np.ndarray | None:
    matches = [(name, values) for name, values in signals if all(term in name.lower() for term in terms)]
    if not matches:
        if required:
            names = sorted({name for name, _ in signals})
            raise ValueError(f"could not find channel terms {terms}; available paths: {names[:20]}")
        return None
    return max(matches, key=lambda item: item[1].size)[1]


def load_paderborn_record(record: PaderbornRecord) -> dict[str, Any]:
    payload = loadmat(record.path, simplify_cells=True)
    signals = list(_walk({key: value for key, value in payload.items() if not key.startswith("__")}))
    vibration = _select(signals, ("vibration",))
    current_1 = _select(signals, ("phase_current_1",), required=False)
    current_2 = _select(signals, ("phase_current_2",), required=False)
    if current_1 is None:
        current_1 = _select(signals, ("current", "1"))
    speed = _select(signals, ("speed",), required=False)
    n = min(len(vibration), len(current_1))
    vibration = vibration[:n]
    current_1 = current_1[:n]
    if current_2 is not None:
        current_2 = current_2[:n]
        current = np.sqrt(np.maximum(current_1**2 + current_2**2 + current_1 * current_2, 0.0))
    else:
        current = current_1
    shaft_rate = record.rpm / 60.0
    return {
        "vibration": vibration,
        "current": current,
        "speed": speed,
        "sampling_rate_hz": PADERBORN_SAMPLE_RATE_HZ,
        "shaft_rate_hz": shaft_rate,
        "fault_frequencies": bearing_fault_frequencies(shaft_rate),
        "label": record.label,
        "bearing_id": record.bearing_id,
        "rpm": record.rpm,
        "operating_condition": {
            "torque_nm": record.torque_nm,
            "radial_force_n": record.radial_force_n,
            "run": record.run,
        },
    }


def nonoverlapping_multichannel_windows(channels, window_samples: int, max_windows: int) -> list[np.ndarray]:
    x = np.asarray(channels, dtype=float)
    if x.ndim != 2:
        raise ValueError("channels must be [channels, samples]")
    size = int(window_samples)
    possible = x.shape[1] // size
    count = min(possible, max(1, int(max_windows)))
    if count < 1:
        return []
    starts = np.linspace(0, x.shape[1] - size, count, dtype=int)
    return [x[:, start:start + size].copy() for start in starts]
