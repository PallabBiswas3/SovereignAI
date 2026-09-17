from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve

import numpy as np
from scipy.io import loadmat


CWRU_BASE_URL = "https://engineering.case.edu/sites/default/files"
CWRU_SAMPLE_RATE_HZ = 12_000.0

# Official Case Western Reserve University drive-end bearing defect-frequency
# multipliers, expressed relative to shaft running speed in Hz for the
# 6205-2RS JEM SKF bearing used by the 0.007/0.014/0.021-inch experiments.
CWRU_DE_FREQUENCY_MULTIPLIERS = {
    "BPFI": 5.4152,
    "BPFO": 3.5848,
    "FTF": 0.39828,
    "BSF": 4.7135,
}


@dataclass(frozen=True)
class CWRUBearingRecord:
    file_id: int
    label: str
    fault_code: str | None
    load_hp: int
    approx_rpm: float
    fault_diameter_in: float | None = None
    outer_race_position: str | None = None

    @property
    def filename(self) -> str:
        return f"{self.file_id}.mat"

    @property
    def url(self) -> str:
        return f"{CWRU_BASE_URL}/{self.filename}"


# Controlled first benchmark: one fault severity (0.007 in), one bearing family
# (SKF), drive-end 12 kHz data, and the same outer-race position across all loads.
# This avoids mixing severity/manufacturer/location confounders into the first
# cross-load validation.
_CWRU_007_DE_RECORDS = (
    CWRUBearingRecord(97, "normal", None, 0, 1797.0),
    CWRUBearingRecord(98, "normal", None, 1, 1772.0),
    CWRUBearingRecord(99, "normal", None, 2, 1750.0),
    CWRUBearingRecord(100, "normal", None, 3, 1730.0),
    CWRUBearingRecord(105, "inner_race", "BPFI", 0, 1797.0, 0.007),
    CWRUBearingRecord(106, "inner_race", "BPFI", 1, 1772.0, 0.007),
    CWRUBearingRecord(107, "inner_race", "BPFI", 2, 1750.0, 0.007),
    CWRUBearingRecord(108, "inner_race", "BPFI", 3, 1730.0, 0.007),
    CWRUBearingRecord(118, "rolling_element", "BSF", 0, 1797.0, 0.007),
    CWRUBearingRecord(119, "rolling_element", "BSF", 1, 1772.0, 0.007),
    CWRUBearingRecord(120, "rolling_element", "BSF", 2, 1750.0, 0.007),
    CWRUBearingRecord(121, "rolling_element", "BSF", 3, 1730.0, 0.007),
    CWRUBearingRecord(130, "outer_race", "BPFO", 0, 1797.0, 0.007, "6:00"),
    CWRUBearingRecord(131, "outer_race", "BPFO", 1, 1772.0, 0.007, "6:00"),
    CWRUBearingRecord(132, "outer_race", "BPFO", 2, 1750.0, 0.007, "6:00"),
    CWRUBearingRecord(133, "outer_race", "BPFO", 3, 1730.0, 0.007, "6:00"),
)


def cwru_007_drive_end_manifest() -> list[CWRUBearingRecord]:
    return list(_CWRU_007_DE_RECORDS)


def download_cwru_007_drive_end(
    destination: str | Path,
    *,
    records: Iterable[CWRUBearingRecord] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    for record in records or _CWRU_007_DE_RECORDS:
        path = root / record.filename
        if overwrite or not path.exists() or path.stat().st_size == 0:
            urlretrieve(record.url, path)
        paths.append(path)
    return paths


def _pick_vector(payload: dict, suffix: str) -> np.ndarray | None:
    candidates = [
        (key, np.asarray(value, dtype=float).squeeze())
        for key, value in payload.items()
        if not key.startswith("__") and key.lower().endswith(suffix.lower())
    ]
    vectors = [(key, value) for key, value in candidates if value.ndim == 1 and value.size > 1]
    if not vectors:
        return None
    # Prefer the longest compatible vector; this robustly selects the waveform
    # when auxiliary arrays are present.
    _, vector = max(vectors, key=lambda item: item[1].size)
    return vector.astype(float, copy=False)


def _pick_scalar(payload: dict, suffix: str) -> float | None:
    for key, value in payload.items():
        if key.startswith("__") or not key.lower().endswith(suffix.lower()):
            continue
        arr = np.asarray(value, dtype=float).squeeze()
        if arr.size == 1 and np.isfinite(float(arr)):
            return float(arr)
    return None


def load_cwru_drive_end(
    path: str | Path,
    record: CWRUBearingRecord,
) -> dict:
    payload = loadmat(Path(path), squeeze_me=True)
    signal = _pick_vector(payload, "_DE_time")
    if signal is None:
        raise ValueError(f"{Path(path).name} does not contain a drive-end waveform")
    rpm = _pick_scalar(payload, "RPM")
    if rpm is None or rpm <= 0.0:
        rpm = float(record.approx_rpm)

    shaft_rate_hz = float(rpm) / 60.0
    return {
        "signal": signal,
        "sampling_rate_hz": CWRU_SAMPLE_RATE_HZ,
        "rpm": float(rpm),
        "shaft_rate_hz": shaft_rate_hz,
        "fault_frequencies": {
            code: float(multiplier * shaft_rate_hz)
            for code, multiplier in CWRU_DE_FREQUENCY_MULTIPLIERS.items()
        },
        "label": record.label,
        "fault_code": record.fault_code,
        "load_hp": int(record.load_hp),
        "file_id": int(record.file_id),
        "fault_diameter_in": record.fault_diameter_in,
        "outer_race_position": record.outer_race_position,
    }


def evenly_spaced_windows(
    signal,
    sampling_rate_hz: float,
    *,
    window_seconds: float = 1.0,
    max_windows: int = 6,
) -> list[np.ndarray]:
    x = np.asarray(signal, dtype=float).ravel()
    n = int(round(float(window_seconds) * float(sampling_rate_hz)))
    if n <= 0:
        raise ValueError("window_seconds and sampling_rate_hz must define a positive window")
    if x.size < n:
        raise ValueError(f"signal has {x.size} samples but a window needs {n}")

    possible = x.size // n
    count = min(max(1, int(max_windows)), possible)
    if count == 1:
        starts = np.asarray([0], dtype=int)
    else:
        starts = np.linspace(0, x.size - n, num=count, dtype=int)
    return [x[int(start): int(start) + n].copy() for start in starts]
