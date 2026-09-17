from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen, urlretrieve
from zipfile import ZipFile

import numpy as np
import pandas as pd
from scipy.io import loadmat


LENZE_SAMPLE_RATE_HZ = 16_000.0
LENZE_RECORD_URL = "https://zenodo.org/records/14762423/files"
LENZE_ARCHIVE_SIZE = 3_220_660_245
LENZE_COLUMNS = (
    "sampling_time", "phase_current_u", "phase_current_v", "phase_current_w",
    "phase_voltage_u", "phase_voltage_v", "phase_voltage_w", "dc_bus_voltage",
    "mechanical_angle", "current_vector", "mechanical_speed", "speed_deviation",
)


@dataclass(frozen=True)
class LenzeRecord:
    path: Path
    record_id: str
    label: str
    rpm: float
    belt_tension: float | None = None
    counter_momentum: float | None = None


def download_lenze_mb(destination: str | Path, *, include_data: bool = True) -> dict[str, Path]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    metadata = root / "Meta_Data.xlsx"
    if not metadata.exists():
        urlretrieve(f"{LENZE_RECORD_URL}/Meta_Data.xlsx?download=1", metadata)
    paths = {"metadata": metadata}
    if not include_data:
        return paths
    archive = root / "Data.zip"
    if not archive.exists() or archive.stat().st_size != LENZE_ARCHIVE_SIZE:
        _download_resumable(f"{LENZE_RECORD_URL}/Data.zip?download=1", archive, LENZE_ARCHIVE_SIZE)
    data_dir = root / "Data"
    if not data_dir.exists():
        root_resolved = root.resolve()
        with ZipFile(archive) as handle:
            for member in handle.infolist():
                target = (root / member.filename).resolve()
                if root_resolved not in target.parents and target != root_resolved:
                    raise ValueError(f"unsafe archive member: {member.filename}")
            handle.extractall(root)
    paths.update({"archive": archive, "data": data_dir})
    return paths


def _download_resumable(url: str, target: Path, expected_size: int, chunk_size: int = 1024 * 1024) -> None:
    current = target.stat().st_size if target.exists() else 0
    if current > expected_size:
        raise ValueError(f"partial archive is larger than expected: {current} > {expected_size}")
    request = Request(url, headers={"Range": f"bytes={current}-"} if current else {})
    with urlopen(request) as response:
        status = getattr(response, "status", None)
        append = current > 0 and status == 206
        if current > 0 and not append:
            current = 0
        with target.open("ab" if append else "wb") as handle:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                handle.write(chunk)
    if target.stat().st_size != expected_size:
        raise IOError(f"incomplete Lenze archive: {target.stat().st_size} of {expected_size} bytes")


def load_lenze_metadata(path: str | Path) -> list[LenzeRecord]:
    source = Path(path)
    frame = pd.read_excel(source)
    columns = {str(column).strip().lower(): column for column in frame.columns}
    id_col = columns.get("id", frame.columns[0])
    condition_col = columns.get("condition")
    rpm_col = columns.get("motor speed (rpm)")
    if condition_col is None or rpm_col is None:
        raise ValueError("Lenze metadata must contain ID, Condition and Motor Speed (RPM)")
    records: list[LenzeRecord] = []
    for _, row in frame.iterrows():
        record_id = str(row[id_col])
        label = str(row[condition_col]).strip().lower()
        records.append(LenzeRecord(
            path=source.parent / "Data" / f"{record_id}.mat",
            record_id=record_id,
            label=label,
            rpm=float(row[rpm_col]),
            belt_tension=_optional_float(row.get(columns.get("belt_tension (nm)"))),
            counter_momentum=_optional_float(row.get(columns.get("counter_momentum (nm)"))),
        ))
    return records


def _optional_float(value) -> float | None:
    try:
        number = float(value)
        return number if np.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def load_lenze_record(record: LenzeRecord) -> dict[str, Any]:
    raw = np.asarray(loadmat(record.path)["StromBox_Werte"])
    if raw.ndim != 2:
        raise ValueError("StromBox_Werte must be a matrix")
    if raw.shape[0] < 12 and raw.shape[1] >= 12:
        raw = raw.T
    if raw.shape[0] < 12:
        raise ValueError("StromBox_Werte must contain at least 12 channels")
    raw = raw[:12].astype(float, copy=False)
    time = raw[0]
    dt = float(np.median(np.diff(time)))
    sampling_rate = 1.0 / dt if np.isfinite(dt) and dt > 0 else LENZE_SAMPLE_RATE_HZ
    current = raw[9]
    speed = raw[10]
    speed -= np.mean(speed)
    return {
        "current": current,
        "speed": speed,
        "sampling_rate_hz": sampling_rate,
        "shaft_rate_hz": record.rpm / 60.0,
        "label": record.label,
        "record_id": record.record_id,
        "rpm": record.rpm,
        "operating_condition": {
            "belt_tension": record.belt_tension,
            "counter_momentum": record.counter_momentum,
        },
    }
