from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Iterable
from urllib.request import urlretrieve
from zipfile import ZipFile

import numpy as np
from scipy.io import loadmat

NASA_BATTERY_URL = "https://phm-datasets.s3.amazonaws.com/NASA/5.+Battery+Data+Set.zip"
NASA_BATTERY_IDS = ("B0005", "B0006", "B0007", "B0018")
NASA_NOMINAL_CAPACITY_AH = 2.0
NASA_EOL_CAPACITY_AH = 1.4


@dataclass(frozen=True)
class NASABatteryDischargeCycle:
    battery_id: str
    cycle_index: int
    capacity_ah: float
    ambient_temperature_c: float | None
    voltage_v: np.ndarray
    current_a: np.ndarray
    temperature_c: np.ndarray
    time_s: np.ndarray

    @property
    def soh(self) -> float:
        return float(self.capacity_ah / NASA_NOMINAL_CAPACITY_AH)


def _field(obj, name: str):
    if hasattr(obj, name):
        return getattr(obj, name)
    if isinstance(obj, np.void) and obj.dtype.names and name in obj.dtype.names:
        return obj[name]
    if isinstance(obj, dict) and name in obj:
        return obj[name]
    raise KeyError(name)


def _scalar(value, default=None):
    try:
        arr = np.asarray(value, dtype=float).squeeze()
        if arr.size == 1 and np.isfinite(float(arr)):
            return float(arr)
    except (TypeError, ValueError):
        pass
    return default


def _vector(value):
    arr = np.asarray(value, dtype=float).squeeze()
    return np.atleast_1d(arr).astype(float).ravel()


def load_nasa_battery_discharge_cycles(path: str | Path, battery_id: str | None = None):
    path = Path(path)
    battery_id = battery_id or path.stem
    payload = loadmat(path, squeeze_me=True, struct_as_record=False)
    root = payload[battery_id]
    cycles = np.atleast_1d(_field(root, "cycle")).ravel()
    output = []
    for cycle in cycles:
        cycle_type = str(np.asarray(_field(cycle, "type")).squeeze()).lower()
        if "discharge" not in cycle_type:
            continue
        data = _field(cycle, "data")
        capacity = _scalar(_field(data, "Capacity"))
        if capacity is None or capacity <= 0:
            continue
        voltage = _vector(_field(data, "Voltage_measured"))
        current = _vector(_field(data, "Current_measured"))
        temperature = _vector(_field(data, "Temperature_measured"))
        time_s = _vector(_field(data, "Time"))
        n = min(len(voltage), len(current), len(temperature), len(time_s))
        if n < 2:
            continue
        finite = np.isfinite(voltage[:n]) & np.isfinite(current[:n]) & np.isfinite(temperature[:n]) & np.isfinite(time_s[:n])
        if int(np.sum(finite)) < 2:
            continue
        output.append(NASABatteryDischargeCycle(
            battery_id=battery_id,
            cycle_index=len(output) + 1,
            capacity_ah=float(capacity),
            ambient_temperature_c=_scalar(_field(cycle, "ambient_temperature"), None),
            voltage_v=voltage[:n][finite],
            current_a=current[:n][finite],
            temperature_c=temperature[:n][finite],
            time_s=time_s[:n][finite],
        ))
    if not output:
        raise ValueError(f"No usable discharge cycles found in {path}")
    return output


def first_eol_cycle(cycles: Iterable[NASABatteryDischargeCycle], *, eol_capacity_ah=NASA_EOL_CAPACITY_AH):
    for cycle in cycles:
        if cycle.capacity_ah <= eol_capacity_ah:
            return int(cycle.cycle_index)
    return None


def download_nasa_battery_subset(destination: str | Path, *, battery_ids=NASA_BATTERY_IDS, overwrite=False):
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    ids = tuple(str(x) for x in battery_ids)
    expected = [root / f"{battery_id}.mat" for battery_id in ids]
    if not overwrite and all(p.exists() and p.stat().st_size > 0 for p in expected):
        return expected
    outer_path = root / "_nasa_battery_dataset.zip"
    urlretrieve(NASA_BATTERY_URL, outer_path)
    try:
        with ZipFile(outer_path) as outer:
            matches = [n for n in outer.namelist() if n.endswith("1. BatteryAgingARC-FY08Q4.zip")]
            if len(matches) != 1:
                raise ValueError(f"Expected one FY08Q4 archive, found {matches}")
            inner_bytes = outer.read(matches[0])
        with ZipFile(BytesIO(inner_bytes)) as inner:
            names = inner.namelist()
            for battery_id, path in zip(ids, expected):
                matches = [n for n in names if n.endswith(f"{battery_id}.mat")]
                if len(matches) != 1:
                    raise ValueError(f"Expected one {battery_id}.mat member, found {matches}")
                path.write_bytes(inner.read(matches[0]))
    finally:
        if outer_path.exists():
            outer_path.unlink()
    return expected
