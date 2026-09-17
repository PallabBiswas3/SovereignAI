from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable
from zipfile import ZipFile

import numpy as np
import pandas as pd

CARE_VERSION = 6
CARE_ZENODO_RECORD = "15846963"
CARE_DOI = "10.5281/zenodo.15846963"
CARE_ARCHIVE_NAME = "CARE_To_Compare.zip"
CARE_SAMPLE_PERIOD_MINUTES = 10.0
CARE_WIND_FARMS = ("A", "B", "C")
CARE_METADATA_COLUMNS = ("id", "train_test", "time_stamp", "asset_id", "status_type_id")


@dataclass(frozen=True)
class CareEvent:
    event_id: int
    wind_farm: str
    asset_id: str
    event_start: pd.Timestamp | None
    event_end: pd.Timestamp | None
    description: str | None
    is_anomaly: bool | None


@dataclass
class CareEventData:
    event: CareEvent
    train: pd.DataFrame
    prediction: pd.DataFrame
    feature_description: pd.DataFrame


def _wind_farm_dir(root: str | Path, wind_farm: str) -> Path:
    wf = str(wind_farm).upper()
    if wf not in CARE_WIND_FARMS:
        raise ValueError(f"wind_farm must be one of {CARE_WIND_FARMS}")
    return Path(root) / f"Wind Farm {wf}"


def _is_archive(root: str | Path) -> bool:
    path = Path(root)
    return path.is_file() and path.suffix.lower() == ".zip"


@lru_cache(maxsize=4)
def _archive_members(path: str) -> tuple[str, ...]:
    with ZipFile(path) as archive:
        return tuple(archive.namelist())


def _archive_member(root: str | Path, relative: str) -> str:
    suffix = relative.replace("\\", "/").lstrip("/")
    matches = [name for name in _archive_members(str(Path(root))) if name.rstrip("/").endswith(suffix)]
    if not matches:
        raise FileNotFoundError(f"CARE archive does not contain {relative}")
    if len(matches) > 1:
        matches.sort(key=len)
    return matches[0]


def _read_csv(root: str | Path, relative: str, **kwargs) -> pd.DataFrame:
    if _is_archive(root):
        member = _archive_member(root, relative)
        with ZipFile(root) as archive, archive.open(member) as handle:
            return pd.read_csv(handle, **kwargs)
    return pd.read_csv(Path(root) / relative, **kwargs)


def validate_care_layout(root: str | Path) -> dict[str, bool]:
    """Validate an extracted CARE v6 directory or the official ZIP archive."""
    root = Path(root)
    status: dict[str, bool] = {}
    if _is_archive(root):
        members = _archive_members(str(root))
        for wf in CARE_WIND_FARMS:
            prefix = f"Wind Farm {wf}/"
            status[wf] = bool(
                any(name.endswith(prefix + "event_info.csv") for name in members)
                and any(name.endswith(prefix + "feature_description.csv") for name in members)
                and any((prefix + "datasets/") in name for name in members)
            )
        return status

    for wf in CARE_WIND_FARMS:
        folder = _wind_farm_dir(root, wf)
        status[wf] = bool(
            folder.exists()
            and (folder / "event_info.csv").exists()
            and (folder / "feature_description.csv").exists()
            and (folder / "datasets").exists()
        )
    return status


def load_care_event_info(root: str | Path, wind_farm: str | None = None) -> pd.DataFrame:
    farms = CARE_WIND_FARMS if wind_farm is None else (str(wind_farm).upper(),)
    frames = []
    for wf in farms:
        frame = _read_csv(root, f"Wind Farm {wf}/event_info.csv", sep=";")
        frame["wind_farm"] = wf
        for col in ("event_start", "event_end"):
            if col in frame.columns:
                frame[col] = pd.to_datetime(frame[col], errors="coerce")
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def load_care_feature_description(root: str | Path, wind_farm: str) -> pd.DataFrame:
    wf = str(wind_farm).upper()
    return _read_csv(root, f"Wind Farm {wf}/feature_description.csv", sep=";")


def load_care_event_asset_ids(root: str | Path, event_id: int) -> tuple[str, ...]:
    """Return every physical asset ID present in one CARE event.

    Event-info metadata is incomplete for some CARE farms, so leakage checks
    must inspect the row-level ``asset_id`` column rather than treating a
    missing event-info value as permission to fall back to event-only splits.
    """

    info = load_care_event_info(root)
    matches = info[info["event_id"].astype(int) == int(event_id)]
    if matches.empty:
        raise KeyError(f"CARE event_id={event_id} not found")
    row = matches.iloc[0]
    wind_farm = str(row["wind_farm"])
    relative = f"Wind Farm {wind_farm}/datasets/{int(event_id)}.csv"
    header = _read_csv(root, relative, sep=";", nrows=0)
    values: list[object] = []
    if "asset_id" in header.columns:
        values.extend(_read_csv(root, relative, sep=";", usecols=["asset_id"])["asset_id"])
    if "asset_id" in row and pd.notna(row["asset_id"]):
        values.append(row["asset_id"])

    normalized = []
    for value in values:
        if pd.isna(value):
            continue
        text = str(value).strip()
        if not text:
            continue
        if text.endswith(".0"):
            try:
                text = str(int(float(text)))
            except ValueError:
                pass
        if text not in normalized:
            normalized.append(text)
    return tuple(sorted(normalized))


def _map_statistics(statistics: Iterable[str] | None) -> tuple[str, ...]:
    if statistics is None:
        return ("avg",)
    mapping = {
        "average": "avg", "avg": "avg",
        "minimum": "min", "min": "min",
        "maximum": "max", "max": "max",
        "std_dev": "std", "standard_deviation": "std", "std": "std",
    }
    out = []
    for value in statistics:
        key = str(value).strip().lower()
        if key not in mapping:
            raise ValueError(f"unsupported CARE statistic: {value}")
        mapped = mapping[key]
        if mapped not in out:
            out.append(mapped)
    return tuple(out)


def _feature_columns(
    dataset_columns: Iterable[str],
    feature_description: pd.DataFrame,
    statistics: Iterable[str] | None,
) -> list[str]:
    existing = set(str(x) for x in dataset_columns)
    stats = _map_statistics(statistics)
    selected = [name for name in CARE_METADATA_COLUMNS if name in existing]

    if "sensor_name" not in feature_description.columns:
        return selected + [
            name for name in existing
            if name not in CARE_METADATA_COLUMNS
            and ("avg" in stats or name.endswith(tuple(f"_{s}" for s in stats)))
        ]

    for _, row in feature_description.iterrows():
        sensor = str(row["sensor_name"])
        declared = str(row.get("statistics_type", "average")).split(",")
        declared_stats = set(_map_statistics(x.strip() for x in declared if str(x).strip()))
        for stat in stats:
            if stat not in declared_stats:
                continue
            if stat == "avg" and sensor in existing and sensor not in selected:
                selected.append(sensor)
            candidate = f"{sensor}_{stat}"
            if candidate in existing and candidate not in selected:
                selected.append(candidate)
    return selected


def _event_from_row(row: pd.Series) -> CareEvent:
    def _optional_timestamp(name: str):
        if name not in row or pd.isna(row[name]):
            return None
        return pd.Timestamp(row[name])

    description = None
    for name in ("event_description", "description", "fault_description"):
        if name in row and pd.notna(row[name]):
            description = str(row[name])
            break

    is_anomaly = None
    for name in ("event_label", "is_anomaly", "anomaly", "label"):
        if name in row and pd.notna(row[name]):
            value = row[name]
            if isinstance(value, str):
                text = value.strip().lower()
                if text in {"true", "1", "anomaly", "fault", "faulty"}:
                    is_anomaly = True
                elif text in {"false", "0", "normal", "healthy"}:
                    is_anomaly = False
            else:
                is_anomaly = bool(value)
            break

    return CareEvent(
        event_id=int(row["event_id"]),
        wind_farm=str(row["wind_farm"]),
        asset_id=str(row.get("asset_id", "")),
        event_start=_optional_timestamp("event_start"),
        event_end=_optional_timestamp("event_end"),
        description=description,
        is_anomaly=is_anomaly,
    )


def load_care_event(
    root: str | Path,
    event_id: int,
    *,
    statistics: Iterable[str] | None = ("avg",),
) -> CareEventData:
    """Load one CARE event from an extracted directory or directly from the ZIP."""
    info = load_care_event_info(root)
    matches = info[info["event_id"].astype(int) == int(event_id)]
    if matches.empty:
        raise KeyError(f"CARE event_id={event_id} not found")
    row = matches.iloc[0]
    event = _event_from_row(row)
    feature_description = load_care_feature_description(root, event.wind_farm)
    relative = f"Wind Farm {event.wind_farm}/datasets/{int(event_id)}.csv"

    header = _read_csv(root, relative, sep=";", nrows=0)
    usecols = _feature_columns(header.columns, feature_description, statistics)
    frame = _read_csv(root, relative, sep=";", usecols=usecols)
    if "time_stamp" in frame.columns:
        frame["time_stamp"] = pd.to_datetime(frame["time_stamp"], errors="coerce")

    numeric = [col for col in frame.columns if col not in CARE_METADATA_COLUMNS and col != "train_test"]
    for col in numeric:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")

    if "train_test" not in frame.columns:
        raise ValueError(f"CARE event {event_id} is missing train_test split labels")
    train = frame[frame["train_test"].astype(str).str.lower() == "train"].drop(columns=["train_test"])
    prediction = frame[frame["train_test"].astype(str).str.lower() == "prediction"].drop(columns=["train_test"])
    return CareEventData(
        event=event,
        train=train.reset_index(drop=True),
        prediction=prediction.reset_index(drop=True),
        feature_description=feature_description,
    )


def care_sensor_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series | None, pd.Series | None]:
    timestamps = frame["time_stamp"] if "time_stamp" in frame.columns else None
    status = frame["status_type_id"] if "status_type_id" in frame.columns else None
    sensor = frame.drop(columns=list(CARE_METADATA_COLUMNS), errors="ignore")
    sensor = sensor.select_dtypes(include=[np.number]).copy()
    return sensor, status, timestamps


def care_normal_mask(frame: pd.DataFrame) -> np.ndarray:
    if "status_type_id" not in frame.columns:
        return np.ones(len(frame), dtype=bool)
    values = pd.to_numeric(frame["status_type_id"], errors="coerce")
    return values.fillna(-1).to_numpy() == 0
