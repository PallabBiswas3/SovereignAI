from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlopen

import numpy as np

SGAH_COMMIT = "bbe1020e3fade83f7861657bb3eaea41c25ec0c9"
SGAH_BASE_URL = f"https://raw.githubusercontent.com/smartlab-hfut/SGAH-datasets/{SGAH_COMMIT}/data"
SGAH_EVENT_SAMPLES = 100
# The CSV headers are Ua/Ub/Uc/Ia/Ib/Ic. The upstream README sentence that
# describes current channels first is inconsistent with the files themselves,
# so the pinned CSV header is treated as authoritative.
SGAH_CHANNEL_NAMES = ("Ua", "Ub", "Uc", "Ia", "Ib", "Ic")
SGAH_CLASSES = {
    1: "single_phase_ground_fault",
    2: "inter_phase_short_circuit_fault",
    3: "two_phase_ground_fault",
    4: "main_transformer_fault",
    5: "normal",
}


@dataclass(frozen=True)
class SgahEvent:
    class_id: int
    label: str
    event_id: int
    signal_matrix: np.ndarray


def _load_csv(path: str | Path) -> np.ndarray:
    matrix = np.genfromtxt(path, delimiter=",", skip_header=1, dtype=float)
    if matrix.ndim == 1:
        matrix = matrix[None, :]
    if matrix.ndim != 2 or matrix.shape[1] != len(SGAH_CHANNEL_NAMES):
        raise ValueError(
            f"SGAH CSV must contain {len(SGAH_CHANNEL_NAMES)} waveform columns; got {matrix.shape}"
        )
    if not np.all(np.isfinite(matrix)):
        raise ValueError("SGAH CSV contains non-finite values")
    if len(matrix) < SGAH_EVENT_SAMPLES:
        raise ValueError("SGAH CSV contains no complete 100-sample event")
    return matrix


def load_sgah_events(path: str | Path, class_id: int) -> list[SgahEvent]:
    class_id = int(class_id)
    if class_id not in SGAH_CLASSES:
        raise ValueError(f"Unsupported SGAH class id: {class_id}")
    matrix = _load_csv(path)
    # The pinned source contains at least one trailing partial event. The README
    # defines an event as exactly 100 samples, so only complete events are used;
    # missing samples are never padded or synthesized.
    event_count = len(matrix) // SGAH_EVENT_SAMPLES
    usable = matrix[: event_count * SGAH_EVENT_SAMPLES]
    return [
        SgahEvent(
            class_id=class_id,
            label=SGAH_CLASSES[class_id],
            event_id=i,
            signal_matrix=usable[i * SGAH_EVENT_SAMPLES : (i + 1) * SGAH_EVENT_SAMPLES],
        )
        for i in range(event_count)
    ]


def validate_sgah_root(root: str | Path) -> list[dict]:
    root = Path(root)
    rows: list[dict] = []
    for class_id, label in SGAH_CLASSES.items():
        path = root / f"{class_id}-data.csv"
        matrix = _load_csv(path)
        events = load_sgah_events(path, class_id)
        rows.append(
            {
                "class_id": class_id,
                "label": label,
                "total_rows": int(len(matrix)),
                "event_count": len(events),
                "discarded_trailing_rows": int(len(matrix) % SGAH_EVENT_SAMPLES),
                "samples_per_event": SGAH_EVENT_SAMPLES,
                "channels": list(SGAH_CHANNEL_NAMES),
            }
        )
    return rows


def download_sgah(destination: str | Path, *, overwrite: bool = False) -> list[Path]:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)
    outputs = [root / f"{class_id}-data.csv" for class_id in SGAH_CLASSES]
    for class_id, output in zip(SGAH_CLASSES, outputs):
        if output.exists() and output.stat().st_size > 0 and not overwrite:
            continue
        url = f"{SGAH_BASE_URL}/{class_id}-data.csv"
        with urlopen(url, timeout=120) as response:
            payload = response.read()
        if not payload:
            raise ValueError(f"Empty SGAH download for class {class_id}")
        output.write_bytes(payload)
    validate_sgah_root(root)
    return outputs
