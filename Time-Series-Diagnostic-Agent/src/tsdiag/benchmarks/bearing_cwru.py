from __future__ import annotations

import csv
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter
from typing import Callable

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from ..datasets.bearing_cwru import (
    CWRUBearingRecord,
    cwru_007_drive_end_manifest,
    evenly_spaced_windows,
    load_cwru_drive_end,
)
from ..domains.bearing_runner import BearingDiagnosticPipeline


LABELS = ("normal", "inner_race", "rolling_element", "outer_race")


@dataclass
class BearingWindowResult:
    file_id: int
    load_hp: int
    window_index: int
    true_label: str
    predicted_label: str
    abstained: bool
    confidence: float
    detection_score: float | None
    top_hypothesis: str | None
    resonance_band_hz: list[float] | None


@dataclass
class BearingRecordResult:
    file_id: int
    load_hp: int
    rpm: float
    true_label: str
    predicted_label: str
    abstained: bool
    confidence: float
    window_count: int
    diagnose_fraction: float
    abnormal_fraction: float
    vote_margin: float
    runtime_seconds: float


def _prediction_from_result(result) -> str:
    if result.abstained or result.decision == "abstain":
        return "abstain"
    if result.detection.abnormal is False:
        return "normal"
    if result.localization.components:
        label = str(result.localization.components[0])
        if label in LABELS:
            return label
    return "abstain"


def _aggregate_record(windows: list[BearingWindowResult]) -> tuple[str, bool, float, float]:
    non_abstain = [row for row in windows if row.predicted_label != "abstain"]
    if not non_abstain:
        return "abstain", True, 0.0, 0.0
    counts = Counter(row.predicted_label for row in non_abstain)
    ranked = counts.most_common()
    top_label, top_count = ranked[0]
    second_count = ranked[1][1] if len(ranked) > 1 else 0
    vote_margin = float((top_count - second_count) / max(len(windows), 1))
    coverage = float(len(non_abstain) / max(len(windows), 1))
    if coverage < 0.5 or top_count == second_count:
        return "abstain", True, vote_margin, coverage
    return str(top_label), False, vote_margin, coverage


def _summary(records: list[BearingRecordResult]) -> dict:
    if not records:
        return {"record_count": 0}
    truth = np.asarray([row.true_label for row in records], dtype=object)
    pred = np.asarray([row.predicted_label for row in records], dtype=object)
    abstained = pred == "abstain"
    covered = ~abstained
    normal = truth == "normal"
    faulty = ~normal
    all_record_accuracy = float(np.mean(truth == pred))
    all_record_macro_f1 = float(f1_score(truth, pred, labels=list(LABELS), average="macro", zero_division=0))
    covered_accuracy = float(accuracy_score(truth[covered], pred[covered])) if np.any(covered) else None
    covered_macro_f1 = float(f1_score(truth[covered], pred[covered], labels=list(LABELS), average="macro", zero_division=0)) if np.any(covered) else None
    false_alarm_flags = (pred != "normal") & ~abstained
    detected_fault_flags = (pred != "normal") & ~abstained
    false_alarm_rate = float(np.mean(false_alarm_flags[normal])) if np.any(normal) else None
    normal_abstention_rate = float(np.mean(abstained[normal])) if np.any(normal) else None
    fault_detection_rate = float(np.mean(detected_fault_flags[faulty])) if np.any(faulty) else None
    by_load: dict[str, dict] = {}
    for load in sorted({row.load_hp for row in records}):
        mask = np.asarray([row.load_hp == load for row in records], dtype=bool)
        load_truth = truth[mask]
        load_pred = pred[mask]
        load_covered = load_pred != "abstain"
        by_load[str(load)] = {
            "record_count": int(np.sum(mask)),
            "accuracy_including_abstention": float(np.mean(load_truth == load_pred)),
            "coverage": float(np.mean(load_covered)),
            "covered_accuracy": float(np.mean(load_truth[load_covered] == load_pred[load_covered])) if np.any(load_covered) else None,
        }
    confusion = confusion_matrix(truth, pred, labels=list(LABELS) + ["abstain"])
    return {
        "record_count": len(records),
        "all_record_accuracy": all_record_accuracy,
        "all_record_macro_f1": all_record_macro_f1,
        "coverage": float(np.mean(covered)),
        "abstention_rate": float(np.mean(abstained)),
        "covered_accuracy": covered_accuracy,
        "covered_macro_f1": covered_macro_f1,
        "normal_false_alarm_rate": false_alarm_rate,
        "normal_abstention_rate": normal_abstention_rate,
        "fault_detection_rate": fault_detection_rate,
        "by_load": by_load,
        "confusion_labels": list(LABELS) + ["abstain"],
        "confusion_matrix": confusion.tolist(),
        "mean_runtime_seconds": float(np.mean([row.runtime_seconds for row in records])),
    }


def run_cwru_benchmark(
    data_dir: str | Path,
    *,
    records: list[CWRUBearingRecord] | None = None,
    window_seconds: float = 1.0,
    max_windows: int = 6,
    minimum_confidence: float = 0.45,
    minimum_harmonics: int = 2,
    output_dir: str | Path | None = None,
    pipeline_factory: Callable[..., object] | None = None,
) -> dict:
    root = Path(data_dir)
    manifest = list(records or cwru_007_drive_end_manifest())
    window_rows: list[BearingWindowResult] = []
    record_rows: list[BearingRecordResult] = []
    failures: list[dict] = []
    factory = pipeline_factory or BearingDiagnosticPipeline

    for record in manifest:
        try:
            loaded = load_cwru_drive_end(root / record.filename, record)
            windows = evenly_spaced_windows(loaded["signal"], loaded["sampling_rate_hz"], window_seconds=window_seconds, max_windows=max_windows)
            started = perf_counter()
            local_rows: list[BearingWindowResult] = []
            for window_index, window in enumerate(windows):
                result = factory(minimum_confidence=minimum_confidence, minimum_harmonics=minimum_harmonics).run(
                    window,
                    loaded["sampling_rate_hz"],
                    fault_frequencies=loaded["fault_frequencies"],
                    shaft_rate_hz=loaded["shaft_rate_hz"],
                    channel_name="drive_end",
                    operating_condition={"load_hp": loaded["load_hp"], "rpm": loaded["rpm"], "dataset": "CWRU"},
                )
                prediction = _prediction_from_result(result)
                band = result.metadata.get("resonance_band_hz")
                row = BearingWindowResult(
                    file_id=record.file_id, load_hp=record.load_hp, window_index=window_index,
                    true_label=record.label, predicted_label=prediction, abstained=result.abstained,
                    confidence=float(result.confidence), detection_score=None if result.detection.score is None else float(result.detection.score),
                    top_hypothesis=result.hypotheses[0].label if result.hypotheses else None,
                    resonance_band_hz=None if band is None else [float(band[0]), float(band[1])],
                )
                local_rows.append(row); window_rows.append(row)
            runtime = perf_counter() - started
            prediction, abstained, vote_margin, coverage = _aggregate_record(local_rows)
            abnormal_fraction = float(np.mean([(row.predicted_label not in {"normal", "abstain"}) for row in local_rows]))
            confidence_values = [row.confidence for row in local_rows if row.predicted_label != "abstain"]
            record_rows.append(BearingRecordResult(
                file_id=record.file_id, load_hp=record.load_hp, rpm=float(loaded["rpm"]), true_label=record.label,
                predicted_label=prediction, abstained=abstained,
                confidence=float(np.mean(confidence_values)) if confidence_values else 0.0,
                window_count=len(local_rows), diagnose_fraction=coverage, abnormal_fraction=abnormal_fraction,
                vote_margin=vote_margin, runtime_seconds=float(runtime),
            ))
        except Exception as exc:
            failures.append({"file_id": int(record.file_id), "error": f"{type(exc).__name__}: {exc}"})

    payload = {
        "source": "Case Western Reserve University Bearing Data Center",
        "protocol": {
            "bearing": "6205-2RS JEM SKF drive-end", "sampling_rate_hz": 12000, "fault_diameter_in": 0.007,
            "outer_race_position": "6:00", "loads_hp": [0, 1, 2, 3], "window_seconds": float(window_seconds),
            "max_windows_per_record": int(max_windows), "primary_unit": "record",
            "label_use": "evaluation only; fault frequencies derive from documented bearing geometry and RPM",
        },
        "configuration": {"minimum_confidence": float(minimum_confidence), "minimum_harmonics": int(minimum_harmonics)},
        "summary": _summary(record_rows), "records": [asdict(row) for row in record_rows],
        "windows": [asdict(row) for row in window_rows], "failures": failures,
    }
    if output_dir is not None:
        out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
        (out / "bearing_cwru_benchmark.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        if record_rows:
            with (out / "bearing_cwru_records.csv").open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(asdict(record_rows[0]).keys()))
                writer.writeheader(); writer.writerows(asdict(row) for row in record_rows)
    return payload
