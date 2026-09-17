from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from tsdiag.benchmarks.bearing_cwru import (
    BearingRecordResult,
    BearingWindowResult,
    _aggregate_record,
    _summary,
)
from tsdiag.datasets.bearing_cwru import (
    CWRU_DE_FREQUENCY_MULTIPLIERS,
    CWRUBearingRecord,
    cwru_007_drive_end_manifest,
    evenly_spaced_windows,
    load_cwru_drive_end,
)


def test_cwru_manifest_is_balanced_across_four_loads():
    records = cwru_007_drive_end_manifest()
    assert len(records) == 16
    labels = {record.label for record in records}
    assert labels == {"normal", "inner_race", "rolling_element", "outer_race"}
    for load in range(4):
        per_load = [record for record in records if record.load_hp == load]
        assert len(per_load) == 4
        assert {record.label for record in per_load} == labels


def test_cwru_loader_uses_mat_rpm_and_documented_frequency_multipliers(tmp_path: Path):
    record = CWRUBearingRecord(999, "inner_race", "BPFI", 2, 1750.0, 0.007)
    path = tmp_path / record.filename
    signal = np.linspace(-1.0, 1.0, 24000)
    savemat(path, {"X999_DE_time": signal[:, None], "X999RPM": np.array([[1800.0]])})

    loaded = load_cwru_drive_end(path, record)
    assert loaded["sampling_rate_hz"] == 12000.0
    assert loaded["rpm"] == 1800.0
    assert loaded["shaft_rate_hz"] == 30.0
    assert np.allclose(loaded["signal"], signal)
    for code, multiplier in CWRU_DE_FREQUENCY_MULTIPLIERS.items():
        assert loaded["fault_frequencies"][code] == multiplier * 30.0


def test_evenly_spaced_windows_do_not_overlap_by_identity_or_change_length():
    x = np.arange(72000, dtype=float)
    windows = evenly_spaced_windows(x, 12000.0, window_seconds=1.0, max_windows=6)
    assert len(windows) == 6
    assert all(window.shape == (12000,) for window in windows)
    assert not np.shares_memory(windows[0], x)
    assert windows[0][0] == 0.0
    assert windows[-1][-1] == 71999.0


def _window(prediction: str, index: int) -> BearingWindowResult:
    return BearingWindowResult(
        file_id=1,
        load_hp=0,
        window_index=index,
        true_label="outer_race",
        predicted_label=prediction,
        abstained=prediction == "abstain",
        confidence=0.8,
        detection_score=0.7,
        top_hypothesis="BPFO",
        resonance_band_hz=[1000.0, 3000.0],
    )


def test_record_aggregation_requires_majority_coverage_and_no_tie():
    rows = [_window("outer_race", i) for i in range(4)] + [_window("abstain", 4), _window("abstain", 5)]
    prediction, abstained, margin, coverage = _aggregate_record(rows)
    assert prediction == "outer_race"
    assert abstained is False
    assert coverage == 4 / 6
    assert margin > 0.0

    tied = [_window("outer_race", 0), _window("outer_race", 1), _window("inner_race", 2), _window("inner_race", 3), _window("abstain", 4), _window("abstain", 5)]
    prediction, abstained, _, _ = _aggregate_record(tied)
    assert prediction == "abstain"
    assert abstained is True

    low_coverage = [_window("outer_race", 0), _window("outer_race", 1)] + [_window("abstain", i) for i in range(2, 6)]
    prediction, abstained, _, coverage = _aggregate_record(low_coverage)
    assert coverage < 0.5
    assert prediction == "abstain"
    assert abstained is True


def _record(file_id: int, truth: str, pred: str, load_hp: int = 0) -> BearingRecordResult:
    return BearingRecordResult(
        file_id=file_id,
        load_hp=load_hp,
        rpm=1800.0,
        true_label=truth,
        predicted_label=pred,
        abstained=pred == "abstain",
        confidence=0.8 if pred != "abstain" else 0.0,
        window_count=6,
        diagnose_fraction=1.0 if pred != "abstain" else 0.0,
        abnormal_fraction=1.0 if pred not in {"normal", "abstain"} else 0.0,
        vote_margin=1.0,
        runtime_seconds=0.1,
    )


def test_summary_masks_before_reducing_false_alarm_and_fault_detection_metrics():
    records = [
        _record(1, "normal", "normal"),
        _record(2, "normal", "outer_race"),
        _record(3, "normal", "abstain"),
        _record(4, "normal", "normal"),
        _record(5, "outer_race", "outer_race"),
        _record(6, "inner_race", "abstain"),
    ]
    summary = _summary(records)
    assert summary["normal_false_alarm_rate"] == 0.25
    assert summary["normal_abstention_rate"] == 0.25
    assert summary["fault_detection_rate"] == 0.5
