from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import savemat

from tsdiag.datasets.bearing_lenze import LenzeRecord, load_lenze_record
from tsdiag.datasets.bearing_paderborn import (
    PaderbornRecord,
    parse_paderborn_filename,
    paderborn_specimen_split,
    load_paderborn_record,
)
from tsdiag.tools.bearing_hybrid import (
    BearingHybridClassifier,
    BearingHybridConfig,
    bearing_fault_frequencies,
    normalize_waveform_channels,
    speed_assisted_physics_features,
)


def test_paderborn_filename_and_specimen_split_are_leakage_safe(tmp_path: Path):
    records = []
    for bearing in ("K001", "K002", "KI01", "KI03", "KA01", "KA03"):
        for run in (1, 2):
            records.append(parse_paderborn_filename(tmp_path / f"N15_M07_F10_{bearing}_{run}.mat"))
    train, test = paderborn_specimen_split(records, test_fraction=0.5)
    assert {row.bearing_id for row in train}.isdisjoint({row.bearing_id for row in test})
    assert {row.label for row in train} == {"healthy", "inner_race", "outer_race"}
    assert {row.label for row in test} == {"healthy", "inner_race", "outer_race"}


def test_speed_assisted_features_have_stable_cross_dataset_shape():
    fs = 64000.0
    t = np.arange(8192) / fs
    shaft = 25.0
    faults = bearing_fault_frequencies(shaft)
    vibration = np.sin(2 * np.pi * faults["BPFO"] * t) + 0.2 * np.sin(2 * np.pi * 2000 * t)
    current = np.sin(2 * np.pi * 50 * t)
    features = speed_assisted_physics_features(np.vstack([vibration, current]), fs, shaft)
    drive_features = speed_assisted_physics_features(np.vstack([current, vibration]), fs, shaft, envelope_channel=None)
    assert features.shape == drive_features.shape
    assert features.shape == (69,)
    assert np.all(np.isfinite(features))
    normalized = normalize_waveform_channels(np.vstack([vibration, current]), 1024)
    assert normalized.shape == (2, 1024)
    assert np.allclose(normalized.mean(axis=1), 0.0, atol=1e-5)


def test_paderborn_loader_reads_reviewed_named_channels(tmp_path: Path):
    path = tmp_path / "N15_M07_F10_KI01_1.mat"
    n = 256
    savemat(path, {
        "vibration_1": np.arange(n, dtype=float),
        "phase_current_1": np.ones(n),
        "phase_current_2": np.ones(n) * 2,
        "speed": np.ones(n) * 1500,
    })
    record = PaderbornRecord(path, "KI01", "inner_race", 1500.0, 0.7, 1000.0, 1)
    loaded = load_paderborn_record(record)
    assert loaded["label"] == "inner_race"
    assert loaded["shaft_rate_hz"] == 25.0
    assert loaded["vibration"].shape == (n,)
    assert loaded["current"].shape == (n,)


def test_lenze_loader_uses_documented_strombox_layout(tmp_path: Path):
    n = 320
    raw = np.zeros((12, n))
    raw[0] = np.arange(n) / 16000.0
    raw[9] = np.sin(np.arange(n) / 10)
    raw[10] = 3.0 + np.sin(np.arange(n) / 20)
    path = tmp_path / "H1.1.mat"
    savemat(path, {"StromBox_Werte": raw})
    loaded = load_lenze_record(LenzeRecord(path, "H1.1", "normal", 900.0))
    assert np.isclose(loaded["sampling_rate_hz"], 16000.0)
    assert loaded["shaft_rate_hz"] == 15.0
    assert abs(float(np.mean(loaded["speed"]))) < 1e-12


def test_hybrid_classifier_fits_predicts_and_persists(tmp_path: Path):
    rng = np.random.default_rng(9)
    labels = np.asarray(["healthy"] * 8 + ["outer_race"] * 8)
    waveforms = rng.normal(size=(16, 2, 256)).astype(np.float32)
    waveforms[8:, 0] += np.sin(np.linspace(0, 30, 256))[None, :]
    physics = rng.normal(size=(16, 69)).astype(np.float32)
    physics[8:, 4] += 4.0
    model = BearingHybridClassifier(BearingHybridConfig(mode="fusion", waveform_samples=256, epochs=1, batch_size=8))
    progress = []
    model.fit(waveforms, physics, labels, progress_callback=progress.append)
    assert len(progress) == 1
    assert progress[0]["epoch"] == 1
    assert progress[0]["epochs"] == 1
    assert progress[0]["loss"] >= 0.0
    assert progress[0]["eta_seconds"] == 0.0
    before = model.predict_proba(waveforms, physics)
    path = model.save(tmp_path / "bearing.pt")
    after = BearingHybridClassifier.load(path).predict_proba(waveforms, physics)
    assert before.shape == (16, 2)
    assert np.allclose(before, after)
