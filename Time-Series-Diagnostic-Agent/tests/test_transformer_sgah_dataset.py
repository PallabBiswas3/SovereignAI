from pathlib import Path

import numpy as np

from tsdiag.datasets.transformer_sgah import SGAH_EVENT_SAMPLES, load_sgah_events
from tsdiag.domains.domain_steps import transformer_representation, transformer_spectral


def _write_csv(path: Path, rows: np.ndarray) -> None:
    header = "Ua(V),Ub(V),Uc(V),Ia(A),Ib(A),Ic(A)"
    np.savetxt(path, rows, delimiter=",", header=header, comments="")


def test_sgah_loader_preserves_whole_event_boundaries(tmp_path):
    rows = np.arange(2 * SGAH_EVENT_SAMPLES * 6, dtype=float).reshape(2 * SGAH_EVENT_SAMPLES, 6)
    path = tmp_path / "4-data.csv"
    _write_csv(path, rows)

    events = load_sgah_events(path, 4)
    assert len(events) == 2
    assert events[0].signal_matrix.shape == (100, 6)
    assert events[1].signal_matrix.shape == (100, 6)
    np.testing.assert_array_equal(events[0].signal_matrix, rows[:100])
    np.testing.assert_array_equal(events[1].signal_matrix, rows[100:])
    assert events[0].event_id == 0
    assert events[1].event_id == 1


def test_sgah_loader_uses_only_complete_events(tmp_path):
    rows = np.arange(199 * 6, dtype=float).reshape(199, 6)
    path = tmp_path / "5-data.csv"
    _write_csv(path, rows)
    events = load_sgah_events(path, 5)
    assert len(events) == 1
    np.testing.assert_array_equal(events[0].signal_matrix, rows[:100])


def test_transformer_spectral_keeps_multiple_frames_for_100_samples():
    t = np.arange(100, dtype=float)
    waveform = np.sin(2 * np.pi * t / 20.0) + 0.25 * np.sin(2 * np.pi * t / 7.0)
    state = {"fused_waveform": waveform, "sampling_rate_hz": 1.0}
    state.update(transformer_spectral(state))
    state.update(transformer_representation(state))

    image = state["feature_image"]
    assert image.ndim == 2
    assert image.shape[1] >= 2
    assert np.any(image[:, 1:] > 0)
