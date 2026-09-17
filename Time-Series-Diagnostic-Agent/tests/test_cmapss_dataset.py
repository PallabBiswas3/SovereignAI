import numpy as np

from tsdiag.datasets.cmapss import (
    CMAPSS_SENSOR_NAMES,
    load_cmapss_rul,
    load_cmapss_trajectories,
)


def _row(unit, cycle, offset=0.0):
    settings = [0.1 + offset, 0.2, 0.3]
    sensors = [float(i) + offset + cycle * 0.01 for i in range(1, 22)]
    return [float(unit), float(cycle), *settings, *sensors]


def test_cmapss_loader_preserves_unit_cycle_settings_and_21_sensors(tmp_path):
    path = tmp_path / "test_FD001.txt"
    matrix = np.asarray([
        _row(1, 1),
        _row(1, 2),
        _row(2, 1, 1.0),
        _row(2, 2, 1.0),
        _row(2, 3, 1.0),
    ])
    np.savetxt(path, matrix)

    trajectories = load_cmapss_trajectories(path, "FD001")

    assert len(trajectories) == 2
    assert trajectories[0].unit_id == 1
    assert trajectories[0].cycle_index.tolist() == [1.0, 2.0]
    assert trajectories[0].operating_settings.shape == (2, 3)
    assert trajectories[0].sensors.shape == (2, len(CMAPSS_SENSOR_NAMES))
    assert trajectories[1].cycle_index.tolist() == [1.0, 2.0, 3.0]


def test_cmapss_rul_loader_reads_nonnegative_targets(tmp_path):
    path = tmp_path / "RUL_FD001.txt"
    np.savetxt(path, np.asarray([12.0, 34.0]))
    assert load_cmapss_rul(path).tolist() == [12.0, 34.0]
