from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class TurbofanTrainingTrajectory:
    unit_id: int
    cycle_index: np.ndarray
    sensors: np.ndarray


def _feature_vector(signal_matrix, cycle_index, *, recent_window: int = 20) -> np.ndarray:
    x = np.asarray(signal_matrix, dtype=float)
    cycles = np.asarray(cycle_index, dtype=float).ravel()
    if x.ndim != 2 or len(x) != len(cycles) or len(x) < 2:
        raise ValueError("signal_matrix and cycle_index must describe at least two aligned cycles")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(cycles)):
        raise ValueError("turbofan RUL features require finite inputs")

    n = min(max(2, int(recent_window)), len(x))
    recent = x[-n:]
    recent_cycles = cycles[-n:]
    span = max(float(recent_cycles[-1] - recent_cycles[0]), 1.0)
    trend = (recent[-1] - recent[0]) / span
    mean = np.mean(recent, axis=0)
    std = np.std(recent, axis=0)
    current = x[-1]

    # Raw operating regime effects are partly encoded in the simultaneous
    # sensor state. HistGradientBoosting handles scale differences and nonlinear
    # interactions without using any test RUL target during fitting.
    return np.concatenate([
        current,
        mean,
        std,
        trend,
        np.asarray([float(cycles[-1]), float(len(cycles))], dtype=float),
    ])


def _sample_endpoints(length: int, *, minimum_history: int, stride: int) -> list[int]:
    if length < minimum_history:
        return []
    points = list(range(minimum_history, length + 1, max(1, int(stride))))
    if not points or points[-1] != length:
        points.append(length)
    return points


class TrainOnlyTurbofanRULModel:
    """Fixed train-only supervised RUL estimator for run-to-failure trajectories.

    Training targets are derived exclusively from each training engine's known
    terminal cycle. The model never receives published test-set RUL values.
    """

    def __init__(
        self,
        *,
        minimum_history: int = 20,
        recent_window: int = 20,
        training_stride: int = 5,
    ):
        self.minimum_history = int(minimum_history)
        self.recent_window = int(recent_window)
        self.training_stride = int(training_stride)
        self._model = HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=180,
            max_leaf_nodes=31,
            min_samples_leaf=20,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=0,
        )
        self._fitted = False
        self.maximum_training_rul_: float | None = None
        self.training_sample_count_: int = 0

    def fit(self, trajectories: Iterable[TurbofanTrainingTrajectory]):
        features: list[np.ndarray] = []
        targets: list[float] = []
        for trajectory in trajectories:
            sensors = np.asarray(trajectory.sensors, dtype=float)
            cycles = np.asarray(trajectory.cycle_index, dtype=float).ravel()
            if len(sensors) != len(cycles):
                raise ValueError(f"Unit {trajectory.unit_id}: sensors/cycles are misaligned")
            for endpoint in _sample_endpoints(
                len(cycles),
                minimum_history=self.minimum_history,
                stride=self.training_stride,
            ):
                features.append(
                    _feature_vector(
                        sensors[:endpoint],
                        cycles[:endpoint],
                        recent_window=self.recent_window,
                    )
                )
                targets.append(float(cycles[-1] - cycles[endpoint - 1]))

        if not features:
            raise ValueError("No training samples were produced for turbofan RUL model")
        X = np.stack(features)
        y = np.asarray(targets, dtype=float)
        self._model.fit(X, y)
        self._fitted = True
        self.maximum_training_rul_ = float(np.max(y))
        self.training_sample_count_ = int(len(y))
        return self

    def predict_rul(self, signal_matrix, cycle_index) -> float:
        if not self._fitted:
            raise RuntimeError("TrainOnlyTurbofanRULModel must be fitted before prediction")
        feature = _feature_vector(
            signal_matrix,
            cycle_index,
            recent_window=self.recent_window,
        )[None, :]
        prediction = float(self._model.predict(feature)[0])
        upper = float(self.maximum_training_rul_ or max(prediction, 0.0))
        return float(np.clip(prediction, 0.0, upper))

    def __call__(self, *, signal_matrix, cycle_index, health_index=None):
        return {"rul_cycles": self.predict_rul(signal_matrix, cycle_index)}
