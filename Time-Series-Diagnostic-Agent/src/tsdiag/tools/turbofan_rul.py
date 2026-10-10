from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor


@dataclass(frozen=True)
class TurbofanTrainingTrajectory:
    unit_id: int
    cycle_index: np.ndarray
    sensors: np.ndarray


@dataclass(frozen=True)
class TurbofanCalibrationCase:
    unit_id: int
    cycle_index: np.ndarray
    sensors: np.ndarray
    remaining_useful_life: float


@dataclass(frozen=True)
class RULResidualCalibration:
    """One fixed prefix per held-out engine; nominal marginal coverage only.

    Coverage relies on exchangeability and does not certify any individual engine
    or out-of-distribution regime. Censored targets cannot be used as exact RUL.
    """
    radius_cycles: float
    coverage: float
    unit_ids: frozenset[int]
    training_unit_ids: frozenset[int]
    model_id: str
    dataset_id: str

    def __post_init__(self):
        if not np.isfinite(self.radius_cycles) or self.radius_cycles < 0 or not 0 < self.coverage < 1:
            raise ValueError("invalid calibration radius or nominal coverage")
        if not self.unit_ids or not self.training_unit_ids or self.unit_ids & self.training_unit_ids:
            raise ValueError("calibration and training identities must be nonempty and disjoint")
        if not self.model_id or not self.dataset_id or math.ceil((len(self.unit_ids)+1)*self.coverage) > len(self.unit_ids):
            raise ValueError("calibration provenance or finite-sample size is invalid")

    @classmethod
    def fit(cls, predictions, targets, unit_ids, *, training_unit_ids, model_id, dataset_id, coverage=.9):
        p, y = np.asarray(predictions, float), np.asarray(targets, float)
        units, training = list(unit_ids), frozenset(training_unit_ids)
        if p.ndim != 1 or p.shape != y.shape or len(p) != len(units) or not len(p):
            raise ValueError("calibration requires aligned predictions, exact RUL targets and engine identities")
        if not np.all(np.isfinite(p)) or not np.all(np.isfinite(y)) or np.any(p < 0) or np.any(y < 0):
            raise ValueError("calibration targets/predictions must be finite nonnegative exact RUL")
        if any(unit is None for unit in units) or len(set(units)) != len(units) or set(units) & training:
            raise ValueError("use one prefix per calibration engine, disjoint from training engines")
        if not training or not model_id or not dataset_id or not 0 < coverage < 1:
            raise ValueError("calibration requires training identities, model/dataset provenance and valid coverage")
        rank = math.ceil((len(y) + 1) * coverage)
        if rank > len(y):
            raise ValueError("too few independent calibration engines for a finite interval at this coverage")
        radius = float(np.sort(np.abs(p-y))[rank-1])
        return cls(radius, float(coverage), frozenset(units), training, str(model_id), str(dataset_id))

    def interval(self, prediction, *, unit_id, model_id):
        if unit_id is None or unit_id in self.unit_ids or unit_id in self.training_unit_ids:
            raise ValueError("prediction engine must be identified and disjoint from training/calibration")
        if model_id != self.model_id or not np.isfinite(prediction) or prediction < 0:
            raise ValueError("calibration model identity or prediction is invalid")
        return [max(0., float(prediction)-self.radius_cycles), float(prediction)+self.radius_cycles]


def _feature_vector(signal_matrix, cycle_index, *, recent_window: int = 20) -> np.ndarray:
    x = np.asarray(signal_matrix, dtype=float)
    cycles = np.asarray(cycle_index, dtype=float).ravel()
    if x.ndim != 2 or len(x) != len(cycles) or len(x) < 2:
        raise ValueError("signal_matrix and cycle_index must describe at least two aligned cycles")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(cycles)):
        raise ValueError("turbofan RUL features require finite inputs")
    if np.any(cycles < 0) or np.any(np.diff(cycles) <= 0):
        raise ValueError("turbofan cycle_index must be nonnegative and strictly increasing")

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
        self.training_unit_ids_: frozenset[int] = frozenset()
        self.model_id_: str | None = None
        self.interval_calibration: RULResidualCalibration | None = None

    def fit(self, trajectories: Iterable[TurbofanTrainingTrajectory]):
        features: list[np.ndarray] = []
        targets: list[float] = []
        units = set()
        for trajectory in trajectories:
            if trajectory.unit_id in units:
                raise ValueError("duplicate training engine identity")
            units.add(trajectory.unit_id)
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
        self.training_unit_ids_ = frozenset(units)
        self.model_id_ = hashlib.sha256(X.tobytes()+y.tobytes()+repr(sorted(units)).encode()).hexdigest()
        self.interval_calibration = None  # Refitting invalidates prior intervals.
        return self

    def calibrate(self, cases: Iterable[TurbofanCalibrationCase], *, dataset_id, coverage=.9):
        if not self._fitted:
            raise RuntimeError("fit the point model before held-out calibration")
        records = list(cases)
        predictions = [self.predict_rul(r.sensors, r.cycle_index) for r in records]
        self.interval_calibration = RULResidualCalibration.fit(
            predictions, [r.remaining_useful_life for r in records], [r.unit_id for r in records],
            training_unit_ids=self.training_unit_ids_, model_id=self.model_id_, dataset_id=dataset_id, coverage=coverage)
        return self

    def predict_with_interval(self, signal_matrix, cycle_index, *, unit_id):
        prediction = self.predict_rul(signal_matrix, cycle_index)
        if self.interval_calibration is None:
            raise ValueError("rul_calibration_required")
        interval = self.interval_calibration.interval(prediction, unit_id=unit_id, model_id=self.model_id_)
        return {"rul_cycles": prediction, "interval": interval, "calibrated": True,
                "coverage": self.interval_calibration.coverage,
                "calibration_dataset": self.interval_calibration.dataset_id,
                "calibration_method": "engine_disjoint_split_conformal_v1"}

    def predict_rul(self, signal_matrix, cycle_index) -> float:
        if not self._fitted:
            raise RuntimeError("TrainOnlyTurbofanRULModel must be fitted before prediction")
        if len(cycle_index) < self.minimum_history:
            raise ValueError("insufficient_turbofan_history")
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
