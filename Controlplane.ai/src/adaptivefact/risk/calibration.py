from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

_EPS = 1e-6


def _as_1d(values) -> np.ndarray:
    return np.asarray(values, dtype=float).reshape(-1)


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = np.clip(_as_1d(probabilities), _EPS, 1.0 - _EPS)
    return np.log(p / (1.0 - p)).reshape(-1, 1)


@dataclass
class ProbabilityCalibrator:
    name: str
    estimator: object | None = None

    def predict(self, probabilities) -> np.ndarray:
        p = np.clip(_as_1d(probabilities), 0.0, 1.0)
        if self.name == "identity":
            return p
        if self.name == "platt":
            return np.asarray(self.estimator.predict_proba(_logit(p)))[:, 1]
        if self.name == "isotonic":
            return np.asarray(self.estimator.predict(p), dtype=float)
        raise ValueError(f"Unknown calibrator: {self.name}")

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ProbabilityCalibrator":
        return joblib.load(path)


def fit_platt(probabilities, y_true, *, random_state: int = 42) -> ProbabilityCalibrator:
    model = LogisticRegression(max_iter=2000, random_state=random_state)
    model.fit(_logit(_as_1d(probabilities)), np.asarray(y_true, dtype=int))
    return ProbabilityCalibrator(name="platt", estimator=model)


def fit_isotonic(probabilities, y_true) -> ProbabilityCalibrator:
    model = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    model.fit(_as_1d(probabilities), np.asarray(y_true, dtype=int))
    return ProbabilityCalibrator(name="isotonic", estimator=model)


def fit_candidate_calibrators(probabilities, y_true, *, random_state: int = 42) -> dict[str, ProbabilityCalibrator]:
    return {
        "identity": ProbabilityCalibrator(name="identity"),
        "platt": fit_platt(probabilities, y_true, random_state=random_state),
        "isotonic": fit_isotonic(probabilities, y_true),
    }
