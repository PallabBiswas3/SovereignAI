from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from adaptivefact.benchmark.runner import Component, ComponentResult
from adaptivefact.data.schema import ResponseRecord
from adaptivefact.risk.features import RiskFeatureExtractor


@dataclass
class RiskModelBundle:
    model_name: str
    estimator: object
    feature_names: list[str]

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        probabilities = self.estimator.predict_proba(X)
        return np.asarray(probabilities)[:, 1]

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "RiskModelBundle":
        return joblib.load(path)


def build_logistic_regression(*, random_state: int = 42, class_weight: str | None = "balanced") -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight=class_weight,
                    random_state=random_state,
                ),
            ),
        ]
    )


def build_xgboost(*, random_state: int = 42, scale_pos_weight: float = 1.0):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise ImportError("XGBoost model requested but xgboost is not installed.") from exc

    return XGBClassifier(
        n_estimators=350,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_weight=2,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        random_state=random_state,
        scale_pos_weight=scale_pos_weight,
    )


class RiskEstimatorComponent(Component):
    label = "Risk Estimator V1"

    def __init__(
        self,
        bundle: RiskModelBundle,
        feature_extractor: RiskFeatureExtractor,
        *,
        threshold: float = 0.5,
    ) -> None:
        if bundle.feature_names != feature_extractor.feature_names:
            raise ValueError("Feature names in saved model do not match the active feature extractor")
        self.bundle = bundle
        self.feature_extractor = feature_extractor
        self.threshold = threshold

    def run(self, record: ResponseRecord) -> ComponentResult:
        X = self.feature_extractor.transform([record])
        risk = float(self.bundle.predict_proba(X)[0])
        return ComponentResult(
            prediction=int(risk >= self.threshold),
            confidence=risk,
        )
