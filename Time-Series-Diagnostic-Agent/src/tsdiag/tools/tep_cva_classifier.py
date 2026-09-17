from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import joblib
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .monitoring import fit_cva, transform_cva


@dataclass
class CVAFaultClassifier:
    """Supervised fault classifier operating on causal CVA state features."""

    cva_state: dict
    reference_mean: np.ndarray
    reference_scale: np.ndarray
    feature_scaler: StandardScaler
    estimator: object
    class_ids: tuple[int, ...]
    method: str

    def save(self, path: str | Path) -> Path:
        """Persist the fitted preprocessing, CVA state, and estimator together."""
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"format": "tsdiag-cva-classifier", "version": 1, "classifier": self}, output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "CVAFaultClassifier":
        """Load a trusted classifier artifact created by :meth:`save`."""
        payload = joblib.load(Path(path))
        if not isinstance(payload, dict) or payload.get("format") != "tsdiag-cva-classifier":
            raise ValueError("not a tsdiag CVA classifier artifact")
        if payload.get("version") != 1 or not isinstance(payload.get("classifier"), cls):
            raise ValueError("unsupported CVA classifier artifact version")
        return payload["classifier"]

    def transform(self, signal_matrix) -> dict:
        x = np.asarray(signal_matrix, dtype=float)
        standardized = (x - self.reference_mean) / self.reference_scale
        transformed = transform_cva(standardized, self.cva_state)
        transformed["classifier_features"] = self.feature_scaler.transform(transformed["scores"])
        return transformed

    def predict_sequence(self, signal_matrix) -> dict:
        transformed = self.transform(signal_matrix)
        features = transformed["classifier_features"]
        probabilities = np.asarray(self.estimator.predict_proba(features), dtype=float)
        predicted = np.asarray(self.estimator.classes_, dtype=int)[probabilities.argmax(axis=1)]
        return {
            "predicted_fault_ids": predicted,
            "probabilities": probabilities,
            "probability_class_ids": np.asarray(self.estimator.classes_, dtype=int),
            "sample_indices": transformed["sample_indices"],
            "cva_scores": transformed["scores"],
        }

    def predict_event(
        self,
        signal_matrix,
        *,
        start_index: int = 0,
        minimum_confidence: float = 0.12,
        minimum_margin: float = 0.01,
        weak_fault_ids: Sequence[int] = (3, 9, 15),
        weak_fault_minimum_confidence: float = 0.20,
        weak_fault_minimum_margin: float = 0.03,
    ) -> dict:
        sequence = self.predict_sequence(signal_matrix)
        keep = sequence["sample_indices"] >= max(0, int(start_index))
        if not np.any(keep):
            raise ValueError("no CVA feature samples remain after start_index")
        mean_probability = sequence["probabilities"][keep].mean(axis=0)
        order = np.argsort(mean_probability)[::-1]
        winner = int(order[0])
        runner_up = int(order[1]) if len(order) > 1 else winner
        class_ids = sequence["probability_class_ids"]
        candidate = int(class_ids[winner])
        confidence = float(mean_probability[winner])
        margin = float(mean_probability[winner] - mean_probability[runner_up]) if len(order) > 1 else confidence
        weak_candidate = candidate in {int(value) for value in weak_fault_ids}
        required_confidence = float(weak_fault_minimum_confidence if weak_candidate else minimum_confidence)
        required_margin = float(weak_fault_minimum_margin if weak_candidate else minimum_margin)
        reasons = []
        if confidence < required_confidence:
            reasons.append("classifier_confidence_below_threshold")
        if margin < required_margin:
            reasons.append("classifier_margin_below_threshold")
        if weak_candidate and reasons:
            reasons.append("weak_fault_requires_stronger_evidence")
        abstained = bool(reasons)
        return {
            "predicted_fault_id": None if abstained else candidate,
            "candidate_fault_id": candidate,
            "confidence": confidence,
            "probability_margin": margin,
            "probabilities": {int(k): float(v) for k, v in zip(class_ids, mean_probability)},
            "sample_count": int(np.sum(keep)),
            "decision": "abstain" if abstained else "diagnose",
            "abstained": abstained,
            "abstain_reason": ";".join(reasons) if reasons else None,
            "thresholds": {
                "minimum_confidence": required_confidence,
                "minimum_margin": required_margin,
                "weak_fault_rule_applied": weak_candidate,
            },
        }


def _as_runs(value) -> list[np.ndarray]:
    if isinstance(value, np.ndarray):
        return [np.asarray(value, dtype=float)]
    return [np.asarray(run, dtype=float) for run in value]


def fit_cva_fault_classifier(
    normal_reference,
    training_runs: Mapping[int, Sequence[np.ndarray] | np.ndarray],
    *,
    method: str = "fda",
    past_lags: int = 2,
    future_lags: int = 2,
    variance_target: float = 0.95,
    max_samples_per_class: int | None = None,
    random_state: int = 0,
) -> CVAFaultClassifier:
    """Fit CVA-FDA or CVA-SVM without using test sequences.

    FDA is implemented as shrinkage linear discriminant analysis. Equal class
    priors prevent long TEP runs from silently dominating the decision rule.
    """
    reference = np.asarray(normal_reference, dtype=float)
    if reference.ndim != 2:
        raise ValueError("normal_reference must have shape [samples, channels]")
    mean = reference.mean(axis=0)
    scale = reference.std(axis=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    standardized_reference = (reference - mean) / scale
    cva_state = fit_cva(
        standardized_reference,
        past_lags=past_lags,
        future_lags=future_lags,
        variance_target=variance_target,
    )

    feature_rows: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    rng = np.random.default_rng(random_state)
    for class_id, raw_runs in sorted(training_runs.items()):
        class_features = []
        for run in _as_runs(raw_runs):
            if run.ndim != 2 or run.shape[1] != reference.shape[1]:
                raise ValueError("every training run must match the reference channels")
            z = (run - mean) / scale
            class_features.append(transform_cva(z, cva_state)["scores"])
        rows = np.vstack(class_features)
        if max_samples_per_class is not None and len(rows) > int(max_samples_per_class):
            chosen = np.sort(rng.choice(len(rows), size=int(max_samples_per_class), replace=False))
            rows = rows[chosen]
        feature_rows.append(rows)
        labels.append(np.full(len(rows), int(class_id), dtype=int))

    if len(feature_rows) < 2:
        raise ValueError("at least two fault classes are required")
    x = np.vstack(feature_rows)
    y = np.concatenate(labels)
    feature_scaler = StandardScaler().fit(x)
    x_scaled = feature_scaler.transform(x)
    class_ids = tuple(int(v) for v in np.unique(y))
    method_key = method.lower().replace("-", "_")
    if method_key in {"fda", "lda"}:
        estimator = LinearDiscriminantAnalysis(
            solver="lsqr",
            shrinkage="auto",
            priors=np.full(len(class_ids), 1.0 / len(class_ids)),
        )
        method_name = "cva_fda"
    elif method_key in {"svm", "svc"}:
        estimator = SVC(
            kernel="rbf",
            class_weight="balanced",
            probability=True,
            random_state=random_state,
        )
        method_name = "cva_svm"
    else:
        raise ValueError("method must be 'fda' or 'svm'")
    estimator.fit(x_scaled, y)
    return CVAFaultClassifier(
        cva_state=cva_state,
        reference_mean=mean,
        reference_scale=scale,
        feature_scaler=feature_scaler,
        estimator=estimator,
        class_ids=class_ids,
        method=method_name,
    )
