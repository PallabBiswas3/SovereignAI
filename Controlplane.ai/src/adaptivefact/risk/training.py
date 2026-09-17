from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import average_precision_score
from sklearn.model_selection import GroupShuffleSplit

from adaptivefact.benchmark.metrics import classification_metrics
from adaptivefact.data.schema import ResponseRecord
from adaptivefact.risk.model import RiskModelBundle, build_logistic_regression, build_xgboost


@dataclass
class TrainValidationSplit:
    train_indices: np.ndarray
    validation_indices: np.ndarray


def labels_from_records(records: list[ResponseRecord]) -> np.ndarray:
    return np.asarray(
        [1 if r.ground_truth_label is not None and r.ground_truth_label.value == "hallucinated" else 0 for r in records],
        dtype=np.int32,
    )


def group_validation_split(
    records: list[ResponseRecord],
    *,
    validation_size: float = 0.2,
    random_state: int = 42,
    candidates: int = 20,
) -> TrainValidationSplit:
    """Choose a source-disjoint validation split with prevalence close to the full train set."""
    y = labels_from_records(records)
    groups = np.asarray([r.source_id or r.id for r in records])
    overall_rate = float(y.mean()) if len(y) else 0.0

    splitter = GroupShuffleSplit(
        n_splits=max(1, candidates),
        test_size=validation_size,
        random_state=random_state,
    )
    best = None
    best_gap = float("inf")
    dummy = np.zeros(len(records))
    for train_idx, val_idx in splitter.split(dummy, y, groups):
        if len(set(y[train_idx])) < 2 or len(set(y[val_idx])) < 2:
            continue
        gap = abs(float(y[val_idx].mean()) - overall_rate)
        if gap < best_gap:
            best_gap = gap
            best = (train_idx, val_idx)

    if best is None:
        raise ValueError("Could not create a source-disjoint train/validation split containing both classes")
    return TrainValidationSplit(train_indices=best[0], validation_indices=best[1])


def evaluate_probabilities(y_true: np.ndarray, probabilities: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    predictions = (probabilities >= threshold).astype(int)
    metrics = classification_metrics(y_true.tolist(), predictions.tolist(), probabilities.tolist())
    if len(set(y_true.tolist())) > 1:
        metrics["auprc"] = float(average_precision_score(y_true, probabilities))
    else:
        metrics["auprc"] = float("nan")
    return metrics


def train_models(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: list[str],
    *,
    random_state: int = 42,
) -> dict[str, RiskModelBundle]:
    logreg = build_logistic_regression(random_state=random_state)
    logreg.fit(X_train, y_train)

    positives = int(y_train.sum())
    negatives = int(len(y_train) - positives)
    scale_pos_weight = negatives / max(1, positives)
    xgb = build_xgboost(random_state=random_state, scale_pos_weight=scale_pos_weight)
    xgb.fit(X_train, y_train)

    return {
        "logreg": RiskModelBundle("logreg", logreg, feature_names),
        "xgboost": RiskModelBundle("xgboost", xgb, feature_names),
    }
