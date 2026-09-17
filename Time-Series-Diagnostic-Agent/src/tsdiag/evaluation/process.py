from __future__ import annotations

from typing import Iterable, Mapping, Any

import numpy as np
from sklearn.metrics import accuracy_score, f1_score


def evaluate_process_predictions(records: Iterable[Mapping[str, Any]]) -> dict[str, float]:
    """Evaluate process-diagnostic outputs, including TEP-style root-cause tasks.

    Each record may contain:
      true_fault_detected, predicted_fault_detected
      true_fault_label, predicted_fault_label
      true_root_cause, predicted_root_cause
      true_onset_index, predicted_onset_index

    Missing optional labels are ignored rather than silently treated as errors.
    """
    rows = list(records)
    if not rows:
        raise ValueError("at least one evaluation record is required")

    y_true_detect = [bool(r["true_fault_detected"]) for r in rows]
    y_pred_detect = [bool(r["predicted_fault_detected"]) for r in rows]

    metrics: dict[str, float] = {
        "detection_f1": float(f1_score(y_true_detect, y_pred_detect, zero_division=0)),
        "detection_accuracy": float(accuracy_score(y_true_detect, y_pred_detect)),
    }

    healthy = np.array([not x for x in y_true_detect], dtype=bool)
    pred_fault = np.array(y_pred_detect, dtype=bool)
    metrics["false_alarm_rate"] = float(pred_fault[healthy].mean()) if healthy.any() else 0.0

    fault_label_pairs = [
        (r.get("true_fault_label"), r.get("predicted_fault_label"))
        for r in rows
        if r.get("true_fault_label") is not None
    ]
    if fault_label_pairs:
        metrics["fault_class_accuracy"] = float(
            np.mean([truth == pred for truth, pred in fault_label_pairs])
        )

    root_pairs = [
        (r.get("true_root_cause"), r.get("predicted_root_cause"))
        for r in rows
        if r.get("true_root_cause") is not None
    ]
    if root_pairs:
        metrics["root_cause_accuracy"] = float(
            np.mean([truth == pred for truth, pred in root_pairs])
        )

    delays = []
    for r in rows:
        true_onset = r.get("true_onset_index")
        pred_onset = r.get("predicted_onset_index")
        if true_onset is not None and pred_onset is not None:
            delays.append(float(pred_onset) - float(true_onset))
    if delays:
        metrics["detection_delay"] = float(np.mean(delays))
        metrics["absolute_detection_delay"] = float(np.mean(np.abs(delays)))

    return metrics
