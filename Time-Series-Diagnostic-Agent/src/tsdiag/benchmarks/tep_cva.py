from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter
from typing import Iterable

import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score

from ..datasets.tep import (
    TEP_FAULTS,
    TEP_TEST_FAULT_START,
    TEP_TRAIN_FAULT_START,
    load_tep_dat,
    load_tep_reference,
)
from ..tools import (
    arbitrate_dpca_cva,
    calibrate_monitoring_config,
    fit_cva_fault_classifier,
    run_monitoring_method,
    standardize_against_normal,
)


def _detection_metrics(alarm_mask, fault_id: int) -> dict:
    alarm = np.asarray(alarm_mask, dtype=bool)
    split = min(TEP_TEST_FAULT_START, len(alarm))
    post = alarm[split:]
    detected = np.flatnonzero(post)
    return {
        "pre_fault_false_alarm_rate": float(np.mean(alarm[:split])) if split else 0.0,
        "post_fault_detection_rate": None if fault_id == 0 else float(np.mean(post)),
        "detection_delay_samples": None if fault_id == 0 or not len(detected) else int(detected[0]),
    }


def _classification_metrics(y_true, y_pred, labels: list[int]) -> dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def run_tep_cva_benchmark(
    data_dir: str | Path,
    *,
    output: str | Path | None = None,
    fault_ids: Iterable[int] = range(0, 21),
    methods: Iterable[str] = ("pca", "dpca", "cva"),
    classifier: str = "svm",
    past_lags: int = 2,
    variance_target: float = 0.95,
    target_false_alarm_rate: float = 0.05,
    max_samples_per_class: int | None = None,
    seed: int = 0,
    fitted_classifier=None,
    model_output: str | Path | None = None,
    minimum_classifier_confidence: float = 0.12,
    minimum_classifier_margin: float = 0.01,
    weak_fault_minimum_confidence: float = 0.20,
    weak_fault_minimum_margin: float = 0.03,
) -> dict:
    """Compare detection methods and evaluate train-only CVA fault diagnosis."""
    root = Path(data_dir)
    ids = tuple(sorted({int(v) for v in fault_ids}))
    if 0 not in ids:
        raise ValueError("fault_ids must include class 0 for normal-operation evaluation")
    reference = load_tep_reference(root)
    standardized_reference = standardize_against_normal(reference, reference)["standardized_reference"]

    if fitted_classifier is None:
        training_runs = {}
        for fault_id in ids:
            run = load_tep_dat(root / f"d{fault_id:02d}.dat")
            # Training fault simulations contain a short normal prefix. Do not label
            # those healthy samples as their eventual fault class.
            training_runs[fault_id] = run if fault_id == 0 else run[TEP_TRAIN_FAULT_START:]
        started = perf_counter()
        fault_classifier = fit_cva_fault_classifier(
            reference, training_runs, method=classifier, past_lags=past_lags,
            future_lags=past_lags, variance_target=variance_target,
            max_samples_per_class=max_samples_per_class, random_state=seed,
        )
        classifier_fit_seconds = perf_counter() - started
        classifier_source = "trained"
    else:
        fault_classifier = fitted_classifier
        classifier_fit_seconds = 0.0
        classifier_source = "loaded_artifact"
    saved_model = None if model_output is None else str(fault_classifier.save(model_output))

    calibrated = {}
    for method in methods:
        key = method.lower()
        calibrated[key] = calibrate_monitoring_config(
            standardized_reference,
            method=key,
            variance_target=variance_target,
            target_false_alarm_rate=target_false_alarm_rate,
            lags_grid=(past_lags,),
        )["config"]

    detection_rows = []
    diagnosis_rows = []
    y_true = []
    y_pred = []
    for fault_id in ids:
        current = load_tep_dat(root / f"d{fault_id:02d}_te.dat")
        standardized = standardize_against_normal(current, reference)
        monitored_by_method = {}
        for method, config in calibrated.items():
            monitored = run_monitoring_method(
                standardized["standardized_signal"],
                standardized["standardized_reference"],
                config,
            )
            monitored_by_method[method] = monitored
            detection_rows.append({
                "method": method,
                "fault_id": fault_id,
                **_detection_metrics(monitored["alarm_mask"], fault_id),
            })
        if "dpca" in monitored_by_method and "cva" in monitored_by_method:
            hybrid = arbitrate_dpca_cva(monitored_by_method["dpca"], monitored_by_method["cva"])
            detection_rows.append({
                "method": "hybrid",
                "fault_id": fault_id,
                "status": hybrid["status"],
                "reason": hybrid["reason"],
                **_detection_metrics(hybrid["alarm_mask"], fault_id),
            })

        event = fault_classifier.predict_event(
            current,
            start_index=0 if fault_id == 0 else TEP_TEST_FAULT_START,
            minimum_confidence=minimum_classifier_confidence,
            minimum_margin=minimum_classifier_margin,
            weak_fault_minimum_confidence=weak_fault_minimum_confidence,
            weak_fault_minimum_margin=weak_fault_minimum_margin,
        )
        y_true.append(fault_id)
        y_pred.append(event["candidate_fault_id"])
        diagnosis_rows.append({
            "fault_id": fault_id,
            "description": TEP_FAULTS[fault_id]["description"],
            **event,
        })

    labels = list(ids)
    detection_summary = {}
    summary_methods = list(calibrated)
    if any(row["method"] == "hybrid" for row in detection_rows):
        summary_methods.append("hybrid")
    for method in summary_methods:
        rows = [r for r in detection_rows if r["method"] == method]
        faulty = [r for r in rows if r["fault_id"] != 0]
        delays = [r["detection_delay_samples"] for r in faulty if r["detection_delay_samples"] is not None]
        detection_summary[method] = {
            "mean_pre_fault_false_alarm_rate": float(np.mean([r["pre_fault_false_alarm_rate"] for r in rows])),
            "mean_post_fault_detection_rate": float(np.mean([r["post_fault_detection_rate"] for r in faulty])),
            "faults_detected": int(sum((r["post_fault_detection_rate"] or 0.0) > 0 for r in faulty)),
            "fault_count": len(faulty),
            "mean_detection_delay_samples": None if not delays else float(np.mean(delays)),
        }

    all_metrics = _classification_metrics(y_true, y_pred, labels)
    fault_true = [value for value in y_true if value != 0]
    fault_pred = [pred for true, pred in zip(y_true, y_pred) if true != 0]
    fault_labels = [value for value in labels if value != 0]
    fault_metrics = _classification_metrics(fault_true, fault_pred, fault_labels)
    covered = [row for row in diagnosis_rows if not row["abstained"]]
    selective_accuracy = None if not covered else float(np.mean([
        row["predicted_fault_id"] == row["fault_id"] for row in covered
    ]))

    payload = {
        "source": "Braatz Tennessee Eastman Process archive",
        "protocol": {
            "fault_ids": labels,
            "detection_calibration": "healthy d00.dat only; thresholds selected by held-out healthy false-alarm rate",
            "classifier_training": "d00.dat plus post-onset samples (index >=20) from selected d01.dat... training files only",
            "classifier_test": "one frozen dXX_te.dat event per class; faulty scoring begins at sample 160",
            "cva": {
                "past_lags": past_lags,
                "future_lags_training_only": past_lags,
                "variance_target": variance_target,
                "online_lookahead": False,
            },
            "classifier": fault_classifier.method,
            "classifier_source": classifier_source,
            "classifier_abstention": {
                "minimum_confidence": minimum_classifier_confidence,
                "minimum_margin": minimum_classifier_margin,
                "weak_fault_ids": [3, 9, 15],
                "weak_fault_minimum_confidence": weak_fault_minimum_confidence,
                "weak_fault_minimum_margin": weak_fault_minimum_margin,
            },
            "seed": seed,
        },
        "classifier_fit_seconds": float(classifier_fit_seconds),
        "model_artifact": saved_model,
        "detection": {"summary": detection_summary, "rows": detection_rows},
        "diagnosis": {
            **all_metrics,
            "all_events": all_metrics,
            "fault_events_only": fault_metrics,
            "selective": {
                "coverage": float(len(covered) / len(diagnosis_rows)),
                "abstention_rate": float(1.0 - len(covered) / len(diagnosis_rows)),
                "covered_accuracy": selective_accuracy,
                "covered_events": len(covered),
            },
            "events": diagnosis_rows,
        },
        "root_cause_stage": {
            "status": "retained",
            "implementation": "ProcessDiagnosticPipeline: contribution + temporal evidence + Granger/process-topology filtering + onset ordering",
            "note": "Fault-ID classification is reported independently and does not overwrite causal evidence.",
        },
    }
    if output is not None:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
