from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from ..datasets.tep import TEP_TRAIN_FAULT_START
from ..tools import (
    MonitoringConfig,
    arbitrate_dpca_cva,
    calibrate_monitoring_config,
    fit_cva_fault_classifier,
    run_monitoring_method,
    standardize_against_normal,
)
from .process_runner import ProcessDiagnosticPipeline, ProcessDiagnosticResult


@dataclass
class CVATEPDiagnosticResult:
    fault_detected: bool
    detection_method: str
    detection_status: str
    early_warning: bool
    alarm_fraction: float
    predicted_fault_id: int | None
    classifier_confidence: float | None
    classifier_decision: str | None
    abstain_reason: str | None
    root_cause: ProcessDiagnosticResult | None
    monitoring: dict


class CVATEPDiagnosticPipeline:
    """DPCA warning + CVA confirmation -> classifier -> causal root analysis."""

    def __init__(
        self,
        normal_reference,
        classifier,
        monitoring_config: MonitoringConfig,
        *,
        dpca_config: MonitoringConfig | None = None,
        minimum_alarm_fraction: float = 0.05,
        strong_dpca_alarm_fraction: float = 0.20,
        root_cause_pipeline: ProcessDiagnosticPipeline | None = None,
    ):
        self.normal_reference = np.asarray(normal_reference, dtype=float)
        self.classifier = classifier
        self.monitoring_config = monitoring_config
        self.dpca_config = dpca_config or MonitoringConfig(
            "dpca",
            monitoring_config.alpha,
            monitoring_config.min_consecutive,
            variance_target=monitoring_config.variance_target,
            lags=max(1, monitoring_config.lags),
        )
        self.minimum_alarm_fraction = float(minimum_alarm_fraction)
        self.strong_dpca_alarm_fraction = float(strong_dpca_alarm_fraction)
        self.root_cause_pipeline = root_cause_pipeline or ProcessDiagnosticPipeline(
            minimum_alarm_fraction=self.minimum_alarm_fraction,
            monitoring_method="hybrid",
        )

    @classmethod
    def fit(
        cls,
        normal_reference,
        training_runs: Mapping[int, Sequence[np.ndarray] | np.ndarray],
        *,
        classifier_method: str = "svm",
        past_lags: int = 2,
        variance_target: float = 0.95,
        target_false_alarm_rate: float = 0.05,
        max_samples_per_class: int | None = None,
        seed: int = 0,
        minimum_alarm_fraction: float = 0.05,
        training_fault_start: int = TEP_TRAIN_FAULT_START,
    ) -> "CVATEPDiagnosticPipeline":
        reference = np.asarray(normal_reference, dtype=float)
        standardized_reference = standardize_against_normal(reference, reference)["standardized_reference"]
        calibrations = {
            method: calibrate_monitoring_config(
                standardized_reference,
                method=method,
                variance_target=variance_target,
                target_false_alarm_rate=target_false_alarm_rate,
                lags_grid=(past_lags,),
            )["config"]
            for method in ("dpca", "cva")
        }
        prepared_runs = {
            int(class_id): (
                np.asarray(runs)
                if int(class_id) == 0 or not isinstance(runs, np.ndarray)
                else np.asarray(runs)[max(0, int(training_fault_start)):]
            )
            for class_id, runs in training_runs.items()
        }
        classifier = fit_cva_fault_classifier(
            reference,
            prepared_runs,
            method=classifier_method,
            past_lags=past_lags,
            future_lags=past_lags,
            variance_target=variance_target,
            max_samples_per_class=max_samples_per_class,
            random_state=seed,
        )
        return cls(
            reference,
            classifier,
            calibrations["cva"],
            dpca_config=calibrations["dpca"],
            minimum_alarm_fraction=minimum_alarm_fraction,
        )

    def run(
        self,
        signal_matrix,
        channel_names,
        *,
        fault_start_index: int = 0,
        process_topology=None,
        fault_catalog=None,
        timestamps=None,
    ) -> CVATEPDiagnosticResult:
        current = np.asarray(signal_matrix, dtype=float)
        standardized = standardize_against_normal(current, self.normal_reference)
        cva = run_monitoring_method(
            standardized["standardized_signal"],
            standardized["standardized_reference"],
            self.monitoring_config,
        )
        dpca = run_monitoring_method(
            standardized["standardized_signal"],
            standardized["standardized_reference"],
            self.dpca_config,
        )
        arbitration = arbitrate_dpca_cva(
            dpca,
            cva,
            minimum_alarm_fraction=self.minimum_alarm_fraction,
            strong_dpca_alarm_fraction=self.strong_dpca_alarm_fraction,
        )
        monitoring = {"dpca": dpca, "cva": cva, "arbitration": arbitration}
        alarm_fraction = float(arbitration["cva_alarm_fraction"])
        detected = bool(arbitration["fault_detected"])
        if not detected:
            return CVATEPDiagnosticResult(
                False, "dpca_cva_hybrid", str(arbitration["status"]),
                bool(arbitration["early_warning"]), alarm_fraction,
                None, None, None, None, None, monitoring,
            )

        diagnosis = self.classifier.predict_event(current, start_index=fault_start_index)
        root = self.root_cause_pipeline.run(
            current,
            self.normal_reference,
            channel_names,
            process_topology=process_topology,
            fault_catalog=fault_catalog,
            timestamps=timestamps,
            detection_override={
                "method": "dpca_cva_hybrid",
                "alarm_mask": arbitration["alarm_mask"],
                "variable_contributions": cva["variable_contributions"],
            },
        )
        predicted = diagnosis.get("predicted_fault_id")
        return CVATEPDiagnosticResult(
            True,
            "dpca_cva_hybrid",
            str(arbitration["status"]),
            bool(arbitration["early_warning"]),
            alarm_fraction,
            None if predicted is None else int(predicted),
            float(diagnosis["confidence"]),
            str(diagnosis.get("decision", "diagnose")),
            diagnosis.get("abstain_reason"),
            root,
            monitoring,
        )
