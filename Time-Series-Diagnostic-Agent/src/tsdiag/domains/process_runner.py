from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..execution import DomainToolRegistry, StepExecutor
from ..tools import (
    MonitoringConfig,
    arbitrate_dpca_cva,
    causal_graph_filter,
    contribution_analysis,
    fault_onset_timing,
    granger_causality,
    knowledge_guided_root_cause_decision,
    run_monitoring_method,
    pca_monitoring,
    pre_post_shift_evidence,
    process_diagnosis,
    root_cause_rank_enhanced,
    standardize_against_normal,
    stationarity_analysis,
    temporal_fault_type_evidence,
)


@dataclass
class ProcessDiagnosticResult:
    fault_detected: bool
    fault_label: str | None
    root_cause: str | None
    affected_variables: list[str]
    propagation_paths: list[list[str]]
    confidence: float
    abstain_reason: str | None
    tool_trace: list[str]
    artifacts: dict[str, Any]
    timings: dict[str, float] | None = None


class ProcessDiagnosticPipeline:
    """Deterministic end-to-end baseline for multivariate process diagnosis.

    The numerical method is intentionally unchanged. Tool execution is routed
    through the shared StepExecutor so timing and failure traces use the same
    infrastructure as the other domain runners.
    """

    def __init__(
        self,
        *,
        variance_target: float = 0.95,
        control_alpha: float = 0.99,
        maxlag: int = 3,
        granger_alpha: float = 0.05,
        onset_z_threshold: float = 3.5,
        onset_persistence: int = 3,
        diagnosis_threshold: float = 0.35,
        minimum_alarm_fraction: float = 0.05,
        use_knowledge_catalog: bool = True,
        monitoring_method: str = "hybrid",
        dynamic_lags: int = 2,
        monitoring_min_consecutive: int = 1,
        strong_dpca_alarm_fraction: float = 0.20,
    ):
        self.variance_target = variance_target
        self.control_alpha = control_alpha
        self.maxlag = maxlag
        self.granger_alpha = granger_alpha
        self.onset_z_threshold = onset_z_threshold
        self.onset_persistence = onset_persistence
        self.diagnosis_threshold = diagnosis_threshold
        self.minimum_alarm_fraction = minimum_alarm_fraction
        self.use_knowledge_catalog = use_knowledge_catalog
        self.monitoring_method = str(monitoring_method).lower()
        self.dynamic_lags = max(1, int(dynamic_lags))
        self.monitoring_min_consecutive = max(1, int(monitoring_min_consecutive))
        self.strong_dpca_alarm_fraction = float(strong_dpca_alarm_fraction)

    @staticmethod
    def _causal_input(x: np.ndarray, stationarity: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
        recommendations = stationarity["differencing_recommendations"]
        difference_channels = [int(r["channel"]) for r in recommendations if r.get("difference")]
        if not difference_channels:
            return x, []

        transformed = x.copy()
        for j in difference_channels:
            transformed[1:, j] = np.diff(x[:, j])
            transformed[0, j] = transformed[1, j]
        return transformed, difference_channels

    def run(
        self,
        signal_matrix,
        normal_reference,
        channel_names,
        *,
        process_topology=None,
        fault_catalog=None,
        timestamps=None,
        detection_override=None,
    ) -> ProcessDiagnosticResult:
        x = np.asarray(signal_matrix, dtype=float)
        ref = np.asarray(normal_reference, dtype=float)
        names = list(channel_names)
        trace: list[str] = []
        artifacts: dict[str, Any] = {}
        timings: dict[str, float] = {}

        registry = DomainToolRegistry()
        executor = StepExecutor("process", registry)
        execution_state: dict[str, Any] = {}

        def execute(name, fn, *args, **kwargs):
            def implementation(_state, _fn=fn, _args=args, _kwargs=kwargs):
                output = _fn(*_args, **_kwargs)
                if not isinstance(output, dict):
                    raise TypeError(f"{name} must return a dictionary")
                return output

            registry.register("process", name, implementation)
            output = executor.run(name, execution_state)
            timings[name] = float(executor.trace[-1].duration_seconds)
            trace.append(name)
            return output

        if x.ndim == 1:
            x = x[:, None]
        if ref.ndim == 1:
            ref = ref[:, None]
        if x.ndim != 2 or ref.ndim != 2 or x.shape[1] != ref.shape[1]:
            raise ValueError("current and normal-reference matrices must be 2-D with matching channels")
        if len(names) != x.shape[1]:
            raise ValueError("channel_names length mismatch")

        standardized = execute("standardize_against_normal", standardize_against_normal, x, ref)
        artifacts["standardization"] = standardized

        hybrid_contributions = None
        if self.monitoring_method == "hybrid":
            dpca = execute(
                "dpca_monitoring",
                run_monitoring_method,
                standardized["standardized_signal"],
                standardized["standardized_reference"],
                MonitoringConfig("dpca", self.control_alpha, self.monitoring_min_consecutive,
                                 variance_target=self.variance_target, lags=self.dynamic_lags),
            )
            cva = execute(
                "cva_monitoring",
                run_monitoring_method,
                standardized["standardized_signal"],
                standardized["standardized_reference"],
                MonitoringConfig("cva", self.control_alpha, self.monitoring_min_consecutive,
                                 variance_target=self.variance_target, lags=self.dynamic_lags),
            )
            arbitration = execute(
                "hybrid_detection_arbitration",
                arbitrate_dpca_cva,
                dpca,
                cva,
                minimum_alarm_fraction=self.minimum_alarm_fraction,
                strong_dpca_alarm_fraction=self.strong_dpca_alarm_fraction,
            )
            monitor = cva
            hybrid_contributions = cva["variable_contributions"]
            artifacts.update({"dpca_monitoring": dpca, "cva_monitoring": cva, "hybrid_detection_arbitration": arbitration})
        elif self.monitoring_method in {"pca", "dpca", "cva"}:
            if self.monitoring_method == "pca":
                monitor = execute(
                    "pca_monitoring", pca_monitoring,
                    standardized["standardized_signal"], standardized["standardized_reference"],
                    variance_target=self.variance_target, alpha=self.control_alpha,
                )
            else:
                monitor = execute(
                    f"{self.monitoring_method}_monitoring", run_monitoring_method,
                    standardized["standardized_signal"], standardized["standardized_reference"],
                    MonitoringConfig(self.monitoring_method, self.control_alpha, self.monitoring_min_consecutive,
                                     variance_target=self.variance_target, lags=self.dynamic_lags),
                )
                if self.monitoring_method == "cva":
                    hybrid_contributions = monitor["variable_contributions"]
            arbitration = {
                "status": "confirmed_fault" if float(np.mean(monitor["alarm_mask"])) >= self.minimum_alarm_fraction else "normal",
                "reason": f"{self.monitoring_method}_only",
                "fault_detected": float(np.mean(monitor["alarm_mask"])) >= self.minimum_alarm_fraction,
                "early_warning": False,
                "alarm_mask": monitor["alarm_mask"],
            }
            artifacts[f"{self.monitoring_method}_monitoring"] = monitor
        else:
            raise ValueError("monitoring_method must be 'hybrid', 'pca', 'dpca', or 'cva'")

        override = dict(detection_override or {})
        alarm_mask = np.asarray(override.get("alarm_mask", arbitration["alarm_mask"]), dtype=bool)
        if alarm_mask.shape != (len(x),):
            raise ValueError("detection_override alarm_mask length mismatch")
        alarm_fraction = float(np.mean(alarm_mask))
        artifacts["detection"] = {
            "alarm_fraction": alarm_fraction,
            "minimum_alarm_fraction": self.minimum_alarm_fraction,
            "method": str(override.get("method", self.monitoring_method)),
            "status": str(arbitration.get("status", "unknown")),
            "early_warning": bool(arbitration.get("early_warning", False)),
            "reason": arbitration.get("reason"),
        }
        fault_detected = (
            alarm_fraction >= self.minimum_alarm_fraction
            if override
            else bool(arbitration.get("fault_detected", alarm_fraction >= self.minimum_alarm_fraction))
        )
        if not fault_detected:
            return ProcessDiagnosticResult(
                fault_detected=False,
                fault_label=None,
                root_cause=None,
                affected_variables=[],
                propagation_paths=[],
                confidence=0.8,
                abstain_reason=None,
                tool_trace=trace,
                artifacts=artifacts,
                timings=timings,
            )

        supplied_contributions = override.get("variable_contributions", hybrid_contributions)
        if supplied_contributions is not None:
            sample_contributions = np.asarray(supplied_contributions, dtype=float)
            if sample_contributions.shape != x.shape:
                raise ValueError("detection_override variable_contributions shape mismatch")
            active = sample_contributions[alarm_mask] if np.any(alarm_mask) else sample_contributions[-min(20, len(x)):]
            values = np.mean(active, axis=0)
            values = values / (float(np.sum(values)) + 1e-12)
            order = np.argsort(values)[::-1]
            suspects = [int(i) for i in order if values[i] >= max(0.1, 1 / (2 * len(values)))]
            contributions = {
                "variable_contributions": values,
                "suspect_variables": suspects,
                "ranked_indices": order,
                "method": str(override.get("method", self.monitoring_method)),
            }
            trace.append("contribution_analysis")
        else:
            contributions = execute(
                "contribution_analysis",
                contribution_analysis,
                standardized["standardized_signal"],
                pca["pca_state"],
                alarm_mask,
            )
        artifacts["contribution_analysis"] = contributions

        shift = execute(
            "pre_post_shift_evidence",
            pre_post_shift_evidence,
            standardized["standardized_signal"],
            names,
            alarm_mask=alarm_mask,
        )
        artifacts["pre_post_shift_evidence"] = shift

        type_evidence = execute(
            "temporal_fault_type_evidence",
            temporal_fault_type_evidence,
            standardized["standardized_signal"],
            names,
            alarm_mask=alarm_mask,
        )
        artifacts["temporal_fault_type_evidence"] = type_evidence

        stationarity = execute(
            "stationarity_analysis",
            stationarity_analysis,
            standardized["standardized_signal"],
        )
        artifacts["stationarity_analysis"] = stationarity

        causal_input, differenced_channels = self._causal_input(
            standardized["standardized_signal"], stationarity
        )
        artifacts["causal_preprocessing"] = {"differenced_channels": differenced_channels}

        granger = execute(
            "granger_causality",
            granger_causality,
            causal_input,
            names,
            maxlag=self.maxlag,
            alpha=self.granger_alpha,
        )
        artifacts["granger_causality"] = granger

        filtered = execute(
            "causal_graph_filter",
            causal_graph_filter,
            granger["directed_edges"],
            process_topology=process_topology,
            p_value_threshold=self.granger_alpha,
        )
        artifacts["causal_graph_filter"] = filtered

        onset = execute(
            "fault_onset_timing",
            fault_onset_timing,
            standardized["standardized_signal"],
            alarm_mask,
            names,
            timestamps=timestamps,
            z_threshold=self.onset_z_threshold,
            persistence=self.onset_persistence,
        )
        artifacts["fault_onset_timing"] = onset

        ranking = execute(
            "root_cause_rank_enhanced",
            root_cause_rank_enhanced,
            contributions["suspect_variables"],
            filtered["filtered_causal_graph"],
            onset["onset_order"],
            variable_contributions=contributions["variable_contributions"],
            shift_scores=shift["shift_scores"],
            channel_names=names,
        )
        artifacts["root_cause_rank"] = ranking

        contribution_scores = {
            names[i]: float(v) for i, v in enumerate(contributions["variable_contributions"])
        }
        if self.use_knowledge_catalog and fault_catalog and isinstance(fault_catalog, dict) and "faults" in fault_catalog:
            kg = execute(
                "knowledge_guided_root_cause_decision",
                knowledge_guided_root_cause_decision,
                ranking["root_cause_ranking"],
                fault_catalog,
                shift_scores=shift["shift_scores"],
                contribution_scores=contribution_scores,
                onset_order=onset["onset_order"],
                fault_type_scores=type_evidence["fault_type_scores"],
            )
            artifacts["knowledge_guided_root_cause"] = kg
            abstain_reason = kg.get("abstain_reason")
            accepted = (
                kg.get("root_cause") is not None
                and kg["confidence"] >= self.diagnosis_threshold
                and abstain_reason is None
            )
            diagnosis = {
                "fault_label": kg["fault_label"] if accepted else None,
                "root_cause": kg["root_cause"] if accepted else None,
                "confidence": kg["confidence"],
                "abstain_reason": None if accepted else (abstain_reason or "knowledge_guided_evidence_below_threshold"),
                "affected_variables": sorted(
                    {
                        v
                        for path in ranking["propagation_paths"]
                        for v in path
                        if v != kg.get("root_cause")
                    }
                ),
                "propagation_paths": ranking["propagation_paths"],
            }
        else:
            diagnosis = execute(
                "process_diagnosis",
                process_diagnosis,
                ranking["root_cause_ranking"],
                ranking["propagation_paths"],
                fault_catalog=fault_catalog,
                confidence_threshold=self.diagnosis_threshold,
            )

        artifacts["process_diagnosis"] = diagnosis
        if trace[-1] != "process_diagnosis":
            trace.append("process_diagnosis")

        return ProcessDiagnosticResult(
            fault_detected=True,
            fault_label=diagnosis["fault_label"],
            root_cause=diagnosis["root_cause"],
            affected_variables=list(diagnosis["affected_variables"]),
            propagation_paths=list(diagnosis["propagation_paths"]),
            confidence=float(diagnosis["confidence"]),
            abstain_reason=diagnosis["abstain_reason"],
            tool_trace=trace,
            artifacts=artifacts,
            timings=timings,
        )
