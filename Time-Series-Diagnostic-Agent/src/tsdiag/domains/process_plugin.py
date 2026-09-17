from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..contracts import DiagnosticRequest
from ..execution import ExecutionTrace, Step, Workflow
from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    UncertaintyEstimate,
)
from ..result_contract import standardize_result
from ..tools import (
    MonitoringConfig,
    arbitrate_dpca_cva,
    causal_graph_filter,
    contribution_analysis,
    fault_onset_timing,
    granger_causality,
    knowledge_guided_root_cause_decision,
    pca_monitoring,
    pre_post_shift_evidence,
    process_diagnosis,
    root_cause_rank_enhanced,
    run_monitoring_method,
    standardize_against_normal,
    stationarity_analysis,
    temporal_fault_type_evidence,
)


def _inactive(state: Mapping[str, Any]) -> bool:
    return state.get("fault_detected") is False


def _causal_input(x: np.ndarray, stationarity: Mapping[str, Any]) -> tuple[np.ndarray, list[int]]:
    recommendations = stationarity["differencing_recommendations"]
    difference_channels = [int(row["channel"]) for row in recommendations if row.get("difference")]
    if not difference_channels:
        return x, []
    transformed = x.copy()
    for channel in difference_channels:
        transformed[1:, channel] = np.diff(x[:, channel])
        transformed[0, channel] = transformed[1, channel]
    return transformed, difference_channels


@dataclass(frozen=True)
class ProcessDecisionPolicy:
    request: DiagnosticRequest
    version: str = "process-policy-v3"

    def decide(self, execution: Mapping[str, Any], trace: ExecutionTrace) -> DiagnosticResult:
        state = dict(execution)
        detected = bool(state.get("fault_detected", False))
        alarm_fraction = float(state.get("alarm_fraction", 0.0))
        if not detected:
            confidence = 0.8
            early_warning = bool(state.get("early_warning", False))
            return standardize_result(DiagnosticResult(
                domain="process",
                task="root_cause",
                decision="monitor",
                detection=DetectionResult(False, score=alarm_fraction, method=str(state.get("monitoring_method", "dpca_cva_hybrid")), details={"alarm_fraction": alarm_fraction, "arbitration": state.get("monitoring_arbitration")}),
                confidence=confidence,
                uncertainty=None,
                uncertainty_estimate=UncertaintyEstimate(None, "not_calibrated_process_baseline"),
                recommended_actions=["Investigate the DPCA early warning and await CVA confirmation."] if early_warning else [],
                tool_trace=trace,
                metadata={
                    "workflow_version": ProcessPlugin.workflow_version,
                    "policy_version": self.version,
                    "alarm_fraction": alarm_fraction,
                    "early_warning": early_warning,
                    "monitoring_arbitration": state.get("monitoring_arbitration"),
                    "allow_confidence_complement_uncertainty": False,
                },
            ))

        diagnosis = dict(state.get("diagnosis") or {})
        confidence = float(np.clip(diagnosis.get("confidence", state.get("root_confidence", 0.0)), 0.0, 1.0))
        root = diagnosis.get("root_cause")
        fault_label = diagnosis.get("fault_label")
        abstain_reason = diagnosis.get("abstain_reason")
        decision = "abstain" if abstain_reason else "diagnose"
        propagation_paths = list(diagnosis.get("propagation_paths") or state.get("propagation_paths") or [])
        affected = list(diagnosis.get("affected_variables") or [])

        evidence: list[Evidence] = []
        if root or state.get("root_cause_ranking"):
            evidence = [Evidence(
                source="process_root_cause",
                statement=f"Detected process deviation; root-cause candidate={root!r}.",
                score=confidence,
                details={
                    "root_cause_ranking": state.get("root_cause_ranking", []),
                    "affected_variables": affected,
                    "propagation_paths": propagation_paths,
                    "alarm_fraction": alarm_fraction,
                },
                evidence_id="process-diagnostic-evidence",
                kind="causal",
            )]
            target_step = "knowledge_guided_root_cause_decision" if state.get("knowledge_guided_used") else "process_diagnosis"
            trace.require(target_step).evidence_ids.append("process-diagnostic-evidence")

        hypotheses: list[DiagnosticHypothesis] = []
        if fault_label or root:
            hypotheses.append(DiagnosticHypothesis(
                label=str(fault_label or root),
                score=confidence,
                rationale="Hybrid dynamic monitoring, contribution, temporal, causal, onset, and topology evidence were combined.",
                evidence_ids=["process-diagnostic-evidence"] if evidence else [],
                details={"root_cause": root},
            ))

        return standardize_result(DiagnosticResult(
            domain="process",
            task="root_cause",
            decision=decision,
            detection=DetectionResult(True, score=float(np.clip(max(alarm_fraction, confidence), 0.0, 1.0)), method=str(state.get("monitoring_method", "dpca_cva_hybrid")), details={"alarm_fraction": alarm_fraction, "arbitration": state.get("monitoring_arbitration")}),
            localization=LocalizationResult(
                components=[str(root)] if root else [],
                channels=affected,
                scores={str(root): confidence} if root else {},
                details={"propagation_paths": propagation_paths},
            ),
            hypotheses=hypotheses,
            evidence=evidence,
            confidence=confidence,
            uncertainty=None,
            uncertainty_estimate=UncertaintyEstimate(None, "not_calibrated_process_baseline"),
            abstained=bool(abstain_reason),
            abstain_reason=abstain_reason,
            recommended_actions=["Verify the proposed root cause against process topology and operating history."] if root else [],
            tool_trace=trace,
            metadata={
                "workflow_version": ProcessPlugin.workflow_version,
                "policy_version": self.version,
                "alarm_fraction": alarm_fraction,
                "differenced_channels": state.get("differenced_channels", []),
                "fault_type_scores": state.get("fault_type_scores", {}),
                "monitoring_arbitration": state.get("monitoring_arbitration"),
                "supervised_classification": state.get("supervised_classification"),
                "allow_confidence_complement_uncertainty": False,
            },
        ))


class ProcessPlugin:
    name = "process"
    workflow_version = "3.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        required = ("signal_matrix", "normal_reference", "channel_names")
        missing = [key for key in required if values.get(key) is None]
        if missing:
            raise ValueError(f"process requires {', '.join(missing)}")
        x = np.asarray(values["signal_matrix"], dtype=float)
        ref = np.asarray(values["normal_reference"], dtype=float)
        if x.ndim == 1: x = x[:, None]
        if ref.ndim == 1: ref = ref[:, None]
        if x.ndim != 2 or ref.ndim != 2 or x.shape[1] != ref.shape[1]:
            raise ValueError("current and normal-reference matrices must be 2-D with matching channels")
        channel_names = [str(row) for row in values["channel_names"]]
        if len(channel_names) != x.shape[1]:
            raise ValueError("channel_names length mismatch")
        values["signal_matrix"] = x
        values["normal_reference"] = ref
        values["channel_names"] = channel_names
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        cfg = dict(request.inputs)

        def standardize(state):
            return standardize_against_normal(state["signal_matrix"], state["normal_reference"])

        def monitor(state):
            method = str(cfg.get("monitoring_method", "hybrid")).lower()
            variance = float(cfg.get("variance_target", 0.95))
            alpha = float(cfg.get("control_alpha", 0.99))
            persistence = int(cfg.get("monitoring_min_consecutive", 1))
            lags = int(cfg.get("dynamic_lags", 2))
            minimum = float(cfg.get("minimum_alarm_fraction", 0.05))
            if method == "hybrid":
                dpca = run_monitoring_method(
                    state["standardized_signal"], state["standardized_reference"],
                    MonitoringConfig("dpca", alpha, persistence, variance_target=variance, lags=lags),
                )
                cva = run_monitoring_method(
                    state["standardized_signal"], state["standardized_reference"],
                    MonitoringConfig("cva", alpha, persistence, variance_target=variance, lags=lags),
                )
                arbitration = arbitrate_dpca_cva(
                    dpca, cva, minimum_alarm_fraction=minimum,
                    strong_dpca_alarm_fraction=float(cfg.get("strong_dpca_alarm_fraction", 0.20)),
                )
                alarm = np.asarray(arbitration["alarm_mask"], dtype=bool)
                return {
                    "monitoring_method": "dpca_cva_hybrid",
                    "dpca_monitoring": dpca,
                    "cva_monitoring": cva,
                    "monitoring_arbitration": arbitration,
                    "alarm_mask": alarm,
                    "alarm_fraction": float(np.mean(alarm)),
                    "fault_detected": bool(arbitration["fault_detected"]),
                    "early_warning": bool(arbitration["early_warning"]),
                    "variable_contributions": cva["variable_contributions"],
                }
            if method not in {"pca", "dpca", "cva"}:
                raise ValueError("monitoring_method must be 'hybrid', 'pca', 'dpca', or 'cva'")
            if method == "pca":
                out = pca_monitoring(
                    state["standardized_signal"], state["standardized_reference"],
                    variance_target=variance, alpha=alpha,
                )
            else:
                out = run_monitoring_method(
                    state["standardized_signal"], state["standardized_reference"],
                    MonitoringConfig(method, alpha, persistence, variance_target=variance, lags=lags),
                )
            alarm_fraction = float(np.mean(np.asarray(out["alarm_mask"], dtype=bool)))
            return {**out, "monitoring_method": method, "alarm_fraction": alarm_fraction,
                    "fault_detected": alarm_fraction >= minimum, "early_warning": False}

        def classify(state):
            if _inactive(state): return {"supervised_classification": None}
            model = state.get("trained_fault_classifier")
            if model is None: return {"supervised_classification": None}
            if not hasattr(model, "predict_event"):
                raise TypeError("trained_fault_classifier must provide predict_event")
            alarm_indices = np.flatnonzero(np.asarray(state["alarm_mask"], dtype=bool))
            start = int(cfg.get("fault_start_index", alarm_indices[0] if len(alarm_indices) else 0))
            return {"supervised_classification": model.predict_event(state["signal_matrix"], start_index=start)}

        def contributions(state):
            if _inactive(state): return {}
            if state.get("monitoring_method") in {"cva", "dpca_cva_hybrid"}:
                sample = np.asarray(state["variable_contributions"], dtype=float)
                alarm = np.asarray(state["alarm_mask"], dtype=bool)
                active = sample[alarm] if np.any(alarm) else sample[-min(20, len(sample)):]
                values = np.mean(active, axis=0)
                values /= float(np.sum(values)) + 1e-12
                order = np.argsort(values)[::-1]
                suspects = [int(i) for i in order if values[i] >= max(0.1, 1 / (2 * len(values)))]
                return {"variable_contributions": values, "suspect_variables": suspects, "ranked_indices": order}
            return contribution_analysis(state["standardized_signal"], state["pca_state"], state["alarm_mask"])

        def shift(state):
            if _inactive(state): return {}
            return pre_post_shift_evidence(state["standardized_signal"], state["channel_names"], alarm_mask=state["alarm_mask"])

        def type_evidence(state):
            if _inactive(state): return {}
            return temporal_fault_type_evidence(state["standardized_signal"], state["channel_names"], alarm_mask=state["alarm_mask"])

        def stationarity(state):
            if _inactive(state): return {}
            out = stationarity_analysis(state["standardized_signal"])
            causal, differenced = _causal_input(state["standardized_signal"], out)
            return {**out, "causal_input": causal, "differenced_channels": differenced}

        def granger(state):
            if _inactive(state): return {}
            return granger_causality(
                state["causal_input"], state["channel_names"],
                maxlag=int(cfg.get("maxlag", 3)), alpha=float(cfg.get("granger_alpha", 0.05)),
            )

        def causal_filter(state):
            if _inactive(state): return {}
            return causal_graph_filter(
                state["directed_edges"],
                process_topology=state.get("process_topology"),
                p_value_threshold=float(cfg.get("granger_alpha", 0.05)),
            )

        def onset(state):
            if _inactive(state): return {}
            return fault_onset_timing(
                state["standardized_signal"], state["alarm_mask"], state["channel_names"],
                timestamps=state.get("timestamps"),
                z_threshold=float(cfg.get("onset_z_threshold", 3.5)),
                persistence=int(cfg.get("onset_persistence", 3)),
            )

        def rank(state):
            if _inactive(state): return {}
            return root_cause_rank_enhanced(
                state["suspect_variables"], state["filtered_causal_graph"], state["onset_order"],
                variable_contributions=state["variable_contributions"],
                shift_scores=state["shift_scores"], channel_names=state["channel_names"],
            )

        def knowledge(state):
            if _inactive(state): return {"knowledge_guided_used": False}
            catalog = state.get("fault_catalog")
            if not (bool(cfg.get("use_knowledge_catalog", True)) and isinstance(catalog, dict) and "faults" in catalog):
                return {"knowledge_guided_used": False}
            scores = {state["channel_names"][i]: float(value) for i, value in enumerate(state["variable_contributions"])}
            out = knowledge_guided_root_cause_decision(
                state["root_cause_ranking"], catalog,
                shift_scores=state["shift_scores"], contribution_scores=scores,
                onset_order=state["onset_order"], fault_type_scores=state["fault_type_scores"],
            )
            accepted = out.get("root_cause") is not None and float(out["confidence"]) >= float(cfg.get("diagnosis_threshold", 0.35)) and out.get("abstain_reason") is None
            diagnosis = {
                "fault_label": out["fault_label"] if accepted else None,
                "root_cause": out["root_cause"] if accepted else None,
                "confidence": float(out["confidence"]),
                "abstain_reason": None if accepted else (out.get("abstain_reason") or "knowledge_guided_evidence_below_threshold"),
                "affected_variables": sorted({v for path in state["propagation_paths"] for v in path if v != out.get("root_cause")}),
                "propagation_paths": state["propagation_paths"],
            }
            return {"knowledge_guided_used": True, "knowledge_guided_result": out, "diagnosis": diagnosis}

        def diagnose_step(state):
            if _inactive(state): return {"diagnosis": {}}
            if state.get("knowledge_guided_used"):
                out = dict(state.get("diagnosis") or {})
            else:
                out = process_diagnosis(
                    state["root_cause_ranking"], state["propagation_paths"],
                    fault_catalog=state.get("fault_catalog"),
                    confidence_threshold=float(cfg.get("diagnosis_threshold", 0.35)),
                )
            supervised = state.get("supervised_classification")
            if supervised:
                if supervised.get("abstained"):
                    out["fault_label"] = None
                    out["abstain_reason"] = supervised.get("abstain_reason") or "ambiguous_supervised_fault_classification"
                elif supervised.get("predicted_fault_id") is not None:
                    out["fault_label"] = f"TEP fault {int(supervised['predicted_fault_id'])}"
                    out["confidence"] = float(supervised["confidence"])
            return {"diagnosis": out}

        steps = (
            Step("standardize_against_normal", standardize, version="baseline-v1"),
            Step("hybrid_process_monitoring", monitor, depends_on=("standardize_against_normal",), version="dpca-cva-v1"),
            Step("supervised_fault_classification", classify, depends_on=("hybrid_process_monitoring",), version="cva-classifier-v1"),
            Step("contribution_analysis", contributions, depends_on=("supervised_fault_classification",), version="cva-contribution-v1"),
            Step("pre_post_shift_evidence", shift, depends_on=("contribution_analysis",), version="baseline-v1"),
            Step("temporal_fault_type_evidence", type_evidence, depends_on=("pre_post_shift_evidence",), version="baseline-v1"),
            Step("stationarity_analysis", stationarity, depends_on=("temporal_fault_type_evidence",), version="baseline-v1"),
            Step("granger_causality", granger, depends_on=("stationarity_analysis",), version="baseline-v1"),
            Step("causal_graph_filter", causal_filter, depends_on=("granger_causality",), version="baseline-v1"),
            Step("fault_onset_timing", onset, depends_on=("causal_graph_filter",), version="baseline-v1"),
            Step("root_cause_rank_enhanced", rank, depends_on=("fault_onset_timing",), version="baseline-v1"),
            Step("knowledge_guided_root_cause_decision", knowledge, depends_on=("root_cause_rank_enhanced",), version="baseline-v1"),
            Step("process_diagnosis", diagnose_step, depends_on=("knowledge_guided_root_cause_decision",), version="baseline-v1"),
        )
        return Workflow(steps, version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> ProcessDecisionPolicy:
        return ProcessDecisionPolicy(request)
