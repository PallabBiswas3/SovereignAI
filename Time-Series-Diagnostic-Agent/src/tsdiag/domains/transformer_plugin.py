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
    VerificationResult,
)
from ..result_contract import standardize_result
from ..tools.transformer_rules import arbitrate_transformer_hybrid, transformer_rule_diagnosis
from .domain_steps import default_domain_tool_registry
from .verification import transformer_physics_verification


def _tool(name: str, state: dict[str, Any]) -> dict[str, Any]:
    return default_domain_tool_registry().get("transformer", name)(state)


@dataclass(frozen=True)
class TransformerDecisionPolicy:
    request: DiagnosticRequest
    version: str = "transformer-policy-v2"

    def decide(self, execution: Mapping[str, Any], trace: ExecutionTrace) -> DiagnosticResult:
        state = dict(execution)
        context = dict(self.request.inputs)
        signal_matrix = np.asarray(state["signal_matrix"], dtype=float)
        rule_reference = context.get("transformer_rule_reference")
        rule_result = None
        if rule_reference is not None:
            rule_result = transformer_rule_diagnosis(
                signal_matrix,
                rule_reference,
                threshold=float(context.get("transformer_rule_threshold", 3.0)),
                external_asymmetry_threshold=float(context.get("transformer_external_asymmetry_threshold", 4.0)),
            )

        label = state.get("fault_label")
        classifier_confidence = float(np.clip(state.get("model_confidence", 0.0), 0.0, 1.0))
        classifier_threshold = float(context.get("classifier_diagnosis_threshold", 0.5))
        classifier_positive = bool(label) and classifier_confidence >= classifier_threshold

        hybrid_enabled = bool(context.get(
            "hybrid_arbitration",
            rule_reference is not None and context.get("trained_image_model") is not None,
        ))
        if hybrid_enabled:
            arbitration = arbitrate_transformer_hybrid(
                classifier_positive=classifier_positive,
                classifier_confidence=classifier_confidence,
                rule_result=rule_result,
            )
            decision = arbitration["decision"]
            confidence = float(arbitration["confidence"])
            reason = None if decision != "abstain" else arbitration["reason"]
            abnormal = decision == "diagnose"
            if abnormal and not label:
                label = "main_transformer_fault"
            method = "hybrid_ml_physics_arbitration_v1"
        elif rule_result is not None:
            rule_positive = bool(rule_result["transformer_fault"])
            external = bool(rule_result["external_fault_signature"])
            if rule_positive:
                abnormal = True; label = "main_transformer_fault"; reason = None
                confidence = float(rule_result["confidence"]); decision = "diagnose"
            elif external:
                abnormal = False; reason = None
                confidence = float(rule_result["confidence"]); decision = "monitor"
            elif classifier_positive and context.get("allow_ml_fallback", False):
                abnormal = True; reason = None; confidence = classifier_confidence; decision = "diagnose"
            else:
                abnormal = False; reason = None
                confidence = float(rule_result["confidence"]); decision = "monitor"
            arbitration = {"reason": "rule_primary", "verification": "INSUFFICIENT"}
            method = "deterministic_transformer_protection_rules_v2"
        else:
            physics_abnormal = bool(state["abnormal"])
            abnormal = physics_abnormal or classifier_positive
            reason = None if classifier_positive else state.get("abstain_reason")
            confidence = max(float(state["confidence"]), classifier_confidence if classifier_positive else 0.0)
            decision = "abstain" if reason else ("diagnose" if abnormal and label else "monitor")
            arbitration = {"reason": "legacy_pipeline", "verification": "INSUFFICIENT"}
            method = "wavelet_multisensor_classifier_fusion"

        weights = dict(zip(state["sensor_positions"], np.asarray(state["sensor_weights"]).tolist()))
        top = max(weights, key=weights.get)
        evidence_payload = {
            "sensor_weights": weights,
            "feature_image_shape": state["representation_metadata"]["shape"],
            "classifier_confidence": classifier_confidence,
            "classifier_positive": classifier_positive,
            "arbitration": arbitration,
            "predicted_class": state.get("predicted_class"),
            "model_family": state.get("classifier_model_family"),
        }
        if rule_result is not None:
            evidence_payload["electrical_rules"] = rule_result
            summary = "; ".join(rule_result["reasons"][:3]) or "No strong electrical rule violation."
            evidence = Evidence(
                "transformer_electrical_rules", summary, confidence, evidence_payload,
                "transformer-rule-evidence", "physics",
            )
        else:
            evidence = Evidence(
                "multisensor_fusion",
                f"Fused-waveform kurtosis={state['harmonic_structure']['kurtosis']:.2f}; classifier label={label!r}.",
                confidence, evidence_payload, "transformer-fusion-evidence", "signal",
            )
        trace.require("spectral_correlation_representation").evidence_ids.append(evidence.evidence_id)

        verification = [transformer_physics_verification(state, evidence.evidence_id)]
        if hybrid_enabled:
            verification.append(VerificationResult(
                "main_transformer_fault", arbitration["verification"],
                f"Hybrid arbiter: {arbitration['reason']}.", [evidence.evidence_id], "ml_physics_arbitration",
            ))

        raw_score = float(
            rule_result["rule_score"] if rule_result is not None
            else max(state["harmonic_structure"]["anomaly_score"], classifier_confidence if classifier_positive else 0.0)
        )
        detection_score = float(np.clip(confidence if hybrid_enabled else raw_score, 0.0, 1.0))
        actions = []
        if decision == "diagnose":
            actions = ["Inspect transformer electrical protection quantities and corroborating measurements."]
        elif decision == "abstain":
            actions = ["Obtain additional transformer-side measurements before acting on conflicting or incomplete evidence."]

        return standardize_result(DiagnosticResult(
            domain="transformer",
            task="fault_diagnosis",
            decision=decision,
            detection=DetectionResult(
                abnormal, detection_score, method=method,
                details={"raw_rule_score": raw_score, "rule_result": rule_result, "classifier_positive": classifier_positive, "arbitration": arbitration},
            ),
            localization=LocalizationResult(channels=[top], scores=weights),
            hypotheses=[DiagnosticHypothesis(str(label), confidence, evidence_ids=[evidence.evidence_id])] if decision == "diagnose" and label else [],
            evidence=[evidence],
            verification=verification,
            confidence=confidence,
            uncertainty=None,
            uncertainty_estimate=UncertaintyEstimate(None, "not_calibrated_transformer_baseline"),
            abstained=decision == "abstain",
            abstain_reason=reason,
            recommended_actions=actions,
            tool_trace=trace,
            metadata={
                "sensor_weights": weights,
                "feature_image_shape": state["representation_metadata"]["shape"],
                "classifier_confidence": classifier_confidence,
                "predicted_class": state.get("predicted_class"),
                "model_family": state.get("classifier_model_family"),
                "electrical_rule_result": rule_result,
                "arbitration": arbitration,
                "workflow_version": TransformerPlugin.workflow_version,
                "policy_version": self.version,
                "allow_confidence_complement_uncertainty": False,
            },
        ))


class TransformerPlugin:
    name = "transformer"
    workflow_version = "2.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        required = ("signal_matrix", "sampling_rate_hz", "sensor_positions")
        missing = [key for key in required if values.get(key) is None]
        if missing:
            raise ValueError(f"transformer requires {', '.join(missing)}")
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        step_ids = (
            "multisensor_sync_check",
            "wavelet_denoising",
            "cross_correlation_analysis",
            "correlation_sensor_weighting",
            "multisensor_fusion",
            "envelope_sanity_check",
            "fast_spectral_correlation",
            "spectral_correlation_representation",
            "transformer_fault_classifier",
            "transformer_decision",
        )
        steps = []
        previous = None
        for step_id in step_ids:
            def execute(state, name=step_id):
                return _tool(name, state)
            steps.append(Step(step_id, execute, depends_on=() if previous is None else (previous,), version="1.0"))
            previous = step_id
        return Workflow(tuple(steps), version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> TransformerDecisionPolicy:
        return TransformerDecisionPolicy(request)
