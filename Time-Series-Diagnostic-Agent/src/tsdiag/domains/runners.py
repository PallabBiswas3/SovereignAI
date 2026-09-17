from __future__ import annotations

from typing import Any

import numpy as np

from ..execution import StepExecutor
from ..models import DetectionResult, DiagnosticHypothesis, DiagnosticResult, Evidence, LocalizationResult, PrognosisResult, UncertaintyEstimate, VerificationResult
from ..result_contract import standardize_result
from ..tools.transformer_rules import arbitrate_transformer_hybrid, transformer_rule_diagnosis
from .domain_steps import default_domain_tool_registry
from .verification import battery_physics_verification, transformer_physics_verification
from .wind_scada_runner import WindScadaDiagnosticPipeline


def _execute(domain: str, values: dict[str, Any]):
    executor = StepExecutor(domain, default_domain_tool_registry())
    from . import DOMAIN_PACKS
    for contract in DOMAIN_PACKS[domain].tools:
        executor.run(contract.name, values)
    return values, executor.trace


def _finish(result: DiagnosticResult):
    return standardize_result(result)


class BatteryDiagnosticPipeline:
    def run(self, cell_voltage, cell_temperature, cell_ids, timestamps, **context):
        state,trace=_execute("battery",{**context,"cell_voltage":cell_voltage,"cell_temperature":cell_temperature,"cell_ids":cell_ids,"timestamps":timestamps})
        confidence=state["confidence"]; abnormal=state["abnormal"]; top=state["top_cell"]; risk=state["failure_probability_by_horizon"]
        ev=Evidence("cell_deviation",f"Largest cell-to-pack deviation is {top} (score={state['top_score']:.2f}).",confidence,
                    {"cell_scores":dict(zip(state["cell_ids"],np.asarray(state["cell_fault_scores"]).tolist()))},"battery-cell-evidence","statistical")
        trace.require("cell_deviation_features").evidence_ids.append(ev.evidence_id)
        verification=[battery_physics_verification(state, ev.evidence_id)]
        prognosis_uncertainty=float(state["prognosis_uncertainty"])
        return _finish(DiagnosticResult(domain="battery",task="anomaly_localization",decision="diagnose" if abnormal else "monitor",
            detection=DetectionResult(abnormal,confidence,method="robust_cell_to_pack_deviation"),
            localization=LocalizationResult(components=[top] if abnormal else [],channels=[top] if abnormal else [],scores={top:confidence} if abnormal else {}),
            hypotheses=[DiagnosticHypothesis("cell_imbalance",confidence,evidence_ids=[ev.evidence_id])] if abnormal else [],evidence=[ev],verification=verification,
            prognosis=PrognosisResult(risk=risk,horizon=state["latent_health_state"].get("horizon"),uncertainty=prognosis_uncertainty),
            confidence=confidence,uncertainty=prognosis_uncertainty,uncertainty_estimate=UncertaintyEstimate(prognosis_uncertainty,"battery_prognostic_uncertainty",calibrated=False),recommended_actions=[f"Inspect {top} sensing and balance state."] if abnormal else [],tool_trace=trace,
            metadata={"ranked_cells":state["ranked_cells"],"window_metadata":state["window_metadata"],"allow_confidence_complement_uncertainty":False}))


class TransformerDiagnosticPipeline:
    def run(self, signal_matrix, sampling_rate_hz, sensor_positions, **context):
        state,trace=_execute("transformer",{**context,"signal_matrix":signal_matrix,"sampling_rate_hz":sampling_rate_hz,"sensor_positions":sensor_positions})
        rule_reference=context.get("transformer_rule_reference")
        rule_result=None
        if rule_reference is not None:
            rule_result=transformer_rule_diagnosis(
                signal_matrix, rule_reference,
                threshold=float(context.get("transformer_rule_threshold",3.0)),
                external_asymmetry_threshold=float(context.get("transformer_external_asymmetry_threshold",4.0)),
            )

        label=state["fault_label"]
        classifier_confidence=float(np.clip(state.get("model_confidence",0.0),0,1))
        classifier_threshold=float(context.get("classifier_diagnosis_threshold",0.5))
        classifier_positive=bool(label) and classifier_confidence>=classifier_threshold

        hybrid_enabled=bool(context.get("hybrid_arbitration", rule_reference is not None and context.get("trained_image_model") is not None))
        if hybrid_enabled:
            arbitration=arbitrate_transformer_hybrid(classifier_positive=classifier_positive,classifier_confidence=classifier_confidence,rule_result=rule_result)
            decision=arbitration["decision"]; confidence=float(arbitration["confidence"])
            reason=None if decision!="abstain" else arbitration["reason"]; abnormal=decision=="diagnose"
            if abnormal and not label: label="main_transformer_fault"
            method="hybrid_ml_physics_arbitration_v1"
        elif rule_result is not None:
            rule_positive=bool(rule_result["transformer_fault"]); external=bool(rule_result["external_fault_signature"])
            if rule_positive: abnormal=True; label="main_transformer_fault"; reason=None; confidence=float(rule_result["confidence"]); decision="diagnose"
            elif external: abnormal=False; reason=None; confidence=float(rule_result["confidence"]); decision="monitor"
            elif classifier_positive and context.get("allow_ml_fallback",False): abnormal=True; reason=None; confidence=classifier_confidence; decision="diagnose"
            else: abnormal=False; reason=None; confidence=float(rule_result["confidence"]); decision="monitor"
            arbitration={"reason":"rule_primary","verification":"INSUFFICIENT"}; method="deterministic_transformer_protection_rules_v2"
        else:
            physics_abnormal=bool(state["abnormal"]); abnormal=physics_abnormal or classifier_positive
            reason=None if classifier_positive else state["abstain_reason"]
            confidence=max(float(state["confidence"]),classifier_confidence if classifier_positive else 0.0)
            decision="abstain" if reason else ("diagnose" if abnormal and label else "monitor")
            arbitration={"reason":"legacy_pipeline","verification":"INSUFFICIENT"}; method="wavelet_multisensor_classifier_fusion"

        abstained=decision=="abstain"
        weights=dict(zip(state["sensor_positions"],np.asarray(state["sensor_weights"]).tolist())); top=max(weights,key=weights.get)
        evidence_payload={"sensor_weights":weights,"feature_image_shape":state["representation_metadata"]["shape"],"classifier_confidence":classifier_confidence,"classifier_positive":classifier_positive,"arbitration":arbitration}
        if rule_result is not None:
            evidence_payload["electrical_rules"]=rule_result
            summary="; ".join(rule_result["reasons"][:3]) or "No strong electrical rule violation."
            ev=Evidence("transformer_electrical_rules",summary,confidence,evidence_payload,"transformer-rule-evidence","physics")
        else:
            ev=Evidence("multisensor_fusion",f"Fused-waveform kurtosis={state['harmonic_structure']['kurtosis']:.2f}; classifier label={label!r}.",confidence,evidence_payload,"transformer-fusion-evidence","signal")
        trace.require("spectral_correlation_representation").evidence_ids.append(ev.evidence_id)

        verification=[transformer_physics_verification(state, ev.evidence_id)]
        if hybrid_enabled:
            verification.append(VerificationResult("main_transformer_fault",arbitration["verification"],f"Hybrid arbiter: {arbitration['reason']}.",[ev.evidence_id],"ml_physics_arbitration"))

        raw_score=float(rule_result["rule_score"] if rule_result is not None else max(state["harmonic_structure"]["anomaly_score"],classifier_confidence if classifier_positive else 0.0))
        detection_score=float(np.clip(confidence if hybrid_enabled else raw_score,0,1))
        actions=[]
        if decision=="diagnose": actions=["Inspect transformer electrical protection quantities and corroborating measurements."]
        elif decision=="abstain": actions=["Escalate because classifier and electrical evidence disagree; obtain additional transformer-side measurements before acting."]

        return _finish(DiagnosticResult(domain="transformer",task="fault_diagnosis",decision=decision,
            detection=DetectionResult(abnormal,detection_score,method=method,details={"raw_rule_score":raw_score,"rule_result":rule_result,"classifier_positive":classifier_positive,"arbitration":arbitration}),
            localization=LocalizationResult(channels=[top],scores=weights),
            hypotheses=[DiagnosticHypothesis(str(label),confidence,evidence_ids=[ev.evidence_id])] if decision=="diagnose" and label else [],evidence=[ev],verification=verification,
            confidence=confidence,uncertainty=None,uncertainty_estimate=UncertaintyEstimate(None,"not_calibrated_transformer_baseline"),abstained=abstained,abstain_reason=reason,
            recommended_actions=actions,tool_trace=trace,
            metadata={"sensor_weights":weights,"feature_image_shape":state["representation_metadata"]["shape"],"classifier_confidence":classifier_confidence,"electrical_rule_result":rule_result,"arbitration":arbitration,"allow_confidence_complement_uncertainty":False}))
