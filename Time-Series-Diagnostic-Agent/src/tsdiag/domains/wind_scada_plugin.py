from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..contracts import DiagnosticRequest
from ..detectors.residual_changepoint import (
    DEFAULT_CUSUM_DRIFT,
    DEFAULT_CUSUM_HOLD_SAMPLES,
    DEFAULT_CUSUM_THRESHOLD,
)
from ..evaluation.wind_scada import wind_event_evidence_decision
from ..evaluation.wind_calibration import load_wind_calibrated_config
from ..execution import Step, Workflow
from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    VerificationResult,
)
from ..result_contract import standardize_result
from .wind_scada_runner import WindScadaDiagnosticPipeline


WIND_POLICY_VERSION = "wind-care-event-policy:1.0"


def normalize_wind_inputs(inputs: Mapping[str, Any]) -> dict[str, Any]:
    """Return the canonical Wind request schema while accepting v1 aliases.

    Canonical callers provide ``train_matrix`` and ``prediction_matrix``.
    ``normal_reference``/``signal_matrix`` and ``timestamps`` remain accepted
    as compatibility aliases until the next major public API version.
    """
    values = dict(inputs)
    aliases = (
        ("train_matrix", "normal_reference"),
        ("prediction_matrix", "signal_matrix"),
        ("prediction_timestamps", "timestamps"),
    )
    for canonical, alias in aliases:
        canonical_value = values.get(canonical)
        alias_value = values.get(alias)
        if canonical_value is None and alias_value is not None:
            values[canonical] = alias_value
        elif canonical_value is not None and alias_value is not None:
            raise ValueError(
                f"Wind SCADA received both {canonical!r} and its deprecated alias {alias!r}; "
                f"provide only {canonical!r}"
            )
        if values.get(canonical) is not None:
            values.pop(alias, None)
    return values


@dataclass(frozen=True)
class WindScadaDecisionPolicy:
    version: str = WIND_POLICY_VERSION

    def decide(self, execution: Mapping[str, Any], trace) -> DiagnosticResult:
        raw = execution["wind_analysis"]
        decision = execution["wind_event_decision"]
        calibration = execution.get("wind_calibration")
        artifacts = raw.artifacts
        physics = artifacts.get("physics_consistency", {})
        physics_findings = list(physics.get("verification_findings", []))

        mapped_decision = {
            "fault": "diagnose",
            "monitor": "monitor",
            "abstain": "abstain",
        }[decision.decision]
        confidence = float(np.clip(decision.confidence, 0.0, 1.0))
        evidence_score = float(np.clip(decision.evidence_score, 0.0, 1.0))

        evidence = Evidence(
            source="wind_event_evidence",
            statement=decision.reason,
            score=evidence_score,
            details={
                "residual_alarm_fraction": decision.residual_alarm_fraction,
                "drift_alarm_fraction": decision.drift_alarm_fraction,
                "corroborated_alarm_fraction": decision.corroborated_alarm_fraction,
                "longest_corroborated_run": decision.longest_corroborated_run,
                "out_of_distribution_fraction": decision.out_of_distribution_fraction,
                "physics_support": decision.physics_support,
            },
            evidence_id="wind-event-evidence",
            kind="statistical",
            provenance={"producer_step": "event_decision", "policy_version": self.version},
        )
        trace.require("event_decision").evidence_ids.append(evidence.evidence_id)

        verification_status = "SUPPORTED" if decision.event_detected and physics_findings else "INSUFFICIENT"
        verification = [
            VerificationResult(
                hypothesis="persistent_scada_anomaly",
                status=verification_status,
                reason=(
                    f"Physics layer produced {len(physics_findings)} finding(s)."
                    if physics_findings
                    else "No independent physics finding was available to corroborate the event decision."
                ),
                evidence_ids=[evidence.evidence_id],
                verifier="wind_physics_consistency",
                details={"finding_count": len(physics_findings)},
            )
        ]

        affected = list(raw.affected_channels)
        hypothesis = [
            DiagnosticHypothesis(
                label="persistent_scada_anomaly",
                score=confidence,
                rationale=decision.reason,
                evidence_ids=[evidence.evidence_id],
            )
        ] if mapped_decision == "diagnose" else []

        result = DiagnosticResult(
            domain="wind_scada",
            task="condition_monitoring",
            decision=mapped_decision,
            detection=DetectionResult(
                abnormal=True if mapped_decision == "diagnose" else (False if mapped_decision == "monitor" else None),
                score=evidence_score,
                method="care_event_evidence_policy",
                details={"decision_reason": decision.reason},
            ),
            localization=LocalizationResult(
                channels=affected if mapped_decision == "diagnose" else [],
                scores={name: confidence for name in affected} if mapped_decision == "diagnose" else {},
            ),
            hypotheses=hypothesis,
            evidence=[evidence],
            verification=verification,
            confidence=confidence,
            uncertainty=None,
            abstained=mapped_decision == "abstain",
            abstain_reason=decision.reason if mapped_decision == "abstain" else None,
            recommended_actions=(
                ["Inspect localized channels, operating regime support, and physics findings."]
                if mapped_decision == "diagnose"
                else (["Do not issue a fault diagnosis until operation returns inside validated regime support."] if mapped_decision == "abstain" else [])
            ),
            tool_trace=trace,
            metadata={
                "workflow_version": WindScadaPlugin.workflow_version,
                "policy_version": self.version,
                "model_version": "wind-nbm-regime-v1",
                "allow_confidence_complement_uncertainty": False,
                "wind_artifacts": artifacts,
                "alarm_mask": raw.alarm_mask,
                "anomaly_scores": raw.anomaly_scores,
                "affected_channels": affected,
                "event_decision": {
                    "decision": decision.decision,
                    "event_detected": decision.event_detected,
                    "evidence_score": decision.evidence_score,
                    "confidence": decision.confidence,
                    "reason": decision.reason,
                    "first_decision_index": decision.first_decision_index,
                },
                "wind_calibration": calibration,
                "uncertainty": {
                    "value": None,
                    "method": "healthy_target_event_far" if calibration else "not_calibrated",
                    "calibrated": bool(calibration),
                    "calibration_dataset": (
                        calibration.get("development_pool_id") if calibration else None
                    ),
                },
            },
        )
        return standardize_result(result)


class WindScadaPlugin:
    name = "wind_scada"
    workflow_version = "1.1"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = normalize_wind_inputs(request.inputs)
        required = ("train_matrix", "prediction_matrix", "channel_names")
        missing = [name for name in required if values.get(name) is None]
        if missing:
            raise ValueError(f"Missing required Wind SCADA inputs: {', '.join(missing)}")
        train = np.asarray(values["train_matrix"], dtype=float)
        pred = np.asarray(values["prediction_matrix"], dtype=float)
        if train.ndim != 2 or pred.ndim != 2 or train.shape[1] != pred.shape[1]:
            raise ValueError("train_matrix and prediction_matrix must be aligned 2-D arrays")
        if len(values["channel_names"]) != train.shape[1]:
            raise ValueError("channel_names length mismatch")
        values["train_matrix"] = train
        values["prediction_matrix"] = pred
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        calibration = load_wind_calibrated_config()
        config = {} if calibration is None else dict(calibration["config"])
        config.update(request.inputs)

        def run_analysis(state: dict[str, Any]) -> dict[str, Any]:
            pipeline = WindScadaDiagnosticPipeline(
                n_regimes=int(config.get("n_regimes", 4)),
                residual_threshold=float(config.get("residual_threshold", 3.5)),
                persistence=int(config.get("persistence", 3)),
                cusum_drift=float(config.get("cusum_drift", DEFAULT_CUSUM_DRIFT)),
                cusum_threshold=float(config.get("cusum_threshold", DEFAULT_CUSUM_THRESHOLD)),
                cusum_hold_samples=int(config.get("cusum_hold_samples", DEFAULT_CUSUM_HOLD_SAMPLES)),
                random_state=int(config.get("random_state", 0)),
            )
            raw = pipeline.run(
                state["train_matrix"],
                state["prediction_matrix"],
                state["channel_names"],
                train_timestamps=state.get("train_timestamps"),
                prediction_timestamps=state.get("prediction_timestamps"),
                healthy_train_mask=state.get("healthy_train_mask"),
                operating_regime_labels=state.get("operating_regime_labels"),
                target_indices=state.get("target_indices"),
            )
            return {
                "wind_analysis": raw,
                "wind_calibration": None if calibration is None else {
                    "development_pool_id": calibration["development_pool_id"],
                    "target_event_far": calibration["target_event_far"],
                    "achieved_event_far": calibration["achieved_event_far"],
                    "achieved_event_far_ci": calibration["achieved_event_far_ci"],
                    "n_events_used": calibration["n_events_used"],
                },
            }

        def event_decision(state: dict[str, Any]) -> dict[str, Any]:
            raw = state["wind_analysis"]
            artifacts = raw.artifacts
            anomaly = artifacts.get("anomaly_detection", {})
            cp = artifacts.get("residual_changepoint", {})
            regime = artifacts.get("regime_assignment", {})
            physics = artifacts.get("physics_consistency", {})
            residual = np.asarray(anomaly.get("alarm_mask", raw.alarm_mask), dtype=bool)
            drift = np.asarray(cp.get("alarm_mask", raw.alarm_mask), dtype=bool)
            ood = np.asarray(regime.get("out_of_distribution_mask", np.zeros(len(raw.alarm_mask), dtype=bool)), dtype=bool)
            decision = wind_event_evidence_decision(
                residual_alarm_mask=residual,
                drift_alarm_mask=drift,
                evaluable_mask=state.get("evaluable_mask"),
                out_of_distribution_mask=ood,
                physics_finding_count=len(physics.get("verification_findings", [])),
                minimum_corroborated_run=int(config.get("event_minimum_corroborated_run", 6)),
                minimum_residual_fraction=float(config.get("event_minimum_residual_fraction", 0.02)),
                minimum_drift_fraction=float(config.get("event_minimum_drift_fraction", 0.02)),
                ood_abstain_fraction=float(config.get("event_ood_abstain_fraction", 0.10)),
            )
            return {"wind_event_decision": decision}

        return Workflow(
            steps=(
                Step("wind_analysis", run_analysis, version="compat-rich-runner-v1"),
                Step("event_decision", event_decision, depends_on=("wind_analysis",), version="1.0"),
            ),
            version=self.workflow_version,
        )

    def policy(self, request: DiagnosticRequest) -> WindScadaDecisionPolicy:
        return WindScadaDecisionPolicy()
