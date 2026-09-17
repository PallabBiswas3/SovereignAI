import pytest
import json
import numpy as np

from tsdiag.models import (
    AgentResult,
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticReport,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    ToolTraceStep,
    VerificationResult,
)


def test_cross_domain_diagnostic_contract_round_trips_to_dict():
    evidence = Evidence(
        source="welch_psd",
        statement="BPFO harmonics are elevated above the healthy baseline.",
        score=0.91,
        evidence_id="ev-1",
        kind="signal",
        provenance={"channel": "DE", "window": 12},
    )
    result = DiagnosticResult(
        domain="bearing",
        task="fault_diagnosis",
        decision="diagnose",
        detection=DetectionResult(abnormal=True, score=0.94, onset=120, method="envelope_spectrum"),
        localization=LocalizationResult(components=["outer_race"], channels=["DE"]),
        hypotheses=[
            DiagnosticHypothesis(
                label="outer_race_fault",
                score=0.89,
                evidence_ids=["ev-1"],
            )
        ],
        evidence=[evidence],
        verification=[
            VerificationResult(
                hypothesis="outer_race_fault",
                status="SUPPORTED",
                reason="Fault-frequency evidence is consistent with BPFO.",
                evidence_ids=["ev-1"],
                verifier="bearing_physics",
            )
        ],
        confidence=0.88,
        uncertainty=0.12,
        recommended_actions=["Inspect the outer race at the next maintenance window."],
        tool_trace=[ToolTraceStep(tool="welch_psd", evidence_ids=["ev-1"])],
    )

    payload = result.to_dict()

    assert payload["schema_version"] == "1.0"
    assert payload["domain"] == "bearing"
    assert payload["detection"]["abnormal"] is True
    assert payload["hypotheses"][0]["label"] == "outer_race_fault"
    assert payload["verification"][0]["status"] == "SUPPORTED"


def test_contract_requires_explicit_abstention_reason():
    result = DiagnosticResult(
        domain="wind_scada",
        task="condition_monitoring",
        decision="abstain",
        detection=DetectionResult(abnormal=None),
        confidence=0.2,
        uncertainty=0.9,
        abstained=True,
    )

    with pytest.raises(ValueError, match="abstain_reason"):
        result.validate()


def test_contract_rejects_unknown_evidence_references():
    result = DiagnosticResult(
        domain="battery",
        task="cell_localization",
        decision="diagnose",
        detection=DetectionResult(abnormal=True, score=0.8),
        hypotheses=[
            DiagnosticHypothesis(
                label="cell_17_degradation",
                score=0.7,
                evidence_ids=["missing-evidence"],
            )
        ],
        confidence=0.7,
    )

    with pytest.raises(ValueError, match="unknown evidence ids"):
        result.validate()


def test_legacy_report_adapter_preserves_evidence_and_recommendations():
    legacy = DiagnosticReport(
        decision="diagnose",
        label="bearing_fault",
        confidence=0.82,
        agent_results=[
            AgentResult(
                agent="BearingDiagnosticAgent",
                status="warning",
                summary="Bearing fault evidence detected.",
                confidence=0.82,
                recommendations=["Inspect bearing."],
            )
        ],
        evidence=[
            Evidence(
                source="envelope_spectrum",
                statement="BPFO peak detected.",
                score=0.9,
            )
        ],
        trace=["signal_processing", "bearing_diagnostic"],
    )

    migrated = DiagnosticResult.from_legacy_report(legacy, domain="bearing")

    assert migrated.decision == "diagnose"
    assert migrated.hypotheses[0].label == "bearing_fault"
    assert migrated.evidence[0].evidence_id == "legacy-evidence-0"
    assert migrated.hypotheses[0].evidence_ids == ["legacy-evidence-0"]
    assert migrated.recommended_actions == ["Inspect bearing."]
    assert [step.tool for step in migrated.tool_trace] == [
        "signal_processing",
        "bearing_diagnostic",
    ]


def test_result_serialization_converts_numpy_values():
    result = DiagnosticResult(
        domain="process", task="condition_monitoring", decision="monitor",
        detection=DetectionResult(abnormal=False, score=np.float64(0.1)),
        confidence=np.float64(0.8), metadata={"array": np.array([1.0, 2.0])},
    )
    payload = json.loads(result.to_json())
    assert payload["metadata"]["array"] == [1.0, 2.0]
