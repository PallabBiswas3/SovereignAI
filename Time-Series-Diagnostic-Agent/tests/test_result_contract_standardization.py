import numpy as np
import pytest

from tsdiag.models import DetectionResult, DiagnosticResult, Evidence, ToolTraceStep
from tsdiag.result_contract import result_summary, standardize_result


def test_standardizer_adds_common_result_envelope():
    result = DiagnosticResult(
        domain="bearing",
        task="fault_diagnosis",
        decision="monitor",
        detection=DetectionResult(abnormal=False, score=0.1, method="baseline"),
        evidence=[Evidence(source="signal", statement="Healthy baseline evidence.", score=0.8)],
        confidence=0.8,
        tool_trace=[ToolTraceStep("signal_integrity")],
    )

    standardized = standardize_result(result)

    assert standardized.uncertainty == pytest.approx(0.2)
    assert standardized.evidence[0].evidence_id == "bearing-evidence-1"
    assert standardized.evidence[0].provenance["domain"] == "bearing"
    assert standardized.metadata["contract"]["name"] == "DiagnosticResult"
    assert standardized.metadata["pipeline_version"] == "unknown"
    assert standardized.metadata["outcome"]["decision"] == "monitor"
    assert standardized.metadata["execution_summary"]["executed_tools"] == ["signal_integrity"]
    assert standardized.metadata["evidence_summary"]["count"] == 1


def test_standardizer_makes_abstention_explicit():
    result = DiagnosticResult(
        domain="transformer",
        task="fault_diagnosis",
        decision="abstain",
        detection=DetectionResult(abnormal=True, score=0.6, method="impulsiveness"),
        confidence=0.4,
        abstained=False,
        abstain_reason="A validated classifier is required.",
    )

    standardized = standardize_result(result)

    assert standardized.abstained is True
    assert standardized.metadata["outcome"]["abstained"] is True
    assert standardized.metadata["outcome"]["abstain_reason"] == "A validated classifier is required."


def test_compact_result_summary_is_domain_agnostic():
    result = DiagnosticResult(
        domain="battery",
        task="anomaly_localization",
        decision="diagnose",
        detection=DetectionResult(abnormal=True, score=np.float64(0.7), method="cell_deviation"),
        confidence=0.7,
    )

    summary = result_summary(result)

    assert summary["domain"] == "battery"
    assert summary["decision"] == "diagnose"
    assert summary["abnormal"] is True
    assert summary["evidence_count"] == 0


def test_standardizer_requires_outer_boundary_for_real_pipeline_version():
    result = DiagnosticResult(
        domain="bearing",
        task="condition_monitoring",
        decision="monitor",
        detection=DetectionResult(abnormal=False, method="fixture"),
    )

    direct = standardize_result(result)
    assert direct.metadata["pipeline_version"] == "unknown"

    explicit = standardize_result(result, pipeline_version="test-pipeline")
    assert explicit.metadata["pipeline_version"] == "test-pipeline"
