from __future__ import annotations

from typing import Any

from .models import Abstention, DiagnosticResult, RunProvenance, UncertaintyEstimate


RESULT_CONTRACT_VERSION = "1.1"
UNSPECIFIED_PIPELINE_VERSION = "unknown"


def _flatten_trace(steps):
    rows = []
    def visit(step):
        rows.append(step)
        for child in getattr(step, "children", []):
            visit(child)
    for step in steps:
        visit(step)
    return rows


def standardize_result(
    result: DiagnosticResult,
    *,
    pipeline_version: str = UNSPECIFIED_PIPELINE_VERSION,
) -> DiagnosticResult:
    result.schema_version = RESULT_CONTRACT_VERSION

    legacy_uncertainty_fallback = bool(result.metadata.get("allow_confidence_complement_uncertainty", True))
    if result.uncertainty is None and legacy_uncertainty_fallback:
        result.uncertainty = max(0.0, min(1.0, 1.0 - float(result.confidence)))

    if result.uncertainty_estimate is None:
        method = "confidence_complement_legacy" if legacy_uncertainty_fallback and result.uncertainty is not None else "not_calibrated"
        result.uncertainty_estimate = UncertaintyEstimate(
            value=None if result.uncertainty is None else float(result.uncertainty),
            method=method,
            calibrated=False,
            calibration_dataset=None,
        )

    if result.decision == "abstain":
        result.abstained = True
    elif result.abstained:
        result.decision = "abstain"
    result.abstention = Abstention(
        abstained=bool(result.abstained),
        reason=result.abstain_reason,
        trigger=result.metadata.get("abstention_trigger"),
    )

    for index, evidence in enumerate(result.evidence):
        if not evidence.evidence_id:
            evidence.evidence_id = f"{result.domain}-evidence-{index + 1}"
        evidence.provenance.setdefault("domain", result.domain)
        evidence.provenance.setdefault("task", result.task)
        evidence.provenance.setdefault("source", evidence.source)

    trace_rows = _flatten_trace(result.tool_trace)
    executed_tools = [step.tool for step in trace_rows if step.status == "ok"]
    failed_tools = [step.tool for step in trace_rows if step.status == "error"]
    skipped_tools = [step.tool for step in trace_rows if step.status == "skipped"]

    # The outer orchestration boundary is authoritative. Nested/legacy runners
    # may standardize results earlier, but they must not freeze an older public
    # pipeline version into the final envelope.
    result.metadata["pipeline_version"] = pipeline_version
    if result.provenance is None:
        result.provenance = RunProvenance(
            workflow_version=result.metadata.get("workflow_version"),
            policy_version=result.metadata.get("policy_version"),
            model_versions=dict(result.metadata.get("resolved_model_versions", {})),
        )

    result.metadata["contract"] = {"name": "DiagnosticResult", "schema_version": RESULT_CONTRACT_VERSION}
    result.metadata["outcome"] = {
        "decision": result.decision,
        "abstained": bool(result.abstained),
        "abstain_reason": result.abstain_reason,
        "confidence": float(result.confidence),
        "uncertainty": None if result.uncertainty is None else float(result.uncertainty),
        "uncertainty_method": result.uncertainty_estimate.method,
    }
    result.metadata["evidence_summary"] = {
        "count": len(result.evidence),
        "kinds": sorted({row.kind for row in result.evidence}),
        "sources": sorted({row.source for row in result.evidence}),
        "verification_count": len(result.verification),
    }
    result.metadata["execution_summary"] = {
        "executed_tools": executed_tools,
        "failed_tools": failed_tools,
        "skipped_tools": skipped_tools,
        "tool_count": len(trace_rows),
        "top_level_step_count": len(result.tool_trace),
    }
    result.metadata["result_summary"] = {
        "abnormal": result.detection.abnormal,
        "detection_method": result.detection.method,
        "top_hypothesis": result.hypotheses[0].label if result.hypotheses else None,
        "localized_components": list(result.localization.components),
        "localized_channels": list(result.localization.channels),
        "has_prognosis": result.prognosis is not None,
    }

    result.validate()
    return result


def result_summary(result: DiagnosticResult) -> dict[str, Any]:
    standardized = standardize_result(
        result,
        pipeline_version=result.metadata.get("pipeline_version", UNSPECIFIED_PIPELINE_VERSION),
    )
    return {
        "domain": standardized.domain,
        "task": standardized.task,
        "decision": standardized.decision,
        "abnormal": standardized.detection.abnormal,
        "label": standardized.hypotheses[0].label if standardized.hypotheses else None,
        "confidence": float(standardized.confidence),
        "uncertainty": standardized.uncertainty,
        "uncertainty_method": standardized.uncertainty_estimate.method if standardized.uncertainty_estimate else None,
        "abstained": standardized.abstained,
        "abstain_reason": standardized.abstain_reason,
        "components": list(standardized.localization.components),
        "channels": list(standardized.localization.channels),
        "evidence_count": len(standardized.evidence),
        "verification_count": len(standardized.verification),
        "tool_count": len(_flatten_trace(standardized.tool_trace)),
        "prognosis": None if standardized.prognosis is None else standardized.prognosis.details,
    }
