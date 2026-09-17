from __future__ import annotations

from typing import Any
from uuid import uuid4

from app.agent.state import AgentPlan, AgentRunState, AgentStep, RunStatus, StepStatus
from app.integrations.models import IntegratedAnalysisResponse
from app.integrations.routing import IntegrationRoutePlan


def missing_input_state(*, request: str, routing: Any, selection: Any, plan: IntegrationRoutePlan) -> AgentRunState:
    domain = plan.diagnostic_domain or "industrial"
    return AgentRunState(
        id=str(uuid4()), request=request, status=RunStatus.completed, routing=routing,
        plan=AgentPlan(goal=request, steps=[AgentStep(
            id=1, action="select_services", title="Select specialist services",
            status=StepStatus.completed, observation=plan.reason,
            verification="No diagnostic service was called without validated measurements.",
        )]),
        final_response=(
            f"I selected the {domain} diagnostic workflow, but it needs measured time-series data before it can run. "
            "Attach a JSON file containing `domain` and `inputs` in the Time-Series Diagnostic Agent request format. "
            "No diagnosis was fabricated and no plant action was taken."
        ),
        warnings=["Diagnostic execution was withheld because validated time-series input was not supplied."],
        requested_execution_mode=selection.requested.value,
        execution_mode=selection.selected.value,
        execution_mode_reason=selection.reason,
        runtime_metrics={"service_plan": plan.services, "route_confidence": plan.confidence},
        context_metrics={"integration_route": plan.model_dump(mode="json")},
    )


def integration_unavailable_state(
    *, request: str, routing: Any, selection: Any, plan: IntegrationRoutePlan, error: str,
) -> AgentRunState:
    return AgentRunState(
        id=str(uuid4()), request=request, status=RunStatus.completed, routing=routing,
        plan=AgentPlan(goal=request, steps=[
            AgentStep(
                id=1, action="select_services", title="Select specialist services",
                status=StepStatus.completed, observation=plan.reason,
                verification=f"Selected services: {', '.join(plan.services)}.",
            ),
            AgentStep(
                id=2, action="integration_fail_closed", title="Stop when a required service is unavailable",
                status=StepStatus.failed, error=error,
                verification="No unverified industrial answer was released.",
            ),
        ]),
        final_response=(
            "I selected the required specialist services, but one of them is currently unavailable. "
            "The workflow stopped safely, so I cannot release an evidence-backed answer yet. "
            "Check the service status in the integration health view and retry."
        ),
        warnings=[error, "The integration failed closed; no unsupported answer was generated."],
        requested_execution_mode=selection.requested.value,
        execution_mode=selection.selected.value,
        execution_mode_reason=selection.reason,
        runtime_metrics={
            "provider": "industrial-integration-orchestrator", "service_plan": plan.services,
            "route_confidence": plan.confidence, "integration_status": "unavailable",
        },
        context_metrics={
            "integration_route": plan.model_dump(mode="json"),
            "controlplane_status": "not_released",
        },
    )


def integrated_result_state(
    *, request: str, routing: Any, selection: Any, plan: IntegrationRoutePlan,
    result: IntegratedAnalysisResponse,
) -> AgentRunState:
    graph = result.graph_evidence or {}
    diagnostic = result.diagnostic or {}
    chunks = graph.get("chunks") or []
    graph_claims = graph.get("claims") or []
    diagnostic_evidence = diagnostic.get("evidence") or []
    hypotheses = diagnostic.get("hypotheses") or []

    sources = [_graph_source(chunk) for chunk in chunks]
    sources.extend(_diagnostic_source(item, diagnostic) for item in diagnostic_evidence)
    claims = [_graph_claim(claim, graph) for claim in graph_claims]
    claims.extend(_diagnostic_claim(item, diagnostic) for item in hypotheses)

    steps = [AgentStep(
        id=1, action="controlplane_precheck", title="Authorize the service plan",
        status=StepStatus.completed, observation=_decision(result.precheck),
        verification="ControlPlane evaluated the request before evidence services ran.",
    )]
    if plan.use_graph:
        steps.append(AgentStep(
            id=len(steps) + 1, action="graph_rag_retrieve", title="Retrieve graph-linked evidence",
            status=StepStatus.completed,
            observation=f"Graph-RAG returned {len(chunks)} source chunks and {len(graph_claims)} supported claims.",
            verification=str((graph.get("verification") or {}).get("summary") or graph.get("status") or "completed"),
        ))
    if plan.diagnostic is not None:
        steps.append(AgentStep(
            id=len(steps) + 1, action="diagnose_time_series", title=f"Run {plan.diagnostic.domain} diagnostics",
            status=StepStatus.completed,
            observation=f"Diagnostic decision: {diagnostic.get('decision', 'unknown')}.",
            verification=f"Confidence: {diagnostic.get('confidence', 'not reported')}.",
        ))
    steps.append(AgentStep(
        id=len(steps) + 1, action="controlplane_release", title="Verify and release the response",
        status=StepStatus.completed if result.released else StepStatus.waiting_for_approval,
        observation=_decision(result.controlplane or result.precheck),
        verification="The response was released only when the final ControlPlane decision allowed it.",
    ))

    warnings: list[str] = []
    if not result.released:
        warnings.append("ControlPlane withheld automatic release; human review is required.")
    if graph.get("status") == "abstained":
        warnings.append("Graph-RAG abstained because it found insufficient supported document evidence.")
    if diagnostic.get("abstained"):
        warnings.append(str(diagnostic.get("abstain_reason") or "The diagnostic service abstained."))

    return AgentRunState(
        id=str(uuid4()), request=request,
        status=RunStatus.completed if result.released else RunStatus.waiting_for_approval,
        routing=routing, plan=AgentPlan(goal=request, steps=steps),
        final_response=result.final_response, warnings=warnings, sources=sources, claims=claims,
        evidence_records=[*chunks, *diagnostic_evidence],
        requested_execution_mode=selection.requested.value,
        execution_mode=selection.selected.value,
        execution_mode_reason=selection.reason,
        runtime_metrics={
            "provider": "industrial-integration-orchestrator",
            "integration_run_id": result.run_id,
            "service_plan": plan.services,
            "service_status": result.service_status,
            "route_confidence": plan.confidence,
        },
        context_metrics={
            "integration_route": plan.model_dump(mode="json"),
            "raw_candidate_count": len(chunks) + len(diagnostic_evidence),
            "final_evidence_count": len(sources),
            "controlplane_status": result.status,
        },
    )


def _graph_source(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata") if isinstance(chunk.get("metadata"), dict) else {}
    return {
        "chunk_id": chunk.get("id"),
        "file": metadata.get("file") or metadata.get("sourceDocId") or metadata.get("source_doc_id") or "Graph-RAG knowledge base",
        "page": metadata.get("page") or metadata.get("page_start"),
        "section": metadata.get("section") or "Graph-linked document evidence",
        "text": chunk.get("content"), "score": chunk.get("similarity"), "source_system": "graph-rag",
    }


def _diagnostic_source(item: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    return {
        "chunk_id": item.get("evidence_id"), "file": f"Diagnostic Agent / {diagnostic.get('domain', 'industrial')}",
        "page": None, "section": item.get("kind") or item.get("source") or "Time-series evidence",
        "text": item.get("statement"), "score": item.get("score"),
        "source_system": "time-series-diagnostic-agent", "provenance": item.get("provenance") or {},
    }


def _graph_claim(claim: dict[str, Any], graph: dict[str, Any]) -> dict[str, Any]:
    claim_id = str(claim.get("id") or f"graph-claim-{uuid4()}")
    evidence_ids = claim.get("supporting_chunk_ids") or claim.get("chunk_ids") or []
    verification = graph.get("verification") or {}
    result = next((row for row in verification.get("results") or [] if row.get("claimId") == claim_id), {})
    return {
        "id": claim_id, "text": str(claim.get("claim_text") or ""), "claim_type": "document_evidence",
        "evidence_ids": evidence_ids, "calculation_ids": [], "support_status": result.get("label") or "SUPPORTED",
        "support_score": claim.get("extraction_confidence"),
        "verification": [{"verifier": "Graph-RAG evidence verifier", "passed": (result.get("label") or "SUPPORTED") == "SUPPORTED", "summary": result.get("reason") or verification.get("summary") or "Claim retained after graph evidence verification."}],
    }


def _diagnostic_claim(hypothesis: dict[str, Any], diagnostic: dict[str, Any]) -> dict[str, Any]:
    verifications = [row for row in diagnostic.get("verification") or [] if row.get("hypothesis") == hypothesis.get("label")]
    return {
        "id": f"diagnostic-{uuid4()}", "text": str(hypothesis.get("label") or "Diagnostic hypothesis"),
        "claim_type": "diagnostic_hypothesis", "evidence_ids": hypothesis.get("evidence_ids") or [],
        "calculation_ids": [], "support_status": "SUPPORTED" if verifications and all(row.get("status") == "SUPPORTED" for row in verifications) else "INSUFFICIENT",
        "support_score": hypothesis.get("score"),
        "verification": [{"verifier": row.get("verifier") or "Diagnostic verifier", "passed": row.get("status") == "SUPPORTED", "summary": row.get("reason") or str(row.get("status") or "Verification completed.")} for row in verifications],
    }


def _decision(report: dict[str, Any]) -> str:
    return str((report.get("decision") or {}).get("action") or "not reported")
