from __future__ import annotations

import asyncio
import json
from time import monotonic
from typing import Any
from uuid import uuid4

from app.core.config import Settings, get_settings

from .clients import ControlPlaneClient, DiagnosticAgentClient, GraphRagClient, IntegrationServiceError
from .models import IntegratedAnalysisRequest, IntegratedAnalysisResponse


HOLD_ACTIONS = {"block", "human_review"}
RELEASE_ACTIONS = {"allow", "allow_with_warning", "redact"}


class IndustrialIntegrationOrchestrator:
    def __init__(
        self,
        *,
        settings: Settings | None = None,
        graph: GraphRagClient | None = None,
        diagnostics: DiagnosticAgentClient | None = None,
        controlplane: ControlPlaneClient | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        client_options = {
            "timeout": self.settings.integration_timeout_seconds,
            "max_retries": getattr(self.settings, "integration_max_retries", 2),
            "retry_backoff_seconds": getattr(self.settings, "integration_retry_backoff_seconds", 0.25),
            "retry_backoff_max_seconds": getattr(self.settings, "integration_retry_backoff_max_seconds", 2.0),
        }
        self.graph = graph or GraphRagClient(self.settings.graphrag_url, **client_options)
        self.diagnostics = diagnostics or DiagnosticAgentClient(self.settings.diagnostics_url, **client_options)
        self.controlplane = controlplane or ControlPlaneClient(self.settings.controlplane_url, **client_options)

    async def health(self) -> dict[str, dict[str, Any]]:
        clients = {
            "graph-rag": self.graph,
            "time-series-diagnostic-agent": self.diagnostics,
            "controlplane": self.controlplane,
        }
        results = await asyncio.gather(*(client.health() for client in clients.values()), return_exceptions=True)
        return {
            name: ({"status": "unavailable", "detail": str(result)} if isinstance(result, Exception) else result)
            for name, result in zip(clients, results)
        }

    async def analyze(
        self,
        request: IntegratedAnalysisRequest,
        *,
        principal_id: str,
        organization_id: str,
    ) -> IntegratedAnalysisResponse:
        total_started = monotonic()
        run_id = f"integration-{uuid4()}"
        metadata = {
            "run_id": run_id,
            "host": "sovereign-ai",
            "principal_id": principal_id,
            "organization_id": organization_id,
        }
        precheck_started = monotonic()
        precheck = await self.controlplane.precheck({
            "id": f"{run_id}-precheck",
            "profile": request.policy_profile,
            "prompt": request.query,
            "consequential": request.consequential,
            "industry": "industrial",
            "metadata": metadata,
        })
        precheck_ms = (monotonic() - precheck_started) * 1000
        precheck_action = self._action(precheck)
        if precheck_action in HOLD_ACTIONS:
            return IntegratedAnalysisResponse(
                run_id=run_id,
                released=False,
                status=f"precheck_{precheck_action}",
                final_response=str(precheck.get("final_response") or "Request held by policy."),
                precheck=precheck,
                service_status={"controlplane": "available", "graph-rag": "not_called", "diagnostics": "not_called"},
                timings_ms={
                    "precheck": round(precheck_ms, 3),
                    "evidence": 0.0,
                    "release_check": 0.0,
                    "total": round((monotonic() - total_started) * 1000, 3),
                },
            )

        calls: list[tuple[str, Any]] = []
        if request.include_graph_evidence:
            calls.append(("graph-rag", self.graph.retrieve(request.query, request.assurance_level.value)))
        if request.diagnostic is not None:
            diagnostic_payload = request.diagnostic.model_dump(mode="python")
            context = dict(diagnostic_payload.get("run_context") or {})
            context_metadata = dict(context.get("metadata") or {})
            if "synthetic" in context:
                context_metadata["synthetic"] = bool(context.pop("synthetic"))
            context.update({"run_id": run_id, "source": "sovereign-ai"})
            if context_metadata:
                context["metadata"] = context_metadata
            diagnostic_payload["run_context"] = context
            calls.append(("diagnostics", self.diagnostics.diagnose(diagnostic_payload)))

        evidence_started = monotonic()
        call_results = await asyncio.gather(*(call for _, call in calls), return_exceptions=True)
        evidence_ms = (monotonic() - evidence_started) * 1000
        evidence: dict[str, dict[str, Any] | None] = {"graph-rag": None, "diagnostics": None}
        service_status = {"controlplane": "available", "graph-rag": "not_requested", "diagnostics": "not_requested"}
        failures: list[tuple[str, Exception]] = []
        for (name, _), result in zip(calls, call_results):
            if isinstance(result, Exception):
                service_status[name] = "unavailable"
                failures.append((name, result))
            else:
                service_status[name] = "available"
                evidence[name] = result

        if failures and self.settings.integration_fail_closed:
            if len(failures) == 1:
                name, failure = failures[0]
                if isinstance(failure, IntegrationServiceError):
                    raise failure
                raise IntegrationServiceError(name, f"{name} integration failed: {failure}") from failure
            failed_services = [name for name, _ in failures]
            detail = "; ".join(f"{name}: {failure}" for name, failure in failures)
            raise IntegrationServiceError(
                "multiple",
                detail,
                failed_services=failed_services,
                retryable=any(isinstance(failure, IntegrationServiceError) and failure.retryable for _, failure in failures),
            )

        candidate = request.candidate_response or self._compose_candidate(
            request.query,
            graph=evidence["graph-rag"],
            diagnostic=evidence["diagnostics"],
        )
        context = json.dumps(
            {"graph_evidence": evidence["graph-rag"], "diagnostic": evidence["diagnostics"]},
            ensure_ascii=True,
            default=str,
        )[:50000]
        grounding_evidence = self._graph_grounding_evidence(evidence["graph-rag"])
        release_started = monotonic()
        report = await self.controlplane.check({
            "id": f"{run_id}-release",
            "profile": request.policy_profile,
            "prompt": request.query,
            "response": candidate,
            "context": context,
            "grounding_evidence": grounding_evidence,
            "consequential": request.consequential,
            "industry": "industrial",
            "metadata": {
                **metadata,
                "grounding_evidence_count": len(grounding_evidence),
                "grounding_evidence_source": "graph-rag" if grounding_evidence else None,
            },
        })
        release_ms = (monotonic() - release_started) * 1000
        action = self._action(report)
        released = action in RELEASE_ACTIONS
        return IntegratedAnalysisResponse(
            run_id=run_id,
            released=released,
            status="released" if released else f"held_{action or 'unknown'}",
            final_response=str(report.get("final_response") or ""),
            precheck=precheck,
            graph_evidence=evidence["graph-rag"],
            diagnostic=evidence["diagnostics"],
            controlplane=report,
            service_status=service_status,
            timings_ms={
                "precheck": round(precheck_ms, 3),
                "evidence": round(evidence_ms, 3),
                "release_check": round(release_ms, 3),
                "total": round((monotonic() - total_started) * 1000, 3),
            },
        )

    @staticmethod
    def _graph_grounding_evidence(graph: dict[str, Any] | None) -> list[dict[str, Any]]:
        """Translate the existing GraphRAG response at the service boundary.

        Only provenance actually supplied by GraphRAG is copied. Missing IDs,
        page numbers, revisions or scores are deliberately omitted rather than
        invented. The original opaque context is still sent as a compatibility
        fallback while ControlPlane v3 consumes this typed evidence directly.
        """

        if not isinstance(graph, dict):
            return []
        output: list[dict[str, Any]] = []
        candidates: list[tuple[str, dict[str, Any]]] = []
        candidates.extend(("claim", item) for item in (graph.get("claims") or []) if isinstance(item, dict))
        candidates.extend(("chunk", item) for item in (graph.get("chunks") or []) if isinstance(item, dict))

        for kind, item in candidates:
            text = item.get("claim_text") if kind == "claim" else item.get("content")
            text = text or item.get("text")
            if not isinstance(text, str) or not text.strip():
                continue
            mapped: dict[str, Any] = {"text": text.strip(), "metadata": {"graph_item_type": kind}}

            evidence_id = item.get("evidence_id") or item.get("claim_id") or item.get("chunk_id") or item.get("id")
            if evidence_id is not None:
                mapped["evidence_id"] = str(evidence_id)
            document_id = item.get("document_id") or item.get("doc_id")
            if document_id is not None:
                mapped["document_id"] = str(document_id)
            chunk_id = item.get("chunk_id")
            if chunk_id is not None:
                mapped["chunk_id"] = str(chunk_id)
            source_name = item.get("source_name") or item.get("source") or item.get("title")
            if source_name is not None:
                mapped["source_name"] = str(source_name)
            if item.get("page") is not None:
                mapped["page"] = item.get("page")
            if item.get("revision") is not None:
                mapped["revision"] = str(item.get("revision"))
            score = item.get("retrieval_score") if item.get("retrieval_score") is not None else item.get("score")
            if isinstance(score, (int, float)) and score >= 0:
                mapped["retrieval_score"] = float(score)
            scope = item.get("authorization_scope") or item.get("access_scope")
            if scope is not None:
                mapped["authorization_scope"] = str(scope)
            if item.get("evidence_set_complete") is not None:
                mapped["metadata"]["evidence_set_complete"] = bool(item.get("evidence_set_complete"))
            output.append(mapped)
        return output

    @staticmethod
    def _action(report: dict[str, Any]) -> str:
        decision = report.get("decision") or {}
        return str(decision.get("action") or "").lower()

    @staticmethod
    def _compose_candidate(
        query: str,
        *,
        graph: dict[str, Any] | None,
        diagnostic: dict[str, Any] | None,
    ) -> str:
        sections = [f"Assessment request: {query}"]
        if diagnostic:
            detection = diagnostic.get("detection") or {}
            hypotheses = diagnostic.get("hypotheses") or []
            actions = diagnostic.get("recommended_actions") or []
            top = hypotheses[0] if hypotheses else {}
            sections.append(
                "Diagnostic evidence: "
                f"decision={diagnostic.get('decision', 'unknown')}; "
                f"abnormal={detection.get('abnormal', 'unknown')}; "
                f"top_hypothesis={top.get('label', 'none')}."
            )
            if actions:
                sections.append("Recommended actions: " + "; ".join(str(item) for item in actions[:5]))
        if graph:
            claims = graph.get("claims") or []
            chunks = graph.get("chunks") or []
            if claims:
                sections.append("Verified document claims: " + " ".join(str(item.get("claim_text", "")) for item in claims[:6]))
            elif chunks and graph.get("status") == "grounded":
                sections.append("Document evidence: " + " ".join(str(item.get("content", "")) for item in chunks[:3]))
            else:
                sections.append("Document evidence was insufficient; no document claim is asserted.")
        if not graph and not diagnostic:
            sections.append("No evidence service returned usable evidence; abstention is required.")
        return "\n\n".join(sections)
