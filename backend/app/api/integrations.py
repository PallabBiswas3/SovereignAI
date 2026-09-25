from __future__ import annotations

from types import SimpleNamespace

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.audit.logger import AuditLogger
from app.core.database import get_db
from app.core.database import AgentRunRecord
from app.core.config import get_settings
from app.artifacts.service import ArtifactService
from app.capsules.builder import EvidenceCapsuleBuilder
from app.identity.dependencies import require_permission
from app.identity.authorization import AuthorizationService
from app.identity.models import Permission, Principal
from app.integrations.clients import IntegrationServiceError
from app.integrations.models import IntegratedAnalysisRequest, IntegratedAnalysisResponse
from app.integrations.orchestrator import IndustrialIntegrationOrchestrator
from app.integrations.routing import IntegrationRoutePlan
from app.integrations.task_adapter import integrated_result_state
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.workcells.defaults import configured_workcell_registry


router = APIRouter(prefix="/api/integrations", tags=["integrations"])


@router.get("/health")
async def integration_health(
    _principal: Principal = Depends(require_permission(Permission.task_read)),
) -> dict[str, object]:
    services = await IndustrialIntegrationOrchestrator().health()
    return {
        "status": "ok" if all(item.get("status") == "ok" for item in services.values()) else "degraded",
        "services": services,
    }


@router.post("/analyze", response_model=IntegratedAnalysisResponse)
async def integrated_analysis(
    payload: IntegratedAnalysisRequest,
    principal: Principal = Depends(require_permission(Permission.workcell_execute)),
    db: Session = Depends(get_db),
) -> IntegratedAnalysisResponse:
    try:
        result = await IndustrialIntegrationOrchestrator().analyze(
            payload,
            principal=principal,
        )
    except IntegrationServiceError as exc:
        failure_meta = {
            "service": exc.service,
            "failed_services": exc.failed_services,
            "status_code": exc.status_code,
            "attempts": exc.attempts,
            "retryable": exc.retryable,
        }
        AuditLogger(db, principal).log(
            "integration:unavailable",
            "INTEGRATION_FAILED_CLOSED",
            "Industrial integration stopped because a required service was unavailable.",
            failure_meta,
        )
        raise HTTPException(
            status_code=503,
            detail={"code": "INTEGRATION_SERVICE_UNAVAILABLE", **failure_meta},
        ) from exc

    if result.released and payload.diagnostic is not None and "pump-102" in payload.query.lower():
        _materialize_pump_102_outputs(result, payload, principal, db)

    AuditLogger(db, principal).log(
        result.run_id,
        "INTEGRATED_ANALYSIS_COMPLETED",
        "Integrated evidence analysis completed through the ControlPlane release gate.",
        {
            "released": result.released,
            "status": result.status,
            "services": result.service_status,
            "controlplane_action": (result.controlplane or {}).get("decision", {}).get("action"),
        },
    )
    return result


def _materialize_pump_102_outputs(
    result: IntegratedAnalysisResponse,
    payload: IntegratedAnalysisRequest,
    principal: Principal,
    db: Session,
) -> None:
    """Create governed outputs only for the canonical released flagship run."""

    settings = get_settings()
    definition = configured_workcell_registry(settings).get("pump-inspection")
    scope = AuthorizationService.owned_scope(principal)
    artifact_root = settings.workspace_root / "artifacts"
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifact_root / f"Pump-102_Maintenance_Recommendation_{result.run_id[-8:]}.md"
    graph = result.graph_evidence or {}
    diagnostic = result.diagnostic or {}
    citations = [
        str(item.get("id") or item.get("chunk_id"))
        for item in (graph.get("chunks") or [])
        if isinstance(item, dict) and (item.get("id") or item.get("chunk_id"))
    ]
    artifact_path.write_text(
        "# Pump-102 Maintenance Recommendation\n\n"
        f"Run: `{result.run_id}`\n\n"
        f"Release status: **{result.status}**\n\n"
        "## Verified recommendation\n\n"
        f"{result.final_response}\n\n"
        "## Diagnostic summary\n\n"
        f"Decision: {diagnostic.get('decision', 'not reported')}\n\n"
        f"Hypotheses: {', '.join(str(item.get('label')) for item in diagnostic.get('hypotheses', []) if isinstance(item, dict)) or 'none'}\n\n"
        "## Evidence citations\n\n"
        + ("\n".join(f"- `{item}`" for item in citations) if citations else "- GraphRAG returned no releasable citation.")
        + "\n\nThis artifact is advisory only. Physical maintenance requires human authorization.\n",
        encoding="utf-8",
    )
    artifact = ArtifactService(db, artifact_root).register(
        artifact_path,
        run_id=result.run_id,
        workcell_id=definition.manifest.id,
        workcell_version=definition.manifest.version,
        artifact_type="maintenance_recommendation",
        derived_from_claims=[
            str(item.get("id")) for item in graph.get("claims", [])
            if isinstance(item, dict) and item.get("id")
        ],
        scope=scope,
    )
    result.artifacts = [{
        "id": artifact.id,
        "name": artifact.name,
        "sha256": artifact.sha256,
        "url": f"/api/artifacts/{artifact.id}",
    }]

    routing = ModelRouter(ModelRegistry(settings.models_config)).route(payload.query)
    plan = IntegrationRoutePlan(
        use_graph=payload.include_graph_evidence,
        diagnostic_domain=payload.diagnostic.domain,
        diagnostic=payload.diagnostic,
        services=[
            *(["graph-rag"] if payload.include_graph_evidence else []),
            "time-series-diagnostic-agent", "local-model", "controlplane",
        ],
        confidence=1.0,
        reason="Canonical Pump-102 evidence-to-maintenance workflow.",
    )
    selection = SimpleNamespace(
        requested=SimpleNamespace(value="AUTOMATIC"),
        selected=SimpleNamespace(value="DEEP"),
        reason="Flagship workflow uses thorough verification.",
    )
    state = integrated_result_state(
        request=payload.query, routing=routing, selection=selection, plan=plan, result=result,
    )
    state.id = result.run_id
    state.workcell_id = definition.manifest.id
    state.workcell_version = definition.manifest.version
    state.workcell_hash = definition.content_hash
    state.workflow_version = definition.workflow.version
    state.artifacts = result.artifacts
    state.principal_id = principal.user_id
    state.organization_id = scope.organization_id
    state.workspace_id = scope.workspace_id
    state.department_id = scope.department_id
    state.classification = scope.classification.name.upper()
    db.add(AgentRunRecord(
        id=state.id, request=state.request, status=state.status.value,
        state_json=state.model_dump_json(),
        organization_id=scope.organization_id, owner_id=principal.user_id,
        workspace_id=scope.workspace_id, department_id=scope.department_id,
        classification=scope.classification.name.upper(),
    ))
    db.commit()

    if principal.has_permission(Permission.capsule_create):
        try:
            capsule = EvidenceCapsuleBuilder(
                db, settings.capsules_root, artifact_root
            ).build(state, definition)
        except (OSError, ValueError) as exc:
            result.capsule = {"state": "FAILED", "error": str(exc)}
        else:
            result.capsule = {
                "id": capsule.id,
                "state": capsule.state,
                "capsule_root_hash": capsule.capsule_root_hash,
                "signature_status": capsule.signature_status,
                "verify_url": f"/api/capsules/{capsule.id}/verify",
                "download_url": f"/api/capsules/{capsule.id}/download",
            }
