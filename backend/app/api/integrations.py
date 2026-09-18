from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.audit.logger import AuditLogger
from app.core.database import get_db
from app.identity.dependencies import require_permission
from app.identity.models import Permission, Principal
from app.integrations.clients import IntegrationServiceError
from app.integrations.models import IntegratedAnalysisRequest, IntegratedAnalysisResponse
from app.integrations.orchestrator import IndustrialIntegrationOrchestrator


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
            principal_id=principal.user_id,
            organization_id=principal.organization_id,
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
