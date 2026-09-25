from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import NetworkEventRecord, get_db
from app.monitoring.network import (
    AirGapVerifier,
    LocalNetworkPolicy,
    NetworkMonitor,
    local_service_status,
)
from app.identity.dependencies import require_permission
from app.identity.models import Permission, Principal


router = APIRouter(prefix="/api/monitor", tags=["monitoring"])


@router.get("/network")
async def network_status(
    principal: Principal = Depends(require_permission(Permission.audit_read)),
    db: Session = Depends(get_db),
) -> dict[str, object]:
    settings = get_settings()
    counts = NetworkMonitor(db).counts()
    config_check = AirGapVerifier().verify_model_config(settings.models_config)
    active_model_url = (
        settings.vllm_url
        if settings.llm_provider.lower() == "vllm"
        else settings.ollama_url
    )
    if not LocalNetworkPolicy.is_local_url(active_model_url):
        config_check["passed"] = False
        config_check["violations"] = [*config_check["violations"], active_model_url]
    config_check["active_text_provider"] = settings.llm_provider
    config_check["active_text_endpoint"] = active_model_url
    attempts = db.query(NetworkEventRecord).filter_by(allowed=False).order_by(NetworkEventRecord.created_at.desc()).limit(50).all()
    return {
        "sovereignty_status": "application-policy-verified" if config_check["passed"] else "violation",
        "external_ai_apis": 0,
        "external_requests": counts["external_attempts"],
        "allowed_local_requests": counts["allowed_local_requests"],
        "configuration": config_check,
        "services": await local_service_status(
            active_model_url, settings.llm_provider, settings.vllm_api_key,
        ),
        "blocked_attempts": [
            {
                "destination": item.destination,
                "component": item.component,
                "timestamp": item.created_at.isoformat(),
            }
            for item in attempts
        ],
        "verification_scope": "application-controlled clients and configured model endpoints",
        "network_isolation_proof": "Run backend/scripts/verify_airgap.py inside the internal Compose network.",
        "note": "Application policy checks are not proof of host-level isolation. Network-level proof requires the internal Compose network or an equivalent firewall plus the supplied active egress test.",
    }
