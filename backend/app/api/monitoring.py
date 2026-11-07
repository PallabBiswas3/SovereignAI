from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.llm.factory import configured_local_provider
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
    remote_mode = settings.llm_provider.strip().lower() in {"remote_vllm", "hf_space"}
    remote_configured = False
    model_health = None
    active_model_url = (
        settings.vllm_url
        if settings.llm_provider.lower() == "vllm"
        else settings.ollama_url
    )
    if remote_mode:
        active_model_url = (settings.hf_space_url if settings.llm_provider.strip().lower() == "hf_space"
                            else settings.remote_vllm_url)
        try:
            selection = configured_local_provider(settings)
            remote_configured = bool(config_check["passed"])
            model_health = await selection.provider.health_check()
        except ValueError:
            model_health = {"available": False}
        if not remote_configured:
            config_check["passed"] = False
    if not LocalNetworkPolicy.is_local_url(active_model_url):
        config_check["passed"] = False
        config_check["violations"] = [*config_check["violations"], active_model_url]
    config_check["active_text_provider"] = settings.llm_provider
    config_check["active_text_endpoint"] = active_model_url
    attempts = db.query(NetworkEventRecord).filter_by(allowed=False).order_by(NetworkEventRecord.created_at.desc()).limit(50).all()
    return {
        "sovereignty_status": ("remote-inference-enabled" if remote_configured else
                               "application-policy-verified" if config_check["passed"] else "violation"),
        "external_ai_apis": int(remote_mode),
        "remote_inference_enabled": remote_configured,
        "external_requests": counts["external_attempts"],
        "allowed_local_requests": counts["allowed_local_requests"],
        "configuration": config_check,
        "services": await local_service_status(
            active_model_url, settings.llm_provider, settings.vllm_api_key,
            model_health=model_health,
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
        "note": ("Remote inference is enabled: prompts and supplied context leave this machine. This deployment is not air-gapped. Request counters below cover recorded policy events, not all HTTP traffic."
                 if remote_configured else
                 "Application policy checks are not proof of host-level isolation. Network-level proof requires the internal Compose network or an equivalent firewall plus the supplied active egress test."),
    }
