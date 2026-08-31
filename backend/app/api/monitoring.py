from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.database import NetworkEventRecord, get_db
from app.monitoring.network import AirGapVerifier, NetworkMonitor, local_service_status


router = APIRouter(prefix="/api/monitor", tags=["monitoring"])


@router.get("/network")
async def network_status(db: Session = Depends(get_db)) -> dict[str, object]:
    settings = get_settings()
    counts = NetworkMonitor(db).counts()
    config_check = AirGapVerifier().verify_model_config(settings.models_config)
    attempts = db.query(NetworkEventRecord).filter_by(allowed=False).order_by(NetworkEventRecord.created_at.desc()).limit(50).all()
    return {
        "sovereignty_status": "application-policy-verified" if config_check["passed"] else "violation",
        "external_ai_apis": 0,
        "external_requests": counts["external_attempts"],
        "allowed_local_requests": counts["allowed_local_requests"],
        "configuration": config_check,
        "services": await local_service_status(settings.ollama_url),
        "blocked_attempts": [{"destination": item.destination, "component": item.component, "timestamp": item.created_at.isoformat()} for item in attempts],
        "verification_scope": "application-controlled clients and configured model endpoints",
        "network_isolation_proof": "Run backend/scripts/verify_airgap.py inside the internal Compose network.",
        "note": "Application policy checks are not proof of host-level isolation. Network-level proof requires the internal Compose network or an equivalent firewall plus the supplied active egress test.",
    }
