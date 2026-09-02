from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.assets.context import AssetContextService
from app.assets.maintenance import APELLocalCMMSConnector
from app.assets.models import AssetFailureCode
from app.assets.repository import AssetRepository
from app.assets.resolver import AssetResolver
from app.assets.telemetry import APELSimulatorTelemetryProvider, FreshnessPolicy
from app.assets.trends import InsufficientHistoryError, TrendAnalyzer
from app.audit.logger import AuditLogger
from app.core.config import get_settings
from app.core.database import get_db
from app.identity.dependencies import require_permission
from app.identity.models import Permission, Principal


router = APIRouter(prefix="/api/assets", tags=["assets"])


class MaintenanceDraftRequest(BaseModel):
    action_type: str = "INSPECTION"
    priority: str = "HIGH"
    title: str = Field(min_length=3, max_length=240)
    description: str = Field(min_length=3, max_length=4000)
    reason_claim_ids: list[str] = Field(min_length=1, max_length=50)


def _provider(db: Session) -> APELSimulatorTelemetryProvider:
    settings = get_settings()
    return APELSimulatorTelemetryProvider(
        AssetRepository(db),
        FreshnessPolicy(settings.telemetry_default_freshness_seconds, settings.telemetry_expired_seconds),
        default_scenario=settings.telemetry_scenario,
    )


def _resolve(db: Session, principal: Principal, reference: str):
    result = AssetResolver(AssetRepository(db)).resolve(principal, reference)
    if result.status == "RESOLVED":
        return result.asset
    status = {AssetFailureCode.asset_not_found.value: 404,
              AssetFailureCode.ambiguous_asset.value: 409,
              AssetFailureCode.asset_access_denied.value: 403}.get(result.status, 400)
    raise HTTPException(status_code=status, detail={"code": result.status, "candidates": result.candidate_ids})


@router.get("")
def list_assets(
    q: str | None = None,
    principal: Principal = Depends(require_permission(Permission.asset_read)),
    db: Session = Depends(get_db),
):
    repository = AssetRepository(db)
    values = repository.search_assets(principal, q, 100) if q else repository.list_assets(principal, 100)
    return {"items": [item.model_dump(mode="json") for item in values], "count": len(values), "data_source": "SIMULATED_PLANT_DATA"}


@router.get("/{asset_id}")
def get_asset(
    asset_id: str,
    principal: Principal = Depends(require_permission(Permission.asset_read)),
    db: Session = Depends(get_db),
):
    return {"asset": _resolve(db, principal, asset_id).model_dump(mode="json"), "data_source": "SIMULATED_PLANT_DATA"}


@router.get("/{asset_id}/measurements/latest")
def latest_measurements(
    asset_id: str, metrics: list[str] | None = Query(default=None), scenario: str | None = None,
    as_of: datetime | None = None,
    principal: Principal = Depends(require_permission(Permission.telemetry_read)),
    db: Session = Depends(get_db),
):
    asset = _resolve(db, principal, asset_id)
    values = _provider(db).get_latest(principal, asset.asset_id, metrics, scenario=scenario, as_of=as_of)
    log = AuditLogger(db, principal)
    log.log(f"asset:{asset.asset_id}", "TELEMETRY_READ", "Authorized latest telemetry snapshot read.",
            {"asset_id": asset.asset_id, "measurement_ids": [item.measurement_id for item in values], "scenario": scenario or get_settings().telemetry_scenario})
    for item in values:
        for warning in item.warnings:
            event = "TELEMETRY_BAD_QUALITY" if warning == AssetFailureCode.bad_telemetry_quality else "TELEMETRY_STALE"
            log.log(f"asset:{asset.asset_id}", event, "Telemetry quality/freshness warning.",
                    {"asset_id": asset.asset_id, "measurement_id": item.measurement_id, "code": warning.value})
    return {"asset_id": asset.asset_id, "provider": _provider(db).provider_name,
            "measurements": [item.model_dump(mode="json") for item in values], "data_source": "SIMULATED_PLANT_DATA"}


@router.get("/{asset_id}/measurements/history")
def measurement_history(
    asset_id: str, metric: str, start: datetime | None = None, end: datetime | None = None,
    scenario: str | None = None, as_of: datetime | None = None, limit: int = Query(120, ge=2, le=500),
    principal: Principal = Depends(require_permission(Permission.telemetry_read)),
    db: Session = Depends(get_db),
):
    asset = _resolve(db, principal, asset_id)
    provider = _provider(db)
    series = provider.get_history(principal, asset.asset_id, metric, start, end, scenario=scenario, as_of=as_of, limit=limit)
    if not series:
        raise HTTPException(status_code=404, detail={"code": AssetFailureCode.history_unavailable.value})
    try:
        trend = TrendAnalyzer().analyze(series)
    except InsufficientHistoryError:
        trend = None
    AuditLogger(db, principal).log(f"asset:{asset.asset_id}", "HISTORY_READ", "Bounded asset history read.",
                                   {"asset_id": asset.asset_id, "metric": metric, "sample_count": len(series.measurements)})
    if trend:
        AuditLogger(db, principal).log(f"asset:{asset.asset_id}", "TREND_CALCULATED", "Deterministic trend calculated.",
                                       {"asset_id": asset.asset_id, "metric": metric, "sample_count": trend.sample_count})
    return {"series": series.model_dump(mode="json"), "trend": trend.model_dump(mode="json") if trend else None,
            "data_source": "SIMULATED_PLANT_DATA"}


@router.get("/{asset_id}/inspections")
def inspections(asset_id: str, principal: Principal = Depends(require_permission(Permission.asset_read)), db: Session = Depends(get_db)):
    asset = _resolve(db, principal, asset_id)
    return {"items": [item.model_dump(mode="json") for item in AssetRepository(db).get_asset_inspections(principal, asset.asset_id)]}


@router.get("/{asset_id}/documents")
def documents(asset_id: str, principal: Principal = Depends(require_permission(Permission.asset_read)), db: Session = Depends(get_db)):
    asset = _resolve(db, principal, asset_id)
    return {"items": [item.model_dump(mode="json") for item in AssetRepository(db).get_asset_documents(principal, asset.asset_id)]}


@router.get("/{asset_id}/maintenance")
def maintenance(asset_id: str, principal: Principal = Depends(require_permission(Permission.maintenance_read)), db: Session = Depends(get_db)):
    asset = _resolve(db, principal, asset_id)
    connector = APELLocalCMMSConnector(db)
    return {"history": [item.model_dump(mode="json") for item in connector.list_asset_work_orders(principal, asset.asset_id)],
            "drafts": [item.model_dump(mode="json") for item in AssetRepository(db).list_drafts(principal, asset.asset_id)],
            "connector": connector.provider_name}


@router.post("/{asset_id}/maintenance/drafts", status_code=201)
def create_maintenance_draft(
    asset_id: str, payload: MaintenanceDraftRequest,
    principal: Principal = Depends(require_permission(Permission.maintenance_draft_create)),
    db: Session = Depends(get_db),
):
    asset = _resolve(db, principal, asset_id)
    try:
        draft = APELLocalCMMSConnector(db).create_work_order_draft(
            principal, asset_id=asset.asset_id, **payload.model_dump(),
        )
    except (PermissionError, ValueError) as exc:
        raise HTTPException(status_code=403 if isinstance(exc, PermissionError) else 400,
                            detail={"code": str(exc)}) from exc
    AuditLogger(db, principal).log(f"asset:{asset.asset_id}", "MAINTENANCE_DRAFT_CREATED",
                                   "Local maintenance action draft created; no plant action was executed.",
                                   {"asset_id": asset.asset_id, "draft_id": draft.draft_id, "approval_id": draft.approval_id})
    AuditLogger(db, principal).log(f"asset:{asset.asset_id}", "MAINTENANCE_DRAFT_APPROVAL_REQUESTED",
                                   "Human approval requested for the exact hashed draft.",
                                   {"draft_id": draft.draft_id, "approval_id": draft.approval_id})
    return {"draft": draft.model_dump(mode="json"), "plant_action_executed": False}


@router.get("/{asset_id}/context")
def asset_context(
    asset_id: str, task: str = "Assess current condition", scenario: str | None = None,
    as_of: datetime | None = None,
    principal: Principal = Depends(require_permission(Permission.telemetry_read)), db: Session = Depends(get_db),
):
    asset = _resolve(db, principal, asset_id)
    context = AssetContextService(AssetRepository(db), _provider(db)).compile(
        principal, asset.asset_id, task, scenario=scenario, as_of=as_of,
    )
    AuditLogger(db, principal).log(f"asset:{asset.asset_id}", "ASSET_CONTEXT_COMPILED", "Bounded authorized asset context compiled.",
                                   {"asset_id": asset.asset_id, "measurement_ids": [item.measurement_id for item in context.latest_measurements],
                                    "trend_metrics": [item.metric for item in context.trends]})
    return {"context": context.model_dump(mode="json"), "data_source": "SIMULATED_PLANT_DATA"}


@router.get("/{asset_id}/timeline")
def timeline(asset_id: str, principal: Principal = Depends(require_permission(Permission.asset_read)), db: Session = Depends(get_db)):
    asset = _resolve(db, principal, asset_id)
    repository = AssetRepository(db)
    events = [
        *({"timestamp": item.inspected_at, "type": "INSPECTION", "title": item.summary, "id": item.id}
          for item in repository.get_asset_inspections(principal, asset.asset_id, 20)),
        *({"timestamp": item.occurred_at, "type": "MAINTENANCE", "title": item.title, "id": item.id}
          for item in repository.get_maintenance_history(principal, asset.asset_id, 50)),
    ]
    events.sort(key=lambda item: item["timestamp"], reverse=True)
    return {"asset_id": asset.asset_id, "items": events[:50], "data_source": "SIMULATED_PLANT_DATA"}
