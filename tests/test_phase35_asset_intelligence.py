from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from sqlalchemy.pool import StaticPool

from app.api.governance import ApprovalDecision, decide_approval
from app.api.assets import router as assets_router
from app.assets.assessment import measurement_discrepancy
from app.assets.context import AssetContextService
from app.assets.maintenance import APELLocalCMMSConnector
from app.assets.models import (
    AssetFailureCode, FreshnessStatus, HistoricalSeries, OperationalMeasurement,
    TelemetryQuality, TrendDirection,
)
from app.assets.repository import AssetRepository
from app.assets.resolver import AssetResolver
from app.assets.telemetry import APELSimulatorTelemetryProvider, FreshnessPolicy, TelemetryProvider
from app.assets.trends import InsufficientHistoryError, TrendAnalyzer
from app.core.database import AssetEvidenceLinkRecord, Base, HumanApprovalRecord, MaintenanceDraftRecord
from app.demo.apel import ApelDemoService
from app.evidence.units import UnitService
from app.identity.authorization import AuthorizationService
from app.identity.models import Permission
from app.identity.provider import LocalIdentityProvider
from app.identity.dependencies import get_current_principal
from app.core.database import get_db
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.factory import configured_hybrid_retriever


ROOT = Path(__file__).resolve().parents[1]
AS_OF = datetime(2026, 9, 2, 12, 44, tzinfo=timezone.utc)


@pytest.fixture()
def seeded(tmp_path: Path):
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False}, poolclass=StaticPool)
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        ApelDemoService(session, ROOT / "demo/apel", tmp_path / "apel").seed()
        identity = LocalIdentityProvider(session, ROOT / "config/access.yaml")
        yield session, identity


def _provider(session: Session) -> APELSimulatorTelemetryProvider:
    return APELSimulatorTelemetryProvider(
        AssetRepository(session), FreshnessPolicy(300, 86400),
    )


def test_asset_passport_hierarchy_aliases_acl_and_links(seeded) -> None:
    session, identity = seeded
    engineer = identity.principal_for_user("apel-maint-001")
    auditor = identity.principal_for_user("apel-auditor-001")
    assert engineer and auditor
    repository = AssetRepository(session)
    resolver = AssetResolver(repository)
    resolved = resolver.resolve(engineer, "P-102")
    assert resolved.status == "RESOLVED"
    assert resolved.asset and resolved.asset.asset_id == "Pump-102"
    assert (resolved.asset.plant_id, resolved.asset.area_id, resolved.asset.unit_id) == (
        "plant-a", "utilities", "cooling-water",
    )
    assert resolver.resolve(engineer, "does-not-exist").status == "ASSET_NOT_FOUND"
    ambiguous = resolver.resolve(engineer, "P102")
    assert ambiguous.status == "AMBIGUOUS_ASSET"
    assert ambiguous.candidate_ids == ["Pump-102", "Pump-103"]
    assert resolver.resolve(auditor, "Pump-102").status == "ASSET_ACCESS_DENIED"
    relationships = {
        row.relationship for row in session.query(AssetEvidenceLinkRecord).filter_by(asset_id="Pump-102")
    }
    assert {"HAS_DOCUMENT", "HAS_INSPECTION", "HAS_MEASUREMENT"} <= relationships


def test_read_only_telemetry_quality_freshness_and_history(seeded) -> None:
    session, identity = seeded
    engineer = identity.principal_for_user("apel-maint-001")
    auditor = identity.principal_for_user("apel-auditor-001")
    assert engineer and auditor
    provider = _provider(session)
    assert set(TelemetryProvider.__abstractmethods__) == {"get_latest", "get_history"}
    latest = provider.get_latest(engineer, "Pump-102", as_of=AS_OF)
    vibration = next(item for item in latest if item.metric == "vibration")
    assert (vibration.value, vibration.unit, vibration.quality, vibration.freshness_status) == (
        8.2, "mm/s", TelemetryQuality.good, FreshnessStatus.fresh,
    )
    stale = provider.get_latest(
        engineer, "Pump-102", ["bearing_temperature"],
        scenario="PUMP_102_STALE_DATA", as_of=AS_OF,
    )[0]
    assert stale.age_seconds == 8 * 3600 + 120
    assert stale.freshness_status == FreshnessStatus.stale
    assert AssetFailureCode.telemetry_stale in stale.warnings
    bad = provider.get_latest(
        engineer, "Pump-102", ["vibration"], scenario="PUMP_102_BAD_QUALITY", as_of=AS_OF,
    )[0]
    assert bad.quality == TelemetryQuality.bad
    assert AssetFailureCode.bad_telemetry_quality in bad.warnings
    history = provider.get_history(engineer, "Pump-102", "vibration", as_of=AS_OF)
    assert history and len(history.measurements) == 6
    assert history.measurements[-1].timestamp.tzinfo is not None
    with pytest.raises(PermissionError, match="ASSET_ACCESS_DENIED"):
        provider.get_latest(auditor, "Pump-102", as_of=AS_OF)


def _measurement(identifier: str, value: float, day: int, quality: TelemetryQuality = TelemetryQuality.good):
    timestamp = datetime(2026, 1, day, tzinfo=timezone.utc)
    return OperationalMeasurement(
        measurement_id=identifier, asset_id="Pump-X", metric="vibration", value=value,
        unit="mm/s", timestamp=timestamp, quality=quality, source="test", source_tag="test/tag",
        original_value=value, original_unit="mm/s",
    )


def test_trend_engine_is_deterministic_and_handles_insufficient_history() -> None:
    analyzer = TrendAnalyzer(slope_tolerance_per_day=0.001)
    increasing = HistoricalSeries(asset_id="Pump-X", metric="vibration", unit="mm/s",
                                  measurements=[_measurement("M1", 2, 1), _measurement("M2", 4, 2), _measurement("M3", 7, 3)], source="test")
    result = analyzer.analyze(increasing, threshold=5)
    assert (result.latest, result.mean, result.minimum, result.maximum, result.change) == (7, pytest.approx(13 / 3), 2, 7, 5)
    assert result.slope_per_day == 2.5
    assert result.trend == TrendDirection.increasing
    assert result.threshold_crossings == 1 and result.abnormal_readings == 1
    flat = HistoricalSeries(asset_id="Pump-X", metric="vibration", unit="mm/s",
                            measurements=[_measurement("F1", 3, 1), _measurement("F2", 3, 2)], source="test")
    assert analyzer.analyze(flat).trend == TrendDirection.flat
    decreasing = HistoricalSeries(asset_id="Pump-X", metric="vibration", unit="mm/s",
                                  measurements=[_measurement("D1", 3, 1), _measurement("D2", 1, 2)], source="test")
    assert analyzer.analyze(decreasing).trend == TrendDirection.decreasing
    with pytest.raises(InsufficientHistoryError):
        analyzer.analyze(HistoricalSeries(asset_id="Pump-X", metric="vibration", unit="mm/s",
                                          measurements=[_measurement("ONE", 3, 1)], source="test"))


def test_units_and_measurement_conflict_are_explicit() -> None:
    pressure = UnitService().convert(500, "kPa", "bar")
    assert pressure.normalized_value == 5.0
    left = _measurement("INSPECTION", 8.2, 1)
    right = _measurement("TELEMETRY", 5.1, 2)
    conflict = measurement_discrepancy(left, right)
    assert conflict and conflict.type == "MEASUREMENT_SOURCE_CONFLICT"
    assert conflict.sources == ["INSPECTION", "TELEMETRY"]


def test_asset_context_is_authorized_structured_and_bounded(seeded) -> None:
    session, identity = seeded
    engineer = identity.principal_for_user("apel-maint-001")
    auditor = identity.principal_for_user("apel-auditor-001")
    assert engineer and auditor
    context = AssetContextService(AssetRepository(session), _provider(session)).compile(
        engineer, "Pump-102", "Assess all evidence", as_of=AS_OF,
    )
    assert context.asset.asset_id == "Pump-102"
    assert 1 <= len(context.latest_measurements) <= 12
    assert len(context.trends) <= 8 and len(context.document_ids) <= 20
    assert len(context.inspections) <= 5 and len(context.maintenance) <= 10
    assert context.rules[0].source.revision == "Rev 4"
    assert context.calculations[0].result is False
    assert context.findings[0].condition.value == "ABNORMAL"
    assert context.recommendations[0].requires_human_approval is True
    assert context.conflicts[0].type == "MEASUREMENT_SOURCE_CONFLICT"
    with pytest.raises(PermissionError, match="ASSET_ACCESS_DENIED"):
        AssetContextService(AssetRepository(session), _provider(session)).compile(auditor, "Pump-102")


def test_asset_aware_hybrid_retrieval_preserves_acl(seeded) -> None:
    session, identity = seeded
    engineer = identity.principal_for_user("apel-maint-001")
    assert engineer
    retriever = configured_hybrid_retriever(session, embeddings=LocalHashEmbeddingProvider(), principal=engineer)
    results = retriever.search("Pump-102 vibration limit", 10, asset_id="Pump-102")
    assert results
    assert results[0].source.get("asset_id") == "Pump-102"
    assert "asset_link" in results[0].retrieval_methods
    assert all(item.source.get("department") != "finance" for item in results)


def test_local_maintenance_draft_is_hashed_governed_and_never_executes_plant_action(seeded) -> None:
    session, identity = seeded
    engineer = identity.principal_for_user("apel-maint-001")
    approver = identity.principal_for_user("apel-manager-001")
    assert engineer and approver and engineer.has_permission(Permission.maintenance_draft_create)
    connector = APELLocalCMMSConnector(session)
    draft = connector.create_work_order_draft(
        engineer, asset_id="Pump-102", action_type="INSPECTION", priority="HIGH",
        title="Inspect Pump-102 bearing and alignment",
        description="Review the supported abnormal-vibration finding before planning work.",
        reason_claim_ids=["CL-P102-VIBRATION"],
    )
    approval = session.get(HumanApprovalRecord, draft.approval_id)
    assert approval and approval.status == "pending" and approval.action_hash
    assert session.get(MaintenanceDraftRecord, draft.draft_id).status == "DRAFT"
    assert AuthorizationService().can_approve_action(engineer, engineer.user_id).allowed is False
    response = asyncio.run(decide_approval(approval.id, ApprovalDecision(approve=True), approver, session))
    assert response["execution_status"] == "draft_accepted", response
    assert response["result"]["plant_action_executed"] is False
    assert session.get(MaintenanceDraftRecord, draft.draft_id).status == "APPROVED"


def test_no_plant_control_method_or_tool_is_present(seeded) -> None:
    session, identity = seeded
    engineer = identity.principal_for_user("apel-maint-001")
    assert engineer
    forbidden = {"write_tag", "set_value", "execute_command", "start_asset", "stop_asset"}
    assert not forbidden.intersection(dir(_provider(session)))
    from app.tools.registry import create_agent_registry
    names = {item["name"] for item in create_agent_registry(__import__("app.core.config", fromlist=["get_settings"]).get_settings(), session, engineer).discover()}
    assert {"get_asset_telemetry", "get_asset_history"} <= names
    assert not forbidden.intersection(names)
    denied = engineer.model_copy(update={"permissions": [item for item in engineer.permissions if item != Permission.telemetry_read]})
    denied_registry = create_agent_registry(__import__("app.core.config", fromlist=["get_settings"]).get_settings(), session, denied)
    result = asyncio.run(denied_registry.get("get_asset_telemetry").execute({"asset_id": "Pump-102"}))
    assert result.success is False and result.error == "ASSET_ACCESS_DENIED"


def test_asset_api_uses_authenticated_principal_and_returns_typed_views(seeded) -> None:
    session, identity = seeded
    engineer = identity.principal_for_user("apel-maint-001")
    assert engineer
    api = FastAPI()
    api.include_router(assets_router)
    api.dependency_overrides[get_current_principal] = lambda: engineer

    def database_override():
        yield session

    api.dependency_overrides[get_db] = database_override
    with TestClient(api) as client:
        catalog = client.get("/api/assets")
        assert catalog.status_code == 200 and catalog.json()["count"] == 20
        passport = client.get("/api/assets/Pump-102")
        assert passport.status_code == 200
        latest = client.get("/api/assets/Pump-102/measurements/latest", params={"as_of": AS_OF.isoformat()})
        assert latest.status_code == 200 and latest.json()["data_source"] == "SIMULATED_PLANT_DATA"
        history = client.get("/api/assets/Pump-102/measurements/history", params={"metric": "vibration", "as_of": AS_OF.isoformat()})
        assert history.status_code == 200 and history.json()["trend"]["trend"] == "INCREASING"
        context = client.get("/api/assets/Pump-102/context", params={"as_of": AS_OF.isoformat()})
        assert context.status_code == 200 and context.json()["context"]["asset"]["asset_id"] == "Pump-102"
        timeline = client.get("/api/assets/Pump-102/timeline")
        assert timeline.status_code == 200 and timeline.json()["items"]
        draft = client.post("/api/assets/Pump-102/maintenance/drafts", json={
            "title": "Inspect Pump-102 bearings", "description": "Review supported abnormal vibration.",
            "reason_claim_ids": ["CL-P102-VIBRATION"],
        })
        assert draft.status_code == 201 and draft.json()["plant_action_executed"] is False


def test_asset_api_denies_telemetry_without_permission_and_leaks_no_values(seeded) -> None:
    session, identity = seeded
    user = identity.principal_for_user("apel-maint-001")
    assert user
    denied = user.model_copy(update={"permissions": [Permission.asset_read]})
    api = FastAPI()
    api.include_router(assets_router)
    api.dependency_overrides[get_current_principal] = lambda: denied

    def database_override():
        yield session

    api.dependency_overrides[get_db] = database_override
    with TestClient(api) as client:
        response = client.get("/api/assets/Pump-102/measurements/latest")
    assert response.status_code == 403
    assert "8.2" not in response.text and "PlantA/Utilities" not in response.text
