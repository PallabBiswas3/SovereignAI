from __future__ import annotations

import json
from datetime import timezone
from uuid import uuid4

from sqlalchemy.orm import Session

from app.assets.models import (
    AssetEvidenceLink, AssetPassport, AssetRelationship, InspectionRecord,
    MaintenanceActionDraft, MaintenanceDraftStatus, MaintenanceRecord,
    OperationalMeasurement,
)
from app.core.database import (
    AssetAliasRecord, AssetEvidenceLinkRecord, AssetRecord, InspectionRecordRow,
    MaintenanceDraftRecord, MaintenanceRecordRow, OperationalMeasurementRecord,
)
from app.identity.authorization import AuthorizationService
from app.identity.models import ClearanceLevel, Principal, ResourceScope, Role


def normalize_reference(value: str) -> str:
    return " ".join(value.strip().upper().split())


class AssetRepository:
    def __init__(self, session: Session) -> None:
        self.session = session
        self.authorization = AuthorizationService()

    @staticmethod
    def scope(row: AssetRecord) -> ResourceScope:
        return ResourceScope(
            resource_id=row.id, organization_id=row.organization_id,
            workspace_id=row.workspace_id, department_id=row.department_id,
            classification=ClearanceLevel.parse(row.classification), owner_id=row.owner_id,
            allowed_roles=[Role(value) for value in json.loads(row.allowed_roles_json or "[]")],
            allowed_users=list(json.loads(row.allowed_users_json or "[]")),
        )

    def is_authorized(self, principal: Principal, row: AssetRecord, *, telemetry: bool = False) -> bool:
        decision = self.authorization.can_read_telemetry(principal, self.scope(row)) if telemetry else self.authorization.can_read_asset(principal, self.scope(row))
        return decision.allowed

    def get_row(self, asset_id: str) -> AssetRecord | None:
        return self.session.get(AssetRecord, asset_id)

    def get_asset(self, principal: Principal, asset_id: str) -> AssetPassport | None:
        row = self.get_row(asset_id)
        return self._passport(row) if row and self.is_authorized(principal, row) else None

    def list_assets(self, principal: Principal, limit: int = 100) -> list[AssetPassport]:
        query = self.session.query(AssetRecord)
        if principal.organization_id != "*":
            query = query.filter(AssetRecord.organization_id == principal.organization_id)
        rows = query.order_by(AssetRecord.id).limit(min(limit, 200)).all()
        return [self._passport(row) for row in rows if self.is_authorized(principal, row)]

    def search_assets(self, principal: Principal, query: str, limit: int = 20) -> list[AssetPassport]:
        needle = query.strip().lower()
        return [item for item in self.list_assets(principal, 200) if needle in item.asset_id.lower() or needle in item.canonical_name.lower() or any(needle in alias.lower() for alias in item.aliases)][:limit]

    def resolution_rows(self, reference: str) -> list[AssetRecord]:
        normalized = normalize_reference(reference)
        ids = {row.asset_id for row in self.session.query(AssetAliasRecord).filter(AssetAliasRecord.alias_normalized == normalized).all()}
        direct = self.session.get(AssetRecord, reference)
        if direct:
            ids.add(direct.id)
        return [row for asset_id in sorted(ids) if (row := self.session.get(AssetRecord, asset_id))]

    def get_asset_documents(self, principal: Principal, asset_id: str) -> list[AssetEvidenceLink]:
        return self._links(principal, asset_id, AssetRelationship.has_document)

    def get_asset_inspections(self, principal: Principal, asset_id: str, limit: int = 20) -> list[InspectionRecord]:
        if not self.get_asset(principal, asset_id):
            return []
        rows = self.session.query(InspectionRecordRow).filter(InspectionRecordRow.asset_id == asset_id).order_by(InspectionRecordRow.inspected_at.desc()).limit(limit).all()
        return [InspectionRecord(id=row.id, asset_id=row.asset_id, inspected_at=self._aware(row.inspected_at), source_document_id=row.source_document_id, summary=row.summary, measurement_ids=json.loads(row.measurement_ids_json or "[]")) for row in rows]

    def get_measurements(self, principal: Principal, asset_id: str, metric: str | None = None, limit: int = 500, scenario: str | None = None) -> list[OperationalMeasurementRecord]:
        asset = self.get_row(asset_id)
        if not asset or not self.is_authorized(principal, asset, telemetry=True):
            return []
        query = self.session.query(OperationalMeasurementRecord).filter(OperationalMeasurementRecord.asset_id == asset_id)
        if metric:
            query = query.filter(OperationalMeasurementRecord.metric == metric)
        if scenario:
            query = query.filter(OperationalMeasurementRecord.scenario == scenario)
        return query.order_by(OperationalMeasurementRecord.timestamp.desc()).limit(min(limit, 2000)).all()

    def get_maintenance_history(self, principal: Principal, asset_id: str, limit: int = 50) -> list[MaintenanceRecord]:
        asset = self.get_row(asset_id)
        if not asset or not self.is_authorized(principal, asset) or not principal.has_permission("maintenance.read"):
            return []
        rows = self.session.query(MaintenanceRecordRow).filter(MaintenanceRecordRow.asset_id == asset_id).order_by(MaintenanceRecordRow.occurred_at.desc()).limit(limit).all()
        return [MaintenanceRecord(id=row.id, asset_id=row.asset_id, occurred_at=self._aware(row.occurred_at), event_type=row.event_type, title=row.title, summary=row.summary, source=row.source, status=row.status) for row in rows]

    def list_drafts(self, principal: Principal, asset_id: str) -> list[MaintenanceActionDraft]:
        if not self.get_asset(principal, asset_id) or not principal.has_permission("maintenance.read"):
            return []
        rows = self.session.query(MaintenanceDraftRecord).filter(MaintenanceDraftRecord.asset_id == asset_id).order_by(MaintenanceDraftRecord.created_at.desc()).all()
        return [self._draft(row) for row in rows]

    def link_evidence(self, asset_id: str, evidence_id: str, relationship: AssetRelationship, source: str, *, confidence: float = 1.0, inferred: bool = False) -> AssetEvidenceLinkRecord:
        existing = self.session.query(AssetEvidenceLinkRecord).filter_by(asset_id=asset_id, evidence_id=evidence_id, relationship=relationship.value).first()
        if existing:
            return existing
        row = AssetEvidenceLinkRecord(id=str(uuid4()), asset_id=asset_id, evidence_id=evidence_id, relationship=relationship.value, source=source, confidence=confidence, inferred=inferred)
        self.session.add(row); self.session.commit(); return row

    def _links(self, principal: Principal, asset_id: str, relationship: AssetRelationship) -> list[AssetEvidenceLink]:
        if not self.get_asset(principal, asset_id):
            return []
        rows = self.session.query(AssetEvidenceLinkRecord).filter_by(asset_id=asset_id, relationship=relationship.value).order_by(AssetEvidenceLinkRecord.created_at).all()
        return [AssetEvidenceLink(id=row.id, asset_id=row.asset_id, evidence_id=row.evidence_id, relationship=AssetRelationship(row.relationship), source=row.source, created_at=self._aware(row.created_at), confidence=row.confidence, inferred=row.inferred) for row in rows]

    def _passport(self, row: AssetRecord) -> AssetPassport:
        aliases = [item.alias for item in self.session.query(AssetAliasRecord).filter(AssetAliasRecord.asset_id == row.id).order_by(AssetAliasRecord.alias).all()]
        return AssetPassport(
            asset_id=row.id, canonical_name=row.canonical_name, asset_type=row.asset_type,
            organization_id=row.organization_id, plant_id=row.plant_id, area_id=row.area_id,
            unit_id=row.unit_id, workspace_id=row.workspace_id, department_id=row.department_id,
            classification=ClearanceLevel.parse(row.classification),
            allowed_roles=[Role(value) for value in json.loads(row.allowed_roles_json or "[]")],
            allowed_users=json.loads(row.allowed_users_json or "[]"), owner_id=row.owner_id,
            criticality=row.criticality, status=row.status, manufacturer=row.manufacturer,
            model=row.model, commissioned_at=self._aware(row.commissioned_at) if row.commissioned_at else None,
            design_parameters=json.loads(row.design_parameters_json or "{}"), tags=json.loads(row.tags_json or "[]"), aliases=aliases,
        )

    @staticmethod
    def _aware(value):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value

    @staticmethod
    def _draft(row: MaintenanceDraftRecord) -> MaintenanceActionDraft:
        return MaintenanceActionDraft(draft_id=row.id, asset_id=row.asset_id, action_type=row.action_type, priority=row.priority, title=row.title, description=row.description, reason_claim_ids=json.loads(row.reason_claim_ids_json or "[]"), status=MaintenanceDraftStatus(row.status), created_by=row.created_by, approval_id=row.approval_id, created_at=AssetRepository._aware(row.created_at))
