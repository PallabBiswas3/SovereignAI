from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.evidence.models import Calculation, EvidenceConflict, Rule
from app.identity.models import ClearanceLevel, Role


class AssetModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class AssetFailureCode(str, Enum):
    asset_not_found = "ASSET_NOT_FOUND"
    ambiguous_asset = "AMBIGUOUS_ASSET"
    asset_access_denied = "ASSET_ACCESS_DENIED"
    telemetry_unavailable = "TELEMETRY_UNAVAILABLE"
    telemetry_stale = "TELEMETRY_STALE"
    telemetry_expired = "TELEMETRY_EXPIRED"
    bad_telemetry_quality = "BAD_TELEMETRY_QUALITY"
    unknown_telemetry_quality = "UNKNOWN_TELEMETRY_QUALITY"
    history_unavailable = "HISTORY_UNAVAILABLE"
    insufficient_history = "INSUFFICIENT_HISTORY"
    measurement_source_conflict = "MEASUREMENT_SOURCE_CONFLICT"
    maintenance_history_unavailable = "MAINTENANCE_HISTORY_UNAVAILABLE"
    maintenance_draft_failed = "MAINTENANCE_DRAFT_FAILED"


class Criticality(str, Enum):
    low = "LOW"
    medium = "MEDIUM"
    high = "HIGH"
    critical = "CRITICAL"


class AssetStatus(str, Enum):
    in_service = "IN_SERVICE"
    standby = "STANDBY"
    out_of_service = "OUT_OF_SERVICE"
    retired = "RETIRED"
    unknown = "UNKNOWN"


class TelemetryQuality(str, Enum):
    good = "GOOD"
    uncertain = "UNCERTAIN"
    bad = "BAD"
    unknown = "UNKNOWN"


class FreshnessStatus(str, Enum):
    fresh = "FRESH"
    stale = "STALE"
    expired = "EXPIRED"
    unknown = "UNKNOWN"


class TrendDirection(str, Enum):
    increasing = "INCREASING"
    decreasing = "DECREASING"
    flat = "FLAT"
    insufficient_history = "INSUFFICIENT_HISTORY"


class ConditionStatus(str, Enum):
    normal = "NORMAL"
    attention = "ATTENTION"
    abnormal = "ABNORMAL"
    critical = "CRITICAL"
    unknown = "UNKNOWN"


class Plant(AssetModel):
    id: str
    organization_id: str
    name: str


class Area(AssetModel):
    id: str
    organization_id: str
    plant_id: str
    name: str


class Unit(AssetModel):
    id: str
    organization_id: str
    plant_id: str
    area_id: str
    name: str


class AssetPassport(AssetModel):
    asset_id: str
    canonical_name: str
    asset_type: str
    organization_id: str
    plant_id: str
    area_id: str
    unit_id: str
    workspace_id: str
    department_id: str | None = None
    classification: ClearanceLevel = ClearanceLevel.internal
    allowed_roles: list[Role] = Field(default_factory=list)
    allowed_users: list[str] = Field(default_factory=list)
    owner_id: str | None = None
    criticality: Criticality
    status: AssetStatus
    manufacturer: str | None = None
    model: str | None = None
    commissioned_at: datetime | None = None
    design_parameters: dict[str, Any] = Field(default_factory=dict)
    tags: list[str] = Field(default_factory=list)
    aliases: list[str] = Field(default_factory=list)


class AssetReference(AssetModel):
    value: str


class AssetResolution(AssetModel):
    status: str
    reference: str
    asset: AssetPassport | None = None
    candidate_ids: list[str] = Field(default_factory=list)


class AssetRelationship(str, Enum):
    has_document = "HAS_DOCUMENT"
    has_inspection = "HAS_INSPECTION"
    has_measurement = "HAS_MEASUREMENT"
    has_finding = "HAS_FINDING"
    supported_by = "SUPPORTED_BY"
    evaluated_using = "EVALUATED_USING"
    derived_from = "DERIVED_FROM"


class AssetEvidenceLink(AssetModel):
    id: str
    asset_id: str
    evidence_id: str
    relationship: AssetRelationship
    source: str
    created_at: datetime
    confidence: float = Field(ge=0.0, le=1.0)
    inferred: bool = False


class InspectionRecord(AssetModel):
    id: str
    asset_id: str
    inspected_at: datetime
    source_document_id: str | None = None
    summary: str
    measurement_ids: list[str] = Field(default_factory=list)


class OperationalMeasurement(AssetModel):
    measurement_id: str
    asset_id: str
    metric: str
    value: float
    unit: str
    timestamp: datetime
    quality: TelemetryQuality
    source: str
    source_tag: str
    original_value: float
    original_unit: str
    age_seconds: float | None = Field(default=None, ge=0.0)
    freshness_status: FreshnessStatus = FreshnessStatus.unknown
    warnings: list[AssetFailureCode] = Field(default_factory=list)

    @model_validator(mode="after")
    def timestamp_must_be_aware(self):
        if self.timestamp.tzinfo is None or self.timestamp.utcoffset() is None:
            raise ValueError("OperationalMeasurement timestamp must include a timezone")
        return self


class HistoricalSeries(AssetModel):
    asset_id: str
    metric: str
    unit: str
    measurements: list[OperationalMeasurement]
    source: str


class TrendWindow(AssetModel):
    start: datetime
    end: datetime


class TrendAnalysis(AssetModel):
    asset_id: str
    metric: str
    unit: str
    window: TrendWindow
    sample_count: int
    latest: float
    mean: float
    minimum: float
    maximum: float
    change: float
    percentage_change: float | None = None
    slope_per_day: float
    rolling_mean: float
    threshold_crossings: int = 0
    time_above_threshold_seconds: float = 0.0
    abnormal_readings: int = 0
    trend: TrendDirection
    engine: str = "deterministic"


class AssetFinding(AssetModel):
    id: str
    asset_id: str
    title: str
    condition: ConditionStatus
    measurement_ids: list[str]
    rule_ids: list[str]
    calculation_ids: list[str]
    trend_metrics: list[str] = Field(default_factory=list)
    status: str = "OPEN"


class AssetRecommendation(AssetModel):
    id: str
    asset_id: str
    text: str
    finding_ids: list[str]
    requires_human_approval: bool = True


class MaintenanceRecord(AssetModel):
    id: str
    asset_id: str
    occurred_at: datetime
    event_type: str
    title: str
    summary: str
    source: str
    status: str = "COMPLETED"


class MaintenanceDraftStatus(str, Enum):
    draft = "DRAFT"
    approved = "APPROVED"
    rejected = "REJECTED"


class MaintenanceActionDraft(AssetModel):
    draft_id: str
    asset_id: str
    action_type: str
    priority: str
    title: str
    description: str
    reason_claim_ids: list[str]
    status: MaintenanceDraftStatus = MaintenanceDraftStatus.draft
    created_by: str
    approval_required: bool = True
    approval_id: str | None = None
    created_at: datetime


class AssetContext(AssetModel):
    asset: AssetPassport
    latest_measurements: list[OperationalMeasurement] = Field(default_factory=list)
    trends: list[TrendAnalysis] = Field(default_factory=list)
    inspections: list[InspectionRecord] = Field(default_factory=list)
    document_ids: list[str] = Field(default_factory=list)
    findings: list[AssetFinding] = Field(default_factory=list)
    recommendations: list[AssetRecommendation] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    maintenance: list[MaintenanceRecord] = Field(default_factory=list)
    maintenance_drafts: list[MaintenanceActionDraft] = Field(default_factory=list)
    warnings: list[AssetFailureCode] = Field(default_factory=list)
    provider: str
    compiled_at: datetime
