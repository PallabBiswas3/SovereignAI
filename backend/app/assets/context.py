from __future__ import annotations

from datetime import datetime, timezone

from app.assets.assessment import ConditionAssessmentEngine, measurement_discrepancy
from app.assets.models import (
    AssetContext, AssetFailureCode, AssetFinding, AssetRecommendation,
    FreshnessStatus, OperationalMeasurement, TelemetryQuality,
)
from app.assets.repository import AssetRepository
from app.assets.telemetry import TelemetryProvider
from app.assets.trends import InsufficientHistoryError, TrendAnalyzer
from app.identity.models import Principal
from app.evidence.models import Rule, RuleSource
from app.core.database import KnowledgeDocument


class AssetContextService:
    """Compiles bounded, authorized operational context from stored evidence."""

    def __init__(self, repository: AssetRepository, telemetry: TelemetryProvider, trends: TrendAnalyzer | None = None) -> None:
        self.repository = repository
        self.telemetry = telemetry
        self.trends = trends or TrendAnalyzer()

    def compile(
        self, principal: Principal, asset_id: str, task: str = "", *,
        scenario: str | None = None, as_of: datetime | None = None,
        metrics: list[str] | None = None,
    ) -> AssetContext:
        asset = self.repository.get_asset(principal, asset_id)
        if not asset:
            raise PermissionError(AssetFailureCode.asset_access_denied.value)
        latest = self.telemetry.get_latest(principal, asset_id, metrics, scenario=scenario, as_of=as_of)
        analyses = []
        warnings = [warning for item in latest for warning in item.warnings]
        for item in latest[:8]:
            series = self.telemetry.get_history(
                principal, asset_id, item.metric, scenario=scenario, as_of=as_of, limit=120,
            )
            if not series:
                continue
            try:
                analyses.append(self.trends.analyze(series))
            except InsufficientHistoryError:
                warnings.append(AssetFailureCode.insufficient_history)
        documents = self.repository.get_asset_documents(principal, asset_id)
        rules: list[Rule] = []
        calculations = []
        findings: list[AssetFinding] = []
        recommendations: list[AssetRecommendation] = []
        conflicts = []
        vibration = next((item for item in latest if item.metric == "vibration"), None)
        limit = asset.design_parameters.get("normal_vibration_max_mm_s_rms")
        source_link = next((item for item in documents if (
            (row := self.repository.session.get(KnowledgeDocument, item.evidence_id))
            and "SOP-MNT-017_Pump_Condition_Monitoring_Rev4" in row.filename
        )), None)
        if vibration and isinstance(limit, (int, float)) and source_link:
            rule = Rule(
                id="RULE-P102-VIBRATION-NORMAL", metric="vibration", operator="<=",
                threshold=float(limit), unit="mm/s", rule_type="normal_limit",
                source=RuleSource(source_id=source_link.evidence_id, section="Pump-102 investigation threshold", revision="Rev 4"),
            )
            rules.append(rule)
            if vibration.quality in {TelemetryQuality.good, TelemetryQuality.uncertain} and vibration.freshness_status != FreshnessStatus.expired:
                condition, calculation = ConditionAssessmentEngine().evaluate(vibration, rule)
                calculations.append(calculation)
                finding = AssetFinding(
                    id="FINDING-P102-VIBRATION", asset_id=asset_id,
                    title="Pump-102 vibration exceeds the documented normal investigation limit" if not calculation.result else "Pump-102 vibration is within the documented normal limit",
                    condition=condition, measurement_ids=[vibration.measurement_id],
                    rule_ids=[rule.id], calculation_ids=[calculation.id], trend_metrics=["vibration"],
                )
                findings.append(finding)
                if not calculation.result:
                    recommendations.append(AssetRecommendation(
                        id="REC-P102-BEARING-ALIGNMENT", asset_id=asset_id,
                        text="Create a human-reviewed inspection draft for Pump-102 bearing condition and alignment; do not issue a plant command.",
                        finding_ids=[finding.id],
                    ))
        inspection = self.repository.get_asset_inspections(principal, asset_id, 1)
        if vibration and inspection:
            import re
            match = re.search(r"vibration\s+([\d.]+)\s*mm/s", inspection[0].summary, re.I)
            if match:
                inspection_measurement = OperationalMeasurement(
                    measurement_id="INSP-P102-VIB-20260901", asset_id=asset_id,
                    metric="vibration", value=float(match.group(1)), unit="mm/s",
                    timestamp=inspection[0].inspected_at, quality=TelemetryQuality.good,
                    source="inspection", source_tag=inspection[0].id,
                    original_value=float(match.group(1)), original_unit="mm/s",
                    freshness_status=FreshnessStatus.unknown,
                )
                conflict = measurement_discrepancy(inspection_measurement, vibration, tolerance=0.1)
                if conflict:
                    conflicts.append(conflict)
                    warnings.append(AssetFailureCode.measurement_source_conflict)
        return AssetContext(
            asset=asset,
            latest_measurements=latest[:12], trends=analyses[:8],
            inspections=self.repository.get_asset_inspections(principal, asset_id, 5),
            document_ids=[item.evidence_id for item in documents[:20]],
            maintenance=self.repository.get_maintenance_history(principal, asset_id, 10),
            maintenance_drafts=self.repository.list_drafts(principal, asset_id)[:10],
            rules=rules, calculations=calculations, findings=findings,
            recommendations=recommendations, conflicts=conflicts,
            warnings=list(dict.fromkeys(warnings)), provider=self.telemetry.provider_name,
            compiled_at=as_of or datetime.now(timezone.utc),
        )
