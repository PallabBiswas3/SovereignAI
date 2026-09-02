from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone

from app.assets.models import (
    AssetFailureCode, FreshnessStatus, HistoricalSeries,
    OperationalMeasurement, TelemetryQuality,
)
from app.assets.repository import AssetRepository
from app.core.database import OperationalMeasurementRecord
from app.evidence.units import UnitService
from app.identity.models import Principal


class FreshnessPolicy:
    def __init__(self, default_freshness_seconds: int = 300, expired_seconds: int = 3600, metric_overrides: dict[str, int] | None = None) -> None:
        self.default = default_freshness_seconds
        self.expired = max(expired_seconds, default_freshness_seconds)
        self.overrides = metric_overrides or {}

    def evaluate(self, timestamp: datetime, metric: str, now: datetime | None = None) -> tuple[float, FreshnessStatus]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None or timestamp.tzinfo is None:
            return 0.0, FreshnessStatus.unknown
        age = max(0.0, (current - timestamp).total_seconds())
        fresh_limit = self.overrides.get(metric, self.default)
        if age <= fresh_limit:
            return age, FreshnessStatus.fresh
        if age <= self.expired:
            return age, FreshnessStatus.stale
        return age, FreshnessStatus.expired


class TelemetryProvider(ABC):
    """Read-only plant-data seam. It intentionally exposes no write or command methods."""

    provider_name: str

    @abstractmethod
    def get_latest(self, principal: Principal, asset_id: str, metrics: list[str] | None = None, *, scenario: str | None = None, as_of: datetime | None = None) -> list[OperationalMeasurement]:
        raise NotImplementedError

    @abstractmethod
    def get_history(self, principal: Principal, asset_id: str, metric: str, start: datetime | None = None, end: datetime | None = None, *, scenario: str | None = None, as_of: datetime | None = None, limit: int = 500) -> HistoricalSeries | None:
        raise NotImplementedError


class APELSimulatorTelemetryProvider(TelemetryProvider):
    provider_name = "apel-readonly-simulator"

    def __init__(self, repository: AssetRepository, freshness: FreshnessPolicy, units: UnitService | None = None, default_scenario: str = "PUMP_102_DEGRADING") -> None:
        self.repository = repository
        self.freshness = freshness
        self.units = units or UnitService()
        self.default_scenario = default_scenario

    def get_latest(self, principal: Principal, asset_id: str, metrics: list[str] | None = None, *, scenario: str | None = None, as_of: datetime | None = None) -> list[OperationalMeasurement]:
        asset = self.repository.get_row(asset_id)
        if not asset or not self.repository.is_authorized(principal, asset, telemetry=True):
            raise PermissionError(AssetFailureCode.asset_access_denied.value)
        rows = self.repository.get_measurements(principal, asset_id, scenario=scenario or self.default_scenario, limit=2000)
        requested = set(metrics or [])
        latest: dict[str, OperationalMeasurementRecord] = {}
        for row in rows:
            if requested and row.metric not in requested:
                continue
            if row.metric not in latest:
                latest[row.metric] = row
        return [self._measurement(row, as_of) for row in sorted(latest.values(), key=lambda item: item.metric)]

    def get_history(self, principal: Principal, asset_id: str, metric: str, start: datetime | None = None, end: datetime | None = None, *, scenario: str | None = None, as_of: datetime | None = None, limit: int = 500) -> HistoricalSeries | None:
        asset = self.repository.get_row(asset_id)
        if not asset or not self.repository.is_authorized(principal, asset, telemetry=True):
            raise PermissionError(AssetFailureCode.asset_access_denied.value)
        rows = self.repository.get_measurements(principal, asset_id, metric, limit, scenario or self.default_scenario)
        values = [self._measurement(row, as_of) for row in reversed(rows) if (start is None or self.repository._aware(row.timestamp) >= start) and (end is None or self.repository._aware(row.timestamp) <= end)]
        if not values:
            return None
        return HistoricalSeries(asset_id=asset_id, metric=metric, unit=values[0].unit, measurements=values, source=self.provider_name)

    def _measurement(self, row: OperationalMeasurementRecord, as_of: datetime | None) -> OperationalMeasurement:
        timestamp = self.repository._aware(row.timestamp)
        normalized = self.units.normalize(row.original_value, row.original_unit)
        age, freshness = self.freshness.evaluate(timestamp, row.metric, as_of)
        quality = TelemetryQuality(row.quality)
        warnings: list[AssetFailureCode] = []
        if quality == TelemetryQuality.bad:
            warnings.append(AssetFailureCode.bad_telemetry_quality)
        elif quality == TelemetryQuality.unknown:
            warnings.append(AssetFailureCode.unknown_telemetry_quality)
        if freshness == FreshnessStatus.stale:
            warnings.append(AssetFailureCode.telemetry_stale)
        elif freshness == FreshnessStatus.expired:
            warnings.append(AssetFailureCode.telemetry_expired)
        return OperationalMeasurement(
            measurement_id=row.id, asset_id=row.asset_id, metric=row.metric,
            value=normalized.normalized_value, unit=normalized.normalized_unit,
            timestamp=timestamp, quality=quality, source=row.source, source_tag=row.source_tag,
            original_value=row.original_value, original_unit=row.original_unit,
            age_seconds=age, freshness_status=freshness, warnings=warnings,
        )


class OPCUAConnector(TelemetryProvider):
    """Read-only future connector contract; no network implementation is enabled in Batch 5."""

    provider_name = "opcua-readonly-unconfigured"

    def get_latest(self, principal: Principal, asset_id: str, metrics: list[str] | None = None, *, scenario: str | None = None, as_of: datetime | None = None) -> list[OperationalMeasurement]:
        raise RuntimeError(AssetFailureCode.telemetry_unavailable.value)

    def get_history(self, principal: Principal, asset_id: str, metric: str, start: datetime | None = None, end: datetime | None = None, *, scenario: str | None = None, as_of: datetime | None = None, limit: int = 500) -> HistoricalSeries | None:
        raise RuntimeError(AssetFailureCode.history_unavailable.value)
