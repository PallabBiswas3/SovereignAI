from __future__ import annotations

from uuid import uuid4

from app.assets.models import AssetFailureCode, ConditionStatus, OperationalMeasurement
from app.evidence.models import Calculation, EvidenceConflict, Rule
from app.evidence.units import UnitService


class ConditionAssessmentEngine:
    """Evaluates explicit evidence rules; it contains no universal plant limits."""

    def __init__(self, units: UnitService | None = None) -> None:
        self.units = units or UnitService()

    def evaluate(self, measurement: OperationalMeasurement, rule: Rule) -> tuple[ConditionStatus, Calculation]:
        normalized = self.units.convert(measurement.value, measurement.unit, rule.unit)
        value = normalized.normalized_value
        if rule.operator == "between":
            result = bool(rule.lower_bound <= value <= rule.upper_bound)  # type: ignore[operator]
            expression = f"{rule.lower_bound} <= {value} <= {rule.upper_bound} {rule.unit}"
        else:
            threshold = float(rule.threshold)  # validated by Rule
            result = {"<": value < threshold, "<=": value <= threshold, "==": value == threshold,
                      ">=": value >= threshold, ">": value > threshold}[rule.operator]
            expression = f"{value} {rule.operator} {threshold} {rule.unit}"
        status = ConditionStatus.normal if result else ConditionStatus.abnormal
        return status, Calculation(id=f"CAL-{uuid4().hex[:12]}", expression=expression,
                                   inputs=[measurement.measurement_id, rule.id], result=result)


def measurement_discrepancy(left: OperationalMeasurement, right: OperationalMeasurement, tolerance: float = 0.01) -> EvidenceConflict | None:
    if left.asset_id != right.asset_id or left.metric != right.metric:
        return None
    converted = UnitService().convert(right.value, right.unit, left.unit).normalized_value
    if abs(left.value - converted) <= tolerance:
        return None
    return EvidenceConflict(
        id=f"CONFLICT-{uuid4().hex[:12]}", type=AssetFailureCode.measurement_source_conflict.value,
        sources=[left.measurement_id, right.measurement_id],
        summary="Different sources report materially different measurements; timestamps, quality, and measurement method require review.",
        values=[left.model_dump(mode="json"), right.model_dump(mode="json")],
    )
