import pytest

from app.evidence.executor import EvidenceFirstExecutor
from app.evidence.models import (
    Calculation,
    Claim,
    EvidenceBundle,
    EvidenceConflict,
    EvidenceFailureCode,
    EvidenceRequirement,
    Measurement,
    Rule,
    RuleSource,
    SupportStatus,
)
from app.evidence.units import IncompatibleUnitsError, UnitAmbiguousError, UnitService
from app.evidence.verification import VerificationEngine


def _pressure_measurement() -> Measurement:
    return Measurement(
        id="M1", asset_id="Pump-102", metric="pressure",
        original_value=500, original_unit="kPa", source_id="E1",
    )


def _pressure_rule() -> Rule:
    return Rule(
        id="R1", metric="pressure", operator="between",
        lower_bound=4.8, upper_bound=5.5, unit="bar",
        rule_type="normal_range", source=RuleSource(source_id="E2", section="7.6", revision="Rev 3"),
    )


def test_unit_service_normalizes_pressure_and_preserves_original() -> None:
    result = UnitService().convert(500, "kPa", "bar")
    assert result.original_value == 500
    assert result.original_unit == "kPa"
    assert result.normalized_value == 5.0
    assert result.normalized_unit == "bar"
    assert result.dimension == "pressure"


def test_unit_service_supports_required_engineering_dimensions() -> None:
    units = UnitService()
    assert units.convert(293.15, "K", "°C").normalized_value == pytest.approx(20.0)
    assert units.convert(1000, "mm", "m").normalized_value == 1.0
    assert units.convert(1, "m/s", "mm/s").normalized_value == 1000.0
    assert units.convert(1000, "W", "kW").normalized_value == 1.0
    assert units.convert(1000, "N", "kN").normalized_value == 1.0
    assert units.convert(1000, "g", "kg").normalized_value == 1.0
    assert units.convert(60, "rpm", "Hz").normalized_value == pytest.approx(1.0)


def test_missing_and_incompatible_units_fail_explicitly() -> None:
    units = UnitService()
    with pytest.raises(UnitAmbiguousError, match="UNIT_AMBIGUOUS"):
        units.convert(5, None, "bar")
    with pytest.raises(IncompatibleUnitsError, match="INCOMPATIBLE_UNITS"):
        units.convert(5, "bar", "mm/s")


def test_evidence_requirements_return_insufficient_evidence() -> None:
    requirements = [
        EvidenceRequirement(id="REQ1", type="measurement", metric="pressure"),
        EvidenceRequirement(id="REQ2", type="rule", metric="pressure", rule_category="normal_range"),
    ]
    bundle = EvidenceFirstExecutor().execute(
        measurements=[_pressure_measurement()], rules=[], requirements=requirements,
    )
    assert EvidenceFailureCode.insufficient_evidence in bundle.failures
    assert bundle.claims[0].support_status == SupportStatus.insufficient_evidence
    assert requirements[0].satisfied
    assert not requirements[1].satisfied


def test_evidence_first_executor_normalizes_calculates_and_verifies() -> None:
    bundle = EvidenceFirstExecutor().execute(
        measurements=[_pressure_measurement()],
        rules=[_pressure_rule()],
        requirements=[
            EvidenceRequirement(id="REQ1", type="measurement", metric="pressure"),
            EvidenceRequirement(id="REQ2", type="rule", metric="pressure", rule_category="normal_range"),
        ],
    )
    assert bundle.measurements[0].original_value == 500
    assert bundle.measurements[0].normalized_value == 5.0
    assert bundle.measurements[0].normalized_unit == "bar"
    assert bundle.calculations[0].result is True
    assert bundle.calculations[0].expression == "4.8 <= 5 <= 5.5"
    assert bundle.claims[0].support_status == SupportStatus.supported
    assert all(item["passed"] for item in bundle.claims[0].verification)


def test_verification_rejects_numerically_inconsistent_claim() -> None:
    measurement, rule = _pressure_measurement(), _pressure_rule()
    calculation = Calculation(
        id="CALC1", expression="4.8 <= 5 <= 5.5", inputs=["M1", "R1"], result=False,
    )
    claim = Claim(
        id="CL1", text="Pump-102 pressure is outside the normal range.",
        claim_type="engineering_finding", evidence_ids=["M1", "R1"],
        calculation_ids=["CALC1"], support_status=SupportStatus.supported,
    )
    bundle = EvidenceBundle(
        measurements=[measurement], rules=[rule], calculations=[calculation], claims=[claim],
    )
    verified = VerificationEngine().verify(bundle)
    assert verified.claims[0].support_status == SupportStatus.unsupported
    assert any(item["verifier"] == "numerical" and not item["passed"] for item in verified.claims[0].verification)


def test_conflicting_revision_marks_claim_for_review() -> None:
    conflict = EvidenceConflict(
        id="C1", sources=["E2", "E3"],
        summary="Rev 2 and Rev 3 contain different pressure limits.",
    )
    bundle = EvidenceFirstExecutor().execute(
        measurements=[_pressure_measurement()], rules=[_pressure_rule()],
        requirements=[EvidenceRequirement(id="REQ1", type="measurement", metric="pressure")],
        conflicts=[conflict],
    )
    assert EvidenceFailureCode.conflicting_evidence in bundle.failures
    assert bundle.claims[0].support_status == SupportStatus.conflicting_evidence
