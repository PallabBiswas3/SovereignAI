from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field

from app.evidence.models import Claim, EvidenceBundle, SupportStatus
from app.evidence.units import IncompatibleUnitsError, UnitAmbiguousError, UnitService


class VerificationResult(BaseModel):
    verifier: str
    passed: bool
    status: str
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class Verifier(ABC):
    name: str

    @abstractmethod
    def verify(self, claim: Claim, bundle: EvidenceBundle) -> VerificationResult:
        raise NotImplementedError


class SchemaVerifier(Verifier):
    name = "schema"

    def verify(self, claim: Claim, bundle: EvidenceBundle) -> VerificationResult:
        passed = bool(claim.id and claim.text and claim.claim_type)
        return VerificationResult(
            verifier=self.name, passed=passed,
            status="VALID" if passed else "VERIFICATION_FAILED",
            summary="Claim schema is complete." if passed else "Claim schema is incomplete.",
        )


class EvidenceVerifier(Verifier):
    name = "evidence"

    def verify(self, claim: Claim, bundle: EvidenceBundle) -> VerificationResult:
        known = {
            item.id for collection in (bundle.sources, bundle.fragments, bundle.measurements, bundle.rules)
            for item in collection
        }
        missing = [value for value in claim.evidence_ids if value not in known]
        passed = bool(claim.evidence_ids) and not missing
        status = "SUPPORTED" if passed else "INSUFFICIENT_EVIDENCE"
        return VerificationResult(
            verifier=self.name, passed=passed, status=status,
            summary="All cited evidence objects exist." if passed else "Claim has missing or no evidence references.",
            details={"missing_evidence_ids": missing},
        )


class UnitVerifier(Verifier):
    name = "unit"

    def __init__(self, units: UnitService | None = None) -> None:
        self.units = units or UnitService()

    def verify(self, claim: Claim, bundle: EvidenceBundle) -> VerificationResult:
        measurements = {item.id: item for item in bundle.measurements}
        rules = {item.id: item for item in bundle.rules}
        checked = 0
        try:
            for calculation_id in claim.calculation_ids:
                calculation = next((item for item in bundle.calculations if item.id == calculation_id), None)
                if not calculation:
                    continue
                measurement = next((measurements[value] for value in calculation.inputs if value in measurements), None)
                rule = next((rules[value] for value in calculation.inputs if value in rules), None)
                if measurement and rule:
                    checked += 1
                    if not self.units.compatible(measurement.original_unit, rule.unit):
                        raise IncompatibleUnitsError("INCOMPATIBLE_UNITS")
        except UnitAmbiguousError as exc:
            return VerificationResult(verifier=self.name, passed=False, status="UNIT_AMBIGUOUS", summary=str(exc))
        except IncompatibleUnitsError as exc:
            return VerificationResult(verifier=self.name, passed=False, status="INCOMPATIBLE_UNITS", summary=str(exc))
        return VerificationResult(
            verifier=self.name, passed=True, status="VALID",
            summary=f"Units are dimensionally compatible for {checked} calculation(s).",
        )


class NumericalVerifier(Verifier):
    name = "numerical"

    def __init__(self, units: UnitService | None = None) -> None:
        self.units = units or UnitService()

    @staticmethod
    def _apply(value: float, operator: str, rule) -> bool:
        if operator == "between":
            assert rule.lower_bound is not None and rule.upper_bound is not None
            return rule.lower_bound <= value <= rule.upper_bound
        assert rule.threshold is not None
        return {"<": value < rule.threshold, "<=": value <= rule.threshold,
                "==": value == rule.threshold, ">=": value >= rule.threshold,
                ">": value > rule.threshold}[operator]

    def verify(self, claim: Claim, bundle: EvidenceBundle) -> VerificationResult:
        measurements = {item.id: item for item in bundle.measurements}
        rules = {item.id: item for item in bundle.rules}
        calculations = {item.id: item for item in bundle.calculations}
        checked = 0
        for calculation_id in claim.calculation_ids:
            calculation = calculations.get(calculation_id)
            if not calculation:
                return VerificationResult(
                    verifier=self.name, passed=False, status="VERIFICATION_FAILED",
                    summary=f"Calculation '{calculation_id}' is missing.",
                )
            measurement = next((measurements[value] for value in calculation.inputs if value in measurements), None)
            rule = next((rules[value] for value in calculation.inputs if value in rules), None)
            if not measurement or not rule:
                return VerificationResult(
                    verifier=self.name, passed=False, status="INSUFFICIENT_EVIDENCE",
                    summary="Calculation inputs do not resolve to a measurement and rule.",
                )
            try:
                normalized = self.units.convert(
                    measurement.original_value, measurement.original_unit, rule.unit
                )
            except UnitAmbiguousError as exc:
                return VerificationResult(verifier=self.name, passed=False, status="UNIT_AMBIGUOUS", summary=str(exc))
            except IncompatibleUnitsError as exc:
                return VerificationResult(verifier=self.name, passed=False, status="INCOMPATIBLE_UNITS", summary=str(exc))
            expected = self._apply(normalized.normalized_value, rule.operator, rule)
            checked += 1
            if bool(calculation.result) != expected:
                return VerificationResult(
                    verifier=self.name, passed=False, status="UNSUPPORTED",
                    summary="Stored calculation result is numerically inconsistent.",
                    details={"expected": expected, "actual": calculation.result},
                )
        return VerificationResult(
            verifier=self.name, passed=True, status="VALID",
            summary=f"Recomputed {checked} calculation(s) deterministically.",
        )


class RuleVerifier(Verifier):
    name = "rule"

    def verify(self, claim: Claim, bundle: EvidenceBundle) -> VerificationResult:
        calculation_ids = {item.id for item in bundle.calculations if item.verified}
        missing = [value for value in claim.calculation_ids if value not in calculation_ids]
        passed = not missing
        return VerificationResult(
            verifier=self.name, passed=passed,
            status="VALID" if passed else "VERIFICATION_FAILED",
            summary="Applicable deterministic rules are linked." if passed else "A linked calculation is absent or unverified.",
            details={"missing_calculation_ids": missing},
        )


class SemanticVerifier(Verifier):
    name = "semantic"

    def verify(self, claim: Claim, bundle: EvidenceBundle) -> VerificationResult:
        # Deterministic claims already carry exact evidence/calculation links. Free-form semantic
        # verification remains in GroundingChecker and is intentionally not mislabeled as proof.
        linked = bool(claim.evidence_ids or claim.calculation_ids)
        return VerificationResult(
            verifier=self.name, passed=linked,
            status="SUPPORTED" if linked else "UNSUPPORTED",
            summary="Claim has explicit evidence lineage." if linked else "Claim has no semantic evidence lineage.",
        )


class VerificationEngine:
    def __init__(self, verifiers: list[Verifier] | None = None) -> None:
        units = UnitService()
        self.verifiers = verifiers or [
            SchemaVerifier(), EvidenceVerifier(), NumericalVerifier(units),
            UnitVerifier(units), RuleVerifier(), SemanticVerifier(),
        ]

    def verify_claim(self, claim: Claim, bundle: EvidenceBundle) -> Claim:
        results: list[VerificationResult] = []
        for verifier in self.verifiers:
            result = verifier.verify(claim, bundle)
            results.append(result)
            if not result.passed and result.status in {
                "INSUFFICIENT_EVIDENCE", "INCOMPATIBLE_UNITS", "UNIT_AMBIGUOUS",
                "UNSUPPORTED", "VERIFICATION_FAILED",
            }:
                break
        claim.verification = [item.model_dump(mode="json") for item in results]
        failure = next((item for item in results if not item.passed), None)
        if failure:
            if failure.status == "INSUFFICIENT_EVIDENCE":
                claim.support_status = SupportStatus.insufficient_evidence
            else:
                claim.support_status = SupportStatus.unsupported
            claim.support_score = 0.0
        elif claim.support_status not in {SupportStatus.conflicting_evidence, SupportStatus.not_applicable}:
            claim.support_status = SupportStatus.supported
            claim.support_score = 1.0
        return claim

    def verify(self, bundle: EvidenceBundle) -> EvidenceBundle:
        bundle.claims = [self.verify_claim(claim, bundle) for claim in bundle.claims]
        return bundle
