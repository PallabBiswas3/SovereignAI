from __future__ import annotations

from app.evidence.models import (
    Calculation,
    Claim,
    EvidenceBundle,
    EvidenceConflict,
    EvidenceFailureCode,
    EvidenceRequirement,
    Measurement,
    Rule,
    SupportStatus,
)
from app.evidence.units import IncompatibleUnitsError, UnitAmbiguousError, UnitService
from app.evidence.verification import NumericalVerifier, VerificationEngine


class EvidenceFirstExecutor:
    """Deterministically resolves evidence before any optional prose synthesis."""

    def __init__(
        self,
        units: UnitService | None = None,
        verification: VerificationEngine | None = None,
    ) -> None:
        self.units = units or UnitService()
        self.verification = verification or VerificationEngine()

    @staticmethod
    def satisfy_requirements(bundle: EvidenceBundle) -> None:
        for requirement in bundle.requirements:
            matches: list[str] = []
            if requirement.type == "asset_identity":
                matches = [item.id for item in bundle.measurements if item.asset_id]
            elif requirement.type == "measurement":
                matches = [item.id for item in bundle.measurements if not requirement.metric or item.metric == requirement.metric]
            elif requirement.type == "rule":
                matches = [
                    item.id for item in bundle.rules
                    if (not requirement.metric or item.metric == requirement.metric)
                    and (not requirement.rule_category or item.rule_type == requirement.rule_category)
                ]
            elif requirement.type == "source_revision":
                matches = [item.id for item in bundle.sources if item.revision]
            requirement.satisfied_by = matches

    def execute(
        self,
        *,
        measurements: list[Measurement],
        rules: list[Rule],
        requirements: list[EvidenceRequirement],
        bundle: EvidenceBundle | None = None,
        conflicts: list[EvidenceConflict] | None = None,
    ) -> EvidenceBundle:
        bundle = bundle or EvidenceBundle()
        bundle.measurements = measurements
        bundle.rules = rules
        bundle.requirements = requirements
        bundle.conflicts = conflicts or bundle.conflicts
        self.satisfy_requirements(bundle)
        missing = [item for item in requirements if not item.satisfied]
        if missing:
            bundle.failures.append(EvidenceFailureCode.insufficient_evidence)
            bundle.claims.append(Claim(
                id="CL-INSUFFICIENT",
                text="The requested conclusion cannot be established from the available authorized evidence.",
                claim_type="evidence_condition",
                evidence_ids=[value for item in requirements for value in item.satisfied_by],
                support_status=SupportStatus.insufficient_evidence,
                support_score=0.0,
            ))
            return bundle
        if bundle.conflicts:
            bundle.failures.append(EvidenceFailureCode.conflicting_evidence)

        calculations: list[Calculation] = []
        claims: list[Claim] = []
        for measurement in measurements:
            applicable = [rule for rule in rules if rule.metric == measurement.metric]
            for rule in applicable:
                calculation_id = f"CALC{len(calculations) + 1}"
                try:
                    normalized = self.units.convert(
                        measurement.original_value, measurement.original_unit, rule.unit
                    )
                    measurement.normalized_value = normalized.normalized_value
                    measurement.normalized_unit = normalized.normalized_unit
                    result = NumericalVerifier._apply(normalized.normalized_value, rule.operator, rule)
                    expression = self._expression(normalized.normalized_value, rule)
                except UnitAmbiguousError:
                    measurement.status = "UNIT_AMBIGUOUS"
                    bundle.failures.append(EvidenceFailureCode.unit_ambiguous)
                    continue
                except IncompatibleUnitsError:
                    measurement.status = "INCOMPATIBLE_UNITS"
                    bundle.failures.append(EvidenceFailureCode.incompatible_units)
                    continue
                calculation = Calculation(
                    id=calculation_id,
                    expression=expression,
                    inputs=[measurement.id, rule.id],
                    result=result,
                )
                calculations.append(calculation)
                conflicting = any(rule.source.source_id in conflict.sources for conflict in bundle.conflicts)
                status = SupportStatus.conflicting_evidence if conflicting else SupportStatus.supported
                comparison = "meets" if result else "exceeds or falls outside"
                claims.append(Claim(
                    id=f"CL{len(claims) + 1}",
                    text=f"{measurement.asset_id or 'The asset'} {comparison} the {rule.rule_type.replace('_', ' ')} for {measurement.metric}.",
                    claim_type="engineering_finding",
                    evidence_ids=[measurement.id, rule.id],
                    calculation_ids=[calculation.id],
                    support_status=status,
                    support_score=1.0 if not conflicting else 0.5,
                ))
        bundle.calculations = calculations
        bundle.claims = claims
        return self.verification.verify(bundle)

    @staticmethod
    def _expression(value: float, rule: Rule) -> str:
        if rule.operator == "between":
            return f"{rule.lower_bound:g} <= {value:g} <= {rule.upper_bound:g}"
        return f"{value:g} {rule.operator} {rule.threshold:g}"
