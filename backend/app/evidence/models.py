from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator


class EvidenceFailureCode(str, Enum):
    reranker_unavailable = "RERANKER_UNAVAILABLE"
    insufficient_evidence = "INSUFFICIENT_EVIDENCE"
    conflicting_evidence = "CONFLICTING_EVIDENCE"
    unit_ambiguous = "UNIT_AMBIGUOUS"
    incompatible_units = "INCOMPATIBLE_UNITS"
    context_budget_exceeded = "CONTEXT_BUDGET_EXCEEDED"
    retrieval_failed = "RETRIEVAL_FAILED"
    verification_failed = "VERIFICATION_FAILED"


class SupportStatus(str, Enum):
    supported = "SUPPORTED"
    partially_supported = "PARTIALLY_SUPPORTED"
    unsupported = "UNSUPPORTED"
    conflicting_evidence = "CONFLICTING_EVIDENCE"
    insufficient_evidence = "INSUFFICIENT_EVIDENCE"
    not_applicable = "NOT_APPLICABLE"


class EvidenceSource(BaseModel):
    id: str
    document_id: str | None = None
    file: str
    page: int | None = None
    section: str | None = None
    revision: str | None = None
    document_hash: str | None = None
    text: str
    retrieval_score: float | None = None
    reranker_score: float | None = None
    access_scope: list[str] = Field(default_factory=lambda: ["internal"])


class EvidenceFragment(BaseModel):
    id: str
    source_id: str
    text: str
    scores: dict[str, float | None] = Field(default_factory=dict)
    retrieval_methods: list[str] = Field(default_factory=list)
    technical_identifiers: list[str] = Field(default_factory=list)
    numerical_values: list[str] = Field(default_factory=list)


class Measurement(BaseModel):
    id: str
    asset_id: str | None = None
    metric: str
    original_value: float
    original_unit: str | None
    normalized_value: float | None = None
    normalized_unit: str | None = None
    source_id: str
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)
    status: str = "VALID"

    @model_validator(mode="after")
    def preserve_original_when_normalized(self):
        if self.normalized_value is None:
            self.normalized_value = self.original_value
        if self.normalized_unit is None:
            self.normalized_unit = self.original_unit
        return self


class RuleSource(BaseModel):
    source_id: str
    section: str | None = None
    revision: str | None = None


class Rule(BaseModel):
    id: str
    metric: str
    operator: Literal["<", "<=", "==", ">=", ">", "between"]
    threshold: float | None = None
    lower_bound: float | None = None
    upper_bound: float | None = None
    unit: str | None
    rule_type: str
    source: RuleSource

    @model_validator(mode="after")
    def validate_threshold_shape(self):
        if self.operator == "between":
            if self.lower_bound is None or self.upper_bound is None:
                raise ValueError("A between rule requires lower_bound and upper_bound")
        elif self.threshold is None:
            raise ValueError("A scalar rule requires threshold")
        return self


class Calculation(BaseModel):
    id: str
    engine: str = "deterministic"
    expression: str
    inputs: list[str]
    result: bool | float | str
    verified: bool = True


class Claim(BaseModel):
    id: str
    text: str
    claim_type: str
    evidence_ids: list[str] = Field(default_factory=list)
    calculation_ids: list[str] = Field(default_factory=list)
    support_status: SupportStatus
    support_score: float | None = Field(default=None, ge=0.0, le=1.0)
    verification: list[dict[str, Any]] = Field(default_factory=list)


class Finding(BaseModel):
    id: str
    title: str
    status: str
    claim_ids: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)


class Recommendation(BaseModel):
    id: str
    text: str
    basis_claim_ids: list[str] = Field(default_factory=list)
    requires_human_approval: bool = True


class EvidenceConflict(BaseModel):
    id: str
    type: str = "DOCUMENT_REVISION_CONFLICT"
    sources: list[str]
    status: str = "REQUIRES_REVIEW"
    summary: str
    values: list[dict[str, Any]] = Field(default_factory=list)


class EvidenceRequirement(BaseModel):
    id: str
    type: str
    metric: str | None = None
    rule_category: str | None = None
    required: bool = True
    satisfied_by: list[str] = Field(default_factory=list)

    @property
    def satisfied(self) -> bool:
        return bool(self.satisfied_by) or not self.required


class EvidenceBundle(BaseModel):
    sources: list[EvidenceSource] = Field(default_factory=list)
    fragments: list[EvidenceFragment] = Field(default_factory=list)
    measurements: list[Measurement] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    calculations: list[Calculation] = Field(default_factory=list)
    claims: list[Claim] = Field(default_factory=list)
    findings: list[Finding] = Field(default_factory=list)
    recommendations: list[Recommendation] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    requirements: list[EvidenceRequirement] = Field(default_factory=list)
    failures: list[EvidenceFailureCode] = Field(default_factory=list)

