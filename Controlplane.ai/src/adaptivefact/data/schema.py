"""Common schema for AdaptiveFact and ControlPlane factuality evaluation.

The schema deliberately separates risk estimation from factual verification.
Factuality v3 adds richer claim states and a structured/decontextualized claim
representation while remaining backward compatible with older datasets that
use ``unknown``.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


class ClaimType(str, Enum):
    NUMERIC = "numeric"
    ENTITY = "entity"
    DATE = "date"
    CITATION = "citation"
    FACTUAL = "factual"
    OTHER = "other"


class VerificationStatus(str, Enum):
    """Claim-level factuality state.

    ``UNKNOWN`` is retained for legacy Phase-5/6 datasets. New ControlPlane v3
    verification should prefer ``UNDECIDABLE`` when evidence is insufficient,
    ``UNSUPPORTED`` when sufficiently complete evidence fails to support a
    checkable claim, and ``CONFLICTING`` when strong sources disagree.
    """

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNSUPPORTED = "unsupported"
    UNDECIDABLE = "undecidable"
    CONFLICTING = "conflicting"
    UNKNOWN = "unknown"  # legacy unresolved state
    UNVERIFIED = "unverified"


class ResponseLabel(str, Enum):
    SUPPORTED = "supported"
    HALLUCINATED = "hallucinated"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class VerificationRoute(str, Enum):
    FAST_ACCEPT = "fast_accept"
    LIGHTWEIGHT = "lightweight"
    AGENTIC = "agentic"
    NOT_ROUTED = "not_routed"


class RiskModelVersion(str, Enum):
    V1 = "v1"
    V2 = "v2"
    V3 = "v3"
    NONE = "none"


class HallucinationSpan(BaseModel):
    start: int
    end: int
    text: str
    label_type: str
    normalized_type: Optional[VerificationStatus] = None
    meta: Optional[str] = None


class Claim(BaseModel):
    """One independently verifiable factual claim.

    ``text`` remains the original extracted unit for compatibility and span
    evaluation. ``decontextualized_text`` is the self-contained form used for
    retrieval/checking. Subject/predicate/object fields are lightweight
    RefChecker-style structure: they are useful for retrieval, diagnostics and
    evaluation but are not assumed to be a perfect semantic parse.
    """

    id: str
    text: str
    type: ClaimType = ClaimType.OTHER

    original_text: Optional[str] = None
    decontextualized_text: Optional[str] = None
    subject: Optional[str] = None
    predicate: Optional[str] = None
    object: Optional[str] = None
    qualifiers: dict[str, str] = Field(default_factory=dict)
    extraction_method: str = "heuristic"
    parent_span: Optional[tuple[int, int]] = None

    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)
    span: Optional[tuple[int, int]] = None

    risk_score: Optional[float] = None
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    confidence: Optional[float] = None

    evidence: list[str] = Field(default_factory=list)
    evidence_ids: list[str] = Field(default_factory=list)
    verifier_used: Optional[str] = None
    verification_scores: dict[str, float] = Field(default_factory=dict)
    verification_metadata: dict[str, Any] = Field(default_factory=dict)

    latency_ms: Optional[float] = None
    cost: Optional[float] = None

    @property
    def verification_text(self) -> str:
        return (self.decontextualized_text or self.text).strip()


class GenerationMetadata(BaseModel):
    model: Optional[str] = None
    temperature: Optional[float] = None
    task_type: Optional[str] = None
    quality: Optional[str] = None
    logits_available: bool = False


class RiskMetadata(BaseModel):
    """Routing signal only; never a factual truth judgment."""

    risk_score: Optional[float] = None
    calibrated_risk_score: Optional[float] = None
    model_version: RiskModelVersion = RiskModelVersion.NONE
    feature_values: dict[str, float] = Field(default_factory=dict)
    latency_ms: Optional[float] = None


class VerificationMetadata(BaseModel):
    route: VerificationRoute = VerificationRoute.NOT_ROUTED
    claims_checked: int = 0
    llm_call_count: int = 0
    search_call_count: int = 0
    retrieval_call_count: int = 0
    agent_iterations: int = 0
    final_decision: Optional[ResponseLabel] = None


class TimingMetadata(BaseModel):
    generation_latency_ms: Optional[float] = None
    risk_latency_ms: Optional[float] = None
    verification_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    stage_breakdown_ms: dict[str, float] = Field(default_factory=dict)


class ResponseRecord(BaseModel):
    id: str
    dataset: str
    source_id: Optional[str] = None
    split: Optional[str] = None

    query: str
    context: Optional[str] = None
    generated_response: str

    ground_truth_label: Optional[ResponseLabel] = None
    ground_truth_spans: list[HallucinationSpan] = Field(default_factory=list)

    atomic_claims: list[Claim] = Field(default_factory=list)
    supporting_evidence: list[str] = Field(default_factory=list)

    generation_metadata: GenerationMetadata = Field(default_factory=GenerationMetadata)
    risk_metadata: Optional[RiskMetadata] = None
    verification_metadata: Optional[VerificationMetadata] = None
    timing_metadata: Optional[TimingMetadata] = None

    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(use_enum_values=False)
