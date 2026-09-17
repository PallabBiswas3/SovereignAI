"""
Common schema for the Latency-Aware Adaptive Factuality / Hallucination
Verification project.

This is the backbone of the whole project (per §49 of the master project
prompt, and the schema discussion in chat): every dataset loader converts
raw data into a `ResponseRecord`, every pipeline stage reads/writes one,
and every downstream ablation or analysis (numeric vs entity vs date
hallucination, which claim types get escalated, etc.) depends on these
fields being populated consistently.

Design notes:
- Every metadata block is optional at construction time because different
  pipeline stages fill them in progressively (generation -> risk ->
  verification -> timing). A record loaded straight from a dataset will
  only have `generation_metadata` and ground truth populated.
- `latency_ms` and `cost` are tracked at BOTH the claim level and the
  response level, so per-claim-type cost/latency ablations (§52) can be
  computed directly from stored records instead of re-derived later.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class ClaimType(str, Enum):
    """Coarse category of an atomic claim. Used for per-type ablations
    (§6 hallucination taxonomy, §52 Experiments 2-4)."""

    NUMERIC = "numeric"
    ENTITY = "entity"
    DATE = "date"
    CITATION = "citation"
    FACTUAL = "factual"  # general factual claim, no specific sub-type
    OTHER = "other"


class VerificationStatus(str, Enum):
    """Output of Problem C (Factual Verification), per §5 / §35."""

    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"
    UNVERIFIED = "unverified"  # not yet processed (default state)


class ResponseLabel(str, Enum):
    """Response-level ground truth or final decision label.
    Mirrors §37 Final System Decision (ACCEPT / CORRECT / ABSTAIN) for
    predictions, and a binary-ish ground truth for dataset labels."""

    SUPPORTED = "supported"
    HALLUCINATED = "hallucinated"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class VerificationRoute(str, Enum):
    """Which tier of the adaptive router a response was sent through.
    Per §19 Adaptive Routing / §64 Final Target Architecture."""

    FAST_ACCEPT = "fast_accept"
    LIGHTWEIGHT = "lightweight"
    AGENTIC = "agentic"
    NOT_ROUTED = "not_routed"  # ground-truth-only record, no pipeline run yet


class RiskModelVersion(str, Enum):
    V1 = "v1"  # provider-independent (text/query/embedding/entity/number/date)
    V2 = "v2"  # + logit-aware features
    V3 = "v3"  # + hidden-state probe
    NONE = "none"


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class HallucinationSpan(BaseModel):
    """Raw ground-truth span annotation, preserved as-is from source
    datasets that provide span-level labels (e.g. RAGTruth). Kept
    separate from `Claim` because claim extraction (Phase 5/6) is a
    pipeline component, not a dataset property -- this is ground truth,
    not a prediction."""

    start: int
    end: int
    text: str
    label_type: str  # dataset-specific raw label, e.g. "Evident Conflict"
    normalized_type: Optional[VerificationStatus] = None
    meta: Optional[str] = None


class Claim(BaseModel):
    """A single atomic factual claim extracted from a response, and the
    result of verifying it. Per schema discussion in chat: every claim
    carries its own latency/cost so per-claim-type ablations don't
    require re-running the pipeline."""

    id: str
    text: str
    type: ClaimType = ClaimType.OTHER

    entities: list[str] = Field(default_factory=list)
    numbers: list[str] = Field(default_factory=list)
    dates: list[str] = Field(default_factory=list)

    # character offsets into the parent response, if known
    span: Optional[tuple[int, int]] = None

    risk_score: Optional[float] = None
    status: VerificationStatus = VerificationStatus.UNVERIFIED
    confidence: Optional[float] = None

    evidence: list[str] = Field(default_factory=list)
    verifier_used: Optional[str] = None  # e.g. "deterministic_numeric", "nli:model_x", "agent"
    verification_scores: dict[str, float] = Field(default_factory=dict)

    latency_ms: Optional[float] = None
    cost: Optional[float] = None


class GenerationMetadata(BaseModel):
    """Everything about how the response was generated."""

    model: Optional[str] = None
    temperature: Optional[float] = None
    task_type: Optional[str] = None  # e.g. "Summary", "QA", "Data2txt"
    quality: Optional[str] = None  # dataset-specific QC flag, e.g. "good"
    logits_available: bool = False


class RiskMetadata(BaseModel):
    """Output of Problem A (Risk Estimation). Per §9: r is a routing
    signal, NOT a factual-correctness judgment -- keep it structurally
    separate from VerificationMetadata."""

    risk_score: Optional[float] = None  # raw, uncalibrated
    calibrated_risk_score: Optional[float] = None
    model_version: RiskModelVersion = RiskModelVersion.NONE
    feature_values: dict[str, float] = Field(default_factory=dict)
    latency_ms: Optional[float] = None


class VerificationMetadata(BaseModel):
    """Output of the routing + verification stages."""

    route: VerificationRoute = VerificationRoute.NOT_ROUTED
    claims_checked: int = 0
    llm_call_count: int = 0
    search_call_count: int = 0
    retrieval_call_count: int = 0
    agent_iterations: int = 0
    final_decision: Optional[ResponseLabel] = None


class TimingMetadata(BaseModel):
    """Wall-clock latency breakdown. Per the correction in chat: this
    must be REAL wall-clock time around complete operations (network,
    serialization, everything included), not framework-reported
    inference time, or latency comparisons become meaningless."""

    generation_latency_ms: Optional[float] = None
    risk_latency_ms: Optional[float] = None
    verification_latency_ms: Optional[float] = None
    total_latency_ms: Optional[float] = None
    stage_breakdown_ms: dict[str, float] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Top-level record
# ---------------------------------------------------------------------------


class ResponseRecord(BaseModel):
    """The common unit of data for the whole project. One instance per
    (query, response) pair, whether it came from a benchmark dataset or
    was produced live by the pipeline."""

    id: str
    dataset: str  # e.g. "ragtruth", "halueval"
    source_id: Optional[str] = None  # dataset-internal id for the source document, if any
    split: Optional[str] = None  # "train" / "test" / "dev"

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
