from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field, model_validator

from adaptivefact.data.schema import VerificationStatus


class SearchResult(BaseModel):
    source_id: str
    source_name: str
    text: str
    url: str | None = None
    authority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    retrieval_score: float = Field(default=0.0, ge=0.0)
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict = Field(default_factory=dict)


class AgentStep(BaseModel):
    iteration: int
    query: str
    tool_names: list[str] = Field(default_factory=list)
    results: list[SearchResult] = Field(default_factory=list)
    best_entailment: float = 0.0
    best_contradiction: float = 0.0
    decision: VerificationStatus = VerificationStatus.UNKNOWN
    latency_ms: float = 0.0


class AgentVerificationConfig(BaseModel):
    max_iterations: int = Field(default=3, ge=1, le=10)
    max_search_calls: int = Field(default=3, ge=1, le=25)
    max_nli_pairs: int = Field(default=20, ge=1, le=200)
    max_sources: int = Field(default=5, ge=1, le=25)
    top_k_per_search: int = Field(default=3, ge=1, le=10)
    max_total_latency_ms: float = Field(default=10_000.0, gt=0.0)
    max_cost: float = Field(default=0.02, ge=0.0)
    search_cost_per_call: float = Field(default=0.0, ge=0.0)
    entailment_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    contradiction_threshold: float = Field(default=0.88, ge=0.0, le=1.0)
    decision_margin: float = Field(default=0.20, ge=0.0, le=1.0)
    min_authority_score: float = Field(default=0.60, ge=0.0, le=1.0)
    min_retrieval_score: float = Field(default=0.10, ge=0.0)
    require_citations: bool = True

    @model_validator(mode="after")
    def validate_cost_budget(self) -> "AgentVerificationConfig":
        minimum_call_cost = self.search_cost_per_call * min(self.max_search_calls, 1)
        if minimum_call_cost > self.max_cost:
            raise ValueError("max_cost is lower than the cost of one permitted search call")
        return self


class AgentVerificationResult(BaseModel):
    claim_id: str
    status: VerificationStatus
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    evidence: list[SearchResult] = Field(default_factory=list)
    trace: list[AgentStep] = Field(default_factory=list)
    iterations: int = 0
    search_calls: int = 0
    nli_pairs: int = 0
    total_cost: float = 0.0
    latency_ms: float = 0.0
    stop_reason: str
