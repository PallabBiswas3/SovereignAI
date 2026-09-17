from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class DiagnosticInvocation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    domain: str = Field(min_length=1)
    task: str | None = None
    inputs: dict[str, Any]
    policy_ref: str | None = None
    model_refs: dict[str, str] = Field(default_factory=dict)
    run_context: dict[str, Any] | None = None


class IntegratedAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=3, max_length=4000)
    candidate_response: str | None = Field(default=None, max_length=30000)
    include_graph_evidence: bool = True
    diagnostic: DiagnosticInvocation | None = None
    policy_profile: str = Field(default="internal_assistant", pattern=r"^[a-z0-9][a-z0-9_-]{1,63}$")
    consequential: bool = True

    @model_validator(mode="after")
    def require_evidence_or_candidate(self):
        if not self.include_graph_evidence and self.diagnostic is None and not self.candidate_response:
            raise ValueError("request must enable an evidence source or provide a candidate response")
        return self


class IntegratedAnalysisResponse(BaseModel):
    run_id: str
    released: bool
    status: str
    final_response: str
    precheck: dict[str, Any]
    graph_evidence: dict[str, Any] | None = None
    diagnostic: dict[str, Any] | None = None
    controlplane: dict[str, Any] | None = None
    service_status: dict[str, str]
