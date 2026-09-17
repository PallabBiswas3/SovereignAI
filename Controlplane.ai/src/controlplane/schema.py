from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


class RiskCategory(str, Enum):
    PRIVACY = "privacy"
    BIAS = "bias"
    HALLUCINATION = "hallucination"
    POLICY = "policy"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


SEVERITY_RANK = {
    Severity.LOW: 1,
    Severity.MEDIUM: 2,
    Severity.HIGH: 3,
    Severity.CRITICAL: 4,
}


class FindingStatus(str, Enum):
    DETECTED = "detected"
    SUSPECTED = "suspected"
    CONTRADICTED = "contradicted"
    UNKNOWN = "unknown"


class EnforcementAction(str, Enum):
    ALLOW = "allow"
    WARN = "allow_with_warning"
    REDACT = "redact"
    REVIEW = "human_review"
    BLOCK = "block"


ACTION_RANK = {
    EnforcementAction.ALLOW: 0,
    EnforcementAction.WARN: 1,
    EnforcementAction.REDACT: 2,
    EnforcementAction.REVIEW: 3,
    EnforcementAction.BLOCK: 4,
}


class VerificationDepth(str, Enum):
    QUICK = "quick"
    STANDARD = "standard"
    DEEP = "deep"


class ConversationTurn(BaseModel):
    role: str
    content: str
    risk_score: float | None = Field(default=None, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class Interaction(BaseModel):
    """A provider-independent interaction submitted to ControlPlane.ai."""

    id: str = Field(default_factory=lambda: f"interaction-{uuid4()}")
    profile: str = "customer_support"
    prompt: str
    response: str
    context: str | None = None
    conversation: list[ConversationTurn] = Field(default_factory=list)
    geography: str | None = None
    industry: str | None = None
    consequential: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class PromptCheckRequest(BaseModel):
    """Input-only request for a pre-generation policy gate."""

    id: str = Field(default_factory=lambda: f"interaction-{uuid4()}")
    profile: str = "customer_support"
    prompt: str
    conversation: list[ConversationTurn] = Field(default_factory=list)
    geography: str | None = None
    industry: str | None = None
    consequential: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_interaction(self) -> Interaction:
        return Interaction(
            id=self.id,
            profile=self.profile,
            prompt=self.prompt,
            response="",
            conversation=self.conversation,
            geography=self.geography,
            industry=self.industry,
            consequential=self.consequential,
            metadata={**self.metadata, "controlplane_stage": "pre_generation"},
        )


class Finding(BaseModel):
    id: str = Field(default_factory=lambda: f"finding-{uuid4()}")
    category: RiskCategory
    subtype: str
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    status: FindingStatus
    message: str
    detector: str
    detector_version: str = "1.0"
    span: tuple[int, int] | None = None
    evidence: list[str] = Field(default_factory=list)
    replacement: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class DetectorResult(BaseModel):
    detector: str
    findings: list[Finding] = Field(default_factory=list)
    latency_ms: float = 0.0
    skipped: bool = False
    skip_reason: str | None = None
    error: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    action: EnforcementAction
    verification_depth: VerificationDepth
    risk_score: float = Field(ge=0.0, le=1.0)
    reason: str
    matched_rules: list[str] = Field(default_factory=list)
    human_review_required: bool = False
    latency_budget_exceeded: bool = False


class CheckReport(BaseModel):
    id: str = Field(default_factory=lambda: f"check-{uuid4()}")
    interaction_id: str
    policy_id: str
    policy_version: str
    original_response: str
    final_response: str
    decision: PolicyDecision
    findings: list[Finding] = Field(default_factory=list)
    detector_results: list[DetectorResult] = Field(default_factory=list)
    total_latency_ms: float
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    audit_event_id: str | None = None


class AuditEvent(BaseModel):
    id: str
    event_type: str = "controlplane.check.completed"
    policy_id: str
    policy_version: str
    report: CheckReport
    policy_snapshot: dict[str, Any]
    raw_sensitive_payloads_stored: bool = False
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class FeedbackEvent(BaseModel):
    id: str = Field(default_factory=lambda: f"feedback-{uuid4()}")
    check_id: str
    interaction_id: str
    outcome: str
    corrected_action: EnforcementAction | None = None
    notes: str | None = None
    actor: str = "human_reviewer"
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
