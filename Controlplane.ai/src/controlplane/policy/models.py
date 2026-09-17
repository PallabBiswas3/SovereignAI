from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator

from controlplane.schema import (
    EnforcementAction,
    FindingStatus,
    RiskCategory,
    Severity,
    VerificationDepth,
)


class DetectorPolicy(BaseModel):
    enabled: bool = True
    depth: VerificationDepth = VerificationDepth.QUICK
    settings: dict[str, Any] = Field(default_factory=dict)


class RuleCondition(BaseModel):
    category: RiskCategory | None = None
    subtype: str | None = None
    location_prefix: str | None = None
    severity_at_least: Severity | None = None
    status: FindingStatus | None = None
    consequential_only: bool = False


class PolicyRule(BaseModel):
    id: str
    description: str = ""
    when: RuleCondition
    action: EnforcementAction


class PolicyProfile(BaseModel):
    id: str
    version: str
    name: str
    description: str = ""
    geography: str = "global"
    industry: str = "general"
    latency_budget_ms: int = Field(default=1000, gt=0)
    default_action: EnforcementAction = EnforcementAction.ALLOW
    checks: dict[str, DetectorPolicy]
    rules: list[PolicyRule]
    warning_template: str = "This response may require verification before it is relied upon."
    blocked_response: str = "This response was blocked by the organization's AI safety policy."
    review_response: str = "This response has been held for human review."

    @model_validator(mode="after")
    def require_unique_rule_ids(self) -> "PolicyProfile":
        ids = [rule.id for rule in self.rules]
        if len(ids) != len(set(ids)):
            raise ValueError("Policy rule ids must be unique")
        return self
