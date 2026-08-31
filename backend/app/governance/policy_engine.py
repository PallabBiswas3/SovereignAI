from __future__ import annotations

from enum import Enum
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class GovernanceDecision(str, Enum):
    allow = "ALLOW"
    allow_with_warning = "ALLOW_WITH_WARNING"
    require_human_approval = "REQUIRE_HUMAN_APPROVAL"
    rewrite = "REWRITE"
    block = "BLOCK"


class Policy(BaseModel):
    name: str
    pii_detection: str = "standard"
    require_sources: bool = False
    grounding_threshold: float = 0.55
    human_review_for: list[str] = Field(default_factory=list)


class PolicyEngine:
    def __init__(self, config_path: Path) -> None:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        self.policies = {name: Policy(name=name, **values) for name, values in raw.get("policies", {}).items()}

    def get(self, use_case: str) -> Policy:
        return self.policies.get(use_case) or self.policies["internal_assistant"]

    def decide(self, policy: Policy, *, pii_count: int = 0, injection_count: int = 0, grounding_score: float | None = None) -> GovernanceDecision:
        if injection_count:
            return GovernanceDecision.block
        if grounding_score is not None and policy.require_sources and grounding_score < policy.grounding_threshold:
            return GovernanceDecision.require_human_approval if "low_grounding" in policy.human_review_for else GovernanceDecision.rewrite
        if pii_count:
            return GovernanceDecision.allow_with_warning
        return GovernanceDecision.allow
