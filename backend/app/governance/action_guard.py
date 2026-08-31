from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel

from app.governance.policy_engine import GovernanceDecision


class ActionDecision(BaseModel):
    tool: str
    risk: str
    decision: GovernanceDecision
    reason: str


class ActionGuard:
    def __init__(self, tools_config: Path) -> None:
        raw = yaml.safe_load(tools_config.read_text(encoding="utf-8")) or {}
        self.tools = raw.get("tools", {})

    def evaluate(self, tool: str, *, approved: bool = False) -> ActionDecision:
        config = self.tools.get(tool)
        if not config:
            return ActionDecision(tool=tool, risk="UNKNOWN", decision=GovernanceDecision.block, reason="Tool is not registered in policy configuration.")
        risk = str(config.get("risk", "HIGH")).upper()
        enabled = bool(config.get("enabled", False))
        if approved:
            if not enabled or not bool(config.get("executable_after_approval", False)):
                return ActionDecision(tool=tool, risk=risk, decision=GovernanceDecision.block, reason="Approval cannot override a disabled or non-executable tool policy.")
            return ActionDecision(tool=tool, risk=risk, decision=GovernanceDecision.allow, reason="Exact registered action was approved and remains enabled after revalidation.")
        if risk == "HIGH" or bool(config.get("approval_required", False)):
            return ActionDecision(tool=tool, risk=risk, decision=GovernanceDecision.require_human_approval, reason=f"Configured {risk.lower()}-risk action requires explicit human authorization.")
        if not enabled:
            return ActionDecision(tool=tool, risk=risk, decision=GovernanceDecision.block, reason="Tool is disabled by configuration.")
        return ActionDecision(tool=tool, risk=risk, decision=GovernanceDecision.allow, reason=f"Configured {risk.lower()}-risk tool.")
