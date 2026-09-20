from __future__ import annotations

from controlplane.policy.models import PolicyProfile, PolicyRule
from controlplane.schema import ACTION_RANK, SEVERITY_RANK, EnforcementAction, Finding, Interaction, PolicyDecision, Severity, VerificationDepth


_SEVERITY_SCORE = {Severity.LOW: 0.25, Severity.MEDIUM: 0.50, Severity.HIGH: 0.75, Severity.CRITICAL: 1.0}


class PolicyEngine:
    """Apply explicit, explainable policy rules to detector findings."""

    def decide(self, interaction: Interaction, findings: list[Finding], profile: PolicyProfile, *, latency_budget_exceeded: bool = False, verification_depth: VerificationDepth | None = None) -> PolicyDecision:
        action = profile.default_action
        matched: list[str] = []
        for finding in findings:
            for rule in profile.rules:
                if self._matches(rule, finding, interaction):
                    matched.append(rule.id)
                    if ACTION_RANK[rule.action] > ACTION_RANK[action]:
                        action = rule.action
        if latency_budget_exceeded and interaction.consequential:
            action = max((action, EnforcementAction.REVIEW), key=lambda item: ACTION_RANK[item])
            matched.append("system:consequential_latency_budget_exceeded")
        risk_score = max((_SEVERITY_SCORE[item.severity] * item.confidence for item in findings), default=0.0)
        depth = verification_depth or self._verification_depth(profile)
        reason = f"Applied {len(set(matched))} policy rule(s); strongest action is {action.value}." if matched else f"No policy rule matched; applied default action {action.value}."
        return PolicyDecision(action=action, verification_depth=depth, risk_score=min(1.0, float(risk_score)), reason=reason, matched_rules=list(dict.fromkeys(matched)), human_review_required=action == EnforcementAction.REVIEW, latency_budget_exceeded=latency_budget_exceeded)

    @staticmethod
    def _matches(rule: PolicyRule, finding: Finding, interaction: Interaction) -> bool:
        condition = rule.when
        if condition.category is not None and finding.category != condition.category:
            return False
        if condition.subtype is not None and finding.subtype != condition.subtype:
            return False
        if condition.location_prefix is not None:
            location = str(finding.metadata.get("location", ""))
            if not location.startswith(condition.location_prefix):
                return False
        if condition.status is not None and finding.status != condition.status:
            return False
        if condition.severity_at_least is not None and SEVERITY_RANK[finding.severity] < SEVERITY_RANK[condition.severity_at_least]:
            return False
        if condition.consequential_only and not interaction.consequential:
            return False
        return True

    @staticmethod
    def _verification_depth(profile: PolicyProfile) -> VerificationDepth:
        enabled = [item.depth for item in profile.checks.values() if item.enabled]
        if VerificationDepth.DEEP in enabled:
            return VerificationDepth.DEEP
        if VerificationDepth.STANDARD in enabled:
            return VerificationDepth.STANDARD
        return VerificationDepth.QUICK
