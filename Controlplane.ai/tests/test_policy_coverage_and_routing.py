"""Policy coverage and fail-closed routing (10 November 2026).

Before this change a HIGH finding that no rule anticipated took the profile's
default action (allow): customer_support released "Initiate a controlled
shutdown now" and a fabricated work record. Separately, with the trained risk
router missing, the deterministic fallback scored unresolved claims ~0.04 and
routed non-consequential answers to QUICK, so NLI never ran and a plainly
contradicted fact was released under every profile.
"""
import re
from pathlib import Path

import pytest

from adaptivefact.agents.schema import AgentVerificationConfig
from adaptivefact.agents.tools.context import ContextSearchTool
from adaptivefact.agents.verifier import BoundedVerificationAgent
from adaptivefact.verification.nli import NLIScorer, NLIScores
from controlplane.checker import ControlPlane
from controlplane.policy import PolicyEngine, PolicyRepository
from controlplane.schema import (
    EnforcementAction,
    Finding,
    FindingStatus,
    GroundingEvidence,
    Interaction,
    RiskCategory,
    Severity,
    VerificationDepth,
)
from controlplane.verification import AdaptiveFactVerificationService

ROOT = Path(__file__).resolve().parents[1]
POLICIES = PolicyRepository(ROOT / "configs/policies")
PROFILES = POLICIES.available()
RELEASED = {EnforcementAction.ALLOW, EnforcementAction.WARN, EnforcementAction.REDACT}

# Every finding subtype literal emitted by the gateway, with its category/status.
EMITTED = {
    "claim_conflicting": (RiskCategory.HALLUCINATION, FindingStatus.CONFLICTING),
    "claim_contradicted": (RiskCategory.HALLUCINATION, FindingStatus.CONTRADICTED),
    "claim_undecidable": (RiskCategory.HALLUCINATION, FindingStatus.UNDECIDABLE),
    "claim_unsupported": (RiskCategory.HALLUCINATION, FindingStatus.UNSUPPORTED),
    "compounding_conversation_risk": (RiskCategory.POLICY, FindingStatus.SUSPECTED),
    "counterfactual_inconsistency": (RiskCategory.BIAS, FindingStatus.SUSPECTED),
    "detector_failure": (RiskCategory.POLICY, FindingStatus.UNKNOWN),
    "elevated_response_risk": (RiskCategory.HALLUCINATION, FindingStatus.SUSPECTED),
    "explicit_stereotype": (RiskCategory.BIAS, FindingStatus.DETECTED),
    "grounded_value_conflict": (RiskCategory.HALLUCINATION, FindingStatus.SUSPECTED),
    "payment_card": (RiskCategory.PRIVACY, FindingStatus.DETECTED),
    "physical_action_requires_review": (RiskCategory.POLICY, FindingStatus.UNKNOWN),
    "protected_attribute_adverse_action": (RiskCategory.BIAS, FindingStatus.SUSPECTED),
    "unsupported_entity_claim": (RiskCategory.HALLUCINATION, FindingStatus.UNDECIDABLE),
    "untrusted_instruction_promotion": (RiskCategory.POLICY, FindingStatus.CONTRADICTED),
    "unverified_operational_event": (RiskCategory.POLICY, FindingStatus.UNKNOWN),
}


def test_every_emitted_subtype_is_listed_here():
    """A new finding subtype must be added to EMITTED (and handled by every policy)."""
    source = "\n".join(path.read_text(encoding="utf-8") for path in (ROOT / "src/controlplane").rglob("*.py"))
    assert set(re.findall(r'subtype="([a-z_]+)"', source)) <= set(EMITTED)


@pytest.mark.parametrize("profile_name", PROFILES)
@pytest.mark.parametrize("subtype", sorted(EMITTED))
def test_every_policy_explicitly_handles_every_high_finding(profile_name, subtype):
    profile = POLICIES.load(profile_name)
    category, status = EMITTED[subtype]
    finding = Finding(category=category, subtype=subtype, severity=Severity.HIGH, confidence=1.0, status=status,
                      message="synthetic", detector="test", metadata={"location": "response"})
    decision = PolicyEngine().decide(Interaction(prompt="q", response="r", consequential=True), [finding], profile)
    explicit = [rule for rule in decision.matched_rules if not rule.startswith("system:")]
    assert explicit or subtype in profile.informational_subtypes, (
        f"{profile_name} has no rule for HIGH {subtype}; add a rule or list it in informational_subtypes"
    )


def test_unmatched_high_finding_fails_closed_to_review():
    profile = POLICIES.load("internal_assistant").model_copy(update={"rules": [], "informational_subtypes": []})
    finding = Finding(category=RiskCategory.POLICY, subtype="brand_new_control", severity=Severity.HIGH,
                      confidence=1.0, status=FindingStatus.UNKNOWN, message="synthetic", detector="test")
    decision = PolicyEngine().decide(Interaction(prompt="q", response="r"), [finding], profile)
    assert decision.action == EnforcementAction.REVIEW
    assert "system:unmatched_high_severity_finding:brand_new_control" in decision.matched_rules


def test_unmatched_low_or_informational_finding_keeps_default():
    profile = POLICIES.load("internal_assistant").model_copy(update={"rules": []})
    low = Finding(category=RiskCategory.POLICY, subtype="minor", severity=Severity.MEDIUM, confidence=1.0,
                  status=FindingStatus.UNKNOWN, message="synthetic", detector="test")
    hint = low.model_copy(update={"subtype": "grounded_value_conflict", "severity": Severity.HIGH})
    decision = PolicyEngine().decide(Interaction(prompt="q", response="r"), [low, hint], profile)
    assert decision.action == profile.default_action


@pytest.mark.parametrize("profile_name", PROFILES)
@pytest.mark.parametrize("response", [
    "Initiate a controlled shutdown of Pump-102 now and isolate it.",
    "Bearing replacement was completed on Pump-102 on 2026-10-04.",
])
def test_physical_action_and_unrecorded_work_are_held_by_every_policy(profile_name, response):
    report = ControlPlane(audit_enabled=False).check(
        Interaction(profile=profile_name, prompt="What should we do with Pump-102?", response=response))
    assert report.decision.action not in RELEASED


# --- Routing when the trained risk router is unavailable ---------------------

class KeywordNLI(NLIScorer):
    def score(self, premises, hypotheses):
        return [NLIScores(0.01, 0.97, 0.02) if "Flowserve" in hypothesis else NLIScores(0.97, 0.01, 0.02)
                for hypothesis in hypotheses]


EVIDENCE = [GroundingEvidence(evidence_id="DS-220", document_id="DS-220", chunk_id="DS-220-1", retrieval_score=0.9,
                              text="Pump-102 was manufactured by KSB.",
                              metadata={"document_status": "current", "document_type": "equipment_manual"})]


def _check(profile_name, response, evidence=EVIDENCE):
    profile = POLICIES.load(profile_name)
    profile.checks["hallucination"].settings["risk_model"] = "missing/model.joblib"  # router unavailable
    nli = KeywordNLI()
    # Same composition as build_adaptive_verification_from_env (deep depth needs the agent).
    agent = BoundedVerificationAgent(nli, [ContextSearchTool()], AgentVerificationConfig())
    gate = ControlPlane(audit_enabled=False,
                        verification_service=AdaptiveFactVerificationService(nli=nli, agent=agent))
    return gate.check(Interaction(prompt="Who made Pump-102?", response=response, grounding_evidence=evidence,
                                  consequential=False), profile)


@pytest.mark.parametrize("profile_name", PROFILES)
def test_missing_risk_router_does_not_skip_nli_for_contradicted_fact(profile_name):
    report = _check(profile_name, "Pump-102 was manufactured by Flowserve [DS-220].")
    assert report.decision.verification_depth != VerificationDepth.QUICK
    assert report.decision.action not in RELEASED


@pytest.mark.parametrize("profile_name", PROFILES)
def test_missing_risk_router_still_releases_supported_fact(profile_name):
    assert _check(profile_name, "Pump-102 was manufactured by KSB [DS-220].").decision.action in RELEASED


def test_answer_without_evidence_still_uses_quick_depth():
    report = _check("internal_assistant", "Pump-102 was manufactured by Flowserve.", evidence=[])
    assert report.decision.verification_depth == VerificationDepth.QUICK
