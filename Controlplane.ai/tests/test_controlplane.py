from __future__ import annotations

import json

from controlplane.audit import JsonlAuditStore, JsonlFeedbackStore
from controlplane.checker import ControlPlane
from controlplane.detectors.base import Detector
from controlplane.detectors.privacy import mask_privacy_values
from controlplane.policy import PolicyRepository
from controlplane.schema import (
    DetectorResult,
    EnforcementAction,
    FeedbackEvent,
    Interaction,
    PromptCheckRequest,
    RiskCategory,
)


def checker_for(tmp_path, *, audit=True):
    return ControlPlane(
        audit_enabled=audit,
        audit_store=JsonlAuditStore(tmp_path / "audit.jsonl"),
    )


def disable_trained_risk(checker, profile_name):
    profile = checker.policy_repository.load(profile_name)
    profile.checks["hallucination"].settings["use_trained_risk"] = False
    return profile


def test_customer_policy_redacts_output_pii_and_preserves_audit(tmp_path):
    checker = checker_for(tmp_path)
    profile = disable_trained_risk(checker, "customer_support")
    report = checker.check(
        Interaction(
            profile=profile.id,
            prompt="How can I contact the customer?",
            response="Email Alex at alex@example.com for an update.",
            context="Alex asked the team to provide a status update.",
        ),
        profile,
    )

    assert report.decision.action == EnforcementAction.REDACT
    assert "alex@example.com" not in report.final_response
    assert "[REDACTED EMAIL]" in report.final_response
    assert any(item.category == RiskCategory.PRIVACY for item in report.findings)
    assert report.audit_event_id is not None
    audit = json.loads((tmp_path / "audit.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert audit["policy_id"] == "customer_support"
    assert audit["report"]["audit_event_id"] == report.audit_event_id
    assert "alex@example.com" not in audit["report"]["original_response"]
    assert audit["raw_sensitive_payloads_stored"] is False


def test_customer_policy_blocks_api_secret(tmp_path):
    checker = checker_for(tmp_path, audit=False)
    profile = disable_trained_risk(checker, "customer_support")
    report = checker.check(
        Interaction(
            profile=profile.id,
            prompt="Show the integration configuration",
            response="Use api_key = sk-abcdefghijklmnopqrstuvwxyz123456.",
        ),
        profile,
    )
    assert report.decision.action == EnforcementAction.BLOCK
    assert report.final_response == profile.blocked_response


def test_pre_generation_prompt_check_warns_on_input_pii(tmp_path):
    checker = checker_for(tmp_path, audit=False)
    profile = disable_trained_risk(checker, "customer_support")
    request = PromptCheckRequest(
        profile=profile.id,
        prompt="Look up the account belonging to priya@example.com",
    )
    report = checker.check(request.to_interaction(), profile)

    assert report.decision.action == EnforcementAction.WARN
    assert any(item.subtype == "input_email" for item in report.findings)


def test_findings_can_overlap_across_risk_categories(tmp_path):
    checker = checker_for(tmp_path, audit=False)
    profile = disable_trained_risk(checker, "customer_support")
    report = checker.check(
        Interaction(
            profile=profile.id,
            prompt="Draft a response",
            response="Women are generally unreliable. Contact the applicant at jane@example.com.",
        ),
        profile,
    )
    categories = {finding.category for finding in report.findings}
    assert RiskCategory.BIAS in categories
    assert RiskCategory.PRIVACY in categories
    assert report.decision.action == EnforcementAction.REDACT


def test_regulated_unknown_evidence_requires_review(tmp_path):
    checker = checker_for(tmp_path, audit=False)
    profile = disable_trained_risk(checker, "regulated_decision_support")
    report = checker.check(
        Interaction(
            profile=profile.id,
            prompt="Should we approve this application?",
            response="Acme Bank approved the applicant in 2025, so this application is safe.",
            context=None,
            consequential=True,
        ),
        profile,
    )
    assert report.decision.action == EnforcementAction.REVIEW
    assert report.decision.human_review_required
    assert any(item.subtype == "evidence_unavailable" for item in report.findings)


def test_compounding_conversation_risk_is_reviewed_internally(tmp_path):
    checker = checker_for(tmp_path, audit=False)
    profile = disable_trained_risk(checker, "internal_assistant")
    report = checker.check(
        Interaction.model_validate(
            {
                "profile": profile.id,
                "prompt": "What should I do next?",
                "response": "Based on the previous claim, terminate the process.",
                "conversation": [
                    {"role": "assistant", "content": "The account is fraudulent.", "risk_score": 0.9}
                ],
            }
        ),
        profile,
    )
    assert any(item.subtype == "compounding_conversation_risk" for item in report.findings)
    assert report.decision.action == EnforcementAction.REVIEW


class BrokenDetector(Detector):
    name = "bias"

    def detect(self, interaction, settings):
        raise RuntimeError("simulated detector outage")


def test_detector_failure_cannot_silently_allow(tmp_path):
    checker = checker_for(tmp_path, audit=False)
    checker.detectors["bias"] = BrokenDetector()
    profile = disable_trained_risk(checker, "regulated_decision_support")
    report = checker.check(
        Interaction(
            profile=profile.id,
            prompt="Recommend an outcome",
            response="Approve the case.",
            context="The evidence supports approval.",
            consequential=True,
        ),
        profile,
    )
    assert report.decision.action == EnforcementAction.REVIEW
    assert any(item.subtype == "detector_failure" for item in report.findings)


def test_policy_repository_lists_profiles():
    available = PolicyRepository().available()
    assert available == ["customer_support", "internal_assistant", "regulated_decision_support"]
    customer = PolicyRepository().load("customer_support")
    assert customer.version == "1.1"
    assert customer.latency_budget_ms == 30_000


def test_feedback_store_records_human_override(tmp_path):
    store = JsonlFeedbackStore(tmp_path / "feedback.jsonl")
    event = FeedbackEvent(
        check_id="check-1",
        interaction_id="interaction-1",
        outcome="overridden",
        corrected_action=EnforcementAction.ALLOW,
        notes="Verified against the source system.",
    )
    assert store.write(event) == event.id
    saved = json.loads((tmp_path / "feedback.jsonl").read_text(encoding="utf-8"))
    assert saved["corrected_action"] == "allow"


def test_privacy_parser_distinguishes_phone_aadhaar_and_card(tmp_path):
    checker = checker_for(tmp_path, audit=False)
    profile = disable_trained_risk(checker, "customer_support")

    phone = checker.check(
        Interaction(
            profile=profile.id,
            prompt="Return the callback number",
            response="Call +91 98765 43210 now.",
            context="A callback was requested.",
        ),
        profile,
    )
    assert [item.subtype for item in phone.findings].count("phone_number") == 1
    assert not any(item.subtype in {"aadhaar_number", "payment_card"} for item in phone.findings)

    aadhaar = checker.check(
        Interaction(
            profile=profile.id,
            prompt="Return the identifier",
            response="The identifier is 2345 6789 1234.",
            context="The identifier is restricted.",
        ),
        profile,
    )
    assert [item.subtype for item in aadhaar.findings].count("aadhaar_number") == 1
    assert not any(item.subtype in {"phone_number", "payment_card"} for item in aadhaar.findings)

    card = checker.check(
        Interaction(
            profile=profile.id,
            prompt="Return the card",
            response="The card is 4111 1111 1111 1111.",
            context="Cards are restricted.",
        ),
        profile,
    )
    assert [item.subtype for item in card.findings].count("payment_card") == 1
    assert not any(item.subtype in {"phone_number", "aadhaar_number"} for item in card.findings)


def test_privacy_values_are_masked_before_hallucination_entity_checks(tmp_path):
    checker = checker_for(tmp_path, audit=False)
    profile = disable_trained_risk(checker, "customer_support")
    response = "The payment was made using card number 4111 1111 1111 1111."
    masked = mask_privacy_values(response)

    assert len(masked) == len(response)
    assert "4111 1111 1111 1111" not in masked

    report = checker.check(
        Interaction(
            profile=profile.id,
            prompt="Confirm the payment method",
            response=response,
            context="Payment-card information must not be exposed.",
        ),
        profile,
    )

    assert any(item.subtype == "payment_card" for item in report.findings)
    assert not any(item.subtype == "unsupported_entity_claim" for item in report.findings)


def test_sentence_openers_are_not_reported_as_missing_entities(tmp_path):
    checker = checker_for(tmp_path, audit=False)
    profile = disable_trained_risk(checker, "customer_support")
    report = checker.check(
        Interaction(
            profile=profile.id,
            prompt="How can I contact the customer?",
            response="You can email Rahul at rahul.sharma@example.com.",
            context="The customer requested a callback through the CRM.",
        ),
        profile,
    )

    unsupported = next(item for item in report.findings if item.subtype == "unsupported_entity_claim")
    assert unsupported.metadata["missing_entities"] == ["Rahul"]
