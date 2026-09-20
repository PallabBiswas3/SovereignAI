from __future__ import annotations

from adaptivefact.data.schema import GenerationMetadata, ResponseRecord, VerificationStatus
from adaptivefact.extraction.claim_extractor import ClaimExtractor
from adaptivefact.extraction.numeric_date import extract_numbers
from adaptivefact.verification.nli import NLIScorer, NLIScores
from controlplane import ControlPlane, Interaction
from controlplane.detectors.hallucination import HallucinationDetector
from controlplane.policy import PolicyRepository
from controlplane.schema import DetectorResult, EnforcementAction, GroundingEvidence, VerificationDepth
from controlplane.support_scorers import SupportScorer
from controlplane.verification import AdaptiveFactVerificationService


class NeutralNLI(NLIScorer):
    def score(self, premises: list[str], hypotheses: list[str]) -> list[NLIScores]:
        return [NLIScores(entailment=0.05, contradiction=0.05, neutral=0.90) for _ in premises]


class ConflictingNLI(NLIScorer):
    def score(self, premises: list[str], hypotheses: list[str]) -> list[NLIScores]:
        output = []
        for premise in premises:
            # Check explicit negation before the shared value substring.
            if "did not reach 95 C" in premise:
                output.append(NLIScores(entailment=0.02, contradiction=0.96, neutral=0.02))
            elif "95 C" in premise:
                output.append(NLIScores(entailment=0.95, contradiction=0.02, neutral=0.03))
            else:
                output.append(NLIScores(entailment=0.02, contradiction=0.96, neutral=0.02))
        return output


class FixedSupportScorer(SupportScorer):
    name = "fixed"

    def __init__(self, value: float) -> None:
        self.value = value
        self.calls = 0
        self.batch_sizes: list[int] = []

    def score(self, documents: list[str], claims: list[str]) -> list[float]:
        self.calls += 1
        self.batch_sizes.append(len(claims))
        return [self.value] * len(claims)


def _evidence(text: str, *, evidence_id: str, chunk_id: str, score: float = 0.9, complete: bool = False) -> GroundingEvidence:
    return GroundingEvidence(
        evidence_id=evidence_id,
        text=text,
        source_name="maintenance-manual",
        document_id="manual-1",
        chunk_id=chunk_id,
        retrieval_score=score,
        authorization_scope="plant-a",
        metadata={"evidence_set_complete": complete},
    )


def _profile(name: str, depth: VerificationDepth) -> object:
    profile = PolicyRepository().load(name)
    profile.checks["hallucination"].depth = depth
    profile.checks["hallucination"].settings["use_trained_risk"] = False
    return profile


def test_asset_identifier_is_not_treated_as_numeric_fact():
    assert extract_numbers("Pump-102 is blue.") == []
    values = extract_numbers("Pump-102 reached 95 C.")
    assert [item.normalized for item in values] == ["95"]


def test_atomic_extractor_decontextualizes_followup_pronoun():
    record = ResponseRecord(
        id="r1",
        dataset="test",
        query="status?",
        context=None,
        generated_response="Pump-102 reached 95 C. It exceeded its safe limit.",
        generation_metadata=GenerationMetadata(),
    )
    claims = ClaimExtractor().extract(record)
    assert len(claims) >= 2
    assert claims[0].subject == "Pump-102"
    assert claims[0].predicate == "reached"
    assert claims[0].object == "95 C"
    assert claims[1].verification_text.startswith("Pump-102 exceeded")
    assert claims[1].parent_span is not None


def test_high_support_resolves_neutral_nli_and_batches_claims():
    scorer = FixedSupportScorer(0.95)
    service = AdaptiveFactVerificationService(nli=NeutralNLI(), support_scorer=scorer)
    result = service.verify(
        Interaction(
            response="Pump-102 is blue. Pump-103 is green.",
            prompt="What colors are the pumps?",
            grounding_evidence=[
                _evidence("Pump-102 is blue. Pump-103 is green.", evidence_id="e1", chunk_id="c1"),
                _evidence("Equipment color register confirms Pump-102 blue and Pump-103 green.", evidence_id="e2", chunk_id="c2"),
            ],
        ),
        VerificationDepth.STANDARD,
    )
    assert scorer.calls == 1
    assert scorer.batch_sizes[0] >= 1
    assert result.metadata["support"]["calls"] == 1
    assert all(item["status"] == VerificationStatus.SUPPORTED.value for item in result.metadata["claim_results"])


def test_low_support_plus_complete_structured_evidence_is_unsupported():
    scorer = FixedSupportScorer(0.05)
    service = AdaptiveFactVerificationService(nli=NeutralNLI(), support_scorer=scorer)
    result = service.verify(
        Interaction(
            prompt="What is the pump state?",
            response="Pump-102 is blue.",
            grounding_evidence=[
                _evidence("Pump-102 is red.", evidence_id="e1", chunk_id="c1", complete=True),
                _evidence("Pump-102 color register lists red.", evidence_id="e2", chunk_id="c2", complete=True),
            ],
        ),
        VerificationDepth.STANDARD,
    )
    claim = result.metadata["claim_results"][0]
    assert claim["status"] == VerificationStatus.UNSUPPORTED.value
    assert any(item.subtype == "claim_unsupported" for item in result.findings)
    assert set(claim["evidence_ids"]) == {"e1", "e2"}


def test_low_support_with_partial_evidence_is_undecidable_not_unsupported():
    service = AdaptiveFactVerificationService(nli=NeutralNLI(), support_scorer=FixedSupportScorer(0.05))
    result = service.verify(
        Interaction(
            prompt="What is the pump state?",
            response="Pump-102 is blue.",
            grounding_evidence=[_evidence("Pump-102 is red.", evidence_id="e1", chunk_id="c1", score=0.2)],
        ),
        VerificationDepth.STANDARD,
    )
    assert result.metadata["claim_results"][0]["status"] == VerificationStatus.UNDECIDABLE.value
    assert any(item.subtype == "claim_undecidable" for item in result.findings)
    assert not any(item.subtype == "claim_unsupported" for item in result.findings)


def test_conflicting_nli_becomes_explicit_conflicting_state():
    service = AdaptiveFactVerificationService(nli=ConflictingNLI())
    result = service.verify(
        Interaction(
            prompt="What temperature did Pump-102 reach?",
            response="Pump-102 reached 95 C.",
            grounding_evidence=[
                _evidence("Pump-102 reached 95 C.", evidence_id="e1", chunk_id="c1"),
                _evidence("Pump-102 did not reach 95 C; maximum was 80 C.", evidence_id="e2", chunk_id="c2"),
            ],
        ),
        VerificationDepth.STANDARD,
    )
    assert result.metadata["claim_results"][0]["status"] == VerificationStatus.CONFLICTING.value
    assert any(item.subtype == "claim_conflicting" for item in result.findings)


def test_risk_prior_does_not_emit_policy_finding_by_default():
    detector = HallucinationDetector()
    result = detector.detect(
        Interaction(prompt="Status?", response="Pump-102 is blue.", context="Pump-102 is blue."),
        {"use_trained_risk": False},
    )
    assert result.metadata["risk_role"] == "verification_router_only"
    assert not any(item.subtype == "elevated_response_risk" for item in result.findings)


def test_risk_routing_is_adaptive_and_capped_by_profile():
    customer = _profile("customer_support", VerificationDepth.STANDARD)
    low = DetectorResult(detector="hallucination", metadata={"risk_score": 0.1, "conflict_candidates": 0})
    high = DetectorResult(detector="hallucination", metadata={"risk_score": 0.95, "conflict_candidates": 0})
    assert ControlPlane._hallucination_depth(customer, Interaction(prompt="q", response="r"), low) == VerificationDepth.QUICK
    assert ControlPlane._hallucination_depth(customer, Interaction(prompt="q", response="r"), high) == VerificationDepth.STANDARD

    regulated = _profile("regulated_decision_support", VerificationDepth.DEEP)
    assert ControlPlane._hallucination_depth(regulated, Interaction(prompt="q", response="r"), high) == VerificationDepth.DEEP
    assert ControlPlane._hallucination_depth(regulated, Interaction(prompt="q", response="r", consequential=True), low) == VerificationDepth.DEEP


def test_only_consequential_ambiguity_requires_regulated_human_review():
    service = AdaptiveFactVerificationService(nli=NeutralNLI())
    checker = ControlPlane(audit_enabled=False, verification_service=service)
    profile = _profile("regulated_decision_support", VerificationDepth.STANDARD)

    ordinary = checker.check(
        Interaction(profile=profile.id, prompt="Who approved it?", response="Northstar approved it."),
        profile,
    )
    assert any(item.subtype == "claim_undecidable" for item in ordinary.findings)
    assert ordinary.decision.action != EnforcementAction.REVIEW

    consequential = checker.check(
        Interaction(profile=profile.id, prompt="Who approved it?", response="Northstar approved it.", consequential=True),
        profile,
    )
    assert any(item.subtype == "claim_undecidable" for item in consequential.findings)
    assert consequential.decision.action == EnforcementAction.REVIEW


def test_missing_required_verifier_fails_closed_for_consequential_claims():
    checker = ControlPlane(audit_enabled=False, verification_service=None)
    profile = _profile("regulated_decision_support", VerificationDepth.STANDARD)
    report = checker.check(
        Interaction(profile=profile.id, prompt="Approve?", response="Northstar approved the case.", consequential=True),
        profile,
    )
    assert any(item.subtype == "detector_failure" and item.metadata.get("failed_detector") == "adaptivefact" for item in report.findings)
    assert report.decision.action == EnforcementAction.REVIEW
