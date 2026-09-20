from adaptivefact.verification.nli import NLIScorer, NLIScores
from controlplane import ControlPlane, Interaction
from controlplane.detectors.hallucination import HallucinationDetector
from controlplane.factuality import build_response_record
from controlplane.schema import GroundingEvidence, VerificationDepth
from controlplane.verification import AdaptiveFactVerificationService


class NeutralNLI(NLIScorer):
    def score(self, premises: list[str], hypotheses: list[str]) -> list[NLIScores]:
        return [NLIScores(entailment=0.05, contradiction=0.05, neutral=0.90) for _ in premises]


def test_controlplane_reuses_phase5_claim_state():
    checker = ControlPlane(
        audit_enabled=False,
        verification_service=AdaptiveFactVerificationService(nli=NeutralNLI()),
    )
    profile = checker.policy_repository.load("customer_support")
    profile.checks["hallucination"].depth = VerificationDepth.STANDARD

    report = checker.check(
        Interaction(
            profile=profile.id,
            prompt="Where is Paris?",
            response="Paris is the capital of France.",
            context="Paris is the capital of France.",
        ),
        profile,
    )

    adaptive = next(item for item in report.detector_results if item.detector == "adaptivefact")
    hallucination = next(item for item in report.detector_results if item.detector == "hallucination")
    assert adaptive.metadata["phase5_reused"] is True
    assert adaptive.metadata["claims_checked"] == hallucination.metadata["claims"]
    assert adaptive.metadata["phase5"] == hallucination.metadata["phase5"]


def test_direct_verifier_remains_backward_compatible():
    service = AdaptiveFactVerificationService()
    result = service.verify(
        Interaction(
            prompt="What was revenue?",
            response="Revenue was $12 million in 2024.",
            context="Revenue was $12 million in 2024.",
        ),
        VerificationDepth.QUICK,
    )
    assert result.metadata["phase5_reused"] is False
    assert result.metadata["claims_checked"] >= 1


def test_unsupported_entity_is_diagnostic_only_by_default():
    detector = HallucinationDetector()
    interaction = Interaction(
        prompt="Who approved it?",
        response="Northstar approved it.",
        context="The request was approved yesterday.",
    )
    result = detector.detect(interaction, {"use_trained_risk": False})

    assert result.metadata["unsupported_entity_diagnostics"] >= 1
    assert not any(item.subtype == "unsupported_entity_claim" for item in result.findings)


def test_unsupported_entity_finding_can_be_explicitly_enabled():
    detector = HallucinationDetector()
    interaction = Interaction(
        prompt="Who approved it?",
        response="Northstar approved it.",
        context="The request was approved yesterday.",
    )
    result = detector.detect(
        interaction,
        {"use_trained_risk": False, "emit_unsupported_entity_findings": True},
    )

    assert any(item.subtype == "unsupported_entity_claim" for item in result.findings)


def test_typed_grounding_evidence_is_preserved_in_verifier_context():
    interaction = Interaction(
        prompt="What is the vibration limit?",
        response="The vibration limit is 7.1 mm/s.",
        grounding_evidence=[
            GroundingEvidence(
                evidence_id="ev-1",
                source_name="Pump SOP",
                document_id="doc-7",
                chunk_id="chunk-3",
                page=4,
                revision="R2",
                text="The vibration alarm limit is 7.1 mm/s.",
                retrieval_score=0.91,
                authorization_scope="maintenance",
            )
        ],
    )

    record = build_response_record(interaction)
    assert "evidence_id=ev-1" in (record.context or "")
    assert "document_id=doc-7" in (record.context or "")
    assert "revision=R2" in (record.context or "")
    assert "7.1 mm/s" in (record.context or "")


def test_unknown_reason_taxonomy_is_exposed_as_undecidable_metadata():
    service = AdaptiveFactVerificationService(nli=NeutralNLI())
    result = service.verify(
        Interaction(
            prompt="Who approved it?",
            response="Northstar approved it.",
            context="The request exists, but the approver is not recorded.",
        ),
        VerificationDepth.STANDARD,
    )

    undecidable = next(item for item in result.findings if item.subtype == "claim_undecidable")
    assert undecidable.status.value == "undecidable"
    # Preserve the v2 reason taxonomy as compatibility metadata while exposing
    # the stronger v3 state at the policy boundary.
    assert undecidable.metadata["unknown_reason"] in {
        "no_evidence",
        "verifier_uncertain",
        "insufficient_support",
        "deterministic_conflict_candidate",
    }
