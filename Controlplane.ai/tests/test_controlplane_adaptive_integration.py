from adaptivefact.agents.schema import AgentVerificationConfig
from adaptivefact.agents.tools.context import ContextSearchTool
from adaptivefact.agents.verifier import BoundedVerificationAgent
from adaptivefact.data.schema import VerificationStatus
from adaptivefact.verification.nli import NLIScorer, NLIScores
from adaptivefact.verification.phase6 import Phase6NLIConfig
from controlplane import ControlPlane, Interaction
from controlplane.schema import EnforcementAction, FindingStatus, VerificationDepth
from controlplane.verification import AdaptiveFactVerificationService, build_adaptive_verification_from_env


def test_adaptive_verification_is_enabled_by_default(monkeypatch):
    monkeypatch.delenv("CONTROLPLANE_ADAPTIVE_VERIFICATION", raising=False)

    assert isinstance(build_adaptive_verification_from_env(), AdaptiveFactVerificationService)


def test_adaptive_verification_can_be_explicitly_disabled(monkeypatch):
    monkeypatch.setenv("CONTROLPLANE_ADAPTIVE_VERIFICATION", "0")

    assert build_adaptive_verification_from_env() is None


class ExactEvidenceNLI(NLIScorer):
    def score(self, premises: list[str], hypotheses: list[str]) -> list[NLIScores]:
        results = []
        for premise, hypothesis in zip(premises, hypotheses):
            p = premise.casefold()
            h = hypothesis.casefold()
            if "joined tesla" in p and "founded tesla" in h:
                results.append(NLIScores(entailment=0.01, contradiction=0.98, neutral=0.01))
            elif h.rstrip(".") in p:
                results.append(NLIScores(entailment=0.98, contradiction=0.01, neutral=0.01))
            else:
                results.append(NLIScores(entailment=0.05, contradiction=0.05, neutral=0.90))
        return results


def test_quick_depth_runs_deterministic_only():
    checker = ControlPlane(
        audit_enabled=False,
        verification_service=AdaptiveFactVerificationService(),
    )
    profile = checker.policy_repository.load("customer_support")
    profile.checks["hallucination"].depth = VerificationDepth.QUICK
    report = checker.check(
        Interaction(
            profile=profile.id,
            prompt="What was revenue?",
            response="Revenue was $12 million in 2024.",
            context="Revenue was $12 million in 2024.",
        ),
        profile,
    )

    adaptive = next(item for item in report.detector_results if item.detector == "adaptivefact")
    assert adaptive.error is None
    assert adaptive.metadata["verification_depth"] == "quick"
    assert adaptive.metadata["phase6"] == {}


def test_standard_depth_runs_retrieval_nli_and_policy_uses_result():
    service = AdaptiveFactVerificationService(
        nli=ExactEvidenceNLI(),
        nli_config=Phase6NLIConfig(min_contradiction_evidence_count=1),
    )
    checker = ControlPlane(audit_enabled=False, verification_service=service)
    profile = checker.policy_repository.load("regulated_decision_support")
    profile.checks["hallucination"].depth = VerificationDepth.STANDARD
    report = checker.check(
        Interaction(
            profile=profile.id,
            prompt="Who founded Tesla?",
            response="Elon Musk founded Tesla.",
            context="Elon Musk joined Tesla in 2004.",
            consequential=True,
        ),
        profile,
    )

    adaptive = next(item for item in report.detector_results if item.detector == "adaptivefact")
    assert adaptive.metadata["phase6"]["nli_pairs"] > 0
    contradiction = next(item for item in adaptive.findings if item.subtype == "claim_contradicted")
    assert contradiction.status == FindingStatus.CONTRADICTED
    assert contradiction.metadata["nli_scores"]["contradiction"] == 0.98
    assert adaptive.metadata["claim_results"][0]["nli_scores"]["contradiction"] == 0.98
    assert report.decision.action == EnforcementAction.REVIEW
    assert "regulated-contradiction-review" in report.decision.matched_rules


def test_customer_policy_blocks_confirmed_contradiction():
    service = AdaptiveFactVerificationService(
        nli=ExactEvidenceNLI(),
        nli_config=Phase6NLIConfig(min_contradiction_evidence_count=1),
    )
    checker = ControlPlane(audit_enabled=False, verification_service=service)
    report = checker.check(
        Interaction(
            profile="customer_support",
            prompt="Who founded Tesla?",
            response="Elon Musk founded Tesla.",
            context="Elon Musk joined Tesla in 2004.",
        )
    )

    assert report.decision.action == EnforcementAction.BLOCK
    assert "customer-contradiction-block" in report.decision.matched_rules
    assert report.final_response == checker.policy_repository.load("customer_support").blocked_response


def test_deep_depth_runs_bounded_agent_for_remaining_unknown_claims():
    nli = ExactEvidenceNLI()
    agent = BoundedVerificationAgent(
        nli,
        [ContextSearchTool()],
        AgentVerificationConfig(max_iterations=1, max_search_calls=1),
    )
    service = AdaptiveFactVerificationService(nli=nli, agent=agent)
    checker = ControlPlane(audit_enabled=False, verification_service=service)
    report = checker.check(
        Interaction(
            profile="regulated_decision_support",
            prompt="Who approved the request?",
            response="Northstar approved the request.",
            context="The request exists, but the approver is not recorded.",
            consequential=True,
        )
    )

    adaptive = next(item for item in report.detector_results if item.detector == "adaptivefact")
    assert adaptive.metadata["verification_depth"] == "deep"
    assert adaptive.metadata["agent"]["claims"] > 0
    assert any(item.status == FindingStatus.UNKNOWN for item in adaptive.findings)
    assert report.decision.action == EnforcementAction.REVIEW


def test_configured_deep_backend_failure_is_policy_visible():
    checker = ControlPlane(
        audit_enabled=False,
        verification_service=AdaptiveFactVerificationService(nli=ExactEvidenceNLI()),
    )
    report = checker.check(
        Interaction(
            profile="regulated_decision_support",
            prompt="Recommend an outcome.",
            response="Approve the request.",
            context="The request meets the requirements.",
            consequential=True,
        )
    )

    adaptive = next(item for item in report.detector_results if item.detector == "adaptivefact")
    assert adaptive.error and "bounded agent" in adaptive.error
    assert any(item.subtype == "detector_failure" for item in report.findings)
    assert report.decision.action == EnforcementAction.REVIEW


def test_normalized_supported_claim_emits_no_deep_hallucination_finding():
    service = AdaptiveFactVerificationService(nli=ExactEvidenceNLI())
    result = service.verify(
        Interaction(
            profile="customer_support",
            prompt="Where is Paris?",
            response="Paris is the capital of France.",
            context="Paris is the capital of France.",
        ),
        VerificationDepth.STANDARD,
    )

    assert result.metadata["status_counts"].get(VerificationStatus.SUPPORTED.value, 0) >= 1
    assert result.findings == []
