from __future__ import annotations

from adaptivefact.agents import AgentVerificationConfig, BoundedVerificationAgent, SearchResult
from adaptivefact.agents.tools import CorpusSearchTool, SearchTool
from adaptivefact.data.schema import (
    Claim,
    ClaimType,
    ResponseLabel,
    ResponseRecord,
    VerificationRoute,
    VerificationStatus,
)
from adaptivefact.pipeline import (
    AdaptivePipelineConfig,
    AdaptiveVerificationComponent,
    AdaptiveVerificationPipeline,
    aggregate_claims,
    claim_priority,
)
from adaptivefact.routing.router import RoutingDecision
from adaptivefact.verification.evidence import EvidenceRetrieverConfig
from adaptivefact.verification.nli import NLIScorer, NLIScores
from adaptivefact.verification.phase5 import Phase5Pipeline
from adaptivefact.verification.phase6 import Phase6NLIConfig, Phase6Pipeline


class ValueAwareNLI(NLIScorer):
    def score(self, premises, hypotheses):
        outputs = []
        for premise, hypothesis in zip(premises, hypotheses):
            if "$12,000" in premise and "$12,000" in hypothesis:
                outputs.append(NLIScores(entailment=0.97, contradiction=0.02, neutral=0.01))
            elif "$8,000" in premise and "$12,000" in hypothesis:
                outputs.append(NLIScores(entailment=0.01, contradiction=0.98, neutral=0.01))
            else:
                outputs.append(NLIScores(entailment=0.35, contradiction=0.25, neutral=0.40))
        return outputs


class EmptySearchTool(SearchTool):
    name = "empty"

    def __init__(self):
        self.calls = 0

    def search(self, query, *, record, claim, top_k=3):
        self.calls += 1
        return []


class ConflictingSearchTool(SearchTool):
    name = "conflicting"

    def search(self, query, *, record, claim, top_k=3):
        return [
            SearchResult(
                source_id="source-support",
                source_name="Supporting system",
                text="The customer account balance is $12,000.",
                authority_score=0.9,
                retrieval_score=1.0,
            ),
            SearchResult(
                source_id="source-contradiction",
                source_name="Contradicting system",
                text="The customer account balance is $8,000.",
                authority_score=1.0,
                retrieval_score=1.0,
            ),
        ]


class StaticRouter:
    def __init__(self, route=VerificationRoute.AGENTIC, risk=0.90):
        self.route_value = route
        self.risk = risk

    def route(self, record):
        return RoutingDecision(
            route=self.route_value,
            raw_risk=self.risk,
            calibrated_risk=self.risk,
            latency_ms=0.1,
        )


def record(response="$12,000 is the customer account balance.", context="External verification is required."):
    return ResponseRecord(
        id="agent-record",
        dataset="synthetic",
        query="What is the verified customer account balance?",
        context=context,
        generated_response=response,
        metadata={"consequential": True},
    )


def numeric_claim():
    return Claim(
        id="claim-1",
        text="The customer account balance is $12,000.",
        type=ClaimType.NUMERIC,
        numbers=["$12,000"],
        status=VerificationStatus.UNKNOWN,
    )


def agent_with_corpus(documents):
    return BoundedVerificationAgent(
        ValueAwareNLI(),
        [CorpusSearchTool(documents)],
        AgentVerificationConfig(
            entailment_threshold=0.85,
            contradiction_threshold=0.85,
            decision_margin=0.20,
        ),
    )


def test_agent_resolves_authoritative_contradiction_with_citation():
    agent = agent_with_corpus(
        [
            {
                "id": "account-db",
                "name": "Authoritative Account Database",
                "text": "The verified customer account balance is $8,000.",
                "authority_score": 1.0,
            }
        ]
    )
    result = agent.verify(record(), numeric_claim())

    assert result.status == VerificationStatus.CONTRADICTED
    assert result.stop_reason == "resolved_contradicted"
    assert result.evidence[0].source_name == "Authoritative Account Database"
    assert result.search_calls == 1
    assert result.nli_pairs == 1


def test_agent_resolves_support():
    agent = agent_with_corpus(
        [
            {
                "id": "account-db",
                "name": "Authoritative Account Database",
                "text": "The customer account balance is $12,000.",
                "authority_score": 1.0,
            }
        ]
    )
    result = agent.verify(record(), numeric_claim())
    assert result.status == VerificationStatus.SUPPORTED
    assert result.stop_reason == "resolved_supported"


def test_conflicting_authoritative_evidence_remains_unknown():
    agent = BoundedVerificationAgent(
        ValueAwareNLI(),
        [ConflictingSearchTool()],
        AgentVerificationConfig(entailment_threshold=0.85, contradiction_threshold=0.85),
    )
    result = agent.verify(record(), numeric_claim())
    assert result.status == VerificationStatus.UNKNOWN
    assert result.stop_reason == "conflicting_evidence"
    assert len(result.evidence) == 2


def test_agent_stops_at_search_budget_without_evidence():
    tool = EmptySearchTool()
    agent = BoundedVerificationAgent(
        ValueAwareNLI(),
        [tool],
        AgentVerificationConfig(max_iterations=5, max_search_calls=2),
    )
    result = agent.verify(record(), numeric_claim())
    assert result.status == VerificationStatus.UNKNOWN
    assert result.stop_reason == "no_reliable_evidence"
    assert result.search_calls == 2
    assert tool.calls == 2


def test_priority_and_aggregation_preserve_important_unknown():
    claim = numeric_claim()
    priority = claim_priority(claim, 0.8, consequential=True)
    claim.risk_score = priority
    result = aggregate_claims([claim], important_unknown_threshold=0.75)
    assert priority == 1.0
    assert result.label == ResponseLabel.UNKNOWN
    assert result.important_unknown_claims == 1


def test_phase7_agentic_route_executes_agent_and_aggregates_contradiction():
    nli = ValueAwareNLI()
    phase6 = Phase6Pipeline(
        nli,
        retrieval_config=EvidenceRetrieverConfig(top_k=1),
        nli_config=Phase6NLIConfig(
            entailment_threshold=0.90,
            contradiction_threshold=0.90,
            decision_margin=0.20,
        ),
    )
    agent = agent_with_corpus(
        [
            {
                "id": "account-db",
                "name": "Authoritative Account Database",
                "text": "The verified customer account balance is $8,000.",
                "authority_score": 1.0,
            }
        ]
    )
    pipeline = AdaptiveVerificationPipeline(
        StaticRouter(),
        Phase5Pipeline(),
        phase6,
        agent=agent,
        config=AdaptivePipelineConfig(
            deep_max_nli_claims=5,
            agent_min_claim_priority=0.75,
            agent_max_claims=2,
        ),
    )
    processed, runtime = pipeline.process(record())

    assert processed.verification_metadata.route == VerificationRoute.AGENTIC
    assert processed.verification_metadata.final_decision == ResponseLabel.HALLUCINATED
    assert runtime["agent"]["claims"] == 1
    assert runtime["agent"]["resolved_contradicted"] == 1
    assert processed.atomic_claims[0].verifier_used == "agent:resolved_contradicted"


def test_lightweight_route_does_not_invoke_agent():
    tool = EmptySearchTool()
    agent = BoundedVerificationAgent(ValueAwareNLI(), [tool])
    pipeline = AdaptiveVerificationPipeline(
        StaticRouter(route=VerificationRoute.LIGHTWEIGHT, risk=0.5),
        Phase5Pipeline(),
        Phase6Pipeline(ValueAwareNLI()),
        agent=agent,
        config=AdaptivePipelineConfig(lightweight_max_nli_claims=1),
    )
    _, runtime = pipeline.process(record())
    assert runtime["agent"]["claims"] == 0
    assert tool.calls == 0


def test_response_agent_budget_can_prevent_new_claim_work():
    tool = EmptySearchTool()
    agent = BoundedVerificationAgent(ValueAwareNLI(), [tool])
    pipeline = AdaptiveVerificationPipeline(
        StaticRouter(route=VerificationRoute.AGENTIC, risk=0.9),
        Phase5Pipeline(),
        Phase6Pipeline(ValueAwareNLI()),
        agent=agent,
        config=AdaptivePipelineConfig(
            deep_max_nli_claims=0,
            agent_min_claim_priority=0.0,
            agent_response_latency_budget_ms=0.0,
        ),
    )

    _, runtime = pipeline.process(record())

    assert runtime["agent"]["claims"] == 0
    assert runtime["agent"]["response_budget_exhausted"] is True
    assert runtime["agent"]["claims_skipped_due_response_budget"] == 1
    assert tool.calls == 0


def test_supported_claim_confidence_is_not_misreported_as_hallucination_risk():
    pipeline = AdaptiveVerificationPipeline(
        StaticRouter(route=VerificationRoute.LIGHTWEIGHT, risk=0.2),
        Phase5Pipeline(),
        Phase6Pipeline(ValueAwareNLI()),
        config=AdaptivePipelineConfig(lightweight_max_nli_claims=1),
    )
    component = AdaptiveVerificationComponent(pipeline)
    supported = record(
        response="The customer account balance is $12,000.",
        context="The customer account balance is $12,000.",
    )
    result = component.run(supported)
    assert result.prediction == 0
    assert result.confidence == 0.2
