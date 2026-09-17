from controlplane import ControlPlane
from controlplane.evaluation import evaluate_scenarios, load_scenarios


def test_controlled_cross_risk_scenarios_run_end_to_end():
    scenarios = load_scenarios("data/controlplane/scenarios.jsonl")
    results = evaluate_scenarios(ControlPlane(audit_enabled=False), scenarios)

    assert results["n"] == 8
    assert results["action_accuracy"] == 1.0
    assert results["unsafe_allow_rate"] == 0.0
    assert {"privacy", "bias", "hallucination", "policy"} <= set(results["category_metrics"])


def test_expanded_development_evaluation_set_is_valid_and_runs():
    scenarios = load_scenarios("data/controlplane/evaluation_v1.jsonl")
    results = evaluate_scenarios(ControlPlane(audit_enabled=False), scenarios)

    assert len(scenarios) >= 30
    assert results["n"] == len(scenarios)
    assert {"privacy", "bias", "hallucination", "policy"} <= set(results["category_metrics"])
    assert {
        "email",
        "api_key",
        "explicit_stereotype",
        "compounding_conversation_risk",
        "evidence_unavailable",
    } <= set(results["subtype_metrics"])
