import numpy as np

from adaptivefact.risk.calibration import fit_candidate_calibrators
from adaptivefact.risk.threshold_optimizer import optimize_routing_thresholds, routing_statistics


def test_calibrators_return_probabilities():
    p = np.array([0.05, 0.15, 0.25, 0.65, 0.80, 0.95])
    y = np.array([0, 0, 0, 1, 1, 1])
    calibrators = fit_candidate_calibrators(p, y)
    for calibrator in calibrators.values():
        out = calibrator.predict(p)
        assert out.shape == p.shape
        assert np.all(out >= 0.0)
        assert np.all(out <= 1.0)


def test_threshold_optimizer_respects_order_and_stats():
    p = np.linspace(0.01, 0.99, 200)
    y = (p > 0.55).astype(int)
    result = optimize_routing_thresholds(
        y,
        p,
        max_fast_hallucination_rate=0.05,
        min_hallucination_recall_outside_fast=0.98,
        min_fast_count=10,
        min_agentic_hallucination_rate=0.8,
        min_agentic_count=10,
    )
    assert 0.0 <= result.tau1 < result.tau2 <= 1.0
    stats = routing_statistics(y, p, result.tau1, result.tau2)
    assert stats["fast_path_hallucination_rate"] <= 0.05
    assert stats["agentic_hallucination_rate"] >= 0.8
