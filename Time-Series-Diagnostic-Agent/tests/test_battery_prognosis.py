import numpy as np

from tsdiag.domains.battery_prognosis_runner import BatteryPrognosticPipeline
from tsdiag.tools.battery_prognosis import estimate_capacity_eol


def _history(n=100):
    cycles = np.arange(1, n + 1, dtype=float)
    capacity = 2.0 - 0.004 * cycles + 0.002 * np.sin(cycles / 4.0)
    return cycles, capacity


def test_prefix_prediction_does_not_depend_on_future_capacity_values():
    cycles, capacity = _history(120)
    pipeline = BatteryPrognosticPipeline()

    first = pipeline.run(cycles[:80], capacity[:80], battery_id="synthetic")

    modified = capacity.copy()
    modified[80:] = modified[80:] - 0.4
    second = pipeline.run(cycles[:80], modified[:80], battery_id="synthetic")

    assert first.prognosis is not None
    assert second.prognosis is not None
    assert first.prognosis.remaining_useful_life == second.prognosis.remaining_useful_life
    assert first.prognosis.details["predicted_eol_cycle"] == second.prognosis.details["predicted_eol_cycle"]
    assert first.prognosis.details["eol_interval_cycles"] == second.prognosis.details["eol_interval_cycles"]


def test_capacity_estimator_returns_forward_uncertainty_interval():
    cycles, capacity = _history(90)
    result = estimate_capacity_eol(cycles, capacity)

    assert result["abstained"] is False
    lower, upper = result["eol_interval_cycles"]
    assert lower >= cycles[-1]
    assert lower <= result["predicted_eol_cycle"] <= upper
    assert result["remaining_useful_life_cycles"] >= 0


def test_capacity_estimator_abstains_without_degradation():
    cycles = np.arange(1, 60, dtype=float)
    capacity = np.full_like(cycles, 1.9)
    result = estimate_capacity_eol(cycles, capacity)

    assert result["abstained"] is True
    assert result["abstain_reason"] == "non_degrading_or_unstable_slope"
