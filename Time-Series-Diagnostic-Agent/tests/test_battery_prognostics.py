import numpy as np
import pytest

from tsdiag.challengers.battery_prognostics import (
    CellData,
    FadeParticleFilter,
    GPFadeModel,
    alpha_lambda_hits,
    crossing_cycle,
    fade_curve,
    fit_fade,
    ic_peak,
    incremental_capacity,
    leave_one_cell_out,
    prognostic_horizon,
    survival_consistency,
    time_to_voltage,
    weighted_quantiles,
)

EOL = 1.4


def _cell(rng, cid, censor_at=None, noise=0.01):
    q0 = rng.uniform(1.95, 2.05)
    s = rng.uniform(0.0022, 0.0032)
    g = rng.uniform(0.004, 0.009)
    k = np.arange(0, 400, dtype=float)
    q = fade_curve(k, q0, s, g) + rng.normal(scale=noise, size=k.size)
    below = np.where(q <= EOL)[0]
    eol = float(k[below[0]])
    n = below[0] + 1
    if censor_at is not None:
        n = int(censor_at * eol)
        return CellData(cid, k[:n], q[:n], None)
    return CellData(cid, k[:n], q[:n], eol)


def test_crossing_matches_curve():
    q0, s, g = 2.0, 0.0025, 0.006
    kc = crossing_cycle(q0, s, g, EOL)[0]
    assert fade_curve(kc, q0, s, g) == pytest.approx(EOL, abs=1e-6)
    assert crossing_cycle(2.0, 0.0025, 0.0, EOL)[0] == pytest.approx(0.6 / 0.0025)
    assert np.isinf(crossing_cycle(2.0, 0.001, -0.01, EOL)[0])     # decelerating fade never crosses
    assert crossing_cycle(1.3, 0.001, 0.0, EOL)[0] == 0.0          # already below threshold


def test_fit_fade_recovers_parameters():
    rng = np.random.default_rng(0)
    k = np.arange(150.0)
    y = fade_curve(k, 2.0, 0.0025, 0.006) + rng.normal(scale=0.005, size=k.size)
    theta, cov = fit_fade(k, y)
    assert theta[0] == pytest.approx(2.0, abs=0.03)
    assert theta[1] == pytest.approx(0.0025, rel=0.3)
    assert np.all(np.isfinite(cov))


def test_weighted_quantiles_handle_inf():
    v = np.array([10.0, 20.0, np.inf, np.inf])
    w = np.array([0.25, 0.25, 0.25, 0.25])
    q = weighted_quantiles(v, w, [0.25, 0.5, 0.9])
    assert q[0] == 10.0 and q[1] == 20.0 and np.isinf(q[2])


def test_particle_filter_loco_accuracy_and_coverage():
    rng = np.random.default_rng(1)
    cells = [_cell(rng, f"c{i}") for i in range(10)]
    res = leave_one_cell_out(cells, EOL, "pf", start_fractions=(0.5, 0.65, 0.8), seed=0)
    assert res["n"] >= 25
    assert res["median_rel_error"] < 0.15
    assert res["interval90_coverage"] >= 0.7          # nominal 0.9; small sample
    assert res["alpha_lambda_accuracy"] >= 0.6
    assert len(res["per_cell_median_abs_error"]) == 10
    assert res["n_cells"] == 10 and res["point_forecast_coverage"] >= 0.9
    assert res["interval_calibrated"] is False


def test_prior_requires_two_cells():
    rng = np.random.default_rng(2)
    with pytest.raises(ValueError):
        FadeParticleFilter.prior_from_cells([_cell(rng, "a")])


def test_censored_cell_is_scored_for_consistency_not_error():
    rng = np.random.default_rng(3)
    cells = [_cell(rng, f"c{i}") for i in range(6)]
    cells.append(_cell(rng, "censored", censor_at=0.9))
    res = leave_one_cell_out(cells, EOL, "pf", start_fractions=(0.5, 0.7), seed=0)
    assert all(r["cell"] != "censored" for r in res["rows"])
    assert "censored_mean_consistency" in res
    assert res["censored_mean_consistency"] > 0.6


def test_survival_consistency_definition():
    r = np.array([5.0, 20.0, 40.0, np.inf])
    w = np.full(4, 0.25)
    assert survival_consistency(r, w, 10.0) == pytest.approx(0.75)


def test_gp_model_gives_ordered_quantiles_and_reasonable_median():
    rng = np.random.default_rng(4)
    c = _cell(rng, "x")
    cut = int(0.7 * len(c.cycles))
    gp = GPFadeModel(EOL, horizon=250, n_samples=150, seed=0).fit(c.cycles[:cut], c.capacity[:cut])
    lo, med, hi = gp.rul_quantiles()
    true = c.eol_cycle - c.cycles[cut - 1]
    assert lo <= med <= hi
    assert abs(med - true) / true < 0.35


def test_gp_loco_runs():
    rng = np.random.default_rng(5)
    cells = [_cell(rng, f"c{i}") for i in range(4)]
    res = leave_one_cell_out(cells, EOL, "gp", start_fractions=(0.6,), seed=0, horizon=250, n_samples=100)
    assert res["n"] == 4 and np.isfinite(res["median_abs_error"])


def test_prognostic_metrics():
    true = np.array([50.0, 40, 30, 20, 10])
    pred = np.array([80.0, 45, 31, 19, 10])
    assert alpha_lambda_hits(true, pred, 0.2).tolist() == [False, True, True, True, True]
    assert prognostic_horizon(true, pred, 0.2) == 40.0
    assert prognostic_horizon(true, np.array([80, 45, 31, 19, 20.0]), 0.2) == 0.0


# ---- discharge-curve features ------------------------------------------------
def _discharge(peak_height, current=2.0, v_lo=2.7, v_hi=4.2):
    vv = np.linspace(v_hi, v_lo, 4000)
    ic = 0.4 + peak_height * np.exp(-(((vv - 3.7) / 0.07) ** 2))      # Ah/V, true IC curve
    q = np.concatenate([[0.0], np.cumsum(0.5 * (ic[1:] + ic[:-1]) * np.abs(np.diff(vv)))])
    t = q * 3600.0 / current
    tt = np.linspace(0, t[-1], 2000)
    v = np.interp(tt, t, vv)
    return tt, v, np.full_like(tt, current)


def test_ic_peak_location_and_degradation_trend():
    t, v, i = _discharge(peak_height=2.0)
    g, ic = incremental_capacity(t, v, i)
    h, pv = ic_peak(g, ic)
    assert pv == pytest.approx(3.7, abs=0.05)
    t2, v2, i2 = _discharge(peak_height=1.2)                          # aged cell: smaller peak
    h2, _ = ic_peak(*incremental_capacity(t2, v2, i2))
    assert h2 < h


def test_time_to_voltage():
    t = np.array([0.0, 10, 20, 30])
    v = np.array([4.0, 3.8, 3.6, 3.2])
    assert time_to_voltage(t, v, 3.7) == pytest.approx(15.0)
    assert np.isnan(time_to_voltage(t, v, 3.0))


@pytest.mark.parametrize("cycles,capacity", [([], []), ([0, 0, 1], [2, 1.9, 1.8]),
    ([0, 2, 1], [2, 1.9, 1.8]), ([0, 1, 2], [2, np.nan, 1.8]),
    ([0, 1, 2], [2, 1.9, -1]), ([0, 1, 2], [2, 1.9])])
def test_invalid_capacity_histories_fail_before_fitting(cycles, capacity):
    with pytest.raises(ValueError):
        fit_fade(cycles, capacity)


def test_cell_records_own_their_arrays_and_reject_duplicate_identity_in_loco():
    cycles, capacity = np.arange(30.0), 2 - 0.002 * np.arange(30.0)
    cell = CellData("cell", cycles, capacity)
    capacity[:] = 1
    assert cell.capacity[0] == 2 and not cell.capacity.flags.writeable
    with pytest.raises(ValueError, match="unique"):
        leave_one_cell_out([cell, cell], EOL)
    with pytest.raises(ValueError):
        CellData("cell", cycles, cell.capacity, eol_cycle=100)


@pytest.mark.parametrize("values,weights,qs", [([1, 2], [0, 0], [0.5]),
    ([1, 2], [1, -1], [0.5]), ([1, np.nan], [1, 1], [0.5]),
    ([1, 2], [1], [0.5]), ([1], [1], [1.1]), ([1], [np.inf], [0.5])])
def test_invalid_weighted_distributions_are_rejected(values, weights, qs):
    with pytest.raises(ValueError):
        weighted_quantiles(values, weights, qs)


def test_zero_weight_atoms_do_not_affect_extreme_quantiles():
    np.testing.assert_array_equal(weighted_quantiles([0, 10, np.inf], [0, 1, 0], [0, 1]), [10, 10])


def test_population_prior_refuses_its_own_training_cell_and_requires_identity():
    rng = np.random.default_rng(10)
    cells = [_cell(rng, "a"), _cell(rng, "b")]
    prior = FadeParticleFilter.prior_from_cells(cells)
    assert prior["training_cell_ids"] == ("a", "b")
    history = cells[0]
    for identity in (None, "a"):
        with pytest.raises(ValueError, match="cell"):
            FadeParticleFilter(EOL, n_particles=50, prior=prior).fit(
                history.cycles[:30], history.capacity[:30], cell_id=identity)


@pytest.mark.parametrize("constructor", [FadeParticleFilter, GPFadeModel])
def test_models_require_fit_and_observed_eol_returns_zero(constructor):
    model = constructor(EOL)
    with pytest.raises(RuntimeError, match="fit"):
        model.rul_samples()
    cycles = np.arange(30.0)
    capacity = 1.6 - 0.01 * cycles
    model.fit(cycles, capacity)
    samples, weights = model.rul_samples()
    assert np.all(samples == 0) and weights.sum() == pytest.approx(1)
    with pytest.raises(ValueError):
        model.fit(cycles, np.full(30, np.nan))
    with pytest.raises(RuntimeError, match="fit"):
        model.rul_samples()


def test_particle_filter_prefix_predictions_ignore_future_source_mutations():
    cycles = np.arange(50.0)
    capacity = fade_curve(cycles, 2.0, 0.0025, 0.006)
    model = FadeParticleFilter(EOL, n_particles=100, seed=1).fit(cycles[:30], capacity[:30])
    before = model.rul_quantiles()
    capacity[:] = 1
    np.testing.assert_array_equal(model.rul_quantiles(), before)


def test_no_prior_filter_does_not_initialize_from_a_full_history_fit(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("full-prefix fit must not be reused as a prior")
    monkeypatch.setattr("tsdiag.challengers.battery_prognostics.fit_fade", forbidden)
    cycles = np.arange(30.0)
    model = FadeParticleFilter(EOL, n_particles=100, sigma_obs=0.01).fit(cycles, 2 - 0.002 * cycles)
    samples, weights = model.rul_samples()
    assert np.all(samples >= 0) and weights.sum() == pytest.approx(1)


def test_gp_no_crossing_tail_retains_its_finite_horizon():
    cycles = np.arange(30.0)
    model = GPFadeModel(EOL, horizon=2, n_samples=10).fit(cycles, 2 - 0.001 * cycles)
    samples, weights = model.rul_samples()
    assert model.censoring_horizon_ == 2 and np.isinf(samples).all()
    assert model.interval_calibrated_ is False
    with pytest.raises(ValueError, match="horizon"):
        survival_consistency(samples, weights, 3, censoring_horizon=model.censoring_horizon_)


def test_survival_does_not_claim_known_survival_past_forecast_horizon():
    assert survival_consistency([5, 10, np.inf], [1, 1, 1], 10) == pytest.approx(1 / 3)
    with pytest.raises(ValueError, match="horizon"):
        survival_consistency([np.inf], [1], 30, censoring_horizon=20)


def test_loco_marks_finite_horizon_censored_checks_unscorable(monkeypatch):
    # Exercise evaluation semantics without fitting a GP for each tiny horizon.
    class HorizonModel:
        def __init__(self, *args, **kwargs):
            self.censoring_horizon_ = 2
        def fit(self, *args, **kwargs):
            return self
        def rul_samples(self):
            return np.array([np.inf]), np.array([1.0])
    monkeypatch.setattr("tsdiag.challengers.battery_prognostics.GPFadeModel", HorizonModel)
    cycles = np.arange(40.0)
    cell = CellData("censored", cycles, 2 - 0.001 * cycles)
    result = leave_one_cell_out([cell], EOL, method="gp", start_fractions=(0.5,))
    assert result["n_censored_unscorable"] == 1 and result["censored_mean_consistency"] is None
    assert result["censored_rows"][0]["status"] == "unscorable_beyond_prediction_horizon"


def test_loco_does_not_count_unknown_gp_tail_as_interval_coverage(monkeypatch):
    class HorizonModel:
        def __init__(self, *args, **kwargs):
            self.censoring_horizon_ = 2
        def fit(self, *args, **kwargs):
            return self
        def rul_samples(self):
            return np.array([1.0, np.inf]), np.array([0.6, 0.4])
    monkeypatch.setattr("tsdiag.challengers.battery_prognostics.GPFadeModel", HorizonModel)
    cycles = np.arange(40.0)
    cell = CellData("observed", cycles, 2 - 0.02 * cycles, eol_cycle=30)
    result = leave_one_cell_out([cell], EOL, method="gp", start_fractions=(0.5,))
    assert result["n_interval_unscorable"] == 1
    assert result["n_interval_scorable"] == 0 and result["interval90_coverage"] is None
    assert result["rows"][0]["interval_contains_truth"] is None
    assert result["n_point_forecasts"] == 1  # finite point error remains measurable


def test_loco_never_passes_held_cell_to_population_prior(monkeypatch):
    from tsdiag.challengers import battery_prognostics as module
    rng = np.random.default_rng(11)
    cells = [_cell(rng, name) for name in ("a", "b", "c")]
    original = module.FadeParticleFilter.prior_from_cells
    observed = []
    def record(training, *args, **kwargs):
        observed.append(tuple(cell.cell_id for cell in training))
        return original(training, *args, **kwargs)
    monkeypatch.setattr(module.FadeParticleFilter, "prior_from_cells", staticmethod(record))
    leave_one_cell_out(cells, EOL, start_fractions=(0.5,), n_particles=100)
    assert observed == [("b", "c"), ("a", "c"), ("a", "b")]


@pytest.mark.parametrize("fractions", [(0.5, 0.5), (0,), (1,), (np.nan,)])
def test_invalid_retrospective_cutoff_protocol_is_rejected(fractions):
    rng = np.random.default_rng(12)
    with pytest.raises(ValueError, match="fractions"):
        leave_one_cell_out([_cell(rng, "cell")], EOL, method="gp", start_fractions=fractions)


def test_censoring_metadata_must_not_hide_an_observed_crossing():
    cell = CellData("bad-censor", np.arange(30.0), 1.6 - 0.01 * np.arange(30.0))
    with pytest.raises(ValueError, match="crossing"):
        leave_one_cell_out([cell], EOL, method="gp")


def test_discharge_features_reject_noncausal_times_reversals_and_bad_smoothing():
    with pytest.raises(ValueError):
        time_to_voltage([0, 2, 1], [4, 3.8, 3.6], 3.7)
    t, v, i = _discharge(2.0)
    v[20] = v[19] + 0.1
    with pytest.raises(ValueError, match="monotone"):
        incremental_capacity(t, v, i)
    t, v, i = _discharge(2.0)
    with pytest.raises(ValueError, match="odd"):
        incremental_capacity(t, v, i, window=6)
    i[0] *= -1
    with pytest.raises(ValueError, match="charge"):
        incremental_capacity(t, v, i)
