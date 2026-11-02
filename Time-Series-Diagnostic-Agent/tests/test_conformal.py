import numpy as np
import pytest

from tsdiag.uncertainty import (
    ConformalClassifier,
    ConformalizedQuantileRegressor,
    SplitConformalRegressor,
    accuracy_at_coverage,
    aurc,
    conformal_quantile,
    expected_calibration_error,
    fit_temperature,
    interval_coverage,
    risk_coverage_curve,
    selective_summary,
    set_coverage,
    softmax,
)


def _make_probs(rng, n, k=4, sep=3.0, noise=1.5):
    y = rng.integers(0, k, n)
    logits = noise * rng.normal(size=(n, k))
    logits[np.arange(n), y] += sep
    return softmax(logits), logits, y


def test_conformal_quantile_needs_enough_points():
    assert np.isinf(conformal_quantile(np.arange(5), 0.1))
    assert np.isfinite(conformal_quantile(np.arange(9), 0.1))


@pytest.mark.parametrize("method", ["lac", "aps"])
@pytest.mark.parametrize("class_conditional", [False, True])
def test_classification_coverage(method, class_conditional):
    rng = np.random.default_rng(0)
    p_cal, _, y_cal = _make_probs(rng, 4000)
    p_te, _, y_te = _make_probs(rng, 8000)
    cc = ConformalClassifier(0.1, method, class_conditional).fit(p_cal, y_cal)
    sets = cc.predict_sets(p_te)
    assert set_coverage(sets, y_te) >= 0.88
    if class_conditional:
        for k in range(4):
            assert set_coverage(sets[y_te == k], y_te[y_te == k]) >= 0.86


def test_decide_abstains_on_non_singletons():
    cc = ConformalClassifier(0.1, "lac").fit(np.tile([0.4, 0.6], (20, 1)), np.zeros(20))
    d = cc.decide([[0.8, 0.2], [0.5, 0.5], [0.2, 0.8]])
    np.testing.assert_array_equal(d.label, [0, -1, 1])
    np.testing.assert_array_equal(d.set_size, [1, 2, 1])
    s = selective_summary(d.label, [0, 1, 1], d.abstain)
    assert s["selective_accuracy"] == 1.0 and s["selective_risk"] == 0.0
    assert s["coverage"] == 2 / 3
    strict = ConformalClassifier(0.1).fit(np.tile([0.9, 0.1], (20, 1)), np.zeros(20))
    assert strict.decide([[0.5, 0.5]]).abstain[0]  # empty set also abstains


def test_split_conformal_and_cqr_coverage():
    rng = np.random.default_rng(2)
    x = rng.uniform(-3, 3, 6000)
    y = x + rng.normal(scale=1 + 0.5 * np.abs(x))
    cal, te = slice(0, 3000), slice(3000, None)
    sc = SplitConformalRegressor(0.1).fit(y[cal], x[cal])
    lo, hi = sc.predict_interval(x[te])
    assert interval_coverage(y[te], lo, hi) >= 0.88
    # deliberately miscalibrated quantile band
    cqr = ConformalizedQuantileRegressor(0.1).fit(y[cal], x[cal] - 0.5, x[cal] + 0.5)
    lo, hi = cqr.predict_interval(x[te] - 0.5, x[te] + 0.5)
    assert interval_coverage(y[te], lo, hi) >= 0.88


def test_clip_min_for_rul():
    sc = SplitConformalRegressor(0.1).fit(np.arange(50.0) + 3, np.arange(50.0))
    lo, _ = sc.predict_interval(np.array([1.0]), clip_min=0.0)
    assert lo[0] == 0.0


def test_temperature_scaling_reduces_ece():
    rng = np.random.default_rng(3)
    n, k = 6000, 4
    z = rng.normal(size=(n, k)) * 1.2
    p = softmax(z)
    y = np.array([rng.choice(k, p=row) for row in p])
    overconfident = 3.0 * z
    t = fit_temperature(overconfident, y)
    assert 2.4 < t < 3.6
    before = expected_calibration_error(softmax(overconfident), y)
    after = expected_calibration_error(softmax(overconfident, t), y)
    assert after < before


def test_risk_coverage_informative_confidence():
    rng = np.random.default_rng(4)
    p, _, y = _make_probs(rng, 4000, sep=2.0)
    conf, correct = p.max(1), p.argmax(1) == y
    cov, risk = risk_coverage_curve(conf, correct)
    assert cov[-1] == 1.0 and abs(risk[-1] - (1 - correct.mean())) < 1e-12
    assert accuracy_at_coverage(conf, correct, 0.3) > accuracy_at_coverage(conf, correct, 1.0)
    assert aurc(conf, correct) < 1 - correct.mean()


@pytest.mark.parametrize("probs,y", [([[0.4, 0.4]], [0]), ([[np.nan, 1]], [0]),
                                     ([[0.2, 0.8]], [0.5]), ([[0.2, 0.8]], [-1]),
                                     (np.empty((0, 2)), [])])
def test_classifier_rejects_invalid_calibration(probs, y):
    with pytest.raises(ValueError):
        ConformalClassifier().fit(probs, y)


def test_missing_class_is_conservative_and_prediction_class_count_is_checked():
    model = ConformalClassifier(class_conditional=True).fit(np.tile([0.9, 0.1], (30, 1)), np.zeros(30))
    assert np.isinf(model.qhat_[1])
    assert model.predict_sets([[0.9, 0.1]])[0, 1]
    with pytest.raises(ValueError, match="classes"):
        model.predict_sets([[0.2, 0.3, 0.5]])


@pytest.mark.parametrize("scale", [0, -1, np.nan, [1, 2]])
def test_regression_rejects_invalid_scale(scale):
    with pytest.raises(ValueError, match="scale"):
        SplitConformalRegressor().fit([1, 2, 3], [1, 2, 3], scale=scale)


def test_regression_shapes_finite_scores_and_nonnegative_support():
    with pytest.raises(ValueError):
        SplitConformalRegressor().fit([1, 2], [[1], [2]])
    with pytest.raises(ValueError):
        conformal_quantile([1, np.nan], 0.1)
    with pytest.raises(ValueError):
        ConformalizedQuantileRegressor().fit([1], [2], [0])
    model = SplitConformalRegressor().fit(np.zeros(20), np.zeros(20))
    lo, hi = model.predict_interval([-10], clip_min=0)
    assert lo[0] == hi[0] == 0


def test_selective_metrics_reject_misalignment_and_handle_empty_inputs():
    with pytest.raises(ValueError):
        risk_coverage_curve([0.8], [True, False])
    with pytest.raises(ValueError):
        selective_summary([0], [0, 1], [False])
    with pytest.raises(ValueError):
        selective_summary([0.9], [0], [False])
    with pytest.raises(ValueError):
        softmax([[0, 1]], 0)
    assert np.isnan(accuracy_at_coverage([], [], 0.5))
    assert np.isnan(selective_summary([], [], [])["coverage"])
    assert np.isnan(expected_calibration_error(np.empty((0, 2)), []))


def test_temperature_stays_in_declared_search_bounds():
    logits = np.array([[3.0, 0.0], [0.0, 3.0]])
    assert 0.5 <= fit_temperature(logits, [0, 1], low=0.5, high=2) <= 2
