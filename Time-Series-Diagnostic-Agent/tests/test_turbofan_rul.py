import numpy as np
import pytest

from tsdiag.challengers.turbofan_rul import (
    RegimeNormalizer,
    RULChallenger,
    cap_rul,
    evaluate_rul,
    make_windows,
    nasa_score,
    rmse,
    truncated_samples,
)
from tsdiag.calibration.split_guard import group_split


def _fleet(rng, n_engines, n_sensors=10, n_regimes=3, id0=0):
    """Synthetic C-MAPSS-like fleet: regime-dependent offsets + health-driven drift."""
    settings_c = rng.uniform(0, 1, size=(n_regimes, 3)) * 10
    rows = []
    gains = rng.normal(size=n_sensors) * 1.0
    offsets = rng.normal(size=(n_regimes, n_sensors)) * 5.0
    for e in range(n_engines):
        life = int(rng.integers(150, 300))
        for t in range(life):
            r = int(rng.integers(0, n_regimes))
            health = (t / life) ** 2.2
            s = offsets[r] + gains * health * 4.0 + rng.normal(scale=0.35, size=n_sensors)
            rows.append((id0 + e, t + 1, life - (t + 1), *(settings_c[r] + rng.normal(scale=0.02, size=3)), *s))
    a = np.array(rows)
    return a[:, 0].astype(int), a[:, 1].astype(int), a[:, 2], a[:, 3:6], a[:, 6:]


def test_metrics_and_cap():
    assert cap_rul([10, 200], 125).tolist() == [10, 125]
    assert rmse([1, 2], [1, 4]) == pytest.approx(np.sqrt(2))
    # late prediction is penalised more than an equally early one
    assert nasa_score([50], [60]) > nasa_score([50], [40])


def test_regime_normalizer_removes_regime_offsets():
    rng = np.random.default_rng(0)
    u, c, r, settings, sensors = _fleet(rng, 20)
    rn = RegimeNormalizer(3).fit(settings, sensors)
    z = rn.transform(settings, sensors)
    reg = rn.regime(settings)
    means = np.array([z[reg == k].mean(0) for k in range(3)])
    assert np.abs(means).max() < 1e-6          # per-regime means are removed


def test_make_windows_shapes_and_padding():
    u = np.array([1, 1, 1, 2, 2])
    c = np.array([1, 2, 3, 1, 2])
    x = np.arange(10, dtype=float).reshape(5, 2)
    W, uid, cyc = make_windows(u, c, x, window=4)
    assert W.shape == (5, 4, 2) and uid.tolist() == [1, 1, 1, 2, 2]
    assert np.all(W[0, :3] == x[0])             # front padding repeats the first row
    Wl, ul, _ = make_windows(u, c, x, window=4, last_only=True)
    assert Wl.shape[0] == 2 and ul.tolist() == [1, 2]


def test_challenger_end_to_end_accuracy_and_coverage():
    rng = np.random.default_rng(1)
    u, c, r, settings, sensors = _fleet(rng, 160)
    # engine-level split: train / calibration / held-out test
    ids = [str(i) for i in np.unique(u)]
    rest, test_ids = group_split(ids, fraction=0.3, seed=0)
    train_ids, cal_ids = group_split(list(rest), fraction=0.25, seed=1)
    sel = lambda ids_: np.isin(u, [int(i) for i in ids_])

    rn = RegimeNormalizer(3).fit(settings[sel(train_ids)], sensors[sel(train_ids)])
    z = rn.transform(settings, sensors)

    tr = sel(train_ids)
    Wtr, utr, _ = make_windows(u[tr], c[tr], z[tr], window=30)
    rul_tr = np.concatenate([r[tr][u[tr] == e][np.argsort(c[tr][u[tr] == e])] for e in np.unique(u[tr])])
    model = RULChallenger("hgb", n_members=4, window=30, alpha=0.1, seed=0).fit(Wtr, rul_tr, utr)

    ca = sel(cal_ids)
    Wc, Rc, uc = truncated_samples(u[ca], c[ca], z[ca], r[ca], window=30, seed=0)
    model.calibrate(Wc, Rc, units=uc)

    te = sel(test_ids)
    # at benchmark time engines are truncated at a random cycle; use random cuts for the test too
    Wt, Rt, ut = truncated_samples(u[te], c[te], z[te], r[te], window=30, seed=7)
    m = evaluate_rul(model, Wt, Rt, units=ut)

    baseline = rmse(Rt, np.full_like(Rt, cap_rul(rul_tr).mean()))
    assert m["rmse"] < 0.6 * baseline
    assert m["interval_target"] == "uncapped"
    assert m["interval_coverage_uncapped"] >= 0.8  # synthetic check, nominal 0.9
    assert m["n_engines"] == len(test_ids)
    assert np.isfinite(m["nasa_score"]) and m["mean_interval_width"] > 0
    mu, lo, hi = model.predict_interval(Wt, units=ut)
    assert np.all(lo >= 0) and np.all(lo <= mu + 1e-9) and np.all(hi >= mu - 1e-9)


def test_predict_interval_requires_calibration():
    m = RULChallenger("hgb", n_members=1)
    with pytest.raises(RuntimeError):
        m.predict_interval(np.zeros((1, 30, 4)))


def test_cnn_backend_fails_clearly_without_torch(monkeypatch):
    import builtins
    original = builtins.__import__

    def without_torch(name, *args, **kwargs):
        if name == "torch" or name.startswith("torch."):
            raise ImportError("synthetic missing optional dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", without_torch)
    with pytest.raises(ImportError, match="PyTorch"):
        RULChallenger("cnn")._member(0)


@pytest.fixture
def tiny_model():
    windows = np.arange(160, dtype=float).reshape(20, 4, 2) / 100
    model = RULChallenger(n_members=1, window=4, cap=20).fit(
        windows, np.full(20, 10.0), units=np.repeat([1, 2], 10))
    return model, windows


def test_calibration_requires_identified_independent_engines(tiny_model):
    model, windows = tiny_model
    with pytest.raises(ValueError, match="identities"):
        model.calibrate(windows[:10], np.full(10, 12.0))
    with pytest.raises(ValueError, match="disjoint"):
        model.calibrate(windows[:10], np.full(10, 12.0), units=["1", *map(str, range(10, 19))])
    with pytest.raises(ValueError, match="one window"):
        model.calibrate(windows[:10], np.full(10, 12.0), units=[10] * 10)
    with pytest.raises(ValueError, match="too few"):
        model.calibrate(windows[:5], np.full(5, 12.0), units=range(10, 15))


def test_calibrated_intervals_reject_reused_engines_and_refit_invalidates(tiny_model):
    model, windows = tiny_model
    model.calibrate(windows[:10], np.full(10, 12.0), units=range(10, 20))
    for unit in (1, 10):
        with pytest.raises(ValueError, match="disjoint"):
            model.predict_interval(windows[:1], units=[unit])
    with pytest.raises(ValueError, match="one window"):
        evaluate_rul(model, windows[:2], [12, 12], units=[99, 99])
    assert evaluate_rul(model, windows[:1], [12], units=[99])["n_engines"] == 1
    model.fit(windows, np.full(20, 10.0), units=np.repeat([1, 2], 10))
    assert model.conformal_ is None and not model.calibration_units_
    with pytest.raises(RuntimeError, match="calibrate"):
        model.predict_interval(windows[:1], units=[99])


def test_capped_and_uncapped_interval_targets_are_explicit(tiny_model):
    model, windows = tiny_model
    model.calibrate(windows[:10], np.full(10, 200.0), units=range(10, 20))
    uncapped = evaluate_rul(model, windows[:1], [200], units=[99])
    assert uncapped["interval_target"] == "uncapped"
    assert uncapped["interval_coverage_uncapped"] == 1
    model.calibrate(windows[:10], np.full(10, 200.0), units=range(10, 20), target="capped")
    capped = evaluate_rul(model, windows[:1], [200], units=[99])
    assert capped["interval_target"] == "capped"
    assert capped["interval_coverage_capped"] == 1
    assert capped["interval_coverage_uncapped"] == 0


def test_failed_recalibration_cannot_leave_a_stale_interval(tiny_model):
    model, windows = tiny_model
    model.calibrate(windows[:10], np.full(10, 12.0), units=range(10, 20))
    with pytest.raises(ValueError):
        model.calibrate(windows[:10], np.full(10, np.nan), units=range(10, 20))
    assert model.conformal_ is None
    with pytest.raises(RuntimeError, match="calibrate"):
        model.predict_interval(windows[:1], units=[99])


def test_prediction_shape_and_mutated_configuration_are_checked(tiny_model):
    model, windows = tiny_model
    with pytest.raises(ValueError, match="shape"):
        model.predict(np.zeros((1, 3, 2)))
    with pytest.raises(ValueError, match="shape"):
        model.predict(np.zeros((1, 4, 3)))
    model.scale_floor = 2
    with pytest.raises(ValueError, match="refit"):
        model.predict(windows[:1])


@pytest.mark.parametrize("kwargs", [{"backend": "typo"}, {"n_members": 0}, {"window": 0},
                                    {"cap": 0}, {"alpha": 1}, {"scale_floor": 0}])
def test_model_configuration_is_validated(kwargs):
    with pytest.raises(ValueError):
        RULChallenger(**kwargs)


def test_window_construction_is_causal_and_never_crosses_engine_boundaries():
    units = np.array([1, 1, 1, 2])
    cycles = np.array([3, 1, 2, 1])
    windows, _, ends = make_windows(units, cycles, [[30], [10], [20], [999]], window=3)
    np.testing.assert_array_equal(windows[1, :, 0], [10, 10, 20])
    assert 999 not in windows[:3]
    np.testing.assert_array_equal(ends, [1, 2, 3, 1])
    with pytest.raises(ValueError, match="duplicate cycle"):
        make_windows([1, 1], [1, 1], [[0], [1]], window=3)


def test_invalid_rows_targets_and_metric_broadcasts_are_rejected():
    with pytest.raises(ValueError):
        make_windows([1], [1], [[np.nan]], window=2)
    with pytest.raises(ValueError):
        make_windows([1], [1], [[1]], window=0)
    with pytest.raises(ValueError):
        truncated_samples([1], [1], [[1]], [2, 3])
    with pytest.raises(ValueError):
        rmse([1, 2], [[1], [2]])
    with pytest.raises(ValueError):
        nasa_score([1, 2], [1])


def test_single_cycle_window_has_zero_slope():
    from tsdiag.challengers.turbofan_rul import window_features
    features = window_features(np.array([[[2.0, 3.0]]]))
    np.testing.assert_array_equal(features[0, -2:], [0, 0])


def test_normalizer_drops_dead_sensors_and_never_refits_on_test_rows():
    settings = np.zeros((20, 3))
    sensors = np.column_stack([np.arange(20), np.ones(20)])
    normalizer = RegimeNormalizer().fit(settings, sensors)
    np.testing.assert_array_equal(normalizer.keep_, [True, False])
    mean, std = normalizer.mean_.copy(), normalizer.std_.copy()
    transformed = normalizer.transform(np.ones((2, 3)), [[100, 1], [200, 1]])
    assert transformed.shape == (2, 1) and transformed[0, 0] > 0
    np.testing.assert_array_equal(normalizer.mean_, mean)
    np.testing.assert_array_equal(normalizer.std_, std)
    with pytest.raises(ValueError, match="variable sensors"):
        RegimeNormalizer().fit(settings, np.ones((20, 2)))


def test_optional_cnn_forward_backward_on_cpu():
    torch = pytest.importorskip("torch")
    from tsdiag.challengers.turbofan_rul import _CNNMember
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            windows = np.random.default_rng(22).normal(size=(8, 6, 2))
            member = _CNNMember(seed=0, epochs=1, batch=4).fit(windows, np.arange(8.0))
            prediction = member.predict(windows[:2])
            assert prediction.shape == (2,) and np.isfinite(prediction).all()
    finally:
        torch.set_num_threads(previous_threads)
