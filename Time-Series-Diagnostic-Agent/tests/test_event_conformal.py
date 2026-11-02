import json

import numpy as np
import pytest

from tsdiag.calibration import (
    DevelopmentSplitManifest,
    EventAlarmCalibration,
    EventFARCalibrator,
    EventResiduals,
    EventScorer,
    LeakageError,
    SplitManifest,
    SplitUnit,
    early_warning_lead_time,
    group_split,
    select_statistic,
    time_to_event_end,
)


def _event(rng, anomaly=0.0, n_train=1500, n_pred=1200, c=3):
    scale = rng.uniform(0.5, 3.0, size=c)           # per-event noise level
    train = rng.normal(size=(n_train, c)) * scale
    pred = rng.normal(size=(n_pred, c)) * scale * 1.1  # out-of-sample inflation
    if anomaly:
        pred[-300:] += anomaly * scale
    return EventResiduals(train=train, predict=pred)


def _pool(rng, n, prefix, anomaly=0.0):
    return {f"{prefix}{i}": _event(rng, anomaly) for i in range(n)}


def test_event_far_is_controlled_and_anomalies_detected():
    rng = np.random.default_rng(0)
    dev = _pool(rng, 240, "d")
    manifest = SplitManifest("event", tuple(dev), tuple(f"e{i}" for i in range(10))).validate()
    rest, val = group_split(list(dev), fraction=0.25, seed=1)
    cal = EventFARCalibrator(EventScorer("persistence", window=6), target_far=0.10)
    res = cal.fit(dev, manifest, validation_ids=val, n_boot=300)
    assert res.n_validation > 40
    assert res.validation_far_wilson95[0] <= 0.10 <= res.validation_far_wilson95[1] + 0.05

    fresh = _pool(rng, 300, "f")
    far = np.mean([cal.detect(e).alarm for e in fresh.values()])
    assert far <= 0.17                                   # target 0.10, tolerance ~4 sigma

    anomalous = _pool(rng, 100, "a", anomaly=4.0)
    recall = np.mean([cal.detect(e).alarm for e in anomalous.values()])
    assert recall >= 0.8


def test_first_alarm_index_and_lead_time():
    rng = np.random.default_rng(1)
    dev = _pool(rng, 120, "d")
    manifest = SplitManifest("event", tuple(dev), ("x",)).validate()
    cal = EventFARCalibrator(EventScorer("persistence", window=6), 0.1)
    cal.fit(dev, manifest, n_boot=0)
    d = cal.detect(_event(rng, anomaly=6.0))
    assert d.alarm and d.first_alarm_index == 905  # first full six-sample anomalous run
    assert early_warning_lead_time(d.first_alarm_index, onset=900) == -5
    assert time_to_event_end(d.first_alarm_index, 1200) == 295
    assert early_warning_lead_time(100, 900) == 800
    assert time_to_event_end(100, 1200) == 1100
    assert early_warning_lead_time(None, 10) is None


def test_unscorable_event_abstains():
    rng = np.random.default_rng(2)
    dev = _pool(rng, 120, "d")
    manifest = SplitManifest("event", tuple(dev), ("x",)).validate()
    cal = EventFARCalibrator(EventScorer(), 0.1)
    cal.fit(dev, manifest, n_boot=0)
    bad = EventResiduals(train=np.zeros((500, 3)), predict=np.ones((500, 3)))
    d = cal.detect(bad)
    assert d.alarm is None and d.status == "abstain_unscorable"
    nan = EventResiduals(train=_event(rng).train, predict=np.full((500, 3), np.nan))
    assert cal.detect(nan).alarm is None


def test_too_few_events_raises():
    rng = np.random.default_rng(3)
    dev = _pool(rng, 5, "d")
    manifest = SplitManifest("event", tuple(dev), ("x",)).validate()
    with pytest.raises(ValueError, match="need at least"):
        EventFARCalibrator(EventScorer(), 0.1).fit(dev, manifest)


def test_calibrator_rejects_events_outside_manifest():
    rng = np.random.default_rng(4)
    dev = _pool(rng, 30, "d")
    manifest = SplitManifest("event", tuple(list(dev)[:-1]), ("x",)).validate()
    with pytest.raises(LeakageError):
        EventFARCalibrator(EventScorer(), 0.1).fit(dev, manifest)


def test_manifest_boundaries():
    assets = {"d1": "A", "d2": "B", "e1": "A", "e2": "C"}
    with pytest.raises(LeakageError):
        SplitManifest("event", ("d1", "d2"), ("d1",)).validate()
    with pytest.raises(LeakageError):
        SplitManifest("asset", ("d1", "d2"), ("e1", "e2"), asset_of=assets).validate()
    with pytest.raises(LeakageError):
        SplitManifest("asset", ("d1",), ("e2",), asset_of={"d1": "A"}).validate()
    SplitManifest("asset", ("d1", "d2"), ("e2",), asset_of=assets).validate()

    iv_ok = {"d1": (0, 10), "d2": (0, 5), "e1": (11, 20), "e2": (0, 5)}
    iv_bad = {"d1": (0, 10), "d2": (0, 5), "e1": (9, 20), "e2": (0, 5)}
    SplitManifest("temporal", ("d1", "d2"), ("e1", "e2"), assets, iv_ok).validate()
    with pytest.raises(LeakageError):
        SplitManifest("temporal", ("d1", "d2"), ("e1", "e2"), assets, iv_bad).validate()


def test_checksum_is_order_independent_and_sensitive():
    a = SplitManifest("event", ("a", "b"), ("c",))
    b = SplitManifest("event", ("b", "a"), ("c",))
    c = SplitManifest("event", ("a", "b"), ("d",))
    assert a.checksum() == b.checksum() != c.checksum()


def test_group_split_keeps_assets_together():
    ids = [f"e{i}" for i in range(40)]
    groups = {i: f"asset{int(i[1:]) % 8}" for i in ids}
    rest, held = group_split(ids, groups, fraction=0.25, seed=0)
    assert {groups[i] for i in rest}.isdisjoint({groups[i] for i in held})


def test_select_statistic_returns_best_dev_recall():
    rng = np.random.default_rng(5)
    healthy = _pool(rng, 150, "h")
    anomalous = _pool(rng, 40, "a", anomaly=4.0)
    manifest = SplitManifest("event", (*healthy, *anomalous), ("x",)).validate()
    cands = [EventScorer("persistence", 6), EventScorer("max_smoothed", 6), EventScorer("cusum")]
    best, report = select_statistic(cands, healthy, anomalous, manifest, 0.1)
    assert best in cands and len(report) == 3


def _small_pool(prefix="d", n=30):
    rng = np.random.default_rng(42)
    return {f"{prefix}{i}": _event(rng, n_train=200, n_pred=120, c=2) for i in range(n)}


def test_existing_care_manifest_is_used_without_another_checksum_schema():
    from tsdiag.evaluation.wind_calibration import build_wind_split_manifest

    dev = _small_pool()
    manifest = build_wind_split_manifest(
        [{"event_id": i, "wind_farm": "farm", "asset_ids": [i]} for i in range(30)],
        [{"event_id": 100, "wind_farm": "farm", "asset_ids": [100]}],
        dataset_id="synthetic", development_pool_id="dev", evaluation_pool_id="frozen",
        leakage_boundary="asset")
    events = {str(i): event for i, event in enumerate(dev.values())}
    result = EventFARCalibrator(EventScorer(window=3)).fit(events, manifest, n_boot=0)
    assert result.manifest_sha256 == manifest.sha256()
    compact = SplitManifest("asset", tuple(events), ("100",),
                            {i: f"asset-{i}" for i in (*events, "100")})
    assert compact.checksum() == compact.to_manifest().sha256()


def test_freezing_preserves_missing_policy_and_detection_configuration():
    dev = _small_pool()
    manifest = SplitManifest("event", tuple(dev), ("x",))
    calibrator = EventFARCalibrator(EventScorer(window=3, max_missing_fraction=0.05))
    artifact = calibrator.fit(dev, manifest, validation_ids=["d28", "d29"], n_boot=0)
    assert artifact.validation_far_bootstrap95 is None
    frozen = EventAlarmCalibration.from_json(artifact.to_json())
    assert frozen == artifact and frozen.config_sha256() == artifact.config_sha256()
    restored = EventFARCalibrator.from_calibration(frozen)
    example = dev["d0"]
    original = restored.detect(example)
    restored.scorer = EventScorer("cusum", window=10)  # cannot change a frozen decision
    assert restored.detect(example) == original
    missing = example.predict.copy()
    missing[:12] = np.nan
    assert restored.detect(EventResiduals(example.train, missing)).alarm is None


@pytest.mark.parametrize("mutation", ["missing_policy", "wrong_version", "overlap", "bad_count"])
def test_frozen_artifact_requires_complete_consistent_configuration(mutation):
    dev = _small_pool()
    result = EventFARCalibrator(EventScorer(window=3)).fit(
        dev, SplitManifest("event", tuple(dev), ("x",)), validation_ids=["d29"], n_boot=0)
    payload = json.loads(result.to_json())
    if mutation == "missing_policy":
        del payload["max_missing_fraction"]
    elif mutation == "wrong_version":
        payload["code_version"] = "future-version"
    elif mutation == "overlap":
        payload["validation_ids"] = ["d0"]
    else:
        payload["n_calibration"] += 1
    with pytest.raises(ValueError):
        EventAlarmCalibration.from_json(json.dumps(payload))


@pytest.mark.parametrize("statistic", ["persistence", "max_smoothed", "cusum"])
def test_missing_rows_cannot_supply_alarm_evidence(statistic):
    rng = np.random.default_rng(9)
    event = _event(rng, n_train=200, n_pred=60, c=2)
    scorer = EventScorer(statistic, window=3, max_missing_fraction=1)
    assert scorer.trace(EventResiduals(event.train, np.full_like(event.predict, np.nan))) is None
    partial = event.predict.copy()
    partial[::3] = np.nan
    trace = scorer.trace(EventResiduals(event.train, partial))
    if statistic in {"persistence", "max_smoothed"}:
        assert trace is None  # every three-row window contains a missing row
    else:
        assert np.isneginf(trace[0][::3]).all()


@pytest.mark.parametrize("kwargs", [{"window": 0}, {"window": 1.5}, {"fuse": "typo"},
                                    {"statistic": "typo"}, {"cusum_drift": -1},
                                    {"max_missing_fraction": 1.1}])
def test_scorer_rejects_invalid_configuration(kwargs):
    with pytest.raises(ValueError):
        EventScorer(**kwargs)


def test_duplicate_validation_and_asset_leakage_are_rejected():
    dev = _small_pool()
    calibrator = EventFARCalibrator(EventScorer(window=3))
    event_manifest = SplitManifest("event", tuple(dev), ("x",))
    with pytest.raises(LeakageError, match="duplicate"):
        calibrator.fit(dev, event_manifest, validation_ids=["d0", "d0"])
    assets = {i: "same-asset" for i in dev}
    assets["x"] = "different-asset"
    asset_manifest = SplitManifest("asset", tuple(dev), ("x",), assets)
    with pytest.raises(LeakageError, match="share asset"):
        calibrator.fit(dev, asset_manifest, validation_ids=["d0"])


def test_temporal_validation_must_follow_calibration_and_require_complete_intervals():
    dev = _small_pool(n=10)
    assets = {i: "A" for i in (*dev, "x")}
    intervals = {f"d{i}": (i * 10, i * 10 + 5) for i in range(10)}
    intervals["x"] = (1000, 1005)
    manifest = SplitManifest("temporal", tuple(dev), ("x",), assets, intervals)
    calibrator = EventFARCalibrator(EventScorer(window=3))
    calibrator.fit(dev, manifest, validation_ids=["d9"], n_boot=0)
    with pytest.raises(LeakageError, match="not strictly earlier"):
        calibrator.fit(dev, manifest, validation_ids=["d0"], n_boot=0)
    # Missing/reversed intervals on non-shared assets are invalid too.
    assets["x"] = "B"
    with pytest.raises(LeakageError, match="timestamps"):
        SplitManifest("temporal", tuple(dev), ("x",), assets, {}).validate()
    intervals["x"] = (1005, 1000)
    with pytest.raises(LeakageError, match="before it starts"):
        SplitManifest("temporal", tuple(dev), ("x",), assets, intervals).validate()


def test_statistic_selection_protects_frozen_ids_and_final_calibration_pool():
    healthy = _small_pool("h")
    anomalous = _small_pool("a", n=5)
    candidates = [EventScorer(window=3)]
    with pytest.raises(LeakageError):
        select_statistic(candidates, healthy, {"frozen": next(iter(anomalous.values()))},
                         SplitManifest("event", tuple(healthy), ("frozen",)))
    tuning_manifest = SplitManifest("event", (*healthy, *anomalous), ("frozen",))
    chosen, _ = select_statistic(candidates, healthy, anomalous, tuning_manifest)
    with pytest.raises(LeakageError, match="overlap"):
        EventFARCalibrator(chosen).fit(healthy, SplitManifest("event", tuple(healthy), ("frozen",)))
    fresh = _small_pool("c")
    final = EventFARCalibrator(chosen).fit(fresh, SplitManifest("event", tuple(fresh), ("frozen",)), n_boot=0)
    assert final.selection_manifest_sha256 == tuning_manifest.checksum()


def test_validation_bootstrap_and_wilson_are_finite_and_exclusions_are_counted():
    dev = _small_pool()
    first = next(iter(dev.values()))
    dev["d0"] = EventResiduals(first.train, np.full_like(first.predict, np.nan))
    dev["d29"] = EventResiduals(first.train, np.full_like(first.predict, np.nan))
    result = EventFARCalibrator(EventScorer(window=3)).fit(
        dev, SplitManifest("event", tuple(dev), ("x",)), validation_ids=["d27", "d28", "d29"], n_boot=20)
    assert result.n_excluded == result.n_validation_excluded == 1
    assert result.n_calibration == 26 and result.n_validation == 2
    assert np.isfinite(result.validation_far_wilson95).all()
    lo, hi = result.validation_far_bootstrap95
    assert 0 <= lo <= hi <= 1


def test_failed_refit_does_not_retain_a_previous_threshold():
    dev = _small_pool()
    manifest = SplitManifest("event", tuple(dev), ("x",))
    calibrator = EventFARCalibrator(EventScorer(window=3))
    calibrator.fit(dev, manifest, n_boot=0)
    with pytest.raises(LeakageError):
        calibrator.fit({}, manifest)
    with pytest.raises(RuntimeError, match="fit"):
        calibrator.detect(dev["d0"])


def test_group_split_rejects_duplicates_missing_assets_and_bad_fraction():
    with pytest.raises(LeakageError):
        group_split(["a", "a"])
    with pytest.raises(LeakageError):
        group_split(["a", "b"], {"a": "asset"})
    with pytest.raises(ValueError):
        group_split(["a", "b"], fraction=1)
