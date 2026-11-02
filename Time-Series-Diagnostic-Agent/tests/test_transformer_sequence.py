import numpy as np
import pytest
from sklearn.metrics import balanced_accuracy_score, f1_score

from tsdiag.challengers.transformer_sequence import (
    SequenceFeatureClassifier,
    SequenceFeatureExtractor,
    augment_waveforms,
    oversample_minority,
    per_class_recall,
)
from tsdiag.uncertainty import ConformalClassifier

T, CYCLES = 100, 2          # 100 samples spanning 2 fundamental cycles
NORMAL, IPSF, GROUND, INRUSH = 0, 1, 2, 3


def _wave(rng, cls, amp_v=1.0, amp_i=0.5, noise=0.02):
    t = np.arange(T) / T * 2 * np.pi * CYCLES + rng.uniform(0, 2 * np.pi)
    ph = np.array([0, -2 * np.pi / 3, 2 * np.pi / 3])
    V = amp_v * np.cos(t[:, None] + ph[None])
    I = amp_i * rng.uniform(0.8, 1.2) * np.cos(t[:, None] + ph[None] - 0.3)
    if cls == IPSF:            # phase B-C short: B and C currents opposed and large, voltage sags
        I = I.copy()
        I[:, 1] = 3.0 * np.cos(t + ph[1] - 0.3)
        I[:, 2] = -3.0 * np.cos(t + ph[1] - 0.3)
        V = V * np.array([1.0, 0.7, 0.7])
    elif cls == GROUND:        # phase A to ground: zero-sequence current, phase A voltage collapses
        I = I.copy()
        I[:, 0] = 3.0 * np.cos(t + ph[0] - 0.3)
        V = V * np.array([0.3, 1.0, 1.0])
    elif cls == INRUSH:        # magnetising inrush: strong 2nd harmonic, mostly one-sided
        I = I + 1.5 * (np.cos(2 * t[:, None] + ph[None] * 2) + 0.5 * np.abs(np.cos(t[:, None] + ph[None])))
    X = np.hstack([V, I])
    return X + rng.normal(scale=noise, size=X.shape)


def _dataset(rng, counts):
    X, y = [], []
    for cls, n in counts.items():
        for _ in range(n):
            X.append(_wave(rng, cls))
            y.append(cls)
    return np.stack(X), np.array(y)


def test_balanced_waveform_has_only_positive_sequence():
    rng = np.random.default_rng(0)
    ext = SequenceFeatureExtractor(cycles_in_window=CYCLES)
    F = ext.transform(_wave(rng, NORMAL)[None])
    f = dict(zip(ext.feature_names_, F[0]))
    assert f["V_neg_over_pos"] < 0.05 and f["V_zero_over_pos"] < 0.05
    assert f["I_neg_over_pos"] < 0.05 and f["I_zero_over_pos"] < 0.05


def test_fault_signatures_in_features():
    rng = np.random.default_rng(1)
    ext = SequenceFeatureExtractor(cycles_in_window=CYCLES)

    def feat(cls):
        values = ext.transform(_wave(rng, cls)[None])[0]
        return dict(zip(ext.feature_names_, values))

    n, ip, g, ir = feat(NORMAL), feat(IPSF), feat(GROUND), feat(INRUSH)
    assert ip["I_neg_over_pos"] > 0.5 > n["I_neg_over_pos"]
    assert g["I_zero_over_pos"] > 0.5 > n["I_zero_over_pos"]
    assert ir["I_h2_share"] > 3 * n["I_h2_share"]
    # phase-to-phase fault: little zero sequence compared with a ground fault
    assert ip["I_zero_over_pos"] < g["I_zero_over_pos"]


def test_dominant_bin_is_estimated_when_unspecified():
    rng = np.random.default_rng(2)
    X = np.stack([_wave(rng, NORMAL) for _ in range(5)])
    a = SequenceFeatureExtractor().transform(X)
    b = SequenceFeatureExtractor(cycles_in_window=CYCLES).transform(X)
    assert np.allclose(a, b, atol=1e-6)


def test_input_validation():
    with pytest.raises(ValueError):
        SequenceFeatureExtractor().transform(np.zeros((3, 100, 5)))


def test_augmentation_and_oversampling():
    rng = np.random.default_rng(3)
    X, y = _dataset(rng, {NORMAL: 60, IPSF: 5})
    Xa = augment_waveforms(X[:4], rng)
    assert Xa.shape == X[:4].shape and not np.allclose(Xa, X[:4])
    Xo, yo = oversample_minority(X, y, 40, rng)
    assert (yo == IPSF).sum() == 40 and (yo == NORMAL).sum() == 60


def test_imbalanced_ipsf_is_recovered_with_features():
    rng = np.random.default_rng(4)
    Xtr, ytr = _dataset(rng, {NORMAL: 400, GROUND: 60, INRUSH: 60, IPSF: 12})   # IPSF is rare
    Xte, yte = _dataset(rng, {NORMAL: 200, GROUND: 40, INRUSH: 40, IPSF: 30})
    clf = SequenceFeatureClassifier(SequenceFeatureExtractor(CYCLES), min_per_class=150).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    rec = per_class_recall(yte, pred)
    assert rec[IPSF]["recall"] >= 0.9
    assert rec[IPSF]["wilson95"][0] > 0.7
    assert f1_score(yte, pred, average="macro") >= 0.9
    assert balanced_accuracy_score(yte, pred) >= 0.9


def test_conformal_abstention_on_top_of_sequence_classifier():
    rng = np.random.default_rng(5)
    Xtr, ytr = _dataset(rng, {NORMAL: 300, GROUND: 80, INRUSH: 80, IPSF: 40})
    Xcal, ycal = _dataset(rng, {NORMAL: 150, GROUND: 60, INRUSH: 60, IPSF: 40})
    Xte, yte = _dataset(rng, {NORMAL: 150, GROUND: 60, INRUSH: 60, IPSF: 40})
    clf = SequenceFeatureClassifier(SequenceFeatureExtractor(CYCLES), min_per_class=150).fit(Xtr, ytr)
    cc = ConformalClassifier(0.1, "lac", class_conditional=True).fit(clf.predict_proba(Xcal), ycal)
    sets = cc.predict_sets(clf.predict_proba(Xte))
    assert np.mean(sets[np.arange(len(yte)), yte]) >= 0.85


def test_stacking_extra_probabilities_disallows_augmentation():
    rng = np.random.default_rng(6)
    X, y = _dataset(rng, {NORMAL: 20, IPSF: 20})
    clf = SequenceFeatureClassifier(SequenceFeatureExtractor(CYCLES), augment=True)
    with pytest.raises(ValueError):
        clf.fit(X, y, extra=np.zeros((40, 2)))
    extra = np.tile([0.4, 0.6], (len(X), 1))  # feature-shape check, not label-derived probabilities
    clf = SequenceFeatureClassifier(SequenceFeatureExtractor(CYCLES), augment=False).fit(X, y, extra=extra)
    assert clf.predict(X, extra=extra).shape == (40,)
    with pytest.raises(ValueError, match="extra"):
        clf.predict(X)
    with pytest.raises(ValueError, match="extra"):
        clf.predict(X, extra=np.ones((len(X), 3)) / 3)


def test_auto_frequency_features_are_independent_of_inference_batch():
    rng = np.random.default_rng(10)
    slow = _wave(rng, NORMAL, noise=0)
    fast = np.tile(slow[::4], (4, 1)) * 20  # eight cycles instead of two
    extractor = SequenceFeatureExtractor()
    alone = extractor.transform(slow)
    together = extractor.transform(np.stack([slow, fast]))
    np.testing.assert_allclose(together[0], alone[0], atol=1e-9)
    np.testing.assert_allclose(together[1], extractor.transform(fast)[0], atol=1e-9)


def test_feature_schema_is_fixed_across_frequencies_and_harmonic_observability():
    rng = np.random.default_rng(11)
    sample = _wave(rng, NORMAL)
    low = SequenceFeatureExtractor(2)
    high = SequenceFeatureExtractor(20)
    a, b = low.transform(sample), high.transform(sample)
    assert low.feature_names_ == high.feature_names_ and a.shape == b.shape
    high_features = dict(zip(high.feature_names_, b[0]))
    assert high_features["V_h3_share"] == high_features["V_h3_observable"] == 0
    assert high_features["V_h2_observable"] == 1


def test_noninteger_cycle_metadata_is_not_rounded_and_zero_sequence_is_true_rms():
    phase = 2 * np.pi * 2.5 * np.arange(T) / T
    offsets = np.array([0, -2 * np.pi / 3, 2 * np.pi / 3])
    v = np.cos(phase[:, None] + offsets)
    i = 0.5 * np.cos(phase[:, None] + offsets - 0.3)
    extractor = SequenceFeatureExtractor(2.5)
    result = extractor.transform(np.column_stack([v, i]))[0]
    feature = dict(zip(extractor.feature_names_, result))
    assert feature["V_neg_over_pos"] < 1e-10
    assert feature["V_zero_over_pos"] < 1e-10
    common = np.column_stack([np.cos(phase)] * 3)
    sample = np.column_stack([v, common])
    feature = dict(zip(extractor.feature_names_, extractor.transform(sample)[0]))
    assert feature["I_td_zero_rms"] == pytest.approx(1.0, abs=1e-8)


@pytest.mark.parametrize("sample", [np.zeros((1, 3, 6)), np.zeros((0, 100, 6)),
                                    np.full((1, 100, 6), np.nan)])
def test_invalid_waveforms_are_rejected(sample):
    with pytest.raises(ValueError):
        SequenceFeatureExtractor(2).transform(sample)


@pytest.mark.parametrize("cycles", [0, -1, np.nan, 50, 60])
def test_invalid_cycle_metadata_is_rejected(cycles):
    with pytest.raises(ValueError):
        SequenceFeatureExtractor(cycles).transform(np.zeros((1, T, 6)))


def test_constant_voltage_frequency_estimate_and_unsafe_shift_are_rejected():
    with pytest.raises(ValueError, match="frequency"):
        SequenceFeatureExtractor().transform(np.zeros((1, T, 6)))
    with pytest.raises(ValueError, match="whole-cycle"):
        augment_waveforms(np.zeros((1, T, 6)), np.random.default_rng(0), max_shift=3)


def test_training_schema_and_extractor_configuration_are_retained():
    rng = np.random.default_rng(12)
    X, y = _dataset(rng, {NORMAL: 25, IPSF: 25})
    extractor = SequenceFeatureExtractor(2)
    model = SequenceFeatureClassifier(extractor, augment=False).fit(X, y)
    expected = model.predict_proba(X)
    extractor.cycles_in_window = 20
    np.testing.assert_allclose(model.predict_proba(X), expected)
    with pytest.raises(ValueError):
        model.fit(X, y.astype(float) + 0.5)
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(X)


def test_recall_rejects_misalignment_and_counts_abstentions_as_misses():
    with pytest.raises(ValueError):
        per_class_recall([0, 1], [0])
    report = per_class_recall([0, 1, 1], [0, 1, -1])
    assert report[1]["recall"] == 0.5 and report[1]["n"] == 2
