import numpy as np
import pytest


torch = pytest.importorskip("torch")

from tsdiag.learned.ad_tfm_at import (
    ADTFMClassifier,
    ADTFMConfig,
    ADTFMAT,
    load_ad_tfm_classifier,
    phase_switch_augment,
    save_ad_tfm_classifier,
)
from tsdiag.domains.transformer_ad_tfm import ADTFMTransformerDiagnosticPipeline


def test_ad_tfm_forward_shapes_and_attention_normalization():
    config = ADTFMConfig(hidden_size=8, time_dimensions=2, frequency_dimensions=2, attention_size=6, num_classes=5)
    model = ADTFMAT(config)
    x = torch.randn(3, 20, 6)
    logits, attention = model(x, return_attention=True)
    assert logits.shape == (3, 5)
    assert attention.shape == (3, 20)
    assert torch.allclose(attention.sum(dim=1), torch.ones(3), atol=1e-5)
    assert torch.isfinite(logits).all()


def test_ad_tfm_checkpoint_round_trip(tmp_path):
    config = ADTFMConfig(
        hidden_size=8,
        time_dimensions=2,
        frequency_dimensions=2,
        attention_size=6,
        num_classes=5,
    )
    classifier = ADTFMClassifier(
        ADTFMAT(config),
        np.zeros(6),
        np.ones(6),
    )
    waveform = np.random.default_rng(3).normal(size=(100, 6))
    before = classifier.predict_batch(waveform)["probabilities"]

    checkpoint = save_ad_tfm_classifier(classifier, tmp_path / "model.pt")
    restored = load_ad_tfm_classifier(checkpoint)
    after = restored.predict_batch(waveform)["probabilities"]

    assert np.allclose(before, after)
    assert restored.class_names == classifier.class_names


def test_phase_switch_keeps_voltage_current_phase_pairs():
    # Each channel carries its channel index so the permutation is explicit.
    x = np.tile(np.arange(6, dtype=np.float32), (2, 4, 1))
    y = np.asarray([0, 3])
    aug_x, aug_y = phase_switch_augment(x, y, mode="paper")
    assert aug_x.shape == (6, 4, 6)
    assert aug_y.tolist() == [0, 3, 0, 3, 0, 3]
    # A<->B: Ua/Ub and Ia/Ib move together.
    assert aug_x[2, 0].tolist() == [1, 0, 2, 4, 3, 5]
    # A<->C: Ua/Uc and Ia/Ic move together.
    assert aug_x[4, 0].tolist() == [2, 1, 0, 5, 4, 3]


def test_raw_classifier_adapter_feeds_existing_pipeline():
    class DummyClassifier:
        def __call__(self, waveform):
            return {
                "label": "main_transformer_fault",
                "confidence": 0.9,
                "probabilities": {"main_transformer_fault": 0.9, "normal": 0.1},
            }

    rng = np.random.default_rng(0)
    signal = rng.normal(size=(100, 6))
    result = ADTFMTransformerDiagnosticPipeline(DummyClassifier()).run(
        signal,
        sampling_rate_hz=1.0,
        sensor_positions=["Ua", "Ub", "Uc", "Ia", "Ib", "Ic"],
        anomaly_threshold=0.0,
    )
    assert result.decision == "diagnose"
    assert result.hypotheses[0].label == "main_transformer_fault"
