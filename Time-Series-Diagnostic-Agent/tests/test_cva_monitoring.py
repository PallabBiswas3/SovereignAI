import numpy as np

from tsdiag.tools import (
    MonitoringConfig,
    CVAFaultClassifier,
    arbitrate_dpca_cva,
    calibrate_monitoring_config,
    cva_monitoring,
    fit_cva_fault_classifier,
    run_monitoring_method,
)
from tsdiag.domains.process_cva import CVATEPDiagnosticPipeline


def _var_process(seed: int, samples: int = 240, channels: int = 4) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros((samples, channels), dtype=float)
    noise = rng.normal(scale=0.35, size=x.shape)
    for t in range(1, samples):
        x[t] = 0.72 * x[t - 1] + noise[t]
    return x


def test_cva_monitoring_detects_dynamic_shift_and_preserves_sample_alignment():
    reference = _var_process(1)
    current = _var_process(2)
    current[120:, 0] += np.linspace(0.0, 4.0, len(current) - 120)
    result = cva_monitoring(
        current,
        reference,
        past_lags=2,
        future_lags=2,
        variance_target=0.95,
        alpha=0.99,
        min_consecutive=2,
    )
    assert result["alarm_mask"].shape == (len(current),)
    assert result["variable_contributions"].shape == current.shape
    assert result["cva_state"]["n_components"] >= 1
    assert np.mean(result["alarm_mask"][160:]) > np.mean(result["alarm_mask"][:100])


def test_cva_online_scores_do_not_depend_on_future_current_samples():
    reference = _var_process(3)
    prefix = _var_process(4, samples=100)
    extended = np.vstack([prefix, np.full((30, prefix.shape[1]), 1000.0)])
    short = cva_monitoring(prefix, reference, past_lags=3, future_lags=3)
    long = cva_monitoring(extended, reference, past_lags=3, future_lags=3)
    np.testing.assert_allclose(short["t2"][2:], long["t2"][2:100], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(short["spe"][2:], long["spe"][2:100], rtol=1e-10, atol=1e-10)


def test_cva_is_supported_by_healthy_only_calibration_dispatch():
    reference = _var_process(5)
    calibrated = calibrate_monitoring_config(
        reference,
        method="cva",
        alpha_grid=(0.99,),
        persistence_grid=(1, 2),
        lags_grid=(2,),
    )
    config = calibrated["config"]
    assert config.method == "cva"
    result = run_monitoring_method(reference, reference, config)
    assert result["calibration"]["method"] == "cva"


def test_cva_fda_uses_dynamic_features_and_predicts_event_classes():
    normal = _var_process(10, samples=300)
    class_one = _var_process(11, samples=260)
    class_two = _var_process(12, samples=260)
    class_one[:, 0] += 3.0
    class_two[:, 1] -= 3.0
    classifier = fit_cva_fault_classifier(
        normal,
        {0: normal, 1: class_one, 2: class_two},
        method="fda",
        past_lags=2,
        future_lags=2,
        variance_target=0.999,
        max_samples_per_class=200,
    )
    assert classifier.method == "cva_fda"
    assert classifier.predict_event(class_one)["predicted_fault_id"] == 1
    assert classifier.predict_event(class_two)["predicted_fault_id"] == 2


def test_monitoring_config_remains_backwards_compatible():
    config = MonitoringConfig("pca", 0.99, 1)
    assert config.lags == 0


def test_cva_pipeline_passes_detection_and_contributions_to_root_layer():
    reference = _var_process(20)
    current = _var_process(21)
    current[80:, 0] += 5.0

    class Classifier:
        def predict_event(self, signal_matrix, *, start_index=0):
            return {"predicted_fault_id": 4, "confidence": 0.9}

    class RootLayer:
        def __init__(self):
            self.kwargs = None

        def run(self, *args, **kwargs):
            self.kwargs = kwargs
            return "causal-result"

    root = RootLayer()
    pipeline = CVATEPDiagnosticPipeline(
        reference,
        Classifier(),
        MonitoringConfig("cva", 0.99, 1, lags=2),
        minimum_alarm_fraction=0.01,
        root_cause_pipeline=root,
    )
    result = pipeline.run(current, [f"x{i}" for i in range(current.shape[1])], fault_start_index=80)
    assert result.fault_detected
    assert result.predicted_fault_id == 4
    assert result.root_cause == "causal-result"
    assert root.kwargs["detection_override"]["method"] == "dpca_cva_hybrid"
    assert root.kwargs["detection_override"]["variable_contributions"].shape == current.shape


def test_dpca_cva_arbitration_distinguishes_warning_and_confirmation():
    def result(fraction):
        mask = np.zeros(100, dtype=bool)
        mask[:round(100 * fraction)] = True
        return {"alarm_mask": mask}

    warning = arbitrate_dpca_cva(result(0.10), result(0.01))
    assert warning["status"] == "early_warning"
    assert not warning["fault_detected"]
    confirmed = arbitrate_dpca_cva(result(0.08), result(0.06))
    assert confirmed["status"] == "confirmed_fault"
    assert confirmed["reason"] == "dpca_cva_agreement"


def test_cva_classifier_persistence_and_ambiguous_abstention(tmp_path):
    normal = _var_process(30, samples=260)
    shifted = _var_process(31, samples=240)
    shifted[:, 0] += 3.0
    classifier = fit_cva_fault_classifier(
        normal, {0: normal, 3: shifted}, method="fda", past_lags=2,
        future_lags=2, max_samples_per_class=180,
    )
    artifact = classifier.save(tmp_path / "model.joblib")
    loaded = CVAFaultClassifier.load(artifact)
    original = classifier.predict_event(shifted, minimum_confidence=0.0, minimum_margin=0.0,
                                        weak_fault_minimum_confidence=0.0, weak_fault_minimum_margin=0.0)
    restored = loaded.predict_event(shifted, minimum_confidence=0.0, minimum_margin=0.0,
                                    weak_fault_minimum_confidence=0.0, weak_fault_minimum_margin=0.0)
    assert restored["candidate_fault_id"] == original["candidate_fault_id"]
    abstained = loaded.predict_event(shifted, weak_fault_minimum_confidence=1.01)
    assert abstained["candidate_fault_id"] == 3
    assert abstained["abstained"]
    assert abstained["predicted_fault_id"] is None
