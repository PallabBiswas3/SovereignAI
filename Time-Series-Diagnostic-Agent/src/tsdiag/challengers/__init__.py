"""Opt-in challengers; production plugins require explicit validated adapters."""

from .transformer_sequence import (
    SequenceFeatureClassifier, SequenceFeatureExtractor, augment_waveforms,
    oversample_minority, per_class_recall,
)
from .turbofan_rul import (
    RULChallenger, RegimeNormalizer, cap_rul, evaluate_rul, make_windows,
    nasa_score, rmse, truncated_samples,
)
from .battery_prognostics import (
    CellData, FadeParticleFilter, GPFadeModel, incremental_capacity,
    leave_one_cell_out, survival_consistency, time_to_voltage, weighted_quantiles,
)
from .bearing_cyclostationary import (
    angular_resample, bearing_cyclo_features, healthy_threshold,
    improved_envelope_spectrum, max_fault_score, order_envelope_spectrum,
    order_fault_scores, select_resonance_band,
)

__all__ = [
    "SequenceFeatureClassifier", "SequenceFeatureExtractor", "augment_waveforms",
    "oversample_minority", "per_class_recall", "RULChallenger", "RegimeNormalizer",
    "cap_rul", "evaluate_rul", "make_windows", "nasa_score", "rmse", "truncated_samples",
    "CellData", "FadeParticleFilter", "GPFadeModel", "incremental_capacity",
    "leave_one_cell_out", "survival_consistency", "time_to_voltage", "weighted_quantiles",
    "angular_resample", "bearing_cyclo_features", "healthy_threshold",
    "improved_envelope_spectrum", "max_fault_score", "order_envelope_spectrum",
    "order_fault_scores", "select_resonance_band",
]
