from __future__ import annotations

from ..contracts import ToolRegistry
from ..detectors.residual_changepoint import residual_cusum
from ..domains.wind_physics import physics_consistency_check
from .change import change_point_detection, cross_sensor_relationships, rolling_statistics
from .monitoring import (
    arbitrate_dpca_cva,
    calibrated_pca_monitoring,
    calibrate_monitoring_config,
    cva_monitoring,
    dpca_monitoring,
    fit_cva,
    make_lagged_matrix,
    run_monitoring_method,
    transform_cva,
)
from .multivariate import (
    contribution_analysis,
    data_quality_check,
    detect_operating_regimes,
    granger_causality,
    pca_monitoring,
    regime_normalization,
    residual_analysis,
    robust_anomaly_detection,
    standardize_against_normal,
    stationarity_analysis,
)
from .process import causal_graph_filter, fault_onset_timing, process_diagnosis, root_cause_rank
from .process_enhanced import pre_post_shift_evidence, root_cause_rank_enhanced
from .signal import (
    bandpass_filter,
    envelope_spectrum,
    hilbert_envelope,
    signal_integrity,
    spectral_kurtosis,
    time_domain_features,
    time_frequency_analysis,
    welch_psd,
)
from .wavelet import (
    correlation_sensor_weighting,
    cross_correlation_analysis,
    multisensor_fusion,
    wavelet_denoising,
)
from .wind_scada import (
    detect_wind_operating_regimes,
    normal_behavior_model,
    scada_quality_check,
    wind_residual_anomaly_detection,
)


def default_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()
    tools = {
        "signal_integrity": signal_integrity,
        "process_data_quality": data_quality_check,
        "scada_quality_check": scada_quality_check,
        "time_domain_features": time_domain_features,
        "welch_psd": welch_psd,
        "time_frequency_analysis": time_frequency_analysis,
        "spectral_kurtosis": spectral_kurtosis,
        "bandpass_filter": bandpass_filter,
        "hilbert_envelope": hilbert_envelope,
        "envelope_spectrum": envelope_spectrum,
        "standardize_against_normal": standardize_against_normal,
        "pca_monitoring": pca_monitoring,
        "calibrated_pca_monitoring": calibrated_pca_monitoring,
        "dpca_monitoring": dpca_monitoring,
        "cva_monitoring": cva_monitoring,
        "fit_cva": fit_cva,
        "transform_cva": transform_cva,
        "make_lagged_matrix": make_lagged_matrix,
        "calibrate_monitoring_config": calibrate_monitoring_config,
        "run_monitoring_method": run_monitoring_method,
        "arbitrate_dpca_cva": arbitrate_dpca_cva,
        "contribution_analysis": contribution_analysis,
        "stationarity_analysis": stationarity_analysis,
        "granger_causality": granger_causality,
        "causal_graph_filter": causal_graph_filter,
        "fault_onset_timing": fault_onset_timing,
        "root_cause_rank": root_cause_rank,
        "root_cause_rank_enhanced": root_cause_rank_enhanced,
        "pre_post_shift_evidence": pre_post_shift_evidence,
        "process_diagnosis": process_diagnosis,
        "operating_regime_detection": detect_wind_operating_regimes,
        "operating_condition_identification": detect_operating_regimes,
        "normal_behavior_model": normal_behavior_model,
        "regime_normalization": regime_normalization,
        "residual_analysis": residual_analysis,
        "scada_anomaly_detection": wind_residual_anomaly_detection,
        "rolling_statistics": rolling_statistics,
        "cross_sensor_relationships": cross_sensor_relationships,
        "change_point_detection": change_point_detection,
        "residual_change_point_detection": residual_cusum,
        "physics_consistency_check": physics_consistency_check,
        "wavelet_denoising": wavelet_denoising,
        "cross_correlation_analysis": cross_correlation_analysis,
        "correlation_sensor_weighting": correlation_sensor_weighting,
        "multisensor_fusion": multisensor_fusion,
    }
    for name, fn in tools.items():
        registry.register(name, fn)
    return registry
