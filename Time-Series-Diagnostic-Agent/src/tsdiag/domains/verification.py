from __future__ import annotations

from typing import Any

import numpy as np

from ..models import VerificationResult


def battery_physics_verification(state: dict[str, Any], evidence_id: str) -> VerificationResult:
    voltage = np.asarray(state["cell_voltage"], dtype=float)
    temperature = np.asarray(state["cell_temperature"], dtype=float)
    top_cell = str(state["top_cell"])
    top_idx = list(state["cell_ids"]).index(top_cell)

    voltage_ok = bool(np.all((voltage >= 1.5) & (voltage <= 5.5)))
    temperature_ok = bool(np.all((temperature >= -50.0) & (temperature <= 120.0)))

    dv = np.median(np.abs(state["delta_voltage"]), axis=0)
    dt = np.median(np.abs(state["delta_temperature"]), axis=0)
    dominant_idx = int(np.argmax(np.maximum(
        dv / max(float(np.max(dv)), 1e-12),
        dt / max(float(np.max(dt)), 1e-12),
    )))
    localization_consistent = dominant_idx == top_idx

    if not voltage_ok or not temperature_ok:
        status = "CONTRADICTED"
        reason = "Telemetry falls outside broad physically plausible cell voltage/temperature bounds."
    elif state["abnormal"] and localization_consistent:
        status = "SUPPORTED"
        reason = "Localized cell imbalance is consistent with independent voltage/temperature deviation evidence."
    elif state["abnormal"]:
        status = "INSUFFICIENT"
        reason = "An anomaly is present, but independent voltage/temperature deviation does not clearly corroborate the localized cell."
    else:
        status = "INSUFFICIENT"
        reason = "No abnormal battery condition was declared; physics checks found no contradiction."

    return VerificationResult(
        hypothesis="cell_imbalance",
        status=status,
        reason=reason,
        evidence_ids=[evidence_id],
        verifier="battery_physics_consistency",
        details={
            "voltage_bounds_ok": voltage_ok,
            "temperature_bounds_ok": temperature_ok,
            "localized_cell": top_cell,
            "independent_dominant_cell": str(state["cell_ids"][dominant_idx]),
            "localization_consistent": localization_consistent,
        },
    )


def turbofan_physics_verification(state: dict[str, Any], evidence_id: str) -> VerificationResult:
    cycles = np.asarray(state["cycle_index"], dtype=float)
    health = np.asarray(state["health_index"], dtype=float)
    slope = float(state["health_slope"])
    rul = state["rul_cycles"]

    health_cycle_corr = 0.0
    if len(health) > 2 and np.std(health) > 1e-12:
        health_cycle_corr = float(np.corrcoef(cycles, health)[0, 1])

    rul_consistent = rul is None or float(rul) >= 0.0
    degrading_trend = slope > 0.0 and health_cycle_corr > 0.25

    if not rul_consistent:
        status = "CONTRADICTED"
        reason = "RUL estimate is physically inconsistent because it is negative."
    elif state["abnormal"] and degrading_trend:
        status = "SUPPORTED"
        reason = "Declared degradation is corroborated by a positive health-index trend over increasing engine cycles."
    elif state["abnormal"]:
        status = "INSUFFICIENT"
        reason = "A degraded state was declared, but the independent health-index trend is not sufficiently monotonic."
    else:
        status = "INSUFFICIENT"
        reason = "No degraded state was declared; trend and RUL checks found no contradiction."

    return VerificationResult(
        hypothesis="degradation",
        status=status,
        reason=reason,
        evidence_ids=[evidence_id],
        verifier="turbofan_degradation_consistency",
        details={
            "health_slope": slope,
            "health_cycle_correlation": health_cycle_corr,
            "rul_nonnegative_or_unavailable": rul_consistent,
            "rul_cycles": rul,
        },
    )


def transformer_physics_verification(state: dict[str, Any], evidence_id: str) -> VerificationResult:
    anomaly_score = float(state["harmonic_structure"]["anomaly_score"])
    kurtosis = float(state["harmonic_structure"]["kurtosis"])
    weights = np.asarray(state["sensor_weights"], dtype=float)
    channel_std = np.asarray(state["channel_statistics"]["std"], dtype=float)
    label = state.get("predicted_fault")
    abnormal = bool(state["abnormal"])

    active_sensors = int(np.sum(channel_std > 1e-9))
    multisensor_support = active_sensors >= min(2, len(channel_std)) and np.sum(weights > 0.05) >= min(2, len(weights))
    signal_support = anomaly_score >= float(state.get("anomaly_threshold", 0.25)) and kurtosis > 3.0

    if label is not None and not abnormal:
        status = "CONTRADICTED"
        reason = "Classifier produced a fault label without corroborating signal-level anomaly evidence."
    elif abnormal and signal_support and multisensor_support:
        status = "SUPPORTED"
        reason = "Fault evidence is corroborated by impulsive signal structure and support from multiple synchronized sensors."
    elif abnormal:
        status = "INSUFFICIENT"
        reason = "Signal anomaly is present, but multisensor or impulsiveness corroboration is incomplete."
    else:
        status = "INSUFFICIENT"
        reason = "No transformer anomaly was declared; multisensor checks found no contradiction."

    return VerificationResult(
        hypothesis=str(label or "transformer_fault"),
        status=status,
        reason=reason,
        evidence_ids=[evidence_id],
        verifier="transformer_multisensor_physics",
        details={
            "anomaly_score": anomaly_score,
            "kurtosis": kurtosis,
            "active_sensor_count": active_sensors,
            "multisensor_support": multisensor_support,
            "classifier_label": label,
        },
    )
