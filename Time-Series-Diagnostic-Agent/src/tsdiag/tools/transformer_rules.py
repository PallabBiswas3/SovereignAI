from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


_EPS = 1e-9


@dataclass(frozen=True)
class TransformerRuleReference:
    center: dict[str, float]
    scale: dict[str, float]


def _rms(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(np.asarray(x, dtype=float) ** 2, axis=0))


def _fundamental_phasors(abc: np.ndarray) -> np.ndarray:
    x = np.asarray(abc, dtype=float)
    x = x - np.mean(x, axis=0, keepdims=True)
    fft = np.fft.rfft(x, axis=0)
    if len(fft) <= 1:
        return np.zeros(3, dtype=complex)
    energy = np.sum(np.abs(fft[1:]) ** 2, axis=1)
    k = int(np.argmax(energy)) + 1
    return fft[k] / max(len(x), 1)


def _sequence(phasors: np.ndarray) -> tuple[complex, complex, complex]:
    a = np.exp(1j * 2 * np.pi / 3)
    aa, ab, ac = phasors
    zero = (aa + ab + ac) / 3
    positive = (aa + a * ab + a**2 * ac) / 3
    negative = (aa + a**2 * ab + a * ac) / 3
    return zero, positive, negative


def verify_grid_fault_subtype(signal_matrix, sampling_rate_hz, channel_names, predicted_class,
                              *, fundamental_hz=50.0, zero_sequence_observable=False) -> dict:
    """Check a grid-fault subtype against named three-phase current evidence.

    This does not retrain the classifier or locate the fault inside a transformer.
    Missing metadata, short windows and weak/ambiguous sequence evidence remain
    insufficient; the model label is never silently replaced by a physics guess.
    """
    classes = {"single_phase_ground_fault", "inter_phase_short_circuit_fault"}
    base = {"status": "INSUFFICIENT", "predicted_class": predicted_class,
            "method": "named_phase_sequence_consistency_v1"}
    if predicted_class not in classes:
        return {**base, "reason": "not_a_grid_fault_subtype"}
    if zero_sequence_observable is not True:
        return {**base, "reason": "measurement_grounding_and_zero_sequence_path_unverified"}
    names = list(channel_names)
    if any(names.count(name) != 1 for name in ("Ia", "Ib", "Ic")):
        return {**base, "reason": "named_current_channels_required"}
    x = np.asarray(signal_matrix, dtype=float)
    fs, fundamental = float(sampling_rate_hz), float(fundamental_hz)
    if x.ndim != 2 or x.shape[1] != len(names) or not np.all(np.isfinite(x)):
        raise ValueError("phase verification requires aligned finite named channels")
    if not np.isfinite(fs) or not np.isfinite(fundamental) or fs <= 0 or not 0 < fundamental < fs / 2:
        raise ValueError("phase verification requires valid sampling and fundamental frequencies")
    if len(x) / fs < 3 / fundamental:
        return {**base, "reason": "at_least_three_fundamental_cycles_required"}
    # Project onto the declared fundamental, not an arbitrary strongest FFT bin.
    t = np.arange(len(x)) / fs
    currents = x[:, [names.index(name) for name in ("Ia", "Ib", "Ic")]]
    phasors = np.mean((currents - currents.mean(axis=0)) * np.exp(-2j*np.pi*fundamental*t)[:, None], axis=0)
    zero, positive, negative = _sequence(phasors)
    scale = float(np.linalg.norm(phasors))
    if scale < 1e-9 or abs(negative) < .1 * scale:
        return {**base, "reason": "insufficient_unbalanced_current_evidence"}
    ratio = float(abs(zero) / max(abs(negative), 1e-12))
    inferred = "inter_phase_short_circuit_fault" if ratio <= .1 else (
        "single_phase_ground_fault" if ratio >= .5 else None)
    if inferred is None:
        return {**base, "reason": "ambiguous_zero_negative_sequence_ratio", "zero_negative_ratio": ratio}
    return {**base, "status": "SUPPORTED" if inferred == predicted_class else "CONTRADICTED",
            "reason": "phase_sequence_matches_subtype" if inferred == predicted_class else "phase_sequence_conflicts_with_subtype",
            "sequence_signature": inferred, "zero_negative_ratio": ratio}


def _thd(abc: np.ndarray) -> float:
    x = np.asarray(abc, dtype=float)
    x = x - np.mean(x, axis=0, keepdims=True)
    fft = np.abs(np.fft.rfft(x, axis=0))
    if len(fft) <= 2:
        return 0.0
    joint = np.sum(fft[1:] ** 2, axis=1)
    k = int(np.argmax(joint)) + 1
    total = np.sum(fft[1:] ** 2, axis=0)
    fundamental = fft[k] ** 2
    harmonic = np.maximum(total - fundamental, 0)
    return float(np.mean(np.sqrt(harmonic / (fundamental + _EPS))))


def transformer_electrical_features(signal_matrix) -> dict[str, float]:
    x = np.asarray(signal_matrix, dtype=float)
    if x.ndim != 2 or x.shape[1] < 6 or len(x) < 16 or not np.all(np.isfinite(x)):
        raise ValueError("transformer electrical rules require finite [samples, >=6] Ua,Ub,Uc,Ia,Ib,Ic data")
    v = x[:, :3]
    i = x[:, 3:6]
    vrms = _rms(v)
    irms = _rms(i)
    vp = _fundamental_phasors(v)
    ip = _fundamental_phasors(i)
    v0, v1, v2 = _sequence(vp)
    i0, i1, i2 = _sequence(ip)
    v_mean = float(np.mean(vrms))
    i_mean = float(np.mean(irms))
    v_imb = float(np.std(vrms) / (v_mean + _EPS))
    i_imb = float(np.std(irms) / (i_mean + _EPS))
    dv = np.diff(v, axis=0)
    di = np.diff(i, axis=0)
    impedance = vrms / (irms + _EPS)
    power_phase = np.mean(v * i, axis=0)
    return {
        "voltage_rms": v_mean,
        "current_rms": i_mean,
        "voltage_imbalance": v_imb,
        "current_imbalance": i_imb,
        "voltage_zero_sequence_ratio": float(abs(v0) / (abs(v1) + _EPS)),
        "voltage_negative_sequence_ratio": float(abs(v2) / (abs(v1) + _EPS)),
        "current_zero_sequence_ratio": float(abs(i0) / (abs(i1) + _EPS)),
        "current_negative_sequence_ratio": float(abs(i2) / (abs(i1) + _EPS)),
        "voltage_transient_ratio": float(np.max(np.sqrt(np.mean(dv**2, axis=1))) / (v_mean + _EPS)),
        "current_transient_ratio": float(np.max(np.sqrt(np.mean(di**2, axis=1))) / (i_mean + _EPS)),
        "voltage_thd": _thd(v),
        "current_thd": _thd(i),
        "apparent_impedance": float(np.mean(impedance)),
        "impedance_phase_spread": float(np.std(impedance) / (np.mean(impedance) + _EPS)),
        "power_phase_imbalance": float(np.std(power_phase) / (np.mean(np.abs(power_phase)) + _EPS)),
    }


def fit_transformer_rule_reference(feature_rows: Iterable[dict[str, float]]) -> TransformerRuleReference:
    rows = list(feature_rows)
    if not rows:
        raise ValueError("normal reference requires at least one feature row")
    center = {}
    scale = {}
    for key in rows[0]:
        values = np.asarray([float(row[key]) for row in rows], dtype=float)
        med = float(np.median(values))
        mad = float(np.median(np.abs(values - med)) * 1.4826)
        fallback = float(np.std(values))
        center[key] = med
        scale[key] = max(mad, fallback * .25, 1e-6)
    return TransformerRuleReference(center=center, scale=scale)


def transformer_rule_diagnosis(signal_matrix, reference: TransformerRuleReference, *, threshold: float = 3.0, external_asymmetry_threshold: float = 4.0) -> dict:
    f = transformer_electrical_features(signal_matrix)
    z = {key: (float(value) - reference.center[key]) / reference.scale[key] for key, value in f.items()}
    absz = {key: abs(value) for key, value in z.items()}
    disturbance = max(absz["current_rms"], absz["voltage_rms"], absz["current_transient_ratio"], absz["voltage_transient_ratio"], absz["apparent_impedance"], absz["current_thd"], absz["voltage_thd"])
    asymmetry = max(absz["current_negative_sequence_ratio"], absz["current_zero_sequence_ratio"], absz["voltage_negative_sequence_ratio"], absz["voltage_zero_sequence_ratio"], absz["current_imbalance"], absz["voltage_imbalance"], absz["power_phase_imbalance"])
    internal_signature = max(absz["apparent_impedance"], absz["impedance_phase_spread"], absz["current_thd"], absz["voltage_thd"], absz["current_transient_ratio"])
    external_fault = bool(asymmetry >= external_asymmetry_threshold)
    transformer_candidate = bool(disturbance >= threshold and internal_signature >= threshold and not external_fault)
    score = float(internal_signature + .35 * disturbance - .45 * max(asymmetry - 2.0, 0.0))
    confidence = float(np.clip((score - threshold + 2.0) / 6.0, .05, .95))
    reasons = []
    for key, value in sorted(absz.items(), key=lambda kv: kv[1], reverse=True)[:5]:
        if value >= 2.0:
            reasons.append(f"{key} deviates {value:.2f} robust-sigma from normal")
    if external_fault:
        reasons.append("sequence/phase asymmetry exceeds the external-fault rejection threshold")
    return {
        "features": f,
        "robust_z": z,
        "disturbance_score": float(disturbance),
        "asymmetry_score": float(asymmetry),
        "internal_signature_score": float(internal_signature),
        "rule_score": score,
        "external_fault_signature": external_fault,
        "transformer_fault": transformer_candidate,
        "confidence": confidence,
        "reasons": reasons,
        "method": "deterministic_transformer_protection_rules_v2",
    }


def arbitrate_transformer_hybrid(*, classifier_positive: bool, classifier_confidence: float, rule_result: dict | None) -> dict:
    """Conservative ML+physics arbitration without test-set-specific tuning."""
    ml_conf = float(np.clip(classifier_confidence, 0.0, 1.0))
    if rule_result is None:
        return {
            "decision": "diagnose" if classifier_positive else "monitor",
            "confidence": ml_conf,
            "reason": "classifier_only",
            "verification": "INSUFFICIENT",
        }

    rule_positive = bool(rule_result.get("transformer_fault", False))
    external = bool(rule_result.get("external_fault_signature", False))
    rule_conf = float(np.clip(rule_result.get("confidence", 0.0), 0.0, 1.0))

    if classifier_positive and rule_positive:
        return {"decision": "diagnose", "confidence": max(ml_conf, rule_conf), "reason": "ml_and_physics_agree", "verification": "SUPPORTED"}
    if classifier_positive and external:
        return {"decision": "abstain", "confidence": min(ml_conf, 1.0 - rule_conf), "reason": "ml_physics_conflict_external_signature", "verification": "CONTRADICTED"}
    if classifier_positive:
        return {"decision": "diagnose", "confidence": 0.75 * ml_conf + 0.25 * rule_conf, "reason": "ml_positive_physics_inconclusive", "verification": "INSUFFICIENT"}
    if rule_positive:
        return {"decision": "abstain", "confidence": rule_conf, "reason": "physics_positive_ml_negative", "verification": "INSUFFICIENT"}
    return {"decision": "monitor", "confidence": max(1.0 - ml_conf, rule_conf if external else 0.0), "reason": "no_transformer_consensus", "verification": "SUPPORTED" if external else "INSUFFICIENT"}
