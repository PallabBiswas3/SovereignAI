from __future__ import annotations

import math
from typing import Any

import numpy as np


def _as_spectrum(frequency_hz, envelope_power) -> tuple[np.ndarray, np.ndarray]:
    freq = np.asarray(frequency_hz, dtype=float).ravel()
    power = np.asarray(envelope_power, dtype=float).ravel()
    if freq.shape != power.shape:
        raise ValueError("frequency_hz and envelope_power must have matching shape")
    if freq.size < 4:
        raise ValueError("envelope spectrum is too short")
    if not np.all(np.isfinite(freq)):
        raise ValueError("frequency_hz contains non-finite values")
    power = np.where(np.isfinite(power) & (power >= 0.0), power, 0.0)
    return freq, power


def _peak_near(freq: np.ndarray, power: np.ndarray, target_hz: float, tolerance_hz: float, floor: float) -> dict[str, float] | None:
    if target_hz <= 0.0 or target_hz > float(freq[-1]):
        return None
    idx = np.flatnonzero(np.abs(freq - float(target_hz)) <= float(tolerance_hz))
    if idx.size == 0:
        return None
    best = int(idx[np.argmax(power[idx])])
    prominence = float(power[best] / max(floor, 1e-12))
    return {
        "target_hz": float(target_hz),
        "peak_hz": float(freq[best]),
        "power": float(power[best]),
        "prominence_ratio": prominence,
        "frequency_error_hz": float(abs(freq[best] - target_hz)),
    }


def bearing_frequency_match(
    frequency_hz,
    envelope_power,
    fault_frequencies: dict[str, float],
    *,
    shaft_rate_hz: float | None = None,
    harmonics: int = 4,
    relative_tolerance: float = 0.03,
    minimum_tolerance_hz: float = 1.0,
) -> dict[str, Any]:
    """Rank bearing-fault hypotheses using harmonics and shaft-rate sidebands.

    Scores are evidence strengths, not calibrated probabilities. Harmonic support
    is weighted more strongly than sidebands, and higher harmonics are mildly
    down-weighted to reduce overconfidence from broadband leakage.
    """
    freq, power = _as_spectrum(frequency_hz, envelope_power)
    if not fault_frequencies:
        return {"fault_ranking": [], "harmonic_matches": {}, "sideband_matches": {}}

    non_dc = power[freq > 0.0]
    floor = float(np.median(non_dc)) if non_dc.size else float(np.median(power))
    floor = max(floor, 1e-12)
    max_harmonics = max(1, int(harmonics))

    ranking = []
    all_harmonics: dict[str, list[dict[str, float]]] = {}
    all_sidebands: dict[str, list[dict[str, float]]] = {}

    for raw_name, raw_base in fault_frequencies.items():
        name = str(raw_name)
        base = float(raw_base)
        if not np.isfinite(base) or base <= 0.0:
            continue

        harmonic_matches: list[dict[str, float]] = []
        sideband_matches: list[dict[str, float]] = []
        harmonic_terms: list[float] = []
        sideband_terms: list[float] = []

        for harmonic in range(1, max_harmonics + 1):
            target = base * harmonic
            tolerance = max(float(minimum_tolerance_hz), float(relative_tolerance) * target)
            match = _peak_near(freq, power, target, tolerance, floor)
            if match is None:
                continue
            match = dict(match)
            match["harmonic"] = int(harmonic)
            harmonic_matches.append(match)
            harmonic_terms.append(math.log1p(match["prominence_ratio"]) / math.sqrt(harmonic))

            if shaft_rate_hz is not None and np.isfinite(shaft_rate_hz) and shaft_rate_hz > 0.0:
                for direction in (-1, 1):
                    side_target = target + direction * float(shaft_rate_hz)
                    side = _peak_near(freq, power, side_target, tolerance, floor)
                    if side is None:
                        continue
                    side = dict(side)
                    side.update({
                        "harmonic": int(harmonic),
                        "sideband_order": int(direction),
                        "carrier_target_hz": float(target),
                    })
                    sideband_matches.append(side)
                    sideband_terms.append(0.35 * math.log1p(side["prominence_ratio"]) / math.sqrt(harmonic))

        if not harmonic_terms:
            continue

        harmonic_score = float(np.mean(harmonic_terms))
        sideband_score = float(np.mean(sideband_terms)) if sideband_terms else 0.0
        total_score = harmonic_score + sideband_score
        ranking.append({
            "fault": name,
            "score": total_score,
            "harmonic_score": harmonic_score,
            "sideband_score": sideband_score,
            "harmonic_count": len(harmonic_matches),
            "sideband_count": len(sideband_matches),
        })
        all_harmonics[name] = harmonic_matches
        all_sidebands[name] = sideband_matches

    ranking.sort(key=lambda row: row["score"], reverse=True)
    return {
        "fault_ranking": ranking,
        "harmonic_matches": all_harmonics,
        "sideband_matches": all_sidebands,
        "noise_floor": floor,
    }


def bearing_evidence_fusion(
    time_features: dict[str, float],
    fault_ranking: list[dict[str, Any]],
    *,
    transient_band=None,
    operating_condition: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fuse impulsiveness, resonance-band and frequency-family evidence.

    This is intentionally conservative: a diagnosis needs both characteristic
    frequency support and enough separation from competing fault hypotheses.
    The output confidence is a bounded heuristic baseline to be calibrated on a
    real bearing benchmark later; it is not presented as a probability.
    """
    if not fault_ranking:
        return {
            "label": None,
            "confidence": 0.0,
            "severity": 0.0,
            "abstain_reason": "No characteristic-frequency hypothesis has usable evidence.",
            "evidence_strength": 0.0,
            "separation": 0.0,
        }

    best = fault_ranking[0]
    second_score = float(fault_ranking[1]["score"]) if len(fault_ranking) > 1 else 0.0
    best_score = float(best["score"])
    separation = max(0.0, (best_score - second_score) / max(best_score, 1e-12))
    frequency_strength = float(1.0 - math.exp(-best_score / 2.5))

    kurtosis = max(0.0, float(time_features.get("kurtosis", 3.0)) - 3.0)
    crest = max(0.0, float(time_features.get("crest_factor", 0.0)) - 3.0)
    impulsiveness = float(np.clip(0.55 * kurtosis / 7.0 + 0.45 * crest / 7.0, 0.0, 1.0))
    band_support = 1.0 if transient_band is not None else 0.0

    confidence = float(np.clip(
        0.50 * frequency_strength
        + 0.25 * separation
        + 0.15 * impulsiveness
        + 0.10 * band_support,
        0.0,
        0.99,
    ))
    severity = float(np.clip(0.65 * frequency_strength + 0.35 * impulsiveness, 0.0, 1.0))

    abstain_reason = None
    if int(best.get("harmonic_count", 0)) < 2:
        abstain_reason = "Fewer than two characteristic-frequency harmonics are supported."
    elif frequency_strength < 0.35:
        abstain_reason = "Characteristic-frequency evidence is too weak for a diagnosis."
    elif separation < 0.08 and len(fault_ranking) > 1:
        abstain_reason = "Competing bearing-fault hypotheses are not sufficiently separated."

    return {
        "label": None if abstain_reason else str(best["fault"]),
        "confidence": confidence,
        "severity": severity,
        "abstain_reason": abstain_reason,
        "evidence_strength": frequency_strength,
        "separation": separation,
        "impulsiveness": impulsiveness,
        "transient_band": None if transient_band is None else list(map(float, transient_band)),
        "operating_condition": dict(operating_condition or {}),
    }
