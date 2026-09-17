from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


@dataclass(frozen=True)
class WindPhysicsConfig:
    """Conservative, non-data-driven sanity bounds for utility-scale turbines."""

    air_density_kg_m3: float = 1.225
    rotor_area_m2: float | None = None
    rated_power_kw: float | None = None
    power_coefficient_limit: float = 0.59
    power_tolerance_fraction: float = 0.20
    curtailment_fraction: float = 0.55
    pitch_curtailment_deg: float = 8.0
    generator_active_threshold: float = 0.10

    bearing_offset_c: float = 2.0
    bearing_torque_coeff_c_per_unit: float = 0.15
    bearing_speed_coeff_c_per_unit: float = 0.30
    gearbox_offset_c: float = 5.0
    gearbox_torque_coeff_c_per_unit: float = 0.18
    gearbox_speed_coeff_c_per_unit: float = 0.35
    generator_offset_c: float = 8.0
    generator_torque_coeff_c_per_unit: float = 0.22
    generator_speed_coeff_c_per_unit: float = 0.40
    thermal_margin_c: float = 12.0
    min_samples: int = 8


def _as_2d(x) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 1:
        arr = arr[:, None]
    if arr.ndim != 2:
        raise ValueError("expected [samples, channels]")
    return arr


def _find_channel(channel_names: Iterable[str], token_groups: Iterable[Iterable[str]]) -> int | None:
    names = [str(x).lower() for x in channel_names]
    for tokens in token_groups:
        tokens = tuple(str(t).lower() for t in tokens)
        for i, name in enumerate(names):
            if all(token in name for token in tokens):
                return i
    return None


def _find_active_power(channel_names: Iterable[str]) -> int | None:
    """Resolve generated active-power channels without accepting reactive power."""
    names = [str(x).lower() for x in channel_names]
    preferred = (
        ("active", "power"),
        ("real", "power"),
        ("electrical", "power"),
        ("generator", "power"),
    )
    for tokens in preferred:
        for i, name in enumerate(names):
            if all(token in name for token in tokens) and "reactive" not in name:
                return i
    for i, name in enumerate(names):
        if "power" in name and "reactive" not in name and "apparent" not in name:
            return i
    return None


def _subsystem(channel: str) -> str:
    value = channel.lower()
    if "gear" in value or "sump" in value or ("oil" in value and "gear" in value):
        return "gearbox"
    if "bearing" in value:
        return "bearing"
    if "generator" in value or "winding" in value:
        return "generator"
    if "pitch" in value or "blade" in value:
        return "pitch_system"
    if "power" in value or "converter" in value or "current" in value or "voltage" in value:
        return "electrical"
    return "turbine"


def _thermal_coefficients(subsystem: str, config: WindPhysicsConfig) -> tuple[float, float, float]:
    if subsystem == "gearbox":
        return config.gearbox_offset_c, config.gearbox_torque_coeff_c_per_unit, config.gearbox_speed_coeff_c_per_unit
    if subsystem == "generator":
        return config.generator_offset_c, config.generator_torque_coeff_c_per_unit, config.generator_speed_coeff_c_per_unit
    return config.bearing_offset_c, config.bearing_torque_coeff_c_per_unit, config.bearing_speed_coeff_c_per_unit


def _thermal_findings(current, names, config):
    ambient = _find_channel(names, (("ambient", "temp"), ("outside", "temp"), ("air", "temp")))
    rotor = _find_channel(names, (("rotor", "speed"), ("rotation", "speed"), ("rotor", "rpm")))
    torque = _find_channel(names, (("generator", "torque"), ("torque",)))
    if ambient is None or rotor is None or torque is None:
        return []

    ambient_values = current[:, ambient]
    rotor_values = np.abs(current[:, rotor])
    torque_values = np.abs(current[:, torque])
    driver_ok = np.isfinite(ambient_values) & np.isfinite(rotor_values) & np.isfinite(torque_values)

    targets = []
    for i, name in enumerate(names):
        lower = name.lower()
        if "temp" not in lower and "temperature" not in lower:
            continue
        subsystem = _subsystem(name)
        if subsystem in {"bearing", "gearbox", "generator"}:
            targets.append((i, subsystem))

    findings = []
    for target, subsystem in targets:
        offset, torque_coeff, speed_coeff = _thermal_coefficients(subsystem, config)
        allowed = ambient_values + float(offset) + float(torque_coeff) * torque_values + float(speed_coeff) * rotor_values + float(config.thermal_margin_c)
        measured = current[:, target]
        excess = measured - allowed
        violation = driver_ok & np.isfinite(measured) & (excess > 0.0)
        if int(np.sum(violation)) >= config.min_samples:
            channel = names[target]
            p90_excess = float(np.nanpercentile(excess[violation], 90))
            findings.append({
                "category": "thermal_mechanical_dissipation",
                "subsystem": subsystem,
                "channel": channel,
                "flag": "temperature_above_steady_state_dissipation_bound",
                "violation_fraction": float(np.mean(violation)),
                "max_excess_c": float(np.nanmax(excess[violation])),
                "p90_excess_c": p90_excess,
                "thermal_margin_c": float(config.thermal_margin_c),
                "model": {
                    "offset_c": float(offset),
                    "torque_coeff_c_per_unit": float(torque_coeff),
                    "speed_coeff_c_per_unit": float(speed_coeff),
                },
                "severity": float(np.clip(1.0 + p90_excess / max(config.thermal_margin_c, 1e-6), 1.0, 5.0)),
            })
    return findings


def _power_findings(reference, current, names, config):
    wind = _find_channel(names, (("wind", "speed"),))
    power = _find_active_power(names)
    pitch = _find_channel(names, (("pitch",), ("blade", "angle")))
    generator = _find_channel(names, (("generator", "speed"), ("generator", "rpm"), ("rotor", "speed")))
    if wind is None or power is None:
        return []

    findings = []
    v = current[:, wind]
    p = current[:, power]
    rated = config.rated_power_kw
    if rated is None:
        finite = reference[:, power][np.isfinite(reference[:, power])]
        if finite.size:
            rated = float(np.nanpercentile(finite, 99.5))

    if config.rotor_area_m2 is not None:
        theoretical_kw = 0.5 * config.air_density_kg_m3 * config.rotor_area_m2 * config.power_coefficient_limit * np.maximum(v, 0.0) ** 3 / 1000.0
        upper = theoretical_kw * (1.0 + config.power_tolerance_fraction)
        if rated is not None and rated > 0:
            upper = np.minimum(upper, rated * (1.0 + config.power_tolerance_fraction))
        violation = np.isfinite(p) & np.isfinite(upper) & (p > upper) & (v > 2.0)
        if int(np.sum(violation)) >= config.min_samples:
            findings.append({
                "category": "aerodynamic_electrical",
                "subsystem": "electrical",
                "channel": names[power],
                "flag": "power_exceeds_aerodynamic_limit",
                "violation_fraction": float(np.mean(violation)),
                "severity": float(np.clip(np.nanpercentile((p / np.maximum(upper, 1e-6))[violation], 90), 1.0, 5.0)),
            })

    if rated is not None and rated > 0:
        active = np.ones(len(p), dtype=bool)
        if generator is not None:
            g = np.abs(current[:, generator])
            finite_g = np.abs(reference[:, generator])
            finite_g = finite_g[np.isfinite(finite_g)]
            threshold = config.generator_active_threshold * float(np.nanpercentile(finite_g, 95)) if finite_g.size else 0.0
            active = g > threshold
        finite_wind = reference[:, wind][np.isfinite(reference[:, wind])]
        if finite_wind.size:
            high_wind = v >= np.nanpercentile(finite_wind, 70)
            low_power = p < config.curtailment_fraction * rated
            pitch_ok = np.ones(len(p), dtype=bool) if pitch is None else np.abs(current[:, pitch]) < config.pitch_curtailment_deg
            unexplained = np.isfinite(p) & np.isfinite(v) & active & high_wind & low_power & pitch_ok
            if int(np.sum(unexplained)) >= config.min_samples:
                findings.append({
                    "category": "aerodynamic_electrical",
                    "subsystem": "electrical",
                    "channel": names[power],
                    "flag": "possible_unannounced_curtailment_or_electrical_loss",
                    "violation_fraction": float(np.mean(unexplained)),
                    "severity": float(np.clip(np.nanpercentile((rated - p[unexplained]) / rated, 90) * 3.0, 1.0, 5.0)),
                })
    return findings


def physics_consistency_check(signal_matrix, channel_names, *, normal_reference=None, config: WindPhysicsConfig | None = None) -> dict[str, Any]:
    cfg = config or WindPhysicsConfig()
    current = _as_2d(signal_matrix)
    names = [str(x) for x in channel_names]
    if len(names) != current.shape[1]:
        raise ValueError("channel_names length mismatch")
    reference = current if normal_reference is None else _as_2d(normal_reference)
    if reference.shape[1] != current.shape[1]:
        raise ValueError("normal_reference channel count mismatch")

    findings = _power_findings(reference, current, names, cfg) + _thermal_findings(current, names, cfg)
    severity = max([float(row.get("severity", 0.0)) for row in findings], default=0.0)
    subsystem_scores: dict[str, float] = {}
    for row in findings:
        subsystem = str(row.get("subsystem", "turbine"))
        subsystem_scores[subsystem] = max(subsystem_scores.get(subsystem, 0.0), float(row.get("severity", 0.0)))
    ranked_subsystems = [name for name, _ in sorted(subsystem_scores.items(), key=lambda item: item[1], reverse=True)]
    return {
        "verification_findings": findings,
        "physics_consistent": len(findings) == 0,
        "max_severity": float(severity),
        "subsystem_scores": subsystem_scores,
        "ranked_subsystems": ranked_subsystems,
        "checks_run": ["aerodynamic_electrical", "thermal_mechanical_dissipation"],
    }
