import numpy as np

from tsdiag.domains.wind_physics import WindPhysicsConfig, physics_consistency_check


def test_power_limit_flags_unphysical_generation():
    rng = np.random.default_rng(3)
    n = 180
    wind = rng.uniform(5.0, 10.0, n)
    pitch = np.zeros(n)
    rotor = 9.0 + 0.2 * wind
    ambient = np.full(n, 20.0)
    power = 45.0 * wind ** 3 / 1000.0
    reference = np.column_stack([wind, power, pitch, rotor, ambient])

    current = reference.copy()
    current[:, 1] *= 8.0
    result = physics_consistency_check(
        current,
        ["wind speed", "active power", "pitch angle", "generator speed", "ambient temp"],
        normal_reference=reference,
        config=WindPhysicsConfig(rotor_area_m2=50.0, rated_power_kw=3000.0, min_samples=4),
    )
    flags = {row["flag"] for row in result["verification_findings"]}
    assert "power_exceeds_aerodynamic_limit" in flags
    assert not result["physics_consistent"]


def test_reactive_power_is_not_used_as_active_power_alias():
    n = 120
    wind = np.full(n, 9.0)
    reactive_power = np.full(n, 5000.0)
    pitch = np.zeros(n)
    generator_speed = np.full(n, 1200.0)
    ambient = np.full(n, 20.0)
    reference = np.column_stack([wind, reactive_power, pitch, generator_speed, ambient])

    result = physics_consistency_check(
        reference,
        ["wind speed", "reactive_power_avg", "pitch angle", "generator speed", "ambient temp"],
        normal_reference=reference,
        config=WindPhysicsConfig(rotor_area_m2=50.0, rated_power_kw=3000.0, min_samples=4),
    )
    assert all(row.get("channel") != "reactive_power_avg" for row in result["verification_findings"])
    assert not any(
        row["flag"] in {"power_exceeds_aerodynamic_limit", "possible_unannounced_curtailment_or_electrical_loss"}
        for row in result["verification_findings"]
    )


def test_thermal_dissipation_flags_persistent_excess_without_fitting_reference():
    rng = np.random.default_rng(4)
    n = 220
    ambient = rng.normal(18.0, 2.0, n)
    rotor = rng.uniform(8.0, 15.0, n)
    torque = rng.uniform(20.0, 80.0, n)
    temp = ambient + 0.15 * torque + 0.30 * rotor + rng.normal(0.0, 0.4, n)
    reference = np.column_stack([ambient, rotor, torque, temp])
    current = reference.copy()
    current[-40:, 3] += 20.0

    result = physics_consistency_check(
        current,
        ["ambient temperature", "rotor speed", "generator torque", "main bearing temperature"],
        normal_reference=reference,
        config=WindPhysicsConfig(thermal_margin_c=8.0, min_samples=6),
    )
    thermal = [
        row
        for row in result["verification_findings"]
        if row["flag"] == "temperature_above_steady_state_dissipation_bound"
    ]
    assert thermal
    assert thermal[0]["subsystem"] == "bearing"
    assert thermal[0]["model"]["torque_coeff_c_per_unit"] == 0.15


def test_thermal_check_requires_physical_drivers():
    current = np.column_stack([np.full(100, 20.0), np.full(100, 95.0)])
    result = physics_consistency_check(
        current,
        ["ambient temperature", "main bearing temperature"],
        config=WindPhysicsConfig(min_samples=4),
    )
    assert result["verification_findings"] == []
