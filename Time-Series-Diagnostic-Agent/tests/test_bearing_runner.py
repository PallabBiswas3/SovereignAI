import numpy as np

from tsdiag.domains.bearing_runner import BearingDiagnosticPipeline


def _synthetic_modulated_bearing_signal(fs: float, seconds: float, fault_hz: float, carrier_hz: float, seed: int = 7):
    rng = np.random.default_rng(seed)
    n = int(fs * seconds)
    t = np.arange(n) / fs
    modulation = 1.0 + 0.85 * np.sin(2 * np.pi * fault_hz * t)
    carrier = modulation * np.sin(2 * np.pi * carrier_hz * t)
    impulses = np.zeros(n)
    period = max(1, int(round(fs / fault_hz)))
    impulses[::period] = 2.0
    kernel_t = np.arange(int(0.015 * fs)) / fs
    ring = np.exp(-180.0 * kernel_t) * np.sin(2 * np.pi * carrier_hz * kernel_t)
    ringing = np.convolve(impulses, ring, mode="same")
    noise = 0.15 * rng.normal(size=n)
    return carrier + ringing + noise


def test_bearing_pipeline_localizes_outer_race_with_harmonic_evidence():
    fs = 12000.0
    bpfo = 90.0
    x = _synthetic_modulated_bearing_signal(fs, seconds=2.0, fault_hz=bpfo, carrier_hz=2600.0)

    result = BearingDiagnosticPipeline(minimum_confidence=0.30).run(
        x,
        fs,
        fault_frequencies={
            "BPFO": bpfo,
            "BPFI": 135.0,
            "BSF": 58.0,
            "FTF": 12.0,
        },
        shaft_rate_hz=30.0,
        channel_name="drive_end",
        operating_condition={"load": "synthetic"},
    )

    assert result.domain == "bearing"
    assert result.evidence
    assert result.metadata["resonance_band_hz"] is not None
    assert result.hypotheses
    assert result.hypotheses[0].label == "BPFO"
    assert "outer_race" in result.localization.components
    assert result.decision in {"diagnose", "abstain"}
    assert result.to_dict()["schema_version"] == "1.0"


def test_bearing_pipeline_abstains_without_fault_frequency_metadata():
    fs = 8000.0
    x = _synthetic_modulated_bearing_signal(fs, seconds=1.5, fault_hz=75.0, carrier_hz=1800.0)

    result = BearingDiagnosticPipeline().run(
        x,
        fs,
        fault_frequencies={},
        shaft_rate_hz=25.0,
        channel_name="drive_end",
    )

    assert result.decision == "abstain"
    assert result.abstained is True
    assert "BPFO/BPFI/BSF/FTF" in result.abstain_reason
    assert result.localization.components == []
    assert result.tool_trace
