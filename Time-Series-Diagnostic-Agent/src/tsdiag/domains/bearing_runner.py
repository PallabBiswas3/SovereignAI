from __future__ import annotations

from typing import Any

import numpy as np

from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    VerificationResult,
)
from ..execution import StepExecutor
from .domain_steps import default_domain_tool_registry


_FAULT_COMPONENTS = {
    "BPFO": "outer_race",
    "BPFI": "inner_race",
    "BSF": "rolling_element",
    "FTF": "cage",
}


class BearingDiagnosticPipeline:
    """Evidence-grounded bearing diagnosis with automatic resonance-band selection.

    Tool execution is delegated to the shared :class:`StepExecutor`; diagnosis,
    thresholds, evidence fusion and abstention logic remain unchanged.
    """

    def __init__(
        self,
        *,
        minimum_confidence: float = 0.45,
        minimum_harmonics: int = 2,
    ):
        self.minimum_confidence = float(minimum_confidence)
        self.minimum_harmonics = int(minimum_harmonics)

    @staticmethod
    def _primary_channel(signal) -> np.ndarray:
        x = np.asarray(signal, dtype=float)
        if x.ndim == 1:
            return x
        if x.ndim == 2 and x.shape[1] >= 1:
            return x[:, 0]
        raise ValueError("signal must be [samples] or [samples, channels]")

    def run(
        self,
        signal,
        sampling_rate_hz: float,
        *,
        fault_frequencies: dict[str, float] | None = None,
        shaft_rate_hz: float | None = None,
        channel_name: str = "ch0",
        operating_condition: dict[str, Any] | None = None,
    ) -> DiagnosticResult:
        x = self._primary_channel(signal)
        fs = float(sampling_rate_hz)
        fault_frequencies = dict(fault_frequencies or {})

        evidence: list[Evidence] = []
        state: dict[str, Any] = {
            "signal": x,
            "sampling_rate_hz": fs,
            "fault_frequencies": fault_frequencies,
            "shaft_rate_hz": shaft_rate_hz,
            "operating_condition": dict(operating_condition or {}),
        }
        executor = StepExecutor("bearing", default_domain_tool_registry())
        trace = executor.trace

        quality = executor.run("signal_integrity", state)
        trace[-1].outputs_summary = {"quality_flags": quality["quality_flags"]}
        fatal_quality = {"nan_or_inf", "too_short", "invalid_sampling_rate", "no_finite_samples"}
        if fatal_quality.intersection(quality["quality_flags"]):
            result = DiagnosticResult(
                domain="bearing",
                task="fault_diagnosis",
                decision="abstain",
                detection=DetectionResult(abnormal=None, method="bearing_pipeline"),
                confidence=0.0,
                uncertainty=1.0,
                abstained=True,
                abstain_reason=f"Signal quality prevents diagnosis: {quality['quality_flags']}",
                tool_trace=trace,
                metadata={"quality": quality},
            )
            result.validate()
            return result

        features = executor.run("time_domain_features", state)
        # The generic tool writes scalar feature keys into shared state. Preserve
        # the structured feature object as well because bearing_evidence_fusion
        # consumes it as one contracted input.
        state["time_domain_features"] = features
        ev_features = Evidence(
            source="time_domain_features",
            statement=(
                f"crest_factor={features['crest_factor']:.3f}, "
                f"kurtosis={features['kurtosis']:.3f}, rms={features['rms']:.4g}"
            ),
            score=float(np.clip(max(features["kurtosis"] - 3.0, 0.0) / 8.0, 0.0, 1.0)),
            evidence_id="bearing-time-features",
            kind="signal",
            provenance={"channel": channel_name},
            details=features,
        )
        evidence.append(ev_features)
        trace[-1].evidence_ids.append(ev_features.evidence_id)

        psd = executor.run("welch_psd", state)
        ev_psd = Evidence(
            source="welch_psd",
            statement=f"Computed Welch PSD with {len(psd['dominant_peaks'])} dominant peaks.",
            score=0.5,
            evidence_id="bearing-welch-psd",
            kind="signal",
            provenance={"channel": channel_name},
            details={"dominant_peaks": psd["dominant_peaks"]},
        )
        evidence.append(ev_psd)
        trace[-1].evidence_ids.append(ev_psd.evidence_id)

        sk = executor.run("spectral_kurtosis", state)
        band = sk.get("recommended_band_hz")
        if band is None:
            result = DiagnosticResult(
                domain="bearing",
                task="fault_diagnosis",
                decision="abstain",
                detection=DetectionResult(abnormal=None, method="spectral_kurtosis"),
                evidence=evidence,
                confidence=0.1,
                uncertainty=0.9,
                abstained=True,
                abstain_reason="No valid resonance band could be selected from spectral kurtosis.",
                tool_trace=trace,
                metadata={"quality": quality, "time_features": features},
            )
            result.validate()
            return result

        ev_band = Evidence(
            source="spectral_kurtosis",
            statement=f"Recommended impulsive resonance band {band[0]:.1f}-{band[1]:.1f} Hz.",
            score=float(np.clip((sk.get("band_scores") or [{"kurtosis": 0.0}])[0]["kurtosis"] / 10.0, 0.0, 1.0)),
            evidence_id="bearing-resonance-band",
            kind="signal",
            provenance={"channel": channel_name},
            details=sk,
        )
        evidence.append(ev_band)
        trace[-1].evidence_ids.append(ev_band.evidence_id)

        filtered = executor.run("bandpass_filter", state)
        trace[-1].outputs_summary = filtered["filter_metadata"]
        envelope = executor.run("hilbert_envelope", state)
        env_spec = executor.run("envelope_spectrum", state)

        if not fault_frequencies:
            result = DiagnosticResult(
                domain="bearing",
                task="fault_diagnosis",
                decision="abstain",
                detection=DetectionResult(
                    abnormal=features["kurtosis"] > 3.5 or features["crest_factor"] > 4.0,
                    score=float(np.clip(max(features["kurtosis"] - 3.0, 0.0) / 7.0, 0.0, 1.0)),
                    method="impulsiveness_without_fault_frequency_metadata",
                ),
                evidence=evidence,
                confidence=0.2,
                uncertainty=0.8,
                abstained=True,
                abstain_reason="BPFO/BPFI/BSF/FTF metadata is required for bearing fault localization.",
                recommended_actions=["Provide bearing geometry or characteristic fault frequencies for localization."],
                tool_trace=trace,
                metadata={"quality": quality, "time_features": features, "resonance_band_hz": band},
            )
            result.validate()
            return result

        match = executor.run("bearing_frequency_match", state)
        ranking = match["fault_ranking"]
        # Keep the exact pre-migration fusion inputs. The shared registry adapter
        # accepts this structured feature object and transient resonance band.
        state["transient_band"] = band
        fusion = executor.run("bearing_evidence_fusion", state)

        best_fault = ranking[0]["fault"] if ranking else None
        harmonic_matches = match["harmonic_matches"].get(best_fault, []) if best_fault else []
        sideband_matches = match["sideband_matches"].get(best_fault, []) if best_fault else []
        if best_fault is not None:
            ev_freq = Evidence(
                source="bearing_frequency_match",
                statement=(
                    f"{best_fault} has {len(harmonic_matches)} harmonic matches and "
                    f"{len(sideband_matches)} shaft-rate sideband matches."
                ),
                score=float(np.clip(fusion["evidence_strength"], 0.0, 1.0)),
                evidence_id="bearing-frequency-family",
                kind="signal",
                provenance={"channel": channel_name},
                details={
                    "ranking": ranking,
                    "harmonic_matches": harmonic_matches,
                    "sideband_matches": sideband_matches,
                },
            )
            evidence.append(ev_freq)
            trace[-1].evidence_ids.append(ev_freq.evidence_id)

        confidence = float(fusion["confidence"])
        abstain_reason = fusion["abstain_reason"]
        if ranking and int(ranking[0].get("harmonic_count", 0)) < self.minimum_harmonics:
            abstain_reason = f"Fewer than {self.minimum_harmonics} characteristic-frequency harmonics are supported."
        if confidence < self.minimum_confidence:
            abstain_reason = abstain_reason or "Evidence confidence is below the bearing diagnosis threshold."

        if abstain_reason:
            decision = "abstain"
            abstained = True
            label = None
        else:
            decision = "diagnose"
            abstained = False
            label = str(fusion["label"])

        component = _FAULT_COMPONENTS.get(label.upper(), label.lower()) if label else None
        hypotheses: list[DiagnosticHypothesis] = []
        verification: list[VerificationResult] = []
        if best_fault is not None:
            evidence_ids = ["bearing-frequency-family", "bearing-resonance-band"] if any(
                row.evidence_id == "bearing-frequency-family" for row in evidence
            ) else ["bearing-resonance-band"]
            hypotheses.append(DiagnosticHypothesis(
                label=str(best_fault),
                score=confidence,
                rationale="Characteristic-frequency harmonics were evaluated in an automatically selected resonance band.",
                alternatives=[str(row["fault"]) for row in ranking[1:3]],
                evidence_ids=evidence_ids,
                details={"ranking": ranking},
            ))
            verification.append(VerificationResult(
                hypothesis=str(best_fault),
                status="SUPPORTED" if not abstain_reason else "INSUFFICIENT",
                reason=(
                    "Characteristic-frequency family and resonance-band evidence are sufficiently consistent."
                    if not abstain_reason
                    else abstain_reason
                ),
                evidence_ids=evidence_ids,
                verifier="bearing_signal_physics",
            ))

        detection_score = float(np.clip(max(
            fusion.get("evidence_strength", 0.0),
            max(features["kurtosis"] - 3.0, 0.0) / 7.0,
        ), 0.0, 1.0))

        result = DiagnosticResult(
            domain="bearing",
            task="fault_diagnosis",
            decision=decision,
            detection=DetectionResult(
                abnormal=detection_score >= 0.35,
                score=detection_score,
                method="spectral_kurtosis_envelope",
                details={"severity": fusion["severity"]},
            ),
            localization=LocalizationResult(
                components=[] if component is None else [component],
                channels=[channel_name],
                scores={} if component is None else {component: confidence},
            ),
            hypotheses=hypotheses,
            evidence=evidence,
            verification=verification,
            confidence=confidence,
            uncertainty=float(np.clip(1.0 - confidence, 0.0, 1.0)),
            abstained=abstained,
            abstain_reason=abstain_reason,
            recommended_actions=(
                ["Confirm the diagnosis on an independent window/load condition before maintenance action."]
                if not abstained
                else ["Collect another clean vibration window and verify bearing geometry/speed metadata."]
            ),
            tool_trace=trace,
            metadata={
                "quality": quality,
                "time_features": features,
                "resonance_band_hz": band,
                "fault_ranking": ranking,
                "fusion": fusion,
                "shaft_rate_hz": shaft_rate_hz,
                "operating_condition": dict(operating_condition or {}),
            },
        )
        result.validate()
        return result
