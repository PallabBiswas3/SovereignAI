from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from ..contracts import DiagnosticRequest
from ..execution import ExecutionTrace, Step, Workflow
from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    LocalizationResult,
    UncertaintyEstimate,
    VerificationResult,
)
from ..result_contract import standardize_result
from .domain_steps import default_domain_tool_registry


_FAULT_COMPONENTS = {
    "BPFO": "outer_race",
    "BPFI": "inner_race",
    "BSF": "rolling_element",
    "FTF": "cage",
}
_FATAL_QUALITY = {"nan_or_inf", "too_short", "invalid_sampling_rate", "no_finite_samples"}


def _tool(name: str, state: dict[str, Any]) -> dict[str, Any]:
    return default_domain_tool_registry().get("bearing", name)(state)


def _inactive(state: Mapping[str, Any]) -> bool:
    quality = state.get("quality") or {}
    return bool(_FATAL_QUALITY.intersection(quality.get("quality_flags", [])))


@dataclass(frozen=True)
class BearingDecisionPolicy:
    minimum_confidence: float = 0.45
    minimum_harmonics: int = 2
    version: str = "bearing-policy-v2"

    def decide(self, execution: Mapping[str, Any], trace: ExecutionTrace) -> DiagnosticResult:
        quality = dict(execution.get("quality") or {})
        features = dict(execution.get("time_domain_features") or {})
        channel_name = str(execution.get("channel_name", "ch0"))
        fault_frequencies = dict(execution.get("fault_frequencies") or {})
        band = execution.get("recommended_band_hz")
        evidence: list[Evidence] = []

        if _FATAL_QUALITY.intersection(quality.get("quality_flags", [])):
            return standardize_result(DiagnosticResult(
                domain="bearing",
                task="fault_diagnosis",
                decision="abstain",
                detection=DetectionResult(abnormal=None, method="bearing_pipeline"),
                confidence=0.0,
                uncertainty=None,
                uncertainty_estimate=UncertaintyEstimate(None, "not_estimable_signal_quality"),
                abstained=True,
                abstain_reason=f"Signal quality prevents diagnosis: {quality.get('quality_flags', [])}",
                tool_trace=trace,
                metadata={"quality": quality, "workflow_version": BearingPlugin.workflow_version, "policy_version": self.version, "allow_confidence_complement_uncertainty": False},
            ))

        if features:
            ev_features = Evidence(
                source="time_domain_features",
                statement=f"crest_factor={features['crest_factor']:.3f}, kurtosis={features['kurtosis']:.3f}, rms={features['rms']:.4g}",
                score=float(np.clip(max(features["kurtosis"] - 3.0, 0.0) / 8.0, 0.0, 1.0)),
                evidence_id="bearing-time-features",
                kind="signal",
                provenance={"channel": channel_name},
                details=features,
            )
            evidence.append(ev_features)
            trace.require("time_domain_features").evidence_ids.append(ev_features.evidence_id)

        psd = execution.get("welch_psd_result") or {}
        if psd:
            ev_psd = Evidence(
                source="welch_psd",
                statement=f"Computed Welch PSD with {len(psd.get('dominant_peaks', []))} dominant peaks.",
                score=0.5,
                evidence_id="bearing-welch-psd",
                kind="signal",
                provenance={"channel": channel_name},
                details={"dominant_peaks": psd.get("dominant_peaks", [])},
            )
            evidence.append(ev_psd)
            trace.require("welch_psd").evidence_ids.append(ev_psd.evidence_id)

        sk = execution.get("spectral_kurtosis_result") or {}
        if band is None:
            return standardize_result(DiagnosticResult(
                domain="bearing",
                task="fault_diagnosis",
                decision="abstain",
                detection=DetectionResult(abnormal=None, method="spectral_kurtosis"),
                evidence=evidence,
                confidence=0.1,
                uncertainty=None,
                uncertainty_estimate=UncertaintyEstimate(None, "not_estimable_missing_resonance_band"),
                abstained=True,
                abstain_reason="No valid resonance band could be selected from spectral kurtosis.",
                tool_trace=trace,
                metadata={"quality": quality, "time_features": features, "workflow_version": BearingPlugin.workflow_version, "policy_version": self.version, "allow_confidence_complement_uncertainty": False},
            ))

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
        trace.require("spectral_kurtosis").evidence_ids.append(ev_band.evidence_id)

        if not fault_frequencies:
            detection_score = float(np.clip(max(features.get("kurtosis", 3.0) - 3.0, 0.0) / 7.0, 0.0, 1.0))
            return standardize_result(DiagnosticResult(
                domain="bearing",
                task="fault_diagnosis",
                decision="abstain",
                detection=DetectionResult(
                    abnormal=features.get("kurtosis", 0.0) > 3.5 or features.get("crest_factor", 0.0) > 4.0,
                    score=detection_score,
                    method="impulsiveness_without_fault_frequency_metadata",
                ),
                evidence=evidence,
                confidence=0.2,
                uncertainty=None,
                uncertainty_estimate=UncertaintyEstimate(None, "not_estimable_missing_fault_metadata"),
                abstained=True,
                abstain_reason="BPFO/BPFI/BSF/FTF metadata is required for bearing fault localization.",
                recommended_actions=["Provide bearing geometry or characteristic fault frequencies for localization."],
                tool_trace=trace,
                metadata={"quality": quality, "time_features": features, "resonance_band_hz": band, "workflow_version": BearingPlugin.workflow_version, "policy_version": self.version, "allow_confidence_complement_uncertainty": False},
            ))

        match = execution.get("bearing_frequency_match_result") or {}
        fusion = execution.get("bearing_evidence_fusion_result") or {}
        ranking = list(match.get("fault_ranking", []))
        best_fault = ranking[0]["fault"] if ranking else None
        harmonic_matches = match.get("harmonic_matches", {}).get(best_fault, []) if best_fault else []
        sideband_matches = match.get("sideband_matches", {}).get(best_fault, []) if best_fault else []
        if best_fault is not None:
            ev_freq = Evidence(
                source="bearing_frequency_match",
                statement=f"{best_fault} has {len(harmonic_matches)} harmonic matches and {len(sideband_matches)} shaft-rate sideband matches.",
                score=float(np.clip(fusion.get("evidence_strength", 0.0), 0.0, 1.0)),
                evidence_id="bearing-frequency-family",
                kind="signal",
                provenance={"channel": channel_name},
                details={"ranking": ranking, "harmonic_matches": harmonic_matches, "sideband_matches": sideband_matches},
            )
            evidence.append(ev_freq)
            trace.require("bearing_evidence_fusion").evidence_ids.append(ev_freq.evidence_id)

        confidence = float(fusion.get("confidence", 0.0))
        abstain_reason = fusion.get("abstain_reason")
        if ranking and int(ranking[0].get("harmonic_count", 0)) < self.minimum_harmonics:
            abstain_reason = f"Fewer than {self.minimum_harmonics} characteristic-frequency harmonics are supported."
        if confidence < self.minimum_confidence:
            abstain_reason = abstain_reason or "Evidence confidence is below the bearing diagnosis threshold."
        decision = "abstain" if abstain_reason else "diagnose"
        label = None if abstain_reason else fusion.get("label")
        component = _FAULT_COMPONENTS.get(str(label).upper(), str(label).lower()) if label else None

        hypotheses: list[DiagnosticHypothesis] = []
        verification: list[VerificationResult] = []
        if best_fault is not None:
            evidence_ids = ["bearing-frequency-family", "bearing-resonance-band"] if any(row.evidence_id == "bearing-frequency-family" for row in evidence) else ["bearing-resonance-band"]
            hypotheses.append(DiagnosticHypothesis(
                label=str(best_fault), score=confidence,
                rationale="Characteristic-frequency harmonics were evaluated in an automatically selected resonance band.",
                alternatives=[str(row["fault"]) for row in ranking[1:3]], evidence_ids=evidence_ids, details={"ranking": ranking},
            ))
            verification.append(VerificationResult(
                hypothesis=str(best_fault), status="SUPPORTED" if not abstain_reason else "INSUFFICIENT",
                reason="Characteristic-frequency family and resonance-band evidence are sufficiently consistent." if not abstain_reason else str(abstain_reason),
                evidence_ids=evidence_ids, verifier="bearing_signal_physics",
            ))

        detection_score = float(np.clip(max(fusion.get("evidence_strength", 0.0), max(features.get("kurtosis", 3.0) - 3.0, 0.0) / 7.0), 0.0, 1.0))
        separation = float(np.clip(fusion.get("separation", 0.0), 0.0, 1.0))
        uncertainty_value = float(1.0 - separation) if ranking else None
        result = DiagnosticResult(
            domain="bearing", task="fault_diagnosis", decision=decision,
            detection=DetectionResult(abnormal=detection_score >= 0.35, score=detection_score, method="spectral_kurtosis_envelope", details={"severity": fusion.get("severity", 0.0)}),
            localization=LocalizationResult(components=[] if component is None else [component], channels=[channel_name], scores={} if component is None else {component: confidence}),
            hypotheses=hypotheses, evidence=evidence, verification=verification, confidence=confidence,
            uncertainty=uncertainty_value,
            uncertainty_estimate=UncertaintyEstimate(uncertainty_value, "competing_hypothesis_separation", calibrated=False, details={"separation": separation}),
            abstained=bool(abstain_reason), abstain_reason=abstain_reason,
            recommended_actions=["Collect another clean vibration window and verify bearing geometry/speed metadata."] if abstain_reason else ["Confirm the diagnosis on an independent window/load condition before maintenance action."],
            tool_trace=trace,
            metadata={
                "quality": quality, "time_features": features, "resonance_band_hz": band, "fault_ranking": ranking,
                "fusion": fusion, "shaft_rate_hz": execution.get("shaft_rate_hz"), "operating_condition": dict(execution.get("operating_condition") or {}),
                "workflow_version": BearingPlugin.workflow_version, "policy_version": self.version,
                "allow_confidence_complement_uncertainty": False,
            },
        )
        return standardize_result(result)


class BearingPlugin:
    name = "bearing"
    workflow_version = "2.0"

    def validate(self, request: DiagnosticRequest) -> Mapping[str, Any]:
        values = dict(request.inputs)
        if values.get("signal") is None:
            values["signal"] = values.get("signal_matrix")
        if values.get("signal") is None or values.get("sampling_rate_hz") is None:
            raise ValueError("bearing requires signal and sampling_rate_hz")
        signal = np.asarray(values["signal"], dtype=float)
        if signal.ndim == 2 and signal.shape[1] >= 1:
            signal = signal[:, 0]
        if signal.ndim != 1:
            raise ValueError("signal must be [samples] or [samples, channels]")
        values["signal"] = signal
        values["sampling_rate_hz"] = float(values["sampling_rate_hz"])
        frequencies = dict(values.get("fault_frequencies") or {})
        for key in ("BPFO", "BPFI", "BSF", "FTF"):
            if values.get(key) is not None:
                frequencies[key] = float(values[key])
        values["fault_frequencies"] = frequencies
        values.setdefault("channel_name", "ch0")
        values.setdefault("operating_condition", {})
        return values

    def workflow(self, request: DiagnosticRequest) -> Workflow:
        def quality(state):
            out = _tool("signal_integrity", state)
            return {**out, "quality": out}

        def features(state):
            if _inactive(state): return {}
            out = _tool("time_domain_features", state)
            return {**out, "time_domain_features": out}

        def psd(state):
            if _inactive(state): return {}
            out = _tool("welch_psd", state)
            return {**out, "welch_psd_result": out}

        def sk(state):
            if _inactive(state): return {}
            out = _tool("spectral_kurtosis", state)
            return {**out, "spectral_kurtosis_result": out}

        def bandpass(state):
            if _inactive(state) or state.get("recommended_band_hz") is None: return {}
            return _tool("bandpass_filter", state)

        def envelope(state):
            if state.get("filtered_signal") is None: return {}
            return _tool("hilbert_envelope", state)

        def envelope_spectrum(state):
            if state.get("envelope") is None: return {}
            return _tool("envelope_spectrum", state)

        def match(state):
            if state.get("envelope_power") is None or not state.get("fault_frequencies"): return {"bearing_frequency_match_result": {"fault_ranking": [], "harmonic_matches": {}, "sideband_matches": {}}}
            out = _tool("bearing_frequency_match", state)
            return {**out, "bearing_frequency_match_result": out}

        def fusion(state):
            match_result = state.get("bearing_frequency_match_result") or {"fault_ranking": []}
            state = dict(state)
            state["fault_ranking"] = match_result.get("fault_ranking", [])
            state["transient_band"] = state.get("recommended_band_hz")
            out = _tool("bearing_evidence_fusion", state)
            return {**out, "bearing_evidence_fusion_result": out}

        return Workflow((
            Step("signal_integrity", quality, version="1.0"),
            Step("time_domain_features", features, depends_on=("signal_integrity",), version="1.0"),
            Step("welch_psd", psd, depends_on=("signal_integrity",), version="1.0"),
            Step("spectral_kurtosis", sk, depends_on=("welch_psd",), version="1.0"),
            Step("bandpass_filter", bandpass, depends_on=("spectral_kurtosis",), version="1.0"),
            Step("hilbert_envelope", envelope, depends_on=("bandpass_filter",), version="1.0"),
            Step("envelope_spectrum", envelope_spectrum, depends_on=("hilbert_envelope",), version="1.0"),
            Step("bearing_frequency_match", match, depends_on=("envelope_spectrum",), version="1.0"),
            Step("bearing_evidence_fusion", fusion, depends_on=("bearing_frequency_match",), version="1.0"),
        ), version=self.workflow_version)

    def policy(self, request: DiagnosticRequest) -> BearingDecisionPolicy:
        return BearingDecisionPolicy(
            minimum_confidence=float(request.inputs.get("minimum_confidence", 0.45)),
            minimum_harmonics=int(request.inputs.get("minimum_harmonics", 2)),
        )
