from __future__ import annotations

import math
from dataclasses import asdict
from typing import Callable

import numpy as np
from scipy.signal import butter, filtfilt, hilbert, stft, welch
from sklearn.decomposition import PCA
from sklearn.neighbors import LocalOutlierFactor
from statsmodels.tsa.stattools import grangercausalitytests

from .base import DiagnosticAgent
from ..models import AgentResult, Evidence, SignalRecord


def _primary_channel(record: SignalRecord) -> np.ndarray:
    record.validate()
    return record.signal[:, 0].astype(float)


def _pearson_kurtosis(x: np.ndarray) -> float:
    z = x - np.mean(x)
    m2 = np.mean(z**2)
    return float(np.mean(z**4) / (m2**2)) if m2 > 0 else 0.0


class SignalProcessingAgent(DiagnosticAgent):
    """Classical signal-processing diagnostics for vibration/current/acoustic data."""

    name = "signal_processing"

    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        x = _primary_channel(record)
        fs = record.fs
        centered = x - np.mean(x)
        rms = float(np.sqrt(np.mean(centered**2)))
        peak = float(np.max(np.abs(centered)))
        kurt = _pearson_kurtosis(centered)
        crest = peak / rms if rms > 0 else 0.0

        f, pxx = welch(centered, fs=fs, nperseg=min(2048, len(x)))
        dominant_idx = int(np.argmax(pxx[1:]) + 1) if len(pxx) > 1 else 0
        dominant_hz = float(f[dominant_idx])

        _, _, zxx = stft(centered, fs=fs, nperseg=min(512, len(x)), noverlap=min(384, max(0, len(x)-1)))
        stft_power = np.abs(zxx) ** 2
        temporal_cv = float(np.mean(np.std(stft_power, axis=1) / (np.mean(stft_power, axis=1) + 1e-12)))

        score = float(np.clip((max(kurt - 3.0, 0) / 10.0 + min(crest / 10.0, 1.0) + min(temporal_cv, 2.0) / 2.0) / 3.0, 0, 1))
        evidence = [
            Evidence("time_domain", f"RMS={rms:.4g}, crest factor={crest:.3f}, Pearson kurtosis={kurt:.3f}", score=max(0.1, score)),
            Evidence("welch_psd", f"Dominant spectral component near {dominant_hz:.3f} Hz", score=0.7),
            Evidence("stft", f"Mean temporal spectral variability={temporal_cv:.3f}", score=0.6),
        ]

        return AgentResult(
            agent=self.name,
            status="warning" if score >= 0.5 else "ok",
            summary="Impulsive/non-stationary behavior detected." if score >= 0.5 else "No strong generic impulsive signature detected.",
            confidence=0.55 + 0.4 * abs(score - 0.5),
            evidence=evidence,
            metrics={"rms": rms, "peak": peak, "crest_factor": crest, "pearson_kurtosis": kurt, "dominant_hz": dominant_hz, "temporal_spectral_cv": temporal_cv, "anomaly_score": score},
        )


class BearingDiagnosticAgent(DiagnosticAgent):
    """Envelope-spectrum matching for rotating-bearing characteristic frequencies."""

    name = "bearing_fault"

    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        if not record.fault_frequencies:
            return AgentResult(self.name, "abstain", "No BPFO/BPFI/BSF/FTF metadata available.", 0.0)

        x = _primary_channel(record)
        fs = record.fs
        centered = x - np.mean(x)

        # Generic high-frequency resonance band. Later phases will replace this with a kurtogram.
        low = max(50.0, 0.15 * fs)
        high = min(0.45 * fs, fs / 2 * 0.95)
        if high <= low:
            return AgentResult(self.name, "abstain", "Sampling rate is too low for the default envelope-analysis band.", 0.0)

        b, a = butter(4, [low, high], btype="bandpass", fs=fs)
        filtered = filtfilt(b, a, centered)
        envelope = np.abs(hilbert(filtered))
        env = envelope - np.mean(envelope)
        spec = np.abs(np.fft.rfft(env)) ** 2
        freq = np.fft.rfftfreq(len(env), d=1 / fs)
        floor = max(float(np.median(spec)), 1e-12)

        ranked: list[tuple[str, float, list[dict]]] = []
        for fault, base in record.fault_frequencies.items():
            matches = []
            values = []
            for harmonic in range(1, 4):
                target = base * harmonic
                if target >= freq[-1]:
                    continue
                tol = max(1.0, 0.03 * target)
                idx = np.where(np.abs(freq - target) <= tol)[0]
                if idx.size == 0:
                    continue
                best = idx[np.argmax(spec[idx])]
                prominence = float(spec[best] / floor)
                values.append(math.log1p(prominence) / math.sqrt(harmonic))
                matches.append({"harmonic": harmonic, "target_hz": float(target), "peak_hz": float(freq[best]), "prominence_ratio": prominence})
            if values:
                ranked.append((fault, float(np.mean(values)), matches))

        if not ranked:
            return AgentResult(self.name, "abstain", "No characteristic-frequency evidence could be evaluated.", 0.2)

        ranked.sort(key=lambda item: item[1], reverse=True)
        best_fault, best_score, matches = ranked[0]
        second = ranked[1][1] if len(ranked) > 1 else 0.0
        separation = max(0.0, (best_score - second) / max(best_score, 1e-12))
        confidence = float(np.clip(0.45 + 0.35 * separation + 0.20 * min(len(matches) / 3, 1), 0, 0.99))

        return AgentResult(
            agent=self.name,
            status="warning",
            summary=f"Strongest bearing-frequency hypothesis: {best_fault}.",
            confidence=confidence,
            evidence=[Evidence("envelope_spectrum", f"{best_fault} harmonic family has the strongest envelope-spectrum support.", confidence, {"matches": matches})],
            metrics={"ranking": [{"fault": f, "score": s, "matches": m} for f, s, m in ranked], "band_hz": [low, high]},
            recommendations=["Confirm the diagnosis across another time window/load condition before maintenance action."],
        )


class StatisticalMonitoringAgent(DiagnosticAgent):
    """PCA/SPE/T2/LOF monitoring. Requires normal-reference windows in context['reference_windows']."""

    name = "statistical_monitoring"

    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        context = context or {}
        reference = context.get("reference_windows")
        if reference is None:
            return AgentResult(self.name, "abstain", "No normal-reference windows supplied for PCA/LOF monitoring.", 0.0)

        ref = np.asarray(reference, dtype=float)
        x = np.asarray(record.signal, dtype=float)
        if ref.ndim != 2 or x.ndim != 2 or ref.shape[1] != x.shape[1]:
            return AgentResult(self.name, "unavailable", "Reference/current feature dimensions do not match.", 0.0)

        n_components = max(1, min(ref.shape[1], ref.shape[0] - 1, context.get("pca_components", min(5, ref.shape[1]))))
        pca = PCA(n_components=n_components).fit(ref)
        scores = pca.transform(x)
        reconstructed = pca.inverse_transform(scores)
        spe = np.sum((x - reconstructed) ** 2, axis=1)
        mean = np.mean(pca.transform(ref), axis=0)
        cov = np.cov(pca.transform(ref), rowvar=False)
        cov = np.atleast_2d(cov) + 1e-8 * np.eye(n_components)
        inv_cov = np.linalg.pinv(cov)
        centered_scores = scores - mean
        t2 = np.einsum("ij,jk,ik->i", centered_scores, inv_cov, centered_scores)

        lof = LocalOutlierFactor(n_neighbors=min(20, max(2, len(ref) - 1)), novelty=True).fit(ref)
        lof_scores = -lof.score_samples(x)

        ref_scores = pca.transform(ref)
        ref_recon = pca.inverse_transform(ref_scores)
        ref_spe = np.sum((ref - ref_recon) ** 2, axis=1)
        ref_centered = ref_scores - mean
        ref_t2 = np.einsum("ij,jk,ik->i", ref_centered, inv_cov, ref_centered)
        ref_lof = -lof.score_samples(ref)

        q_spe = float(np.quantile(ref_spe, 0.99))
        q_t2 = float(np.quantile(ref_t2, 0.99))
        q_lof = float(np.quantile(ref_lof, 0.99))
        exceed = np.mean((spe > q_spe) | (t2 > q_t2) | (lof_scores > q_lof))

        return AgentResult(
            self.name,
            "warning" if exceed > 0.1 else "ok",
            f"{100*exceed:.1f}% of current samples exceed at least one 99% reference limit.",
            float(np.clip(0.6 + abs(exceed - 0.1), 0, 0.95)),
            evidence=[Evidence("pca_lof", "Reference-based PCA/SPE, Hotelling-T2 and LOF monitoring completed.", 0.8)],
            metrics={"exceedance_fraction": float(exceed), "spe_limit": q_spe, "t2_limit": q_t2, "lof_limit": q_lof, "explained_variance": pca.explained_variance_ratio_.tolist()},
        )


class ProbabilisticDiagnosticAgent(DiagnosticAgent):
    """Gaussian likelihood / uncertainty scoring against a normal reference distribution."""

    name = "probabilistic"

    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        context = context or {}
        reference = context.get("reference_windows")
        if reference is None:
            return AgentResult(self.name, "abstain", "No normal-reference data supplied for probabilistic scoring.", 0.0)
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(record.signal, dtype=float)
        if ref.ndim != 2 or cur.ndim != 2 or ref.shape[1] != cur.shape[1]:
            return AgentResult(self.name, "unavailable", "Reference/current dimensions do not match.", 0.0)

        mu = ref.mean(axis=0)
        cov = np.cov(ref, rowvar=False)
        cov = np.atleast_2d(cov) + 1e-6 * np.eye(ref.shape[1])
        inv = np.linalg.pinv(cov)
        d2 = np.einsum("ij,jk,ik->i", cur - mu, inv, cur - mu)
        ref_d2 = np.einsum("ij,jk,ik->i", ref - mu, inv, ref - mu)
        limit = float(np.quantile(ref_d2, 0.99))
        exceed = float(np.mean(d2 > limit))
        return AgentResult(
            self.name,
            "warning" if exceed > 0.1 else "ok",
            f"Gaussian-reference likelihood test flags {100*exceed:.1f}% of samples.",
            float(np.clip(0.6 + abs(exceed - 0.1), 0, 0.95)),
            evidence=[Evidence("mahalanobis", "Mahalanobis-distance likelihood surrogate evaluated against normal reference.", 0.75)],
            metrics={"mahalanobis_limit_99": limit, "exceedance_fraction": exceed, "mean_distance": float(np.mean(d2))},
        )


class CausalRootCauseAgent(DiagnosticAgent):
    """Pairwise Granger-causal screening for multivariate process signals."""

    name = "causal_root_cause"

    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        record.validate()
        x = record.signal
        if x.shape[1] < 2:
            return AgentResult(self.name, "abstain", "At least two synchronized channels are required for causal analysis.", 0.0)
        maxlag = int((context or {}).get("maxlag", 3))
        alpha = float((context or {}).get("alpha", 0.05))
        names = record.channels or [f"ch{i}" for i in range(x.shape[1])]
        edges = []

        for cause in range(x.shape[1]):
            for effect in range(x.shape[1]):
                if cause == effect:
                    continue
                pair = np.column_stack([x[:, effect], x[:, cause]])
                try:
                    res = grangercausalitytests(pair, maxlag=maxlag, verbose=False)
                    pvals = [res[lag][0]["ssr_ftest"][1] for lag in range(1, maxlag + 1)]
                    p = float(min(pvals))
                    if p < alpha:
                        edges.append({"cause": names[cause], "effect": names[effect], "p_value": p})
                except Exception:
                    continue

        outgoing = {name: 0 for name in names}
        for edge in edges:
            outgoing[edge["cause"]] += 1
        ranked = sorted(outgoing.items(), key=lambda kv: kv[1], reverse=True)
        root = ranked[0][0] if ranked and ranked[0][1] > 0 else None

        return AgentResult(
            self.name,
            "warning" if root else "ok",
            f"Candidate upstream/root-cause channel: {root}." if root else "No significant Granger-causal root candidate found.",
            0.65 if root else 0.5,
            evidence=[Evidence("granger_graph", f"Detected {len(edges)} significant directed predictive relationships.", 0.7, {"edges": edges})],
            metrics={"edges": edges, "outgoing_counts": outgoing, "candidate_root": root},
            recommendations=["Treat Granger links as predictive, not proof of physical causality; verify against process topology/physics."],
        )


class TransferRobustnessAgent(DiagnosticAgent):
    """Detect operating-condition/domain shift before trusting a trained diagnostic model."""

    name = "transfer_robustness"

    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        context = context or {}
        reference = context.get("reference_windows")
        if reference is None:
            return AgentResult(self.name, "abstain", "No source-domain reference supplied for shift analysis.", 0.0)
        ref = np.asarray(reference, dtype=float)
        cur = np.asarray(record.signal, dtype=float)
        if ref.shape[1] != cur.shape[1]:
            return AgentResult(self.name, "unavailable", "Source/target channel dimensions differ.", 0.0)

        mu0, mu1 = ref.mean(0), cur.mean(0)
        sd0, sd1 = ref.std(0) + 1e-12, cur.std(0) + 1e-12
        mean_shift = np.abs(mu1 - mu0) / sd0
        scale_shift = np.abs(np.log(sd1 / sd0))
        shift = float(np.mean(mean_shift + scale_shift))
        status = "warning" if shift > 1.0 else "ok"
        return AgentResult(
            self.name,
            status,
            f"Estimated source-to-target operating-domain shift score={shift:.3f}.",
            float(np.clip(0.55 + 0.15 * min(shift, 2), 0, 0.9)),
            evidence=[Evidence("domain_shift", "Compared channel means and scales between reference and current operating domains.", 0.7)],
            metrics={"shift_score": shift, "normalized_mean_shift": mean_shift.tolist(), "log_scale_shift": scale_shift.tolist()},
            recommendations=["Use domain adaptation/recalibration before relying on a classifier when shift is high."] if status == "warning" else [],
        )


class LearnedModelAgent(DiagnosticAgent):
    """Adapter for CNN/LSTM/GRU/Transformer/autoencoder/ensemble models supplied by the application."""

    name = "learned_model"

    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        predictor: Callable | None = (context or {}).get("predictor")
        if predictor is None:
            return AgentResult(self.name, "unavailable", "No trained learned-model predictor is registered.", 0.0, recommendations=["Register a callable predictor for CNN/LSTM/GRU/Transformer/autoencoder or ensemble inference."])
        output = predictor(record)
        if not isinstance(output, dict):
            return AgentResult(self.name, "unavailable", "Predictor must return a dictionary.", 0.0)
        label = output.get("label")
        confidence = float(output.get("confidence", 0.5))
        return AgentResult(
            self.name,
            "warning" if label not in (None, "normal", "healthy") else "ok",
            f"Learned-model prediction: {label}.",
            confidence,
            evidence=[Evidence("learned_model", f"Registered model predicted {label}.", confidence, output)],
            metrics=output,
        )


class MultimodalFusionAgent(DiagnosticAgent):
    """Fuse signal, text/KG, image, and model evidence passed through context."""

    name = "multimodal_fusion"

    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        context = context or {}
        items = context.get("modal_evidence", [])
        if not items:
            return AgentResult(self.name, "abstain", "No multimodal evidence supplied.", 0.0)
        normalized = []
        for item in items:
            if isinstance(item, Evidence):
                normalized.append(item)
            elif isinstance(item, dict) and "statement" in item:
                normalized.append(Evidence(item.get("source", "external"), item["statement"], float(item.get("score", 0.5)), item.get("details", {})))
        if not normalized:
            return AgentResult(self.name, "abstain", "Modal evidence could not be normalized.", 0.0)
        confidence = float(np.average([e.score for e in normalized]))
        return AgentResult(self.name, "ok", f"Fused {len(normalized)} signal/text/image/model evidence items.", confidence, normalized)


class EvidenceVerificationAgent(DiagnosticAgent):
    """Run domain/physics/manual verification hooks against candidate claims."""

    name = "evidence_verification"

    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        context = context or {}
        claims = context.get("claims", [])
        verifiers = context.get("verifiers", [])
        if not claims:
            return AgentResult(self.name, "abstain", "No candidate diagnostic claims supplied for verification.", 0.0)
        if not verifiers:
            return AgentResult(self.name, "unavailable", "No physics/manual/domain verification hooks registered.", 0.0)

        findings = []
        supported = 0
        contradicted = 0
        for claim in claims:
            labels = []
            for verifier in verifiers:
                try:
                    verdict = verifier(claim, record)
                    if verdict:
                        labels.append(verdict)
                except Exception:
                    continue
            label_values = [v.get("label") for v in labels if isinstance(v, dict)]
            if "CONTRADICTED" in label_values:
                final = "CONTRADICTED"
                contradicted += 1
            elif "SUPPORTED" in label_values:
                final = "SUPPORTED"
                supported += 1
            else:
                final = "INSUFFICIENT"
            findings.append({"claim": claim, "label": final, "verdicts": labels})

        total = len(claims)
        confidence = float(supported / total) if total else 0.0
        return AgentResult(
            self.name,
            "warning" if contradicted else "ok",
            f"Verified {total} claim(s): {supported} supported, {contradicted} contradicted.",
            confidence,
            evidence=[Evidence("verification", "Physics/manual/domain verification completed.", confidence, {"findings": findings})],
            metrics={"supported": supported, "contradicted": contradicted, "insufficient": total - supported - contradicted, "findings": findings},
        )


class PrognosticsAgent(DiagnosticAgent):
    """Simple health-trend/RUL baseline. Requires context['health_history']=[(time, health_index), ...]."""

    name = "prognostics"

    def run(self, record: SignalRecord, context: dict | None = None) -> AgentResult:
        history = (context or {}).get("health_history")
        if history is None or len(history) < 3:
            return AgentResult(self.name, "abstain", "At least three health-index observations are required for RUL trend estimation.", 0.0)
        arr = np.asarray(history, dtype=float)
        t = arr[:, 0]
        h = arr[:, 1]
        slope, intercept = np.polyfit(t, h, 1)
        threshold = float((context or {}).get("failure_threshold", 1.0))
        current_t = float(t[-1])
        if slope <= 0:
            return AgentResult(self.name, "ok", "Health index is not trending toward the configured failure threshold.", 0.55, metrics={"slope": float(slope), "rul": None})
        failure_t = (threshold - intercept) / slope
        rul = max(0.0, float(failure_t - current_t))
        residual = h - (slope * t + intercept)
        fit_error = float(np.sqrt(np.mean(residual**2)))
        confidence = float(np.clip(1.0 / (1.0 + 5.0 * fit_error), 0.2, 0.9))
        return AgentResult(
            self.name,
            "warning",
            f"Linear baseline estimates RUL≈{rul:.3f} time units to health-index threshold {threshold}.",
            confidence,
            evidence=[Evidence("health_trend", f"Health index slope={slope:.5g}; fitted threshold crossing gives RUL≈{rul:.3f}.", confidence)],
            metrics={"slope": float(slope), "intercept": float(intercept), "fit_rmse": fit_error, "failure_threshold": threshold, "rul": rul},
            recommendations=["Treat this as a baseline prognosis; use a validated degradation/RUL model for maintenance decisions."],
        )
