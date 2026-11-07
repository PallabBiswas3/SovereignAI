"""Explicit adapters for server-registered diagnostic research artifacts.

No models are trained or loaded from request-supplied paths. Registered snapshots
are copied and checked for mutation; per-request battery fitting uses a new copy.
Supplementary bearing/Wind analyses never override the domain policy's decision.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from time import perf_counter
from typing import Any, Literal

import joblib
import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .models import Evidence, ToolTraceStep, UncertaintyEstimate


class DiagnosticUpdate(BaseModel):
    """Authoritative additive API response for one selected diagnostic update."""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model_ref: str = ""
    version: str = ""
    method: str
    status: Literal["completed", "abstained"] = "completed"
    summary: str
    calibrated: bool = False
    calibration_dataset: str | None = None
    synthetic_artifact: bool = False
    prediction_set: list[str] = Field(default_factory=list)
    point_estimate: float | None = Field(default=None, ge=0)
    interval: tuple[float, float] | None = None
    unit: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def valid_forecast(self):
        if self.interval is not None:
            lo, hi = self.interval
            if not 0 <= lo <= hi:
                raise ValueError("interval must be ordered and nonnegative")
            if self.point_estimate is not None and not lo <= self.point_estimate <= hi:
                raise ValueError("point estimate must lie within its interval")
        if self.status == "abstained" and (self.point_estimate is not None or self.interval is not None):
            raise ValueError("abstained analysis cannot publish a finite forecast")
        return self


def _finite_interval(values):
    values = np.asarray(values, dtype=float)
    if values.shape != (3,) or not np.isfinite(values).all():
        return None
    lo, med, hi = map(float, values)
    if not 0 <= lo <= med <= hi:
        raise ValueError("forecast must have ordered nonnegative quantiles")
    return lo, med, hi


@dataclass
class TransformerSequenceAdapter:
    model: Any
    conformal: Any
    labels: dict[int, str]
    normal_labels: tuple[str, ...] = ("normal",)
    model_family = "sequence-features-conformal-v1"
    domain = "transformer"
    task = "fault_diagnosis"
    slot = "raw_waveform_model"

    def validate(self):
        classes = getattr(self.model, "classes_", None)
        if classes is None or self.conformal.qhat_ is None:
            raise ValueError("sequence model and conformal calibrator must be fitted")
        if set(map(int, classes)) != set(self.labels) or len(classes) != len(self.conformal.qhat_):
            raise ValueError("class labels must align with fitted probability columns")
        if not set(self.normal_labels) <= set(self.labels.values()):
            raise ValueError("normal_labels must name registered classes")

    def __call__(self, waveform):
        self.validate()
        x = np.asarray(waveform, dtype=float)
        if x.ndim != 2 or x.shape[1] != 6:
            raise ValueError("sequence adapter requires Ua, Ub, Uc, Ia, Ib, Ic waveforms")
        probs = self.model.predict_proba(x[None])
        decision = self.conformal.decide(probs)
        names = [self.labels[int(c)] for c in self.model.classes_]
        prediction_set = [name for name, included in zip(names, decision.sets[0]) if included]
        answered = not bool(decision.abstain[0])
        label = names[int(decision.label[0])] if answered else None
        report = DiagnosticUpdate(
            method=self.model_family, status="completed" if answered else "abstained",
            summary=f"Singleton prediction: {label}." if answered else "Prediction set is empty or ambiguous; classification withheld.",
            calibrated=True, prediction_set=prediction_set,
            details={"probabilities": dict(zip(names, map(float, probs[0]))),
                     "nominal_set_coverage": 1 - self.conformal.alpha,
                     "coverage_is_marginal_not_answer_accuracy": True},
        )
        return {"label": label if label not in self.normal_labels else None,
                "predicted_class": label, "confidence": float(probs[0, int(decision.label[0])]) if answered else 0.,
                "probabilities": dict(zip(names, map(float, probs[0]))),
                "model_family": self.model_family, "diagnostic_update": report.model_dump(mode="json")}

    def assessment(self, inputs, execution, context):
        if tuple(inputs.get("sensor_positions", ())) != ("Ua", "Ub", "Uc", "Ia", "Ib", "Ic"):
            raise ValueError("sequence adapter requires named phase order Ua, Ub, Uc, Ia, Ib, Ic")
        return DiagnosticUpdate.model_validate(execution["classifier_update"])


@dataclass
class TurbofanRULAdapter:
    model: Any
    normalizer: Any = None
    domain = "turbofan"
    task = "remaining_useful_life"
    slot = "trained_rul_model"

    def validate(self):
        self.model._check_fitted()
        if self.model.conformal_ is None or self.model.interval_target_ != "uncapped":
            raise ValueError("public RUL requires fitted uncapped conformal calibration")

    def predict_diagnostic(self, inputs):
        from .challengers.turbofan_rul import make_windows
        self.validate()
        signal = np.asarray(inputs["signal_matrix"], dtype=float)
        cycles = np.asarray(inputs["cycle_index"], dtype=float)
        engine = str(inputs["engine_id"])
        if self.normalizer is not None:
            settings = inputs.get("operating_settings")
            if settings is None:
                raise ValueError("registered RUL preprocessing requires operating_settings")
            signal = self.normalizer.transform(np.asarray(settings, dtype=float), signal)
        windows, _, _ = make_windows(np.repeat(engine, len(cycles)), cycles, signal,
                                     window=self.model.window, last_only=True)
        mu, lo, hi = self.model.predict_interval(windows, units=[engine])
        interval = _finite_interval([lo[0], mu[0], hi[0]])
        if interval is None:
            raise ValueError("calibrated RUL interval is unbounded")
        low, point, high = interval
        report = DiagnosticUpdate(method="turbofan-ensemble-conformal-v1", summary="Calibrated remaining-life forecast.",
            calibrated=True, point_estimate=point, interval=(low, high), unit="cycles",
            details={"nominal_coverage": 1 - self.model.alpha, "interval_target": "uncapped"})
        return {"rul_cycles": point, "interval": [low, high], "calibrated": True,
                "coverage": 1 - self.model.alpha, "calibration_dataset": getattr(self, "calibration_dataset", None),
                "calibration_method": report.method, "diagnostic_update": report.model_dump(mode="json")}

    def assessment(self, inputs, execution, context):
        return DiagnosticUpdate.model_validate(execution["rul_calibration_output"]["diagnostic_update"])


@dataclass
class BatteryFadeAdapter:
    model: Any
    domain = "battery"
    task = "prognosis"
    slot = "capacity_prognostic_model"

    def validate(self):
        from .challengers.battery_prognostics import FadeParticleFilter, GPFadeModel
        if not isinstance(self.model, (FadeParticleFilter, GPFadeModel)):
            raise ValueError("capacity adapter requires a PF or GP fade model template")

    def predict_diagnostic(self, inputs):
        from .challengers.battery_prognostics import FadeParticleFilter, weighted_quantiles
        self.validate()
        model = deepcopy(self.model)
        if not np.isclose(model.eol, inputs["eol_capacity_ah"], rtol=0, atol=1e-12):
            raise ValueError("battery EOL threshold differs from registered model configuration")
        kwargs = {"cell_id": inputs["battery_id"]} if isinstance(model, FadeParticleFilter) else {}
        model.fit(inputs["cycle_index"], inputs["capacity_ah"], **kwargs)
        samples, weights = model.rul_samples()
        interval = _finite_interval(weighted_quantiles(samples, weights, [.05, .5, .95]))
        method = "battery-particle-filter-v1" if isinstance(model, FadeParticleFilter) else "battery-gp-fade-v1"
        if interval is None:
            return DiagnosticUpdate(method=method, status="abstained", unit="cycles",
                summary="Remaining-life quantiles are unbounded or censored; finite forecast withheld.",
                details={"interval_calibrated": False, "forecast_horizon": getattr(model, "censoring_horizon_", None)})
        lo, med, hi = interval
        return DiagnosticUpdate(method=method, summary="Capacity-history forecast with model-based, uncalibrated uncertainty.",
            point_estimate=med, interval=(lo, hi), unit="cycles",
            details={"interval_calibrated": False, "nominal_interval": .9,
                     "forecast_horizon": getattr(model, "censoring_horizon_", None)})

    def assessment(self, inputs, execution, context):
        return DiagnosticUpdate.model_validate(execution["capacity_update"])


@dataclass
class BearingCycloAdapter:
    healthy_scores: tuple[float, ...] = ()
    alpha: float = .1
    band: tuple[float, float] | None = None
    domain = "bearing"
    task = "fault_diagnosis"
    slot = "diagnostic_update"

    def validate(self):
        from .challengers.bearing_cyclostationary import healthy_threshold
        if self.healthy_scores:
            healthy_threshold(self.healthy_scores, self.alpha)
        elif not 0 < self.alpha < 1:
            raise ValueError("alpha must be between zero and one")

    def assessment(self, inputs, execution, context):
        from .challengers.bearing_cyclostationary import bearing_cyclo_features, max_fault_score, healthy_threshold, order_envelope_spectrum
        self.validate()
        features = bearing_cyclo_features(inputs["signal"], inputs["sampling_rate_hz"], inputs["fault_frequencies"], band=self.band)
        score = max_fault_score(features)
        details = {"features": features, "score": float(score), "supplementary_only": True}
        if self.healthy_scores:
            threshold = healthy_threshold(self.healthy_scores, self.alpha)
            details.update(threshold=float(threshold), alarm=bool(score > threshold), target_far=self.alpha,
                           calibration_recordings=len(self.healthy_scores))
        if self.band is not None and inputs.get("shaft_speed_rpm_trace") is not None:
            orders, amplitude = order_envelope_spectrum(inputs["signal"], inputs["sampling_rate_hz"],
                np.asarray(inputs["shaft_speed_rpm_trace"]) / 60., self.band)
            peak = int(np.argmax(amplitude[1:])) + 1
            details["dominant_order"] = float(orders[peak])
        return DiagnosticUpdate(method="bearing-modulation-order-v1", calibrated=bool(self.healthy_scores),
            summary="Additional bearing features; frequency and waveform-quality policy remains authoritative.", details=details)


@dataclass
class WindResidualProvider:
    """Fixed train-only NBM; optionally a separately fitted regime predictor."""
    state: Any
    regime_model: Any = None

    def __call__(self, inputs):
        from .calibration import EventResiduals
        from .tools.wind_scada import predict_wind_normal_behavior
        residuals = []
        for key in ("train_matrix", "prediction_matrix"):
            x = np.asarray(inputs[key], dtype=float)
            if self.regime_model is None:
                if len(self.state.regime_ids) != 1:
                    raise ValueError("multiregime NBM requires its frozen regime predictor")
                regimes = np.full(len(x), self.state.regime_ids[0])
            else:
                regimes = self.regime_model.predict(x)
            residuals.append(predict_wind_normal_behavior(x, regimes, self.state)["residuals"])
        return EventResiduals(train=residuals[0], predict=residuals[1])


@dataclass
class WindEventAdapter:
    calibrator: Any
    residual_provider: WindResidualProvider
    domain = "wind_scada"
    task = "condition_monitoring"
    slot = "diagnostic_update"

    def validate(self):
        if self.calibrator.calibration_ is None or not isinstance(self.residual_provider, WindResidualProvider):
            raise ValueError("Wind update requires frozen calibration and a fixed residual provider")

    def assessment(self, inputs, execution, context):
        self.validate()
        artifact = self.calibrator.calibration_
        event_id = str(context.get("specimen_id", ""))
        if not event_id or event_id in set(artifact.calibration_ids) | set(artifact.validation_ids):
            raise ValueError("Wind prediction event must be identified and disjoint from calibration/validation")
        decision = self.calibrator.detect(self.residual_provider(inputs))
        scorable = decision.alarm is not None
        return DiagnosticUpdate(method="wind-event-conformal-v1", calibrated=True,
            status="completed" if scorable else "abstained",
            summary="Frozen event-threshold assessment; quality and regime holds remain authoritative." if scorable else "Event is unscorable; no healthy or fault claim is made.",
            details={"alarm": decision.alarm, "score": float(decision.score) if scorable else None,
                     "threshold": float(decision.threshold), "first_alarm_index": decision.first_alarm_index,
                     "target_far": artifact.target_far, "calibration_sha256": artifact.config_sha256(),
                     "supplementary_only": True})


ADAPTERS = (TransformerSequenceAdapter, TurbofanRULAdapter, BatteryFadeAdapter, BearingCycloAdapter, WindEventAdapter)


def _fingerprint(adapter):
    return sha256(joblib.hash(adapter, hash_name="sha1").encode()).hexdigest()


def register_diagnostic_update(ref, adapter, *, version, input_profile, training_dataset,
                               calibration_dataset=None, synthetic=False, excluded_specimen_ids=()):
    """Call during service startup with trusted artifacts, never from an HTTP request.

    input_profile is produced by input_contract.model_input_profile(). Callers
    own physical-asset split declarations and supply excluded training/tuning IDs.
    """
    from .registry import model_registry
    if type(adapter) not in ADAPTERS:
        raise TypeError("unsupported diagnostic update adapter")
    if not version or not training_dataset:
        raise ValueError("version and training_dataset are required")
    if input_profile.get("domain") != adapter.domain or input_profile.get("task") != adapter.task:
        raise ValueError("adapter does not match the public input profile")
    snapshot = deepcopy(adapter)
    if isinstance(snapshot, TurbofanRULAdapter):
        snapshot.calibration_dataset = calibration_dataset
    snapshot.validate()
    needs_calibration = isinstance(snapshot, (TransformerSequenceAdapter, TurbofanRULAdapter, WindEventAdapter)) or (isinstance(snapshot, BearingCycloAdapter) and bool(snapshot.healthy_scores))
    if needs_calibration and not calibration_dataset:
        raise ValueError("calibrated adapters require calibration_dataset provenance")
    checksum = _fingerprint(snapshot)

    def validate(artifact):
        artifact.validate()
        if _fingerprint(artifact) != checksum:
            raise ValueError("registered diagnostic update changed after registration; register a new version")

    return model_registry.register(ref, version=version, artifact=snapshot, checksum=checksum, validator=validate,
        metadata={"kind": "diagnostic_update", "slot": adapter.slot, "input_profile": deepcopy(input_profile),
                  "training_dataset": training_dataset, "calibration_dataset": calibration_dataset,
                  "synthetic": bool(synthetic), "excluded_specimen_ids": tuple(map(str, excluded_specimen_ids))})


def apply_diagnostic_updates(request, result, execution):
    from .registry import model_registry
    context = dict(request.run_context.metadata) if request.run_context else {}
    for slot, ref in request.model_refs.items():
        record = model_registry.resolve(ref)
        if record.metadata.get("kind") != "diagnostic_update":
            continue
        started = perf_counter()
        adapter = record.artifact
        if type(adapter) not in ADAPTERS or (adapter.domain, adapter.task, adapter.slot) != (request.domain, request.task, slot):
            raise ValueError("diagnostic update registered for another domain/task/slot")
        if str(context.get("specimen_id", "")) in record.metadata["excluded_specimen_ids"]:
            raise ValueError("prediction specimen overlaps registered training/tuning units")
        report = adapter.assessment(request.inputs, execution, context)
        report.model_ref, report.version = ref, record.version
        report.calibration_dataset = record.metadata.get("calibration_dataset")
        report.synthetic_artifact = record.metadata["synthetic"]
        report.details["training_dataset"] = record.metadata["training_dataset"]
        # Ensure NumPy and unbounded values never cross the public JSON boundary.
        encoded = json.loads(json.dumps(report.model_dump(mode="json"), allow_nan=False))
        result.diagnostic_updates.append(encoded)
        evidence_id = f"diagnostic-update-{len(result.diagnostic_updates)}"
        # Evidence statements are displayed as sources. When the domain policy
        # holds the result, the statement must say so and must not restate a
        # forecast the policy withheld (e.g. an interval that is too wide).
        held = result.decision == "abstain" and report.status == "completed"
        statement = report.summary
        if held:
            statement = (f"Policy hold ({result.abstain_reason}): supplementary analysis only, "
                         f"not a released result. {statement}")
            if report.point_estimate is not None:
                statement += " The model forecast is withheld."
        elif report.point_estimate is not None:
            statement += f" Model RUL estimate: {report.point_estimate:.6g} {report.unit}; interval: {report.interval} {report.unit}."
        if report.synthetic_artifact:
            statement = "Synthetic demonstration artifact. " + statement
        result.evidence.append(Evidence(source="registered_diagnostic_update", statement=statement, score=0.,
            evidence_id=evidence_id, kind="model", details=encoded,
            provenance={"model_ref": ref, "version": record.version, "checksum": record.checksum,
                        "synthetic_artifact": report.synthetic_artifact}))
        result.tool_trace.append(ToolTraceStep("diagnostic_update", status="warning" if report.status == "abstained" else "ok",
            duration_seconds=perf_counter() - started, evidence_ids=[evidence_id],
            details={"model_ref": ref, "method": report.method, "version": record.version}))
        if isinstance(adapter, TransformerSequenceAdapter):
            result.uncertainty = None
            result.uncertainty_estimate = UncertaintyEstimate(None, report.method, calibrated=True,
                calibration_dataset=report.calibration_dataset, details={"prediction_set": report.prediction_set,
                "applies_to": "sequence_classifier", "coverage_is_marginal_not_answer_accuracy": True})
        if report.status == "abstained" and not report.details.get("supplementary_only"):
            result.decision, result.abstained = "abstain", True
            result.abstain_reason = report.summary
            result.hypotheses = []
            result.confidence = 0.
            if result.prognosis is not None:
                result.prognosis.remaining_useful_life = None
                result.prognosis.details = {"method": report.method, "interval": None, "unavailable_reason": report.summary}
                result.prognosis.uncertainty = None
                result.uncertainty_estimate = UncertaintyEstimate(None, report.method, calibrated=report.calibrated,
                    calibration_dataset=report.calibration_dataset, details={"forecast_unavailable": True})
            result.uncertainty = None
            result.metadata["allow_confidence_complement_uncertainty"] = False
        if held:
            encoded["details"]["policy_hold_remains"] = True
    return result
