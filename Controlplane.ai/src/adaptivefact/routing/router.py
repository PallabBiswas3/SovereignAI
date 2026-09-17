from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

from adaptivefact.data.schema import ResponseRecord, VerificationRoute
from adaptivefact.risk.calibration import ProbabilityCalibrator
from adaptivefact.risk.features import RiskFeatureExtractor
from adaptivefact.risk.model import RiskModelBundle


@dataclass
class RoutingDecision:
    route: VerificationRoute
    raw_risk: float
    calibrated_risk: float
    latency_ms: float


class AdaptiveRouter:
    def __init__(
        self,
        model: RiskModelBundle,
        feature_extractor: RiskFeatureExtractor,
        calibrator: ProbabilityCalibrator,
        *,
        tau1: float,
        tau2: float,
    ) -> None:
        if not 0.0 <= tau1 < tau2 <= 1.0:
            raise ValueError("Expected 0 <= tau1 < tau2 <= 1")
        if model.feature_names != feature_extractor.feature_names:
            raise ValueError("Saved risk model and feature extractor use different features")
        self.model = model
        self.feature_extractor = feature_extractor
        self.calibrator = calibrator
        self.tau1 = float(tau1)
        self.tau2 = float(tau2)

    def route_probability(self, calibrated_risk: float) -> VerificationRoute:
        if calibrated_risk < self.tau1:
            return VerificationRoute.FAST_ACCEPT
        if calibrated_risk < self.tau2:
            return VerificationRoute.LIGHTWEIGHT
        return VerificationRoute.AGENTIC

    def route(self, record: ResponseRecord) -> RoutingDecision:
        start = perf_counter()
        X = self.feature_extractor.transform([record])
        raw = float(self.model.predict_proba(X)[0])
        calibrated = float(self.calibrator.predict([raw])[0])
        route = self.route_probability(calibrated)
        return RoutingDecision(
            route=route,
            raw_risk=raw,
            calibrated_risk=calibrated,
            latency_ms=(perf_counter() - start) * 1000.0,
        )
