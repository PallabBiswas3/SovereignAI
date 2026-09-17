from __future__ import annotations

import numpy as np

from ..models import (
    DetectionResult,
    DiagnosticHypothesis,
    DiagnosticResult,
    Evidence,
    PrognosisResult,
    ToolTraceStep,
    VerificationResult,
)
from ..result_contract import standardize_result
from ..tools.battery_prognosis import capacity_features, estimate_capacity_eol


class BatteryPrognosticPipeline:
    """Capacity-history prognosis path for battery aging benchmarks."""

    def __init__(self, *, nominal_capacity_ah=2.0, eol_capacity_ah=1.4, minimum_observations=20, slope_window=20):
        self.nominal_capacity_ah = float(nominal_capacity_ah)
        self.eol_capacity_ah = float(eol_capacity_ah)
        self.minimum_observations = int(minimum_observations)
        self.slope_window = int(slope_window)

    def run(self, cycle_index, capacity_ah, *, battery_id="cell") -> DiagnosticResult:
        x = np.asarray(cycle_index, dtype=float).ravel()
        y = np.asarray(capacity_ah, dtype=float).ravel()
        features = capacity_features(
            x,
            y,
            nominal_capacity_ah=self.nominal_capacity_ah,
            slope_window=self.slope_window,
        )
        estimate = estimate_capacity_eol(
            x,
            y,
            eol_capacity_ah=self.eol_capacity_ah,
            nominal_capacity_ah=self.nominal_capacity_ah,
            minimum_observations=self.minimum_observations,
            slope_window=self.slope_window,
        )

        evidence = [Evidence(
            source="capacity_trajectory",
            statement=(
                f"SOH={features['soh'] if features['soh'] is not None else float('nan'):.3f}; "
                f"local slope={features['local_slope_ah_per_cycle'] if features['local_slope_ah_per_cycle'] is not None else float('nan'):.6f} Ah/cycle."
            ),
            score=float(np.clip(1.0 - float(features['soh'] or 0.0), 0.0, 1.0)),
            evidence_id="battery-prognosis-capacity",
            kind="prognostic",
            provenance={"battery_id": battery_id, "observed_cycles": int(len(x))},
            details=features,
        )]
        trace = [ToolTraceStep("capacity_features", evidence_ids=["battery-prognosis-capacity"])]

        if estimate.get("abstained", True):
            return standardize_result(DiagnosticResult(
                domain="battery",
                task="prognosis",
                decision="abstain",
                detection=DetectionResult(
                    abnormal=None if features["soh"] is None else bool(float(features["soh"]) <= 0.8),
                    score=None if features["soh"] is None else float(np.clip(1.0 - float(features["soh"]), 0.0, 1.0)),
                    method="capacity_soh",
                ),
                evidence=evidence,
                prognosis=PrognosisResult(
                    risk=None,
                    horizon="cycles_to_1.4Ah_EOL",
                    remaining_useful_life=None,
                    uncertainty=None,
                    details=estimate,
                ),
                confidence=0.2,
                uncertainty=0.8,
                abstained=True,
                abstain_reason=str(estimate.get("abstain_reason")),
                recommended_actions=["Collect more discharge history before issuing a point RUL forecast."],
                tool_trace=trace + [ToolTraceStep("estimate_capacity_eol", status="warning")],
                metadata={"battery_id": battery_id},
            ))

        rul = float(estimate["remaining_useful_life_cycles"])
        uncertainty_cycles = float(estimate["uncertainty_cycles"])
        relative_uncertainty = uncertainty_cycles / max(rul + uncertainty_cycles, 1.0)
        confidence = float(np.clip(1.0 - relative_uncertainty, 0.0, 1.0))
        verification_status = "SUPPORTED" if float(features["local_slope_ah_per_cycle"]) < 0 else "INSUFFICIENT"

        return standardize_result(DiagnosticResult(
            domain="battery",
            task="prognosis",
            decision="monitor",
            detection=DetectionResult(
                abnormal=bool(float(features["soh"]) <= 0.8),
                score=float(np.clip(1.0 - float(features["soh"]), 0.0, 1.0)),
                method="capacity_soh",
                details={"soh": features["soh"]},
            ),
            hypotheses=[DiagnosticHypothesis(
                label="capacity_fade",
                score=confidence,
                rationale="Observed discharge-capacity history supports a degrading trajectory.",
                evidence_ids=["battery-prognosis-capacity"],
                details={"rul_cycles": rul},
            )],
            evidence=evidence,
            verification=[VerificationResult(
                hypothesis="capacity_fade",
                status=verification_status,
                reason="Recent capacity slope is negative and supports forward EOL extrapolation.",
                evidence_ids=["battery-prognosis-capacity"],
                verifier="capacity_trajectory_physics",
            )],
            prognosis=PrognosisResult(
                risk=float(estimate["risk"]),
                horizon="cycles_to_1.4Ah_EOL",
                remaining_useful_life=rul,
                uncertainty=float(np.clip(relative_uncertainty, 0.0, 1.0)),
                details={
                    "predicted_eol_cycle": float(estimate["predicted_eol_cycle"]),
                    "eol_interval_cycles": list(estimate["eol_interval_cycles"]),
                    "uncertainty_cycles": uncertainty_cycles,
                    "method": estimate["method"],
                },
            ),
            confidence=confidence,
            uncertainty=float(np.clip(relative_uncertainty, 0.0, 1.0)),
            tool_trace=trace + [ToolTraceStep("estimate_capacity_eol", evidence_ids=["battery-prognosis-capacity"])],
            metadata={"battery_id": battery_id, "features": features},
        ))
