from __future__ import annotations

from collections import Counter

import numpy as np

from adaptivefact.benchmark.metrics import calibration_metrics, classification_metrics, latency_percentiles
from adaptivefact.data.schema import ResponseRecord, VerificationRoute
from adaptivefact.routing.router import AdaptiveRouter


def evaluate_router(router: AdaptiveRouter, records: list[ResponseRecord]) -> dict:
    y_true = []
    calibrated = []
    latencies = []
    routes = []

    for record in records:
        decision = router.route(record)
        truth = int(record.ground_truth_label is not None and record.ground_truth_label.value == "hallucinated")
        y_true.append(truth)
        calibrated.append(decision.calibrated_risk)
        latencies.append(decision.latency_ms)
        routes.append(decision.route)

    y = np.asarray(y_true, dtype=int)
    p = np.asarray(calibrated, dtype=float)
    fast = np.asarray([r == VerificationRoute.FAST_ACCEPT for r in routes])
    agentic = np.asarray([r == VerificationRoute.AGENTIC for r in routes])
    needs_verification = ~fast

    counts = Counter(r.value for r in routes)
    total_h = int(y.sum())
    missed_h = int(y[fast].sum()) if fast.any() else 0

    binary_metrics = classification_metrics(
        y.tolist(),
        needs_verification.astype(int).tolist(),
        p.tolist(),
    )

    return {
        "n": len(records),
        "hallucination_rate": float(y.mean()) if len(y) else float("nan"),
        "thresholds": {"tau1": router.tau1, "tau2": router.tau2},
        "route_counts": dict(counts),
        "route_fractions": {key: value / len(records) for key, value in counts.items()} if records else {},
        "fast_path_hallucination_rate": float(y[fast].mean()) if fast.any() else 0.0,
        "hallucination_recall_outside_fast_path": float((total_h - missed_h) / total_h) if total_h else 1.0,
        "lightweight_hallucination_rate": float(y[np.asarray([r == VerificationRoute.LIGHTWEIGHT for r in routes])].mean())
        if any(r == VerificationRoute.LIGHTWEIGHT for r in routes)
        else 0.0,
        "agentic_hallucination_rate": float(y[agentic].mean()) if agentic.any() else 0.0,
        "needs_verification_binary_metrics": binary_metrics,
        "calibration": calibration_metrics(y.tolist(), p.tolist()),
        "latency_ms": latency_percentiles(latencies),
    }
