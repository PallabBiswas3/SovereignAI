from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class RoutingThresholds:
    tau1: float
    tau2: float
    fast_path_fraction: float
    fast_path_hallucination_rate: float
    hallucination_recall_outside_fast_path: float
    agentic_fraction: float
    agentic_hallucination_rate: float
    lightweight_fraction: float

    def to_dict(self) -> dict:
        return asdict(self)


def routing_statistics(y_true, probabilities, tau1: float, tau2: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) == 0:
        raise ValueError("Cannot compute routing statistics for an empty set")
    if not 0.0 <= tau1 < tau2 <= 1.0:
        raise ValueError("Expected 0 <= tau1 < tau2 <= 1")

    fast = p < tau1
    agentic = p >= tau2
    lightweight = ~(fast | agentic)

    total_h = int(y.sum())
    fast_h = int(y[fast].sum()) if fast.any() else 0

    return {
        "n": int(len(y)),
        "tau1": float(tau1),
        "tau2": float(tau2),
        "fast_path_fraction": float(fast.mean()),
        "fast_path_count": int(fast.sum()),
        "fast_path_hallucination_rate": float(y[fast].mean()) if fast.any() else 0.0,
        "hallucination_recall_outside_fast_path": float((total_h - fast_h) / total_h) if total_h else 1.0,
        "lightweight_fraction": float(lightweight.mean()),
        "lightweight_count": int(lightweight.sum()),
        "lightweight_hallucination_rate": float(y[lightweight].mean()) if lightweight.any() else 0.0,
        "agentic_fraction": float(agentic.mean()),
        "agentic_count": int(agentic.sum()),
        "agentic_hallucination_rate": float(y[agentic].mean()) if agentic.any() else 0.0,
    }


def _candidate_thresholds(probabilities: np.ndarray, points: int = 300) -> np.ndarray:
    p = np.clip(np.asarray(probabilities, dtype=float), 0.0, 1.0)
    quantiles = np.linspace(0.0, 1.0, max(20, points))
    values = np.quantile(p, quantiles)
    return np.unique(np.concatenate(([0.0], values, [1.0])))


def optimize_routing_thresholds(
    y_true,
    probabilities,
    *,
    max_fast_hallucination_rate: float = 0.02,
    min_hallucination_recall_outside_fast: float = 0.98,
    min_fast_count: int = 25,
    min_agentic_hallucination_rate: float = 0.75,
    min_agentic_count: int = 25,
    grid_points: int = 300,
) -> RoutingThresholds:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if len(y) != len(p) or len(y) == 0:
        raise ValueError("y_true and probabilities must be non-empty and have the same length")

    candidates = _candidate_thresholds(p, grid_points)

    best_tau1 = None
    best_fast_fraction = -1.0
    for tau1 in candidates:
        fast = p < tau1
        count = int(fast.sum())
        if count < min_fast_count:
            continue
        fast_rate = float(y[fast].mean()) if count else 0.0
        total_h = int(y.sum())
        missed_h = int(y[fast].sum()) if count else 0
        retained = float((total_h - missed_h) / total_h) if total_h else 1.0
        if fast_rate <= max_fast_hallucination_rate and retained >= min_hallucination_recall_outside_fast:
            fraction = count / len(y)
            if fraction > best_fast_fraction:
                best_fast_fraction = fraction
                best_tau1 = float(tau1)

    if best_tau1 is None:
        best_tau1 = float(np.min(p))

    best_tau2 = None
    best_agentic_fraction = -1.0
    for tau2 in candidates:
        if tau2 <= best_tau1:
            continue
        agentic = p >= tau2
        count = int(agentic.sum())
        if count < min_agentic_count:
            continue
        rate = float(y[agentic].mean())
        if rate >= min_agentic_hallucination_rate:
            fraction = count / len(y)
            if fraction > best_agentic_fraction:
                best_agentic_fraction = fraction
                best_tau2 = float(tau2)

    if best_tau2 is None:
        higher = candidates[candidates > best_tau1]
        if len(higher) == 0:
            best_tau2 = min(1.0, best_tau1 + 1e-6)
        else:
            target = float(np.quantile(p, 0.85))
            viable = higher[higher >= target]
            best_tau2 = float(viable[0] if len(viable) else higher[-1])

    if best_tau2 <= best_tau1:
        best_tau2 = min(1.0, best_tau1 + 1e-6)

    stats = routing_statistics(y, p, best_tau1, best_tau2)
    return RoutingThresholds(
        tau1=best_tau1,
        tau2=best_tau2,
        fast_path_fraction=stats["fast_path_fraction"],
        fast_path_hallucination_rate=stats["fast_path_hallucination_rate"],
        hallucination_recall_outside_fast_path=stats["hallucination_recall_outside_fast_path"],
        agentic_fraction=stats["agentic_fraction"],
        agentic_hallucination_rate=stats["agentic_hallucination_rate"],
        lightweight_fraction=stats["lightweight_fraction"],
    )
