from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Literal

import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score


@dataclass
class WindEventEvidenceDecision:
    """Label-free event-level decision from diagnostic evidence.

    ``evidence_score`` is a bounded heuristic evidence-strength score, not a
    calibrated probability. Event labels are never consumed here.
    """

    decision: Literal["fault", "monitor", "abstain"]
    event_detected: bool
    evidence_score: float
    confidence: float
    residual_alarm_fraction: float
    drift_alarm_fraction: float
    corroborated_alarm_fraction: float
    longest_residual_run: int
    longest_drift_run: int
    longest_corroborated_run: int
    out_of_distribution_fraction: float
    physics_support: float
    first_decision_index: int | None
    reason: str


@dataclass
class WindEventEvaluation:
    event_id: int
    is_anomaly_event: bool
    event_detected: bool
    event_decision: str
    evidence_score: float
    decision_confidence: float
    abstained: bool
    decision_reason: str
    legacy_criticality_detected: bool
    max_criticality: int
    alarm_fraction: float
    residual_alarm_fraction: float
    drift_alarm_fraction: float
    corroborated_alarm_fraction: float
    longest_corroborated_run: int
    out_of_distribution_fraction: float
    event_window_recall: float | None
    event_window_precision: float | None
    first_detection_index: int | None
    time_to_event_end_samples: int | None
    time_to_event_end_minutes: float | None
    early_warning_lead_time_samples: int | None
    early_warning_lead_time_minutes: float | None


def calculate_criticality(
    alarm_mask,
    normal_mask=None,
    *,
    initial: int = 0,
    maximum: int = 1000,
) -> np.ndarray:
    """CARE-style bounded cumulative criticality.

    Criticality is retained as a benchmark/reporting feature for historical
    comparability. It is no longer the primary event classifier because long
    prediction windows can accumulate criticality from scattered false alarms.
    """
    alarm = np.asarray(alarm_mask, dtype=bool).ravel()
    if normal_mask is None:
        normal = np.ones_like(alarm, dtype=bool)
    else:
        normal = np.asarray(normal_mask, dtype=bool).ravel()
        if normal.shape != alarm.shape:
            raise ValueError("normal_mask length mismatch")
    out = np.empty(alarm.size, dtype=int)
    value = int(initial)
    for i, (flag, is_normal) in enumerate(zip(alarm, normal)):
        delta = 1 if is_normal and flag else (-1 if is_normal and not flag else 0)
        value = max(0, min(int(maximum), value + delta))
        out[i] = value
    return out


def _longest_true_run(mask: np.ndarray) -> tuple[int, int | None]:
    """Return longest run length and the index where that run first qualifies."""
    longest = 0
    current = 0
    first_qualifying_end = None
    for i, value in enumerate(np.asarray(mask, dtype=bool).ravel()):
        if value:
            current += 1
            if current > longest:
                longest = current
        else:
            current = 0
    return int(longest), first_qualifying_end


def _first_run_end(mask: np.ndarray, minimum_run: int) -> int | None:
    run = 0
    required = max(1, int(minimum_run))
    for i, value in enumerate(np.asarray(mask, dtype=bool).ravel()):
        run = run + 1 if value else 0
        if run >= required:
            return int(i)
    return None


def wind_event_evidence_decision(
    *,
    residual_alarm_mask,
    drift_alarm_mask,
    evaluable_mask=None,
    out_of_distribution_mask=None,
    physics_finding_count: int = 0,
    minimum_corroborated_run: int = 6,
    minimum_residual_fraction: float = 0.02,
    minimum_drift_fraction: float = 0.02,
    ood_abstain_fraction: float = 0.10,
) -> WindEventEvidenceDecision:
    """Convert sample-level evidence into a conservative event decision.

    Design choices are intentionally label-free:

    * a fault requires simultaneous persistent residual + CUSUM evidence;
    * scattered alarms from different times do not create a fault decision;
    * OOD operation is uncertainty, not fault evidence. The regime support uses a
      healthy-reference p99 distance threshold, so >10% OOD samples represents a
      large departure from the expected ~1% reference exceedance rate;
    * physics findings add only a small evidence-strength contribution because
      missing/anonymized turbine specifications can make those checks incomplete.

    The defaults are a transparent baseline and must later be calibrated only on
    healthy/reference data, never on CARE event labels.
    """
    residual = np.asarray(residual_alarm_mask, dtype=bool).ravel()
    drift = np.asarray(drift_alarm_mask, dtype=bool).ravel()
    if residual.shape != drift.shape:
        raise ValueError("residual_alarm_mask and drift_alarm_mask length mismatch")

    if evaluable_mask is None:
        evaluable = np.ones_like(residual, dtype=bool)
    else:
        evaluable = np.asarray(evaluable_mask, dtype=bool).ravel()
        if evaluable.shape != residual.shape:
            raise ValueError("evaluable_mask length mismatch")

    residual_eval = residual & evaluable
    drift_eval = drift & evaluable
    corroborated = residual_eval & drift_eval

    denominator = max(int(np.sum(evaluable)), 1)
    residual_fraction = float(np.sum(residual_eval) / denominator)
    drift_fraction = float(np.sum(drift_eval) / denominator)
    corroborated_fraction = float(np.sum(corroborated) / denominator)

    longest_residual, _ = _longest_true_run(residual_eval)
    longest_drift, _ = _longest_true_run(drift_eval)
    longest_corroborated, _ = _longest_true_run(corroborated)

    if out_of_distribution_mask is None:
        ood_fraction = 0.0
    else:
        ood = np.asarray(out_of_distribution_mask, dtype=bool).ravel()
        if ood.shape != residual.shape:
            raise ValueError("out_of_distribution_mask length mismatch")
        ood_fraction = float(np.sum(ood & evaluable) / denominator)

    minimum_run = max(1, int(minimum_corroborated_run))
    physics_support = float(np.clip(int(physics_finding_count) / 3.0, 0.0, 1.0))

    # Score is descriptive/continuous so later healthy-only calibration can use
    # it. The rule-based decision below does not threshold this score.
    residual_strength = float(np.clip(residual_fraction / 0.10, 0.0, 1.0))
    drift_strength = float(np.clip(drift_fraction / 0.10, 0.0, 1.0))
    corroboration_strength = float(np.clip(corroborated_fraction / 0.05, 0.0, 1.0))
    run_strength = float(np.clip(longest_corroborated / minimum_run, 0.0, 1.0))
    evidence_score = (
        0.25 * residual_strength
        + 0.20 * drift_strength
        + 0.30 * corroboration_strength
        + 0.20 * run_strength
        + 0.05 * physics_support
    )
    evidence_score = float(np.clip(evidence_score, 0.0, 1.0))

    high_ood = ood_fraction > float(ood_abstain_fraction)
    sufficient_burden = (
        residual_fraction >= float(minimum_residual_fraction)
        and drift_fraction >= float(minimum_drift_fraction)
    )
    sustained_corroboration = longest_corroborated >= minimum_run

    if high_ood:
        decision = "abstain"
        detected = False
        first_index = None
        reason = (
            f"{ood_fraction:.1%} of evaluable samples are outside healthy regime support; "
            "the normal-behavior residual model is not trusted for an event diagnosis."
        )
        confidence = float(np.clip(ood_fraction / max(float(ood_abstain_fraction), 1e-12), 0.0, 1.0))
    elif sustained_corroboration and sufficient_burden:
        decision = "fault"
        detected = True
        first_index = _first_run_end(corroborated, minimum_run)
        reason = (
            f"Residual and CUSUM evidence corroborate for {longest_corroborated} consecutive samples "
            f"with residual/drift burdens {residual_fraction:.1%}/{drift_fraction:.1%}."
        )
        confidence = evidence_score
    else:
        decision = "monitor"
        detected = False
        first_index = None
        if longest_corroborated < minimum_run:
            reason = (
                f"No sustained corroborated episode: longest residual+CUSUM overlap is "
                f"{longest_corroborated} samples (need {minimum_run})."
            )
        else:
            reason = (
                "Corroborated evidence exists but event-level residual/drift burden is below "
                "the conservative baseline requirement."
            )
        confidence = float(np.clip(1.0 - evidence_score, 0.0, 1.0))

    return WindEventEvidenceDecision(
        decision=decision,
        event_detected=bool(detected),
        evidence_score=evidence_score,
        confidence=float(confidence),
        residual_alarm_fraction=residual_fraction,
        drift_alarm_fraction=drift_fraction,
        corroborated_alarm_fraction=corroborated_fraction,
        longest_residual_run=longest_residual,
        longest_drift_run=longest_drift,
        longest_corroborated_run=longest_corroborated,
        out_of_distribution_fraction=ood_fraction,
        physics_support=physics_support,
        first_decision_index=first_index,
        reason=reason,
    )


def event_window_mask(timestamps, event_start, event_end) -> np.ndarray:
    ts = np.asarray(timestamps).astype("datetime64[ns]")
    if event_start is None or event_end is None:
        return np.zeros(ts.shape[0], dtype=bool)
    start = np.datetime64(event_start, "ns")
    end = np.datetime64(event_end, "ns")
    return (ts >= start) & (ts <= end)


def evaluate_wind_event(
    *,
    event_id: int,
    is_anomaly_event: bool,
    alarm_mask,
    residual_alarm_mask=None,
    drift_alarm_mask=None,
    out_of_distribution_mask=None,
    physics_finding_count: int = 0,
    timestamps=None,
    event_start=None,
    event_end=None,
    normal_mask=None,
    criticality_threshold: int = 72,
    sample_period_minutes: float = 10.0,
    minimum_corroborated_run: int = 6,
    minimum_residual_fraction: float = 0.02,
    minimum_drift_fraction: float = 0.02,
    ood_abstain_fraction: float = 0.10,
) -> WindEventEvaluation:
    alarm = np.asarray(alarm_mask, dtype=bool).ravel()
    normal = np.ones_like(alarm) if normal_mask is None else np.asarray(normal_mask, dtype=bool).ravel()
    if normal.shape != alarm.shape:
        raise ValueError("normal_mask length mismatch")

    residual = alarm if residual_alarm_mask is None else np.asarray(residual_alarm_mask, dtype=bool).ravel()
    drift = alarm if drift_alarm_mask is None else np.asarray(drift_alarm_mask, dtype=bool).ravel()
    if residual.shape != alarm.shape or drift.shape != alarm.shape:
        raise ValueError("component alarm masks must match alarm_mask length")

    criticality = calculate_criticality(alarm, normal)
    max_criticality = int(np.max(criticality)) if criticality.size else 0
    legacy_detected = bool(max_criticality >= int(criticality_threshold))
    alarm_fraction = float(np.mean(alarm[normal])) if np.any(normal) else 0.0

    evidence_decision = wind_event_evidence_decision(
        residual_alarm_mask=residual,
        drift_alarm_mask=drift,
        evaluable_mask=normal,
        out_of_distribution_mask=out_of_distribution_mask,
        physics_finding_count=physics_finding_count,
        minimum_corroborated_run=minimum_corroborated_run,
        minimum_residual_fraction=minimum_residual_fraction,
        minimum_drift_fraction=minimum_drift_fraction,
        ood_abstain_fraction=ood_abstain_fraction,
    )

    window_recall = None
    window_precision = None
    first_detection = evidence_decision.first_decision_index
    time_to_end_samples = None
    time_to_end_minutes = None
    early_warning_samples = None
    early_warning_minutes = None

    if timestamps is not None and event_start is not None and event_end is not None:
        window = event_window_mask(timestamps, event_start, event_end)
        evaluable = normal.copy()
        truth = window & bool(is_anomaly_event)
        if np.any(evaluable):
            y_true = truth[evaluable].astype(int)
            y_pred = alarm[evaluable].astype(int)
            if is_anomaly_event:
                window_recall = float(recall_score(y_true, y_pred, zero_division=0))
                window_precision = float(precision_score(y_true, y_pred, zero_division=0))

        if first_detection is not None and is_anomaly_event:
            ts = np.asarray(timestamps).astype("datetime64[ns]")
            if first_detection < 0 or first_detection >= len(ts):
                raise ValueError("first_detection_index is outside the timestamp array")
            start = np.datetime64(event_start, "ns")
            end = np.datetime64(event_end, "ns")
            detected_time = ts[first_detection]
            time_to_end_minutes = float(
                (end - detected_time) / np.timedelta64(1, "m")
            )
            time_to_end_samples = int(
                round(time_to_end_minutes / float(sample_period_minutes))
            )
            # Positive means the diagnostic decision preceded event onset;
            # negative means detection occurred after onset (a detection delay).
            early_warning_minutes = float(
                (start - detected_time) / np.timedelta64(1, "m")
            )
            early_warning_samples = int(
                round(early_warning_minutes / float(sample_period_minutes))
            )

    return WindEventEvaluation(
        event_id=int(event_id),
        is_anomaly_event=bool(is_anomaly_event),
        event_detected=evidence_decision.event_detected,
        event_decision=evidence_decision.decision,
        evidence_score=evidence_decision.evidence_score,
        decision_confidence=evidence_decision.confidence,
        abstained=evidence_decision.decision == "abstain",
        decision_reason=evidence_decision.reason,
        legacy_criticality_detected=legacy_detected,
        max_criticality=max_criticality,
        alarm_fraction=alarm_fraction,
        residual_alarm_fraction=evidence_decision.residual_alarm_fraction,
        drift_alarm_fraction=evidence_decision.drift_alarm_fraction,
        corroborated_alarm_fraction=evidence_decision.corroborated_alarm_fraction,
        longest_corroborated_run=evidence_decision.longest_corroborated_run,
        out_of_distribution_fraction=evidence_decision.out_of_distribution_fraction,
        event_window_recall=window_recall,
        event_window_precision=window_precision,
        first_detection_index=first_detection,
        time_to_event_end_samples=time_to_end_samples,
        time_to_event_end_minutes=time_to_end_minutes,
        early_warning_lead_time_samples=early_warning_samples,
        early_warning_lead_time_minutes=early_warning_minutes,
    )


def summarize_wind_events(rows: Iterable[WindEventEvaluation | dict]) -> dict:
    records = [asdict(row) if isinstance(row, WindEventEvaluation) else dict(row) for row in rows]
    if not records:
        return {
            "event_count": 0,
            "event_recall": None,
            "event_precision": None,
            "normal_event_false_alarm_rate": None,
            "event_f1": None,
            "abstention_rate": None,
            "mean_time_to_event_end_minutes": None,
            "mean_early_warning_lead_time_minutes": None,
            "mean_max_criticality": None,
        }

    truth = np.asarray([bool(row["is_anomaly_event"]) for row in records], dtype=bool)
    pred = np.asarray([bool(row["event_detected"]) for row in records], dtype=bool)
    legacy_pred = np.asarray(
        [bool(row.get("legacy_criticality_detected", row["event_detected"])) for row in records],
        dtype=bool,
    )
    abstained = np.asarray([bool(row.get("abstained", False)) for row in records], dtype=bool)
    normal = ~truth
    covered = ~abstained

    time_to_end = [
        float(row.get("time_to_event_end_minutes", row.get("lead_time_minutes")))
        for row in records
        if row.get("time_to_event_end_minutes", row.get("lead_time_minutes")) is not None
        and bool(row["is_anomaly_event"])
    ]
    early_warning = [
        float(row["early_warning_lead_time_minutes"])
        for row in records
        if row.get("early_warning_lead_time_minutes") is not None
        and bool(row["is_anomaly_event"])
    ]
    coverage = [
        float(row["event_window_recall"])
        for row in records
        if row.get("event_window_recall") is not None and bool(row["is_anomaly_event"])
    ]
    max_criticality = [float(row.get("max_criticality", 0.0)) for row in records]
    anomaly_criticality = [
        float(row.get("max_criticality", 0.0)) for row in records if bool(row["is_anomaly_event"])
    ]
    normal_criticality = [
        float(row.get("max_criticality", 0.0)) for row in records if not bool(row["is_anomaly_event"])
    ]
    anomaly_scores = [
        float(row.get("evidence_score", 0.0)) for row in records if bool(row["is_anomaly_event"])
    ]
    normal_scores = [
        float(row.get("evidence_score", 0.0)) for row in records if not bool(row["is_anomaly_event"])
    ]
    anomaly_ood = [
        float(row.get("out_of_distribution_fraction", 0.0)) for row in records if bool(row["is_anomaly_event"])
    ]
    normal_ood = [
        float(row.get("out_of_distribution_fraction", 0.0)) for row in records if not bool(row["is_anomaly_event"])
    ]

    covered_anomaly = truth & covered
    return {
        "event_count": len(records),
        "anomaly_event_count": int(np.sum(truth)),
        "normal_event_count": int(np.sum(normal)),
        "event_recall": float(recall_score(truth, pred, zero_division=0)),
        "event_precision": float(precision_score(truth, pred, zero_division=0)),
        "event_f1": float(f1_score(truth, pred, zero_division=0)),
        "normal_event_false_alarm_rate": float(np.mean(pred[normal])) if np.any(normal) else None,
        "abstention_rate": float(np.mean(abstained)),
        "anomaly_event_abstention_rate": float(np.mean(abstained[truth])) if np.any(truth) else None,
        "normal_event_abstention_rate": float(np.mean(abstained[normal])) if np.any(normal) else None,
        "covered_anomaly_recall": (
            float(np.mean(pred[covered_anomaly])) if np.any(covered_anomaly) else None
        ),
        "legacy_criticality_recall": float(recall_score(truth, legacy_pred, zero_division=0)),
        "legacy_criticality_precision": float(precision_score(truth, legacy_pred, zero_division=0)),
        "legacy_criticality_f1": float(f1_score(truth, legacy_pred, zero_division=0)),
        "legacy_normal_event_false_alarm_rate": (
            float(np.mean(legacy_pred[normal])) if np.any(normal) else None
        ),
        "mean_event_window_recall": float(np.mean(coverage)) if coverage else None,
        "mean_time_to_event_end_minutes": float(np.mean(time_to_end)) if time_to_end else None,
        "median_time_to_event_end_minutes": float(np.median(time_to_end)) if time_to_end else None,
        "mean_early_warning_lead_time_minutes": (
            float(np.mean(early_warning)) if early_warning else None
        ),
        "median_early_warning_lead_time_minutes": (
            float(np.median(early_warning)) if early_warning else None
        ),
        "mean_evidence_score_anomaly": float(np.mean(anomaly_scores)) if anomaly_scores else None,
        "mean_evidence_score_normal": float(np.mean(normal_scores)) if normal_scores else None,
        "mean_ood_fraction_anomaly": float(np.mean(anomaly_ood)) if anomaly_ood else None,
        "mean_ood_fraction_normal": float(np.mean(normal_ood)) if normal_ood else None,
        "mean_max_criticality": float(np.mean(max_criticality)) if max_criticality else None,
        "mean_anomaly_event_criticality": float(np.mean(anomaly_criticality)) if anomaly_criticality else None,
        "mean_normal_event_criticality": float(np.mean(normal_criticality)) if normal_criticality else None,
    }
