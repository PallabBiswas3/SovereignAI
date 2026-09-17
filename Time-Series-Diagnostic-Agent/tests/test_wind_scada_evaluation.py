import numpy as np

from tsdiag.evaluation.wind_scada import (
    calculate_criticality,
    evaluate_wind_event,
    summarize_wind_events,
    wind_event_evidence_decision,
)


def test_criticality_increases_and_decays_only_on_normal_operation():
    alarm = np.array([1, 1, 0, 1, 0, 0], dtype=bool)
    normal = np.array([1, 1, 1, 0, 0, 1], dtype=bool)
    assert calculate_criticality(alarm, normal).tolist() == [1, 2, 1, 1, 1, 0]


def test_event_evidence_requires_sustained_residual_and_drift_corroboration():
    residual = np.zeros(100, dtype=bool)
    drift = np.zeros(100, dtype=bool)
    residual[30:45] = True
    drift[35:48] = True

    result = wind_event_evidence_decision(
        residual_alarm_mask=residual,
        drift_alarm_mask=drift,
        minimum_corroborated_run=6,
        minimum_residual_fraction=0.02,
        minimum_drift_fraction=0.02,
    )

    assert result.decision == "fault"
    assert result.event_detected is True
    assert result.longest_corroborated_run == 10
    assert result.first_decision_index == 40


def test_event_evidence_does_not_promote_single_detector_alarm():
    residual = np.zeros(100, dtype=bool)
    drift = np.zeros(100, dtype=bool)
    residual[20:60] = True

    result = wind_event_evidence_decision(
        residual_alarm_mask=residual,
        drift_alarm_mask=drift,
        minimum_corroborated_run=6,
    )

    assert result.decision == "monitor"
    assert result.event_detected is False
    assert result.longest_corroborated_run == 0


def test_event_evidence_abstains_on_out_of_distribution_operation():
    residual = np.zeros(100, dtype=bool)
    drift = np.zeros(100, dtype=bool)
    residual[20:60] = True
    drift[20:60] = True
    ood = np.zeros(100, dtype=bool)
    ood[:20] = True

    result = wind_event_evidence_decision(
        residual_alarm_mask=residual,
        drift_alarm_mask=drift,
        out_of_distribution_mask=ood,
        ood_abstain_fraction=0.10,
    )

    assert result.decision == "abstain"
    assert result.event_detected is False
    assert result.out_of_distribution_fraction == 0.20


def test_event_evaluation_reports_evidence_detection_and_legacy_criticality():
    timestamps = np.arange(
        np.datetime64("2024-01-01T00:00"),
        np.datetime64("2024-01-01T03:20"),
        np.timedelta64(10, "m"),
    )
    alarm = np.zeros(len(timestamps), dtype=bool)
    alarm[5:15] = True
    result = evaluate_wind_event(
        event_id=4,
        is_anomaly_event=True,
        alarm_mask=alarm,
        residual_alarm_mask=alarm,
        drift_alarm_mask=alarm,
        timestamps=timestamps,
        event_start=np.datetime64("2024-01-01T01:00"),
        event_end=np.datetime64("2024-01-01T02:30"),
        criticality_threshold=3,
        minimum_corroborated_run=3,
        sample_period_minutes=10,
    )
    assert result.event_detected is True
    assert result.event_decision == "fault"
    assert result.legacy_criticality_detected is True
    assert result.first_detection_index == 7
    assert result.time_to_event_end_minutes == 80.0
    assert result.early_warning_lead_time_minutes == -10.0
    assert result.event_window_recall is not None


def test_event_summary_separates_primary_legacy_and_abstention_metrics():
    rows = [
        {
            "event_id": 1,
            "is_anomaly_event": True,
            "event_detected": True,
            "legacy_criticality_detected": True,
            "abstained": False,
            "time_to_event_end_minutes": 60.0,
            "early_warning_lead_time_minutes": 20.0,
            "event_window_recall": 0.8,
            "evidence_score": 0.8,
            "out_of_distribution_fraction": 0.0,
        },
        {
            "event_id": 2,
            "is_anomaly_event": True,
            "event_detected": False,
            "legacy_criticality_detected": False,
            "abstained": True,
            "time_to_event_end_minutes": None,
            "early_warning_lead_time_minutes": None,
            "event_window_recall": 0.2,
            "evidence_score": 0.4,
            "out_of_distribution_fraction": 0.2,
        },
        {
            "event_id": 3,
            "is_anomaly_event": False,
            "event_detected": False,
            "legacy_criticality_detected": False,
            "abstained": False,
            "time_to_event_end_minutes": None,
            "early_warning_lead_time_minutes": None,
            "event_window_recall": None,
            "evidence_score": 0.2,
            "out_of_distribution_fraction": 0.0,
        },
        {
            "event_id": 4,
            "is_anomaly_event": False,
            "event_detected": True,
            "legacy_criticality_detected": True,
            "abstained": False,
            "time_to_event_end_minutes": None,
            "early_warning_lead_time_minutes": None,
            "event_window_recall": None,
            "evidence_score": 0.7,
            "out_of_distribution_fraction": 0.0,
        },
    ]
    summary = summarize_wind_events(rows)
    assert summary["event_recall"] == 0.5
    assert summary["normal_event_false_alarm_rate"] == 0.5
    assert summary["abstention_rate"] == 0.25
    assert summary["covered_anomaly_recall"] == 1.0
    assert summary["legacy_criticality_recall"] == 0.5
    assert summary["legacy_normal_event_false_alarm_rate"] == 0.5
    assert summary["mean_event_window_recall"] == 0.5
    assert summary["mean_time_to_event_end_minutes"] == 60.0
    assert summary["mean_early_warning_lead_time_minutes"] == 20.0
