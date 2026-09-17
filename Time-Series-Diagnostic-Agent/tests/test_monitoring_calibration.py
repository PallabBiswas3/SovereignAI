import numpy as np

from tsdiag.evaluation.reporting import write_fault_table, write_markdown_summary
from tsdiag.tools import (
    calibrate_monitoring_config,
    dpca_monitoring,
    pre_post_shift_evidence,
    root_cause_rank_enhanced,
    run_monitoring_method,
    standardize_against_normal,
)


def test_alarm_calibration_controls_healthy_false_alarms():
    rng = np.random.default_rng(10)
    ref = rng.normal(size=(500, 4))
    std = standardize_against_normal(ref, ref)
    calibrated = calibrate_monitoring_config(
        std["standardized_reference"],
        method="pca",
        target_false_alarm_rate=0.05,
        alpha_grid=(0.99, 0.995, 0.999),
        persistence_grid=(1, 2, 3),
    )
    cfg = calibrated["config"]
    monitored = run_monitoring_method(std["standardized_reference"], std["standardized_reference"], cfg)
    assert monitored["alarm_mask"].mean() < 0.12


def test_dpca_detects_autocorrelated_shift():
    rng = np.random.default_rng(11)
    n = 450
    ref = rng.normal(size=(n, 3))
    cur = rng.normal(size=(n, 3))
    cur[220:, 1] += 2.5
    std = standardize_against_normal(cur, ref)
    result = dpca_monitoring(
        std["standardized_signal"],
        std["standardized_reference"],
        lags=2,
        alpha=0.99,
        min_consecutive=2,
    )
    assert result["alarm_mask"].shape == (n,)
    assert result["alarm_mask"][230:].mean() > 0.4


def test_enhanced_root_cause_uses_shift_evidence():
    graph = {"edges": [{"cause": "A", "effect": "B", "p_value": 0.001}]}
    ranked = root_cause_rank_enhanced(
        ["B"],
        graph,
        ["A", "B"],
        variable_contributions=np.array([0.2, 0.8]),
        shift_scores={"A": 1.0, "B": 0.4},
        channel_names=["A", "B"],
    )
    assert ranked["root_cause_ranking"][0]["variable"] == "A"
    assert ranked["propagation_paths"] == [["A", "B"]]


def test_pre_post_shift_evidence_ranks_shifted_channel():
    rng = np.random.default_rng(12)
    x = rng.normal(size=(300, 3))
    x[120:, 2] += 4.0
    alarm = np.zeros(300, dtype=bool)
    alarm[120:] = True
    result = pre_post_shift_evidence(x, ["a", "b", "c"], alarm_mask=alarm)
    assert result["ranked_variables"][0] == "c"


def test_reporting_writes_csv_and_markdown(tmp_path):
    rows = [
        {
            "method": "pca",
            "fault_id": 1,
            "pre_fault_false_alarm_rate": 0.02,
            "post_fault_detection_rate": 0.9,
        }
    ]
    csv_path = write_fault_table(rows, tmp_path / "table.csv")
    md_path = write_markdown_summary({"pca": {"mean": 1.0}}, rows, tmp_path / "summary.md")
    assert csv_path.exists()
    assert md_path.exists()
    assert "fault_id" in csv_path.read_text()
    assert "Aggregate metrics" in md_path.read_text()
