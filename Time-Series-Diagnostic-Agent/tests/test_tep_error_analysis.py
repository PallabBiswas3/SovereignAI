from tsdiag.evaluation.tep_error_analysis import analyze_root_cause_errors


def _candidate(variable, contribution, shift, onset, topology, type_score, catalog):
    return {
        "variable": variable,
        "contribution_score": contribution,
        "pre_post_shift_score": shift,
        "onset_earliness_score": onset,
        "topology_upstreamness_score": topology,
        "fault_type_agreement_score": type_score,
        "catalog_root_prior_score": catalog,
    }


def test_error_analysis_reports_rank_gap_and_category():
    records = [
        {
            "fault_id": 1,
            "expected_roots": ["A"],
            "candidates": [
                _candidate("A", 0.3, 0.5, 0.5, 1.0, 0.7, 0.8),
                _candidate("B", 1.0, 1.0, 0.5, 0.2, 0.2, 0.2),
            ],
        }
    ]
    held_out = [
        {
            "fault_id": 1,
            "expected_roots": ["A"],
            "raw_top_root": "B",
            "predicted_root": "B",
            "abstained": False,
            "score_margin": 0.2,
            "ranking": [
                {"variable": "B", "score": 0.7},
                {"variable": "A", "score": 0.5},
            ],
        }
    ]
    weights = {
        "contribution_score": 0.2,
        "pre_post_shift_score": 0.2,
        "onset_earliness_score": 0.1,
        "topology_upstreamness_score": 0.2,
        "fault_type_agreement_score": 0.1,
        "catalog_root_prior_score": 0.2,
    }
    report = analyze_root_cause_errors(records, held_out, weights)
    assert report["error_count"] == 1
    row = report["errors"][0]
    assert row["true_root_rank"] == 2
    assert row["top_wrong_variable"] == "B"
    assert abs(row["root_support_gap"] - 0.2) < 1e-9
    assert row["primary_failure_cause"] == "DOWNSTREAM_OVER_WEIGHTING"


def test_weak_signal_is_identified_first():
    records = [
        {
            "fault_id": 2,
            "expected_roots": ["A"],
            "candidates": [
                _candidate("A", 0.2, 0.1, 0.4, 0.8, 0.5, 0.5),
                _candidate("B", 0.8, 0.7, 0.6, 0.2, 0.5, 0.5),
            ],
        }
    ]
    held_out = [{
        "fault_id": 2,
        "raw_top_root": "B",
        "predicted_root": "B",
        "abstained": False,
        "score_margin": 0.1,
        "ranking": [
            {"variable": "B", "score": 0.6},
            {"variable": "A", "score": 0.5},
        ],
    }]
    report = analyze_root_cause_errors(records, held_out, {"pre_post_shift_score": 1.0})
    assert report["errors"][0]["primary_failure_cause"] == "WEAK_SIGNAL"


def test_screened_out_true_root_is_not_silently_dropped():
    records = [{
        "fault_id": 9,
        "expected_roots": ["TRUE_ROOT"],
        "candidates": [
            _candidate("WRONG", 1.0, 1.0, 0.8, 0.3, 0.2, 0.1),
        ],
    }]
    held_out = [{
        "fault_id": 9,
        "raw_top_root": "WRONG",
        "predicted_root": "WRONG",
        "abstained": False,
        "score_margin": 0.3,
        "ranking": [{"variable": "WRONG", "score": 0.8}],
    }]
    report = analyze_root_cause_errors(records, held_out, {"pre_post_shift_score": 1.0})
    assert report["error_count"] == 1
    assert report["candidate_omission_count"] == 1
    row = report["errors"][0]
    assert row["true_root"] == "TRUE_ROOT"
    assert row["true_root_in_candidates"] is False
    assert row["true_root_rank"] is None
    assert row["primary_failure_cause"] == "WEAK_SIGNAL"
