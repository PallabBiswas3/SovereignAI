from tsdiag.tools.root_rank_calibration import (
    FEATURE_NAMES,
    RootRankerConfig,
    cross_validate_root_ranker,
    evaluate_root_ranker,
    score_root_candidates,
)


def _record(fault_id, root, wrong, root_features, wrong_features):
    return {
        "fault_id": fault_id,
        "expected_roots": [root],
        "candidates": [
            {"variable": root, **root_features},
            {"variable": wrong, **wrong_features},
        ],
    }


def _features(contrib, shift, onset, topology, type_score, catalog):
    return dict(
        contribution_score=contrib,
        pre_post_shift_score=shift,
        onset_earliness_score=onset,
        topology_upstreamness_score=topology,
        fault_type_agreement_score=type_score,
        catalog_root_prior_score=catalog,
    )


def test_score_root_candidates_uses_requested_six_terms():
    rows = [
        {"variable": "root", **_features(0.4, 0.8, 0.7, 1.0, 0.8, 0.9)},
        {"variable": "symptom", **_features(1.0, 1.0, 0.5, 0.1, 0.2, 0.2)},
    ]
    weights = {name: 1.0 for name in FEATURE_NAMES}
    ranked = score_root_candidates(rows, weights)
    assert ranked[0].variable == "root"
    assert set(ranked[0].features) == set(FEATURE_NAMES)


def test_evaluate_root_ranker_reports_safety_metrics():
    records = [
        _record(
            1,
            "A",
            "B",
            _features(0.5, 0.7, 0.8, 1.0, 0.7, 0.9),
            _features(0.9, 0.9, 0.6, 0.0, 0.1, 0.1),
        )
    ]
    cfg = RootRankerConfig(weights={
        "contribution_score": 0.0,
        "pre_post_shift_score": 0.0,
        "onset_earliness_score": 0.0,
        "topology_upstreamness_score": 0.5,
        "fault_type_agreement_score": 0.0,
        "catalog_root_prior_score": 0.5,
    })
    metrics = evaluate_root_ranker(records, cfg)
    assert metrics["top1_root_accuracy"] == 1.0
    assert metrics["top3_root_accuracy"] == 1.0
    assert metrics["false_confident_rate"] == 0.0


def test_cross_validation_holds_out_fault_ids():
    records = []
    for fault_id in range(1, 9):
        records.append(
            _record(
                fault_id,
                f"R{fault_id}",
                f"W{fault_id}",
                _features(0.3, 0.5, 0.7, 1.0, 0.7, 1.0),
                _features(1.0, 0.9, 0.5, 0.0, 0.1, 0.0),
            )
        )
    result = cross_validate_root_ranker(
        records,
        n_splits=4,
        weight_step=0.25,
        margin_grid=(0.0, 0.05),
        max_abstention_rate=0.35,
        max_false_confident_rate=0.50,
    )
    assert len(result["folds"]) == 4
    assert result["cv_metrics"]["cases"] == 8
    assert result["cv_metrics"]["top1_root_accuracy"] == 1.0
    for fold in result["folds"]:
        assert not (set(fold["validation_fault_ids"]) & set(fold["train_fault_ids"]))
