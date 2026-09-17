import numpy as np

from tsdiag.tools import (
    knowledge_guided_root_cause_decision,
    rank_tep_fault_catalog,
    temporal_fault_type_evidence,
)


def test_catalog_fault_type_prior_separates_shared_root_neighbors():
    catalog = {
        "faults": [
            {
                "fault_id": 5,
                "fault_label": "condenser step",
                "type": "step",
                "expected_roots": ["XMV(11)"],
                "affected_variables": ["XMEAS(11)"],
            },
            {
                "fault_id": 12,
                "fault_label": "condenser random variation",
                "type": "random variation",
                "expected_roots": ["XMV(11)"],
                "affected_variables": ["XMEAS(11)"],
            },
        ]
    }

    out = rank_tep_fault_catalog(
        ["XMV(11)", "XMEAS(11)"],
        catalog,
        fault_type_scores={"random_variation": 1.0, "step": 0.0},
    )

    assert out["best_match"]["fault_id"] == 12
    assert out["best_match"]["fault_type_score"] == 1.0


def test_knowledge_guided_decision_abstains_on_downstream_only_match():
    catalog = {
        "faults": [
            {
                "fault_id": 5,
                "fault_label": "condenser cooling fault",
                "type": "step",
                "expected_roots": ["XMV(11)"],
                "affected_variables": ["XMEAS(11)"],
            }
        ]
    }

    kg = knowledge_guided_root_cause_decision(
        [{"variable": "XMEAS(11)", "score": 0.8}],
        catalog,
        shift_scores={"XMEAS(11)": 1.0},
        contribution_scores={"XMEAS(11)": 1.0},
        onset_order=["XMEAS(11)"],
        fault_type_scores={"step": 0.2},
        threshold=0.18,
    )

    assert kg["root_cause"] is None
    assert kg["fault_id"] is None
    assert kg["abstain_reason"] == "catalog_root_not_directly_supported"


def test_temporal_fault_type_evidence_returns_bounded_scores():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(120, 2))
    x[60:, 0] += 4.0
    alarm = np.arange(120) >= 60

    evidence = temporal_fault_type_evidence(
        x,
        ["XMEAS(1)", "XMV(1)"],
        alarm_mask=alarm,
    )

    expected = {
        "step",
        "random_variation",
        "slow_drift",
        "valve_sticking",
        "constant_position",
        "unknown",
    }
    assert expected.issubset(evidence["fault_type_scores"])
    assert all(0.0 <= value <= 1.0 for value in evidence["fault_type_scores"].values())
    assert evidence["top_channels"]
