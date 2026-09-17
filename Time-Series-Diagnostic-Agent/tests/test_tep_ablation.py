import numpy as np

from tsdiag.datasets.tep_knowledge import tep_fault_catalog, tep_topology
from tsdiag.evaluation.tep_ablation import run_tep_root_cause_ablation, summarize_tep_ablation


def test_summarize_tep_ablation_reports_variant_metrics():
    rows = [
        {
            "fault_id": 1,
            "variant": "generic",
            "root_hit": False,
            "top3_root_hit": True,
            "fault_id_hit": None,
            "false_confident": True,
            "confidence": 0.8,
            "abstain_reason": None,
        },
        {
            "fault_id": 1,
            "variant": "topology_catalog",
            "root_hit": True,
            "top3_root_hit": True,
            "fault_id_hit": True,
            "false_confident": False,
            "confidence": 0.7,
            "abstain_reason": None,
        },
    ]

    summary = summarize_tep_ablation(rows)

    assert summary["generic"]["root_accuracy"] == 0.0
    assert summary["generic"]["top3_root_accuracy"] == 1.0
    assert summary["generic"]["false_confident_rate"] == 1.0
    assert summary["topology_catalog"]["root_accuracy"] == 1.0
    assert summary["topology_catalog"]["fault_id_accuracy"] == 1.0


def test_run_tep_root_cause_ablation_returns_all_variants():
    rng = np.random.default_rng(7)
    n = 140
    names = ["XMV(10)", "XMEAS(21)", "XMEAS(9)", "XMEAS(7)"]
    reference = rng.normal(size=(n, len(names)))
    current = rng.normal(size=(n, len(names)))
    current[65:, 0] += 4.0
    current[70:, 1] += 3.0
    current[75:, 2] += 2.0
    current[80:, 3] += 1.0
    alarm = np.zeros(n, dtype=bool)
    alarm[65:] = True

    result = run_tep_root_cause_ablation(
        current,
        reference,
        names,
        alarm,
        fault_id=11,
        expected_roots=["XMV(10)", "XMEAS(21)", "XMEAS(9)"],
        process_topology=tep_topology(),
        fault_catalog=tep_fault_catalog(),
        maxlag=1,
        confidence_threshold=0.05,
    )

    variants = {row["variant"] for row in result["rows"]}
    assert variants == {"generic", "topology_only", "catalog_only", "topology_catalog"}
    assert all("top3_roots" in row for row in result["rows"])
    assert "generic_edge_count" in result["artifacts"]
    assert "topology_edge_count" in result["artifacts"]
