from tsdiag.datasets.tep_knowledge import TEP_FAULT_SPECS, tep_fault_catalog, tep_topology
from tsdiag.tools.tep_reasoning import knowledge_guided_root_cause_decision, rank_tep_fault_catalog


def test_tep_fault_catalog_contains_known_mechanisms():
    catalog = tep_fault_catalog()
    assert len(catalog["faults"]) == 21
    idv4 = next(row for row in catalog["faults"] if row["fault_id"] == 4)
    assert "XMV(10)" in idv4["expected_roots"]
    assert "XMEAS(21)" in idv4["affected_variables"]


def test_tep_topology_contains_cooling_path():
    edges = {(e["cause"], e["effect"]) for e in tep_topology()["edges"]}
    assert ("XMV(10)", "XMEAS(21)") in edges
    assert ("XMEAS(21)", "XMEAS(9)") in edges


def test_rank_tep_catalog_prefers_reactor_cooling_fault():
    catalog = tep_fault_catalog()
    ranked = ["XMV(10)", "XMEAS(21)", "XMEAS(9)", "XMEAS(7)"]
    scores = {name: 1.0 - i * 0.1 for i, name in enumerate(ranked)}
    result = rank_tep_fault_catalog(ranked, catalog, variable_scores=scores)
    assert result["best_match"]["fault_id"] in {4, 11, 14}
    assert result["best_match"]["best_root"] in TEP_FAULT_SPECS[result["best_match"]["fault_id"]].expected_roots


def test_knowledge_guided_decision_can_override_generic_downstream_variable():
    catalog = tep_fault_catalog()
    generic = [{"variable": "XMEAS(9)", "score": 0.9}, {"variable": "XMV(10)", "score": 0.45}]
    shift = {"XMV(10)": 1.0, "XMEAS(21)": 0.8, "XMEAS(9)": 0.7}
    decision = knowledge_guided_root_cause_decision(
        generic,
        catalog,
        shift_scores=shift,
        onset_order=["XMV(10)", "XMEAS(21)", "XMEAS(9)"],
    )
    assert decision["decision_source"] == "tep_fault_catalog"
    assert decision["root_cause"] == "XMV(10)"
    assert decision["fault_id"] in {4, 11, 14}
