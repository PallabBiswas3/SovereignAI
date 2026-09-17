from tsdiag.domains import DOMAIN_PACKS, get_domain_pack
from tsdiag.domains.planner import build_execution_plan, describe_contract


def test_exactly_six_v1_domain_packs():
    assert set(DOMAIN_PACKS) == {
        "bearing",
        "process",
        "wind_scada",
        "battery",
        "turbofan",
        "transformer",
    }


def test_every_domain_has_ordered_nonempty_tool_contracts():
    for pack in DOMAIN_PACKS.values():
        assert len(pack.tools) >= 8
        names = pack.tool_names()
        assert len(names) == len(set(names))
        for tool in pack.tools:
            assert tool.name
            assert tool.purpose
            assert tool.required_inputs
            assert tool.outputs


def test_domain_outputs_and_benchmarks_are_declared():
    for pack in DOMAIN_PACKS.values():
        assert pack.outputs
        assert pack.benchmark_targets


def test_execution_plan_blocks_missing_required_metadata():
    plan = build_execution_plan("bearing", {})
    assert not plan.ready
    assert "sampling_rate_hz" in plan.missing_required_metadata


def test_execution_plan_becomes_ready_with_required_metadata():
    plan = build_execution_plan(
        "bearing",
        {"sampling_rate_hz": 12000.0, "shaft_rate_hz": 30.0},
    )
    assert plan.ready
    assert plan.missing_required_metadata == ()
    assert "shaft_rate_hz" in plan.available_optional_metadata
    assert plan.tool_sequence[0] == "signal_integrity"
    assert plan.tool_sequence[-1] == "bearing_evidence_fusion"


def test_contract_is_serializable_for_ui_or_agent_router():
    rows = describe_contract(get_domain_pack("process"))
    assert rows[0]["name"] == "process_data_quality"
    assert any(row["name"] == "granger_causality" for row in rows)
