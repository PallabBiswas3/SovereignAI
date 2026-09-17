import json
from types import SimpleNamespace

from app.integrations.models import IntegratedAnalysisResponse
from app.integrations.routing import IntegrationRoutePlanner
from app.integrations.task_adapter import integrated_result_state, missing_input_state
from app.core.config import get_settings
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter


def routing_decision(settings):
    return ModelRouter(ModelRegistry(settings.models_config)).route("diagnose bearing vibration", None)


def test_internal_evidence_question_selects_graph_and_controlplane(tmp_path):
    plan = IntegrationRoutePlanner().plan(
        "What do our internal documents say about the pump limit?", [], tmp_path,
    )

    assert plan.use_graph is True
    assert plan.diagnostic is None
    assert plan.services == ["graph-rag", "controlplane"]
    assert plan.handles_request is True


def test_diagnostic_intent_without_measurements_does_not_invent_input(tmp_path):
    plan = IntegrationRoutePlanner().plan(
        "Diagnose this bearing vibration anomaly", [], tmp_path,
    )

    assert plan.diagnostic_domain == "bearing"
    assert plan.missing_diagnostic_input is True
    assert plan.services == ["time-series-diagnostic-agent", "controlplane"]

    selection = SimpleNamespace(requested=SimpleNamespace(value="AUTOMATIC"), selected=SimpleNamespace(value="STANDARD"), reason="test")
    state = missing_input_state(
        request="Diagnose this bearing vibration anomaly",
        routing=routing_decision(get_settings()), selection=selection, plan=plan,
    )
    assert "needs measured time-series data" in state.final_response
    assert state.runtime_metrics["service_plan"] == plan.services


def test_valid_diagnostic_attachment_selects_the_diagnostic_service(tmp_path):
    uploads = tmp_path / "uploads"
    uploads.mkdir()
    payload = {
        "domain": "bearing",
        "inputs": {"signal": [0.0] * 32, "sampling_rate_hz": 8000},
    }
    (uploads / "bearing.json").write_text(json.dumps(payload), encoding="utf-8")

    plan = IntegrationRoutePlanner().plan(
        "Please assess these measurements", ["uploads/bearing.json"], tmp_path,
    )

    assert plan.diagnostic is not None
    assert plan.diagnostic.domain == "bearing"
    assert plan.use_graph is False
    assert plan.missing_diagnostic_input is False


def test_combined_result_maps_evidence_into_existing_task_contract():
    app_settings = get_settings()
    plan = IntegrationRoutePlanner().plan(
        "Use internal document evidence for this bearing diagnostic", [], app_settings.workspace_root,
    )
    plan.diagnostic = None
    result = IntegratedAnalysisResponse(
        run_id="integration-1", released=True, status="released",
        final_response="Supported answer.", precheck={"decision": {"action": "allow"}},
        graph_evidence={
            "status": "grounded",
            "chunks": [{"id": "chunk-1", "content": "Limit 7.1 mm/s", "metadata": {"file": "manual.pdf", "page": 4}, "similarity": 0.9}],
            "claims": [{"id": "claim-1", "claim_text": "The limit is 7.1 mm/s.", "supporting_chunk_ids": ["chunk-1"]}],
            "verification": {"summary": "supported", "results": [{"claimId": "claim-1", "label": "SUPPORTED", "reason": "Matched source."}]},
        },
        controlplane={"decision": {"action": "allow"}},
        service_status={"controlplane": "available", "graph-rag": "available", "diagnostics": "not_requested"},
    )
    selection = SimpleNamespace(requested=SimpleNamespace(value="AUTOMATIC"), selected=SimpleNamespace(value="STANDARD"), reason="test")

    state = integrated_result_state(
        request="Use internal document evidence for this bearing diagnostic",
        routing=routing_decision(app_settings), selection=selection, plan=plan, result=result,
    )

    assert state.sources[0]["file"] == "manual.pdf"
    assert state.claims[0]["support_status"] == "SUPPORTED"
    assert state.runtime_metrics["service_plan"] == plan.services


def test_unreleased_integration_status_is_available_to_the_governance_layer():
    from app.integrations.task_adapter import integration_unavailable_state

    settings = get_settings()
    plan = IntegrationRoutePlanner().plan("Search our internal documents", [], settings.workspace_root)
    selection = SimpleNamespace(requested=SimpleNamespace(value="AUTOMATIC"), selected=SimpleNamespace(value="STANDARD"), reason="test")
    state = integration_unavailable_state(
        request="Search our internal documents", routing=routing_decision(settings),
        selection=selection, plan=plan, error="graph-rag unavailable",
    )

    assert state.context_metrics["controlplane_status"] == "not_released"
    assert "no unsupported answer" in state.warnings[-1].lower()
