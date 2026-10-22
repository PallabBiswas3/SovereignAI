import asyncio
from app.integrations.orchestrator import IndustrialIntegrationOrchestrator


def chunk(identifier, kind="maintenance_sop", status="current", asset="Pump-731"):
    return {"id": identifier, "content": f"{asset} requires isolation authorization.",
            "metadata": {"document_type": kind, "document_status": status}}


def test_procedure_lookup_is_bounded_and_preserves_original_evidence_and_scope():
    scope = {"organization_id": "org", "fingerprint": "exact-scope"}
    original = {"status": "grounded", "chunks": [chunk("log", "operations_log")], "claims": [{"id": "claim"}]}
    class Graph:
        calls = []
        async def retrieve(self, query, assurance, access):
            self.calls.append((query, assurance, access))
            return original if len(self.calls) == 1 else {"status": "grounded", "chunks": [
                chunk("current"), chunk("old", status="superseded"), chunk("other", asset="Pump-999")], "timings_ms": {"total": 5}}
    graph = Graph()
    service = IndustrialIntegrationOrchestrator(graph=graph)
    result = asyncio.run(service._retrieve_graph("Diagnose Pump-731 process; compare the SOP and generate a maintenance report.", "thorough", scope))
    assert len(graph.calls) == 2 and all(c[1:] == ("thorough", scope) for c in graph.calls)
    assert [c["id"] for c in result["chunks"]] == ["log", "current"]
    assert result["claims"] == original["claims"]
    assert result["procedure_retrieval"]["matched_current_sops"] == 1


def test_existing_current_sop_does_not_trigger_second_lookup():
    class Graph:
        calls = 0
        async def retrieve(self, *args):
            self.calls += 1
            return {"status": "grounded", "chunks": [chunk("current")]}
    graph = Graph()
    service = IndustrialIntegrationOrchestrator(graph=graph)
    asyncio.run(service._retrieve_graph("Pump-731 SOP; generate a maintenance report.", "thorough", {}))
    assert graph.calls == 1


def test_sop_comparison_selects_document_retrieval_without_special_prompt_words(tmp_path):
    from app.integrations.routing import IntegrationRoutePlanner
    plan = IntegrationRoutePlanner().plan("Diagnose Pump-731 process pressure; compare with the SOP and generate a maintenance report.", [], tmp_path)
    assert plan.use_graph and plan.missing_diagnostic_input
