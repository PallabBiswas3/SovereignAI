from app.integrations.orchestrator import IndustrialIntegrationOrchestrator


def test_graph_grounding_evidence_preserves_available_provenance_only():
    graph = {
        "claims": [
            {
                "claim_id": "claim-7",
                "claim_text": "Pump-102 reached 95 C.",
                "document_id": "manual-12",
                "chunk_id": "chunk-9",
                "source_name": "Maintenance Manual",
                "page": 18,
                "revision": "R4",
                "retrieval_score": 0.91,
                "authorization_scope": "plant-a",
            }
        ],
        "chunks": [
            {
                "content": "The alarm threshold is 90 C.",
                "chunk_id": "chunk-10",
                "score": 0.82,
            }
        ],
    }

    evidence = IndustrialIntegrationOrchestrator._graph_grounding_evidence(graph)

    assert evidence[0]["evidence_id"] == "claim-7"
    assert evidence[0]["document_id"] == "manual-12"
    assert evidence[0]["chunk_id"] == "chunk-9"
    assert evidence[0]["page"] == 18
    assert evidence[0]["revision"] == "R4"
    assert evidence[0]["retrieval_score"] == 0.91
    assert evidence[0]["authorization_scope"] == "plant-a"

    assert evidence[1]["evidence_id"] == "chunk-10"
    assert evidence[1]["chunk_id"] == "chunk-10"
    assert evidence[1]["retrieval_score"] == 0.82
    assert "document_id" not in evidence[1]
    assert "page" not in evidence[1]
    assert "revision" not in evidence[1]


def test_graph_grounding_evidence_does_not_invent_missing_provenance():
    evidence = IndustrialIntegrationOrchestrator._graph_grounding_evidence(
        {"chunks": [{"content": "Only text is available."}]}
    )
    assert evidence == [{"text": "Only text is available.", "metadata": {"graph_item_type": "chunk"}}]
