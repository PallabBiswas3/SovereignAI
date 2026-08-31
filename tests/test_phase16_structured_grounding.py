from app.governance.grounding import ClaimStatus, GroundingChecker
from app.rag.embeddings import configured_embedding_provider


def test_paraphrased_claim_is_semantically_supported_with_provenance() -> None:
    report = GroundingChecker(.48, .32, configured_embedding_provider()).evaluate(
        "The pump should be shut down if bearing temperature goes beyond 90 C.",
        [{
            "text": "A bearing temperature above 90 C requires immediate shutdown.",
            "source": {"file": "Maintenance_SOP.md", "section": "Temperature"},
            "retrieval_score": .8,
        }],
    )
    assert report.claims[0].status == ClaimStatus.supported
    assert report.claims[0].source["section"] == "Temperature"
    assert report.claims[0].semantic_score > .5


def test_invented_material_claim_is_explicitly_unsupported() -> None:
    report = GroundingChecker(.55, .35, configured_embedding_provider()).evaluate(
        "The manufacturer requires complete pump replacement within 24 hours.",
        [{
            "text": "If vibration exceeds 9 mm/s, remove the pump from service for engineering inspection.",
            "source": {"file": "Maintenance_SOP.md", "section": "Vibration"},
        }],
    )
    assert report.claims[0].status == ClaimStatus.unsupported
    assert report.unsupported_material_claims
