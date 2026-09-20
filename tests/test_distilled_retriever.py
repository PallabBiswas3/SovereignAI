from app.core.config import Settings
from app.rag.distilled_retriever import DistillingHybridRetriever
from app.rag.retrieval import RetrievedChunk


class StubRetriever:
    def __init__(self, rows: list[RetrievedChunk]) -> None:
        self.rows = rows
        self.calls: list[int] = []

    def search(self, query: str, limit: int | None = None, *, asset_id: str | None = None):
        bounded = limit or len(self.rows)
        self.calls.append(bounded)
        return self.rows[:bounded]


def chunk(chunk_id: str, text: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text,
        score=score,
        document_id=f"doc-{chunk_id}",
        source={
            "file": "SOP-MNT-017.pdf",
            "page": 4,
            "section": "4.2",
            "revision": "Rev 4",
            "document_hash": f"hash-{chunk_id}",
        },
        scores={"dense": score, "sparse": score, "fusion": score, "reranker": score},
        retrieval_methods=["dense", "bm25"],
        access_scope=["internal"],
    )


def test_wrapper_is_identity_when_distillation_disabled():
    rows = [chunk("c1", "Pump-102 maximum vibration limit is 7.1 mm/s RMS.", 0.9)]
    inner = StubRetriever(rows)
    settings = Settings(context_distillation_enabled=False)
    wrapper = DistillingHybridRetriever(inner, settings)

    output = wrapper.search("Pump-102 vibration limit", 1)

    assert output[0].text == rows[0].text
    assert wrapper.last_distillation["distillation_applied"] is False
    assert inner.calls == [1]


def test_wrapper_distills_real_model_facing_text_when_enabled():
    rows = [
        chunk(
            "c1",
            "This section contains general administrative background. "
            "Pump-102 maximum vibration limit is 7.1 mm/s RMS. "
            "Historical procurement notes follow. "
            "Additional unrelated narrative is included for archival purposes.",
            0.9,
        )
    ]
    inner = StubRetriever(rows)
    settings = Settings(
        context_distillation_enabled=True,
        context_distillation_target_tokens=18,
        context_distillation_sentence_redundancy_threshold=0.82,
        hybrid_rerank_top_k=10,
    )
    wrapper = DistillingHybridRetriever(inner, settings)

    output = wrapper.search("What is the Pump-102 vibration limit?", 1)

    assert output
    assert "7.1 mm/s" in output[0].text
    assert len(output[0].text) < len(rows[0].text)
    assert wrapper.last_distillation["distillation_applied"] is True
    assert wrapper.last_distillation["selected_evidence_tokens"] <= 18
    assert output[0].telemetry["evidence_distillation"]["distillation_applied"] is True
