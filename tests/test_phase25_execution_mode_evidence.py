from app.core.config import Settings
from app.evidence.context import ContextCompiler
from app.rag.decomposition import ModeAwareRetrievalPipeline, QueryDecomposer
from app.rag.retrieval import RetrievedChunk


class FakeRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def search(self, query: str, limit: int):
        self.queries.append(query)
        identifier = "same" if "requirements" in query else str(len(self.queries))
        return [RetrievedChunk(
            chunk_id=identifier, text=f"Evidence for {query}", score=1 / len(self.queries),
            source={"file": "sop.md", "section": "7.4"},
            scores={"fusion": 1 / len(self.queries)}, retrieval_methods=["dense", "bm25"],
        )]


def test_fast_and_standard_do_not_decompose_queries() -> None:
    decomposer = QueryDecomposer(max_subqueries=4)
    query = "Assess Pump-102 using latest inspection, historical readings and maintenance requirements"
    assert decomposer.decompose(query, "FAST") == [query]
    assert decomposer.decompose(query, "STANDARD") == [query]


def test_deep_mode_decomposes_bounded_queries_and_deduplicates_results() -> None:
    retriever = FakeRetriever()
    pipeline = ModeAwareRetrievalPipeline(retriever, max_subqueries=4)
    results = pipeline.search(
        "Assess Pump-102 using latest inspection, historical readings, maintenance requirements and replacement criteria",
        "DEEP",
        10,
    )
    assert 2 <= len(pipeline.last_subqueries) <= 4
    assert len(retriever.queries) == len(pipeline.last_subqueries)
    assert len({item.chunk_id for item in results}) == len(results)


def test_fast_context_is_bounded_more_tightly_than_standard() -> None:
    settings = Settings(context_max_evidence_chunks=8, context_max_evidence_tokens=3000)
    candidates = [RetrievedChunk(
        chunk_id=str(index), text=f"Pump evidence {index} vibration 6.0 mm/s. " * 20,
        score=1 / index, source={"file": "sop.md", "page": index},
        scores={"fusion": 1 / index}, retrieval_methods=["dense", "bm25"],
    ) for index in range(1, 9)]
    compiler = ContextCompiler(settings)
    fast = compiler.compile(task="Pump vibration", evidence=candidates, selected_model="general",
                            context_window=8192, execution_mode="FAST")
    standard = compiler.compile(task="Pump vibration", evidence=candidates, selected_model="general",
                                context_window=8192, execution_mode="STANDARD")
    assert fast.budget.max_evidence_chunks == 3
    assert fast.budget.max_evidence_tokens == 1000
    assert standard.budget.max_evidence_chunks == 8
    assert standard.metrics.final_evidence_count >= fast.metrics.final_evidence_count
