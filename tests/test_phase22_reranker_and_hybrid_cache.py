from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.database import Base
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.hybrid import HybridRetriever
from app.rag.ingestion import KnowledgeIngestionService
from app.rag.reranking import LexicalTestReranker, LocalCrossEncoderReranker
from app.rag.retrieval import RetrievedChunk
from app.resources.cache import SQLiteCache
from app.resources.scheduler import ResourceScheduler


def test_reranker_reorders_candidates_and_records_score() -> None:
    candidates = [
        RetrievedChunk("a", "Monthly lubrication schedule", 0.03, {"file": "a"}),
        RetrievedChunk("b", "PU-102 vibration limit is 6.0 mm/s RMS", 0.02, {"file": "b"}),
    ]
    output = LexicalTestReranker().rerank("PU-102 vibration limit", candidates, 2)
    assert output[0].chunk_id == "b"
    assert output[0].scores["reranker"] == 1.0


def test_unavailable_local_cross_encoder_falls_back_to_fusion(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    path = tmp_path / "sop.md"
    path.write_text("[PAGE 1]\n## Section 7.4\nPU-102 vibration limit is 6.0 mm/s RMS.", encoding="utf-8")
    with Session(engine) as session:
        embeddings = LocalHashEmbeddingProvider(32)
        KnowledgeIngestionService(session, embeddings).ingest(path, {"classification": "internal"})
        reranker = LocalCrossEncoderReranker(
            "definitely-not-present/cross-encoder",
            local_files_only=True,
            scheduler=ResourceScheduler(),
        )
        retriever = HybridRetriever(session, embeddings, reranker=reranker)
        output = retriever.search("PU-102 vibration limit", 1)
    assert output
    assert retriever.last_telemetry.reranker_available is False
    assert retriever.last_telemetry.warning.startswith("RERANKER_UNAVAILABLE:")
    assert output[0].scores["fusion"] is not None


def test_hybrid_cache_invalidates_when_pipeline_version_changes(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    cache = SQLiteCache(factory)
    path = tmp_path / "sop.md"
    path.write_text("[PAGE 1]\n## Section 7.4\nPU-102 vibration limit is 6.0 mm/s RMS.", encoding="utf-8")
    with factory() as session:
        embeddings = LocalHashEmbeddingProvider(32)
        KnowledgeIngestionService(session, embeddings).ingest(path, {"classification": "internal"})
        v1 = Settings(fusion_strategy_version="rrf-v1", reranker_enabled=False)
        first = HybridRetriever(session, embeddings, cache, settings=v1).search("PU-102", 1)
        second = HybridRetriever(session, embeddings, cache, settings=v1).search("PU-102", 1)
        misses_before = cache.stats()["misses"]
        v2 = Settings(fusion_strategy_version="rrf-v2", reranker_enabled=False)
        third = HybridRetriever(session, embeddings, cache, settings=v2).search("PU-102", 1)
    assert first[0].cache_hit is False
    assert second[0].cache_hit is True
    assert third[0].cache_hit is False
    assert cache.stats()["misses"] == misses_before + 1
