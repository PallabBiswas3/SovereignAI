from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import Base
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.hybrid import BM25Retriever, HybridRetriever, ReciprocalRankFusion
from app.rag.ingestion import KnowledgeIngestionService
from app.rag.retrieval import LocalRetriever, RetrievedChunk


def _corpus(tmp_path: Path):
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = Session(engine)
    embeddings = LocalHashEmbeddingProvider(64)
    documents = [
        (
            "SOP-MNT-42.md",
            "[PAGE 37]\n## Section 7.4\nSOP-MNT-42 applies to Pump-102 / PU-102. "
            "Maximum normal vibration shall not exceed 6.0 mm/s RMS.",
            {"classification": "internal", "department": "maintenance", "revision": "Rev 3"},
        ),
        (
            "lubrication.md",
            "[PAGE 2]\n## Section 3.1\nBearing lubrication is performed monthly for XV-101.",
            {"classification": "internal", "department": "maintenance"},
        ),
        (
            "restricted.md",
            "[PAGE 1]\n## Section 1\nPU-102 confidential procurement record.",
            {"classification": "restricted", "department": "procurement"},
        ),
    ]
    for filename, text, metadata in documents:
        path = tmp_path / filename
        path.write_text(text, encoding="utf-8")
        KnowledgeIngestionService(session, embeddings).ingest(path, metadata)
    return session, embeddings


def test_existing_dense_retriever_remains_compatible(tmp_path: Path) -> None:
    session, embeddings = _corpus(tmp_path)
    try:
        result = LocalRetriever(session, embeddings).search("acceptable vibration threshold", 1)[0]
        assert result.source["section"] == "7.4"
        assert result.retrieval_methods == ["dense"]
        assert result.scores["dense"] is not None
    finally:
        session.close()


def test_bm25_retrieves_exact_industrial_identifier_with_provenance(tmp_path: Path) -> None:
    session, _ = _corpus(tmp_path)
    try:
        result = BM25Retriever(session).search("PU-102 vibration limit", 2)[0]
        assert result.source["file"] == "SOP-MNT-42.md"
        assert result.source["page"] == 37
        assert result.source["section"] == "7.4"
        assert result.source["revision"] == "Rev 3"
        assert result.retrieval_methods == ["bm25"]
    finally:
        session.close()


def test_rrf_merges_duplicate_candidates_without_adding_raw_scores() -> None:
    dense = RetrievedChunk("one", "dense", 0.91, {"file": "a"}, retrieval_methods=["dense"])
    sparse_same = RetrievedChunk("one", "dense", 12.4, {"file": "a"}, retrieval_methods=["bm25"])
    sparse_other = RetrievedChunk("two", "other", 10.0, {"file": "b"}, retrieval_methods=["bm25"])
    fused = ReciprocalRankFusion(k=60).fuse(
        [("dense", [dense]), ("bm25", [sparse_same, sparse_other])], 10
    )
    assert [item.chunk_id for item in fused].count("one") == 1
    assert fused[0].retrieval_methods == ["dense", "bm25"]
    assert fused[0].scores["dense"] == 0.91
    assert fused[0].scores["sparse"] == 12.4
    assert fused[0].score < 0.1


def test_hybrid_retains_dense_paraphrase_and_sparse_identifier(tmp_path: Path) -> None:
    session, embeddings = _corpus(tmp_path)
    settings = Settings(
        hybrid_dense_top_k=10,
        hybrid_sparse_top_k=10,
        hybrid_fusion_candidate_limit=10,
        hybrid_final_context_k=3,
    )
    try:
        identifier = HybridRetriever(session, embeddings, settings=settings).search(
            "PU-102 vibration limit", 3
        )
        paraphrase = HybridRetriever(session, embeddings, settings=settings).search(
            "What level of oscillation is acceptable?", 3
        )
        assert identifier[0].source["section"] == "7.4"
        assert "bm25" in identifier[0].retrieval_methods
        assert any(item.source["section"] == "7.4" for item in paraphrase)
    finally:
        session.close()


def test_access_scope_is_applied_before_ranking(tmp_path: Path) -> None:
    session, _ = _corpus(tmp_path)
    try:
        internal = BM25Retriever(session, access_scope="internal").search("PU-102", 10)
        restricted = BM25Retriever(session, access_scope="restricted").search("PU-102", 10)
        assert all(item.source["classification"] != "restricted" for item in internal)
        assert restricted[0].source["file"] == "restricted.md"
    finally:
        session.close()
