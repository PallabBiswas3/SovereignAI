from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.database import Base
from app.evaluation.batch2 import Batch2EvaluationRunner
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.hybrid import HybridRetriever
from app.rag.ingestion import KnowledgeIngestionService
from app.workflows.inspection import InspectionWorkflow


def _session():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    return Session(engine)


def test_flagship_normalizes_500_kpa_and_keeps_original(tmp_path: Path) -> None:
    session = _session()
    try:
        embeddings = LocalHashEmbeddingProvider(64)
        sop = tmp_path / "SOP.md"
        sop.write_text("[PAGE 3]\n## Section 7.6\nPump-102 normal discharge pressure is 4.8 to 5.5 bar.", encoding="utf-8")
        report = tmp_path / "inspection.md"
        report.write_text("Pump-102\nDischarge pressure: 500 kPa", encoding="utf-8")
        KnowledgeIngestionService(session, embeddings).ingest(sop, {"classification": "internal", "revision": "Rev 3"})
        retriever = HybridRetriever(session, embeddings, settings=Settings(reranker_enabled=False))
        result = InspectionWorkflow(retriever).analyze(report, tmp_path / "approval.docx")
    finally:
        session.close()
    measurement = result.evidence_bundle["measurements"][0]
    assert measurement["original_value"] == 500
    assert measurement["original_unit"] == "kPa"
    assert measurement["normalized_value"] == 5.0
    assert measurement["normalized_unit"] == "bar"
    assert result.findings[0]["status"] == "NORMAL"
    assert result.evidence_bundle["calculations"][0]["result"] is True


def test_flagship_surfaces_revision_conflict_without_arbitrary_limit(tmp_path: Path) -> None:
    session = _session()
    try:
        embeddings = LocalHashEmbeddingProvider(64)
        for revision, limit in (("Rev 2", "6.0"), ("Rev 3", "5.5")):
            sop = tmp_path / f"SOP-{revision}.md"
            sop.write_text(
                f"[PAGE 37]\n## Section 7.4\nPump-102 vibration shall not exceed {limit} mm/s RMS.",
                encoding="utf-8",
            )
            KnowledgeIngestionService(session, embeddings).ingest(
                sop, {"classification": "internal", "revision": revision}
            )
        report = tmp_path / "inspection.md"
        report.write_text("Pump-102\nVibration: 5.8 mm/s", encoding="utf-8")
        result = InspectionWorkflow(
            HybridRetriever(session, embeddings, settings=Settings(reranker_enabled=False))
        ).analyze(report, tmp_path / "approval.docx", "What is the current vibration limit?")
    finally:
        session.close()
    assert result.evidence_bundle["conflicts"]
    assert result.findings[0]["status"] == "CONFLICTING_EVIDENCE"
    assert "human establishes the applicable revision" in result.recommendation
    assert any(claim["support_status"] == "CONFLICTING_EVIDENCE" for claim in result.evidence_bundle["claims"])


def test_batch2_evaluation_reports_actual_comparable_metrics() -> None:
    result = Batch2EvaluationRunner().run()
    comparison = result["comparison"]
    assert result["benchmark_version"] == "2026.09-batch2-v1"
    assert set(comparison) == {"dense", "hybrid", "hybrid_rerank"}
    for strategy in comparison.values():
        assert 0 <= strategy["recall_at_1"] <= 1
        assert 0 <= strategy["recall_at_3"] <= 1
        assert 0 <= strategy["recall_at_5"] <= 1
        assert 0 <= strategy["mrr"] <= 1
        assert strategy["mean_total_latency_ms"] >= 0
    assert comparison["hybrid"]["sparse_latency_ms"] is not None
    assert result["context_compiler"]["required_fact_preserved"] is True
    assert result["claim_verification"]["deterministic_correctness"] == 1.0
