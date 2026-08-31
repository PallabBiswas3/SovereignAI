from pathlib import Path
from uuid import uuid4

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.database import Base
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.ingestion import KnowledgeIngestionService
from app.rag.retrieval import LocalRetriever


def test_local_rag_retrieves_sop_threshold_with_provenance(tmp_path: Path) -> None:
    source = tmp_path / f"SOP_{uuid4().hex}.md"
    source.write_text(
        "[PAGE 37]\n## Section 7.4\nMaximum acceptable pump vibration is 6 mm/s RMS.\n"
        "[PAGE 38]\n## Section 7.5\nBearing lubrication intervals are monthly.",
        encoding="utf-8",
    )
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        embeddings = LocalHashEmbeddingProvider()
        KnowledgeIngestionService(session, embeddings).ingest(source, {"department": "maintenance"})
        results = LocalRetriever(session, embeddings).search("acceptable vibration threshold", limit=1)

    assert "6 mm/s" in results[0].text
    assert results[0].source["file"] == source.name
    assert results[0].source["page"] == 37
    assert results[0].source["section"] == "7.4"

