from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from app.core.database import Base
from app.rag.embeddings import SentenceTransformerEmbeddingProvider
from app.rag.ingestion import KnowledgeIngestionService
from app.rag.retrieval import LocalRetriever


def test_semantic_embeddings_retrieve_paraphrased_sop_sections(tmp_path: Path) -> None:
    sop = tmp_path / "semantic_sop.md"
    sop.write_text(
        "[PAGE 1]\n## Section 7.4\nOverall pump vibration shall not exceed 6.0 mm/s RMS.\n"
        "[PAGE 2]\n## Section 7.5\nNormal bearing temperature is up to 80 degrees Celsius.\n"
        "[PAGE 3]\n## Section 7.6\nExpected discharge pressure is between 4.8 and 5.5 bar.\n",
        encoding="utf-8",
    )
    provider = SentenceTransformerEmbeddingProvider(local_files_only=True)
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    cases = [
        ("How much mechanical shaking is acceptable before service is needed?", "7.4"),
        ("How hot may the bearing run during normal operation?", "7.5"),
        ("What delivery pressure band should the pump maintain?", "7.6"),
    ]
    ranks = []
    with Session(engine) as session:
        KnowledgeIngestionService(session, provider).ingest(
            sop, {"department": "maintenance", "classification": "internal"}
        )
        retriever = LocalRetriever(session, provider)
        for query, expected in cases:
            found = retriever.search(query, 3)
            ranks.append(next((index for index, item in enumerate(found, start=1) if item.source["section"] == expected), 0))
            assert found[0].source["department"] == "maintenance"
            assert found[0].source["classification"] == "internal"
    assert sum(rank == 1 for rank in ranks) / len(ranks) >= 2 / 3
    assert all(rank in {1, 2, 3} for rank in ranks)
    assert sum(1 / rank for rank in ranks) / len(ranks) >= 0.75
