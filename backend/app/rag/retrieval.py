from __future__ import annotations

import json
from dataclasses import dataclass

from sqlalchemy.orm import Session

from app.core.database import KnowledgeChunkRecord, KnowledgeDocument
from app.rag.embeddings import EmbeddingProvider


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: dict[str, object]


class LocalRetriever:
    def __init__(self, session: Session, embeddings: EmbeddingProvider) -> None:
        self.session = session
        self.embeddings = embeddings

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        query_vector = self.embeddings.embed_query(query)
        rows = self.session.query(KnowledgeChunkRecord, KnowledgeDocument).join(
            KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunkRecord.document_id
        ).filter(KnowledgeDocument.embedding_provider == self.embeddings.provider_name).all()
        ranked: list[RetrievedChunk] = []
        for chunk, document in rows:
            stored_vector = json.loads(chunk.embedding_json)
            if len(stored_vector) != self.embeddings.dimension:
                continue
            score = self._cosine(query_vector, stored_vector)
            metadata = json.loads(chunk.metadata_json or "{}")
            ranked.append(RetrievedChunk(
                chunk_id=chunk.id, text=chunk.text, score=score,
                source={"file": document.filename, "page": chunk.page, "section": chunk.section,
                        "chunk_id": chunk.id, "department": metadata.get("department"),
                        "classification": metadata.get("classification")},
            ))
        return sorted(ranked, key=lambda item: item.score, reverse=True)[: max(1, min(limit, 20))]
