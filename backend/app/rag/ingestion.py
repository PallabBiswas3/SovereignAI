from __future__ import annotations

import hashlib
import json
from pathlib import Path
from uuid import uuid4

from sqlalchemy.orm import Session

from app.core.database import KnowledgeChunkRecord, KnowledgeDocument
from app.rag.chunking import ProvenanceChunker
from app.rag.embeddings import EmbeddingProvider
from app.tools.file_tools import extract_text


class KnowledgeIngestionService:
    def __init__(self, session: Session, embeddings: EmbeddingProvider, chunker: ProvenanceChunker | None = None) -> None:
        self.session = session
        self.embeddings = embeddings
        self.chunker = chunker or ProvenanceChunker()

    def ingest(self, path: Path, metadata: dict[str, object] | None = None) -> KnowledgeDocument:
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        existing = self.session.query(KnowledgeDocument).filter_by(checksum=checksum).one_or_none()
        if existing:
            if existing.embedding_provider != self.embeddings.provider_name:
                records = self.session.query(KnowledgeChunkRecord).filter_by(
                    document_id=existing.id
                ).order_by(KnowledgeChunkRecord.chunk_index).all()
                vectors = self.embeddings.embed_documents([record.text for record in records])
                for record, vector in zip(records, vectors):
                    record.embedding_json = json.dumps(vector)
                existing.embedding_provider = self.embeddings.provider_name
                existing.embedding_dimension = self.embeddings.dimension
                self.session.commit()
            return existing
        text = extract_text(path)
        chunks = self.chunker.chunk(text, metadata)
        if not chunks:
            raise ValueError("No extractable text found in document")
        vectors = self.embeddings.embed_documents([chunk.text for chunk in chunks])
        document = KnowledgeDocument(
            id=str(uuid4()), filename=path.name, checksum=checksum,
            metadata_json=json.dumps(metadata or {}), chunk_count=len(chunks),
            embedding_provider=self.embeddings.provider_name,
            embedding_dimension=self.embeddings.dimension,
        )
        self.session.add(document)
        for chunk, vector in zip(chunks, vectors):
            self.session.add(KnowledgeChunkRecord(
                id=str(uuid4()), document_id=document.id, chunk_index=chunk.index,
                text=chunk.text, page=chunk.page, section=chunk.section,
                metadata_json=json.dumps(chunk.metadata), embedding_json=json.dumps(vector),
            ))
        self.session.commit()
        return document
