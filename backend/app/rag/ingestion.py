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
from app.identity.models import DocumentACL


class KnowledgeIngestionService:
    def __init__(self, session: Session, embeddings: EmbeddingProvider, chunker: ProvenanceChunker | None = None) -> None:
        self.session = session
        self.embeddings = embeddings
        self.chunker = chunker or ProvenanceChunker()

    def ingest(
        self,
        path: Path,
        metadata: dict[str, object] | None = None,
        *,
        acl: DocumentACL | None = None,
        require_acl: bool = False,
    ) -> KnowledgeDocument:
        if require_acl and acl is None:
            raise ValueError("ACCESS_SCOPE_REQUIRED")
        metadata = dict(metadata or {})
        if acl:
            metadata.update({
                "organization_id": acl.organization_id,
                "department_id": acl.department_id,
                "department": acl.department_id,
                "workspace_id": acl.workspace_id,
                "classification": acl.classification.name.upper(),
                "allowed_roles": [role.value for role in acl.allowed_roles],
                "allowed_users": acl.allowed_users,
                "owner_id": acl.owner_id,
            })
        content = path.read_bytes()
        checksum = hashlib.sha256(content).hexdigest()
        existing = self.session.query(KnowledgeDocument).filter_by(checksum=checksum).one_or_none()
        if existing:
            if acl and (
                existing.organization_id not in {None, acl.organization_id}
                or existing.workspace_id not in {None, acl.workspace_id}
            ):
                raise ValueError("DOCUMENT_ALREADY_SCOPED")
            if acl and existing.organization_id is None:
                existing.organization_id = acl.organization_id
                existing.owner_id = acl.owner_id
                existing.workspace_id = acl.workspace_id
                existing.department_id = acl.department_id
                existing.classification = acl.classification.name.upper()
                existing.allowed_roles_json = json.dumps([role.value for role in acl.allowed_roles])
                existing.allowed_users_json = json.dumps(acl.allowed_users)
                existing_metadata = json.loads(existing.metadata_json or "{}")
                existing_metadata.update(metadata)
                existing.metadata_json = json.dumps(existing_metadata)
                for record in self.session.query(KnowledgeChunkRecord).filter_by(document_id=existing.id).all():
                    chunk_metadata = json.loads(record.metadata_json or "{}")
                    chunk_metadata.update(metadata)
                    record.metadata_json = json.dumps(chunk_metadata)
                self.session.commit()
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
            metadata_json=json.dumps(metadata), chunk_count=len(chunks),
            embedding_provider=self.embeddings.provider_name,
            embedding_dimension=self.embeddings.dimension,
            organization_id=acl.organization_id if acl else None,
            owner_id=acl.owner_id if acl else None,
            workspace_id=acl.workspace_id if acl else None,
            department_id=acl.department_id if acl else None,
            classification=acl.classification.name.upper() if acl else str(metadata.get("classification", "INTERNAL")).upper(),
            allowed_roles_json=json.dumps([role.value for role in acl.allowed_roles] if acl else []),
            allowed_users_json=json.dumps(acl.allowed_users if acl else []),
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
