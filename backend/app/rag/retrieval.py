from __future__ import annotations

import json
from dataclasses import dataclass
from dataclasses import field

from sqlalchemy.orm import Session

from app.core.database import KnowledgeChunkRecord, KnowledgeDocument
from app.rag.embeddings import EmbeddingProvider
from app.resources.cache import CacheBackend, CacheKeyBuilder, CacheNamespace, stable_hash
from app.identity.authorization import AuthorizationService
from app.identity.models import ClearanceLevel, DocumentACL, Principal, Role


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    source: dict[str, object]
    cache_hit: bool = False
    document_id: str | None = None
    scores: dict[str, float | None] = field(default_factory=dict)
    retrieval_methods: list[str] = field(default_factory=list)
    access_scope: list[str] = field(default_factory=list)
    telemetry: dict[str, object] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "text": self.text,
            "score": self.score,
            "source": self.source,
            "scores": self.scores,
            "retrieval_methods": self.retrieval_methods,
            "access_scope": self.access_scope,
            "cache_hit": self.cache_hit,
            "telemetry": self.telemetry,
        }


class LocalRetriever:
    def __init__(
        self,
        session: Session,
        embeddings: EmbeddingProvider,
        cache: CacheBackend | None = None,
        *,
        acl_scope: str | list[str] = "internal",
        retriever_version: str = "cosine-v1",
        principal: Principal | None = None,
    ) -> None:
        self.session = session
        self.embeddings = embeddings
        self.cache = cache
        self.access_scope = _normalize_scope(acl_scope)
        self.acl_scope = ",".join(self.access_scope)
        self.retriever_version = retriever_version
        self.principal = principal

    @staticmethod
    def _cosine(left: list[float], right: list[float]) -> float:
        return sum(a * b for a, b in zip(left, right))

    def collection_version(self) -> str:
        documents = self.session.query(KnowledgeDocument).filter(
            KnowledgeDocument.embedding_provider == self.embeddings.provider_name
        ).all()
        return stable_hash(sorted(
            (document.checksum, document.chunk_count, document.embedding_provider, document.metadata_json)
            for document in documents
        ))

    def search(self, query: str, limit: int = 5) -> list[RetrievedChunk]:
        bounded_limit = max(1, min(limit, 20))
        collection_version = self.collection_version()
        cache_key = CacheKeyBuilder.retrieval(
            query,
            collection_version,
            self.acl_scope,
            self.retriever_version,
            bounded_limit,
        )
        if self.cache:
            cached = self.cache.get(CacheNamespace.retrieval.value, cache_key)
            if isinstance(cached, list):
                return [RetrievedChunk(**{**item, "cache_hit": True}) for item in cached if isinstance(item, dict)]
        query_vector = self.embeddings.embed_query(query)
        rows = self.session.query(KnowledgeChunkRecord, KnowledgeDocument).join(
            KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunkRecord.document_id
        ).filter(KnowledgeDocument.embedding_provider == self.embeddings.provider_name).all()
        ranked: list[RetrievedChunk] = []
        for chunk, document in rows:
            metadata = json.loads(chunk.metadata_json or "{}")
            document_metadata = json.loads(document.metadata_json or "{}")
            metadata = {**document_metadata, **metadata}
            if not _scope_allows(metadata, self.access_scope, self.principal):
                continue
            stored_vector = json.loads(chunk.embedding_json)
            if len(stored_vector) != self.embeddings.dimension:
                continue
            score = self._cosine(query_vector, stored_vector)
            ranked.append(RetrievedChunk(
                chunk_id=chunk.id, text=chunk.text, score=score,
                source={"file": document.filename, "page": chunk.page, "section": chunk.section,
                        "chunk_id": chunk.id, "document_id": document.id,
                        "document_hash": document.checksum, "revision": metadata.get("revision"),
                        "department": metadata.get("department"),
                        "classification": metadata.get("classification"),
                        "asset_id": metadata.get("asset_id")},
                document_id=document.id,
                scores={"dense": score, "sparse": None, "fusion": None, "reranker": None},
                retrieval_methods=["dense"],
                access_scope=_record_scope(metadata),
            ))
        results = sorted(ranked, key=lambda item: item.score, reverse=True)[:bounded_limit]
        if self.cache:
            self.cache.set(
                CacheNamespace.retrieval.value,
                cache_key,
                [item.to_dict() for item in results],
                metadata={
                    "collection_version": collection_version,
                    "acl_scope": self.acl_scope,
                    "retriever_version": self.retriever_version,
                },
            )
        return results


def _normalize_scope(scope: str | list[str]) -> list[str]:
    values = [scope] if isinstance(scope, str) else scope
    normalized = sorted({str(value).strip().lower() for value in values if str(value).strip()})
    return normalized or ["internal"]


def _record_scope(metadata: dict[str, object]) -> list[str]:
    explicit = metadata.get("access_scope")
    if isinstance(explicit, list):
        values = [str(value).lower() for value in explicit]
    elif explicit:
        values = [str(explicit).lower()]
    else:
        values = []
    for key in ("classification", "department"):
        if metadata.get(key):
            values.append(str(metadata[key]).lower())
    return sorted(set(values or ["internal"]))


def _document_acl(metadata: dict[str, object]) -> DocumentACL | None:
    organization_id = metadata.get("organization_id")
    workspace_id = metadata.get("workspace_id")
    if not organization_id or not workspace_id:
        return None
    try:
        return DocumentACL(
            organization_id=str(organization_id),
            department_id=str(metadata["department_id"]) if metadata.get("department_id") else (
                str(metadata["department"]) if metadata.get("department") else None
            ),
            workspace_id=str(workspace_id),
            classification=ClearanceLevel.parse(str(metadata.get("classification", "INTERNAL"))),
            allowed_roles=[Role(str(value).upper()) for value in metadata.get("allowed_roles", []) if str(value).upper() in {item.value for item in Role}],
            allowed_users=[str(value) for value in metadata.get("allowed_users", [])],
            owner_id=str(metadata["owner_id"]) if metadata.get("owner_id") else None,
        )
    except (KeyError, ValueError, TypeError):
        return None


def _scope_allows(
    metadata: dict[str, object], requested_scope: list[str], principal: Principal | None = None
) -> bool:
    if principal is not None:
        acl = _document_acl(metadata)
        return bool(acl and AuthorizationService().can_read_document(principal, acl).allowed)
    if "*" in requested_scope:
        return True
    record_scope = _record_scope(metadata)
    return bool(set(requested_scope) & set(record_scope)) or (
        requested_scope == ["internal"] and not metadata.get("classification")
    )
