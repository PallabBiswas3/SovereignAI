from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.database import KnowledgeChunkRecord, KnowledgeDocument
from app.rag.embeddings import EmbeddingProvider
from app.rag.retrieval import LocalRetriever, RetrievedChunk, _normalize_scope, _record_scope, _scope_allows
from app.resources.cache import CacheBackend, CacheKeyBuilder, CacheNamespace, stable_hash


class CandidateReranker(Protocol):
    @property
    def identity(self) -> str: ...

    @property
    def version(self) -> str: ...

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]: ...


@dataclass(slots=True)
class HybridRetrievalTelemetry:
    dense_duration_ms: float = 0.0
    sparse_duration_ms: float = 0.0
    fusion_duration_ms: float = 0.0
    reranker_duration_ms: float | None = None
    dense_count: int = 0
    sparse_count: int = 0
    fused_count: int = 0
    output_count: int = 0
    reranker_available: bool | None = None
    warning: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "dense_duration_ms": self.dense_duration_ms,
            "sparse_duration_ms": self.sparse_duration_ms,
            "fusion_duration_ms": self.fusion_duration_ms,
            "reranker_duration_ms": self.reranker_duration_ms,
            "dense_count": self.dense_count,
            "sparse_count": self.sparse_count,
            "fused_count": self.fused_count,
            "output_count": self.output_count,
            "reranker_available": self.reranker_available,
            "warning": self.warning,
        }


class BM25Retriever:
    """Fully local Okapi BM25 over the current SQLite chunk corpus."""

    def __init__(
        self,
        session: Session,
        *,
        access_scope: str | list[str] = "internal",
        version: str = "bm25-v1",
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self.session = session
        self.access_scope = _normalize_scope(access_scope)
        self.version = version
        self.k1 = k1
        self.b = b

    @staticmethod
    def tokenize(text: str) -> list[str]:
        lowered = text.lower().replace("≤", " <= ").replace("≥", " >= ")
        primary = re.findall(r"[a-z]+(?:-[a-z0-9]+)+|[a-z]+\d+|\d+(?:\.\d+)?|[a-z]+", lowered)
        expanded: list[str] = []
        for token in primary:
            expanded.append(token)
            if "-" in token:
                expanded.extend(part for part in token.split("-") if part)
        return expanded

    def search(self, query: str, limit: int = 30) -> list[RetrievedChunk]:
        rows = self.session.query(KnowledgeChunkRecord, KnowledgeDocument).join(
            KnowledgeDocument, KnowledgeDocument.id == KnowledgeChunkRecord.document_id
        ).all()
        corpus: list[tuple[KnowledgeChunkRecord, KnowledgeDocument, dict[str, object], list[str]]] = []
        for chunk, document in rows:
            import json

            metadata = {
                **json.loads(document.metadata_json or "{}"),
                **json.loads(chunk.metadata_json or "{}"),
            }
            if _scope_allows(metadata, self.access_scope):
                corpus.append((chunk, document, metadata, self.tokenize(chunk.text)))
        if not corpus:
            return []
        query_terms = self.tokenize(query)
        if not query_terms:
            return []
        document_frequency = Counter()
        for *_, tokens in corpus:
            document_frequency.update(set(tokens))
        average_length = sum(len(tokens) for *_, tokens in corpus) / len(corpus)
        ranked: list[RetrievedChunk] = []
        for chunk, document, metadata, tokens in corpus:
            counts = Counter(tokens)
            score = 0.0
            for term in query_terms:
                frequency = counts[term]
                if not frequency:
                    continue
                df = document_frequency[term]
                idf = math.log(1 + (len(corpus) - df + 0.5) / (df + 0.5))
                denominator = frequency + self.k1 * (
                    1 - self.b + self.b * len(tokens) / max(1.0, average_length)
                )
                score += idf * (frequency * (self.k1 + 1)) / denominator
            if score <= 0:
                continue
            source = {
                "file": document.filename,
                "page": chunk.page,
                "section": chunk.section,
                "chunk_id": chunk.id,
                "document_id": document.id,
                "document_hash": document.checksum,
                "revision": metadata.get("revision"),
                "department": metadata.get("department"),
                "classification": metadata.get("classification"),
            }
            ranked.append(RetrievedChunk(
                chunk_id=chunk.id,
                document_id=document.id,
                text=chunk.text,
                score=score,
                source=source,
                scores={"dense": None, "sparse": score, "fusion": None, "reranker": None},
                retrieval_methods=["bm25"],
                access_scope=_record_scope(metadata),
            ))
        return sorted(ranked, key=lambda item: (
            -item.score, str(item.source.get("file", "")),
            int(item.source.get("page") or 0), str(item.source.get("section") or ""), item.chunk_id,
        ))[:max(1, limit)]


class ReciprocalRankFusion:
    version = "rrf-v1"

    def __init__(self, k: int = 60) -> None:
        if k < 1:
            raise ValueError("RRF k must be positive")
        self.k = k

    def fuse(
        self,
        rankings: list[tuple[str, list[RetrievedChunk]]],
        limit: int,
    ) -> list[RetrievedChunk]:
        candidates: dict[str, RetrievedChunk] = {}
        fusion_scores: defaultdict[str, float] = defaultdict(float)
        for method, ranking in rankings:
            for rank, item in enumerate(ranking, start=1):
                fusion_scores[item.chunk_id] += 1.0 / (self.k + rank)
                current = candidates.get(item.chunk_id)
                if current is None:
                    current = RetrievedChunk(**item.to_dict())
                    candidates[item.chunk_id] = current
                if method not in current.retrieval_methods:
                    current.retrieval_methods.append(method)
                score_key = "dense" if method == "dense" else "sparse"
                current.scores[score_key] = item.score
        for chunk_id, item in candidates.items():
            item.scores["fusion"] = fusion_scores[chunk_id]
            item.score = fusion_scores[chunk_id]
        return sorted(
            candidates.values(), key=lambda item: (
                -item.score, str(item.source.get("file", "")),
                int(item.source.get("page") or 0), str(item.source.get("section") or ""), item.chunk_id,
            )
        )[:max(1, limit)]


class HybridRetriever:
    def __init__(
        self,
        session: Session,
        embeddings: EmbeddingProvider,
        cache: CacheBackend | None = None,
        *,
        access_scope: str | list[str] = "internal",
        reranker: CandidateReranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self.access_scope = _normalize_scope(access_scope)
        self.embeddings = embeddings
        self.cache = cache
        self.dense = LocalRetriever(
            session,
            embeddings,
            None,
            acl_scope=self.access_scope,
            retriever_version=self.settings.dense_retriever_version,
        )
        self.sparse = BM25Retriever(
            session,
            access_scope=self.access_scope,
            version=self.settings.bm25_index_version,
        )
        self.fusion = ReciprocalRankFusion(self.settings.hybrid_rrf_k)
        self.reranker = reranker
        self.last_telemetry = HybridRetrievalTelemetry()

    def collection_version(self) -> str:
        return self.dense.collection_version()

    def search(self, query: str, limit: int | None = None) -> list[RetrievedChunk]:
        final_limit = max(1, min(limit or self.settings.hybrid_final_context_k, 20))
        limits = {
            "dense_top_k": self.settings.hybrid_dense_top_k,
            "sparse_top_k": self.settings.hybrid_sparse_top_k,
            "fusion_candidate_limit": self.settings.hybrid_fusion_candidate_limit,
            "rerank_top_k": self.settings.hybrid_rerank_top_k,
            "final_context_k": final_limit,
            "rrf_k": self.settings.hybrid_rrf_k,
        }
        reranker_identity = (
            str(getattr(self.reranker, "cache_identity", self.reranker.identity))
            if self.reranker else "none"
        )
        reranker_version = self.reranker.version if self.reranker else "none"
        key = CacheKeyBuilder.hybrid_retrieval(
            query,
            collection_version=self.collection_version(),
            embedding_model=self.embeddings.provider_name,
            dense_version=self.settings.dense_retriever_version,
            bm25_version=self.settings.bm25_index_version,
            fusion_version=self.settings.fusion_strategy_version,
            reranker_identity=reranker_identity,
            reranker_version=reranker_version,
            access_scope=self.access_scope,
            limits=limits,
        )
        if self.cache:
            cached = self.cache.get(CacheNamespace.retrieval.value, key)
            if isinstance(cached, dict) and isinstance(cached.get("results"), list):
                self.last_telemetry = HybridRetrievalTelemetry(**cached.get("telemetry", {}))
                return [
                    RetrievedChunk(**{**item, "cache_hit": True})
                    for item in cached["results"] if isinstance(item, dict)
                ]

        telemetry = HybridRetrievalTelemetry()
        started = monotonic()
        dense = self.dense.search(query, self.settings.hybrid_dense_top_k)
        telemetry.dense_duration_ms = round((monotonic() - started) * 1000, 6)
        telemetry.dense_count = len(dense)
        started = monotonic()
        sparse = self.sparse.search(query, self.settings.hybrid_sparse_top_k)
        telemetry.sparse_duration_ms = round((monotonic() - started) * 1000, 6)
        telemetry.sparse_count = len(sparse)
        started = monotonic()
        fused = self.fusion.fuse(
            [("dense", dense), ("bm25", sparse)],
            self.settings.hybrid_fusion_candidate_limit,
        )
        telemetry.fusion_duration_ms = round((monotonic() - started) * 1000, 6)
        telemetry.fused_count = len(fused)

        ranked = fused
        if self.reranker and fused:
            started = monotonic()
            try:
                ranked = self.reranker.rerank(
                    query, fused, min(self.settings.hybrid_rerank_top_k, len(fused))
                )
                telemetry.reranker_available = True
            except Exception as exc:
                telemetry.reranker_available = False
                telemetry.warning = f"RERANKER_UNAVAILABLE: {exc}"
                ranked = fused
            telemetry.reranker_duration_ms = round((monotonic() - started) * 1000, 6)
        output = ranked[:final_limit]
        telemetry.output_count = len(output)
        shared_telemetry = telemetry.to_dict()
        for item in output:
            item.telemetry = shared_telemetry
        self.last_telemetry = telemetry
        if self.cache:
            self.cache.set(
                CacheNamespace.retrieval.value,
                key,
                {"results": [item.to_dict() for item in output], "telemetry": shared_telemetry},
                metadata={
                    "pipeline": "hybrid",
                    "identity": stable_hash({
                        "dense": self.settings.dense_retriever_version,
                        "bm25": self.settings.bm25_index_version,
                        "fusion": self.settings.fusion_strategy_version,
                        "reranker": reranker_identity,
                    }),
                },
            )
        return output
