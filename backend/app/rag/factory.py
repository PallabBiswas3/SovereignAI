from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.rag.embeddings import EmbeddingProvider, configured_embedding_provider
from app.rag.hybrid import HybridRetriever
from app.rag.reranking import LocalCrossEncoderReranker, Reranker
from app.resources.cache import CacheBackend, get_cache_backend
from app.identity.models import Principal


def configured_reranker(settings: Settings | None = None) -> Reranker | None:
    settings = settings or get_settings()
    if not settings.reranker_enabled:
        return None
    return LocalCrossEncoderReranker(
        settings.reranker_model,
        local_files_only=settings.reranker_local_files_only,
        version=settings.reranker_version,
    )


def configured_hybrid_retriever(
    session: Session,
    *,
    embeddings: EmbeddingProvider | None = None,
    cache: CacheBackend | None = None,
    access_scope: str | list[str] = "internal",
    reranker: Reranker | None = None,
    settings: Settings | None = None,
    execution_mode: str = "STANDARD",
    principal: Principal | None = None,
) -> HybridRetriever:
    settings = settings or get_settings()
    return HybridRetriever(
        session,
        embeddings or configured_embedding_provider(),
        cache if cache is not None else get_cache_backend() if settings.cache_enabled else None,
        access_scope=access_scope,
        reranker=(
            None if execution_mode.upper() == "FAST"
            else reranker if reranker is not None
            else configured_reranker(settings)
        ),
        settings=settings,
        principal=principal,
    )
