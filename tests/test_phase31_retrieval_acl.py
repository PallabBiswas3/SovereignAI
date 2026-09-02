from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import Settings
from app.core.database import Base
from app.evidence.context import ContextCompiler
from app.identity.models import ClearanceLevel, DocumentACL, Permission, Principal, Role
from app.rag.embeddings import LocalHashEmbeddingProvider
from app.rag.hybrid import HybridRetriever
from app.rag.ingestion import KnowledgeIngestionService
from app.resources.cache import SQLiteCache


class SpyReranker:
    identity = "spy-local"
    version = "test-v1"

    def __init__(self) -> None:
        self.seen: list[str] = []

    @property
    def cache_identity(self) -> str:
        return self.identity

    def rerank(self, query, candidates, top_k):
        self.seen = [item.source["file"] for item in candidates]
        return candidates[:top_k]


def principal(*, manager: bool = False) -> Principal:
    return Principal(
        user_id="manager-1" if manager else "maint-1",
        email="user@apel.local", display_name="User", organization_id="apel",
        department_ids=["management"] if manager else ["maintenance"],
        workspace_ids=["plant-a"],
        roles=[Role.manager] if manager else [Role.engineer],
        clearance=ClearanceLevel.restricted if manager else ClearanceLevel.confidential,
        permissions=[Permission.knowledge_read] + ([Permission.knowledge_read_cross_department] if manager else []),
    )


def corpus(tmp_path: Path):
    engine = create_engine(f"sqlite:///{(tmp_path / 'acl.db').as_posix()}")
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    db = factory()
    embeddings = LocalHashEmbeddingProvider(64)
    ingest = KnowledgeIngestionService(db, embeddings)
    maintenance = tmp_path / "Maintenance_SOP.md"
    maintenance.write_text("Plant maintenance schedule and rotating equipment inspection guidance.", encoding="utf-8")
    finance = tmp_path / "Executive_Compensation_2026.md"
    finance.write_text("Executive compensation 2026 payroll amount is 999999 units.", encoding="utf-8")
    ingest.ingest(maintenance, acl=DocumentACL(
        organization_id="apel", department_id="maintenance", workspace_id="plant-a",
        classification=ClearanceLevel.confidential, owner_id="maint-1",
    ), require_acl=True)
    ingest.ingest(finance, acl=DocumentACL(
        organization_id="apel", department_id="finance", workspace_id="plant-a",
        classification=ClearanceLevel.restricted, owner_id="finance-1",
    ), require_acl=True)
    return db, embeddings, SQLiteCache(factory)


def settings() -> Settings:
    return Settings(
        reranker_enabled=False, hybrid_dense_top_k=20, hybrid_sparse_top_k=20,
        hybrid_fusion_candidate_limit=20, hybrid_rerank_top_k=20,
        hybrid_final_context_k=10,
    )


def test_unauthorized_perfect_match_never_reaches_ranking_or_context(tmp_path: Path) -> None:
    db, embeddings, cache = corpus(tmp_path)
    spy = SpyReranker()
    retriever = HybridRetriever(
        db, embeddings, cache, principal=principal(), reranker=spy, settings=settings()
    )
    results = retriever.search("executive compensation 2026 payroll", 10)
    serialized = str([item.to_dict() for item in results])
    assert "Executive_Compensation" not in serialized
    assert "999999" not in serialized
    assert all(name != "Executive_Compensation_2026.md" for name in spy.seen)
    compiled = ContextCompiler().compile(
        task="executive compensation 2026 payroll", evidence=results,
        selected_model="test", context_window=4096, execution_mode="STANDARD",
    )
    assert "999999" not in compiled.model_dump_json()


def test_authorization_fingerprint_prevents_privileged_cache_reuse(tmp_path: Path) -> None:
    db, embeddings, cache = corpus(tmp_path)
    broad = HybridRetriever(db, embeddings, cache, principal=principal(manager=True), settings=settings())
    broad_results = broad.search("executive compensation 2026 payroll", 10)
    assert any(item.source["file"] == "Executive_Compensation_2026.md" for item in broad_results)
    narrow = HybridRetriever(db, embeddings, cache, principal=principal(), settings=settings())
    narrow_results = narrow.search("executive compensation 2026 payroll", 10)
    assert all(item.source["file"] != "Executive_Compensation_2026.md" for item in narrow_results)
    assert broad.authorization_fingerprint != narrow.authorization_fingerprint
    assert all(not item.cache_hit for item in narrow_results)
    assert cache.stats()["entries"] == 2


def test_secure_ingestion_requires_explicit_acl(tmp_path: Path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    db = Session(engine)
    path = tmp_path / "unscoped.md"
    path.write_text("unscoped organizational content", encoding="utf-8")
    try:
        KnowledgeIngestionService(db, LocalHashEmbeddingProvider(32)).ingest(path, require_acl=True)
    except ValueError as exc:
        assert str(exc) == "ACCESS_SCOPE_REQUIRED"
    else:
        raise AssertionError("Unscoped secure ingestion was accepted")
