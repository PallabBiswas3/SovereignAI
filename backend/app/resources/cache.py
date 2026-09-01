from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from datetime import datetime, timedelta, timezone
from enum import Enum
from functools import lru_cache
from pathlib import Path
from threading import RLock
from typing import Any

from sqlalchemy.orm import Session, sessionmaker

from app.core.database import CacheRecord, SessionLocal


class CacheNamespace(str, Enum):
    ocr = "ocr"
    embedding = "embedding"
    retrieval = "retrieval"
    vision = "vision"
    deterministic = "deterministic"


class CacheBackend(ABC):
    @abstractmethod
    def get(self, namespace: str, key: str) -> Any | None:
        raise NotImplementedError

    @abstractmethod
    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def delete(self, namespace: str, key: str) -> bool:
        raise NotImplementedError

    @abstractmethod
    def stats(self) -> dict[str, int]:
        raise NotImplementedError


class SQLiteCache(CacheBackend):
    """Replaceable local JSON cache. It never stores final model responses by convention."""

    def __init__(self, session_factory: sessionmaker[Session] = SessionLocal) -> None:
        self.session_factory = session_factory
        self._hits = 0
        self._misses = 0
        self._lock = RLock()

    def get(self, namespace: str, key: str) -> Any | None:
        with self.session_factory() as session:
            record = session.get(CacheRecord, self._id(namespace, key))
            if record is None:
                self._count(hit=False)
                return None
            now = datetime.now(timezone.utc)
            expires_at = _as_utc(record.expires_at)
            if expires_at is not None and expires_at <= now:
                session.delete(record)
                session.commit()
                self._count(hit=False)
                return None
            record.last_accessed_at = now
            record.hit_count += 1
            session.commit()
            self._count(hit=True)
            return json.loads(record.value_json)

    def set(
        self,
        namespace: str,
        key: str,
        value: Any,
        *,
        ttl_seconds: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        now = datetime.now(timezone.utc)
        expires = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None
        record_id = self._id(namespace, key)
        with self.session_factory() as session:
            record = session.get(CacheRecord, record_id)
            if record is None:
                record = CacheRecord(id=record_id, namespace=namespace, cache_key=key)
                session.add(record)
            record.value_json = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
            record.metadata_json = json.dumps(metadata or {}, ensure_ascii=False, sort_keys=True, default=str)
            record.updated_at = now
            record.last_accessed_at = now
            record.expires_at = expires
            session.commit()

    def delete(self, namespace: str, key: str) -> bool:
        with self.session_factory() as session:
            record = session.get(CacheRecord, self._id(namespace, key))
            if record is None:
                return False
            session.delete(record)
            session.commit()
            return True

    def stats(self) -> dict[str, int]:
        with self.session_factory() as session:
            entries = session.query(CacheRecord).count()
        with self._lock:
            return {"hits": self._hits, "misses": self._misses, "entries": entries}

    def _count(self, *, hit: bool) -> None:
        with self._lock:
            if hit:
                self._hits += 1
            else:
                self._misses += 1

    @staticmethod
    def _id(namespace: str, key: str) -> str:
        return hashlib.sha256(f"{namespace}:{key}".encode("utf-8")).hexdigest()


class CacheKeyBuilder:
    @staticmethod
    def file(path: Path, pipeline_version: str, **identity: object) -> str:
        return stable_hash({
            "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "pipeline_version": pipeline_version,
            **identity,
        })

    @staticmethod
    def embedding(text: str, model_identity: str) -> str:
        return stable_hash({
            "text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
            "embedding_model_identity": model_identity,
        })

    @staticmethod
    def retrieval(
        query: str,
        collection_version: str,
        acl_scope: str,
        retriever_version: str,
        limit: int,
    ) -> str:
        return stable_hash({
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "collection_version": collection_version,
            "acl_scope": acl_scope,
            "retriever_version": retriever_version,
            "limit": limit,
        })

    @staticmethod
    def hybrid_retrieval(
        query: str,
        *,
        collection_version: str,
        embedding_model: str,
        dense_version: str,
        bm25_version: str,
        fusion_version: str,
        reranker_identity: str,
        reranker_version: str,
        access_scope: list[str],
        limits: dict[str, int],
    ) -> str:
        return stable_hash({
            "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
            "collection_version": collection_version,
            "embedding_model": embedding_model,
            "dense_version": dense_version,
            "bm25_version": bm25_version,
            "fusion_version": fusion_version,
            "reranker_identity": reranker_identity,
            "reranker_version": reranker_version,
            "access_scope": sorted(access_scope),
            "limits": limits,
        })

    @staticmethod
    def vision(path: Path, model: str, prompt: str, schema_version: str) -> str:
        return CacheKeyBuilder.file(
            path,
            schema_version,
            model=model,
            prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        )

    @staticmethod
    def deterministic(input_hashes: list[str], workflow_version: str, rule_version: str) -> str:
        return stable_hash({
            "input_hashes": sorted(input_hashes),
            "workflow_version": workflow_version,
            "rule_version": rule_version,
        })


def stable_hash(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _as_utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


@lru_cache(maxsize=1)
def get_cache_backend() -> CacheBackend:
    return SQLiteCache()
