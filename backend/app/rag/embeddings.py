from __future__ import annotations

import hashlib
import math
import re
from abc import ABC, abstractmethod
from functools import lru_cache
from app.resources.cache import CacheBackend, CacheKeyBuilder, CacheNamespace, get_cache_backend


class EmbeddingProvider(ABC):
    """Model-independent interface for local document and query embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        raise NotImplementedError

    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError

    @abstractmethod
    def embed_query(self, text: str) -> list[float]:
        raise NotImplementedError

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_documents(texts)


class LocalHashEmbeddingProvider(EmbeddingProvider):
    """Dependency-free deterministic baseline and emergency fallback."""

    def __init__(self, dimensions: int = 384) -> None:
        self._dimension = dimensions

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return "local-feature-hash"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_one(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_one(text)

    def _embed_one(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = re.findall(r"[a-z0-9]+", text.lower())
        features = tokens + [f"{left}_{right}" for left, right in zip(tokens, tokens[1:])]
        for feature in features:
            digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "little") % self.dimension
            sign = 1.0 if digest[4] & 1 else -1.0
            vector[index] += sign
        magnitude = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / magnitude for value in vector]


class SentenceTransformerEmbeddingProvider(EmbeddingProvider):
    """Local transformer encoder with mean pooling and normalized embeddings.

    The model is loaded from the local Hugging Face cache by default. Runtime
    downloads are disabled, preserving air-gap behavior.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        *,
        local_files_only: bool = True,
        device: str = "cpu",
        batch_size: int = 32,
    ) -> None:
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.batch_size = batch_size
        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=local_files_only)
        self._model = AutoModel.from_pretrained(model_name, local_files_only=local_files_only)
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._dimension = int(self._model.config.hidden_size)

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def provider_name(self) -> str:
        return f"sentence-transformer:{self.model_name}"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        results: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            results.extend(self._encode_batch(texts[start : start + self.batch_size]))
        return results

    def embed_query(self, text: str) -> list[float]:
        return self._encode_batch([text])[0]

    def _encode_batch(self, texts: list[str]) -> list[list[float]]:
        encoded = self._tokenizer(
            texts, padding=True, truncation=True, max_length=512, return_tensors="pt"
        )
        encoded = {key: value.to(self._device) for key, value in encoded.items()}
        with self._torch.inference_mode():
            hidden = self._model(**encoded).last_hidden_state
            mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
            pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
        return normalized.cpu().tolist()


class CachedEmbeddingProvider(EmbeddingProvider):
    """Caches vectors by text hash plus the exact embedding model identity."""

    def __init__(self, provider: EmbeddingProvider, cache: CacheBackend) -> None:
        self.provider = provider
        self.cache = cache

    @property
    def dimension(self) -> int:
        return self.provider.dimension

    @property
    def provider_name(self) -> str:
        return self.provider.provider_name

    @property
    def model_identity(self) -> str:
        return f"{self.provider_name}:dim={self.dimension}:pipeline=v1"

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        results: list[list[float] | None] = [None] * len(texts)
        missing_indexes: list[int] = []
        missing_texts: list[str] = []
        for index, text in enumerate(texts):
            key = CacheKeyBuilder.embedding(text, self.model_identity)
            cached = self.cache.get(CacheNamespace.embedding.value, key)
            if isinstance(cached, list) and len(cached) == self.dimension:
                results[index] = [float(value) for value in cached]
            else:
                missing_indexes.append(index)
                missing_texts.append(text)
        if missing_texts:
            generated = self.provider.embed_documents(missing_texts)
            for index, text, vector in zip(missing_indexes, missing_texts, generated):
                results[index] = vector
                self._store(text, vector)
        return [vector for vector in results if vector is not None]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_cached(text)

    def _embed_cached(self, text: str) -> list[float]:
        key = CacheKeyBuilder.embedding(text, self.model_identity)
        cached = self.cache.get(CacheNamespace.embedding.value, key)
        if isinstance(cached, list) and len(cached) == self.dimension:
            return [float(value) for value in cached]
        vector = self.provider.embed_query(text)
        self._store(text, vector)
        return vector

    def _store(self, text: str, vector: list[float]) -> None:
        key = CacheKeyBuilder.embedding(text, self.model_identity)
        self.cache.set(
            CacheNamespace.embedding.value,
            key,
            vector,
            metadata={"model_identity": self.model_identity},
        )


@lru_cache(maxsize=2)
def configured_embedding_provider() -> EmbeddingProvider:
    from app.core.config import get_settings

    settings = get_settings()
    if settings.embedding_provider.lower() == "hash":
        provider: EmbeddingProvider = LocalHashEmbeddingProvider()
    else:
        try:
            provider = SentenceTransformerEmbeddingProvider(
            settings.embedding_model,
            local_files_only=settings.embedding_local_files_only,
        )
        except Exception:
            if not settings.embedding_allow_hash_fallback:
                raise
            provider = LocalHashEmbeddingProvider()
    return CachedEmbeddingProvider(provider, get_cache_backend()) if settings.cache_enabled else provider
