from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from time import monotonic
from pathlib import Path
import os
import hashlib

from app.rag.retrieval import RetrievedChunk
from app.resources.scheduler import ResourceScheduler, get_resource_scheduler


class RerankerUnavailable(RuntimeError):
    code = "RERANKER_UNAVAILABLE"


@dataclass(slots=True)
class RerankerTelemetry:
    reranker_duration_ms: float
    candidate_count: int
    output_count: int
    available: bool
    error: str | None = None


class Reranker(ABC):
    @property
    @abstractmethod
    def identity(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        raise NotImplementedError


class LocalCrossEncoderReranker(Reranker):
    """Lazy, CPU-only Transformers cross-encoder with runtime downloads disabled."""

    def __init__(
        self,
        model_name: str,
        *,
        local_files_only: bool = True,
        version: str = "cross-encoder-v1",
        batch_size: int = 8,
        scheduler: ResourceScheduler | None = None,
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self._version = version
        self.batch_size = batch_size
        self.scheduler = scheduler or get_resource_scheduler()
        self._tokenizer = None
        self._model = None
        self._torch = None
        self._load_error: str | None = None
        self.last_telemetry: RerankerTelemetry | None = None

    @property
    def identity(self) -> str:
        return f"local-cross-encoder:{self.model_name}"

    @property
    def version(self) -> str:
        return self._version

    @property
    def cache_identity(self) -> str:
        try:
            path = Path(self._resolve_local_model())
            config_hash = hashlib.sha256((path / "config.json").read_bytes()).hexdigest()
            return f"{self.identity}:local:{config_hash}"
        except (RerankerUnavailable, OSError):
            return f"{self.identity}:unavailable"

    @property
    def available(self) -> bool | None:
        if self._model is not None:
            return True
        if self._load_error is not None:
            return False
        return None

    def status(self) -> dict[str, object]:
        try:
            path = self._resolve_local_model()
            available = True
            error = None
        except RerankerUnavailable as exc:
            path = None
            available = False
            error = str(exc)
        return {
            "identity": self.identity,
            "version": self.version,
            "available": available,
            "device": "cpu",
            "local_path": path,
            "runtime_downloads": False,
            "fallback": "RRF fusion ranking" if not available else None,
            "error": error,
        }

    def _load(self) -> None:
        if self._model is not None:
            return
        if self._load_error:
            raise RerankerUnavailable(self._load_error)
        try:
            model_path = self._resolve_local_model()
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                model_path, local_files_only=True
            )
            self._model = AutoModelForSequenceClassification.from_pretrained(
                model_path, local_files_only=True
            )
            self._model.to("cpu")
            self._model.eval()
            self._torch = torch
        except Exception as exc:
            self._load_error = (
                f"{self.identity} is not available in the local model cache. "
                "Pre-stage the configured cross-encoder; runtime downloads are disabled."
            )
            raise RerankerUnavailable(self._load_error) from exc

    def _resolve_local_model(self) -> str:
        direct = Path(self.model_name)
        if direct.is_dir() and (direct / "config.json").is_file():
            return str(direct)
        cache_root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
        hub = cache_root / "hub" if cache_root.name != "hub" else cache_root
        model_root = hub / f"models--{self.model_name.replace('/', '--')}" / "snapshots"
        if model_root.is_dir():
            snapshots = sorted(
                (path for path in model_root.iterdir() if (path / "config.json").is_file()),
                key=lambda path: path.stat().st_mtime,
                reverse=True,
            )
            if snapshots:
                return str(snapshots[0])
        raise RerankerUnavailable(
            f"{self.identity} is not available in the local model cache. "
            "Pre-stage the configured cross-encoder; runtime downloads are disabled."
        )

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        started = monotonic()
        try:
            with self.scheduler.acquire_cpu_sync():
                self._load()
                assert self._tokenizer is not None and self._model is not None and self._torch is not None
                scores: list[float] = []
                for start in range(0, len(candidates), self.batch_size):
                    batch = candidates[start : start + self.batch_size]
                    encoded = self._tokenizer(
                        [query] * len(batch),
                        [item.text for item in batch],
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors="pt",
                    )
                    with self._torch.inference_mode():
                        logits = self._model(**encoded).logits
                    if logits.shape[-1] == 1:
                        batch_scores = self._torch.sigmoid(logits.squeeze(-1)).tolist()
                    else:
                        batch_scores = self._torch.softmax(logits, dim=-1)[:, -1].tolist()
                    if isinstance(batch_scores, float):
                        batch_scores = [batch_scores]
                    scores.extend(float(value) for value in batch_scores)
            for item, score in zip(candidates, scores):
                item.scores["reranker"] = score
                item.score = score
            output = sorted(
                candidates, key=lambda item: (-(item.scores.get("reranker") or 0.0), item.chunk_id)
            )[:max(1, top_k)]
            self.last_telemetry = RerankerTelemetry(
                reranker_duration_ms=round((monotonic() - started) * 1000, 6),
                candidate_count=len(candidates),
                output_count=len(output),
                available=True,
            )
            return output
        except RerankerUnavailable as exc:
            self.last_telemetry = RerankerTelemetry(
                reranker_duration_ms=round((monotonic() - started) * 1000, 6),
                candidate_count=len(candidates),
                output_count=0,
                available=False,
                error=str(exc),
            )
            raise


class LexicalTestReranker(Reranker):
    """Deterministic lightweight reranker for tests and offline diagnostics, not production default."""

    identity = "lexical-test-reranker"
    version = "lexical-v1"

    def rerank(
        self, query: str, candidates: list[RetrievedChunk], top_k: int
    ) -> list[RetrievedChunk]:
        from app.rag.hybrid import BM25Retriever

        query_terms = set(BM25Retriever.tokenize(query))
        for item in candidates:
            terms = set(BM25Retriever.tokenize(item.text))
            score = len(query_terms & terms) / max(1, len(query_terms))
            item.scores["reranker"] = score
            item.score = score
        return sorted(candidates, key=lambda item: (-item.score, item.chunk_id))[:max(1, top_k)]
