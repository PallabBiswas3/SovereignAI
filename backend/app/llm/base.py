from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
import asyncio
from typing import Any


@dataclass(slots=True)
class GenerationResult:
    text: str
    model: str
    provider: str
    fallback: bool = False
    runtime_stats: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class GenerationChunk:
    text: str
    model: str
    provider: str
    done: bool = False
    fallback: bool = False
    runtime_stats: dict[str, Any] = field(default_factory=dict)


class ModelGenerationCancelled(RuntimeError):
    """Raised when the caller or disconnected SSE client cancels local generation."""


@dataclass(slots=True)
class StructuredGenerationResult(GenerationResult):
    data: dict[str, Any] | None = None


class LocalModelProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, model: str, system: str | None = None) -> GenerationResult:
        raise NotImplementedError

    async def stream(
        self,
        prompt: str,
        model: str,
        system: str | None = None,
        *,
        cancellation_event: asyncio.Event | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        if cancellation_event and cancellation_event.is_set():
            raise ModelGenerationCancelled("Local model generation was cancelled before it started.")
        result = await self.generate(prompt, model, system)
        yield GenerationChunk(
            text=result.text,
            model=result.model,
            provider=result.provider,
            done=True,
            fallback=result.fallback,
            runtime_stats=result.runtime_stats,
        )

    @abstractmethod
    async def generate_json(
        self, prompt: str, model: str, schema: dict[str, Any], system: str | None = None
    ) -> StructuredGenerationResult:
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> dict[str, object]:
        raise NotImplementedError

    async def list_available_models(self) -> list[dict[str, object]]:
        status = await self.health_check()
        models = status.get("models", [])
        return [item for item in models if isinstance(item, dict)] if isinstance(models, list) else []

    async def model_runtime_stats(self, model: str | None = None) -> dict[str, object]:
        return {"model": model, "available": False, "detail": "Runtime statistics are not exposed by this provider."}
