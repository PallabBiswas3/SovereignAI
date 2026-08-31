from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class GenerationResult:
    text: str
    model: str
    provider: str
    fallback: bool = False


@dataclass(slots=True)
class StructuredGenerationResult(GenerationResult):
    data: dict[str, Any] | None = None


class LocalModelProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, model: str, system: str | None = None) -> GenerationResult:
        raise NotImplementedError

    async def stream(self, prompt: str, model: str, system: str | None = None) -> AsyncIterator[str]:
        result = await self.generate(prompt, model, system)
        yield result.text

    @abstractmethod
    async def generate_json(
        self, prompt: str, model: str, schema: dict[str, Any], system: str | None = None
    ) -> StructuredGenerationResult:
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> dict[str, object]:
        raise NotImplementedError
