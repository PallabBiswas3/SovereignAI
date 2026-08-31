from __future__ import annotations

from abc import ABC, abstractmethod


class GraphRetriever(ABC):
    """Extension seam for future entity/relationship retrieval."""

    @abstractmethod
    def retrieve(self, query: str, entity_types: list[str] | None = None) -> list[dict[str, object]]:
        raise NotImplementedError


class DisabledGraphRetriever(GraphRetriever):
    def retrieve(self, query: str, entity_types: list[str] | None = None) -> list[dict[str, object]]:
        return []

