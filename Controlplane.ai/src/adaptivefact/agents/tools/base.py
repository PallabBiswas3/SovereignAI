from __future__ import annotations

import abc

from adaptivefact.agents.schema import SearchResult
from adaptivefact.data.schema import Claim, ResponseRecord


class SearchTool(abc.ABC):
    """An approved evidence source available to the verification agent."""

    name: str

    @abc.abstractmethod
    def search(
        self,
        query: str,
        *,
        record: ResponseRecord,
        claim: Claim,
        top_k: int = 3,
    ) -> list[SearchResult]:
        raise NotImplementedError
