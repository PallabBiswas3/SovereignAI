from __future__ import annotations

from adaptivefact.agents.schema import SearchResult
from adaptivefact.agents.tools.base import SearchTool
from adaptivefact.data.schema import Claim, ResponseRecord
from adaptivefact.verification.evidence import ContextTfidfIndex, EvidenceRetrieverConfig


class ContextSearchTool(SearchTool):
    name = "provided_context"

    def __init__(
        self,
        *,
        authority_score: float = 0.85,
        retrieval_config: EvidenceRetrieverConfig | None = None,
    ) -> None:
        self.authority_score = authority_score
        self.retrieval_config = retrieval_config or EvidenceRetrieverConfig()

    def search(
        self,
        query: str,
        *,
        record: ResponseRecord,
        claim: Claim,
        top_k: int = 3,
    ) -> list[SearchResult]:
        index = ContextTfidfIndex(record.context, self.retrieval_config)
        retrieved = index.retrieve(query, top_k=top_k)
        return [
            SearchResult(
                source_id=f"{record.source_id or record.id}:context:{item.index}",
                source_name="Provided grounding context",
                text=item.text,
                authority_score=self.authority_score,
                retrieval_score=max(0.0, item.score),
                metadata={"tool": self.name, "chunk_index": item.index},
            )
            for item in retrieved
        ]
