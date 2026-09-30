from __future__ import annotations

import re
from typing import Any

from app.core.config import Settings
from app.evidence.distiller import EvidenceDistiller
from app.rag.hybrid import HybridRetriever
from app.rag.retrieval import RetrievedChunk


class DistillingHybridRetriever:
    """Post-retrieval evidence view for generation-facing RAG consumers.

    The wrapped HybridRetriever still owns authorization, dense/BM25 retrieval,
    RRF, reranking, caching and ranking. This wrapper only replaces each selected
    chunk's model-facing text with a deterministic query-aware excerpt when the
    experimental distillation flag is enabled.
    """

    def __init__(self, retriever: HybridRetriever, settings: Settings) -> None:
        self.retriever = retriever
        self.settings = settings
        self.last_distillation: dict[str, Any] = {
            "distillation_applied": False,
            "selected_evidence_tokens": 0,
            "selected_sentence_count": 0,
            "technical_sentence_count": 0,
            "conflict_preserved_chunks": 0,
        }

    def __getattr__(self, name: str) -> Any:
        return getattr(self.retriever, name)

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)))

    @staticmethod
    def _clone_with_text(item: RetrievedChunk, text: str) -> RetrievedChunk:
        payload = item.to_dict()
        payload["text"] = text
        return RetrievedChunk(**payload)

    def search(
        self,
        query: str,
        limit: int | None = None,
        *,
        asset_id: str | None = None,
    ) -> list[RetrievedChunk]:
        requested_limit = max(
            1,
            min(limit or self.settings.hybrid_final_context_k, 20),
        )
        enabled = bool(self.settings.context_distillation_enabled)
        if not enabled:
            results = self.retriever.search(query, requested_limit, asset_id=asset_id)
            self.last_distillation = {
                "distillation_applied": False,
                "selected_evidence_tokens": sum(self.estimate_tokens(item.text) for item in results),
                "selected_sentence_count": 0,
                "technical_sentence_count": 0,
                "conflict_preserved_chunks": 0,
                "candidate_count": len(results),
                "output_count": len(results),
            }
            return results

        # Give the distiller a small ranked reserve beyond the final chunk cap so
        # it can retain a conflicting revision that would otherwise fall just
        # below top-k. Ranking/retrieval itself is unchanged.
        candidate_limit = max(
            requested_limit,
            min(max(requested_limit, self.settings.hybrid_rerank_top_k), 20),
        )
        candidates = self.retriever.search(query, candidate_limit, asset_id=asset_id)
        target = max(1, int(self.settings.context_distillation_target_tokens))
        result = EvidenceDistiller(
            estimate_tokens=self.estimate_tokens,
            sentence_redundancy_threshold=float(
                self.settings.context_distillation_sentence_redundancy_threshold
            ),
        ).distill(
            task=query,
            chunks=candidates,
            token_budget=target,
            chunk_cap=requested_limit,
        )
        output = [self._clone_with_text(item, text) for item, text in result.selected]
        self.last_distillation = {
            "distillation_applied": True,
            "target_evidence_tokens": target,
            "selected_evidence_tokens": result.selected_tokens,
            "selected_sentence_count": result.sentence_count,
            "technical_sentence_count": result.technical_sentence_count,
            "conflict_preserved_chunks": result.conflict_preserved_chunks,
            "candidate_count": len(candidates),
            "output_count": len(output),
        }
        for item in output:
            item.telemetry = {
                **item.telemetry,
                "evidence_distillation": dict(self.last_distillation),
            }
        return output
