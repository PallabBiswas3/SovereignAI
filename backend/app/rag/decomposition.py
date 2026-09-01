from __future__ import annotations

import re


class QueryDecomposer:
    """Bounded deterministic decomposition for DEEP retrieval only."""

    ASPECTS = (
        (("latest", "inspection", "finding"), "latest inspection findings"),
        (("histor", "trend", "reading"), "historical readings and trends"),
        (("limit", "requirement", "sop", "standard"), "applicable operating limits and maintenance requirements"),
        (("replace", "replacement"), "replacement criteria"),
        (("shutdown", "critical"), "shutdown and critical intervention criteria"),
    )

    def __init__(self, max_subqueries: int = 4) -> None:
        self.max_subqueries = max(1, max_subqueries)

    def decompose(self, query: str, execution_mode: str) -> list[str]:
        if execution_mode.upper() != "DEEP":
            return [query]
        lowered = query.lower()
        identifiers = re.findall(r"\b[A-Z]{1,8}(?:-[A-Z0-9]+)+\b|\bPump[- ]?\d+\b", query, re.I)
        asset = identifiers[0] if identifiers else "the referenced asset"
        subqueries: list[str] = []
        for triggers, description in self.ASPECTS:
            if any(trigger in lowered for trigger in triggers):
                subqueries.append(f"{asset} {description}")
        if not subqueries:
            clauses = [part.strip() for part in re.split(r",|\band\b|;", query, flags=re.I) if len(part.strip()) > 5]
            subqueries.extend(f"{asset} {clause}" for clause in clauses)
        deduplicated: list[str] = []
        for item in subqueries:
            if item.lower() not in {existing.lower() for existing in deduplicated}:
                deduplicated.append(item)
        return (deduplicated or [query])[: self.max_subqueries]


class ModeAwareRetrievalPipeline:
    def __init__(self, retriever, max_subqueries: int = 4) -> None:
        self.retriever = retriever
        self.decomposer = QueryDecomposer(max_subqueries)
        self.last_subqueries: list[str] = []

    def search(self, query: str, execution_mode: str, limit: int) -> list:
        self.last_subqueries = self.decomposer.decompose(query, execution_mode)
        merged: dict[str, object] = {}
        for subquery in self.last_subqueries:
            for candidate in self.retriever.search(subquery, limit):
                existing = merged.get(candidate.chunk_id)
                if existing is None or candidate.score > existing.score:
                    merged[candidate.chunk_id] = candidate
        return sorted(merged.values(), key=lambda item: (
            -item.score, str(item.source.get("file", "")),
            int(item.source.get("page") or 0), str(item.source.get("section") or ""), item.chunk_id,
        ))[:limit]
