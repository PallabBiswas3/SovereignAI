from __future__ import annotations

import re
from time import monotonic
from typing import Any

from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.evidence.models import (
    Calculation,
    EvidenceConflict,
    EvidenceFragment,
    EvidenceSource,
    Measurement,
    Rule,
)
from app.rag.retrieval import RetrievedChunk


class ContextBudget(BaseModel):
    context_window: int
    max_fraction_of_window: float
    output_reserve_tokens: int
    max_evidence_tokens: int
    max_evidence_chunks: int
    available_input_tokens: int


class ContextCompilationMetrics(BaseModel):
    raw_candidate_count: int
    reranked_candidate_count: int
    final_evidence_count: int
    raw_evidence_tokens: int
    compiled_context_tokens: int
    deduplicated_chunks: int
    dropped_chunks: int
    compression_ratio: float
    compilation_duration_ms: float


class CompiledContext(BaseModel):
    task: dict[str, Any]
    model: dict[str, Any]
    evidence: list[EvidenceSource]
    fragments: list[EvidenceFragment]
    measurements: list[Measurement] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    verified_calculations: list[Calculation] = Field(default_factory=list)
    tool_observations: list[dict[str, Any]] = Field(default_factory=list)
    previous_verified_findings: list[dict[str, Any]] = Field(default_factory=list)
    conflicts: list[EvidenceConflict] = Field(default_factory=list)
    open_questions: list[str] = Field(default_factory=list)
    user_constraints: list[str] = Field(default_factory=list)
    budget: ContextBudget
    metrics: ContextCompilationMetrics
    warnings: list[str] = Field(default_factory=list)


class ContextCompiler:
    """Builds compact, provenance-preserving context; it never stores model reasoning."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()

    @staticmethod
    def estimate_tokens(text: str) -> int:
        return max(1, len(re.findall(r"\w+|[^\w\s]", text)))

    @staticmethod
    def _normalized(text: str) -> str:
        return " ".join(re.findall(r"[a-z0-9.°/%<>=-]+", text.lower()))

    @staticmethod
    def _token_set(text: str) -> set[str]:
        return set(re.findall(r"[a-z]+(?:-[a-z0-9]+)+|\d+(?:\.\d+)?|[a-z]+", text.lower()))

    def _near_duplicate(self, left: str, right: str) -> bool:
        left_tokens, right_tokens = self._token_set(left), self._token_set(right)
        if not left_tokens or not right_tokens:
            return False
        return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) >= self.settings.context_near_duplicate_threshold

    @staticmethod
    def _relevance(item: RetrievedChunk) -> float:
        for key in ("reranker", "fusion", "dense", "sparse"):
            value = item.scores.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return item.score

    @staticmethod
    def _revision_number(revision: object) -> int:
        match = re.search(r"\d+", str(revision or ""))
        return int(match.group()) if match else -1

    def _select_text(self, task: str, text: str, token_limit: int) -> str:
        if self.estimate_tokens(text) <= token_limit:
            return text.strip()
        task_terms = self._token_set(task)
        sentences = [value.strip() for value in re.split(r"(?<=[.!?])\s+|\n+", text) if value.strip()]
        ranked = sorted(
            enumerate(sentences),
            key=lambda pair: (
                -len(task_terms & self._token_set(pair[1])),
                -int(bool(re.search(r"\d|[A-Z]{1,5}-\d|mm/s|bar|°C|kPa", pair[1], re.I))),
                pair[0],
            ),
        )
        selected: list[tuple[int, str]] = []
        used = 0
        for index, sentence in ranked:
            tokens = self.estimate_tokens(sentence)
            if selected and used + tokens > token_limit:
                continue
            selected.append((index, sentence))
            used += tokens
            if used >= token_limit:
                break
        return " ".join(sentence for _, sentence in sorted(selected))

    def compile(
        self,
        *,
        task: str,
        evidence: list[RetrievedChunk],
        selected_model: str,
        context_window: int,
        execution_mode: str,
        task_profile: dict[str, Any] | None = None,
        measurements: list[Measurement] | None = None,
        rules: list[Rule] | None = None,
        calculations: list[Calculation] | None = None,
        tool_observations: list[dict[str, Any]] | None = None,
        previous_verified_findings: list[dict[str, Any]] | None = None,
        open_questions: list[str] | None = None,
        user_constraints: list[str] | None = None,
    ) -> CompiledContext:
        started = monotonic()
        raw_tokens = sum(self.estimate_tokens(item.text) for item in evidence)
        max_fraction_tokens = int(context_window * self.settings.context_max_fraction_of_window)
        available = max(0, max_fraction_tokens - self.settings.context_output_reserve_tokens)
        mode = execution_mode.upper()
        mode_token_cap = 1000 if mode == "FAST" else self.settings.context_max_evidence_tokens
        mode_chunk_cap = min(3, self.settings.context_max_evidence_chunks) if mode == "FAST" else self.settings.context_max_evidence_chunks
        evidence_budget = min(mode_token_cap, available)
        budget = ContextBudget(
            context_window=context_window,
            max_fraction_of_window=self.settings.context_max_fraction_of_window,
            output_reserve_tokens=self.settings.context_output_reserve_tokens,
            max_evidence_tokens=evidence_budget,
            max_evidence_chunks=mode_chunk_cap,
            available_input_tokens=available,
        )

        ordered = sorted(
            evidence,
            key=lambda item: (
                -self._revision_number(item.source.get("revision")),
                -self._relevance(item),
                item.chunk_id,
            ),
        )
        unique: list[RetrievedChunk] = []
        exact_seen: set[str] = set()
        deduplicated = 0
        for item in ordered:
            normalized = self._normalized(item.text)
            same_document_duplicate = any(
                existing.document_id == item.document_id
                and self._near_duplicate(existing.text, item.text)
                for existing in unique
            )
            if normalized in exact_seen or same_document_duplicate:
                deduplicated += 1
                continue
            exact_seen.add(normalized)
            unique.append(item)

        selected: list[tuple[RetrievedChunk, str]] = []
        used_tokens = 0
        warnings: list[str] = []
        if evidence_budget <= 0 and unique:
            warnings.append("CONTEXT_BUDGET_EXCEEDED")
        for item in unique:
            if len(selected) >= mode_chunk_cap:
                break
            remaining = evidence_budget - used_tokens
            if remaining <= 0:
                break
            text = self._select_text(task, item.text, remaining)
            tokens = self.estimate_tokens(text)
            if not text or tokens > remaining:
                continue
            selected.append((item, text))
            used_tokens += tokens

        sources: list[EvidenceSource] = []
        fragments: list[EvidenceFragment] = []
        for index, (item, text) in enumerate(selected, start=1):
            source_id = f"E{index}"
            sources.append(EvidenceSource(
                id=source_id,
                document_id=item.document_id,
                file=str(item.source.get("file", "unknown")),
                page=item.source.get("page") if isinstance(item.source.get("page"), int) else None,
                section=str(item.source["section"]) if item.source.get("section") is not None else None,
                revision=str(item.source["revision"]) if item.source.get("revision") else None,
                document_hash=str(item.source["document_hash"]) if item.source.get("document_hash") else None,
                text=text,
                retrieval_score=float(item.scores.get("fusion") or item.score),
                reranker_score=float(item.scores["reranker"]) if isinstance(item.scores.get("reranker"), (int, float)) else None,
                access_scope=item.access_scope or ["internal"],
            ))
            fragments.append(EvidenceFragment(
                id=f"F{index}",
                source_id=source_id,
                text=text,
                scores=item.scores,
                retrieval_methods=item.retrieval_methods,
                technical_identifiers=sorted(set(re.findall(r"\b[A-Z]{1,8}(?:-[A-Z0-9]+)+\b", text))),
                numerical_values=re.findall(r"\b\d+(?:\.\d+)?\s*(?:MPa|kPa|Pa|bar|mm/s|m/s|°C|K|rpm|Hz|kW|MW|W)?", text, re.I),
            ))

        conflicts = self._revision_conflicts(sources)
        compiled_tokens = used_tokens + self.estimate_tokens(task)
        metrics = ContextCompilationMetrics(
            raw_candidate_count=len(evidence),
            reranked_candidate_count=len(ordered),
            final_evidence_count=len(sources),
            raw_evidence_tokens=raw_tokens,
            compiled_context_tokens=compiled_tokens,
            deduplicated_chunks=deduplicated,
            dropped_chunks=max(0, len(evidence) - len(sources) - deduplicated),
            compression_ratio=round(compiled_tokens / max(1, raw_tokens + self.estimate_tokens(task)), 4),
            compilation_duration_ms=round((monotonic() - started) * 1000, 6),
        )
        return CompiledContext(
            task={"goal": task, "profile": task_profile or {}, "execution_mode": execution_mode},
            model={"id": selected_model, "context_window": context_window},
            evidence=sources,
            fragments=fragments,
            measurements=measurements or [],
            rules=rules or [],
            verified_calculations=calculations or [],
            tool_observations=tool_observations or [],
            previous_verified_findings=previous_verified_findings or [],
            conflicts=conflicts,
            open_questions=open_questions or [],
            user_constraints=user_constraints or [],
            budget=budget,
            metrics=metrics,
            warnings=warnings,
        )

    @staticmethod
    def _revision_conflicts(sources: list[EvidenceSource]) -> list[EvidenceConflict]:
        groups: dict[tuple[str, str | None], list[EvidenceSource]] = {}
        for source in sources:
            key = (source.file.lower(), source.section)
            groups.setdefault(key, []).append(source)
        conflicts: list[EvidenceConflict] = []
        for group in groups.values():
            revisions = {source.revision for source in group if source.revision}
            numeric_sets = {tuple(re.findall(r"\d+(?:\.\d+)?", source.text)) for source in group}
            if len(revisions) > 1 and len(numeric_sets) > 1:
                conflicts.append(EvidenceConflict(
                    id=f"CONFLICT-{len(conflicts) + 1}",
                    sources=[source.id for source in group],
                    summary="Different document revisions contain conflicting numerical evidence; applicability requires review.",
                    values=[{"source_id": source.id, "revision": source.revision, "text": source.text} for source in group],
                ))
        return conflicts
