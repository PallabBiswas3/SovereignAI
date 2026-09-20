from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable

from app.rag.retrieval import RetrievedChunk


_TECHNICAL_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:MPa|kPa|Pa|bar|mm/s|m/s|°C|C|K|rpm|Hz|kW|MW|W|V|A|%|psi)?\b"
    r"|\b[A-Z]{1,8}(?:-[A-Z0-9]+)+\b",
    re.I,
)
_CONSTRAINT_RE = re.compile(
    r"\b(?:limit|threshold|maximum|minimum|must|shall|required|alarm|trip|setpoint|"
    r"above|below|exceed(?:s|ed)?|greater than|less than|at least|at most|within)\b",
    re.I,
)
_NEGATION_RE = re.compile(
    r"\b(?:no|not|never|without|prohibited|must not|shall not|do not)\b",
    re.I,
)
_IDENTIFIER_RE = re.compile(r"\b[A-Z]{1,8}(?:-[A-Z0-9]+)+\b", re.I)
_TOKEN_RE = re.compile(r"[a-z]+(?:-[a-z0-9]+)+|\d+(?:\.\d+)?|[a-z]+", re.I)
_SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
_NUMERIC_RE = re.compile(
    r"\b\d+(?:\.\d+)?\s*(?:MPa|kPa|Pa|bar|mm/s|m/s|°C|C|K|rpm|Hz|kW|MW|W|V|A|%|psi)?",
    re.I,
)


@dataclass(slots=True)
class DistillationResult:
    selected: list[tuple[RetrievedChunk, str]]
    selected_tokens: int
    sentence_count: int
    technical_sentence_count: int
    conflict_preserved_chunks: int


class EvidenceDistiller:
    """Deterministic query-aware evidence compression for local RAG prompts.

    Retrieval and provenance stay untouched. This component only chooses the
    smallest useful sentence subset to expose to the generation model.
    """

    def __init__(
        self,
        *,
        estimate_tokens: Callable[[str], int],
        sentence_redundancy_threshold: float = 0.82,
    ) -> None:
        self.estimate_tokens = estimate_tokens
        self.sentence_redundancy_threshold = sentence_redundancy_threshold

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return {value.lower() for value in _TOKEN_RE.findall(text)}

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return [value.strip() for value in _SENTENCE_SPLIT_RE.split(text) if value.strip()]

    @staticmethod
    def _source_key(item: RetrievedChunk) -> tuple[str, str]:
        return (
            str(item.source.get("file") or "").strip().lower(),
            str(item.source.get("section") or "").strip().lower(),
        )

    @staticmethod
    def _numeric_signature(text: str) -> tuple[str, ...]:
        return tuple(value.lower().strip() for value in _NUMERIC_RE.findall(text))

    def _near_duplicate(self, left: str, right: str) -> bool:
        left_tokens = self._tokens(left)
        right_tokens = self._tokens(right)
        if not left_tokens or not right_tokens:
            return False
        similarity = len(left_tokens & right_tokens) / len(left_tokens | right_tokens)
        return similarity >= self.sentence_redundancy_threshold

    def _sentence_score(self, task: str, sentence: str, position: int) -> tuple[float, int, int]:
        task_terms = self._tokens(task)
        sentence_terms = self._tokens(sentence)
        overlap = len(task_terms & sentence_terms) / max(1, len(task_terms))
        task_ids = {value.lower() for value in _IDENTIFIER_RE.findall(task)}
        sentence_ids = {value.lower() for value in _IDENTIFIER_RE.findall(sentence)}
        identifier_overlap = len(task_ids & sentence_ids)
        technical = int(bool(_TECHNICAL_RE.search(sentence)))
        constraint = int(bool(_CONSTRAINT_RE.search(sentence)))
        negation = int(bool(_NEGATION_RE.search(sentence)))
        score = (
            (4.0 * overlap)
            + (1.5 * identifier_overlap)
            + (0.8 * technical)
            + (0.7 * constraint)
            + (0.4 * negation)
        )
        return score, technical + constraint + negation, -position

    def _conflict_ids(self, chunks: list[RetrievedChunk]) -> set[str]:
        groups: dict[tuple[str, str], list[RetrievedChunk]] = {}
        for item in chunks:
            key = self._source_key(item)
            if key[0]:
                groups.setdefault(key, []).append(item)

        output: set[str] = set()
        for group in groups.values():
            revisions = {
                str(item.source.get("revision"))
                for item in group
                if item.source.get("revision")
            }
            numeric_sets = {
                self._numeric_signature(item.text)
                for item in group
                if self._numeric_signature(item.text)
            }
            if len(revisions) > 1 and len(numeric_sets) > 1:
                output.update(item.chunk_id for item in group)
        return output

    def _candidates(
        self,
        chunks: list[RetrievedChunk],
        chunk_cap: int,
    ) -> tuple[list[RetrievedChunk], set[str]]:
        if chunk_cap <= 0:
            return [], set()
        chosen = list(chunks[:chunk_cap])
        if chunk_cap < 2:
            return chosen, set()

        conflict_ids = self._conflict_ids(chunks)
        chosen_ids = {item.chunk_id for item in chosen}
        protected: set[str] = set()
        for item in list(chosen):
            if item.chunk_id not in conflict_ids:
                continue
            key = self._source_key(item)
            counterpart = next(
                (
                    candidate
                    for candidate in chunks
                    if candidate.chunk_id in conflict_ids
                    and candidate.chunk_id != item.chunk_id
                    and self._source_key(candidate) == key
                ),
                None,
            )
            if counterpart is None:
                continue
            protected.update({item.chunk_id, counterpart.chunk_id})
            if counterpart.chunk_id in chosen_ids:
                continue
            replace_at = next(
                (
                    index
                    for index in range(len(chosen) - 1, -1, -1)
                    if chosen[index].chunk_id not in protected
                ),
                None,
            )
            if replace_at is None:
                continue
            chosen_ids.discard(chosen[replace_at].chunk_id)
            chosen[replace_at] = counterpart
            chosen_ids.add(counterpart.chunk_id)

        order = {item.chunk_id: index for index, item in enumerate(chunks)}
        chosen.sort(key=lambda item: order.get(item.chunk_id, len(chunks)))
        return chosen, protected & {item.chunk_id for item in chosen}

    def distill(
        self,
        *,
        task: str,
        chunks: list[RetrievedChunk],
        token_budget: int,
        chunk_cap: int,
    ) -> DistillationResult:
        candidates, protected_conflicts = self._candidates(chunks, chunk_cap)
        if token_budget <= 0 or not candidates:
            return DistillationResult([], 0, 0, 0, 0)

        ranked: dict[str, list[tuple[int, str, tuple[float, int, int]]]] = {}
        for item in candidates:
            entries = [
                (index, sentence, self._sentence_score(task, sentence, index))
                for index, sentence in enumerate(self._sentences(item.text))
            ]
            entries.sort(key=lambda value: value[2], reverse=True)
            ranked[item.chunk_id] = entries

        selected: dict[str, list[tuple[int, str]]] = {item.chunk_id: [] for item in candidates}
        selected_texts: list[str] = []
        used = 0
        technical_count = 0

        def add(item: RetrievedChunk, index: int, sentence: str) -> bool:
            nonlocal used, technical_count
            tokens = self.estimate_tokens(sentence)
            if tokens > token_budget - used:
                return False
            if any(self._near_duplicate(sentence, existing) for existing in selected_texts):
                return False
            selected[item.chunk_id].append((index, sentence))
            selected_texts.append(sentence)
            used += tokens
            if _TECHNICAL_RE.search(sentence) or _CONSTRAINT_RE.search(sentence):
                technical_count += 1
            return True

        # First preserve source diversity, prioritizing detected revision conflicts.
        coverage_order = sorted(
            enumerate(candidates),
            key=lambda pair: (pair[1].chunk_id not in protected_conflicts, pair[0]),
        )
        consumed: set[tuple[str, int]] = set()
        for _, item in coverage_order:
            for index, sentence, _score in ranked[item.chunk_id]:
                if add(item, index, sentence):
                    consumed.add((item.chunk_id, index))
                    break

        # Spend the remaining budget on the globally highest-value sentences.
        chunk_rank = {item.chunk_id: index for index, item in enumerate(candidates)}
        remainder: list[tuple[tuple[float, int, int], int, RetrievedChunk, int, str]] = []
        for item in candidates:
            for index, sentence, score in ranked[item.chunk_id]:
                if (item.chunk_id, index) not in consumed:
                    remainder.append((score, -chunk_rank[item.chunk_id], item, index, sentence))
        remainder.sort(key=lambda value: (value[0], value[1]), reverse=True)
        for _score, _rank, item, index, sentence in remainder:
            if used >= token_budget:
                break
            add(item, index, sentence)

        output: list[tuple[RetrievedChunk, str]] = []
        for item in candidates:
            sentences = selected[item.chunk_id]
            if sentences:
                output.append((item, " ".join(text for _, text in sorted(sentences))))

        return DistillationResult(
            selected=output,
            selected_tokens=used,
            sentence_count=sum(len(values) for values in selected.values()),
            technical_sentence_count=technical_count,
            conflict_preserved_chunks=sum(
                1 for item, _text in output if item.chunk_id in protected_conflicts
            ),
        )
