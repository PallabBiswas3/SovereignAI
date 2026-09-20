from __future__ import annotations

import re
from dataclasses import dataclass

from adaptivefact.data.schema import Claim, ClaimType, ResponseRecord
from adaptivefact.extraction.ner import EntityExtractor
from adaptivefact.extraction.numeric_date import extract_dates, extract_numbers

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+")
_CLAUSE_BOUNDARY_RE = re.compile(r"\s*;\s*|\s+—\s+|\s+–\s+")
_ATOMIC_CONJUNCTION_RE = re.compile(
    r"\s+(?:and|but|while|whereas)\s+(?=(?:[A-Z][\w.-]*|the\s+\w+|it\s+|they\s+|this\s+|that\s+))",
    re.IGNORECASE,
)
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_CITATION_RE = re.compile(r"https?://|\bdoi\s*:|\bet\s+al\.\b|\[[0-9]+\]", re.IGNORECASE)
_PRONOUN_START_RE = re.compile(r"^(it|they|this|that|these|those)\b", re.IGNORECASE)
_SIMPLE_RELATION_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?P<predicate>is|are|was|were|has|have|had|contains?|includes?|reported|recorded|reached|exceeded|requires?|recommends?|approved|joined|founded)\s+(?P<object>.+)$",
    re.IGNORECASE,
)


@dataclass
class ClaimExtractionConfig:
    use_spacy: bool = False
    spacy_model: str = "en_core_web_sm"
    max_claims: int = 40
    min_claim_chars: int = 8
    split_semicolons: bool = True
    split_atomic_conjunctions: bool = True
    decontextualize_pronouns: bool = True


class ClaimExtractor:
    """Low-latency atomic/decontextualized claim extractor.

    The extractor keeps offsets tied to the original response but verifies a
    self-contained form when possible. It deliberately uses deterministic
    heuristics locally; a learned/LLM decomposer can be benchmarked later as an
    ablation without making every factuality check expensive.
    """

    def __init__(self, config: ClaimExtractionConfig | None = None) -> None:
        self.config = config or ClaimExtractionConfig()
        self.entities = EntityExtractor(
            use_spacy=self.config.use_spacy,
            model_name=self.config.spacy_model,
        )

    def extract(self, record: ResponseRecord) -> list[Claim]:
        claims: list[Claim] = []
        previous_subject: str | None = None
        for raw, start, end, parent_span in self._atomic_segments(record.generated_response or ""):
            prefix = _BULLET_PREFIX_RE.match(raw)
            if prefix:
                start += prefix.end()
                raw = raw[prefix.end():]
            clean = " ".join(raw.split()).strip(" -•\t\n")
            if len(clean) < self.config.min_claim_chars:
                continue

            numbers = extract_numbers(clean)
            dates = extract_dates(clean)
            entities = self.entities.extract(clean)
            subject, predicate, obj = self._structure(clean)
            decontextualized = clean
            if self.config.decontextualize_pronouns and previous_subject and _PRONOUN_START_RE.match(clean):
                decontextualized = _PRONOUN_START_RE.sub(previous_subject, clean, count=1)
                subject, predicate, obj = self._structure(decontextualized)
            if subject and not _PRONOUN_START_RE.match(subject):
                previous_subject = subject
            elif entities:
                previous_subject = entities[0]

            qualifiers: dict[str, str] = {}
            if numbers:
                qualifiers["numbers"] = ", ".join(value.raw for value in numbers)
            if dates:
                qualifiers["dates"] = ", ".join(value.raw for value in dates)

            claims.append(
                Claim(
                    id=f"{record.id}:claim:{len(claims)}",
                    text=clean,
                    original_text=clean,
                    decontextualized_text=decontextualized,
                    subject=subject,
                    predicate=predicate,
                    object=obj,
                    qualifiers=qualifiers,
                    extraction_method="heuristic_atomic_v3",
                    parent_span=parent_span,
                    type=self._claim_type(decontextualized, numbers, dates, entities),
                    entities=entities,
                    numbers=[value.raw for value in numbers],
                    dates=[value.raw for value in dates],
                    span=(start, end),
                )
            )
            if len(claims) >= self.config.max_claims:
                break
        return claims

    def _atomic_segments(self, text: str) -> list[tuple[str, int, int, tuple[int, int]]]:
        output: list[tuple[str, int, int, tuple[int, int]]] = []
        for raw, start, end in self._segments_with_offsets(text):
            parent = (start, end)
            if not self.config.split_atomic_conjunctions:
                output.append((raw, start, end, parent))
                continue
            local = 0
            for match in _ATOMIC_CONJUNCTION_RE.finditer(raw):
                if match.start() > local:
                    output.append((raw[local:match.start()], start + local, start + match.start(), parent))
                local = match.end()
            if local < len(raw):
                output.append((raw[local:], start + local, end, parent))
        return output

    def _segments_with_offsets(self, text: str) -> list[tuple[str, int, int]]:
        if not text.strip():
            return []
        sentence_ranges: list[tuple[int, int]] = []
        start = 0
        for boundary in _SENTENCE_BOUNDARY_RE.finditer(text):
            if boundary.start() > start:
                sentence_ranges.append((start, boundary.start()))
            start = boundary.end()
        if start < len(text):
            sentence_ranges.append((start, len(text)))

        segments: list[tuple[str, int, int]] = []
        for sent_start, sent_end in sentence_ranges:
            sentence = text[sent_start:sent_end]
            if not self.config.split_semicolons:
                segments.append((sentence, sent_start, sent_end))
                continue
            local = 0
            for boundary in _CLAUSE_BOUNDARY_RE.finditer(sentence):
                if boundary.start() > local:
                    segments.append((sentence[local:boundary.start()], sent_start + local, sent_start + boundary.start()))
                local = boundary.end()
            if local < len(sentence):
                segments.append((sentence[local:], sent_start + local, sent_end))
        return segments

    @staticmethod
    def _structure(text: str) -> tuple[str | None, str | None, str | None]:
        match = _SIMPLE_RELATION_RE.match(text.strip().rstrip("."))
        if not match:
            return None, None, None
        return (
            match.group("subject").strip(),
            match.group("predicate").strip().casefold(),
            match.group("object").strip(),
        )

    @staticmethod
    def _claim_type(text, numbers, dates, entities) -> ClaimType:
        if _CITATION_RE.search(text):
            return ClaimType.CITATION
        if dates:
            return ClaimType.DATE
        if numbers:
            return ClaimType.NUMERIC
        if entities:
            return ClaimType.ENTITY
        if len(text.split()) >= 4:
            return ClaimType.FACTUAL
        return ClaimType.OTHER
