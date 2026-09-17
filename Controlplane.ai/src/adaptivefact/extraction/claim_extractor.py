from __future__ import annotations

import re
from dataclasses import dataclass

from adaptivefact.data.schema import Claim, ClaimType, ResponseRecord
from adaptivefact.extraction.ner import EntityExtractor
from adaptivefact.extraction.numeric_date import extract_dates, extract_numbers

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+")
_CLAUSE_BOUNDARY_RE = re.compile(r"\s*;\s*|\s+—\s+|\s+–\s+")
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_CITATION_RE = re.compile(r"https?://|\bdoi\s*:|\bet\s+al\.\b|\[[0-9]+\]", re.IGNORECASE)


@dataclass
class ClaimExtractionConfig:
    use_spacy: bool = False
    spacy_model: str = "en_core_web_sm"
    max_claims: int = 30
    min_claim_chars: int = 8
    split_semicolons: bool = True


class ClaimExtractor:
    """Low-latency Phase-5 claim extractor.

    This intentionally produces sentence/semicolon-sized factual units, not a
    claim of perfect atomicity. Phase 7 can add a stronger atomic decomposer.
    Offsets remain tied to the original response for RAGTruth span diagnostics.
    """

    def __init__(self, config: ClaimExtractionConfig | None = None) -> None:
        self.config = config or ClaimExtractionConfig()
        self.entities = EntityExtractor(
            use_spacy=self.config.use_spacy,
            model_name=self.config.spacy_model,
        )

    def extract(self, record: ResponseRecord) -> list[Claim]:
        claims: list[Claim] = []
        for raw, start, end in self._segments_with_offsets(record.generated_response or ""):
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
            claims.append(
                Claim(
                    id=f"{record.id}:claim:{len(claims)}",
                    text=clean,
                    type=self._claim_type(clean, numbers, dates, entities),
                    entities=entities,
                    numbers=[value.raw for value in numbers],
                    dates=[value.raw for value in dates],
                    span=(start, end),
                )
            )
            if len(claims) >= self.config.max_claims:
                break
        return claims

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
