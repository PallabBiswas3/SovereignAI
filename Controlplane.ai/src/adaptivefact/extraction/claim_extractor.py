from __future__ import annotations

import re
from dataclasses import dataclass

from adaptivefact.data.schema import Claim, ClaimType, ResponseRecord
from adaptivefact.extraction.ner import EntityExtractor
from adaptivefact.extraction.numeric_date import extract_dates, extract_numbers

_SENTENCE_BOUNDARY_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])|\n+")
_CLAUSE_BOUNDARY_RE = re.compile(r"\s*;\s*|\s+—\s+|\s+–\s+")
_ATOMIC_CONJUNCTION_RE = re.compile(
    r"\s+(?i:and|but|while|whereas)\s+",
)
_FINITE_PREDICATE_RE = re.compile(
    r"\b(?:is|are|was|were|has|have|had|does|do|did|can|could|must|should|"
    r"may|might|will|would|recorded|reported|reached|exceeded|requires?|"
    r"recommends?|approved|requested|occurred|shows?|indicates?|lacks?|supports?|exceeds?|constitutes?)\b",
    re.IGNORECASE,
)
_BULLET_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_DISCOURSE_PREFIX_RE = re.compile(r"^\s*(?:however|nevertheless),\s+", re.IGNORECASE)
_CITATION_RE = re.compile(r"https?://|\bdoi\s*:|\bet\s+al\.\b|\[[0-9]+\]", re.IGNORECASE)
_PRONOUN_START_RE = re.compile(r"^(it|they|this|that|these|those)\b", re.IGNORECASE)
_SIMPLE_RELATION_RE = re.compile(
    r"^(?P<subject>.+?)\s+(?P<predicate>is|are|was|were|has|have|had|contains?|includes?|reported|recorded|reached|exceeded|requires?|recommends?|approved|joined|founded)\s+(?P<object>.+)$",
    re.IGNORECASE,
)
# A conjunction inside a condition is part of that condition ("must not be
# restarted until the seal is replaced and the permit is closed"); splitting it
# would turn a precondition into a stand-alone fact and drop it from the rule.
_CONDITION_RE = re.compile(
    r"\b(?:until|unless|if|provided(?:\s+that)?|as\s+long\s+as|so\s+long\s+as|in\s+case|whenever|except\s+when|"
    r"only\s+(?:after|when|if|once))\b|"
    r"\b(?:when|once|before|after|while)\s+(?:the\s+|a\s+|an\s+|it\s+|they\s+|its\s+)?[\w-]+(?:\s+(?!and\b|but\b)[\w-]+){0,3}\s+"
    r"(?:is|are|was|were|has|have|had|reaches|exceeds|falls|drops|rises|returns|completes|closes|opens)\b",
    re.IGNORECASE,
)
_ASSET_ID_RE = re.compile(r"(?<![\w-])[A-Za-z]+-\d+\b")
# A capitalised first word of an instruction is not a subject ("Do not ...").
_NOT_A_SUBJECT = frozenset({
    "do", "don't", "check", "ensure", "replace", "inspect", "verify", "please", "note", "initiate", "stop",
    "start", "restart", "use", "isolate", "monitor", "review", "confirm", "record", "schedule", "perform",
    "apply", "keep", "avoid", "never", "always", "then", "also", "however", "the", "this", "that", "these",
    "those", "if", "when", "after", "before", "until", "it", "they", "we", "you", "a", "an",
})
_STRUCTURAL_HEADINGS = frozenset({
    "diagnosis:", "evidence:", "uncertainty/conflicts:",
    "recommended inspection:", "human approval required:", "conclusion:",
})


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
        # The claim cap bounds computation. Record when it leaves claim-bearing
        # text unextracted so callers can abstain instead of treating the
        # unexamined remainder as verified.
        record.metadata.pop("claim_extraction_truncated", None)
        record.metadata.pop("claim_extraction_unextracted_start", None)
        for raw, start, end, parent_span, inherited_asset in self._atomic_segments(record.generated_response or ""):
            prefix = _BULLET_PREFIX_RE.match(raw)
            if prefix:
                start += prefix.end()
                raw = raw[prefix.end():]
            discourse = _DISCOURSE_PREFIX_RE.match(raw)
            if discourse:
                start += discourse.end()
                raw = raw[discourse.end():]
            clean = " ".join(raw.split()).strip(" -•\t\n")
            # Section labels are presentation scaffolding, not factual claims.
            # A narrow allowlist avoids hiding arbitrary instruction-like text.
            heading = re.sub(r"\s*/\s*", "/", clean.casefold())
            if heading in _STRUCTURAL_HEADINGS:
                continue
            if len(clean) < self.config.min_claim_chars:
                continue
            if len(claims) >= self.config.max_claims:
                record.metadata["claim_extraction_truncated"] = True
                record.metadata["claim_extraction_unextracted_start"] = start
                break

            numbers = extract_numbers(clean)
            dates = extract_dates(clean)
            entities = self.entities.extract(clean)
            subject, predicate, obj = self._structure(clean)
            decontextualized = clean
            if inherited_asset and not _ASSET_ID_RE.search(clean):
                # Right conjunct of "Pump-102 vibration is 7.4 mm/s and the
                # bearing temperature is 92 °C": keep the asset it is about.
                decontextualized = self._with_asset(clean, inherited_asset)
                subject, predicate, obj = self._structure(decontextualized)
            elif self.config.decontextualize_pronouns and previous_subject and _PRONOUN_START_RE.match(clean):
                decontextualized = _PRONOUN_START_RE.sub(previous_subject, clean, count=1)
                subject, predicate, obj = self._structure(decontextualized)
            previous_subject = self._salient_subject(decontextualized, subject, entities) or previous_subject

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
        return claims

    def _atomic_segments(self, text: str) -> list[tuple[str, int, int, tuple[int, int], str | None]]:
        output: list[tuple[str, int, int, tuple[int, int], str | None]] = []
        for raw, start, end in self._segments_with_offsets(text):
            parent = (start, end)
            # Keep a conditional sentence whole so its condition stays attached
            # to every requirement it governs.
            if not self.config.split_atomic_conjunctions or _CONDITION_RE.search(raw):
                output.append((raw, start, end, parent, None))
                continue
            local = 0
            inherited: str | None = None
            for match in _ATOMIC_CONJUNCTION_RE.finditer(raw):
                # A coordinated noun phrase is not two factual assertions.
                # Split only when both sides have their own finite predicate and
                # the right side starts with an explicit subject, not "but is".
                right_first_word = re.match(r"[A-Za-z][\w.-]*", raw[match.end():])
                if right_first_word is None or _FINITE_PREDICATE_RE.fullmatch(right_first_word.group()):
                    continue
                if not (_FINITE_PREDICATE_RE.search(raw[local:match.start()])
                        and _FINITE_PREDICATE_RE.search(raw[match.end():])):
                    continue
                if match.start() > local:
                    output.append((raw[local:match.start()], start + local, start + match.start(), parent, inherited))
                # The right conjunct inherits the asset only when the text so far
                # names exactly one, never one picked from a mixed-asset sentence.
                assets = {asset.casefold(): asset for asset in _ASSET_ID_RE.findall(raw[:match.start()])}
                inherited = next(iter(assets.values())) if len(assets) == 1 else None
                local = match.end()
            if local < len(raw):
                output.append((raw[local:], start + local, end, parent, inherited))
        return output

    @staticmethod
    def _with_asset(text: str, asset: str) -> str:
        determiner = re.match(r"(?i)(?:the|its|their)\s+", text)
        if determiner:
            return f"{asset} {text[determiner.end():]}"
        return f"{asset}: {text}"

    @staticmethod
    def _salient_subject(text: str, subject: str | None, entities: list[str]) -> str | None:
        """Subject a following pronoun may refer to, or None if unclear."""
        if subject and not _PRONOUN_START_RE.match(subject) and subject.casefold() not in _NOT_A_SUBJECT:
            return subject
        asset = _ASSET_ID_RE.search(text)
        if asset:
            return asset.group()
        candidates = [entity for entity in entities if entity.casefold() not in _NOT_A_SUBJECT]
        return candidates[0] if candidates else None

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
