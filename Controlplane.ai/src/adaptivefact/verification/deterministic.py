from __future__ import annotations

import re
from dataclasses import dataclass
from time import perf_counter

from adaptivefact.data.schema import Claim, VerificationStatus
from adaptivefact.extraction.ner import EntityExtractor
from adaptivefact.extraction.numeric_date import DateValue, NumberValue, extract_dates, extract_numbers
from adaptivefact.verification.text import split_sentences

_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
_STOPWORDS = {
    "a", "an", "the", "is", "was", "were", "are", "be", "been", "being",
    "of", "to", "in", "on", "at", "for", "from", "by", "with", "and", "or",
    "that", "this", "it", "its", "as", "has", "have", "had", "will", "would",
    "percent", "thousand", "million", "billion", "crore", "lakh", "bn", "mn",
}


@dataclass
class DeterministicVerifierConfig:
    use_spacy: bool = False
    spacy_model: str = "en_core_web_sm"
    min_anchor_overlap_for_conflict: float = 0.45
    min_anchor_tokens_for_conflict: int = 1
    support_confidence: float = 0.98
    conflict_confidence: float = 0.90
    entity_support_confidence: float = 0.92


class DeterministicVerifier:
    """Conservative Phase-5 verifier.

    Rules:
    - exact normalized numeric/date matches can support a structured claim;
    - a numeric/date contradiction is emitted only when a context sentence is
      strongly anchored to the same non-value words but contains a different
      comparable value;
    - entity presence is only an evidence signal; relational entity claims remain
      UNKNOWN until semantic verification;
    - anything not safely resolvable is UNKNOWN for Phase 6.
    """

    def __init__(self, config: DeterministicVerifierConfig | None = None) -> None:
        self.config = config or DeterministicVerifierConfig()
        self.entities = EntityExtractor(
            use_spacy=self.config.use_spacy,
            model_name=self.config.spacy_model,
        )

    def verify(self, claim: Claim, context: str | None) -> Claim:
        started = perf_counter()
        context = context or ""
        status = VerificationStatus.UNKNOWN
        confidence = 0.0
        evidence: list[str] = []
        method = "deterministic_unresolved"

        if not context.strip():
            return self._finish(claim, status, confidence, evidence, method, started)

        claim_numbers = extract_numbers(claim.text)
        claim_dates = extract_dates(claim.text)
        claim_entities = claim.entities or self.entities.extract(claim.text)

        context_numbers = extract_numbers(context)
        context_dates = extract_dates(context)
        context_casefold = context.casefold()

        if claim_dates:
            status, confidence, evidence, method = self._verify_dates(
                claim.text, claim_dates, context, context_dates
            )
        elif claim_numbers:
            status, confidence, evidence, method = self._verify_numbers(
                claim.text, claim_numbers, context, context_numbers
            )
        elif claim_entities:
            supported_entities = [e for e in claim_entities if e.casefold() in context_casefold]
            if supported_entities:
                status = VerificationStatus.UNKNOWN
                confidence = 0.0
                method = "deterministic_entity_presence_unresolved"
                evidence = self._best_evidence_sentences(claim.text, context, limit=1)

        return self._finish(claim, status, confidence, evidence, method, started)

    def _verify_numbers(
        self,
        claim_text: str,
        claim_values: list[NumberValue],
        context: str,
        context_values: list[NumberValue],
    ):
        claim_keys = {(v.normalized, v.kind, v.unit) for v in claim_values}
        exact_evidence = self._find_exact_value_sentence(claim_text, claim_values, context, value_type="number")
        if claim_keys and exact_evidence is not None:
            return (
                VerificationStatus.SUPPORTED,
                self.config.support_confidence,
                [exact_evidence],
                "deterministic_numeric_exact",
            )

        conflict = self._find_conflicting_value_sentence(claim_text, claim_values, context, value_type="number")
        if conflict is not None:
            return (
                VerificationStatus.UNKNOWN,
                0.0,
                [conflict],
                "deterministic_numeric_conflict_candidate",
            )

        return VerificationStatus.UNKNOWN, 0.0, [], "deterministic_numeric_unresolved"

    def _verify_dates(
        self,
        claim_text: str,
        claim_values: list[DateValue],
        context: str,
        context_values: list[DateValue],
    ):
        claim_keys = {v.normalized for v in claim_values}
        exact_evidence = self._find_exact_value_sentence(claim_text, claim_values, context, value_type="date")
        if claim_keys and exact_evidence is not None:
            return (
                VerificationStatus.SUPPORTED,
                self.config.support_confidence,
                [exact_evidence],
                "deterministic_date_exact",
            )

        conflict = self._find_conflicting_value_sentence(claim_text, claim_values, context, value_type="date")
        if conflict is not None:
            return (
                VerificationStatus.UNKNOWN,
                0.0,
                [conflict],
                "deterministic_date_conflict_candidate",
            )

        return VerificationStatus.UNKNOWN, 0.0, [], "deterministic_date_unresolved"


    def _find_exact_value_sentence(self, claim_text, claim_values, context, *, value_type: str):
        claim_anchor = self._anchor_tokens(claim_text)
        claim_value_keys = self._value_keys(claim_values, value_type)
        best = None
        best_overlap = 0.0
        for sentence in split_sentences(context):
            sentence_values = extract_numbers(sentence) if value_type == "number" else extract_dates(sentence)
            if not sentence_values:
                continue
            sentence_keys = self._value_keys(sentence_values, value_type)
            if not claim_value_keys.issubset(sentence_keys):
                continue
            sentence_anchor = self._anchor_tokens(sentence)
            overlap = len(claim_anchor & sentence_anchor) / max(1, len(claim_anchor))
            if overlap >= self.config.min_anchor_overlap_for_conflict and overlap > best_overlap:
                best = sentence
                best_overlap = overlap
        return best

    def _find_conflicting_value_sentence(self, claim_text, claim_values, context, *, value_type: str):
        claim_anchor = self._anchor_tokens(claim_text)
        if len(claim_anchor) < self.config.min_anchor_tokens_for_conflict:
            return None

        claim_value_keys = self._value_keys(claim_values, value_type)
        best = None
        best_overlap = 0.0
        for sentence in split_sentences(context):
            sentence_anchor = self._anchor_tokens(sentence)
            if not sentence_anchor:
                continue
            overlap = len(claim_anchor & sentence_anchor) / max(1, len(claim_anchor))
            if overlap < self.config.min_anchor_overlap_for_conflict:
                continue

            sentence_values = extract_numbers(sentence) if value_type == "number" else extract_dates(sentence)
            if not sentence_values:
                continue
            sentence_keys = self._value_keys(sentence_values, value_type)
            if claim_value_keys.isdisjoint(sentence_keys) and overlap > best_overlap:
                best = sentence
                best_overlap = overlap
        return best

    @staticmethod
    def _value_keys(values, value_type: str):
        if value_type == "number":
            return {(v.normalized, v.kind, v.unit) for v in values}
        return {v.normalized for v in values}

    @classmethod
    def _anchor_tokens(cls, text: str) -> set[str]:
        # Remove numeric/date surface forms from lexical anchors so value changes do
        # not artificially reduce evidence similarity.
        stripped = re.sub(r"\d+(?:[.,:/-]\d+)*", " ", text.lower())
        return {
            token
            for token in _WORD_RE.findall(stripped)
            if token not in _STOPWORDS and len(token) > 2
        }

    @classmethod
    def _best_evidence_sentences(cls, claim_text: str, context: str, *, limit: int = 1) -> list[str]:
        anchors = cls._anchor_tokens(claim_text)
        scored = []
        for sentence in split_sentences(context):
            sent = cls._anchor_tokens(sentence)
            score = len(anchors & sent) / max(1, len(anchors))
            if score > 0:
                scored.append((score, sentence))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [sentence for _, sentence in scored[:limit]]

    @staticmethod
    def _finish(claim, status, confidence, evidence, method, started):
        claim.status = status
        claim.confidence = confidence if confidence > 0 else None
        claim.evidence = evidence
        claim.verifier_used = method
        claim.latency_ms = (perf_counter() - started) * 1000.0
        return claim
