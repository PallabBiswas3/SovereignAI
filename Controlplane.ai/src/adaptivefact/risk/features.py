from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from adaptivefact.data.schema import ResponseRecord
from adaptivefact.verification.text import chunk_text, split_sentences

_WORD_RE = re.compile(r"\b[\w'-]+\b", re.UNICODE)
_NUMBER_RE = re.compile(r"(?<!\w)[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?%?")
_PERCENT_RE = re.compile(r"\b\d+(?:\.\d+)?\s*%")
_CURRENCY_RE = re.compile(r"(?:[$€£₹]\s*\d[\d,]*(?:\.\d+)?)|(?:\b(?:USD|EUR|GBP|INR)\s*\d[\d,]*(?:\.\d+)?)", re.I)
_YEAR_RE = re.compile(r"\b(?:18|19|20|21)\d{2}\b")
_DATE_RE = re.compile(
    r"\b(?:\d{1,2}[/-]\d{1,2}[/-](?:\d{2}|\d{4})|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
    r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2}(?:,\s*\d{4})?)\b",
    re.I,
)
_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.I)
_PROPER_NOUN_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}|[A-Z]{2,})\b")
_CITATION_RE = re.compile(r"\[[0-9]{1,3}\]|\([A-Z][A-Za-z-]+(?:\s+et\s+al\.)?,?\s*\d{4}\)")

_HEDGES = {"maybe", "perhaps", "possibly", "likely", "probably", "appears", "seems", "might", "could"}
_CERTAINTY = {"definitely", "certainly", "clearly", "always", "never", "undoubtedly", "proven"}
_CURRENT_TERMS = {"current", "currently", "latest", "today", "now", "recent", "recently"}
_CITATION_TERMS = {"source", "citation", "cite", "reference", "paper", "doi"}


@dataclass
class FeatureExtractionConfig:
    use_spacy: bool = False
    spacy_model: str = "en_core_web_sm"
    use_semantic_support: bool = False
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_batch_size: int = 32
    max_context_chars: int = 16000
    max_context_chunks: int = 24
    max_response_sentences: int = 20

    def to_dict(self) -> dict:
        return asdict(self)


class RiskFeatureExtractor:
    """Provider-independent Phase 2 feature extractor.

    It has no dependence on generator logits. spaCy NER and semantic embeddings
    are optional; deterministic regex/lexical fallbacks keep V1 runnable on CPU.
    """

    def __init__(self, config: FeatureExtractionConfig | None = None) -> None:
        self.config = config or FeatureExtractionConfig()
        self._nlp = None
        self._semantic_model = None
        self._context_embedding_cache: dict[str, tuple[list[str], np.ndarray]] = {}

        if self.config.use_spacy:
            try:
                import spacy
                self._nlp = spacy.load(self.config.spacy_model)
            except (ImportError, OSError):
                self._nlp = None

        if self.config.use_semantic_support:
            try:
                from sentence_transformers import SentenceTransformer
                self._semantic_model = SentenceTransformer(self.config.semantic_model)
            except ImportError as exc:
                raise ImportError(
                    "Semantic support is enabled but sentence-transformers is not installed."
                ) from exc

    @property
    def feature_names(self) -> list[str]:
        names = [
            "response_chars",
            "response_words",
            "response_sentences",
            "avg_sentence_words",
            "response_question_ratio",
            "response_number_count",
            "response_percent_count",
            "response_currency_count",
            "response_year_count",
            "response_date_count",
            "response_url_count",
            "response_citation_count",
            "response_entity_count",
            "response_hedge_count",
            "response_certainty_count",
            "query_chars",
            "query_words",
            "query_entity_count",
            "query_number_count",
            "query_year_count",
            "query_current_flag",
            "query_citation_flag",
            "lexical_context_overlap",
            "min_sentence_lexical_support",
            "mean_sentence_lexical_support",
            "max_sentence_lexical_support",
            "entity_support_ratio",
            "number_support_ratio",
            "date_support_ratio",
        ]
        if self.config.use_semantic_support:
            names += [
                "min_sentence_semantic_support",
                "mean_sentence_semantic_support",
                "max_sentence_semantic_support",
            ]
        return names

    def transform(self, records: Iterable[ResponseRecord]) -> np.ndarray:
        rows = [self.extract(record) for record in records]
        if not rows:
            return np.empty((0, len(self.feature_names)), dtype=np.float32)
        return np.asarray([[row[name] for name in self.feature_names] for row in rows], dtype=np.float32)

    def extract(self, record: ResponseRecord) -> dict[str, float]:
        response = record.generated_response or ""
        query = record.query or ""
        context = (record.context or "")[: self.config.max_context_chars]

        response_words = self._words(response)
        query_words = self._words(query)
        response_sentences = split_sentences(response)[: self.config.max_response_sentences]

        response_entities = self._entities(response)
        query_entities = self._entities(query)
        response_numbers = self._normalized_numbers(response)
        context_numbers = set(self._normalized_numbers(context))
        response_dates = self._normalized_dates(response)
        context_dates = set(self._normalized_dates(context))

        context_chunks = chunk_text(context, max_chars=800)[: self.config.max_context_chunks]
        lexical_supports = [
            max((self._containment(sentence, chunk) for chunk in context_chunks), default=0.0)
            for sentence in response_sentences
        ]
        lexical_supports = lexical_supports or [0.0]

        features = {
            "response_chars": float(len(response)),
            "response_words": float(len(response_words)),
            "response_sentences": float(len(response_sentences)),
            "avg_sentence_words": float(len(response_words) / max(1, len(response_sentences))),
            "response_question_ratio": float(response.count("?") / max(1, len(response_sentences))),
            "response_number_count": float(len(response_numbers)),
            "response_percent_count": float(len(_PERCENT_RE.findall(response))),
            "response_currency_count": float(len(_CURRENCY_RE.findall(response))),
            "response_year_count": float(len(_YEAR_RE.findall(response))),
            "response_date_count": float(len(response_dates)),
            "response_url_count": float(len(_URL_RE.findall(response))),
            "response_citation_count": float(len(_CITATION_RE.findall(response))),
            "response_entity_count": float(len(response_entities)),
            "response_hedge_count": float(self._count_terms(response_words, _HEDGES)),
            "response_certainty_count": float(self._count_terms(response_words, _CERTAINTY)),
            "query_chars": float(len(query)),
            "query_words": float(len(query_words)),
            "query_entity_count": float(len(query_entities)),
            "query_number_count": float(len(self._normalized_numbers(query))),
            "query_year_count": float(len(_YEAR_RE.findall(query))),
            "query_current_flag": float(self._contains_terms(query_words, _CURRENT_TERMS)),
            "query_citation_flag": float(self._contains_terms(query_words, _CITATION_TERMS)),
            "lexical_context_overlap": self._containment(response, context),
            "min_sentence_lexical_support": float(min(lexical_supports)),
            "mean_sentence_lexical_support": float(np.mean(lexical_supports)),
            "max_sentence_lexical_support": float(max(lexical_supports)),
            "entity_support_ratio": self._support_ratio(response_entities, context, casefold=True),
            "number_support_ratio": self._set_support_ratio(response_numbers, context_numbers),
            "date_support_ratio": self._set_support_ratio(response_dates, context_dates),
        }

        if self.config.use_semantic_support:
            semantic = self._semantic_support(record, response_sentences, context)
            features.update(semantic)

        return {name: self._finite(features.get(name, 0.0)) for name in self.feature_names}

    def _semantic_support(
        self,
        record: ResponseRecord,
        response_sentences: list[str],
        context: str,
    ) -> dict[str, float]:
        if self._semantic_model is None or not response_sentences or not context:
            return {
                "min_sentence_semantic_support": 0.0,
                "mean_sentence_semantic_support": 0.0,
                "max_sentence_semantic_support": 0.0,
            }

        cache_key = record.source_id or f"context:{hash(context)}"
        cached = self._context_embedding_cache.get(cache_key)
        if cached is None:
            chunks = chunk_text(context, max_chars=800)[: self.config.max_context_chunks]
            if not chunks:
                return {
                    "min_sentence_semantic_support": 0.0,
                    "mean_sentence_semantic_support": 0.0,
                    "max_sentence_semantic_support": 0.0,
                }
            chunk_embeddings = self._semantic_model.encode(
                chunks,
                batch_size=self.config.semantic_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            chunk_embeddings = np.asarray(chunk_embeddings, dtype=np.float32)
            self._context_embedding_cache[cache_key] = (chunks, chunk_embeddings)
        else:
            _, chunk_embeddings = cached

        sentence_embeddings = self._semantic_model.encode(
            response_sentences,
            batch_size=self.config.semantic_batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        sentence_embeddings = np.asarray(sentence_embeddings, dtype=np.float32)
        similarities = sentence_embeddings @ chunk_embeddings.T
        per_sentence = similarities.max(axis=1)
        return {
            "min_sentence_semantic_support": float(per_sentence.min()),
            "mean_sentence_semantic_support": float(per_sentence.mean()),
            "max_sentence_semantic_support": float(per_sentence.max()),
        }

    def _entities(self, text: str) -> list[str]:
        if self._nlp is not None:
            doc = self._nlp(text)
            allowed = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP"}
            return [ent.text.strip() for ent in doc.ents if ent.label_ in allowed and ent.text.strip()]
        return [m.group(0).strip() for m in _PROPER_NOUN_RE.finditer(text)]

    @staticmethod
    def _words(text: str) -> list[str]:
        return [m.group(0).lower() for m in _WORD_RE.finditer(text)]

    @staticmethod
    def _normalized_numbers(text: str) -> list[str]:
        values = []
        for raw in _NUMBER_RE.findall(text):
            value = raw.replace(",", "").replace(" ", "").lower()
            values.append(value)
        return values

    @staticmethod
    def _normalized_dates(text: str) -> list[str]:
        values = [m.group(0).lower().replace(",", "").strip() for m in _DATE_RE.finditer(text)]
        values.extend(m.group(0) for m in _YEAR_RE.finditer(text))
        return list(dict.fromkeys(values))

    @classmethod
    def _jaccard(cls, a: str, b: str) -> float:
        a_tokens = set(cls._words(a))
        b_tokens = set(cls._words(b))
        if not a_tokens:
            return 0.0
        union = a_tokens | b_tokens
        return float(len(a_tokens & b_tokens) / max(1, len(union)))

    @classmethod
    def _containment(cls, needle_text: str, evidence_text: str) -> float:
        needle = set(cls._words(needle_text))
        evidence = set(cls._words(evidence_text))
        if not needle:
            return 0.0
        return float(len(needle & evidence) / len(needle))

    @staticmethod
    def _support_ratio(items: list[str], context: str, *, casefold: bool = False) -> float:
        if not items:
            return 1.0
        haystack = context.casefold() if casefold else context
        supported = 0
        for item in items:
            needle = item.casefold() if casefold else item
            if needle and needle in haystack:
                supported += 1
        return float(supported / len(items))

    @staticmethod
    def _set_support_ratio(items: list[str], supported_set: set[str]) -> float:
        if not items:
            return 1.0
        return float(sum(item in supported_set for item in items) / len(items))

    @staticmethod
    def _count_terms(words: list[str], terms: set[str]) -> int:
        return sum(word in terms for word in words)

    @staticmethod
    def _contains_terms(words: list[str], terms: set[str]) -> bool:
        return any(word in terms for word in words)

    @staticmethod
    def _finite(value: float) -> float:
        try:
            value = float(value)
        except (TypeError, ValueError):
            return 0.0
        return value if math.isfinite(value) else 0.0
