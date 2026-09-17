from __future__ import annotations

import re

_ALLOWED_LABELS = {"PERSON", "ORG", "GPE", "LOC", "PRODUCT", "EVENT", "WORK_OF_ART", "NORP"}
_PROPER_NOUN_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,4}|[A-Z]{2,}(?:\s+[A-Z]{2,})*)\b")
_FALLBACK_FALSE_ENTITIES = {
    "a",
    "an",
    "how",
    "i",
    "it",
    "its",
    "our",
    "she",
    "that",
    "the",
    "their",
    "there",
    "these",
    "they",
    "this",
    "those",
    "we",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "you",
    "your",
}


class EntityExtractor:
    def __init__(self, *, use_spacy: bool = False, model_name: str = "en_core_web_sm") -> None:
        self.use_spacy = use_spacy
        self.model_name = model_name
        self._nlp = None
        if use_spacy:
            try:
                import spacy

                self._nlp = spacy.load(model_name)
            except Exception as exc:
                raise RuntimeError(
                    f"spaCy entity extraction requested but model '{model_name}' could not be loaded. "
                    f"Run: python -m spacy download {model_name}"
                ) from exc

    def extract(self, text: str) -> list[str]:
        text = text or ""
        if self._nlp is not None:
            entities = [
                ent.text.strip()
                for ent in self._nlp(text).ents
                if ent.label_ in _ALLOWED_LABELS and ent.text.strip()
            ]
        else:
            entities = [match.group(0).strip() for match in _PROPER_NOUN_RE.finditer(text)]

        result: list[str] = []
        seen: set[str] = set()
        for entity in entities:
            key = entity.casefold()
            if len(entity) < 2 or key in seen:
                continue
            # The regex fallback has no part-of-speech information, so a
            # capitalized sentence opener such as "The" or "You" otherwise
            # becomes a fake named entity. spaCy already handles these words.
            if self._nlp is None and key in _FALLBACK_FALSE_ENTITIES:
                continue
            seen.add(key)
            result.append(entity)
        return result
