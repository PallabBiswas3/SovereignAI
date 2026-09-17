from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from adaptivefact.verification.text import chunk_text


@dataclass
class RetrievedEvidence:
    text: str
    score: float
    index: int


class TfidfRetriever:
    """Cheap lexical retriever used as the Phase 1 retrieval baseline."""

    def __init__(self, *, max_chunk_chars: int = 900, ngram_range: tuple[int, int] = (1, 2)):
        self.max_chunk_chars = max_chunk_chars
        self.ngram_range = ngram_range

    def retrieve(self, query: str, context: str, top_k: int = 3) -> list[RetrievedEvidence]:
        chunks = chunk_text(context, max_chars=self.max_chunk_chars)
        if not chunks or not query.strip():
            return []

        corpus = chunks + [query]
        try:
            matrix = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=self.ngram_range,
            ).fit_transform(corpus)
        except ValueError:
            return []

        query_vec = matrix[-1]
        chunk_matrix = matrix[:-1]
        scores = (chunk_matrix @ query_vec.T).toarray().ravel()
        order = np.argsort(scores)[::-1][: max(1, top_k)]
        return [
            RetrievedEvidence(text=chunks[int(i)], score=float(scores[int(i)]), index=int(i))
            for i in order
        ]
