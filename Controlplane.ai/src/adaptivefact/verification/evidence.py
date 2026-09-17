from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from adaptivefact.verification.retriever import RetrievedEvidence
from adaptivefact.verification.text import chunk_text


@dataclass
class EvidenceRetrieverConfig:
    top_k: int = 3
    max_chunk_chars: int = 700
    ngram_min: int = 1
    ngram_max: int = 2
    # Do not send unrelated top-k chunks to NLI merely because fewer than k
    # relevant chunks exist. Deterministic seed evidence keeps score 1.0.
    min_score: float = 0.05


class ContextTfidfIndex:
    """Fit TF-IDF once for one response context, then reuse it for all claims."""

    def __init__(self, context: str | None, config: EvidenceRetrieverConfig | None = None) -> None:
        self.config = config or EvidenceRetrieverConfig()
        self.chunks = chunk_text(context or "", max_chars=self.config.max_chunk_chars)
        self.vectorizer: TfidfVectorizer | None = None
        self.chunk_matrix = None

        if not self.chunks:
            return

        try:
            self.vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words="english",
                ngram_range=(self.config.ngram_min, self.config.ngram_max),
            )
            self.chunk_matrix = self.vectorizer.fit_transform(self.chunks)
        except ValueError:
            self.vectorizer = None
            self.chunk_matrix = None

    def retrieve(self, query: str, *, top_k: int | None = None) -> list[RetrievedEvidence]:
        if not query.strip() or self.vectorizer is None or self.chunk_matrix is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = (self.chunk_matrix @ query_vec.T).toarray().ravel()
        k = max(1, top_k or self.config.top_k)
        order = np.argsort(scores)[::-1][:k]

        results: list[RetrievedEvidence] = []
        for raw_index in order:
            index = int(raw_index)
            score = float(scores[index])
            if score < self.config.min_score:
                continue
            results.append(
                RetrievedEvidence(
                    text=self.chunks[index],
                    score=score,
                    index=index,
                )
            )
        return results


def merge_seed_evidence(
    seed_evidence: list[str],
    retrieved: list[RetrievedEvidence],
    *,
    top_k: int,
) -> list[RetrievedEvidence]:
    """Keep Phase-5 candidate evidence first, then fill remaining slots by retrieval."""

    merged: list[RetrievedEvidence] = []
    seen: set[str] = set()

    for text in seed_evidence:
        normalized = " ".join(text.split())
        if not normalized or normalized in seen:
            continue
        merged.append(RetrievedEvidence(text=text, score=1.0, index=-1))
        seen.add(normalized)
        if len(merged) >= top_k:
            return merged

    for item in retrieved:
        normalized = " ".join(item.text.split())
        if not normalized or normalized in seen:
            continue
        merged.append(item)
        seen.add(normalized)
        if len(merged) >= top_k:
            break

    return merged
