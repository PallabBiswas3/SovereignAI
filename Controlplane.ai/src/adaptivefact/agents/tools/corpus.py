from __future__ import annotations

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer

from adaptivefact.agents.schema import SearchResult
from adaptivefact.agents.tools.base import SearchTool
from adaptivefact.data.schema import Claim, ResponseRecord


class CorpusDocument(BaseModel):
    id: str
    name: str
    text: str
    url: str | None = None
    authority_score: float = Field(default=0.8, ge=0.0, le=1.0)
    metadata: dict = Field(default_factory=dict)


class CorpusSearchTool(SearchTool):
    """Offline enterprise-search stand-in backed by an approved document corpus."""

    name = "approved_corpus"

    def __init__(self, documents: list[CorpusDocument | dict]) -> None:
        self.documents = [
            item if isinstance(item, CorpusDocument) else CorpusDocument.model_validate(item)
            for item in documents
        ]
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix = None
        if self.documents:
            self.vectorizer = TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2))
            try:
                self.matrix = self.vectorizer.fit_transform([item.text for item in self.documents])
            except ValueError:
                self.vectorizer = None

    def search(
        self,
        query: str,
        *,
        record: ResponseRecord,
        claim: Claim,
        top_k: int = 3,
    ) -> list[SearchResult]:
        if not query.strip() or self.vectorizer is None or self.matrix is None:
            return []
        query_vector = self.vectorizer.transform([query])
        scores = (self.matrix @ query_vector.T).toarray().ravel()
        order = np.argsort(scores)[::-1][: max(1, top_k)]
        results = []
        for raw_index in order:
            index = int(raw_index)
            score = float(scores[index])
            if score <= 0.0:
                continue
            document = self.documents[index]
            results.append(
                SearchResult(
                    source_id=document.id,
                    source_name=document.name,
                    text=document.text,
                    url=document.url,
                    authority_score=document.authority_score,
                    retrieval_score=score,
                    metadata={**document.metadata, "tool": self.name},
                )
            )
        return results
