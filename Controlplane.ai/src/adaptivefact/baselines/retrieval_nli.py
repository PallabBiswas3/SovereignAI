from __future__ import annotations

from adaptivefact.benchmark.runner import Component, ComponentResult
from adaptivefact.data.schema import ResponseRecord
from adaptivefact.verification.nli import NLIScorer
from adaptivefact.verification.retriever import TfidfRetriever
from adaptivefact.verification.text import split_sentences


class RetrievalNLIBaseline(Component):
    """Always-on grounded verification using sentence units, retrieval, then NLI.

    Phase 1 deliberately uses sentences instead of a full atomic-claim extractor;
    atomic claim extraction is evaluated later in Phase 5/7.
    """

    label = "Always-on retrieval + NLI"

    def __init__(
        self,
        nli: NLIScorer,
        *,
        retriever: TfidfRetriever | None = None,
        top_k: int = 2,
        contradiction_threshold: float = 0.70,
        unsupported_threshold: float = 0.70,
        max_sentences: int = 20,
    ) -> None:
        self.nli = nli
        self.retriever = retriever or TfidfRetriever()
        self.top_k = top_k
        self.contradiction_threshold = contradiction_threshold
        self.unsupported_threshold = unsupported_threshold
        self.max_sentences = max_sentences

    def run(self, record: ResponseRecord) -> ComponentResult:
        sentences = split_sentences(record.generated_response)[: self.max_sentences]
        if not sentences or not record.context:
            return ComponentResult(prediction=1, confidence=1.0, retrieval_calls=0)

        premises: list[str] = []
        hypotheses: list[str] = []
        sentence_slices: list[tuple[int, int]] = []
        retrieval_calls = 0

        for sentence in sentences:
            evidence = self.retriever.retrieve(sentence, record.context, top_k=self.top_k)
            retrieval_calls += 1
            start = len(premises)
            for item in evidence:
                premises.append(item.text)
                hypotheses.append(sentence)
            sentence_slices.append((start, len(premises)))

        if not premises:
            return ComponentResult(prediction=1, confidence=1.0, retrieval_calls=retrieval_calls)

        scores = self.nli.score(premises, hypotheses)
        hallucination_scores: list[float] = []
        sentence_flags: list[bool] = []

        for start, end in sentence_slices:
            local = scores[start:end]
            if not local:
                hallucination_scores.append(1.0)
                sentence_flags.append(True)
                continue
            best_entailment = max(s.entailment for s in local)
            best_contradiction = max(s.contradiction for s in local)
            unsupported = 1.0 - best_entailment
            sentence_risk = max(best_contradiction, unsupported)
            hallucination_scores.append(sentence_risk)
            sentence_flags.append(
                best_contradiction >= self.contradiction_threshold
                or unsupported >= self.unsupported_threshold
            )

        overall_risk = max(hallucination_scores, default=0.0)
        is_hallucinated = any(sentence_flags)

        return ComponentResult(
            prediction=int(is_hallucinated),
            confidence=float(overall_risk),
            retrieval_calls=retrieval_calls,
        )
