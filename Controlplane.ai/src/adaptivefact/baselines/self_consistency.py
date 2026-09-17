from __future__ import annotations

import re

from adaptivefact.benchmark.runner import Component, ComponentResult
from adaptivefact.data.schema import ResponseRecord
from adaptivefact.generation.base import ChatGenerator
from adaptivefact.verification.nli import NLIScorer
from adaptivefact.verification.text import split_sentences


class SelfConsistencyBaseline(Component):
    """Black-box consistency baseline using same-generator stochastic resampling.

    The original RAGTruth answer is checked against K fresh samples from the same
    configured generator. This avoids treating outputs from unrelated models as
    if they were genuine self-consistency samples.
    """

    label = "Self-consistency"

    def __init__(
        self,
        generator: ChatGenerator,
        nli: NLIScorer,
        *,
        n_samples: int = 3,
        temperature: float = 0.8,
        max_new_tokens: int = 256,
        max_context_chars: int = 9000,
        hallucination_threshold: float = 0.55,
        max_sentences: int = 15,
        record_model_aliases: list[str] | None = None,
        require_model_match: bool = True,
    ) -> None:
        self.generator = generator
        self.nli = nli
        self.n_samples = n_samples
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.max_context_chars = max_context_chars
        self.hallucination_threshold = hallucination_threshold
        self.max_sentences = max_sentences
        self.require_model_match = require_model_match
        aliases = record_model_aliases or []
        if getattr(generator, "model_name", None):
            aliases.append(str(generator.model_name))
        self.record_model_aliases = {self._normalize_model_name(name) for name in aliases if name}

    def run(self, record: ResponseRecord) -> ComponentResult:
        if self.require_model_match and not self._matches_record_model(record):
            return ComponentResult(
                prediction=0,
                confidence=None,
                skip=True,
                skip_reason="Configured generator does not match the RAGTruth source generator",
            )
        context = (record.context or "")[: self.max_context_chars]
        system_prompt = (
            "Answer the user using only the supplied context. Be factual and concise. "
            "Do not mention that this is a verification experiment."
        )
        user_prompt = f"CONTEXT:\n{context}\n\nQUESTION:\n{record.query}"

        samples: list[str] = []
        total_cost = 0.0
        for _ in range(self.n_samples):
            generated = self.generator.generate(
                system_prompt,
                user_prompt,
                temperature=self.temperature,
                max_new_tokens=self.max_new_tokens,
            )
            samples.append(generated.text)
            total_cost += generated.cost

        sentences = split_sentences(record.generated_response)[: self.max_sentences]
        if not sentences:
            return ComponentResult(
                prediction=0,
                confidence=0.0,
                cost=total_cost,
                llm_calls=self.n_samples,
            )

        premises: list[str] = []
        hypotheses: list[str] = []
        slices: list[tuple[int, int]] = []
        for sentence in sentences:
            start = len(premises)
            for sample in samples:
                premises.append(sample)
                hypotheses.append(sentence)
            slices.append((start, len(premises)))

        nli_scores = self.nli.score(premises, hypotheses)
        sentence_risks: list[float] = []
        for start, end in slices:
            local = nli_scores[start:end]
            if not local:
                sentence_risks.append(1.0)
                continue
            mean_entail = sum(s.entailment for s in local) / len(local)
            mean_contradiction = sum(s.contradiction for s in local) / len(local)
            sentence_risks.append(max(1.0 - mean_entail, mean_contradiction))

        risk = max(sentence_risks, default=0.0)
        return ComponentResult(
            prediction=int(risk >= self.hallucination_threshold),
            confidence=float(risk),
            cost=total_cost,
            llm_calls=self.n_samples,
        )

    @staticmethod
    def _normalize_model_name(name: str) -> str:
        return re.sub(r"[^a-z0-9]+", "", name.lower())

    def _matches_record_model(self, record: ResponseRecord) -> bool:
        record_model = record.generation_metadata.model
        if not record_model or not self.record_model_aliases:
            return False
        normalized = self._normalize_model_name(record_model)
        return normalized in self.record_model_aliases
