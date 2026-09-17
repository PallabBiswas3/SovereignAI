from __future__ import annotations

import json
import re

from adaptivefact.benchmark.runner import Component, ComponentResult
from adaptivefact.data.schema import ResponseRecord
from adaptivefact.generation.base import ChatGenerator

_JSON_RE = re.compile(r"\{.*?\}", re.DOTALL)


class LLMJudgeBaseline(Component):
    label = "Always-on LLM judge"

    def __init__(
        self,
        generator: ChatGenerator,
        *,
        max_context_chars: int = 12000,
        max_new_tokens: int = 120,
    ) -> None:
        self.generator = generator
        self.max_context_chars = max_context_chars
        self.max_new_tokens = max_new_tokens

    def run(self, record: ResponseRecord) -> ComponentResult:
        context = (record.context or "")[: self.max_context_chars]
        system_prompt = (
            "You are a factuality verifier for a retrieval-grounded QA system. "
            "Use only the supplied context as evidence. Mark the answer hallucinated if any factual "
            "statement is contradicted by the context or is presented as fact without support in the context. "
            "Do not use your own world knowledge. Return only JSON with keys hallucinated and confidence."
        )
        user_prompt = (
            f"QUESTION:\n{record.query}\n\n"
            f"CONTEXT:\n{context}\n\n"
            f"ANSWER TO CHECK:\n{record.generated_response}\n\n"
            'Return exactly: {"hallucinated": true|false, "confidence": 0.0-1.0}'
        )
        result = self.generator.generate(
            system_prompt,
            user_prompt,
            temperature=0.0,
            max_new_tokens=self.max_new_tokens,
        )
        pred, confidence = self._parse(result.text)
        return ComponentResult(
            prediction=pred,
            confidence=confidence,
            cost=result.cost,
            llm_calls=1,
        )

    @staticmethod
    def _parse(text: str) -> tuple[int, float]:
        match = _JSON_RE.search(text)
        if match:
            try:
                obj = json.loads(match.group(0))
                raw_hallucinated = obj.get("hallucinated", False)
                if isinstance(raw_hallucinated, str):
                    hallucinated = raw_hallucinated.strip().lower() in {"true", "yes", "1", "hallucinated"}
                else:
                    hallucinated = bool(raw_hallucinated)
                confidence = float(obj.get("confidence", 0.5))
                return int(hallucinated), max(0.0, min(1.0, confidence))
            except (json.JSONDecodeError, TypeError, ValueError):
                pass

        lowered = text.lower()
        if "not hallucinated" in lowered or "supported" in lowered:
            return 0, 0.6
        if "hallucinated" in lowered or "unsupported" in lowered or "contradicted" in lowered:
            return 1, 0.6
        return 1, 0.5
