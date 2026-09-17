from __future__ import annotations

import json

from adaptivefact.verification.nli import NLIScorer, NLIScores


class OpenAIEvidenceJudge(NLIScorer):
    """Evidence-only NLI adapter using Responses API structured outputs."""

    def __init__(self, model: str, *, reasoning_effort: str = "low") -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError("Install the optional 'judge' dependencies to use OpenAI judges") from exc
        self.client = OpenAI()
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def score(self, premises: list[str], hypotheses: list[str]) -> list[NLIScores]:
        if len(premises) != len(hypotheses):
            raise ValueError("premises and hypotheses must have the same length")
        scores = []
        for premise, hypothesis in zip(premises, hypotheses):
            response = self.client.responses.create(
                model=self.model,
                reasoning={"effort": self.reasoning_effort},
                store=False,
                input=(
                    "Classify the claim using only the supplied evidence. "
                    "Use unknown when evidence is missing, irrelevant, or conflicting.\n\n"
                    f"EVIDENCE:\n{premise}\n\nCLAIM:\n{hypothesis}"
                ),
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "evidence_judgment",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "status": {"type": "string", "enum": ["supported", "contradicted", "unknown"]},
                                "confidence": {"type": "number", "minimum": 0, "maximum": 1},
                            },
                            "required": ["status", "confidence"],
                            "additionalProperties": False,
                        },
                    }
                },
            )
            payload = json.loads(response.output_text)
            confidence = float(payload["confidence"])
            status = payload["status"]
            residual = (1.0 - confidence) / 2.0
            values = {"entailment": residual, "contradiction": residual, "neutral": residual}
            values[{"supported": "entailment", "contradicted": "contradiction", "unknown": "neutral"}[status]] = confidence
            scores.append(NLIScores(**values))
            self.calls += 1
            usage = getattr(response, "usage", None)
            self.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)
        return scores
