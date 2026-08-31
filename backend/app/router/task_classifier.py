from __future__ import annotations

import re

from app.router.schemas import TaskProfile


class TaskClassifier:
    """Transparent local classifier used before any LLM call."""

    CODING = {"code", "python", "script", "debug", "function", "algorithm", "compile", "test"}
    VISION = {"image", "photograph", "drawing", "diagram", "visual", "scan", "scanned", "pid", "p&id"}
    DOCUMENT = {"document", "pdf", "report", "sop", "manual", "policy", "spreadsheet", "file"}
    SUMMARY = {"summarize", "summary", "briefing", "executive", "condense", "one-page"}
    REASONING = {"analyze", "compare", "calculate", "recommend", "evaluate", "why", "optimization"}

    @staticmethod
    def _density(words: set[str], vocabulary: set[str], floor: float = 0.0) -> float:
        matches = len(words & vocabulary)
        return min(1.0, max(floor, matches * 0.28 + (0.18 if matches else 0.0)))

    def classify(self, request: str) -> TaskProfile:
        words = set(re.findall(r"[a-z0-9&+-]+", request.lower()))
        coding = self._density(words, self.CODING)
        vision = self._density(words, self.VISION)
        document = self._density(words, self.DOCUMENT)
        summary = self._density(words, self.SUMMARY)
        reasoning = self._density(words, self.REASONING, floor=0.35)

        requirements = {
            "coding": coding,
            "vision": vision,
            "document": document,
            "summarization": summary,
        }
        task_type = max(requirements, key=requirements.get)
        if requirements[task_type] == 0:
            task_type = "general"
        estimated_tokens = max(512, min(131_072, len(request) * 8))
        return TaskProfile(
            task_type=task_type,
            coding_requirement=coding,
            reasoning_requirement=reasoning,
            vision_requirement=vision,
            document_requirement=document,
            summarization_requirement=summary,
            latency_priority=0.5,
            context_length_required=estimated_tokens,
        )

