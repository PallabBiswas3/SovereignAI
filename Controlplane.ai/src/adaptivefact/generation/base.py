from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class GenerationResult:
    text: str
    latency_ms: float
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost: float = 0.0


class ChatGenerator(abc.ABC):
    """Minimal provider-neutral interface used by judge and self-consistency baselines."""

    model_name: str | None = None

    @abc.abstractmethod
    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        temperature: float = 0.0,
        max_new_tokens: int = 256,
    ) -> GenerationResult:
        raise NotImplementedError
