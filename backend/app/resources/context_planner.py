from __future__ import annotations

from dataclasses import dataclass
from math import ceil


@dataclass(frozen=True, slots=True)
class ContextPlan:
    execution_mode: str
    estimated_prompt_tokens: int
    output_reserve_tokens: int
    required_tokens: int
    selected_num_ctx: int
    profile_max_num_ctx: int
    truncated_risk: bool
    source: str = "adaptive-context-v1"

    def to_dict(self) -> dict[str, object]:
        return {
            "execution_mode": self.execution_mode,
            "estimated_prompt_tokens": self.estimated_prompt_tokens,
            "output_reserve_tokens": self.output_reserve_tokens,
            "required_tokens": self.required_tokens,
            "selected_num_ctx": self.selected_num_ctx,
            "profile_max_num_ctx": self.profile_max_num_ctx,
            "truncated_risk": self.truncated_risk,
            "source": self.source,
        }


class AdaptiveContextPlanner:
    """Choose the smallest safe Ollama context bucket for a CPU-only request.

    Qwen tokenization is not available locally before the Ollama request, so this deliberately
    uses a conservative UTF-8 byte estimate (roughly <=3 bytes/token for typical English/code)
    and then adds the requested output budget. The planner never exceeds the execution profile's
    context ceiling, preserving FAST/STANDARD/DEEP semantics.
    """

    def __init__(
        self,
        *,
        buckets: tuple[int, ...] = (2048, 4096, 8192, 16384),
        minimum_context: int = 2048,
        bytes_per_token: float = 3.0,
        prompt_margin_tokens: int = 128,
    ) -> None:
        cleaned = sorted({int(value) for value in buckets if int(value) > 0})
        if not cleaned:
            raise ValueError("At least one positive context bucket is required")
        self.buckets = tuple(cleaned)
        self.minimum_context = max(256, int(minimum_context))
        self.bytes_per_token = max(1.0, float(bytes_per_token))
        self.prompt_margin_tokens = max(0, int(prompt_margin_tokens))

    def estimate_prompt_tokens(self, prompt: str, system: str | None = None) -> int:
        combined = prompt if not system else f"{system}\n{prompt}"
        byte_count = len(combined.encode("utf-8"))
        return max(1, ceil(byte_count / self.bytes_per_token) + self.prompt_margin_tokens)

    def plan(
        self,
        *,
        prompt: str,
        system: str | None,
        execution_mode: str,
        profile_max_num_ctx: int,
        output_reserve_tokens: int,
    ) -> ContextPlan:
        estimated_prompt = self.estimate_prompt_tokens(prompt, system)
        reserve = max(1, int(output_reserve_tokens))
        required = estimated_prompt + reserve
        ceiling = max(self.minimum_context, int(profile_max_num_ctx))

        eligible = [
            value
            for value in self.buckets
            if value >= self.minimum_context and value <= ceiling
        ]
        if ceiling not in eligible:
            eligible.append(ceiling)
        eligible = sorted(set(eligible))

        selected = ceiling
        for value in eligible:
            if value >= required:
                selected = value
                break

        return ContextPlan(
            execution_mode=str(execution_mode or "STANDARD").upper(),
            estimated_prompt_tokens=estimated_prompt,
            output_reserve_tokens=reserve,
            required_tokens=required,
            selected_num_ctx=selected,
            profile_max_num_ctx=ceiling,
            truncated_risk=required > ceiling,
        )
