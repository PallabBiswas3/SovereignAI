from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GenerationRuntimeProfile:
    mode: str
    num_ctx: int
    num_predict: int
    temperature: float

    def ollama_options(self) -> dict[str, int | float]:
        return {
            "temperature": self.temperature,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
        }

    def metrics(self) -> dict[str, int | float | str]:
        return {
            "mode": self.mode,
            "num_ctx": self.num_ctx,
            "num_predict": self.num_predict,
            "temperature": self.temperature,
        }


_STREAMING_PROFILES = {
    "FAST": GenerationRuntimeProfile("FAST", num_ctx=4096, num_predict=384, temperature=0.2),
    "STANDARD": GenerationRuntimeProfile("STANDARD", num_ctx=8192, num_predict=1024, temperature=0.2),
    "DEEP": GenerationRuntimeProfile("DEEP", num_ctx=16384, num_predict=1536, temperature=0.2),
}

_STRUCTURED_PROFILES = {
    "FAST": GenerationRuntimeProfile("FAST", num_ctx=4096, num_predict=1024, temperature=0.1),
    "STANDARD": GenerationRuntimeProfile("STANDARD", num_ctx=8192, num_predict=2048, temperature=0.1),
    "DEEP": GenerationRuntimeProfile("DEEP", num_ctx=16384, num_predict=4096, temperature=0.1),
}


def generation_runtime_profile(
    execution_mode: str | None,
    *,
    structured: bool = False,
) -> GenerationRuntimeProfile:
    """Return a bounded local-inference budget for the selected execution depth.

    Automatic mode should already have been resolved by the orchestration layer. Unknown values
    intentionally fall back to STANDARD rather than silently selecting the most expensive profile.
    """

    mode = str(execution_mode or "STANDARD").upper()
    profiles = _STRUCTURED_PROFILES if structured else _STREAMING_PROFILES
    return profiles.get(mode, profiles["STANDARD"])
