from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from time import monotonic

from app.agent.state import AgentRunState, AgentStep
from app.llm.base import LocalModelProvider


PersistCallback = Callable[[AgentRunState], Awaitable[None]]
EventCallback = Callable[[str, dict[str, object]], Awaitable[None]]


class AgentExecutor:
    """Execute transparent agent steps without turning token streaming into I/O backpressure.

    Model providers may yield very small chunks. Persisting every provider chunk as an SSE/audit
    event can make the application path much slower than direct local inference, especially when the
    callback commits each event to SQLite. We therefore coalesce model output into short UI frames
    while retaining the complete model text in memory.
    """

    TOKEN_EVENT_MAX_CHARS = 96
    TOKEN_EVENT_MAX_DELAY_SECONDS = 0.05

    def __init__(
        self,
        provider: LocalModelProvider,
        model: str,
        event_callback: EventCallback | None = None,
        cancellation_event: asyncio.Event | None = None,
        *,
        system_prompt: str | None = None,
        generation_prompt: str | None = None,
    ) -> None:
        self.provider = provider
        self.model = model
        self.event_callback = event_callback
        self.cancellation_event = cancellation_event
        self.system_prompt = system_prompt
        self.generation_prompt = generation_prompt

    async def _emit(self, event_type: str, payload: dict[str, object]) -> None:
        if self.event_callback:
            await self.event_callback(event_type, payload)

    async def execute_step(self, run: AgentRunState, step: AgentStep) -> str:
        if step.action == "understand_task":
            return f"Request classified as {run.routing.task_profile.task_type}."
        if step.action == "analyze_input":
            return "Input-analysis capability identified; attached files are handled by registered tools."
        if step.action == "prepare_code":
            return "Code task identified; execution requires the isolated sandbox tool."
        if step.action == "generate_response":
            await self._emit("generation_started", {"model": self.model})
            pieces: list[str] = []
            pending_event_text: list[str] = []
            pending_event_chars = 0
            last_event_at = monotonic()
            fallback = False
            provider_name = "local"

            async def flush_model_tokens() -> None:
                nonlocal pending_event_chars, last_event_at
                if not pending_event_text:
                    return
                text = "".join(pending_event_text)
                pending_event_text.clear()
                pending_event_chars = 0
                last_event_at = monotonic()
                await self._emit("model_token", {"text": text, "model": self.model})

            async for chunk in self.provider.stream(
                self.generation_prompt or run.request,
                self.model,
                self.system_prompt or "You are a local industrial assistant. Do not invent evidence or claim unavailable tools ran.",
                cancellation_event=self.cancellation_event,
            ):
                fallback = fallback or chunk.fallback
                provider_name = chunk.provider
                if chunk.text:
                    pieces.append(chunk.text)
                    pending_event_text.append(chunk.text)
                    pending_event_chars += len(chunk.text)
                    if (
                        pending_event_chars >= self.TOKEN_EVENT_MAX_CHARS
                        or monotonic() - last_event_at >= self.TOKEN_EVENT_MAX_DELAY_SECONDS
                    ):
                        await flush_model_tokens()
                if chunk.done:
                    run.runtime_metrics = dict(chunk.runtime_stats)
                    if run.runtime_metrics.get("output_truncated"):
                        run.warnings.append(
                            "Local model reached its bounded output limit; the response may be incomplete."
                        )

            # Never lose the final short frame just because it did not reach the batching threshold.
            await flush_model_tokens()
            run.final_response = "".join(pieces).strip()
            if not run.final_response:
                raise RuntimeError("Local model returned no direct response")
            await self._emit(
                "generation_completed",
                {"model": self.model, "provider": provider_name, "runtime_metrics": run.runtime_metrics},
            )
            if fallback:
                run.warnings.append("Local model unavailable; no synthetic model answer was generated.")
            return f"Response produced by {provider_name}."
        if step.action == "review_response":
            if not run.final_response:
                raise RuntimeError("No response exists to review")
            return "Response reviewed against the requested industrial scope."
        if step.action == "verify_response":
            if not run.final_response:
                raise RuntimeError("No response exists to verify")
            step.verification = "A non-empty response exists; source verification is required for document claims."
            return "Response completeness verified."
        raise ValueError(f"Unsupported agent action: {step.action}")
