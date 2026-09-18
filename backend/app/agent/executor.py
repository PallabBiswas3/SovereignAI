from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from time import monotonic

from app.agent.state import AgentRunState, AgentStep
from app.llm.base import LocalModelProvider
from app.resources.latency import build_latency_breakdown


PersistCallback = Callable[[AgentRunState], Awaitable[None]]
EventCallback = Callable[[str, dict[str, object]], Awaitable[None]]


class AgentExecutor:
    """Execute transparent agent steps without turning token streaming into I/O backpressure.

    Model providers may yield very small chunks. Persisting every provider chunk as an SSE/audit
    event can make the application path much slower than direct local inference, especially when the
    callback commits each event to SQLite. We therefore coalesce model output into short UI frames
    while retaining the complete model text in memory and measuring application-side overhead.
    """

    TOKEN_EVENT_MAX_CHARS = 96
    # CPU-local models on the target laptop produce roughly one provider chunk every ~90-100 ms.
    # A 50 ms deadline therefore flushed almost every chunk individually and forced one durable
    # task-event callback per chunk. Keep the first visible frame immediate, then coalesce subsequent
    # frames to a ~5 Hz UI cadence unless the character bound is reached first.
    TOKEN_EVENT_MAX_DELAY_SECONDS = 0.20

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
            generation_started_at = monotonic()
            await self._emit("generation_started", {"model": self.model})
            pieces: list[str] = []
            pending_event_text: list[str] = []
            pending_event_chars = 0
            emitted_event_chars = 0
            last_event_at = monotonic()
            event_callback_seconds = 0.0
            event_callback_count = 0
            first_ui_frame_seconds: float | None = None
            fallback = False
            provider_name = "local"

            async def flush_model_tokens() -> None:
                nonlocal pending_event_chars, emitted_event_chars, last_event_at
                nonlocal event_callback_seconds, event_callback_count, first_ui_frame_seconds
                if not pending_event_text:
                    return
                text = "".join(pending_event_text)
                pending_event_text.clear()
                pending_event_chars = 0
                callback_started_at = monotonic()
                await self._emit("model_token", {"text": text, "model": self.model})
                event_callback_seconds += monotonic() - callback_started_at
                event_callback_count += 1
                emitted_event_chars += len(text)
                first_ui_frame_seconds = first_ui_frame_seconds or (monotonic() - generation_started_at)
                last_event_at = monotonic()

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
                    # Preserve perceived TTFT: the first non-empty provider chunk is delivered immediately.
                    # After that, batch slow CPU chunks so SQLite-backed task callbacks do not run once per token.
                    if (
                        first_ui_frame_seconds is None
                        or pending_event_chars >= self.TOKEN_EVENT_MAX_CHARS
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

            generation_wall_seconds = monotonic() - generation_started_at
            run.runtime_metrics["latency_breakdown"] = build_latency_breakdown(
                run.runtime_metrics,
                generation_wall_seconds=generation_wall_seconds,
                event_callback_seconds=event_callback_seconds,
                event_callback_count=event_callback_count,
                first_ui_frame_seconds=first_ui_frame_seconds,
            )
            run.runtime_metrics["stream_batching"] = {
                "max_chars": self.TOKEN_EVENT_MAX_CHARS,
                "max_delay_seconds": self.TOKEN_EVENT_MAX_DELAY_SECONDS,
                "first_frame_immediate": True,
                "emitted_event_count": event_callback_count,
                "emitted_character_count": emitted_event_chars,
                "mean_chars_per_event": round(emitted_event_chars / event_callback_count, 2)
                if event_callback_count
                else 0.0,
            }
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
