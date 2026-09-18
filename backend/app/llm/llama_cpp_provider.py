from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from time import monotonic
from typing import Any, AsyncIterator

import httpx

from app.core.config import get_settings
from app.llm.base import (
    GenerationChunk,
    GenerationResult,
    LocalModelProvider,
    ModelGenerationCancelled,
    StructuredGenerationResult,
)
from app.monitoring.network import LocalNetworkPolicy
from app.resources.context_planner import AdaptiveContextPlanner, ContextPlan
from app.resources.lifecycle import ModelLifecycleManager, get_model_lifecycle_manager
from app.resources.runtime_profiles import GenerationRuntimeProfile, generation_runtime_profile
from app.resources.scheduler import ModelJob, ResourceScheduler, get_resource_scheduler


class LlamaCppProvider(LocalModelProvider):
    """Adapter for a local llama.cpp HTTP server using OpenAI-compatible chat completions.

    llama.cpp owns model loading and KV allocation at server start. SovereignAI still serializes
    generation through the shared CPU scheduler and applies the same execution-mode output budgets
    and request-level context-risk accounting used by the Ollama path.
    """

    def __init__(
        self,
        endpoint: str,
        allow_fallback: bool = True,
        *,
        client: httpx.AsyncClient | None = None,
        scheduler: ResourceScheduler | None = None,
        lifecycle: ModelLifecycleManager | None = None,
        role: str = "GENERAL",
        memory_requirement: str = "medium",
        execution_mode: str = "STANDARD",
        priority: int = 50,
        timeout_seconds: float | None = None,
        server_context_size: int | None = None,
    ) -> None:
        LocalNetworkPolicy.require_local(endpoint)
        settings = get_settings()
        self.endpoint = endpoint.rstrip("/")
        self.allow_fallback = allow_fallback
        self.client = client
        self.scheduler = scheduler or get_resource_scheduler()
        self.lifecycle = lifecycle or get_model_lifecycle_manager()
        self.role = role
        self.memory_requirement = memory_requirement
        self.execution_mode = execution_mode
        self.priority = priority
        self.timeout_seconds = timeout_seconds or settings.model_generation_timeout_seconds
        self.server_context_size = max(512, int(server_context_size or settings.llama_cpp_context_size))
        buckets = tuple(
            int(item.strip())
            for item in str(settings.model_context_buckets).split(",")
            if item.strip() and item.strip().isdigit()
        ) or (2048, 4096, 8192, 16384)
        self.context_planner = AdaptiveContextPlanner(
            buckets=buckets,
            minimum_context=min(settings.model_context_minimum, self.server_context_size),
            bytes_per_token=settings.model_context_bytes_per_token,
            prompt_margin_tokens=settings.model_context_prompt_margin_tokens,
        )
        self._last_stats: dict[str, dict[str, object]] = {}

    def _job(self, model: str) -> ModelJob:
        # The llama.cpp server is a resident process; admission should preserve the OS reserve but
        # must not charge the already-loaded model a second time.
        return ModelJob(
            model=model,
            role=self.role,
            memory_requirement=self.memory_requirement,
            resident=True,
            execution_mode=self.execution_mode,
            priority=self.priority,
        )

    def _runtime(
        self,
        prompt: str,
        system: str | None,
        *,
        structured: bool,
    ) -> tuple[GenerationRuntimeProfile, ContextPlan]:
        profile = generation_runtime_profile(self.execution_mode, structured=structured)
        ceiling = min(profile.num_ctx, self.server_context_size)
        plan = self.context_planner.plan(
            prompt=prompt,
            system=system,
            execution_mode=self.execution_mode,
            profile_max_num_ctx=ceiling,
            output_reserve_tokens=profile.num_predict,
        )
        return profile, plan

    @staticmethod
    def _messages(prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _payload(
        self,
        prompt: str,
        model: str,
        system: str | None,
        profile: GenerationRuntimeProfile,
        *,
        stream: bool,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model,
            "messages": self._messages(prompt, system),
            "temperature": profile.temperature,
            "max_tokens": profile.num_predict,
            "stream": stream,
            # Supported by current llama.cpp chat templates; harmless for templates that ignore it.
            "chat_template_kwargs": {"enable_thinking": False},
            "reasoning_effort": "none",
        }
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    async def generate(self, prompt: str, model: str, system: str | None = None) -> GenerationResult:
        pieces: list[str] = []
        final: GenerationChunk | None = None
        async for chunk in self.stream(prompt, model, system):
            if chunk.text:
                pieces.append(chunk.text)
            if chunk.done:
                final = chunk
        text = "".join(pieces).strip()
        if not text:
            raise RuntimeError("llama.cpp returned no response")
        return GenerationResult(
            text=text,
            model=model,
            provider=final.provider if final else "llama_cpp",
            fallback=bool(final and final.fallback),
            runtime_stats=final.runtime_stats if final else {},
        )

    async def stream(
        self,
        prompt: str,
        model: str,
        system: str | None = None,
        *,
        cancellation_event: asyncio.Event | None = None,
    ) -> AsyncIterator[GenerationChunk]:
        if cancellation_event and cancellation_event.is_set():
            raise ModelGenerationCancelled("Local model generation was cancelled before it started.")
        profile, context_plan = self._runtime(prompt, system, structured=False)
        payload = self._payload(prompt, model, system, profile, stream=True)
        started = monotonic()
        first_token_at: float | None = None
        final_data: dict[str, Any] = {}
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds), follow_redirects=False
        )
        try:
            async with self.scheduler.acquire_model(self._job(model)) as permit:
                self.lifecycle.begin(model, was_loaded=True, queue_depth=self.scheduler.queue_depth)
                try:
                    async with client.stream(
                        "POST", f"{self.endpoint}/v1/chat/completions", json=payload
                    ) as response:
                        response.raise_for_status()
                        async for line in self._cancellable_lines(
                            response, cancellation_event, self.timeout_seconds
                        ):
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if not raw or raw == "[DONE]":
                                continue
                            item = json.loads(raw)
                            final_data = item
                            text = _stream_text(item)
                            if text:
                                first_token_at = first_token_at or monotonic()
                                self.lifecycle.mark_busy(model)
                                yield GenerationChunk(
                                    text=text,
                                    model=model,
                                    provider="llama_cpp",
                                    done=False,
                                )
                    stats = self._runtime_stats(
                        final_data,
                        started=started,
                        first_token_at=first_token_at,
                        queue_wait_seconds=permit.queue_wait_seconds,
                        profile=profile,
                        context_plan=context_plan,
                    )
                    self._last_stats[model] = stats
                    self.lifecycle.complete(model, stats)
                    yield GenerationChunk(
                        text="",
                        model=model,
                        provider="llama_cpp",
                        done=True,
                        runtime_stats=stats,
                    )
                except ModelGenerationCancelled:
                    self.lifecycle.fail(model, "Generation cancelled by caller or disconnected client.")
                    raise
                except (httpx.HTTPError, asyncio.TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
                    self.lifecycle.fail(model, str(exc))
                    if not self.allow_fallback:
                        raise RuntimeError(f"llama.cpp inference unavailable: {exc}") from exc
                    stats = self._failure_stats(started, first_token_at, permit.queue_wait_seconds, exc)
                    stats["runtime_profile"] = profile.metrics()
                    stats["context_plan"] = context_plan.to_dict()
                    yield GenerationChunk(
                        text="Local llama.cpp inference was unavailable. Check the local server and model before retrying.",
                        model=model,
                        provider="deterministic-unavailable",
                        fallback=True,
                    )
                    yield GenerationChunk(
                        text="",
                        model=model,
                        provider="deterministic-unavailable",
                        done=True,
                        fallback=True,
                        runtime_stats=stats,
                    )
        finally:
            if owns_client:
                await client.aclose()

    async def generate_json(
        self, prompt: str, model: str, schema: dict[str, Any], system: str | None = None
    ) -> StructuredGenerationResult:
        profile, context_plan = self._runtime(prompt, system, structured=True)
        payload = self._payload(
            prompt,
            model,
            system,
            profile,
            stream=False,
            response_format={"type": "json_schema", "schema": schema},
        )
        started = monotonic()
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds), follow_redirects=False
        )
        try:
            async with self.scheduler.acquire_model(self._job(model)) as permit:
                self.lifecycle.begin(model, was_loaded=True, queue_depth=self.scheduler.queue_depth)
                try:
                    response = await client.post(f"{self.endpoint}/v1/chat/completions", json=payload)
                    response.raise_for_status()
                    item = response.json()
                    text = _message_text(item).strip()
                    data = json.loads(text)
                    if not isinstance(data, dict):
                        raise ValueError("Structured llama.cpp response is not a JSON object")
                    stats = self._runtime_stats(
                        item,
                        started=started,
                        first_token_at=None,
                        queue_wait_seconds=permit.queue_wait_seconds,
                        profile=profile,
                        context_plan=context_plan,
                    )
                    self._last_stats[model] = stats
                    self.lifecycle.complete(model, stats)
                    return StructuredGenerationResult(
                        text=text,
                        data=data,
                        model=model,
                        provider="llama_cpp",
                        runtime_stats=stats,
                    )
                except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
                    self.lifecycle.fail(model, str(exc))
                    if not self.allow_fallback:
                        raise RuntimeError(f"llama.cpp structured inference unavailable: {exc}") from exc
                    return StructuredGenerationResult(
                        text="",
                        data=None,
                        model=model,
                        provider="deterministic-unavailable",
                        fallback=True,
                        runtime_stats=self._failure_stats(
                            started, None, permit.queue_wait_seconds, exc
                        ),
                    )
        finally:
            if owns_client:
                await client.aclose()

    async def health_check(self) -> dict[str, object]:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=2, follow_redirects=False)
        try:
            response = await client.get(f"{self.endpoint}/health")
            response.raise_for_status()
            return {"available": True, "endpoint": self.endpoint, "backend": "llama_cpp", **response.json()}
        except (httpx.HTTPError, ValueError) as exc:
            return {"available": False, "endpoint": self.endpoint, "backend": "llama_cpp", "error": str(exc)}
        finally:
            if owns_client:
                await client.aclose()

    async def list_available_models(self) -> list[dict[str, object]]:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=2, follow_redirects=False)
        try:
            response = await client.get(f"{self.endpoint}/v1/models")
            response.raise_for_status()
            items = response.json().get("data", [])
            return [item for item in items if isinstance(item, dict)] if isinstance(items, list) else []
        finally:
            if owns_client:
                await client.aclose()

    async def model_runtime_stats(self, model: str | None = None) -> dict[str, object]:
        return {
            "model": model,
            "backend": "llama_cpp",
            "endpoint": self.endpoint,
            "server_context_size": self.server_context_size,
            "last_generation": self._last_stats.get(model or "", {}),
            "lifecycle": self.lifecycle.get(model).model_dump(mode="json") if model else None,
        }

    @staticmethod
    async def _cancellable_lines(
        response: httpx.Response,
        cancellation_event: asyncio.Event | None,
        total_timeout_seconds: float | None,
    ) -> AsyncIterator[str]:
        iterator = response.aiter_lines().__aiter__()
        deadline = monotonic() + total_timeout_seconds if total_timeout_seconds else None
        while True:
            if cancellation_event and cancellation_event.is_set():
                raise ModelGenerationCancelled("Local model generation was cancelled.")
            remaining = max(0.001, deadline - monotonic()) if deadline else None
            try:
                if remaining is None:
                    line = await anext(iterator)
                else:
                    line = await asyncio.wait_for(anext(iterator), remaining)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError:
                raise asyncio.TimeoutError(
                    f"llama.cpp generation exceeded the {total_timeout_seconds:g} second total timeout."
                )
            yield line

    @staticmethod
    def _runtime_stats(
        response: dict[str, Any],
        *,
        started: float,
        first_token_at: float | None,
        queue_wait_seconds: float,
        profile: GenerationRuntimeProfile,
        context_plan: ContextPlan,
    ) -> dict[str, object]:
        timings = response.get("timings") if isinstance(response.get("timings"), dict) else {}
        usage = response.get("usage") if isinstance(response.get("usage"), dict) else {}
        predicted_n = int(timings.get("predicted_n") or usage.get("completion_tokens") or 0)
        prompt_n = int(
            (timings.get("prompt_n") or 0)
            + (timings.get("cache_n") or 0)
            or usage.get("prompt_tokens")
            or 0
        )
        choices = response.get("choices") if isinstance(response.get("choices"), list) else []
        finish_reason = "stop"
        if choices and isinstance(choices[0], dict):
            finish_reason = str(choices[0].get("finish_reason") or "stop")
        return {
            "model": str(response.get("model") or ""),
            "backend": "llama_cpp",
            "time_to_first_token_seconds": round(first_token_at - started, 6) if first_token_at else None,
            "tokens_per_second": _round_number(timings.get("predicted_per_second")),
            "prompt_tokens_per_second": _round_number(timings.get("prompt_per_second")),
            "total_duration_seconds": round(monotonic() - started, 6),
            "load_duration_seconds": 0.0,
            "token_count": predicted_n,
            "prompt_token_count": prompt_n,
            "warm_status": "resident-server",
            "queue_wait_seconds": round(queue_wait_seconds, 6),
            "done_reason": finish_reason,
            "output_truncated": finish_reason == "length",
            "completed": True,
            "runtime_profile": profile.metrics(),
            "context_plan": context_plan.to_dict(),
        }

    @staticmethod
    def _failure_stats(
        started: float,
        first_token_at: float | None,
        queue_wait_seconds: float,
        error: Exception,
    ) -> dict[str, object]:
        return {
            "backend": "llama_cpp",
            "time_to_first_token_seconds": round(first_token_at - started, 6) if first_token_at else None,
            "total_duration_seconds": round(monotonic() - started, 6),
            "warm_status": "unknown",
            "queue_wait_seconds": round(queue_wait_seconds, 6),
            "completed": False,
            "error": str(error),
        }


def _stream_text(item: dict[str, Any]) -> str:
    choices = item.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return ""
    delta = choices[0].get("delta")
    if not isinstance(delta, dict):
        return ""
    return str(delta.get("content") or "")


def _message_text(item: dict[str, Any]) -> str:
    choices = item.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise ValueError("llama.cpp response contains no completion choice")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise ValueError("llama.cpp response contains no assistant message")
    return str(message.get("content") or "")


def _round_number(value: object) -> float | None:
    return round(float(value), 3) if isinstance(value, (int, float)) else None
