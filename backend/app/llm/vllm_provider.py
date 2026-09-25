from __future__ import annotations

import asyncio
from contextlib import suppress
import json
from time import monotonic
from typing import Any, AsyncIterator

import httpx

from app.llm.base import (
    GenerationChunk,
    GenerationResult,
    LocalModelProvider,
    ModelGenerationCancelled,
    StructuredGenerationResult,
)
from app.monitoring.network import LocalNetworkPolicy
from app.resources.lifecycle import ModelLifecycleManager, get_model_lifecycle_manager
from app.resources.scheduler import ModelJob, ResourceScheduler, get_resource_scheduler


class VLLMProvider(LocalModelProvider):
    """OpenAI-compatible vLLM adapter restricted to an internal endpoint."""

    def __init__(
        self,
        endpoint: str,
        *,
        api_key: str = "local",
        allow_fallback: bool = True,
        enable_thinking: bool = False,
        max_tokens: int = 1280,
        timeout_seconds: float = 600.0,
        client: httpx.AsyncClient | None = None,
        scheduler: ResourceScheduler | None = None,
        lifecycle: ModelLifecycleManager | None = None,
        role: str = "GENERAL",
        memory_requirement: str = "medium",
        execution_mode: str = "STANDARD",
        priority: int = 50,
    ) -> None:
        LocalNetworkPolicy.require_local(endpoint)
        base = endpoint.rstrip("/")
        self.endpoint = base if base.endswith("/v1") else f"{base}/v1"
        self.api_key = api_key
        self.allow_fallback = allow_fallback
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.client = client
        self.scheduler = scheduler or get_resource_scheduler()
        self.lifecycle = lifecycle or get_model_lifecycle_manager()
        self.role = role
        self.memory_requirement = memory_requirement
        self.execution_mode = execution_mode
        self.priority = priority
        self._last_stats: dict[str, dict[str, object]] = {}

    def _job(self, model: str) -> ModelJob:
        return ModelJob(
            model=model,
            role=self.role,
            memory_requirement=self.memory_requirement,
            execution_mode=self.execution_mode,
            priority=self.priority,
        )

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _base_payload(self, prompt: str, model: str, system: str | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model,
            "messages": self._messages(prompt, system),
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        if not self.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
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
            raise RuntimeError("Local model returned no direct response")
        return GenerationResult(
            text=text,
            model=model,
            provider=final.provider if final else "vllm",
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
        payload = self._base_payload(prompt, model, system)
        payload.update({"stream": True, "stream_options": {"include_usage": True}})
        started = monotonic()
        first_token_at: float | None = None
        usage: dict[str, object] = {}
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds), follow_redirects=False)
        try:
            async with self.scheduler.acquire_model(self._job(model)) as permit:
                was_loaded = await self._is_model_available(client, model)
                self.lifecycle.begin(
                    model, was_loaded=was_loaded, queue_depth=self.scheduler.queue_depth
                )
                try:
                    saw_done = False
                    async with client.stream(
                        "POST", f"{self.endpoint}/chat/completions", headers=self.headers, json=payload
                    ) as response:
                        response.raise_for_status()
                        async for line in self._cancellable_lines(
                            response, cancellation_event, self.timeout_seconds
                        ):
                            if not line.startswith("data:"):
                                continue
                            raw = line[5:].strip()
                            if not raw:
                                continue
                            if raw == "[DONE]":
                                saw_done = True
                                break
                            item = json.loads(raw)
                            if isinstance(item.get("usage"), dict):
                                usage = item["usage"]
                            choices = item.get("choices") or []
                            token = ""
                            if choices and isinstance(choices[0], dict):
                                token = str((choices[0].get("delta") or {}).get("content") or "")
                            if token:
                                first_token_at = first_token_at or monotonic()
                                self.lifecycle.mark_busy(model)
                                yield GenerationChunk(text=token, model=model, provider="vllm")
                    if not saw_done:
                        raise ValueError("vLLM stream ended without a [DONE] record")
                    stats = self._runtime_stats(
                        usage, started, first_token_at, permit.queue_wait_seconds,
                        model, was_loaded,
                    )
                    self._last_stats[model] = stats
                    self.lifecycle.complete(model, stats)
                    yield GenerationChunk(
                        text="", model=model, provider="vllm", done=True, runtime_stats=stats
                    )
                except ModelGenerationCancelled:
                    self.lifecycle.fail(model, "Generation cancelled by caller or disconnected client.")
                    raise
                except (httpx.HTTPError, asyncio.TimeoutError, json.JSONDecodeError, ValueError) as exc:
                    self.lifecycle.fail(model, str(exc))
                    if not self.allow_fallback:
                        raise RuntimeError(f"Local inference unavailable: {exc}") from exc
                    stats = self._failure_stats(
                        started, first_token_at, was_loaded, permit.queue_wait_seconds, exc
                    )
                    self._last_stats[model] = stats
                    message = (
                        "Local model generation is deterministically unavailable. No model-generated "
                        "maintenance recommendation was fabricated; restore the configured vLLM runtime and retry."
                    )
                    yield GenerationChunk(text=message, model=model, provider="deterministic-unavailable", fallback=True)
                    yield GenerationChunk(
                        text="", model=model, provider="deterministic-unavailable", done=True,
                        fallback=True, runtime_stats=stats,
                    )
        finally:
            if owns_client:
                await client.aclose()

    async def generate_json(
        self, prompt: str, model: str, schema: dict[str, Any], system: str | None = None
    ) -> StructuredGenerationResult:
        payload: dict[str, object] = {
            "model": model,
            "messages": self._messages(prompt, system),
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "sovereign_response", "schema": schema},
            },
        }
        if not self.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        started = monotonic()
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds), follow_redirects=False)
        try:
            async with self.scheduler.acquire_model(self._job(model)) as permit:
                was_loaded = await self._is_model_available(client, model)
                self.lifecycle.begin(
                    model, was_loaded=was_loaded, queue_depth=self.scheduler.queue_depth
                )
                try:
                    response = await client.post(
                        f"{self.endpoint}/chat/completions", headers=self.headers, json=payload
                    )
                    response.raise_for_status()
                    body = response.json()
                    text = str(body["choices"][0]["message"]["content"]).strip()
                    data = json.loads(text)
                    if not isinstance(data, dict):
                        raise ValueError("Structured model response is not a JSON object")
                    usage = body.get("usage") or {}
                    stats = self._runtime_stats(
                        usage if isinstance(usage, dict) else {}, started, None,
                        permit.queue_wait_seconds, model, was_loaded,
                    )
                    self._last_stats[model] = stats
                    self.lifecycle.complete(model, stats)
                    return StructuredGenerationResult(
                        text=text, data=data, model=model, provider="vllm", runtime_stats=stats
                    )
                except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
                    self.lifecycle.fail(model, str(exc))
                    if not self.allow_fallback:
                        raise RuntimeError(f"Local structured inference unavailable: {exc}") from exc
                    stats = self._failure_stats(
                        started, None, was_loaded, permit.queue_wait_seconds, exc
                    )
                    self._last_stats[model] = stats
                    return StructuredGenerationResult(
                        text="", data=None, model=model, provider="deterministic-unavailable", fallback=True,
                        runtime_stats=stats,
                    )
        finally:
            if owns_client:
                await client.aclose()

    async def list_available_models(self) -> list[dict[str, object]]:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=2, follow_redirects=False)
        try:
            response = await client.get(f"{self.endpoint}/models", headers=self.headers)
            response.raise_for_status()
            data = response.json().get("data", [])
            return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
        finally:
            if owns_client:
                await client.aclose()

    async def health_check(self) -> dict[str, object]:
        try:
            return {"available": True, "endpoint": self.endpoint, "models": await self.list_available_models()}
        except (httpx.HTTPError, ValueError) as exc:
            return {"available": False, "endpoint": self.endpoint, "error": str(exc)}

    async def model_runtime_stats(self, model: str | None = None) -> dict[str, object]:
        try:
            models = await self.list_available_models()
            runtime_error: str | None = None
        except (httpx.HTTPError, ValueError) as exc:
            models = []
            runtime_error = str(exc)
        return {
            "model": model,
            "running_models": models,
            "last_generation": self._last_stats.get(model or "", {}),
            "lifecycle": self.lifecycle.get(model).model_dump(mode="json") if model else None,
            "runtime_error": runtime_error,
        }

    async def _is_model_available(
        self, client: httpx.AsyncClient, model: str
    ) -> bool | None:
        try:
            response = await client.get(f"{self.endpoint}/models", headers=self.headers)
            response.raise_for_status()
            data = response.json().get("data", [])
            if not isinstance(data, list):
                return None
            identifiers = {
                str(item.get("id") or item.get("model") or item.get("name"))
                for item in data
                if isinstance(item, dict)
            }
            return model in identifiers
        except (httpx.HTTPError, ValueError):
            return None

    @staticmethod
    async def _cancellable_lines(
        response: httpx.Response,
        cancellation_event: asyncio.Event | None,
        total_timeout_seconds: float | None = None,
    ) -> AsyncIterator[str]:
        iterator = response.aiter_lines().__aiter__()
        deadline = monotonic() + total_timeout_seconds if total_timeout_seconds else None
        while True:
            if deadline is not None and monotonic() >= deadline:
                raise asyncio.TimeoutError(
                    f"Local generation exceeded the {total_timeout_seconds:g} second total timeout."
                )
            if cancellation_event is None:
                try:
                    if deadline is None:
                        yield await anext(iterator)
                    else:
                        yield await asyncio.wait_for(
                            anext(iterator), max(0.001, deadline - monotonic())
                        )
                except StopAsyncIteration:
                    return
                continue
            if cancellation_event.is_set():
                raise ModelGenerationCancelled("Local model generation was cancelled.")
            line_task = asyncio.create_task(anext(iterator))
            cancel_task = asyncio.create_task(cancellation_event.wait())
            completed, _ = await asyncio.wait(
                {line_task, cancel_task},
                timeout=max(0.001, deadline - monotonic()) if deadline else None,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if not completed:
                line_task.cancel()
                cancel_task.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await line_task
                with suppress(asyncio.CancelledError):
                    await cancel_task
                raise asyncio.TimeoutError(
                    f"Local generation exceeded the {total_timeout_seconds:g} second total timeout."
                )
            if cancel_task in completed and cancellation_event.is_set():
                line_task.cancel()
                with suppress(asyncio.CancelledError, StopAsyncIteration):
                    await line_task
                raise ModelGenerationCancelled("Local model generation was cancelled.")
            cancel_task.cancel()
            with suppress(asyncio.CancelledError):
                await cancel_task
            try:
                yield line_task.result()
            except StopAsyncIteration:
                return

    @staticmethod
    def _runtime_stats(
        usage: dict[str, object],
        started: float,
        first_token_at: float | None,
        queue_wait_seconds: float,
        model: str,
        was_loaded: bool | None,
    ) -> dict[str, object]:
        finished = monotonic()
        completion_tokens = int(usage.get("completion_tokens") or 0)
        prompt_tokens = int(usage.get("prompt_tokens") or 0)
        decode_seconds = finished - first_token_at if first_token_at else None
        return {
            "available": True,
            "model": model,
            "ttft_ms": round(((first_token_at or finished) - started) * 1000, 3),
            "time_to_first_token_seconds": (
                round(first_token_at - started, 6) if first_token_at else None
            ),
            "total_duration_ms": round((finished - started) * 1000, 3),
            "total_duration_seconds": round(finished - started, 6),
            "prompt_tokens": prompt_tokens,
            "prompt_token_count": prompt_tokens,
            "completion_tokens": completion_tokens,
            "token_count": completion_tokens,
            "tokens_per_second": (
                round(completion_tokens / decode_seconds, 3)
                if decode_seconds and completion_tokens
                else None
            ),
            "warm_status": (
                "warm" if was_loaded is True else "cold" if was_loaded is False else "unknown"
            ),
            "queue_wait_seconds": round(queue_wait_seconds, 6),
            "completed": True,
        }

    @staticmethod
    def _failure_stats(
        started: float,
        first_token_at: float | None,
        was_loaded: bool | None,
        queue_wait_seconds: float,
        error: Exception,
    ) -> dict[str, object]:
        return {
            "available": False,
            "time_to_first_token_seconds": (
                round(first_token_at - started, 6) if first_token_at else None
            ),
            "total_duration_seconds": round(monotonic() - started, 6),
            "warm_status": (
                "warm" if was_loaded is True else "cold" if was_loaded is False else "unknown"
            ),
            "queue_wait_seconds": round(queue_wait_seconds, 6),
            "completed": False,
            "error": str(error),
        }
