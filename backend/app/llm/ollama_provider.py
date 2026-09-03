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
from app.resources.lifecycle import ModelLifecycleManager, get_model_lifecycle_manager
from app.resources.scheduler import ModelJob, ResourceScheduler, get_resource_scheduler


class OllamaProvider(LocalModelProvider):
    """Local Ollama adapter with true NDJSON streaming and bounded admission."""

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
        keep_alive: str | None = None,
        timeout_seconds: float | None = None,
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
        self.keep_alive = keep_alive or settings.model_keep_alive or f"{settings.model_idle_timeout_seconds}s"
        self.timeout_seconds = timeout_seconds or settings.model_generation_timeout_seconds
        self._last_stats: dict[str, dict[str, object]] = {}

    def _job(self, model: str) -> ModelJob:
        return ModelJob(
            model=model,
            role=self.role,
            memory_requirement=self.memory_requirement,
            execution_mode=self.execution_mode,
            priority=self.priority,
        )

    @staticmethod
    def _direct_prompt(prompt: str, model: str) -> str:
        lowered = model.lower()
        if lowered.startswith("qwen3") and "-instruct" not in lowered:
            return f"/no_think\n{prompt}"
        return prompt

    def _base_payload(self, prompt: str, model: str, system: str | None) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": model,
            "prompt": self._direct_prompt(prompt, model),
            "think": False,
            "keep_alive": self.keep_alive,
        }
        if system:
            payload["system"] = system
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
            provider=final.provider if final else "ollama",
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
        payload.update({
            "stream": True,
            "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1280},
        })
        started = monotonic()
        first_token_at: float | None = None
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds), follow_redirects=False
        )
        try:
            async with self.scheduler.acquire_model(self._job(model)) as permit:
                was_loaded = await self._is_model_loaded(client, model)
                self.lifecycle.begin(model, was_loaded=was_loaded, queue_depth=self.scheduler.queue_depth)
                try:
                    async with client.stream(
                        "POST", f"{self.endpoint}/api/generate", json=payload
                    ) as response:
                        response.raise_for_status()
                        async for line in self._cancellable_lines(
                            response, cancellation_event, self.timeout_seconds
                        ):
                            if not line.strip():
                                continue
                            item = json.loads(line)
                            token = str(item.get("response", ""))
                            if token:
                                first_token_at = first_token_at or monotonic()
                                self.lifecycle.mark_busy(model)
                                yield GenerationChunk(
                                    text=token, model=model, provider="ollama", done=False
                                )
                            if item.get("done"):
                                stats = self._runtime_stats(
                                    item,
                                    started=started,
                                    first_token_at=first_token_at,
                                    was_loaded=was_loaded,
                                    queue_wait_seconds=permit.queue_wait_seconds,
                                )
                                self._last_stats[model] = stats
                                self.lifecycle.complete(model, stats)
                                yield GenerationChunk(
                                    text="",
                                    model=model,
                                    provider="ollama",
                                    done=True,
                                    runtime_stats=stats,
                                )
                                return
                    raise ValueError("Ollama stream ended without a completion record")
                except ModelGenerationCancelled:
                    self.lifecycle.fail(model, "Generation cancelled by caller or disconnected client.")
                    raise
                except (httpx.HTTPError, asyncio.TimeoutError, KeyError, json.JSONDecodeError, ValueError) as exc:
                    self.lifecycle.fail(model, str(exc))
                    if not self.allow_fallback:
                        raise RuntimeError(f"Local inference unavailable: {exc}") from exc
                    stats = self._failure_stats(started, first_token_at, was_loaded, permit.queue_wait_seconds, exc)
                    self._last_stats[model] = stats
                    fallback = (
                        "Local model generation was unavailable or exceeded its timeout. The request was "
                        "stored locally, but no model-generated answer was fabricated. Check Ollama, the "
                        "configured model, and available CPU/RAM before retrying."
                    )
                    yield GenerationChunk(
                        text=fallback,
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
                        yield await asyncio.wait_for(anext(iterator), max(0.001, deadline - monotonic()))
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

    async def list_available_models(self) -> list[dict[str, object]]:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=2, follow_redirects=False)
        try:
            response = await client.get(f"{self.endpoint}/api/tags")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [item for item in models if isinstance(item, dict)] if isinstance(models, list) else []
        finally:
            if owns_client:
                await client.aclose()

    async def health_check(self) -> dict[str, object]:
        try:
            return {
                "available": True,
                "endpoint": self.endpoint,
                "models": await self.list_available_models(),
            }
        except httpx.HTTPError as exc:
            return {"available": False, "endpoint": self.endpoint, "error": str(exc)}

    async def model_runtime_stats(self, model: str | None = None) -> dict[str, object]:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=2, follow_redirects=False)
        try:
            response = await client.get(f"{self.endpoint}/api/ps")
            response.raise_for_status()
            running = response.json().get("models", [])
            running_models = [item for item in running if isinstance(item, dict)] if isinstance(running, list) else []
        except httpx.HTTPError as exc:
            running_models = []
            runtime_error: str | None = str(exc)
        else:
            runtime_error = None
        finally:
            if owns_client:
                await client.aclose()
        return {
            "model": model,
            "running_models": running_models,
            "last_generation": self._last_stats.get(model or "", {}),
            "lifecycle": self.lifecycle.get(model).model_dump(mode="json") if model else None,
            "runtime_error": runtime_error,
        }

    async def _is_model_loaded(self, client: httpx.AsyncClient, model: str) -> bool | None:
        try:
            response = await client.get(f"{self.endpoint}/api/ps")
            response.raise_for_status()
            running = response.json().get("models", [])
            if not isinstance(running, list):
                return None
            names = {
                str(item.get("name") or item.get("model"))
                for item in running
                if isinstance(item, dict)
            }
            return model in names or f"{model}:latest" in names
        except httpx.HTTPError:
            return None

    async def generate_json(
        self, prompt: str, model: str, schema: dict[str, Any], system: str | None = None
    ) -> StructuredGenerationResult:
        payload = self._base_payload(prompt, model, system)
        payload.update({
            "format": schema,
            "stream": False,
            "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 4096},
        })
        started = monotonic()
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds), follow_redirects=False
        )
        try:
            async with self.scheduler.acquire_model(self._job(model)) as permit:
                was_loaded = await self._is_model_loaded(client, model)
                self.lifecycle.begin(model, was_loaded=was_loaded, queue_depth=self.scheduler.queue_depth)
                try:
                    response = await client.post(f"{self.endpoint}/api/generate", json=payload)
                    response.raise_for_status()
                    response_data = response.json()
                    text = str(response_data.get("response", "")).strip()
                    if not text:
                        candidate = str(response_data.get("thinking", "")).strip()
                        parsed_candidate = json.loads(candidate)
                        if not isinstance(parsed_candidate, dict):
                            raise ValueError("Structured model response is not a JSON object")
                        text = candidate
                    data = json.loads(text)
                    if not isinstance(data, dict):
                        raise ValueError("Structured model response is not a JSON object")
                    stats = self._runtime_stats(
                        response_data,
                        started=started,
                        first_token_at=None,
                        was_loaded=was_loaded,
                        queue_wait_seconds=permit.queue_wait_seconds,
                    )
                    self._last_stats[model] = stats
                    self.lifecycle.complete(model, stats)
                    return StructuredGenerationResult(
                        text=text,
                        data=data,
                        model=model,
                        provider="ollama",
                        runtime_stats=stats,
                    )
                except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
                    self.lifecycle.fail(model, str(exc))
                    if not self.allow_fallback:
                        raise RuntimeError(f"Local structured inference unavailable: {exc}") from exc
                    return StructuredGenerationResult(
                        text="",
                        data=None,
                        model=model,
                        provider="deterministic-unavailable",
                        fallback=True,
                        runtime_stats=self._failure_stats(
                            started, None, was_loaded, permit.queue_wait_seconds, exc
                        ),
                    )
        finally:
            if owns_client:
                await client.aclose()

    @staticmethod
    def _runtime_stats(
        response: dict[str, object],
        *,
        started: float,
        first_token_at: float | None,
        was_loaded: bool | None,
        queue_wait_seconds: float,
    ) -> dict[str, object]:
        total_seconds = _nanoseconds_to_seconds(response.get("total_duration")) or (monotonic() - started)
        load_seconds = _nanoseconds_to_seconds(response.get("load_duration"))
        eval_seconds = _nanoseconds_to_seconds(response.get("eval_duration"))
        token_count = int(response.get("eval_count", 0) or 0)
        done_reason = str(response.get("done_reason") or "stop")
        return {
            "model": str(response.get("model", "")),
            "time_to_first_token_seconds": round(first_token_at - started, 6) if first_token_at else None,
            "tokens_per_second": round(token_count / eval_seconds, 3) if eval_seconds and token_count else None,
            "total_duration_seconds": round(total_seconds, 6),
            "load_duration_seconds": round(load_seconds, 6) if load_seconds is not None else None,
            "token_count": token_count,
            "prompt_token_count": int(response.get("prompt_eval_count", 0) or 0),
            "warm_status": "warm" if was_loaded is True else "cold" if was_loaded is False else "unknown",
            "queue_wait_seconds": round(queue_wait_seconds, 6),
            "done_reason": done_reason,
            "output_truncated": done_reason == "length",
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
            "time_to_first_token_seconds": round(first_token_at - started, 6) if first_token_at else None,
            "total_duration_seconds": round(monotonic() - started, 6),
            "warm_status": "warm" if was_loaded is True else "cold" if was_loaded is False else "unknown",
            "queue_wait_seconds": round(queue_wait_seconds, 6),
            "completed": False,
            "error": str(error),
        }


def _nanoseconds_to_seconds(value: object) -> float | None:
    return float(value) / 1_000_000_000 if isinstance(value, (int, float)) else None
