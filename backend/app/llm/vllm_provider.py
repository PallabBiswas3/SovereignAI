from __future__ import annotations

import asyncio
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
    ) -> None:
        LocalNetworkPolicy.require_local(endpoint)
        self.endpoint = endpoint.rstrip("/")
        self.api_key = api_key
        self.allow_fallback = allow_fallback
        self.enable_thinking = enable_thinking
        self.max_tokens = max_tokens
        self.timeout_seconds = timeout_seconds
        self.client = client
        self._last_stats: dict[str, dict[str, object]] = {}

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _messages(self, prompt: str, system: str | None) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return messages

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
        payload: dict[str, object] = {
            "model": model,
            "messages": self._messages(prompt, system),
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": 0.2,
            "max_tokens": self.max_tokens,
        }
        if not self.enable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        started = monotonic()
        first_token_at: float | None = None
        usage: dict[str, object] = {}
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=httpx.Timeout(self.timeout_seconds), follow_redirects=False)
        try:
            try:
                async with client.stream(
                    "POST", f"{self.endpoint}/chat/completions", headers=self.headers, json=payload
                ) as response:
                    response.raise_for_status()
                    async for line in response.aiter_lines():
                        if cancellation_event and cancellation_event.is_set():
                            raise ModelGenerationCancelled("Local model generation was cancelled.")
                        if not line.startswith("data:"):
                            continue
                        raw = line[5:].strip()
                        if not raw or raw == "[DONE]":
                            continue
                        item = json.loads(raw)
                        if isinstance(item.get("usage"), dict):
                            usage = item["usage"]
                        choices = item.get("choices") or []
                        token = ""
                        if choices and isinstance(choices[0], dict):
                            token = str((choices[0].get("delta") or {}).get("content") or "")
                        if token:
                            first_token_at = first_token_at or monotonic()
                            yield GenerationChunk(text=token, model=model, provider="vllm")
                finished = monotonic()
                completion_tokens = int(usage.get("completion_tokens") or 0)
                generation_seconds = max(0.000001, finished - (first_token_at or finished))
                stats: dict[str, object] = {
                    "available": True,
                    "ttft_ms": round(((first_token_at or finished) - started) * 1000, 3),
                    "total_duration_ms": round((finished - started) * 1000, 3),
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": completion_tokens,
                    "tokens_per_second": round(completion_tokens / generation_seconds, 3) if completion_tokens else None,
                }
                self._last_stats[model] = stats
                yield GenerationChunk(text="", model=model, provider="vllm", done=True, runtime_stats=stats)
            except ModelGenerationCancelled:
                raise
            except (httpx.HTTPError, asyncio.TimeoutError, json.JSONDecodeError, ValueError) as exc:
                if not self.allow_fallback:
                    raise RuntimeError(f"Local inference unavailable: {exc}") from exc
                stats = {
                    "available": False,
                    "error": str(exc),
                    "total_duration_ms": round((monotonic() - started) * 1000, 3),
                }
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
                stats = {
                    "available": True,
                    "total_duration_ms": round((monotonic() - started) * 1000, 3),
                    "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                    "completion_tokens": int(usage.get("completion_tokens") or 0),
                }
                self._last_stats[model] = stats
                return StructuredGenerationResult(
                    text=text, data=data, model=model, provider="vllm", runtime_stats=stats
                )
            except (httpx.HTTPError, KeyError, IndexError, json.JSONDecodeError, ValueError) as exc:
                if not self.allow_fallback:
                    raise RuntimeError(f"Local structured inference unavailable: {exc}") from exc
                return StructuredGenerationResult(
                    text="", data=None, model=model, provider="deterministic-unavailable", fallback=True,
                    runtime_stats={"available": False, "error": str(exc)},
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
        except httpx.HTTPError as exc:
            return {"available": False, "endpoint": self.endpoint, "error": str(exc)}

    async def model_runtime_stats(self, model: str | None = None) -> dict[str, object]:
        return {"model": model, "last_generation": self._last_stats.get(model or "", {})}
