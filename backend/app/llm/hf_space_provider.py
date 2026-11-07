"""Authenticated Gradio API adapter for our private Hugging Face ZeroGPU Space."""
from __future__ import annotations

import asyncio
from contextlib import aclosing
import json
import re
from time import monotonic
from urllib.parse import urlsplit

import httpx

from app.llm.base import GenerationChunk, ModelGenerationCancelled, StructuredGenerationResult
from app.llm.remote_vllm_provider import RemoteVLLMProvider, validate_remote_endpoint
from app.monitoring.timing import measure_generation, measure_stream


def validate_hf_settings(settings) -> None:
    if not settings.remote_inference_enabled:
        raise ValueError("Set SOVEREIGN_REMOTE_INFERENCE_ENABLED=true to use a Hugging Face GPU")
    validate_space_url(settings.hf_space_url)
    if not settings.hf_space_model.strip():
        raise ValueError("Set SOVEREIGN_HF_SPACE_MODEL to the deployed model ID")
    token = settings.hf_token.get_secret_value()
    if not token.startswith("hf_") or any(char.isspace() for char in token):
        raise ValueError("Configure SOVEREIGN_HF_TOKEN or SOVEREIGN_HF_TOKEN_FILE")


def validate_space_url(endpoint: str) -> None:
    validate_remote_endpoint(endpoint)
    parsed = urlsplit(endpoint)
    if (not parsed.hostname.endswith(".hf.space") or parsed.path not in {"", "/"}
            or parsed.port not in {None, 443}):
        raise ValueError("Use the deployed Space's HTTPS hf.space root URL")


class HuggingFaceSpaceProvider(RemoteVLLMProvider):
    provider_name = "hf_space"
    inference_label = "Hugging Face"
    _validate_endpoint = staticmethod(validate_space_url)

    def __init__(self, endpoint: str, **kwargs):
        super().__init__(endpoint, **kwargs)
        # The inherited OpenAI adapter appends /v1; Gradio has its own API root.
        self.endpoint = endpoint.rstrip("/")

    async def _packets(self, client, api_name, data, cancellation_event=None, timeout=None):
        if cancellation_event and cancellation_event.is_set():
            raise ModelGenerationCancelled("Hugging Face generation cancelled")
        deadline = timeout or self.timeout_seconds
        async with asyncio.timeout(deadline):
            response = await client.post(
                f"{self.endpoint}/gradio_api/call/{api_name}", headers=self.headers,
                json={"data": data}, follow_redirects=False)
            response.raise_for_status()
            event_id = response.json().get("event_id")
            if not isinstance(event_id, str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,128}", event_id):
                raise ValueError("Space returned an invalid event ID")
            event = ""
            async with client.stream(
                "GET", f"{self.endpoint}/gradio_api/call/{api_name}/{event_id}",
                headers=self.headers, follow_redirects=False,
            ) as response:
                response.raise_for_status()
                async for line in self._cancellable_lines(response, cancellation_event, deadline):
                    if line.startswith("event:"):
                        event = line[6:].strip()
                    elif line.startswith("data:"):
                        if event == "error":
                            raise ValueError("Space rejected the request or exhausted its GPU quota")
                        if event not in {"generating", "complete"}:
                            continue
                        output = json.loads(line[5:].strip())
                        if not isinstance(output, list) or len(output) != 1 or not isinstance(output[0], dict):
                            raise ValueError("Invalid Space response")
                        yield event, output[0]
                        if event == "complete":
                            return
            raise ValueError("Space stream ended before completion")

    async def _generation(self, prompt, model, system, schema, cancellation_event=None):
        if cancellation_event and cancellation_event.is_set():
            raise ModelGenerationCancelled("Hugging Face generation cancelled")
        started = monotonic()
        first = None
        previous = ""
        owns_client = self.client is None
        client = self.client or self._new_client()
        try:
            async with self.scheduler.acquire_model(self._job(model)) as permit:
                self.lifecycle.begin(model, was_loaded=None, queue_depth=self.scheduler.queue_depth)
                try:
                    async for event, packet in self._packets(
                        client, "generate", [prompt, system or "", min(self.max_tokens, 1024),
                                             json.dumps(schema) if schema is not None else ""], cancellation_event,
                    ):
                        text = packet.get("text")
                        if packet.get("model") != model or not isinstance(text, str) or not text.startswith(previous):
                            raise ValueError("Space model or stream contract mismatch")
                        delta, previous = text[len(previous):], text
                        if delta:
                            first = first or monotonic()
                            self.lifecycle.mark_busy(model)
                            yield GenerationChunk(text=delta, model=model, provider=self.provider_name)
                        if event == "complete":
                            if packet.get("done") is not True or not text.strip():
                                raise ValueError("Space returned an incomplete answer")
                            usage = packet.get("usage") if isinstance(packet.get("usage"), dict) else {}
                            stats = self._runtime_stats(usage, started, first, permit.queue_wait_seconds, model, None)
                            stats["remote_gpu_quota"] = "Hugging Face account quota applies"
                            self._last_stats[model] = stats
                            self.lifecycle.complete(model, stats)
                            yield GenerationChunk(text="", model=model, provider=self.provider_name,
                                                  done=True, runtime_stats=stats)
                except (asyncio.CancelledError, ModelGenerationCancelled):
                    self.lifecycle.fail(model, "Generation cancelled")
                    raise
                except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError) as exc:
                    detail = self._error_detail(exc)
                    self.lifecycle.fail(model, detail)
                    raise RuntimeError(f"Hugging Face inference unavailable: {detail}") from None
        finally:
            if owns_client:
                await client.aclose()

    @measure_stream
    async def stream(self, prompt, model, system=None, *, cancellation_event=None):
        async with aclosing(self._generation(prompt, model, system, None, cancellation_event)) as chunks:
            async for chunk in chunks:
                yield chunk

    @measure_generation
    async def generate_json(self, prompt, model, schema, system=None):
        pieces, stats = [], {}
        async for chunk in self._generation(prompt, model, system, schema):
            pieces.append(chunk.text)
            if chunk.done:
                stats = chunk.runtime_stats
        text = "".join(pieces).strip()
        try:
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Expected JSON object")
        except ValueError:
            raise RuntimeError("Hugging Face returned invalid structured output") from None
        return StructuredGenerationResult(text=text, model=model, provider=self.provider_name,
                                           data=data, runtime_stats=stats)

    async def list_available_models(self):
        owns_client = self.client is None
        client = self.client or self._new_client(timeout=10)
        try:
            async for event, packet in self._packets(
                    client, "runtime_info", [], timeout=min(10, client.timeout.read or 10)):
                if event == "complete":
                    if packet.get("protocol") != "sovereign-hf-v1" or not packet.get("model"):
                        raise ValueError("Space does not implement the SovereignAI inference API")
                    return [{"id": packet["model"]}]
        finally:
            if owns_client:
                await client.aclose()

    async def health_check(self):
        try:
            return {"available": True, "endpoint": self.endpoint,
                    "models": await self.list_available_models()}
        except (httpx.HTTPError, TimeoutError, ValueError, KeyError, TypeError) as exc:
            return {"available": False, "endpoint": self.endpoint, "error": self._error_detail(exc)}

    async def model_runtime_stats(self, model=None):
        return {"model": model, "running_models": [], "last_generation": self._last_stats.get(model or "", {}),
                "warm_status": "unknown", "detail": "ZeroGPU allocates hardware per request"}
