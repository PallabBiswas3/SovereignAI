from __future__ import annotations

import httpx
import json
from typing import Any

from app.llm.base import GenerationResult, LocalModelProvider, StructuredGenerationResult
from app.monitoring.network import LocalNetworkPolicy


class OllamaProvider(LocalModelProvider):
    """Local Ollama adapter. It never redirects and only accepts loopback endpoints."""

    def __init__(self, endpoint: str, allow_fallback: bool = True) -> None:
        LocalNetworkPolicy.require_local(endpoint)
        self.endpoint = endpoint.rstrip("/")
        self.allow_fallback = allow_fallback

    async def generate(self, prompt: str, model: str, system: str | None = None) -> GenerationResult:
        direct_prompt = f"/no_think\n{prompt}" if model.lower().startswith("qwen3") else prompt
        payload: dict[str, object] = {
            "model": model, "prompt": direct_prompt, "stream": False, "think": False,
            "options": {"temperature": 0.2, "num_ctx": 8192, "num_predict": 1024},
        }
        if system:
            payload["system"] = system
        try:
            # CPU-only Qwen inference can legitimately take more than two minutes.
            async with httpx.AsyncClient(timeout=300, follow_redirects=False) as client:
                response = await client.post(f"{self.endpoint}/api/generate", json=payload)
                response.raise_for_status()
            response_text = str(response.json()["response"]).strip()
            if not response_text:
                raise ValueError("Local model returned no direct response")
            return GenerationResult(text=response_text, model=model, provider="ollama")
        except (httpx.HTTPError, KeyError, ValueError) as exc:
            if not self.allow_fallback:
                raise RuntimeError(f"Local inference unavailable: {exc}") from exc
            return GenerationResult(
                text=(
                    "Local model generation was unavailable or exceeded its timeout. The request was "
                    "stored locally, but no model-generated answer was fabricated. Check Ollama, the "
                    "configured model, and available CPU/RAM before retrying."
                ),
                model=model,
                provider="deterministic-unavailable",
                fallback=True,
            )

    async def health_check(self) -> dict[str, object]:
        try:
            async with httpx.AsyncClient(timeout=2, follow_redirects=False) as client:
                response = await client.get(f"{self.endpoint}/api/tags")
                response.raise_for_status()
            return {"available": True, "endpoint": self.endpoint, "models": response.json().get("models", [])}
        except httpx.HTTPError as exc:
            return {"available": False, "endpoint": self.endpoint, "error": str(exc)}

    async def generate_json(
        self, prompt: str, model: str, schema: dict[str, Any], system: str | None = None
    ) -> StructuredGenerationResult:
        direct_prompt = f"/no_think\n{prompt}" if model.lower().startswith("qwen3") else prompt
        payload: dict[str, object] = {
            "model": model,
            "prompt": direct_prompt,
            "format": schema,
            "stream": False,
            # Tool/code protocols need the JSON answer, not an unreturned thinking trace.
            "think": False,
            "options": {"temperature": 0.1, "num_ctx": 8192, "num_predict": 4096},
        }
        if system:
            payload["system"] = system
        try:
            async with httpx.AsyncClient(timeout=300, follow_redirects=False) as client:
                response = await client.post(f"{self.endpoint}/api/generate", json=payload)
                response.raise_for_status()
            response_data = response.json()
            text = str(response_data.get("response", "")).strip()
            # Some Qwen/Ollama template combinations place schema JSON in the
            # thinking field even with thinking disabled. Accept it only when
            # it is itself the requested JSON object; never expose raw traces.
            if not text:
                candidate = str(response_data.get("thinking", "")).strip()
                parsed_candidate = json.loads(candidate)
                if not isinstance(parsed_candidate, dict):
                    raise ValueError("Structured model response is not a JSON object")
                text = candidate
            data = json.loads(text)
            if not isinstance(data, dict):
                raise ValueError("Structured model response is not a JSON object")
            return StructuredGenerationResult(
                text=text, data=data, model=model, provider="ollama"
            )
        except (httpx.HTTPError, KeyError, json.JSONDecodeError, ValueError) as exc:
            if not self.allow_fallback:
                raise RuntimeError(f"Local structured inference unavailable: {exc}") from exc
            return StructuredGenerationResult(
                text="", data=None, model=model, provider="deterministic-unavailable",
                fallback=True,
            )
