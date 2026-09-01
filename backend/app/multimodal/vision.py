from __future__ import annotations

import base64
import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel, Field
from app.monitoring.network import LocalNetworkPolicy
from app.resources.cache import CacheBackend, CacheKeyBuilder, CacheNamespace


class VisionAnalysis(BaseModel):
    description: str
    detected_components: list[str] = Field(default_factory=list)
    observations: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    available: bool = True
    warning: str | None = None
    model: str
    cache_hit: bool = False
    schema_version: str = "vision-analysis-v1"


class VisionProvider(ABC):
    @abstractmethod
    async def analyze_image(self, path: Path, prompt: str) -> VisionAnalysis:
        raise NotImplementedError


class OllamaVisionProvider(VisionProvider):
    """Local-only Ollama VLM adapter with structured, uncertainty-aware output."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        client: httpx.AsyncClient | None = None,
        cache: CacheBackend | None = None,
        schema_version: str = "vision-analysis-v1",
    ) -> None:
        LocalNetworkPolicy.require_local(endpoint)
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.client = client
        self.cache = cache
        self.schema_version = schema_version

    async def analyze_image(self, path: Path, prompt: str) -> VisionAnalysis:
        cache_key = CacheKeyBuilder.vision(path, self.model, prompt, self.schema_version)
        if self.cache:
            cached = self.cache.get(CacheNamespace.vision.value, cache_key)
            if isinstance(cached, dict):
                return VisionAnalysis.model_validate({**cached, "cache_hit": True})
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        schema = VisionAnalysis.model_json_schema()
        payload = {
            "model": self.model,
            "prompt": (
                "Analyze the supplied enterprise image as evidence. Do not infer precise engineering "
                "measurements that are not visible. Return JSON only. User task: " + prompt
            ),
            "images": [encoded],
            "format": schema,
            "stream": False,
        }
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=180, follow_redirects=False)
        try:
            response = await client.post(f"{self.endpoint}/api/generate", json=payload)
            response.raise_for_status()
            raw = response.json().get("response", "{}")
            values: dict[str, Any] = json.loads(raw) if isinstance(raw, str) else raw
            values.setdefault("model", self.model)
            values.setdefault("available", True)
            values.setdefault("schema_version", self.schema_version)
            result = VisionAnalysis.model_validate(values)
            if self.cache:
                self.cache.set(
                    CacheNamespace.vision.value,
                    cache_key,
                    result.model_dump(mode="json"),
                    metadata={"model": self.model, "schema_version": self.schema_version},
                )
            return result
        except (httpx.HTTPError, json.JSONDecodeError, ValueError, KeyError) as exc:
            return VisionAnalysis(
                description="No visual analysis was generated.", available=False, model=self.model,
                warning=f"Configured local vision model is unavailable or returned invalid structured output: {exc}",
            )
        finally:
            if owns_client:
                await client.aclose()
