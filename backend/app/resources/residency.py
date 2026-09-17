from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import httpx

from app.monitoring.network import LocalNetworkPolicy


@dataclass(frozen=True, slots=True)
class ResidencyAction:
    model: str
    action: str
    resident_before: bool
    resident_after: bool
    keep_alive: str | int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "action": self.action,
            "resident_before": self.resident_before,
            "resident_after": self.resident_after,
            "keep_alive": self.keep_alive,
        }


class OllamaResidencyManager:
    """Explicitly manage Ollama model residency without hiding memory policy.

    This layer deliberately does not make eviction decisions. It provides the load/unload
    primitives and observable resident-model state that the RAM/VRAM-aware scheduler can use
    in the next phase.
    """

    def __init__(
        self,
        endpoint: str,
        *,
        default_keep_alive: str | int = "15m",
        timeout_seconds: float = 120.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        LocalNetworkPolicy.require_local(endpoint)
        self.endpoint = endpoint.rstrip("/")
        self.default_keep_alive = default_keep_alive
        self.timeout_seconds = timeout_seconds
        self.client = client

    async def list_resident_models(self) -> list[dict[str, object]]:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(
            timeout=httpx.Timeout(min(self.timeout_seconds, 5.0)), follow_redirects=False
        )
        try:
            response = await client.get(f"{self.endpoint}/api/ps")
            response.raise_for_status()
            models = response.json().get("models", [])
            return [item for item in models if isinstance(item, dict)] if isinstance(models, list) else []
        finally:
            if owns_client:
                await client.aclose()

    async def is_resident(self, model: str) -> bool:
        target = _normalized_names(model)
        return any(
            str(item.get("name") or item.get("model") or "") in target
            for item in await self.list_resident_models()
        )

    async def ensure_warm(
        self,
        model: str,
        *,
        keep_alive: str | int | None = None,
    ) -> ResidencyAction:
        keep_alive = self.default_keep_alive if keep_alive is None else keep_alive
        before = await self.is_resident(model)
        if before:
            return ResidencyAction(model, "already_resident", True, True, keep_alive)

        await self._generate_control_request(model, keep_alive=keep_alive)
        after = await self.is_resident(model)
        return ResidencyAction(model, "preload", False, after, keep_alive)

    async def unload(self, model: str) -> ResidencyAction:
        before = await self.is_resident(model)
        if not before:
            return ResidencyAction(model, "already_unloaded", False, False, 0)

        await self._generate_control_request(model, keep_alive=0)
        after = await self.is_resident(model)
        return ResidencyAction(model, "unload", True, after, 0)

    async def warm_pool(
        self,
        models: Iterable[str],
        *,
        keep_alive: str | int | None = None,
    ) -> list[ResidencyAction]:
        """Warm models sequentially to avoid simultaneous cold-load memory spikes."""
        actions: list[ResidencyAction] = []
        for model in dict.fromkeys(str(item).strip() for item in models if str(item).strip()):
            actions.append(await self.ensure_warm(model, keep_alive=keep_alive))
        return actions

    async def _generate_control_request(self, model: str, *, keep_alive: str | int) -> dict[str, Any]:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(
            timeout=httpx.Timeout(self.timeout_seconds), follow_redirects=False
        )
        try:
            response = await client.post(
                f"{self.endpoint}/api/generate",
                json={"model": model, "stream": False, "keep_alive": keep_alive},
            )
            response.raise_for_status()
            data = response.json()
            return data if isinstance(data, dict) else {}
        finally:
            if owns_client:
                await client.aclose()


def _normalized_names(model: str) -> set[str]:
    value = model.strip()
    if not value:
        return set()
    return {value, f"{value}:latest"} if ":" not in value else {value}
