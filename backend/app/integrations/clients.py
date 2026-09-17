from __future__ import annotations

import ipaddress
from typing import Any
from urllib.parse import urlparse

import httpx


class IntegrationServiceError(RuntimeError):
    def __init__(self, service: str, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.service = service
        self.status_code = status_code


def validate_internal_service_url(value: str) -> str:
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise ValueError(f"invalid integration service URL: {value!r}")
    hostname = parsed.hostname.lower()
    allowed = hostname == "localhost" or "." not in hostname
    if not allowed:
        try:
            allowed = ipaddress.ip_address(hostname).is_private or ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            allowed = False
    if not allowed:
        raise ValueError(f"integration service must use a loopback, private, or internal service address: {value!r}")
    return value.rstrip("/")


class JsonServiceClient:
    service_name = "service"

    def __init__(self, base_url: str, *, timeout: float = 60.0, client: httpx.AsyncClient | None = None) -> None:
        self.base_url = validate_internal_service_url(base_url)
        self.timeout = timeout
        self.client = client

    async def request(self, method: str, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.timeout, follow_redirects=False)
        try:
            response = await client.request(method, f"{self.base_url}{path}", json=payload)
            response.raise_for_status()
            value = response.json()
            if not isinstance(value, dict):
                raise ValueError("response was not a JSON object")
            return value
        except (httpx.HTTPError, ValueError) as exc:
            status = exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None
            raise IntegrationServiceError(
                self.service_name,
                f"{self.service_name} request failed: {exc}",
                status_code=status,
            ) from exc
        finally:
            if owns_client:
                await client.aclose()

    async def health(self) -> dict[str, Any]:
        return await self.request("GET", "/health")


class GraphRagClient(JsonServiceClient):
    service_name = "graph-rag"

    async def retrieve(self, query: str, verification_mode: str = "thorough") -> dict[str, Any]:
        return await self.request(
            "POST", "/api/integration/retrieve",
            payload={"query": query, "verification_mode": verification_mode},
        )


class DiagnosticAgentClient(JsonServiceClient):
    service_name = "time-series-diagnostic-agent"

    async def diagnose(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.request("POST", "/v1/diagnose", payload=payload)


class ControlPlaneClient(JsonServiceClient):
    service_name = "controlplane"

    async def precheck(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.request("POST", "/v1/precheck", payload=payload)

    async def check(self, payload: dict[str, Any]) -> dict[str, Any]:
        return await self.request("POST", "/v1/check", payload=payload)
