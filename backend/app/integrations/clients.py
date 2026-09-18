from __future__ import annotations

import asyncio
import ipaddress
from typing import Any
from urllib.parse import urlparse

import httpx


RETRYABLE_STATUS_CODES = {408, 425, 429, 500, 502, 503, 504}


class IntegrationServiceError(RuntimeError):
    def __init__(
        self,
        service: str,
        message: str,
        *,
        status_code: int | None = None,
        attempts: int = 1,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.service = service
        self.status_code = status_code
        self.attempts = attempts
        self.retryable = retryable


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

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 60.0,
        client: httpx.AsyncClient | None = None,
        max_retries: int = 2,
        retry_backoff_seconds: float = 0.25,
        retry_backoff_max_seconds: float = 2.0,
    ) -> None:
        self.base_url = validate_internal_service_url(base_url)
        self.timeout = timeout
        self.client = client
        self.max_retries = max(0, int(max_retries))
        self.retry_backoff_seconds = max(0.0, float(retry_backoff_seconds))
        self.retry_backoff_max_seconds = max(self.retry_backoff_seconds, float(retry_backoff_max_seconds))

    async def request(self, method: str, path: str, *, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        owns_client = self.client is None
        client = self.client or httpx.AsyncClient(timeout=self.timeout, follow_redirects=False)
        attempts = 0
        last_error: Exception | None = None
        last_status: int | None = None
        last_retryable = False
        try:
            while attempts <= self.max_retries:
                attempts += 1
                try:
                    response = await client.request(method, f"{self.base_url}{path}", json=payload)
                    last_status = response.status_code
                    if response.status_code >= 400:
                        retryable = response.status_code in RETRYABLE_STATUS_CODES
                        last_retryable = retryable
                        if retryable and attempts <= self.max_retries:
                            await self._sleep_before_retry(attempts)
                            continue
                        response.raise_for_status()

                    value = response.json()
                    if not isinstance(value, dict):
                        raise ValueError("response was not a JSON object")
                    return value
                except httpx.HTTPStatusError as exc:
                    last_error = exc
                    last_status = exc.response.status_code
                    last_retryable = last_status in RETRYABLE_STATUS_CODES
                    break
                except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
                    last_error = exc
                    last_retryable = True
                    if attempts <= self.max_retries:
                        await self._sleep_before_retry(attempts)
                        continue
                    break
                except (httpx.HTTPError, ValueError) as exc:
                    last_error = exc
                    last_retryable = False
                    break

            detail = str(last_error) if last_error is not None else "request failed"
            raise IntegrationServiceError(
                self.service_name,
                f"{self.service_name} request failed after {attempts} attempt(s): {detail}",
                status_code=last_status,
                attempts=attempts,
                retryable=last_retryable,
            ) from last_error
        finally:
            if owns_client:
                await client.aclose()

    async def _sleep_before_retry(self, attempts: int) -> None:
        delay = min(
            self.retry_backoff_seconds * (2 ** max(0, attempts - 1)),
            self.retry_backoff_max_seconds,
        )
        if delay > 0:
            await asyncio.sleep(delay)

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
