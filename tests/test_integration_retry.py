from __future__ import annotations

import httpx
import pytest

from app.integrations.clients import GraphRagClient, IntegrationServiceError


@pytest.mark.asyncio
async def test_retries_transient_status_then_succeeds() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            return httpx.Response(503, json={"detail": "warming up"})
        return httpx.Response(200, json={"status": "grounded", "chunks": []})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        graph = GraphRagClient(
            "http://127.0.0.1:3100",
            client=client,
            max_retries=2,
            retry_backoff_seconds=0,
        )
        result = await graph.retrieve("pump evidence")

    assert result["status"] == "grounded"
    assert attempts == 2


@pytest.mark.asyncio
async def test_permanent_client_error_is_not_retried() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, json={"detail": "bad request"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        graph = GraphRagClient(
            "http://127.0.0.1:3100",
            client=client,
            max_retries=3,
            retry_backoff_seconds=0,
        )
        with pytest.raises(IntegrationServiceError) as exc_info:
            await graph.retrieve("bad payload")

    assert attempts == 1
    assert exc_info.value.status_code == 400
    assert exc_info.value.attempts == 1
    assert exc_info.value.retryable is False


@pytest.mark.asyncio
async def test_retry_exhaustion_reports_attempt_count() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, json={"detail": "unavailable"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        graph = GraphRagClient(
            "http://127.0.0.1:3100",
            client=client,
            max_retries=2,
            retry_backoff_seconds=0,
        )
        with pytest.raises(IntegrationServiceError) as exc_info:
            await graph.retrieve("pump evidence")

    assert attempts == 3
    assert exc_info.value.status_code == 503
    assert exc_info.value.attempts == 3
    assert exc_info.value.retryable is True


@pytest.mark.asyncio
async def test_network_error_retries_then_succeeds() -> None:
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise httpx.ConnectError("connection refused", request=request)
        return httpx.Response(200, json={"status": "ok"})

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport) as client:
        graph = GraphRagClient(
            "http://127.0.0.1:3100",
            client=client,
            max_retries=2,
            retry_backoff_seconds=0,
        )
        result = await graph.health()

    assert result["status"] == "ok"
    assert attempts == 2
