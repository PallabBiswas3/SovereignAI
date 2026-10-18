"""Backend credentials stay on the service hop; unauthorized failures never retry."""
import asyncio

import httpx
import pytest

from app.integrations.clients import GraphRagClient, IntegrationServiceError


def test_missing_credentials_fail_before_transport():
    async def run():
        def forbidden(request):
            pytest.fail("unauthenticated request reached transport")
        async with httpx.AsyncClient(transport=httpx.MockTransport(forbidden)) as transport:
            client = GraphRagClient("http://localhost:3100", client=transport, service_token="")
            with pytest.raises(IntegrationServiceError, match="authentication"):
                await client.retrieve("synthetic query")
    asyncio.run(run())


@pytest.mark.parametrize("status", [200, 401, 403])
def test_service_bearer_and_permanent_auth_failure(status):
    calls = []
    async def run():
        def respond(request):
            calls.append(request)
            assert request.headers["Authorization"] == "Bearer " + "s" * 32
            return httpx.Response(status, json={"evidence": []})
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as transport:
            client = GraphRagClient("http://localhost:3100", client=transport, service_token="s" * 32)
            if status == 200:
                assert await client.retrieve("synthetic query") == {"evidence": []}
            else:
                with pytest.raises(IntegrationServiceError) as error:
                    await client.retrieve("synthetic query")
                assert error.value.retryable is False
                assert error.value.status_code == status
    asyncio.run(run())
    assert len(calls) == 1
