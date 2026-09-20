import asyncio
import json

import httpx

from app.resources.residency import OllamaResidencyManager


def test_residency_manager_preloads_and_unloads_models() -> None:
    resident: set[str] = set()
    control_requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(
                200,
                json={"models": [{"name": name} for name in sorted(resident)]},
            )
        if request.url.path == "/api/generate":
            payload = json.loads(request.content)
            control_requests.append(payload)
            model = str(payload["model"])
            if payload.get("keep_alive") == 0:
                resident.discard(model)
            else:
                resident.add(model)
            return httpx.Response(200, json={"model": model, "done": True})
        return httpx.Response(404)

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = OllamaResidencyManager(
            "http://127.0.0.1:11434",
            default_keep_alive="15m",
            client=client,
        )
        preload = await manager.ensure_warm("qwen3:4b-instruct")
        repeated = await manager.ensure_warm("qwen3:4b-instruct")
        unload = await manager.unload("qwen3:4b-instruct")
        await client.aclose()
        return preload, repeated, unload

    preload, repeated, unload = asyncio.run(run())
    assert preload.action == "preload"
    assert preload.resident_after is True
    assert repeated.action == "already_resident"
    assert unload.action == "unload"
    assert unload.resident_after is False
    assert control_requests[0]["keep_alive"] == "15m"
    assert control_requests[-1]["keep_alive"] == 0


def test_warm_pool_loads_models_sequentially_without_duplicates() -> None:
    resident: set[str] = set()
    requested_models: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/ps":
            return httpx.Response(200, json={"models": [{"name": name} for name in resident]})
        payload = json.loads(request.content)
        model = str(payload["model"])
        requested_models.append(model)
        resident.add(model)
        return httpx.Response(200, json={"model": model, "done": True})

    async def run():
        client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        manager = OllamaResidencyManager("http://localhost:11434", client=client)
        actions = await manager.warm_pool(["general", "coder", "general"])
        await client.aclose()
        return actions

    actions = asyncio.run(run())
    assert [item.model for item in actions] == ["general", "coder"]
    assert requested_models == ["general", "coder"]
