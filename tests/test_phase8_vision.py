import asyncio
import json
from pathlib import Path

import httpx

from app.multimodal.vision import OllamaVisionProvider


def test_local_vision_adapter_returns_structured_uncertainty(tmp_path: Path) -> None:
    image = tmp_path / "pump.png"
    image.write_bytes(b"local-image-bytes")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.host == "127.0.0.1"
        payload = json.loads(request.content)
        assert payload["images"]
        return httpx.Response(200, json={"response": json.dumps({
            "description": "A pump assembly is visible.",
            "detected_components": ["pump", "motor"],
            "observations": ["Nameplate text is not legible"],
            "confidence": 0.78,
            "model": "local-vlm",
        })})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await OllamaVisionProvider("http://127.0.0.1:11434", "local-vlm", client).analyze_image(image, "Inspect")

    result = asyncio.run(run())
    assert result.available
    assert result.confidence == 0.78
    assert "motor" in result.detected_components


def test_vision_rejects_external_endpoint() -> None:
    try:
        OllamaVisionProvider("https://api.example.com", "cloud-model")
    except ValueError as exc:
        assert "blocked" in str(exc).lower()
    else:
        raise AssertionError("External endpoint was accepted")
