from pathlib import Path

from app.api.models import runtime_status
from app.router.model_registry import ModelRegistry


ROOT = Path(__file__).resolve().parents[1]


def test_configured_roles_share_installed_model_without_duplication() -> None:
    registry = ModelRegistry(ROOT / "config" / "models.yaml")
    assert registry.get("general").model_tag == "qwen3-vl:4b"
    assert registry.get("vision").model_tag == "qwen3-vl:4b"
    assert registry.get("coder").model_tag == "qwen2.5-coder:7b"
    assert len({model.model_tag for model in registry.all()}) == 2


def test_runtime_status_distinguishes_ready_missing_and_unavailable() -> None:
    model = ModelRegistry(ROOT / "config" / "models.yaml").get("coder")
    ready = runtime_status(model, {"available": True, "models": [{"name": "qwen2.5-coder:7b", "capabilities": ["completion", "tools"]}]})
    missing = runtime_status(model, {"available": True, "models": [{"name": "qwen3-vl:4b"}]})
    unavailable = runtime_status(model, {"available": False, "error": "connection refused"})
    assert ready.availability.value == "READY"
    assert ready.capabilities == ["completion", "tools"]
    assert missing.availability.value == "MODEL_NOT_INSTALLED"
    assert unavailable.availability.value == "OLLAMA_UNAVAILABLE"
