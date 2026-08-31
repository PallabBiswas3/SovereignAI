from pathlib import Path

from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter


CONFIG = Path(__file__).resolve().parents[1] / "config" / "models.yaml"


def test_routes_code_task_to_coder() -> None:
    decision = ModelRouter(ModelRegistry(CONFIG)).route(
        "Write and test a Python function, debug it and execute the code"
    )
    assert decision.model_id == "coder"
    assert decision.task_profile.task_type == "coding"
    assert decision.confidence > 0.5


def test_routes_scanned_drawing_to_vision() -> None:
    decision = ModelRouter(ModelRegistry(CONFIG)).route(
        "Analyze this scanned engineering drawing and photograph"
    )
    assert decision.model_id == "vision"
    assert decision.task_profile.vision_requirement > 0.5


def test_manual_override_is_respected() -> None:
    decision = ModelRouter(ModelRegistry(CONFIG)).route("Write Python code", override="general")
    assert decision.model_id == "general"
    assert decision.manual_override is True

