from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.llm.ollama_provider import OllamaProvider
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.router.schemas import ModelAvailability, ModelDefinition, ModelRuntimeStatus, RoutingDecision


router = APIRouter(prefix="/api/models", tags=["models"])


def get_registry() -> ModelRegistry:
    return ModelRegistry(get_settings().models_config)


@router.get("", response_model=list[ModelDefinition])
async def list_models() -> list[ModelDefinition]:
    return get_registry().all()


@router.get("/status")
async def model_status() -> dict[str, object]:
    try:
        registry = get_registry()
    except (ValueError, OSError) as exc:
        return {
            "configured": 0,
            "ready": 0,
            "ollama_available": False,
            "models": [],
            "configuration_error": str(exc),
        }
    endpoint_results: dict[str, dict[str, object]] = {}
    for endpoint in sorted({model.endpoint for model in registry.all()}):
        try:
            endpoint_results[endpoint] = await OllamaProvider(endpoint).health_check()
        except ValueError as exc:
            endpoint_results[endpoint] = {"available": False, "configuration_error": str(exc)}
    statuses = [runtime_status(model, endpoint_results.get(model.endpoint, {})) for model in registry.all()]
    return {
        "configured": len(statuses),
        "ready": sum(item.availability == ModelAvailability.ready for item in statuses),
        "ollama_available": any(bool(result.get("available")) for result in endpoint_results.values()),
        "models": [item.model_dump(mode="json") for item in statuses],
    }


def runtime_status(model: ModelDefinition, provider_status: dict[str, object]) -> ModelRuntimeStatus:
    if model.provider.lower() != "ollama" or not model.model_tag.strip():
        return ModelRuntimeStatus(
            id=model.id, role=model.role, display_name=model.display_name,
            model_tag=model.model_tag, endpoint=model.endpoint,
            availability=ModelAvailability.configuration_error, installed=False,
            detail="Only a non-empty local Ollama model tag is supported.",
        )
    if provider_status.get("configuration_error"):
        return ModelRuntimeStatus(
            id=model.id, role=model.role, display_name=model.display_name,
            model_tag=model.model_tag, endpoint=model.endpoint,
            availability=ModelAvailability.configuration_error, installed=False,
            detail=str(provider_status["configuration_error"]),
        )
    if not provider_status.get("available"):
        return ModelRuntimeStatus(
            id=model.id, role=model.role, display_name=model.display_name,
            model_tag=model.model_tag, endpoint=model.endpoint,
            availability=ModelAvailability.ollama_unavailable, installed=False,
            detail=str(provider_status.get("error", "Ollama did not respond.")),
        )
    raw_models = provider_status.get("models", [])
    installed: dict[str, dict[str, object]] = {}
    if isinstance(raw_models, list):
        for item in raw_models:
            if isinstance(item, dict):
                for key in ("name", "model"):
                    if item.get(key):
                        installed[str(item[key])] = item
    matched = installed.get(model.model_tag) or installed.get(f"{model.model_tag}:latest")
    if not matched:
        return ModelRuntimeStatus(
            id=model.id, role=model.role, display_name=model.display_name,
            model_tag=model.model_tag, endpoint=model.endpoint,
            availability=ModelAvailability.model_not_installed, installed=False,
            detail=f"Ollama is running, but '{model.model_tag}' is not installed.",
        )
    capabilities = matched.get("capabilities", [])
    return ModelRuntimeStatus(
        id=model.id, role=model.role, display_name=model.display_name,
        model_tag=model.model_tag, endpoint=model.endpoint,
        availability=ModelAvailability.ready, installed=True,
        detail="Model is installed and Ollama is reachable.",
        capabilities=[str(value) for value in capabilities] if isinstance(capabilities, list) else [],
    )


@router.get("/route", response_model=RoutingDecision)
async def preview_route(request: str, override: str | None = None) -> RoutingDecision:
    try:
        return ModelRouter(get_registry()).route(request, override)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
