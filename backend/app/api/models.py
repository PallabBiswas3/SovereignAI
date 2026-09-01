from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.config import get_settings
from app.llm.ollama_provider import OllamaProvider
from app.router.model_registry import ModelRegistry
from app.router.model_router import ModelRouter
from app.router.schemas import ModelAvailability, ModelDefinition, ModelRuntimeStatus, RoutingDecision
from app.resources.lifecycle import get_model_lifecycle_manager
from app.resources.scheduler import get_resource_scheduler
from app.resources.cache import get_cache_backend
from app.rag.factory import configured_reranker
from app.rag.reranking import LocalCrossEncoderReranker


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
    endpoint_runtime: dict[str, dict[str, object]] = {}
    for endpoint in sorted({model.endpoint for model in registry.all()}):
        try:
            provider = OllamaProvider(endpoint)
            endpoint_results[endpoint] = await provider.health_check()
            endpoint_runtime[endpoint] = await provider.model_runtime_stats()
        except ValueError as exc:
            endpoint_results[endpoint] = {"available": False, "configuration_error": str(exc)}
    statuses = [
        runtime_status(
            model,
            endpoint_results.get(model.endpoint, {}),
            endpoint_runtime.get(model.endpoint, {}),
        )
        for model in registry.all()
    ]
    try:
        cache_stats = get_cache_backend().stats()
    except Exception as exc:
        cache_stats = {"hits": 0, "misses": 0, "entries": 0, "error": str(exc)}
    reranker = configured_reranker(get_settings())
    reranker_status = (
        reranker.status() if isinstance(reranker, LocalCrossEncoderReranker)
        else {"available": False, "disabled": True, "fallback": "RRF fusion ranking"}
    )
    return {
        "configured": len(statuses),
        "ready": sum(item.availability == ModelAvailability.ready for item in statuses),
        "ollama_available": any(bool(result.get("available")) for result in endpoint_results.values()),
        "models": [item.model_dump(mode="json") for item in statuses],
        "resources": get_resource_scheduler().snapshot().model_dump(mode="json"),
        "cache": cache_stats,
        "reranker": reranker_status,
        "retrieval": {
            "dense_version": get_settings().dense_retriever_version,
            "sparse_version": get_settings().bm25_index_version,
            "fusion_version": get_settings().fusion_strategy_version,
            "rrf_k": get_settings().hybrid_rrf_k,
        },
    }


def runtime_status(
    model: ModelDefinition,
    provider_status: dict[str, object],
    provider_runtime: dict[str, object] | None = None,
) -> ModelRuntimeStatus:
    provider_runtime = provider_runtime or {}
    if model.provider.lower() != "ollama" or not model.model_tag.strip():
        return ModelRuntimeStatus(
            id=model.id, role=model.role, display_name=model.display_name,
            model_tag=model.model_tag, endpoint=model.endpoint,
            availability=ModelAvailability.configuration_error, installed=False,
            detail="Only a non-empty local Ollama model tag is supported.",
            lifecycle_state="ERROR", warm_status="unavailable",
        )
    if provider_status.get("configuration_error"):
        return ModelRuntimeStatus(
            id=model.id, role=model.role, display_name=model.display_name,
            model_tag=model.model_tag, endpoint=model.endpoint,
            availability=ModelAvailability.configuration_error, installed=False,
            detail=str(provider_status["configuration_error"]),
            lifecycle_state="ERROR", warm_status="unavailable",
        )
    if not provider_status.get("available"):
        return ModelRuntimeStatus(
            id=model.id, role=model.role, display_name=model.display_name,
            model_tag=model.model_tag, endpoint=model.endpoint,
            availability=ModelAvailability.ollama_unavailable, installed=False,
            detail=str(provider_status.get("error", "Ollama did not respond.")),
            lifecycle_state="ERROR", warm_status="unknown",
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
            lifecycle_state="NOT_INSTALLED", warm_status="unavailable",
        )
    capabilities = matched.get("capabilities", [])
    running_models = provider_runtime.get("running_models", [])
    loaded: dict[str, object] | None = None
    if isinstance(running_models, list):
        loaded = next((
            item for item in running_models
            if isinstance(item, dict)
            and str(item.get("name") or item.get("model")) in {model.model_tag, f"{model.model_tag}:latest"}
        ), None)
    raw_memory = loaded.get("size_vram") if loaded else None
    memory_mb = round(float(raw_memory) / 1024 / 1024, 2) if isinstance(raw_memory, (int, float)) else None
    lifecycle = get_model_lifecycle_manager().observe(
        model.model_tag, installed=True, loaded=loaded is not None, memory_usage_mb=memory_mb
    )
    return ModelRuntimeStatus(
        id=model.id, role=model.role, display_name=model.display_name,
        model_tag=model.model_tag, endpoint=model.endpoint,
        availability=ModelAvailability.ready, installed=True,
        detail="Model is installed and Ollama is reachable.",
        capabilities=[str(value) for value in capabilities] if isinstance(capabilities, list) else [],
        lifecycle_state=lifecycle.state.value,
        warm_status=lifecycle.warm_status,
        memory_usage_mb=lifecycle.memory_usage_mb,
        runtime_metrics=lifecycle.model_dump(mode="json"),
    )


@router.get("/route", response_model=RoutingDecision)
async def preview_route(request: str, override: str | None = None) -> RoutingDecision:
    try:
        return ModelRouter(get_registry()).route(request, override)
    except KeyError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
