"""Bounded, non-generating checks for the supported industrial workflow.

Responses deliberately omit credentials, document contents, endpoints and raw
exceptions. Liveness remains independent at /health.
"""
from __future__ import annotations

import asyncio
import os

import httpx
from sqlalchemy import select, text
import yaml

from app.core.config import Settings, validate_runtime_settings
from app.integrations.clients import validate_internal_service_url
from app.router.model_registry import ModelRegistry


def configuration_checks(settings: Settings) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    try:
        validate_runtime_settings(settings)
        checks["runtime_configuration"] = True
    except (ValueError, RuntimeError):
        checks["runtime_configuration"] = False
    for name, path, key in (
        ("access_configuration", settings.access_config, "roles"),
        ("policy_configuration", settings.policies_config, "policies"),
        ("tool_configuration", settings.tools_config, "tools"),
    ):
        try:
            value = yaml.safe_load(path.read_text(encoding="utf-8"))
            checks[name] = isinstance(value, dict) and bool(value.get(key))
        except (OSError, ValueError, yaml.YAMLError):
            checks[name] = False
    try:
        models = ModelRegistry(settings.models_config).all()
        checks["model_configuration"] = any(m.role.upper() == "GENERAL" and m.supports_text for m in models)
    except (OSError, ValueError, yaml.YAMLError):
        checks["model_configuration"] = False
    checks["retrieval_credential"] = len(settings.graphrag_service_token) >= 32
    for name, path in (("workspace_storage", settings.workspace_root),
                       ("knowledge_storage", settings.knowledge_root),
                       ("capsule_storage", settings.capsules_root)):
        checks[name] = path.is_dir() and os.access(path, os.R_OK | os.W_OK)
    return checks


async def readiness_status(settings: Settings, *, client=None, session_factory=None) -> dict:
    checks = configuration_checks(settings)
    if session_factory is None:
        from app.core.database import SessionLocal
        session_factory = SessionLocal
    try:
        from app.core.database import UserRecord
        with session_factory() as db:
            db.execute(text("SELECT 1"))
            checks["database"] = True
            checks["identity_provisioned"] = db.scalar(select(UserRecord.id).where(UserRecord.enabled.is_(True)).limit(1)) is not None
    except Exception:
        checks["database"] = False
        checks["identity_provisioned"] = False

    async def service(name, endpoint):
        try:
            validate_internal_service_url(endpoint)
            response = await http.get(f"{endpoint.rstrip('/')}/health")
            response.raise_for_status()
            value = response.json()
            return name, isinstance(value, dict) and value.get("status") == "ok"
        except (httpx.HTTPError, ValueError, TypeError):
            return name, False

    async def model():
        try:
            # The first deployment profile supports Ollama. Unsupported providers
            # must not accidentally pass a readiness check intended for Ollama.
            if settings.llm_provider.lower() != "ollama":
                return "generation_model", False
            validate_internal_service_url(settings.ollama_url)
            response = await http.get(f"{settings.ollama_url.rstrip('/')}/api/tags")
            response.raise_for_status()
            installed = response.json()["models"]
            if not isinstance(installed, list) or any(not isinstance(item, dict) for item in installed):
                return "generation_model", False
            names = {str(item.get("name") or item.get("model")) for item in installed}
            required = [m for m in ModelRegistry(settings.models_config).all() if m.role.upper() == "GENERAL"]
            return "generation_model", bool(required) and all(
                m.provider.lower() == "ollama" and m.endpoint.rstrip('/') == settings.ollama_url.rstrip('/')
                and (m.model_tag in names or f"{m.model_tag}:latest" in names) for m in required)
        except (httpx.HTTPError, OSError, ValueError, KeyError, TypeError, yaml.YAMLError):
            return "generation_model", False

    owned = client is None
    http = client or httpx.AsyncClient(timeout=settings.readiness_timeout_seconds, follow_redirects=False, trust_env=False)
    try:
        checks.update(await asyncio.gather(
            service("graphrag", settings.graphrag_url),
            service("controlplane", settings.controlplane_url),
            service("diagnostics", settings.diagnostics_url), model(),
        ))
    finally:
        if owned:
            await http.aclose()
    return {"ready": all(checks.values()), "workflow": "pump102-assessment", "checks": checks,
            "scope": "configuration, storage, identity, service health and installed model; not workflow acceptance"}
