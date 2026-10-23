"""Production configuration and outage checks without models or live services."""
import asyncio
from contextlib import nullcontext
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings, validate_runtime_settings
from app.monitoring.readiness import readiness_status


@pytest.fixture
def settings(tmp_path):
    return Settings(_env_file=None, environment="production", auth_mode="local", auth_cookie_secure=True,
                    allowed_origins=["https://localhost:8443"], allow_deterministic_fallback=False,
                    ollama_url="http://localhost:11434",
                    graphrag_service_token="test-credential-" * 3,
                    workspace_root=tmp_path, knowledge_root=tmp_path, capsules_root=tmp_path)


@pytest.mark.parametrize("change", [
    {"auth_mode": "disabled"}, {"auth_mode": "unsupported"}, {"auth_cookie_secure": False},
    {"demo_org_enabled": True}, {"integration_fail_closed": False},
    {"allow_deterministic_fallback": True}, {"graphrag_service_token": "short"},
    {"allowed_origins": ["http://localhost"]}, {"allowed_origins": []},
    {"allowed_origins": ["https://user:secret@localhost"]},
    {"allowed_origins": ["https://*"]}, {"allowed_origins": ["https://localhost:invalid"]},
])
def test_unsafe_production_settings_fail_before_startup(settings, change):
    with pytest.raises(RuntimeError):
        validate_runtime_settings(settings.model_copy(update=change))


def test_valid_profile_and_development_defaults(settings):
    validate_runtime_settings(settings)
    validate_runtime_settings(Settings(_env_file=None))


def probe(settings, *, down=None, installed=True, user=True):
    def handler(request):
        if down and down in str(request.url):
            return httpx.Response(503, json={"detail": "private exception must not be exposed"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": "qwen3:4b-instruct"}] if installed else []})
        return httpx.Response(200, json={"status": "ok"})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            factory = lambda: nullcontext(SimpleNamespace(execute=lambda _: None, scalar=lambda _: "user" if user else None))
            return await readiness_status(settings, client=client, session_factory=factory)
    return asyncio.run(run())


def test_healthy_dependencies_and_provisioned_identity_pass(settings):
    result = probe(settings)
    assert result["ready"] is True
    assert all(result["checks"].values())


@pytest.mark.parametrize("endpoint,check", [("3100", "graphrag"), ("8100", "controlplane"), ("8200", "diagnostics"), ("11434", "generation_model")])
def test_each_dependency_outage_holds_readiness_without_leaking_details(settings, endpoint, check):
    result = probe(settings, down=endpoint)
    assert result["ready"] is False and result["checks"][check] is False
    assert "private exception" not in str(result)
    assert settings.graphrag_service_token not in str(result)


def test_missing_model_and_empty_identity_store_do_not_report_ready(settings):
    assert probe(settings, installed=False)["ready"] is False
    assert probe(settings, user=False)["checks"]["identity_provisioned"] is False


def test_health_response_status_unavailable_is_not_treated_as_success(settings):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"status": "unavailable"}))) as client:
            return await readiness_status(settings, client=client, session_factory=lambda: nullcontext(SimpleNamespace(execute=lambda _: None, scalar=lambda _: "u")))
    result = asyncio.run(run())
    assert result["ready"] is False
    assert result["checks"]["graphrag"] is False


def test_model_registry_endpoint_must_match_active_provider(settings):
    result = probe(settings.model_copy(update={"ollama_url": "http://ollama:11434"}))
    assert result["checks"]["generation_model"] is False


@pytest.mark.parametrize("installed", [[None], {"unexpected": "mapping"}])
def test_malformed_model_service_payload_is_unready_instead_of_crashing(settings, installed):
    def handler(request):
        return httpx.Response(200, json={"models": installed} if request.url.path == "/api/tags" else {"status": "ok"})
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await readiness_status(settings, client=client, session_factory=lambda: nullcontext(SimpleNamespace(execute=lambda _: None, scalar=lambda _: "u")))
    assert asyncio.run(run())["checks"]["generation_model"] is False
