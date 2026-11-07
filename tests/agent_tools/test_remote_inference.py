"""Remote inference contracts use mock transports, never real cloud resources."""
from __future__ import annotations

import asyncio
from contextlib import nullcontext
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings, validate_runtime_settings
from app.llm.factory import configured_local_provider
from app.llm.ollama_provider import OllamaProvider
from app.llm.remote_vllm_provider import RemoteVLLMProvider, validate_remote_endpoint
from app.llm.vllm_provider import VLLMProvider
from app.llm.base import ModelGenerationCancelled
from app.monitoring.network import LocalNetworkPolicy
from app.monitoring.readiness import readiness_status
from app.monitoring.timing import capture_task_timing
from app.resources.lifecycle import ModelLifecycleManager
from app.resources.scheduler import ResourceScheduler


ROOT = Path(__file__).resolve().parents[2]
KEY = "synthetic-test-key-" * 3
ENDPOINT = "https://gpu.example/v1"


def remote_settings(**changes):
    return Settings(_env_file=None, **{
        "llm_provider": "remote_vllm", "remote_inference_enabled": True,
        "remote_vllm_url": ENDPOINT, "remote_vllm_model": "test",
        "remote_vllm_api_key": KEY, **changes})


def remote(client=None, **kwargs):
    return RemoteVLLMProvider(ENDPOINT, enabled=True, api_key=KEY, client=client,
                              scheduler=ResourceScheduler(), lifecycle=ModelLifecycleManager(), **kwargs)


@pytest.mark.parametrize("endpoint", [
    "http://gpu.example", "https://localhost", "https://127.0.0.1", "https://10.0.0.1",
    "https://169.254.169.254", "https://[::1]", "https://ollama", "https://gpu.local",
    "https://gpu.example:bad", "https://gpu.example:99999", "https://gpu.example?key=secret",
    "https://gpu.example#secret", "https://user:secret@gpu.example", "https://gpu.example/has space",
    "https://gpu.example\\other", "https://127%2e0%2e0%2e1",
    "https://127.1", "https://2130706433", "https://127.0.0.01",
])
def test_remote_endpoint_validation_never_echoes_invalid_url(endpoint):
    with pytest.raises(ValueError) as error:
        validate_remote_endpoint(endpoint)
    assert endpoint not in str(error.value) and "secret" not in str(error.value)


def test_local_policy_still_rejects_remote_and_defaults_stay_local():
    settings = Settings(_env_file=None)
    assert settings.llm_provider == "ollama" and not settings.remote_inference_enabled
    assert configured_local_provider(settings).provider_name == "ollama"
    for provider in (OllamaProvider, VLLMProvider):
        with pytest.raises(ValueError, match="sovereignty policy"):
            provider(ENDPOINT)
    with pytest.raises(ValueError):
        LocalNetworkPolicy.require_local(ENDPOINT)


@pytest.mark.parametrize("changes", [
    {"remote_inference_enabled": False}, {"remote_vllm_api_key": ""},
    {"remote_vllm_api_key": "local"}, {"remote_vllm_api_key": "has\nnewline"},
    {"remote_vllm_model": ""}, {"remote_vllm_url": "http://gpu.example"},
])
def test_invalid_remote_configuration_fails_before_startup_and_provider_selection(changes):
    settings = remote_settings(**changes)
    with pytest.raises(ValueError):
        validate_runtime_settings(settings)
    with pytest.raises(ValueError):
        configured_local_provider(settings)


def test_remote_factory_selects_exact_model_redacts_secret_and_disables_fallback():
    settings = remote_settings(remote_vllm_url="https://gpu.example/")
    validate_runtime_settings(settings)
    selection = configured_local_provider(settings)
    assert isinstance(selection.provider, RemoteVLLMProvider)
    assert selection.endpoint == ENDPOINT and selection.model == "test"
    assert selection.provider_name == "remote_vllm" and not selection.provider.allow_fallback
    assert KEY not in repr(settings) and KEY not in settings.model_dump_json()
    with pytest.raises(ValueError, match="disabled"):
        RemoteVLLMProvider(ENDPOINT, api_key=KEY)


def test_streaming_structured_output_auth_and_metrics_share_remote_provider():
    async def run():
        requests = []

        def handler(request):
            requests.append(request)
            assert request.headers["authorization"] == "Bearer " + KEY
            assert str(request.url) == ENDPOINT + "/chat/completions"
            payload = json.loads(request.content)
            assert payload["model"] == "test"
            if payload.get("stream"):
                return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"hello"}}]}\n\n'
                                      'data: {"usage":{"prompt_tokens":9,"completion_tokens":2}}\n\n'
                                      'data: [DONE]\n\n')
            return httpx.Response(200, json={"choices": [{"message": {"content": '{"answer":42}'}}]})

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = remote(client)
            with capture_task_timing() as timing:
                result = await provider.generate("synthetic prompt", "test")
                structured = await provider.generate_json("synthetic prompt", "test", {"type": "object"})
            assert result.text == "hello" and result.provider == "remote_vllm" and not result.fallback
            assert result.runtime_stats["time_to_first_token_seconds"] is not None
            assert result.runtime_stats["warm_status"] == "unknown"
            assert structured.data == {"answer": 42} and structured.provider == "remote_vllm"
            assert timing.snapshot()["llm_call_count"] == 2
            assert KEY not in json.dumps(timing.snapshot())
            assert len(requests) == 2  # No extra /models round trip before inference.
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["redirect", "unauthorized", "timeout", "truncated", "malformed"])
def test_remote_failures_never_fabricate_a_success_or_forward_credentials(mode):
    async def run():
        requests = []

        def handler(request):
            requests.append(request)
            if mode == "timeout":
                raise httpx.ReadTimeout("sensitive message " + KEY, request=request)
            if mode == "redirect":
                return httpx.Response(307, headers={"Location": "https://other.example/collect"})
            if mode == "unauthorized":
                return httpx.Response(401, text=KEY)
            if mode == "malformed":
                return httpx.Response(200, text="data: {not-json}\n\n")
            return httpx.Response(200, text='data: {"choices":[{"delta":{"content":"partial"}}]}\n\n')

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            provider = remote(client, allow_fallback=True)
            with pytest.raises(RuntimeError, match="Remote inference unavailable") as error:
                await provider.generate("synthetic", "test")
            assert KEY not in str(error.value) and "sensitive" not in str(error.value)
            assert len(requests) == 1 and str(requests[0].url).startswith(ENDPOINT)
            assert KEY not in provider.lifecycle.get("test").last_error
    asyncio.run(run())


def test_remote_pre_cancel_sends_nothing_and_owned_client_ignores_proxies():
    async def run():
        provider = remote()
        client = provider._new_client()
        try:
            assert client.follow_redirects is False and client.trust_env is False
            event = asyncio.Event()
            event.set()
            with pytest.raises(ModelGenerationCancelled):
                await anext(provider.stream("synthetic", "test", cancellation_event=event))
        finally:
            await client.aclose()
    asyncio.run(run())


@pytest.mark.parametrize("status,models,ready", [(200, [{"id": "test"}], True),
                                               (200, [{"id": "other"}], False), (503, [], False)])
def test_readiness_checks_authenticated_remote_model_without_generating(status, models, ready, tmp_path):
    async def run():
        def handler(request):
            if request.url.host == "gpu.example":
                assert request.url.path == "/v1/models" and request.method == "GET"
                assert request.headers["authorization"] == "Bearer " + KEY
                return httpx.Response(status, json={"data": models})
            return httpx.Response(200, json={"status": "ok"})
        settings = remote_settings(workspace_root=tmp_path, knowledge_root=tmp_path, capsules_root=tmp_path,
                                   graphrag_service_token="synthetic-" * 4)
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            factory = lambda: nullcontext(SimpleNamespace(execute=lambda _: None, scalar=lambda _: "user"))
            result = await readiness_status(settings, client=client, session_factory=factory)
        assert result["checks"]["generation_model"] is ready
        assert KEY not in str(result)
    asyncio.run(run())


def script(name, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_host_launch_binds_loopback_and_passes_key_only_via_environment(monkeypatch):
    host = script("serve_remote_vllm", monkeypatch)
    monkeypatch.setattr(host.shutil, "which", lambda _: "/env/bin/vllm")
    args = SimpleNamespace(model="test", context_length=8192, gpu_memory_utilization=.85, revision="sha")
    command, environment = host.launch_config(args, {"SOVEREIGN_REMOTE_VLLM_API_KEY": KEY})
    assert KEY not in " ".join(command)
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert environment["VLLM_API_KEY"] == KEY and environment["VLLM_NO_USAGE_STATS"] == "1"
    with pytest.raises(ValueError):
        host.launch_config(args, {})


def test_comparison_interleaves_equal_prompts_records_failures_and_redacts(monkeypatch, tmp_path):
    benchmark = script("benchmark_remote_inference", monkeypatch)
    monkeypatch.setenv("SOVEREIGN_REMOTE_VLLM_API_KEY", KEY)
    requests = []

    def opened(request, **kwargs):
        requests.append(request)
        if request.data is None:
            return io.BytesIO(json.dumps({"models": [{"name": "test"}], "data": [{"id": "test"}]}).encode())
        if request.full_url.endswith("/api/chat"):
            return io.BytesIO(b'{"message":{"content":"private generated text"},"done":false}\n'
                              b'{"done":true,"eval_count":2,"prompt_eval_count":9}\n')
        return io.BytesIO(b'data: {"choices":[{"delta":{"content":"private generated text"}}]}\n\n'
                          b'data: {"usage":{"completion_tokens":2}}\n\ndata: [DONE]\n\n')

    monkeypatch.setattr(benchmark.urllib.request, "build_opener", lambda *a: SimpleNamespace(open=opened))
    monkeypatch.setattr(benchmark.protocol, "hardware_info", lambda: {})
    args = benchmark.parse_args(["--local-model", "test", "--remote-model", "test",
                                "--remote-endpoint", ENDPOINT, "--warmups", "1", "--repeats", "2",
                                "--output", str(tmp_path / "comparison.json")])
    assert benchmark.run(args) == 0
    saved = args.output.read_text()
    assert KEY not in saved and "private generated text" not in saved
    report = json.loads(saved)
    for item in report["runs"].values():
        assert len(item["warmup_samples"]) == 1 and item["summary"]["successful"] == 2
    posts = [request for request in requests if request.data]
    assert len(posts) == 6
    assert all(json.loads(request.data)["messages"][0]["content"] == benchmark.protocol.PROMPT for request in posts)
    for request in posts:
        if request.full_url.endswith("/api/chat"):
            assert "authorization" not in {key.lower() for key in request.headers}
    report["runs"]["remote"]["summary"] = benchmark.protocol.summarize([{"success": False}])
    assert benchmark.comparison(report)["total_seconds_median_speedup"] is None
    assert not benchmark.comparison(report)["all_requests_succeeded"]


def test_ollama_benchmark_rejects_truncated_and_error_streams(monkeypatch):
    benchmark = script("benchmark_remote_inference", monkeypatch)
    for body in (b'{"message":{"content":"partial"}}\n', b'{"error":"bad model"}\n'):
        opener = SimpleNamespace(open=lambda *a, **k: io.BytesIO(body))
        with pytest.raises(ValueError):
            benchmark.ollama_sample(opener, "http://localhost:11434", "test", 128, 30)


@pytest.mark.parametrize("enabled", [True, False])
def test_monitor_reports_remote_exception_without_claiming_airgap_or_exposing_key(monkeypatch, enabled):
    from app.api import monitoring

    class Query:
        def filter_by(self, **kwargs): return self
        def order_by(self, *args): return self
        def limit(self, *args): return self
        def all(self): return []

    settings = remote_settings(remote_inference_enabled=enabled)
    monkeypatch.setattr(monitoring, "get_settings", lambda: settings)
    monkeypatch.setattr(monitoring, "NetworkMonitor", lambda db: SimpleNamespace(
        counts=lambda: {"external_attempts": 0, "allowed_local_requests": 0}))

    async def services(*args, **kwargs):
        assert kwargs["model_health"] is not None
        return []

    monkeypatch.setattr(monitoring, "local_service_status", services)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, json={"data": [{"id": "test"}]}))) as client:
            original = monitoring.configured_local_provider

            def selection(value):
                result = original(value)
                result.provider.client = client
                return result

            monkeypatch.setattr(monitoring, "configured_local_provider", selection)
            result = await monitoring.network_status(principal=None, db=SimpleNamespace(query=lambda *a: Query()))
        assert result["sovereignty_status"] == ("remote-inference-enabled" if enabled else "violation")
        assert result["configuration"]["passed"] is False
        assert result["external_ai_apis"] == 1 and result["remote_inference_enabled"] is enabled
        assert KEY not in json.dumps(result)
    asyncio.run(run())


def test_settings_load_remote_credential_file_without_changing_defaults(monkeypatch, tmp_path):
    from app.core import config
    key_file = tmp_path / "key"
    key_file.write_text(KEY + "\n")
    settings = remote_settings(remote_vllm_api_key="", remote_vllm_api_key_file=key_file,
                               workspace_root=tmp_path, knowledge_root=tmp_path,
                               organizations_root=tmp_path, organization_reports_root=tmp_path)
    monkeypatch.setattr(config, "Settings", lambda: settings)
    config.get_settings.cache_clear()
    try:
        loaded = config.get_settings()
        assert loaded.remote_vllm_api_key.get_secret_value() == KEY
        assert KEY not in repr(loaded)
    finally:
        config.get_settings.cache_clear()
