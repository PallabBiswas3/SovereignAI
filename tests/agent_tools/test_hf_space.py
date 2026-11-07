"""Hugging Face protocol/setup checks use mocks; no account or GPU is accessed."""
import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.core.config import Settings, validate_runtime_settings
from app.llm.base import ModelGenerationCancelled
from app.llm.factory import configured_local_provider
from app.llm.hf_space_provider import HuggingFaceSpaceProvider, validate_space_url
from app.monitoring.timing import capture_task_timing
from app.resources.lifecycle import ModelLifecycleManager
from app.resources.scheduler import ResourceScheduler

ROOT = Path(__file__).resolve().parents[2]
TOKEN = "hf_synthetic_test_credential"
URL = "https://test-owner-space.hf.space"
MODEL = "Qwen/Qwen3-4B-Instruct-2507"


def provider(client):
    return HuggingFaceSpaceProvider(URL, enabled=True, api_key=TOKEN, client=client,
                                    scheduler=ResourceScheduler(), lifecycle=ModelLifecycleManager())


def event(kind, packet):
    return "event: " + kind + "\ndata: " + json.dumps([packet]) + "\n\n"


def setup_script():
    spec = importlib.util.spec_from_file_location("deploy_hf_space", ROOT / "scripts/deploy_hf_space.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text_json_status_auth_and_call_metrics():
    async def run():
        submitted = []

        def handler(request):
            assert request.headers["authorization"] == "Bearer " + TOKEN
            if request.method == "POST":
                submitted.append(json.loads(request.content)["data"])
                return httpx.Response(200, json={"event_id": "abc"})
            if "/runtime_info/" in request.url.path:
                return httpx.Response(200, text=event("complete", {"model": MODEL, "protocol": "sovereign-hf-v1"}))
            text = '{"answer":42}' if submitted[-1][-1] else "hello"
            return httpx.Response(200, text=event("generating", {"model": MODEL, "text": text[:2], "done": False})
                                  + event("complete", {"model": MODEL, "text": text, "done": True,
                                                        "usage": {"prompt_tokens": 9, "completion_tokens": 3}}))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            remote = provider(client)
            assert (await remote.health_check())["available"]
            with capture_task_timing() as timing:
                text = await remote.generate("synthetic", MODEL)
                structured = await remote.generate_json("synthetic", MODEL, {"type": "object"})
            assert text.text == "hello" and text.provider == "hf_space"
            assert structured.data == {"answer": 42} and not structured.fallback
            assert text.runtime_stats["warm_status"] == "unknown"
            assert text.runtime_stats["token_count"] == 3
            assert timing.snapshot()["llm_call_count"] == 2
            assert TOKEN not in json.dumps(timing.snapshot())
            assert json.loads(submitted[-1][-1]) == {"type": "object"}
    asyncio.run(run())


@pytest.mark.parametrize("mode", ["redirect", "bad_id", "quota", "truncated", "wrong_model", "invalid_json"])
def test_failures_leave_no_fallback_and_never_forward_tokens(mode):
    async def run():
        requests = []

        def handler(request):
            requests.append(request)
            if mode == "redirect":
                return httpx.Response(307, headers={"Location": "https://other.example/steal"})
            if request.method == "POST":
                return httpx.Response(200, json={"event_id": "../steal" if mode == "bad_id" else "abc"})
            if mode == "quota":
                return httpx.Response(200, text="event: error\ndata: " + json.dumps(TOKEN) + "\n\n")
            if mode == "truncated":
                return httpx.Response(200, text=event("generating", {"model": MODEL, "text": "partial"}))
            if mode == "invalid_json":
                return httpx.Response(200, text="event: complete\ndata: broken\n\n")
            return httpx.Response(200, text=event("complete", {"model": "unexpected", "text": "bad", "done": True}))

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(RuntimeError) as exc:
                await provider(client).generate("synthetic", MODEL)
            assert TOKEN not in str(exc.value)
            assert all(request.url.host.endswith(".hf.space") for request in requests)
    asyncio.run(run())


def test_precancellation_and_health_deadlines():
    async def run():
        requests = []

        def handler(request):
            requests.append(request)
            raise httpx.ReadTimeout(TOKEN, request=request)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            remote = provider(client)
            cancelled = asyncio.Event()
            cancelled.set()
            with pytest.raises(ModelGenerationCancelled):
                await anext(remote.stream("synthetic", MODEL, cancellation_event=cancelled))
            assert not requests
            health = await remote.health_check()
            assert health["available"] is False and TOKEN not in json.dumps(health)
    asyncio.run(run())


@pytest.mark.parametrize("url", ["http://test.hf.space", "https://huggingface.co/spaces/owner/name",
                                "https://evil.example", "https://test.hf.space/v1", "https://test.hf.space:9000"])
def test_only_space_root_urls_are_accepted(url):
    with pytest.raises(ValueError):
        validate_space_url(url)


def test_factory_requires_explicit_optin_and_hides_token():
    settings = Settings(_env_file=None, llm_provider="hf_space", remote_inference_enabled=True,
                        hf_space_url=URL, hf_token=TOKEN, hf_space_model=MODEL)
    validate_runtime_settings(settings)
    selected = configured_local_provider(settings)
    assert selected.provider_name == "hf_space" and selected.endpoint == URL
    assert TOKEN not in settings.model_dump_json() and not selected.provider.allow_fallback
    with pytest.raises(ValueError):
        configured_local_provider(settings.model_copy(update={"remote_inference_enabled": False}))


def test_setup_preserves_unrelated_settings_and_keeps_token_out_of_env(tmp_path):
    module = setup_script()
    original = '# preserve\nOTHER="keep"\nSOVEREIGN_LLM_PROVIDER=ollama\nSOVEREIGN_LLM_PROVIDER=vllm\n'
    (tmp_path / ".env").write_text(original)
    target = module.save_connection(URL, TOKEN, root=tmp_path)
    value = target.read_text()
    assert '# preserve\nOTHER="keep"\n' in value
    assert value.count("SOVEREIGN_LLM_PROVIDER=") == 1 and 'SOVEREIGN_LLM_PROVIDER="hf_space"' in value
    assert TOKEN not in value and (tmp_path / "deploy/secrets/hf_token").read_text().strip() == TOKEN
    target = module.save_connection(URL, TOKEN, docker=True, root=tmp_path)
    assert 'SOVEREIGN_HF_TOKEN_FILE="/run/secrets/hf_token"' in target.read_text()


@pytest.mark.parametrize("hardware", ["cpu-basic", "t4-small"])
def test_setup_never_activates_cpu_or_paid_hardware(hardware):
    module = setup_script()
    api = SimpleNamespace(get_space_runtime=lambda _: SimpleNamespace(stage="RUNNING", hardware=hardware))
    with pytest.raises(RuntimeError):
        module.wait_for_space(api, "owner/name")


def test_setup_refuses_wrong_account_before_creating_or_uploading(monkeypatch, tmp_path):
    module = setup_script()
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    monkeypatch.setattr(module, "HfApi", lambda **kw: SimpleNamespace(whoami=lambda: {"name": "someone-else"}))
    assert module.main(["--username", "pallabbiswas891"]) == 1


def test_server_diagnostic_redacts_tokens_urls_emails_and_control_characters():
    module = setup_script()
    message = ("Cannot create Space; token " + TOKEN + " extra hf_other_private_token "
               "Bearer unrelated_secret https://example.org/?secret=hidden person@example.org\n\x1b[31m")
    error = RuntimeError("Raw exception is intentionally not shown: " + TOKEN)
    error.response = SimpleNamespace(status_code=403, json=lambda: {"error": message})
    safe = module.safe_hub_message(error, TOKEN)
    assert "Cannot create Space" in safe
    assert all(secret not in safe for secret in (TOKEN, "hf_other_private_token", "unrelated_secret",
                                                 "example.org", "person@example.org", "\n", "\x1b"))


def test_creation_denial_reports_http_and_server_reason_without_false_resume(monkeypatch, capsys):
    module = setup_script()
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    calls = []
    error = RuntimeError("Raw private exception " + TOKEN)
    error.response = SimpleNamespace(status_code=403, json=lambda: {
        "error": "ZeroGPU hosting limit reached. " + TOKEN})

    def create(**kwargs):
        calls.append(kwargs)
        raise error

    api = SimpleNamespace(whoami=lambda: {"name": "pallabbiswas891", "auth": {
        "accessToken": {"role": "write"}}}, create_repo=create)
    monkeypatch.setattr(module, "HfApi", lambda **kwargs: api)
    assert module.main(["--username", "pallabbiswas891"]) == 1
    printed = capsys.readouterr().out
    assert "Authenticated as pallabbiswas891. Token permission: write." in printed
    assert "HTTP 403" in printed and "private ZeroGPU Space creation" in printed
    assert "ZeroGPU hosting limit reached" in printed
    assert "To resume:" not in printed and "creation was not confirmed" in printed
    assert TOKEN not in printed and "Raw private exception" not in printed
    assert len(calls) == 1 and calls[0]["private"] is True
    assert calls[0]["space_hardware"].value == "zero-a10g"


def test_read_only_token_is_rejected_before_creation_with_actionable_guidance(monkeypatch, capsys):
    module = setup_script()
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    api = SimpleNamespace(whoami=lambda: {"name": "pallabbiswas891", "auth": {
        "accessToken": {"role": "read"}}})
    monkeypatch.setattr(module, "HfApi", lambda **kwargs: api)
    assert module.main(["--username", "pallabbiswas891"]) == 1
    printed = capsys.readouterr().out
    assert "This token is read-only" in printed and "Create a write token" in printed
    assert "To resume:" not in printed


def test_free_hosting_402_is_an_eligibility_block_without_token_retry_or_paid_fallback(monkeypatch, capsys):
    module = setup_script()
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    calls = []
    error = RuntimeError(TOKEN)
    error.response = SimpleNamespace(status_code=402, json=lambda: {
        "error": "Your account is not eligible to host ZeroGPU Spaces for free. Subscribe to PRO or request a community grant."})

    def create(**kwargs):
        calls.append(kwargs)
        raise error

    api = SimpleNamespace(whoami=lambda: {"name": "pallabbiswas891", "auth": {
        "accessToken": {"role": "fineGrained"}}}, create_repo=create)
    monkeypatch.setattr(module, "HfApi", lambda **kwargs: api)
    assert module.main(["--username", "pallabbiswas891"]) == 1
    printed = capsys.readouterr().out
    assert "HTTP 402" in printed and "Changing or recreating tokens will not resolve" in printed
    assert "community GPU grant" in printed and "requires Hugging Face approval" in printed
    assert "Retry only after" in printed and "To resume:" not in printed
    assert "Check Space build/runtime logs" not in printed
    assert TOKEN not in printed
    assert len(calls) == 1 and calls[0]["private"] is True
    assert calls[0]["space_hardware"].value == "zero-a10g"


def test_failed_upload_resume_includes_upload_instead_of_waiting_on_empty_space(monkeypatch, capsys):
    module = setup_script()
    monkeypatch.setenv("HF_TOKEN", TOKEN)

    def failed_upload(**kwargs):
        raise RuntimeError(TOKEN)

    api = SimpleNamespace(whoami=lambda: {"name": "pallabbiswas891"},
                          create_repo=lambda **kwargs: None, upload_folder=failed_upload)
    monkeypatch.setattr(module, "HfApi", lambda **kwargs: api)
    assert module.main(["--username", "pallabbiswas891", "--docker"]) == 1
    printed = capsys.readouterr().out
    assert "inference file upload" in printed
    assert "--existing-space pallabbiswas891/sovereign-qwen3-gpu --upload-existing" in printed
    assert "--docker" in printed and TOKEN not in printed


def test_resume_checks_existing_space_before_upload_or_wait(monkeypatch, capsys):
    module = setup_script()
    monkeypatch.setenv("HF_TOKEN", TOKEN)
    error = RuntimeError("Not found")
    error.response = SimpleNamespace(status_code=404, json=lambda: {"error": "Space not found"})

    def missing_space(*args):
        raise error

    api = SimpleNamespace(whoami=lambda: {"name": "pallabbiswas891"}, space_info=missing_space)
    monkeypatch.setattr(module, "HfApi", lambda **kwargs: api)
    assert module.main(["--existing-space", "pallabbiswas891/sovereign-qwen3-gpu"]) == 1
    printed = capsys.readouterr().out
    assert "existing Space lookup" in printed and "HTTP 404" in printed
    assert "To resume:" not in printed


@pytest.mark.parametrize("value", ["", "a-token-name", '"hf_private_value"', "hf_private value", "hf_"])
def test_bad_token_entry_is_explained_without_echoing_input_or_contacting_hub(monkeypatch, capsys, value):
    module = setup_script()
    monkeypatch.delenv("HF_TOKEN", raising=False)
    monkeypatch.setattr(module, "get_token", lambda: None)
    monkeypatch.setattr(module.sys, "stdin", SimpleNamespace(isatty=lambda: True))
    monkeypatch.setattr(module, "getpass", lambda _: value)

    def forbidden_api(**kwargs):
        pytest.fail("Invalid input must be rejected before contacting Hugging Face")

    monkeypatch.setattr(module, "HfApi", forbidden_api)
    with pytest.raises(SystemExit) as error:
        module.main(["--username", "pallabbiswas891"])
    assert error.value.code == 2
    captured = capsys.readouterr()
    assert "Nothing will appear" in captured.out
    assert ("No token received" if not value else "Expected the full access token") in captured.err
    assert "hf_private" not in captured.out + captured.err


def test_host_sources_compile_and_upload_directory_contains_only_inference_files():
    paths = {path.name for path in (ROOT / "deploy/huggingface").iterdir() if path.is_file()}
    assert paths == {"app.py", "README.md", "requirements.txt"}
    compile((ROOT / "deploy/huggingface/app.py").read_text(encoding="utf-8"), "hf_app", "exec")
