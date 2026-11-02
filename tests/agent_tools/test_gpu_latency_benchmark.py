"""Benchmark protocol tests use synthetic SSE, never report GPU performance."""
from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


client = load_script("benchmark_vllm_latency")


@pytest.fixture
def server():
    state = {"requests": [], "mode": "ok"}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            state["requests"].append(("GET", self.path))
            if state["mode"] == "redirect":
                self.send_response(302)
                self.send_header("Location", "/redirected")
                self.end_headers()
                return
            self.send_response(200)
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "test"}]}).encode())

        def do_POST(self):
            payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state["requests"].append(("POST", payload))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            self.wfile.write(b'data: {"choices":[{"delta":{"content":"answer"}}]}\n\n')
            if state["mode"] != "missing_usage":
                self.wfile.write(b'data: {"choices":[],"usage":{"prompt_tokens":25,"completion_tokens":3}}\n\n')
            if state["mode"] != "truncated":
                self.wfile.write(b'data: [DONE]\n\n')

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_port}", state
    httpd.shutdown()
    httpd.server_close()
    thread.join(timeout=2)


def arguments(tmp_path, endpoint, *extra):
    return client.parse_args(["--endpoint", endpoint, "--model", "test", "--repeats", "2",
                              "--warmups", "1", "--output", str(tmp_path / "result.json"), *extra])


def test_stream_protocol_warmups_usage_and_secret_redaction(server, tmp_path, monkeypatch):
    endpoint, state = server
    monkeypatch.setenv("SOVEREIGN_VLLM_API_KEY", "secret-api-value")
    args = arguments(tmp_path, endpoint)
    assert client.run(args) == 0
    saved = args.output.read_text()
    assert "secret-api-value" not in saved
    report = json.loads(saved)
    assert len(report["warmup_samples"]) == 1
    assert report["summary"]["successful"] == 2
    assert report["summary"]["ttft_seconds"]["count"] == 2
    assert all(sample["completion_tokens"] == 3 for sample in report["samples"])
    assert all(sample["decode_tokens_per_second_estimate"] is None for sample in report["samples"])
    posts = [payload for method, payload in state["requests"] if method == "POST"]
    assert len(posts) == 3
    assert posts[0]["chat_template_kwargs"] == {"enable_thinking": False}


def test_missing_usage_is_unknown_and_truncated_streams_fail(server, tmp_path):
    endpoint, state = server
    state["mode"] = "missing_usage"
    args = arguments(tmp_path, endpoint)
    assert client.run(args) == 0
    report = json.loads(args.output.read_text())
    assert report["samples"][0]["completion_tokens"] is None
    state["mode"] = "truncated"
    assert client.run(args) == 1
    report = json.loads(args.output.read_text())
    assert report["summary"]["failed"] == 2
    assert report["summary"]["ttft_seconds"]["p50"] is None


def test_preflight_does_not_follow_redirects_or_write_measurements(server, tmp_path):
    endpoint, state = server
    state["mode"] = "redirect"
    args = arguments(tmp_path, endpoint)
    assert client.run(args) == 2
    assert state["requests"] == [("GET", "/v1/models")]
    assert not args.output.exists()


@pytest.mark.parametrize("endpoint", ["http://gpu.example:8001", "https://gpu.example/v1",
                                      "https://user:secret@gpu.example", "https://gpu.example?key=secret"])
def test_remote_endpoints_require_explicit_https_without_url_credentials(endpoint):
    with pytest.raises(ValueError):
        client.normalize_endpoint(endpoint)
    if endpoint == "https://gpu.example/v1":
        assert client.normalize_endpoint(endpoint, True) == endpoint
    else:
        with pytest.raises(ValueError):
            client.normalize_endpoint(endpoint, True)


def test_summary_excludes_failures_and_reports_sample_percentiles():
    summary = client.summarize([
        {"success": True, "ttft_seconds": 1, "total_seconds": 3},
        {"success": True, "ttft_seconds": 3, "total_seconds": 5},
        {"success": False, "ttft_seconds": 100}])
    assert summary["ttft_seconds"] == {"count": 2, "p50": 2, "p95": 2.9}
    assert summary["failed"] == 1


def test_first_visible_chunk_and_last_content_define_timing(monkeypatch):
    ticks = iter([10, 11, 12, 14, 15, 16, 17])
    monkeypatch.setattr(client.time, "perf_counter", lambda: next(ticks))
    body = (b'data: {"choices":[{"delta":{"reasoning_content":"hidden"}}]}\n'
            b'data: {"choices":[{"delta":{"content":"first"}}]}\n'
            b'data: {"choices":[{"delta":{"content":"last"}}]}\n'
            b'data: {"usage":{"completion_tokens":5,"prompt_tokens":9}}\n'
            b'data: [DONE]\n')
    opener = SimpleNamespace(open=lambda *args, **kwargs: io.BytesIO(body))
    sample = client.stream_sample(opener, "http://localhost/v1", "test", "local", 128, 120)
    assert sample["ttft_seconds"] == 2
    assert sample["total_seconds"] == 7
    assert sample["decode_tokens_per_second_estimate"] == 2


@pytest.mark.parametrize("value", ["0", "-1"])
def test_repeats_must_be_positive(value):
    with pytest.raises(SystemExit):
        client.parse_args(["--model", "test", "--repeats", value])


@pytest.mark.parametrize("provider", ["vllm", "ollama"])
@pytest.mark.parametrize("outcome", ["completed", "fallback"])
def test_workflow_selects_models_and_rejects_fallback_without_saved_config_changes(tmp_path, monkeypatch, provider, outcome):
    workflow = load_script("benchmark_workflow_latency")
    from app.core.config import Settings
    from app.llm.factory import configured_local_provider
    config = tmp_path / "models.yaml"
    config.write_text("models:\n  general:\n    model_tag: saved\n    endpoint: http://localhost:11434\n")
    original = config.read_text()
    settings = Settings(models_config=config, workspace_root=tmp_path / "workspace")
    monkeypatch.setattr(workflow, "get_settings", lambda: settings)
    requests, selections = [], []

    def request(url, data=None, **kwargs):
        requests.append((url, data))
        if url.endswith("/models"):
            return {"data": [{"id": "one"}, {"id": "two"}]}
        if url.endswith("/api/tags"):
            return {"models": [{"name": "one"}, {"name": "two"}]}
        return {}

    async def execute(*args):
        isolated = workflow.tasks.get_settings()
        assert not isolated.allow_deterministic_fallback and not isolated.cache_enabled
        if provider == "vllm":
            selection = configured_local_provider(isolated)
            selections.append(selection.model)
        else:
            selections.append(workflow.yaml.safe_load(isolated.models_config.read_text())["models"]["general"]["model_tag"])
        return SimpleNamespace(
            runtime_metrics={"timing": {"llm_calls": [{"outcome": outcome}],
                                       "llm_call_count": 1, "task_handler_seconds": 0}},
            status=SimpleNamespace(value="completed"), governance={"decision": "allow"},
            structured_plan={}, tool_records=[], final_response="test")

    monkeypatch.setattr(workflow, "request_json", request)
    monkeypatch.setattr(workflow.tasks, "_execute_task", execute)
    monkeypatch.setattr(workflow, "check", lambda *args: (True, "verified"))
    args = SimpleNamespace(provider=provider, endpoint="http://127.0.0.1:8001", models=["one", "two"],
                           cases=["numeric_chain"], append=False, retry_failed=False,
                           keep_going=False, output=tmp_path / "workflow.json")
    assert asyncio.run(workflow.main(args)) == (0 if outcome == "completed" else 1)
    assert selections == ["one", "two"]
    assert config.read_text() == original and settings.llm_provider == "ollama"
    generate_calls = [url for url, data in requests if url.endswith("/api/generate")]
    assert len(generate_calls) == (4 if provider == "ollama" else 0)


def test_workflow_still_rejects_public_endpoints():
    workflow = load_script("benchmark_workflow_latency")
    from app.core.config import Settings
    with pytest.raises(ValueError, match="sovereignty policy"):
        workflow.benchmark_settings(SimpleNamespace(provider="vllm", endpoint="https://gpu.example"), Settings())


def test_notebook_contains_current_client_and_all_cells_compile():
    notebook = json.loads((ROOT / "notebooks" / "vllm_gpu_latency.ipynb").read_text())
    assert notebook["metadata"]["accelerator"] == "GPU"
    embedded = {}
    for index, item in enumerate(notebook["cells"]):
        if item["cell_type"] == "code":
            source = "".join(item["source"])
            compile(source, f"cell-{index}", "exec")
            if source.startswith("client_source = "):
                # Read the literal without executing the notebook's file write.
                import ast
                embedded["client"] = ast.literal_eval(ast.parse(source).body[0].value)
            assert item["outputs"] == [] and item["execution_count"] is None
    assert embedded["client"] == (ROOT / "scripts" / "benchmark_vllm_latency.py").read_text()
