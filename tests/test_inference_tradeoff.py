"""Synthetic protocol/HTTP fixtures; these are not live inference measurements."""
import asyncio
import importlib.util
import json
import sys
from argparse import Namespace
from collections import Counter
from pathlib import Path

import httpx
import pytest

spec = importlib.util.spec_from_file_location(
    "inference_tradeoff", Path(__file__).resolve().parents[1] / "benchmarks/inference_tradeoff.py")
bench = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = bench
spec.loader.exec_module(bench)


def arguments(tmp_path, **overrides):
    values = dict(protocol="legacy-controlled", seed=102, repetitions=5,
                  providers=["vllm", "ollama"], context_lengths=[64], evidence_budgets=[128],
                  timeout=1, output=tmp_path,
                  vllm_url="http://127.0.0.1:8001/v1", ollama_url="http://127.0.0.1:11434",
                  vllm_model="fixture", ollama_model="fixture",
                  vllm_format="synthetic", ollama_format="synthetic")
    return Namespace(**(values | overrides))


@pytest.mark.parametrize("url", ["https://example.com", "http://8.8.8.8", "http://169.254.169.254",
                                  "http://0.0.0.0", "http://[::]", "http://user:secret@127.0.0.1",
                                  "http://127.0.0.1?key=secret", "ftp://127.0.0.1",
                                  "http://127.0.0.1#fragment", "http://[::ffff:8.8.8.8]"])
def test_external_or_ambiguous_endpoint_rejected(url):
    with pytest.raises(ValueError):
        bench.validate_endpoint(url)


@pytest.mark.parametrize("url", ["http://127.0.0.1:8001/v1", "http://10.0.0.2", "https://[::1]",
                                  "http://192.168.1.2", "http://[fd00::1]"])
def test_private_literal_endpoint_allowed(url):
    assert bench.validate_endpoint(url) == url


def test_localhost_is_pinned():
    assert bench.validate_endpoint("http://localhost:11434/") == "http://127.0.0.1:11434"


def test_randomized_blocks_reproducible_and_balanced(tmp_path):
    args = arguments(tmp_path)
    schedule = bench.controlled_schedule(args)
    assert schedule == bench.controlled_schedule(args)
    assert schedule != bench.controlled_schedule(arguments(tmp_path, seed=103))
    assert set(Counter(row[:3] for row in schedule).values()) == {5}
    for repetition in range(1, 6):
        assert len({row[:3] for row in schedule if row[3] == repetition}) == 4


def test_unknown_citations_reduce_precision():
    assert bench.quality("[E1] [E999]")[:2] == (0.5, 0.25)


@pytest.mark.parametrize("provider,body", [
    ("vllm", 'data: {"choices":[{"delta":{"content":"partial"}}]}\n\n'),
    ("vllm", 'data: [DONE]\n\n'),
    ("ollama", '{"response":"partial","done":false}\n'),
    ("ollama", '{"response":"","done":true}\n'),
])
def test_incomplete_and_empty_streams_fail(provider, body):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=body))) as client:
            with pytest.raises(ValueError, match="Incomplete or empty"):
                await getattr(bench, f"stream_{provider}")(client, "http://127.0.0.1", "fixture", "prompt")
    asyncio.run(run())


@pytest.mark.parametrize("provider,body", [
    ("vllm", 'data: {"choices":[{"delta":{"content":"answer"}}]}\n\ndata: [DONE]\n\n'),
    ("ollama", '{"response":"answer","done":true,"eval_count":1}\n'),
])
def test_complete_stream(provider, body):
    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(
                lambda request: httpx.Response(200, text=body))) as client:
            answer, metrics = await getattr(bench, f"stream_{provider}")(
                client, "http://127.0.0.1", "fixture", "prompt")
            assert answer == "answer"
            assert 0 <= metrics["ttft_ms"] <= metrics["total_ms"]
    asyncio.run(run())


def test_controlled_output_excludes_warmup_and_preserves_raw(tmp_path, monkeypatch):
    calls = []
    async def observe(provider, endpoint, model, prompt, experiment, target, concurrency, repetition, timeout):
        calls.append((provider, prompt, repetition))
        duration = 999 if len(calls) % 2 else 10
        return bench.Observation(experiment, provider, model, target, concurrency, repetition,
                                 1, duration, 10, 2, 200, 1, 1, 1, answer="synthetic [E1]")
    monkeypatch.setattr(bench, "observe", observe)
    assert asyncio.run(bench.main_async(arguments(tmp_path))) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    raw = json.loads((tmp_path / "raw.json").read_text())
    assert summary["runs"] == 40
    assert summary["warmup_runs"] == summary["measured_successful"] == 20
    assert all(group["samples"] == 5 and group["latency_p50_ms"] == 10
               for group in summary["groups"].values())
    assert summary["memory"] is None
    assert [row["phase"] for row in raw] == ["warmup", "warm"] * 20
    assert all(calls[i] == calls[i + 1] for i in range(0, 40, 2))
    assert (tmp_path / "raw.csv").is_file()


def test_failed_warmup_never_becomes_measurement(tmp_path, monkeypatch):
    async def observe(provider, endpoint, model, prompt, experiment, target, concurrency, repetition, timeout):
        return bench.Observation(experiment, provider, model, target, concurrency, repetition,
                                 None, 0, None, None, None, error="synthetic outage")
    monkeypatch.setattr(bench, "observe", observe)
    assert asyncio.run(bench.main_async(arguments(tmp_path))) == 2
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["runs"] == 20
    assert summary["measured_successful"] == 0
    assert summary["groups"] == {}


@pytest.mark.parametrize("overrides", [dict(repetitions=4), dict(context_lengths=[0]),
                                       dict(ollama_url="https://example.com"),
                                       dict(vllm_format=None), dict(timeout=0)])
def test_invalid_protocol_fails_before_requests(tmp_path, monkeypatch, overrides):
    async def forbidden(*args, **kwargs):
        pytest.fail("No request is permitted for an invalid protocol")
    monkeypatch.setattr(bench, "observe", forbidden)
    with pytest.raises(ValueError):
        asyncio.run(bench.main_async(arguments(tmp_path, **overrides)))


def test_observe_disables_proxies_and_redirects_and_records_provenance(monkeypatch):
    real_client = httpx.AsyncClient
    def client(**kwargs):
        assert kwargs["trust_env"] is False
        assert kwargs["follow_redirects"] is False
        return real_client(**kwargs, transport=httpx.MockTransport(lambda request: httpx.Response(
            200, text='{"response":"[E1]","done":true,"eval_count":2,"prompt_eval_count":9}\n')))
    monkeypatch.setattr(bench.httpx, "AsyncClient", client)
    result = asyncio.run(bench.observe("ollama", "http://127.0.0.1", "fixture", "prompt",
                                      "evidence_budget", 128, 1, 1, 1))
    assert result.error is None
    assert result.answer == "[E1]"
    assert len(result.prompt_sha256) == 64
    assert result.prompt_tokens == 9
    assert result.tokens_per_second == pytest.approx(2 / (result.total_ms / 1000))


def test_redirect_is_failure_without_following(monkeypatch):
    real_client = httpx.AsyncClient
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(307, headers={"location": "https://example.com"})
    monkeypatch.setattr(bench.httpx, "AsyncClient", lambda **kwargs: real_client(
        **kwargs, transport=httpx.MockTransport(respond)))
    result = asyncio.run(bench.observe("ollama", "http://127.0.0.1", "fixture", "prompt",
                                      "evidence_budget", 128, 1, 1, 1))
    assert result.error
    assert result.groundedness is result.ttft_ms is None
    assert len(requests) == 1


def test_legacy_smoke_protocol_still_writes_summary(tmp_path, monkeypatch):
    async def observe(provider, endpoint, model, prompt, experiment, target, concurrency, repetition, timeout):
        return bench.Observation(experiment, provider, model, target, concurrency, repetition,
                                 1, 10, 10, 2, 200, 1, 1, 1)
    monkeypatch.setattr(bench, "observe", observe)
    assert asyncio.run(bench.main_async(arguments(tmp_path, protocol="smoke"))) == 0
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["runs"] == summary["measured_successful"] == 28
    assert summary["warmup_runs"] == 0
    assert len(summary["prefix_cache"]) == 10
    assert set(summary["pareto_evidence_budgets"]) == {"vllm", "ollama"}
