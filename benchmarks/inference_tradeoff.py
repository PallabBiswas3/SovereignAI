#!/usr/bin/env python3
"""Reproducible local-inference experiments for evidence-budget research.

The harness writes raw observations; it never ships prompts to an external host
and never substitutes synthetic numbers when a runtime is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import ipaddress
import random
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import httpx


@dataclass
class Observation:
    experiment: str
    provider: str
    model: str
    target_tokens: int
    concurrency: int
    repetition: int
    ttft_ms: float | None
    total_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None
    tokens_per_second: float | None
    citation_precision: float | None = None
    citation_recall: float | None = None
    groundedness: float | None = None
    error: str | None = None
    phase: str = "smoke"
    answer: str | None = None
    prompt_sha256: str | None = None


def validate_endpoint(endpoint: str) -> str:
    """Allow literal loopback/RFC1918/ULA addresses; never resolve arbitrary DNS."""
    parsed = urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("Runtime endpoint must be a private HTTP(S) URL without credentials/query/fragment")
    host = parsed.hostname
    if host == "localhost":
        host = "127.0.0.1"
    try:
        address = ipaddress.ip_address(host)
    except ValueError as exc:
        raise ValueError("Use a literal private runtime IP or localhost") from exc
    networks = ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16", "fc00::/7")
    if not (address.is_loopback or any(address in ipaddress.ip_network(n) for n in networks)):
        raise ValueError("Public, link-local, and unspecified runtime addresses are forbidden")
    netloc = f"[{host}]" if address.version == 6 else host
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return urlunsplit((parsed.scheme, netloc, parsed.path.rstrip("/"), "", ""))


def controlled_schedule(args: argparse.Namespace) -> list[tuple[str, int, str, int]]:
    """Randomized blocks: each condition occurs once per warm repetition."""
    conditions = [(provider, target, experiment)
                  for provider in args.providers
                  for experiment, targets in (("context_length", args.context_lengths),
                                              ("evidence_budget", args.evidence_budgets))
                  for target in targets]
    rng = random.Random(args.seed)
    schedule = []
    for repetition in range(1, args.repetitions + 1):
        block = conditions.copy()
        rng.shuffle(block)
        schedule.extend((*condition, repetition) for condition in block)
    return schedule


def bounded_prompt(target_tokens: int, *, evidence: bool = False) -> str:
    records = [
        "[E1] Pump-102 vibration is 8.2 mm/s RMS in the latest authorized snapshot.",
        "[E2] SOP revision 4 section 7.4 sets the alert threshold at 7.1 mm/s RMS.",
        "[E3] The maintenance history records bearing inspection as the next non-invasive action.",
        "[E4] No source authorizes automatic shutdown or a control-system command.",
    ]
    prefix = (
        "Use only the evidence. State the supported Pump-102 finding, recommend a bounded next step, "
        "and cite bracketed evidence IDs. Abstain from any unsupported claim.\n\n"
        if evidence else
        "Summarize the following authorized industrial context in two sentences.\n\n"
    )
    block = " ".join(records) if evidence else "Authorized maintenance context for Pump-102. "
    words = prefix.split()
    block_words = block.split()
    while len(words) < target_tokens:
        words.extend(block_words)
    return " ".join(words[:target_tokens])


def quality(answer: str) -> tuple[float, float, float]:
    cited = set(re.findall(r"\[E(\d+)\]", answer.upper()))
    valid = {"1", "2", "3", "4"}
    precision = len(cited & valid) / max(1, len(cited))
    recall = len(cited & valid) / len(valid)
    supported_terms = ("8.2", "7.1", "bearing", "inspection", "no", "automatic")
    groundedness = sum(term in answer.lower() for term in supported_terms) / len(supported_terms)
    return precision, recall, groundedness


async def stream_vllm(client: httpx.AsyncClient, endpoint: str, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    first: float | None = None
    text: list[str] = []
    usage: dict[str, Any] = {}
    complete = False
    async with client.stream("POST", f"{endpoint.rstrip('/')}/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 256, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }, headers={"Authorization": "Bearer local"}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if line.startswith("data:") and line[5:].strip() == "[DONE]":
                complete = True
                break
            if not line.startswith("data:") or not line[5:].strip():
                continue
            item = json.loads(line[5:])
            usage = item.get("usage") or usage
            choices = item.get("choices") or []
            token = str((choices[0].get("delta") or {}).get("content") or "") if choices else ""
            if token:
                first = first or time.perf_counter()
                text.append(token)
    finished = time.perf_counter()
    if not complete or not text:
        raise ValueError("Incomplete or empty vLLM stream")
    usage.update({"ttft_ms": ((first or finished) - started) * 1000, "total_ms": (finished - started) * 1000})
    return "".join(text), usage


async def stream_ollama(client: httpx.AsyncClient, endpoint: str, model: str, prompt: str) -> tuple[str, dict[str, Any]]:
    started = time.perf_counter()
    first: float | None = None
    text: list[str] = []
    final: dict[str, Any] = {}
    async with client.stream("POST", f"{endpoint.rstrip('/')}/api/generate", json={
        "model": model, "prompt": prompt, "stream": True, "think": False,
        "keep_alive": "15m", "options": {"temperature": 0, "num_predict": 256},
    }) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.strip():
                continue
            item = json.loads(line)
            token = str(item.get("response") or "")
            if token:
                first = first or time.perf_counter()
                text.append(token)
            if item.get("done"):
                final = item
    finished = time.perf_counter()
    if not final.get("done") or not text:
        raise ValueError("Incomplete or empty Ollama stream")
    final.update({
        "ttft_ms": ((first or finished) - started) * 1000,
        "total_ms": (finished - started) * 1000,
        "prompt_tokens": final.get("prompt_eval_count"),
        "completion_tokens": final.get("eval_count"),
    })
    return "".join(text), final


async def observe(provider: str, endpoint: str, model: str, prompt: str, experiment: str,
                  target: int, concurrency: int, repetition: int, timeout: float) -> Observation:
    started = time.perf_counter()
    try:
        endpoint = validate_endpoint(endpoint)
        if provider not in {"vllm", "ollama"}:
            raise ValueError("Unknown inference provider")
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False, trust_env=False) as client:
            answer, metrics = await (
                stream_vllm(client, endpoint, model, prompt)
                if provider == "vllm" else stream_ollama(client, endpoint, model, prompt)
            )
        completion = metrics.get("completion_tokens")
        total_ms = float(metrics.get("total_ms") or 0)
        ttft_ms = float(metrics.get("ttft_ms")) if metrics.get("ttft_ms") is not None else None
        # End-to-end request throughput; do not inflate it by counting the first
        # token again over a post-first-token interval.
        generation_s = total_ms / 1000
        precision = recall = grounding = None
        if experiment == "evidence_budget":
            precision, recall, grounding = quality(answer)
        return Observation(
            experiment, provider, model, target, concurrency, repetition, ttft_ms, total_ms,
            metrics.get("prompt_tokens") or metrics.get("prompt_eval_count"), completion,
            (float(completion) / generation_s) if completion is not None and generation_s > 0 else None,
            precision, recall, grounding, answer=answer,
            prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
        )
    except Exception as exc:
        return Observation(experiment, provider, model, target, concurrency, repetition,
                           None, (time.perf_counter() - started) * 1000, None, None, None,
                           error=str(exc), prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest())


async def main_async(args: argparse.Namespace) -> int:
    # Validate all inputs before making even the first request.
    if args.repetitions < (5 if args.protocol == "controlled" else 1):
        raise ValueError("Controlled protocol requires at least five warm repetitions")
    if (not args.context_lengths or not args.evidence_budgets
            or any(n <= 0 for n in args.context_lengths + args.evidence_budgets)
            or args.timeout <= 0):
        raise ValueError("Budgets and timeout must be positive")
    if len(set(args.providers)) != len(args.providers):
        raise ValueError("Duplicate providers are not independent conditions")
    configured = {
        "vllm": (args.vllm_url, args.vllm_model),
        "ollama": (args.ollama_url, args.ollama_model),
    }
    for name in args.providers:
        endpoint, model = configured[name]
        configured[name] = (validate_endpoint(endpoint), model)
        if args.protocol == "controlled" and not getattr(args, f"{name}_format"):
            raise ValueError(f"Controlled protocol requires --{name}-format for model provenance")
    providers = [(name, *configured[name]) for name in args.providers]
    observations: list[Observation] = []
    if args.protocol == "controlled":
        # Each measured request follows a successful same-condition warm-up.
        # This controls loading/prefix order without asserting a verified cold start.
        for provider, target, experiment, repetition in controlled_schedule(args):
            endpoint, model = configured[provider]
            prompt = bounded_prompt(target, evidence=experiment == "evidence_budget")
            warmup = await observe(provider, endpoint, model, prompt, experiment,
                                   target, 1, repetition, args.timeout)
            warmup.phase = "warmup"
            observations.append(warmup)
            if warmup.error:
                continue
            measured = await observe(provider, endpoint, model, prompt, experiment,
                                     target, 1, repetition, args.timeout)
            measured.phase = "warm"
            observations.append(measured)
    else:
        for provider, endpoint, model in providers:
            for target in args.context_lengths:
                observations.append(await observe(provider, endpoint, model, bounded_prompt(target), "context_length", target, 1, 1, args.timeout))
            for target in args.evidence_budgets:
                observations.append(await observe(provider, endpoint, model, bounded_prompt(target, evidence=True), "evidence_budget", target, 1, 1, args.timeout))
            cached_prompt = bounded_prompt(max(args.context_lengths))
            for repetition in range(1, args.repetitions + 1):
                observations.append(await observe(provider, endpoint, model, cached_prompt, "prefix_cache", max(args.context_lengths), 1, repetition, args.timeout))
            for concurrency in (1, 2, 4):
                rows = await asyncio.gather(*[
                    observe(provider, endpoint, model, bounded_prompt(1024), "concurrency", 1024, concurrency, index + 1, args.timeout)
                    for index in range(concurrency)
                ])
                observations.extend(rows)

    args.output.mkdir(parents=True, exist_ok=True)
    raw = [asdict(item) for item in observations]
    (args.output / "raw.json").write_text(json.dumps(raw, indent=2), encoding="utf-8")
    with (args.output / "raw.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(raw[0]))
        writer.writeheader(); writer.writerows(raw)
    successful = [item for item in observations if not item.error and item.phase != "warmup"]
    summary: dict[str, Any] = {
        "schema_version": 2,
        "protocol": args.protocol, "seed": args.seed,
        "harness_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "experiment_complete": False,
        "failed_requests": sum(bool(item.error) for item in observations),
        "runs": len(observations), "successful": sum(not item.error for item in observations),
        "measured_successful": len(successful),
        "warmup_runs": sum(item.phase == "warmup" for item in observations),
        "requested_repetitions": args.repetitions,
        "budget_unit": "whitespace words (legacy target_tokens field), not tokenizer tokens",
        "throughput_definition": "completion tokens / end-to-end request seconds",
        "quality_definition": "synthetic lexical/citation proxies, not adjudicated factuality",
        "cold_start": "not measured; external verified runtime reset required",
        "memory": None,
        "memory_status": "not measured; runtime process/device instrumentation required",
        "model_identity_status": "operator-declared, not independently verified",
        "models": {name: {"model": model, "format": getattr(args, f"{name}_format"),
                          "endpoint": endpoint} for name, endpoint, model in providers},
        "groups": {},
    }
    for key in sorted({(item.experiment, item.provider, item.target_tokens, item.concurrency) for item in successful}):
        rows = [item for item in successful if (item.experiment, item.provider, item.target_tokens, item.concurrency) == key]
        summary["groups"]["|".join(map(str, key))] = {
            "samples": len(rows),
            "ttft_p50_ms": statistics.median(item.ttft_ms for item in rows if item.ttft_ms is not None),
            "latency_p50_ms": statistics.median(item.total_ms for item in rows),
            "throughput_tok_s": statistics.fmean(item.tokens_per_second for item in rows if item.tokens_per_second is not None) if any(item.tokens_per_second is not None for item in rows) else None,
            "groundedness_mean": statistics.fmean(item.groundedness for item in rows if item.groundedness is not None) if any(item.groundedness is not None for item in rows) else None,
            "citation_recall_mean": statistics.fmean(item.citation_recall for item in rows if item.citation_recall is not None) if any(item.citation_recall is not None for item in rows) else None,
            "aggregate_throughput_tok_s": (
                sum(item.completion_tokens or 0 for item in rows)
                / max(item.total_ms for item in rows) * 1000
                if key[0] == "concurrency" and rows else None
            ),
        }
    summary["prefix_cache"] = [
        {"provider": item.provider, "repetition": item.repetition, "ttft_ms": item.ttft_ms}
        for item in successful if item.experiment == "prefix_cache"
    ]
    summary["pareto_evidence_budgets"] = {}
    for provider in sorted({item.provider for item in successful}):
        points = []
        for budget in sorted({
            item.target_tokens for item in successful
            if item.provider == provider and item.experiment == "evidence_budget"
        }):
            rows = [item for item in successful if item.provider == provider and item.experiment == "evidence_budget" and item.target_tokens == budget]
            ttft = statistics.fmean(item.ttft_ms for item in rows if item.ttft_ms is not None)
            grounding = statistics.fmean(item.groundedness or 0 for item in rows)
            citation = statistics.fmean(item.citation_recall or 0 for item in rows)
            points.append({"evidence_tokens": budget, "ttft_ms": ttft, "quality": (grounding + citation) / 2})
        summary["pareto_evidence_budgets"][provider] = [
            point for point in points if not any(
                other["ttft_ms"] <= point["ttft_ms"]
                and other["quality"] >= point["quality"]
                and (other["ttft_ms"] < point["ttft_ms"] or other["quality"] > point["quality"])
                for other in points
            )
        ]
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0 if observations and all(not item.error for item in observations) else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", required=True)
    parser.add_argument(
        "--providers", nargs="+", choices=("vllm", "ollama"),
        default=["vllm", "ollama"],
        help="Runtime(s) to measure; use one value to append a missing provider run independently.",
    )
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--evidence-budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--protocol", choices=("smoke", "controlled"), default="smoke")
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--vllm-format", help="Operator-declared weight identity/precision; not auto-verified")
    parser.add_argument("--ollama-format", help="Operator-declared weight identity/quantization; not auto-verified")
    parser.add_argument("--repetitions", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/inference_tradeoff"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
