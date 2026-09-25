#!/usr/bin/env python3
"""Reproducible local-inference experiments for evidence-budget research.

The harness writes raw observations; it never ships prompts to an external host
and never substitutes synthetic numbers when a runtime is unavailable.
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

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
    cited = set(re.findall(r"\[E([1-4])\]", answer.upper()))
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
    async with client.stream("POST", f"{endpoint.rstrip('/')}/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "temperature": 0, "max_tokens": 256, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False},
    }, headers={"Authorization": "Bearer local"}) as response:
        response.raise_for_status()
        async for line in response.aiter_lines():
            if not line.startswith("data:") or line[5:].strip() in {"", "[DONE]"}:
                continue
            item = json.loads(line[5:])
            usage = item.get("usage") or usage
            choices = item.get("choices") or []
            token = str((choices[0].get("delta") or {}).get("content") or "") if choices else ""
            if token:
                first = first or time.perf_counter()
                text.append(token)
    finished = time.perf_counter()
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
    final.update({
        "ttft_ms": ((first or finished) - started) * 1000,
        "total_ms": (finished - started) * 1000,
        "prompt_tokens": final.get("prompt_eval_count"),
        "completion_tokens": final.get("eval_count"),
    })
    return "".join(text), final


async def observe(provider: str, endpoint: str, model: str, prompt: str, experiment: str,
                  target: int, concurrency: int, repetition: int, timeout: float) -> Observation:
    try:
        async with httpx.AsyncClient(timeout=timeout, follow_redirects=False) as client:
            answer, metrics = await (
                stream_vllm(client, endpoint, model, prompt)
                if provider == "vllm" else stream_ollama(client, endpoint, model, prompt)
            )
        completion = metrics.get("completion_tokens")
        total_ms = float(metrics.get("total_ms") or 0)
        ttft_ms = float(metrics.get("ttft_ms")) if metrics.get("ttft_ms") is not None else None
        generation_s = max(0.000001, (total_ms - (ttft_ms or 0)) / 1000)
        precision = recall = grounding = None
        if experiment == "evidence_budget":
            precision, recall, grounding = quality(answer)
        return Observation(
            experiment, provider, model, target, concurrency, repetition, ttft_ms, total_ms,
            metrics.get("prompt_tokens") or metrics.get("prompt_eval_count"), completion,
            (float(completion) / generation_s) if completion else None,
            precision, recall, grounding,
        )
    except Exception as exc:
        return Observation(experiment, provider, model, target, concurrency, repetition, None, 0, None, None, None, error=str(exc))


async def main_async(args: argparse.Namespace) -> int:
    providers = [("vllm", args.vllm_url, args.vllm_model), ("ollama", args.ollama_url, args.ollama_model)]
    observations: list[Observation] = []
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
    successful = [item for item in observations if not item.error]
    summary: dict[str, Any] = {"runs": len(observations), "successful": len(successful), "groups": {}}
    for key in sorted({(item.experiment, item.provider, item.target_tokens, item.concurrency) for item in successful}):
        rows = [item for item in successful if (item.experiment, item.provider, item.target_tokens, item.concurrency) == key]
        summary["groups"]["|".join(map(str, key))] = {
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
    return 0 if len(successful) == len(observations) else 2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-url", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--vllm-model", required=True)
    parser.add_argument("--ollama-url", default="http://127.0.0.1:11434")
    parser.add_argument("--ollama-model", required=True)
    parser.add_argument("--context-lengths", type=int, nargs="+", default=[512, 1024, 2048, 4096])
    parser.add_argument("--evidence-budgets", type=int, nargs="+", default=[256, 512, 1024, 2048])
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--output", type=Path, default=Path("experiments/results/inference_tradeoff"))
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main_async(parse_args())))
