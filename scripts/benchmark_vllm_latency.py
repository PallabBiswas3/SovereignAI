"""Measure streaming OpenAI-compatible inference using synthetic inputs only.

Standard library only; works beside vLLM on a GPU VM or in Colab. This measures
client-observed serving latency, not the application's full diagnostic workflow.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import ipaddress
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request


PROMPT = (
    "These are synthetic example numbers, not plant measurements. "
    "For [10, 20, 30], explain how to calculate the mean, multiply it by eight, "
    "and convert that value from kPa to bar. Show the arithmetic and units."
)


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def normalize_endpoint(endpoint, allow_remote=False):
    parsed = urllib.parse.urlsplit(endpoint)
    if (parsed.scheme not in {"http", "https"} or not parsed.hostname
            or parsed.username is not None or parsed.password is not None
            or parsed.query or parsed.fragment):
        raise ValueError("Use an HTTP(S) base URL without credentials, query or fragment")
    try:
        local = ipaddress.ip_address(parsed.hostname).is_loopback
    except ValueError:
        local = parsed.hostname.lower() in {"localhost", "vllm"}
    if not local and not (allow_remote and parsed.scheme == "https"):
        raise ValueError("Remote benchmarking requires --allow-remote and an HTTPS endpoint")
    base = endpoint.rstrip("/")
    return base if base.endswith("/v1") else base + "/v1"


def percentile(values, percent):
    """Linearly interpolated sample percentile; small samples remain descriptive."""
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent / 100
    low, high = math.floor(index), math.ceil(index)
    return round(ordered[low] + (ordered[high] - ordered[low]) * (index - low), 6)


def summarize(samples):
    good = [sample for sample in samples if sample["success"]]
    summary = {"attempted": len(samples), "successful": len(good),
               "failed": len(samples) - len(good)}
    for key in ("ttft_seconds", "total_seconds", "decode_tokens_per_second_estimate"):
        values = [sample[key] for sample in good if sample.get(key) is not None]
        summary[key] = {"count": len(values), "p50": percentile(values, 50),
                        "p95": percentile(values, 95)}
    return summary


def stream_sample(opener, endpoint, model, api_key, max_tokens, timeout):
    payload = {"model": model, "messages": [{"role": "user", "content": PROMPT}],
               "temperature": 0, "max_tokens": max_tokens, "stream": True,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": False}}
    request = urllib.request.Request(
        endpoint + "/chat/completions", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"})
    started = time.perf_counter()
    first = last = None
    usage, finish_reason = {}, None
    saw_done = False
    with opener.open(request, timeout=timeout) as response:
        for line in response:
            now = time.perf_counter()
            if now - started > timeout:
                raise TimeoutError("Total request deadline exceeded")
            if not line.startswith(b"data:"):
                continue
            raw = line[5:].strip()
            if raw == b"[DONE]":
                saw_done = True
                break
            if not raw:
                continue
            event = json.loads(raw)
            if event.get("error"):
                raise ValueError("Server returned an error event")
            if isinstance(event.get("usage"), dict):
                usage = event["usage"]
            choices = event.get("choices") or []
            if choices:
                choice = choices[0]
                finish_reason = choice.get("finish_reason") or finish_reason
                if (choice.get("delta") or {}).get("content"):
                    first = now if first is None else first
                    last = now
    if not saw_done or first is None:
        raise ValueError("Stream did not finish with [DONE] and nonempty visible output")
    count = usage.get("completion_tokens")
    prompt_count = usage.get("prompt_tokens")
    count = count if type(count) is int and count > 0 else None
    prompt_count = prompt_count if type(prompt_count) is int and prompt_count >= 0 else None
    decode = last - first
    return {"success": True, "ttft_seconds": round(first - started, 6),
            "total_seconds": round(time.perf_counter() - started, 6),
            "prompt_tokens": prompt_count, "completion_tokens": count,
            "finish_reason": finish_reason,
            "decode_tokens_per_second_estimate": (
                round((count - 1) / decode, 3) if count and count > 1 and decode > 0 else None)}


def hardware_info():
    info = {"client_platform": platform.platform(), "client_python": platform.python_version()}
    if shutil.which("nvidia-smi"):
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=10,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), check=True)
            info["client_gpus"] = result.stdout.strip().splitlines()
        except (OSError, subprocess.SubprocessError):
            info["client_gpus"] = None
    return info


def positive_int(value):
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("Must be at least 1")
    return parsed


def nonnegative_int(value):
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("Must be nonnegative")
    return parsed


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="http://127.0.0.1:8001/v1")
    parser.add_argument("--allow-remote", action="store_true",
                        help="Permit a remote HTTPS endpoint in this standalone benchmark only")
    parser.add_argument("--model", required=True, help="Exact served model ID from /v1/models")
    parser.add_argument("--api-key-env", default="SOVEREIGN_VLLM_API_KEY",
                        help="Environment variable containing the API key (never written to results)")
    parser.add_argument("--repeats", type=positive_int, default=20)
    parser.add_argument("--warmups", type=nonnegative_int, default=2)
    parser.add_argument("--max-tokens", type=positive_int, default=128)
    parser.add_argument("--timeout", type=positive_int, default=120)
    parser.add_argument("--server-label", default="unspecified",
                        help="Record GPU type, model revision, quantization and server flags")
    parser.add_argument("--output", type=Path, default=Path("vllm-latency.json"))
    args = parser.parse_args(argv)
    try:
        args.endpoint = normalize_endpoint(args.endpoint, args.allow_remote)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def run(args):
    api_key = os.environ.get(args.api_key_env, "local")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    # Verify the requested model before starting any measured generation.
    request = urllib.request.Request(args.endpoint + "/models",
                                     headers={"Authorization": f"Bearer {api_key}"})
    try:
        with opener.open(request, timeout=args.timeout) as response:
            models = json.load(response)["data"]
        if args.model not in {item["id"] for item in models}:
            raise ValueError("Requested model is not served by this endpoint")
    except (OSError, ValueError, KeyError) as exc:
        # Do not print HTTP response bodies or request headers containing credentials.
        print(json.dumps({"preflight": "failed", "error_type": type(exc).__name__,
                          "hint": "Start vLLM and check endpoint, served model ID and API key"}))
        return 2
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Client-observed streaming inference; includes network, server queueing and generation. Excludes application retrieval/tools, UI and server startup/model loading.",
        "endpoint": args.endpoint, "model": args.model, "server_label": args.server_label,
        "served_model_metadata": [{key: item[key] for key in ("id", "root", "max_model_len") if key in item}
                                  for item in models if item["id"] == args.model],
        "client_hardware": hardware_info(),
        "settings": {"repeats": args.repeats, "warmups": args.warmups,
                     "max_tokens": args.max_tokens, "temperature": 0, "concurrency": 1,
                     "thinking": False, "prompt": PROMPT},
        "measurement_notes": [
            "TTFT ends at the first visible content chunk; reasoning chunks do not count.",
            "Each request opens a connection; client network/connection overhead is included.",
            "Decode speed uses (completion_tokens - 1)/(last_content - first_content); chunking makes it an estimate.",
            "Missing usage remains unknown. Failed requests are excluded from latency percentiles.",
            "Repeated identical inputs may hit server prefix caches; record cache settings in server-label.",
            "Client GPU metadata describes the server only when the benchmark runs on that server.",
            "Nonempty output is checked; arithmetic correctness and workflow quality are not evaluated.",
            "Warmups are excluded from summary; these are warm serving samples, not cold-start measurements."],
        "warmup_samples": [], "samples": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.warmups + args.repeats):
        warmup = index < args.warmups
        try:
            sample = stream_sample(opener, args.endpoint, args.model, api_key,
                                   args.max_tokens, args.timeout)
        except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
            sample = {"success": False, "error_type": type(exc).__name__}
        sample["index"] = index + 1 if warmup else index - args.warmups + 1
        report["warmup_samples" if warmup else "samples"].append(sample)
        report["summary"] = summarize(report["samples"])
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps({"warmup": warmup, **sample}), flush=True)
    print(json.dumps({"summary": report["summary"], "output": str(args.output)}), flush=True)
    return 0 if report["summary"]["failed"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
