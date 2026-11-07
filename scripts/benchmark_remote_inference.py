"""Compare local Ollama and remote vLLM with identical synthetic input.

Standard library only. Run from the laptop to include internet latency. No
documents, generated answers or credentials are stored in the report.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import time
import urllib.request

import benchmark_vllm_latency as protocol


def ollama_sample(opener, endpoint, model, max_tokens, timeout):
    payload = {"model": model, "messages": [{"role": "user", "content": protocol.PROMPT}],
               "think": False, "stream": True, "keep_alive": "5m",
               "options": {"temperature": 0, "num_predict": max_tokens}}
    request = urllib.request.Request(endpoint + "/api/chat", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = last = None
    final = None
    with opener.open(request, timeout=timeout) as response:
        for line in response:
            now = time.perf_counter()
            if now - started > timeout:
                raise TimeoutError("Total request deadline exceeded")
            if not line.strip():
                continue
            item = json.loads(line)
            if item.get("error"):
                raise ValueError("Ollama returned an error event")
            if (item.get("message") or {}).get("content"):
                first = now if first is None else first
                last = now
            if item.get("done") is True:
                final = item
                break
    if final is None or first is None:
        raise ValueError("Ollama stream did not finish with nonempty output")
    count = final.get("eval_count")
    prompt_count = final.get("prompt_eval_count")
    count = count if type(count) is int and count > 0 else None
    prompt_count = prompt_count if type(prompt_count) is int and prompt_count >= 0 else None
    decode = last - first
    return {"success": True, "ttft_seconds": round(first - started, 6),
            "total_seconds": round(time.perf_counter() - started, 6),
            "prompt_tokens": prompt_count, "completion_tokens": count,
            "finish_reason": final.get("done_reason"),
            "decode_tokens_per_second_estimate": (
                round((count - 1) / decode, 3) if count and count > 1 and decode > 0 else None)}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-endpoint", default="http://127.0.0.1:11434")
    parser.add_argument("--local-model", default="qwen3:4b-instruct")
    parser.add_argument("--local-only", action="store_true", help="Record a baseline before a GPU host is available")
    parser.add_argument("--remote-endpoint", default=os.environ.get("SOVEREIGN_REMOTE_VLLM_URL", ""))
    parser.add_argument("--remote-model", default=os.environ.get("SOVEREIGN_REMOTE_VLLM_MODEL", ""))
    parser.add_argument("--api-key-env", default="SOVEREIGN_REMOTE_VLLM_API_KEY")
    parser.add_argument("--local-label", default="Record local hardware, model revision and quantization")
    parser.add_argument("--remote-label", default="Record GPU, model revision, quantization and server flags")
    parser.add_argument("--repeats", type=protocol.positive_int, default=20)
    parser.add_argument("--warmups", type=protocol.nonnegative_int, default=2)
    parser.add_argument("--max-tokens", type=protocol.positive_int, default=128)
    parser.add_argument("--timeout", type=protocol.positive_int, default=120)
    parser.add_argument("--output", type=Path, default=Path("workspace/artifacts/inference-comparison.json"))
    args = parser.parse_args(argv)
    try:
        args.local_endpoint = protocol.normalize_endpoint(args.local_endpoint).removesuffix("/v1")
        if not args.local_only:
            if not args.remote_model.strip():
                raise ValueError("Set --remote-model to the exact served model ID")
            args.remote_endpoint = protocol.normalize_endpoint(args.remote_endpoint, allow_remote=True)
            if not args.remote_endpoint.startswith("https://"):
                raise ValueError("The remote endpoint must use HTTPS")
    except ValueError:
        parser.error("Check endpoint syntax: local HTTP(S), remote HTTPS, no URL credentials/query/fragment; supply a remote model")
    return args


def comparison(report):
    if "remote" not in report["runs"]:
        return {}
    local, remote = (report["runs"][key]["summary"] for key in ("local", "remote"))
    ratios = {}
    for metric in ("ttft_seconds", "total_seconds"):
        baseline, measured = local[metric]["p50"], remote[metric]["p50"]
        ratios[metric + "_median_speedup"] = (
            round(baseline / measured, 3) if baseline is not None and measured else None)
    return {**ratios, "all_requests_succeeded": not (local["failed"] or remote["failed"]),
            "interpretation": "Above 1 means remote was faster. Compare model identity, quantization, output counts and failures before attributing the difference to the GPU."}


def run(args):
    api_key = os.environ.get(args.api_key_env, "")
    if not args.local_only and (not api_key.strip() or api_key.lower() in {"local", "changeme", "replace-me"}
                                or any(char.isspace() for char in api_key)):
        print(json.dumps({"preflight": "failed", "hint": "Set the remote API key in the selected environment variable"}))
        return 2
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), protocol.NoRedirect())
    runs = {"local": {"endpoint": args.local_endpoint, "model": args.local_model,
                      "server_label": args.local_label, "warmup_samples": [], "samples": []}}
    if not args.local_only:
        runs["remote"] = {"endpoint": args.remote_endpoint, "model": args.remote_model,
                          "server_label": args.remote_label, "warmup_samples": [], "samples": []}
    # Non-generating preflight must finish before creating a measurement report.
    try:
        for name, run_info in runs.items():
            local = name == "local"
            request = urllib.request.Request(
                run_info["endpoint"] + ("/api/tags" if local else "/models"),
                headers={} if local else {"Authorization": "Bearer " + api_key})
            with opener.open(request, timeout=args.timeout) as response:
                body = json.load(response)
            identifiers = ({item["name"] for item in body["models"]} if local else
                           {item["id"] for item in body["data"]})
            if run_info["model"] not in identifiers:
                raise ValueError("Configured model is not served")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(json.dumps({"preflight": "failed", "error_type": type(exc).__name__,
                          "hint": "Check both runtimes, model IDs, TLS and the remote API key"}))
        return 2
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "scope": "Laptop-observed synthetic streaming inference; retrieval/tools/UI and server startup excluded",
              "settings": {"prompt": protocol.PROMPT, "max_tokens": args.max_tokens, "temperature": 0,
                           "thinking": False, "repeats": args.repeats, "warmups": args.warmups, "concurrency": 1},
              "notes": ["Warmups excluded. Identical prompt/token budget does not guarantee identical output length.",
                        "Ollama GGUF and vLLM weights can differ; this is an end-to-end deployment comparison.",
                        "Network/TLS overhead and server queues are included. Prefix caches can affect repeated requests.",
                        "First visible content defines TTFT; missing usage stays unknown; failed samples stay in the report.",
                        "No generated text is saved; answer quality is not assessed. Cold starts can still occur during measured requests."],
              "client_hardware": protocol.hardware_info(), "runs": runs}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for index in range(args.warmups + args.repeats):
        warmup = index < args.warmups
        for name, run_info in runs.items():
            try:
                sample = (ollama_sample(opener, run_info["endpoint"], run_info["model"], args.max_tokens, args.timeout)
                          if name == "local" else protocol.stream_sample(
                              opener, run_info["endpoint"], run_info["model"], api_key, args.max_tokens, args.timeout))
            except (OSError, ValueError, KeyError, TypeError, IndexError) as exc:
                sample = {"success": False, "error_type": type(exc).__name__}
            run_info["warmup_samples" if warmup else "samples"].append(sample)
            run_info["summary"] = protocol.summarize(run_info["samples"])
            print(json.dumps({"run": name, "warmup": warmup, **sample}), flush=True)
        report["comparison"] = comparison(report)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"comparison": report["comparison"], "output": str(args.output)}))
    return int(any(info["summary"]["failed"] for info in runs.values()))


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
