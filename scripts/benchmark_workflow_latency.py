"""Compare Ollama or vLLM planners on isolated, verifiable workflow samples.

Run from the project root with .venv/Scripts/python.exe. No downloads or saved
configuration changes. Results are smoke observations, not statistical guarantees.
"""
from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys
import tempfile
import time
from unittest.mock import patch
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
import yaml
import psutil
from sqlalchemy import create_engine
from sqlalchemy.orm import Session
from docx import Document
from app.api import tasks
from app.core.config import get_settings
from app.core.database import Base
from app.identity.dependencies import development_principal
from app.monitoring.network import LocalNetworkPolicy
from app.orchestration.structured_plan import bindings


CASES = [
    ("numeric_chain", "Use summarize_numbers on [10, 20, 30]. Then use calculate to multiply the observed mean by 8. Then use convert_units to convert that calculated value from kPa to bar. Report the actual results from all three tools. These are example numbers, not plant measurements.", "NUMERIC"),
    ("conversion_chain", "Use convert_units to convert 250 kPa to bar. Then use calculate to multiply the observed converted value by 4. Report the actual results.", "NUMERIC"),
    ("file_report", "Use read_file to read input.txt. Then use generate_docx to create evidence.docx titled Inspection evidence with a Recorded observations section containing the exact observed file text. Report the created artifact.", "ALL"),
]


def request_json(url, data=None, *, api_key=None):
    LocalNetworkPolicy.require_local(url)
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(url, data=json.dumps(data).encode() if data is not None else None,
                                     headers=headers)
    # Match the inference adapters: never follow redirects to another destination.
    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, req, fp, code, msg, response_headers, newurl):
            return None
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    with opener.open(request, timeout=120) as response:
        return json.load(response)


def benchmark_settings(args, settings):
    provider = args.provider or settings.llm_provider
    updates = {"llm_provider": provider, "allow_deterministic_fallback": False,
               "cache_enabled": False}
    if args.endpoint:
        LocalNetworkPolicy.require_local(args.endpoint)
        updates["vllm_url" if provider == "vllm" else "ollama_url"] = args.endpoint.rstrip("/")
    if provider == "vllm":
        endpoint = str(updates.get("vllm_url", settings.vllm_url)).rstrip("/")
        updates["vllm_url"] = endpoint if endpoint.endswith("/v1") else endpoint + "/v1"
    return settings.model_copy(update=updates)


def available_models(settings):
    if settings.llm_provider == "vllm":
        payload = request_json(settings.vllm_url + "/models", api_key=settings.vllm_api_key)
        return {item["id"]: item for item in payload["data"]}
    payload = request_json(settings.ollama_url + "/api/tags")
    return {item["name"]: item for item in payload["models"]}


def check(case, state, workspace):
    if state.status.value != "completed" or state.structured_plan.get("status") != "completed":
        return False, "Workflow did not complete as a structured plan"
    if not state.tool_records or not all(item.get("success") for item in state.tool_records):
        return False, "A requested tool failed or did not run"
    records = {item["tool"]: item for item in state.tool_records}
    plan = state.structured_plan["steps"]
    if case == "numeric_chain":
        if set(records) != {"summarize_numbers", "calculate", "convert_units"}:
            return False, "Unexpected numeric tool selection"
        result = json.loads(records["calculate"]["output_summary"])["result"]
        converted = json.loads(records["convert_units"]["output_summary"])["normalized_value"]
        bound = all(bindings(step["arguments"]) for step in plan if step["tool"] in {"calculate", "convert_units"})
        return result == 160 and converted == 1.6 and records["convert_units"]["arguments"]["value"] == 160 and bound, "Check mean/scaling/conversion and actual dependencies"
    if case == "conversion_chain":
        if set(records) != {"convert_units", "calculate"}:
            return False, "Unexpected conversion tool selection"
        result = json.loads(records["calculate"]["output_summary"])["result"]
        bound = any(bindings(step["arguments"]) for step in plan if step["tool"] == "calculate")
        return result == 10 and bound, "Check converted 2.5 bar times four and observed-value binding"
    if set(records) != {"read_file", "generate_docx"} or not state.artifacts:
        return False, "Expected read_file and a registered report artifact"
    report = workspace / "artifacts" / "evidence.docx"
    if not report.exists():
        return False, "Requested report filename was not produced"
    text = "\n".join(paragraph.text for paragraph in Document(report).paragraphs)
    bound = any(bindings(step["arguments"]) for step in plan if step["tool"] == "generate_docx")
    return "Observation Z7: recorded vibration was 2 mm/s." in text and bound, "Check exact observed text in actual Word artifact"


async def main(args):
    settings = benchmark_settings(args, get_settings())
    args.models = args.models or ([settings.vllm_model] if settings.llm_provider == "vllm"
                                  else ["qwen3:4b-instruct", "qwen3:0.6b"])
    available = available_models(settings)
    if any(model not in available for model in args.models):
        raise RuntimeError("Requested model is not served; this script never downloads models")
    endpoint = settings.vllm_url if settings.llm_provider == "vllm" else settings.ollama_url
    report = {"scope": "Sequential isolated workflow smoke tasks with fixed synthetic inputs. Model preloading excluded. Application response cache and deterministic inference fallback disabled. Task-handler timings exclude HTTP/SSE delivery and browser rendering. Not a controlled statistical model comparison.",
              "provider": settings.llm_provider, "endpoint": endpoint,
              "models": {name: available[name] for name in args.models}, "samples": []}
    default_name = "vllm-latency-comparison.json" if settings.llm_provider == "vllm" else "latency-comparison.json"
    output = args.output or ROOT / "local_testing" / "agent_tools" / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    if args.append and output.exists():
        previous = json.loads(output.read_text(encoding="utf-8"))
        if (previous.get("provider", "ollama") != settings.llm_provider
                or previous.get("endpoint", settings.ollama_url) != endpoint):
            raise RuntimeError("Cannot append measurements from a different provider or endpoint")
        report["models"].update(previous.get("models", {}))
        report["samples"] = previous.get("samples", [])
    original_models = yaml.safe_load(settings.models_config.read_text(encoding="utf-8"))
    for model in args.models:
        started = time.perf_counter()
        if settings.llm_provider == "ollama":
            request_json(settings.ollama_url + "/api/generate", {"model": model, "prompt": "", "keep_alive": "15m", "options": {"num_ctx": 8192}})
            print(json.dumps({"preloaded": model, "seconds": round(time.perf_counter() - started, 3)}), flush=True)
        for case, goal, scope in CASES:
            if case not in args.cases:
                continue
            existing = next((sample for sample in report["samples"] if sample["model"] == model and sample["case"] == case), None)
            if existing:
                if existing["correct"] or not args.retry_failed:
                    if not existing["correct"] and not args.keep_going:
                        print(json.dumps({"model": model, "rejected": "Existing failed quality check; remaining cases skipped"}), flush=True)
                        break
                    continue
                report.setdefault("previous_attempts", []).append(existing)
                report["samples"].remove(existing)
            with tempfile.TemporaryDirectory(prefix="sovereign-latency-") as directory:
                workspace = Path(directory)
                models = json.loads(json.dumps(original_models))
                models["models"]["general"]["model_tag"] = model
                models["models"]["general"]["endpoint"] = endpoint
                models["models"]["general"]["provider"] = settings.llm_provider
                config = workspace / "models.yaml"
                config.write_text(yaml.safe_dump(models), encoding="utf-8")
                (workspace / "input.txt").write_text("Observation Z7: recorded vibration was 2 mm/s.", encoding="utf-8")
                isolated = settings.model_copy(update={"workspace_root": workspace, "models_config": config,
                                                       "vllm_model": model})
                engine = create_engine(f"sqlite:///{(workspace / 'task.db').as_posix()}")
                Base.metadata.create_all(engine)
                try:
                    with Session(engine) as db, patch.object(tasks, "get_settings", return_value=isolated):
                        state = await tasks._execute_task(tasks.CreateTaskRequest(
                            request=goal, chat_mode="CONTROLLED", tool_scope=scope, model_override="general"),
                            db, development_principal())
                    passed, validation = check(case, state, workspace)
                    timing = state.runtime_metrics["timing"]
                    calls = timing.get("llm_calls", [])
                    if not calls or any(call["outcome"] != "completed" for call in calls):
                        passed, validation = False, "Missing, failed or fallback model inference"
                    sample = {"model": model, "case": case, "correct": passed, "validation": validation,
                              "ram_percent_after": psutil.virtual_memory().percent,
                              "status": state.status.value, "governance": state.governance["decision"],
                              "response_generation": state.runtime_metrics.get("response_generation"),
                              "prompt_compaction": state.runtime_metrics.get("prompt_compaction"), "timing": timing,
                              "plan": state.structured_plan, "tools": state.tool_records, "response": state.final_response}
                    report["samples"].append(sample)
                    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
                    print(json.dumps({"model": model, "case": case, "correct": passed, "calls": timing["llm_call_count"],
                                      "task_seconds": timing["task_handler_seconds"], "governance": sample["governance"]}), flush=True)
                    if not passed and not args.keep_going:
                        print(json.dumps({"model": model, "rejected": "Failed quality check; remaining cases skipped"}), flush=True)
                        break
                finally:
                    engine.dispose()
        # Release only the planner this benchmark just used; leave the Ollama service running.
        if settings.llm_provider == "ollama":
            request_json(settings.ollama_url + "/api/generate", {"model": model, "prompt": "", "keep_alive": 0})
    return 0 if report["samples"] and all(sample["correct"] for sample in report["samples"]) else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=["ollama", "vllm"], help="Override provider for this process only")
    parser.add_argument("--endpoint", help="Loopback/internal endpoint; run beside a cloud GPU or use an SSH tunnel")
    parser.add_argument("--models", nargs="+", help="Served model IDs; defaults depend on the provider")
    parser.add_argument("--output", type=Path, help="JSON results path")
    parser.add_argument("--cases", nargs="+", choices=[item[0] for item in CASES], default=[item[0] for item in CASES])
    parser.add_argument("--append", action="store_true", help="Retain existing samples and run only missing model/case pairs")
    parser.add_argument("--retry-failed", action="store_true", help="Recheck failed samples after a fix; retain previous attempts")
    parser.add_argument("--keep-going", action="store_true", help="Continue sampling a model after a failed quality check")
    raise SystemExit(asyncio.run(main(parser.parse_args())))
