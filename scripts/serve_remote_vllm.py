"""Start vLLM on a Linux GPU host, bound to loopback behind HTTPS ingress.

Requires vLLM installed on that host. Credentials come from the environment and
are never placed in process arguments. This script provisions no cloud account.
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess


def launch_config(args, environ):
    key = environ.get("SOVEREIGN_REMOTE_VLLM_API_KEY", "")
    if len(key) < 32 or any(char.isspace() for char in key):
        raise ValueError("Set SOVEREIGN_REMOTE_VLLM_API_KEY to a random secret of at least 32 characters")
    executable = shutil.which("vllm")
    if not executable:
        raise ValueError("Install vLLM in the GPU host's active Python environment first")
    command = [executable, "serve", args.model, "--host", "127.0.0.1", "--port", "8001",
               "--dtype", "half", "--max-model-len", str(args.context_length),
               "--gpu-memory-utilization", str(args.gpu_memory_utilization),
               "--disable-log-requests"]
    if args.revision:
        command += ["--revision", args.revision]
    environment = dict(environ)
    environment["VLLM_API_KEY"] = key
    environment["VLLM_NO_USAGE_STATS"] = "1"
    environment["DO_NOT_TRACK"] = "1"
    return command, environment


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    parser.add_argument("--revision", help="Optional Hugging Face commit SHA for reproducible weights")
    parser.add_argument("--context-length", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args(argv)
    if args.context_length < 1024 or not 0 < args.gpu_memory_utilization < 1:
        parser.error("Use context length >=1024 and GPU memory utilization between 0 and 1")
    try:
        command, environment = launch_config(args, os.environ)
    except ValueError as exc:
        parser.error(str(exc))
    print("Starting authenticated vLLM on 127.0.0.1:8001; expose only the HTTPS ingress.", flush=True)
    try:
        return subprocess.call(command, env=environment)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
