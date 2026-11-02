"""Build a self-contained Colab notebook from the maintained benchmark client."""
from __future__ import annotations

import json
from pathlib import Path
import textwrap


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "notebooks" / "vllm_gpu_latency.ipynb"


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {},
              "source": textwrap.dedent(source).strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    client = (ROOT / "scripts" / "benchmark_vllm_latency.py").read_text(encoding="utf-8")
    cells = [
        cell("markdown", """
            # Measure Qwen latency on a cloud GPU using vLLM

            In Google Colab, select **Runtime > Change runtime type > GPU**, then run
            these cells in order. GPU access depends on your account and availability.
            No cloud account credentials, repository access, or project files are needed.
            Installation and model weights are downloaded into this notebook runtime.

            The default model is Qwen3 4B Instruct, matching the intended planner family.
            Its float16 weights differ from quantized Ollama weights: do not attribute
            a latency difference solely to the runtime. Use `Qwen/Qwen3-0.6B` for a
            smaller smoke run; that measures a different model.

            This notebook measures streaming LLM serving latency with synthetic numbers.
            It does not run the application's diagnostic, retrieval, or tool workflow.
            The server binds to loopback. No public endpoint or tunnel is created.
            Stop the server and disconnect/delete the runtime when finished.

            References: [vLLM GPU requirements](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/),
            [vLLM server](https://docs.vllm.ai/en/latest/cli/serve/),
            [Colab GPU availability](https://research.google.com/colaboratory/faq.html).
        """),
        cell("code", """
            import json, os, pathlib, shutil, subprocess, sys, time, urllib.request

            MODEL = "Qwen/Qwen3-4B-Instruct-2507"
            VLLM_PACKAGE = "vllm"  # Exact installed versions are saved with the results.
            REPEATS, WARMUPS, MAX_TOKENS = 20, 2, 128
            work = pathlib.Path("/content/vllm-latency")
            work.mkdir(parents=True, exist_ok=True)
            gpu = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, check=True).stdout.strip()
            print("GPU:", gpu)
            assert sys.version_info >= (3, 11), "Use a current Python 3.11+ runtime"
        """),
        cell("markdown", """
            Installation can take several minutes. vLLM uses its own Python environment
            so notebook preinstalled Torch/Transformers packages do not determine its dependencies.
            GPU/driver compatibility is checked by the server; inspect `server.log` if it fails.
        """),
        cell("code", """
            subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "virtualenv"], check=True)
            env = work / "env"
            subprocess.run([sys.executable, "-m", "virtualenv", str(env)], check=True)
            python = str(env / "bin" / "python")
            subprocess.run([python, "-m", "pip", "install", VLLM_PACKAGE], check=True)
            freeze = subprocess.run([python, "-m", "pip", "freeze"], capture_output=True,
                                    text=True, check=True).stdout
            (work / "requirements-resolved.txt").write_text(freeze)
        """),
        cell("code", """
            # Rerunning this cell stops only the server started by this notebook.
            if "server" in globals() and server.poll() is None:
                server.terminate()
                server.wait(timeout=30)
            envvars = dict(os.environ, VLLM_NO_USAGE_STATS="1", DO_NOT_TRACK="1",
                           HF_HUB_DISABLE_TELEMETRY="1")
            help_text = subprocess.run([str(env / "bin" / "vllm"), "serve", "--help"],
                                       capture_output=True, text=True, check=True, env=envvars).stdout
            # vLLM versions use either spelling for disabling automatic prefix caching.
            cache_flag = ("--disable-prefix-caching" if "--disable-prefix-caching" in help_text
                          else "--no-enable-prefix-caching")
            command = [str(env / "bin" / "vllm"), "serve", MODEL,
                       "--host", "127.0.0.1", "--port", "8001", "--dtype", "half",
                       "--max-model-len", "8192", "--max-num-seqs", "1",
                       "--gpu-memory-utilization", "0.85", "--enforce-eager", cache_flag]
            (work / "server-command.json").write_text(json.dumps(command, indent=2))
            log = (work / "server.log").open("w")
            server = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, env=envvars)
            endpoint = "http://127.0.0.1:8001/v1"
            startup = time.monotonic()
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            try:
                while time.monotonic() - startup < 1200:
                    if server.poll() is not None:
                        raise RuntimeError("vLLM exited. See server.log below.")
                    try:
                        with opener.open(endpoint + "/models", timeout=3) as response:
                            available = json.load(response)
                        if MODEL in {item["id"] for item in available["data"]}:
                            break
                    except (OSError, ValueError):
                        pass
                    print("Waiting for model download/load:", round(time.monotonic() - startup), "seconds", flush=True)
                    time.sleep(10)
                else:
                    raise TimeoutError("Model startup exceeded 20 minutes")
                print("Ready:", available)
                print("Startup seconds (excluded from serving latency):", round(time.monotonic() - startup, 2))
            except Exception:
                if server.poll() is None:
                    server.terminate()
                print((work / "server.log").read_text()[-10000:])
                raise
        """),
        cell("markdown", """
            Run two warmups and twenty measured requests at concurrency one. Record
            median/p95 time to first visible token and total response time. Decode speed
            is an estimate based on server usage counts and client stream timestamps.
            `--enforce-eager` and disabled prefix caching are reproducible starter settings,
            not a claim of optimal vLLM performance. Missing counts remain unknown;
            errors are retained and excluded from successful latency percentiles.
        """),
        cell("code", "client_source = " + repr(client) + "\n(work / 'benchmark_vllm_latency.py').write_text(client_source)"),
        cell("code", """
            result_path = work / "vllm-latency.json"
            result = subprocess.run(
                [python, str(work / "benchmark_vllm_latency.py"), "--endpoint", endpoint,
                 "--model", MODEL, "--repeats", str(REPEATS), "--warmups", str(WARMUPS),
                 "--max-tokens", str(MAX_TOKENS), "--server-label", gpu + "; " + " ".join(command),
                 "--output", str(result_path)])
            if result_path.exists():
                report = json.loads(result_path.read_text())
                print(json.dumps(report["summary"], indent=2))
            else:
                raise RuntimeError("Preflight failed; no measured results were produced")
            if result.returncode:
                print("Some requests failed; inspect samples before comparing results.")
        """),
        cell("code", """
            # Download results plus environment/launch details for a reproducible report.
            archive = work / "results"
            archive.mkdir(exist_ok=True)
            for name in ("vllm-latency.json", "requirements-resolved.txt", "server-command.json", "server.log"):
                if (work / name).exists():
                    shutil.copy2(work / name, archive / name)
            zip_path = shutil.make_archive(str(work / "vllm-latency-results"), "zip", archive)
            from google.colab import files
            files.download(zip_path)
        """),
        cell("code", """
            # Stop the process we started. Also disconnect/delete the Colab runtime.
            if "server" in globals() and server.poll() is None:
                server.terminate()
                server.wait(timeout=30)
            if "log" in globals():
                log.close()
            print("vLLM server stopped")
        """),
    ]
    notebook = {"cells": cells, "metadata": {
        "accelerator": "GPU", "colab": {"name": OUTPUT.name, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"}},
        "nbformat": 4, "nbformat_minor": 5}
    for index, item in enumerate(cells):
        item["id"] = f"vllm-latency-{index:02d}"
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(notebook, indent=2) + "\n", encoding="utf-8")
    return OUTPUT


if __name__ == "__main__":
    print(build())
