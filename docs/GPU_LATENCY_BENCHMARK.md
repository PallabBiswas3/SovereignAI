# GPU latency benchmark

## Run without a GPU machine

Open [the self-contained notebook](../notebooks/vllm_gpu_latency.ipynb) in
[Google Colab](https://colab.research.google.com/): File > Upload notebook.
Select Runtime > Change runtime type > GPU, then run the cells in order.
Colab GPU access depends on account limits and availability; a GPU allocation
is required before any real performance measurements can be produced.

The notebook installs vLLM in a separate environment, downloads
`Qwen/Qwen3-4B-Instruct-2507`, starts a loopback server, performs two warmups
and twenty measured requests, and downloads a ZIP with JSON results, package
versions, launch flags, and server logs. The final cell stops its server.
Disconnect/delete the Colab runtime when finished. No paid instance is created
by the repository tooling.

For a smaller smoke run, change `MODEL` to `Qwen/Qwen3-0.6B` before starting
the server. Label this as a different model. Float16 Hugging Face weights and
quantized Ollama weights also produce different comparisons.

The notebook contains the maintained client script. After changing that script,
regenerate the notebook with:

```console
python scripts/create_vllm_benchmark_notebook.py
```

## Run on an existing Linux GPU machine

Install vLLM in a separate Python environment from the application's pinned
Torch/Transformers environment. The example starts with eager execution and
disabled prefix caching so repeated inputs do not benefit from cached prefixes.
These are starter settings, not optimized deployment claims.

```console
vllm serve Qwen/Qwen3-4B-Instruct-2507 --host 127.0.0.1 --port 8001 --dtype half --max-model-len 8192 --max-num-seqs 1 --gpu-memory-utilization 0.85 --enforce-eager --disable-prefix-caching
```

Some older versions spell the final flag `--no-enable-prefix-caching`; the
notebook detects the installed spelling. Run the client beside the server:

```console
python scripts/benchmark_vllm_latency.py --model Qwen/Qwen3-4B-Instruct-2507 --repeats 20 --warmups 2 --server-label "GPU name; model revision; float16; prefix cache disabled; vLLM version" --output local_testing/gpu_latency/vllm.json
```

The client uses only the Python standard library. It checks `/v1/models`
before generating and uses a fixed synthetic prompt. It never downloads weights.
The default endpoint is `http://127.0.0.1:8001/v1`. Run it on the GPU machine
to minimize network overhead; running it elsewhere includes that network's latency.
Each request opens a connection, so connection setup is included.

For a separately provisioned HTTPS inference endpoint:

```console
python scripts/benchmark_vllm_latency.py --endpoint https://YOUR-ENDPOINT/v1 --allow-remote --model YOUR-SERVED-MODEL --output local_testing/gpu_latency/remote.json
```

Set `SOVEREIGN_VLLM_API_KEY` in your environment when authentication is needed.
Keys are never command-line arguments or report fields. Redirects are rejected.
Remote access is confined to this standalone synthetic benchmark; the
application's endpoint policy is unchanged.

## Run the application's actual workflow

With the application dependencies installed and vLLM reachable on loopback or
an approved internal service name, run from the project root:

```console
python scripts/benchmark_workflow_latency.py --provider vllm --endpoint http://127.0.0.1:8001/v1 --models Qwen/Qwen3-4B-Instruct-2507 --output local_testing/gpu_latency/workflow.json
```

Run both processes on the cloud machine to measure its workflow performance.
An existing SSH connection can alternatively forward port 8001 to the GPU
machine; this measures a cloud deployment through a tunnel and does not
establish local/on-premises or air-gapped performance.

Provider, endpoint, and model overrides affect only this benchmark process and
temporary workflow workspaces. Saved settings/model files are not edited.
Inference fallback and the application response cache are disabled. The script
verifies actual tool results and retains per-call timing plus task-handler time.
It remains a sequential workflow smoke test, not a statistical load test.
Ollama mode and its existing append/retry options remain available.

## Read the results

- `ttft_seconds`: request start to first visible response content, including
  client connection, network, and queue time. Reasoning chunks do not count.
- `total_seconds`: request start through the terminal `[DONE]` event.
- `decode_tokens_per_second_estimate`: `(completion_tokens - 1)` divided by
  the interval from first to last visible content chunk. Stream chunking can
  distort this estimate; missing usage or a single chunk produces `null`.
- `summary`: successful measured samples only, with counts and median/p95.
  Warmups and failed samples are retained separately. Twenty requests provide
  descriptive results, not a reliable tail-latency guarantee.
- Client GPU metadata describes the inference server only when the client runs
  on that machine. Record server GPU, model revision, quantization, vLLM version,
  input/output lengths, concurrency, and cache flags when comparing runs.

The standalone client checks completed, nonempty streams, not mathematical or
diagnostic correctness. The workflow benchmark additionally checks tool outputs.
Neither measures browser rendering; the standalone client also excludes retrieval
and tool execution. Model download, server startup and model loading are excluded.
Exit codes are `0` for successful measured samples, `1` for measured failures,
and `2` for standalone preflight/argument failures. A preflight failure writes
no new measurements; do not reuse an earlier output file as a new run.

References: [vLLM GPU requirements](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/),
[vLLM server options](https://docs.vllm.ai/en/latest/cli/serve/),
[Colab resource availability](https://research.google.com/colaboratory/faq.html).
