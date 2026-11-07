# Remote GPU inference setup

**For your Hugging Face account:** use [Hugging Face setup](HUGGING_FACE_SETUP.md).
It creates the private ZeroGPU model and connects your app; no benchmark is required.
The manual vLLM instructions below are an alternative.

Status: integration prepared and tested with simulated HTTP responses. No GPU
host is provisioned, no credentials are configured, and no latency improvement
has been measured. Local Ollama remains the default.

```text
Browser -> local backend / retrieval / governance -> HTTPS GPU API -> answer
```

The `remote_vllm` provider moves text generation to an authenticated vLLM server.
Prompts, system instructions and any retrieved context included in a generation
request leave the machine. Stored documents, retrieval, embeddings, tools and
ControlPlane stay in their existing locations. Vision uses its existing local
adapter. All text roles use the single configured remote model; the local role
registry still describes local models. Choose a model suitable for your tasks.

Local Ollama and local vLLM keep their network restrictions. Remote mode requires
an explicit opt-in, an HTTPS base URL, a model ID and an API key. It does not
follow redirects, use environment HTTP proxies, retry failed generations, or
silently replace failed remote answers with fallback output. Certificate
verification stays enabled. The network monitor labels the deployment
`remote-inference-enabled`, and air-gap verification should fail in this mode.

## 1. Choose a host

Use a Linux NVIDIA GPU host on which you are allowed to serve an inference API.
The setup below needs a domain name with DNS pointing to the host, HTTPS ingress
and enough VRAM for your model plus context. A quantized GGUF model on Ollama and
FP16 weights on vLLM have different memory needs and may generate different text.
Start with a modest context length and one request at a time.

No cloud resources, paid subscriptions or model downloads are created by these
repository changes. A free notebook GPU allocation does not automatically permit
external API hosting. Colab's free tier restricts bypassing the notebook UI;
Kaggle external API hosting permission has not been established here. The existing
`notebooks/vllm_gpu_latency.ipynb` can still measure generation inside a notebook;
that does not measure the laptop-to-GPU path.

References: [Colab limits](https://research.google.com/colaboratory/faq.html),
[vLLM GPU installation](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/),
[vLLM serving](https://docs.vllm.ai/en/latest/serving/online_serving/),
[vLLM security](https://docs.vllm.ai/en/latest/usage/security/).

## 2. Start the GPU server

On the GPU host, install the vLLM version compatible with its driver/GPU in an
isolated environment following the official installation guide. Copy
`scripts/serve_remote_vllm.py` and `deploy/Caddyfile.remote-gpu` to that host.
The launcher uses the existing Qwen3 4B model family as its starter model:

```bash
# Linux GPU host; run in the vLLM environment.
# Generate a secret into the environment without printing it or typing it into history.
export SOVEREIGN_REMOTE_VLLM_API_KEY="$(python -c 'import secrets; print(secrets.token_urlsafe(32))')"
python serve_remote_vllm.py --model Qwen/Qwen3-4B-Instruct-2507 --context-length 8192
```

The launcher binds port 8001 to `127.0.0.1`, passes the key through `VLLM_API_KEY`,
disables request logging and usage statistics, and leaves the model resident
while the server runs. Add `--revision <commit-sha>` to pin weights. Record the
installed vLLM version, model revision, GPU and server flags with benchmark results.
The first launch downloads weights and loads the model; GPU execution has not
been validated on a real host by this change.

Run Caddy **natively on the same GPU host**, in a separate terminal/service:

```bash
export SOVEREIGN_GPU_HOST=gpu.your-domain.com
caddy run --config Caddyfile.remote-gpu --adapter caddyfile
```

Configure the host firewall/provider ingress for TCP 80/443 and your administrative
access; keep port 8001 private. Caddy obtains HTTPS certificates for the domain.
The supplied configuration exposes only `/v1/models` and
`/v1/chat/completions`, both authenticated by vLLM. It leaves SSE streaming
unbuffered through Caddy's automatic event-stream handling and preserves client
disconnect cancellation. Other vLLM routes, including administrative routes,
are not published. See [Caddy reverse proxy](https://caddyserver.com/docs/caddyfile/directives/reverse_proxy).

Use the host's secret manager or another secure transfer method to give the local
backend the same API key. Never paste it into chat, commit it, or put it in a URL.

## 3. Connect a native backend

Run from `1_SIH`, with the project Python environment. Put these values in your
ignored root `.env` (or in the backend process environment):

```dotenv
SOVEREIGN_LLM_PROVIDER=remote_vllm
SOVEREIGN_REMOTE_INFERENCE_ENABLED=true
SOVEREIGN_REMOTE_VLLM_URL=https://gpu.your-domain.com/v1
SOVEREIGN_REMOTE_VLLM_MODEL=Qwen/Qwen3-4B-Instruct-2507
SOVEREIGN_REMOTE_VLLM_API_KEY=<the actual GPU host secret>
SOVEREIGN_VLLM_ENABLE_THINKING=false
SOVEREIGN_VLLM_MAX_TOKENS=1280
SOVEREIGN_MODEL_GENERATION_TIMEOUT_SECONDS=120
```

`SOVEREIGN_REMOTE_VLLM_API_KEY_FILE` can instead point to a protected text file
containing the key. The `/v1` suffix is added automatically when omitted. Use the
exact model ID served by `/v1/models`. Restart the backend to reload settings.
The backend calls the endpoint directly; browser CORS on the GPU server is not
needed. Existing governance/authorization still apply before response release.

Inspect `/api/models/status` -> `active_text_provider` and the authorized
`/api/monitor/network` view. `/ready` checks the remote model listing without
generating. A listed model does not prove it is warm; runtime warm status remains
unknown until the host supplies trustworthy measurements. The backend readiness
response also depends on the existing local services, identity and storage.

Return to local generation with `SOVEREIGN_LLM_PROVIDER=ollama` and
`SOVEREIGN_REMOTE_INFERENCE_ENABLED=false`, then restart the backend.

## 4. Optional Docker connection

The default production Compose networks intentionally block outbound traffic.
After provisioning the existing stack as documented in `DOCKER_DEPLOYMENT.md`,
add the remote URL and model above to ignored `deploy/.env`, and save the key in
`deploy/secrets/remote_vllm_api_key` (already covered by `.gitignore`). Then:

```powershell
docker compose --env-file deploy/.env -f compose.yaml -f compose.remote-inference.yaml config --quiet
docker compose --env-file deploy/.env -f compose.yaml -f compose.remote-inference.yaml up -d --build backend frontend
```

The override mounts the API key as a Docker secret and adds an outbound bridge
to the backend only. It permits general backend egress at the Docker network
layer; it is not an endpoint-specific firewall. Other application containers
retain their existing internal networks. Ollama is still required for GraphRAG
embeddings and the existing local deployment smoke checks. Do not expose the
GPU server through the local application's gateway.

The existing deployment smoke command accepts the same override:

```powershell
python scripts/docker_smoke.py --file compose.yaml --file compose.remote-inference.yaml --inference
```

That command performs existing synthetic audit checks, generates a short answer
using the remote provider, and checks a local embedding. It does not measure
latency. To return to the isolated topology, recreate backend/frontend with the
default Compose files and the default local provider.

## 5. Measure the laptop-to-GPU improvement

Run the comparison from the laptop. This standalone benchmark reads **process
environment variables**, not `.env`. Put the key in the shell without echoing it:

```powershell
$gpuCredential = Read-Host 'GPU API key' -AsSecureString
$env:SOVEREIGN_REMOTE_VLLM_API_KEY = [System.Net.NetworkCredential]::new('', $gpuCredential).Password
python scripts/benchmark_remote_inference.py --local-model qwen3:4b-instruct --remote-endpoint https://gpu.your-domain.com/v1 --remote-model Qwen/Qwen3-4B-Instruct-2507 --local-label 'Record CPU/GPU and GGUF quantization' --remote-label 'Record GPU, FP16, revision and vLLM version'
Remove-Item Env:SOVEREIGN_REMOTE_VLLM_API_KEY
```

Before a host is available, record just the local baseline:

```powershell
python scripts/benchmark_remote_inference.py --local-only --output workspace/artifacts/local-inference-baseline.json
```

Default comparison output: `workspace/artifacts/inference-comparison.json`.
It records two excluded warmups and twenty measured requests per runtime,
interleaved at concurrency one, with the same synthetic prompt, temperature,
thinking setting and output token budget. Each connection includes client
network/TLS overhead. No retrieved documents or generated answer text are saved.

Compare median/p95 time to first visible token and total response time, output
token counts and failures. Median speedup above 1 means the remote deployment
was faster in that sample. Failed requests are retained and excluded from latency
percentiles; missing token counts stay unknown. Different weights/quantization,
different output lengths and prefix caches are confounders. This is an inference
comparison, not proof of full workflow latency or answer quality. Startup/model
download is excluded; session restarts can still cause cold measured requests.

No live remote measurements exist until a host is connected and this benchmark
has actually completed. Protocol tests are not GPU performance evidence.
