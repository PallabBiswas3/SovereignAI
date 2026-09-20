# SovereignAI overall-system benchmark

Run this benchmark on the same machine that runs SovereignAI. It measures the real local stack; GitHub CI cannot reproduce laptop CPU/RAM/Ollama latency.

## What it measures

1. Backend and integrated-service health.
2. Plain `/api/chat` latency.
3. User-visible time-to-first-token (TTFT) and total latency through `/api/tasks/start` + SSE.
4. Authorized/RAG task latency.
5. ControlPlane precheck → GraphRAG → ControlPlane release latency.
6. Optional full GraphRAG + Time-Series Diagnostic Agent + ControlPlane latency using deterministic synthetic process data.
7. Host CPU/RAM snapshots and per-component integration timings.

The benchmark does not change GraphRAG internals and does not automatically stop services.

## Run

From the repository root, with the local workspace already running:

```powershell
python scripts/benchmark_system.py --runs 3
```

To include the Time-Series Diagnostic Agent:

```powershell
python scripts/benchmark_system.py --runs 3 --include-diagnostic
```

For a more stable latency distribution:

```powershell
python scripts/benchmark_system.py --runs 10 --include-diagnostic
```

### Local-auth mode

Either provide an existing session/CSRF pair:

```powershell
python scripts/benchmark_system.py --cookie "sovereign_session=..." --csrf-token "..."
```

or let the benchmark log in:

```powershell
python scripts/benchmark_system.py --email "user@example.com" --password "..."
```

Do not commit credentials or generated session values.

## Output

The default output directory is `reports/system_benchmark/`:

- `system_benchmark.json`: full machine-readable health/results.
- `system_benchmark.csv`: one row per request.
- `system_benchmark.md`: compact p50/p95 scorecard.

Important fields include:

- total request latency
- TTFT
- local model duration
- tokens/second
- warm/cold status
- service status
- ControlPlane release status
- integration component timings
- CPU and RAM snapshots
- failures/degraded runs

## Failure-mode validation

The harness deliberately does not kill services. To validate safe degradation, run the normal benchmark once, then intentionally stop exactly one local service and rerun the relevant scenario.

Examples:

1. Stop GraphRAG, rerun the integration benchmark, and confirm the report records an unavailable GraphRAG path rather than fabricated evidence.
2. Stop the diagnostic service and rerun with `--include-diagnostic`.
3. Stop ControlPlane and confirm the integrated workflow fails closed.
4. Stop Ollama and confirm the model path reports explicit unavailability/fallback rather than a fabricated model answer.

Restore each service before testing the next failure.

## Interpretation

Compare scenarios separately rather than quoting one “SovereignAI latency” number:

- `chat_sync`: simple non-streaming local chat.
- `task_general_stream`: user-visible general-chat TTFT.
- `task_authorized_stream`: authorized/RAG path.
- `integration_graph_controlplane`: evidence + policy path.
- `integration_full_industrial`: GraphRAG + diagnostic agent + ControlPlane.

Optimize the slowest measured component only after the first baseline is captured.
