# SovereignAI system benchmark

- Generated: `2026-09-18T21:20:27.800573+00:00`
- Base URL: `http://127.0.0.1:8000`
- Runs per scenario: `3`

## Health

| Probe | OK | HTTP | Latency (ms) |
|---|---:|---:|---:|
| backend | True | 200 | 28.097 |
| integrations | True | 200 | 77.381 |

## Latency and reliability

| Scenario | Success | p50 total (s) | p95 total (s) | p50 TTFT (s) | Avg tok/s |
|---|---:|---:|---:|---:|---:|
| chat_sync | 3/3 | 4.075 | 11.485 | — | — |
| task_general_stream | 3/3 | 5.544 | 8.858 | 0.489 | 10.570 |
| task_authorized_stream | 3/3 | 185.091 | 186.191 | 65.565 | 4.821 |
| integration_graph_controlplane | 3/3 | 9.290 | 34.922 | — | — |
| integration_full_industrial | 0/3 | — | — | — | — |

## Integrated component timing

### integration_graph_controlplane

| Component | p50 (ms) | p95 (ms) |
|---|---:|---:|
| evidence | 6344.000 | 26073.800 |
| precheck | 47.000 | 47.000 |
| release_check | 2875.000 | 8752.900 |
| total | 9266.000 | 34873.700 |


## Failures / degraded runs

- `integration_full_industrial` run 1: HTTP=503; {"detail": {"code": "INTEGRATION_SERVICE_UNAVAILABLE", "service": "integration"}}
- `integration_full_industrial` run 2: HTTP=503; {"detail": {"code": "INTEGRATION_SERVICE_UNAVAILABLE", "service": "integration"}}
- `integration_full_industrial` run 3: HTTP=503; {"detail": {"code": "INTEGRATION_SERVICE_UNAVAILABLE", "service": "integration"}}

## Interpretation

- `chat_sync` measures the simple non-streaming chat API.
- `task_general_stream` measures user-visible TTFT from task admission to the first `model_token` SSE event.
- `task_authorized_stream` exercises the authorized/RAG path and records TTFT when model generation occurs.
- `integration_graph_controlplane` exercises ControlPlane precheck → GraphRAG → ControlPlane release.
- `integration_full_industrial` additionally sends deterministic synthetic process data through the Time-Series Diagnostic Agent.
- The harness does not stop services automatically. Run it again while intentionally stopping one service to capture degraded/fail-closed behavior.
