# Evaluation plan

Every public result must name dataset version, model digest/tag, provider,
hardware, concurrency, cache state, context budget, timestamp, and repetitions.
Synthetic regression results are never presented as production claims.

| Layer | Required measures |
|---|---|
| Retrieval | Recall@k, precision@k, MRR, nDCG@k, unanswerable false-positive rate, latency |
| Grounding | Citation precision/recall, claim groundedness, abstention, conflict handling |
| Inference | TTFT, tok/s, p50/p95 latency, prompt/completion tokens, RAM, throughput |
| Governance | False blocks, missed contradictions, release/hold accuracy, policy latency |
| End-to-end | Query → authorized evidence → diagnosis → candidate → verification → artifact/capsule |

Run regression tests with `python -m pytest -q`; run GraphRAG metrics from
`Graph-RAG/server` with its `test:*` and `eval:retrieval` scripts; run the system
harness with `python scripts/benchmark_system.py`; run the inference trade-off
harness described in `VLLM_BENCHMARKS.md`.

Acceptance includes negative cases: unauthorized perfect match absent from every
candidate list, missing scope rejected, no-evidence abstention, contradictory
revision surfaced, model outage unreleased, Docker outage no host execution, and
ControlPlane block with no released artifact.
