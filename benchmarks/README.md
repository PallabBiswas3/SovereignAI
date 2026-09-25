# Benchmark suite

This directory contains executable measurement harnesses, not claimed results.

- `inference_tradeoff.py`: context length, evidence budget, prefix-cache repetition,
  concurrency 1/2/4, and same-workload Ollama/vLLM comparisons.
- `../scripts/benchmark_system.py`: end-to-end API, streaming TTFT, component latency,
  throughput, host RAM, and fail-closed service behavior.
- `../Graph-RAG/server/evaluation`: Recall@k, precision@k, MRR, nDCG@k,
  unanswerable false-positive rate, and retrieval latency.

Results belong under `experiments/results/` and must include runtime/model identity.
