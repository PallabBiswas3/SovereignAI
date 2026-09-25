# vLLM validation closeout

This concise plan is retained from the reconciled `integrate-vllm-provider`
branch. Its implementation was not merged wholesale. The shared provider
factory, payload coverage, endpoint normalization, scheduler admission,
lifecycle state, queue-wait metrics, cancellation, and timeout handling were
selectively ported into the newer provider.

Before publishing a vLLM-versus-Ollama comparison:

1. Serve the same checkpoint and comparable precision/quantization in both.
2. Record hardware, model digest, context/output caps, cache state, repetitions,
   concurrency order, RAM, TTFT, total latency, and output throughput.
3. Run cold and warm observations separately.
4. Exercise cancellation, timeout, malformed output, context overflow, and
   mid-stream server loss.
5. Run GraphRAG evidence budgets together with citation and groundedness scores.
6. Run the networked Pump-102 path and prove unauthorized evidence never enters
   scoring or the model context.

Historical measurements from the deleted branch are preserved only as prior
observations: on its documented CPU setup, a unique long prompt reported about
53.9 s TTFT and an immediate repeated-prefix request about 2.5 s TTFT. They are
not mixed with the current run because the runtime/model conditions differ.

Current measured results and limitations are published under
`experiments/results/2026-09-26/`.
