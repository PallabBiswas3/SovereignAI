# Preliminary live check — not the final controlled comparison

The original raw JSON/CSV and summary are preserved unchanged: 40 successful
requests, comprising 20 warm-ups and 20 warm measurements, with zero failures.
`experiment_complete=false` is intentional.

Ollama: qwen3:0.6b, GGUF Q4_K_M, declared digest
`7df6b6e09427a769808717c0a93cadc4ae99ed4eb8bf5ca557c90846becea435`.
vLLM: Qwen/Qwen3-0.6B, BF16 under WSL. This compares runtime plus weight format
and host environment, not isolated equivalent-precision runtime performance.

The v2 harness used whitespace-word budgets despite `target_tokens` naming.
It did not verify cold states, capture process memory, verify full runtime
identity, or cover the randomized prefix/concurrency matrix. Quality was a
synthetic lexical/citation proxy without human adjudication. Warm-up requests
are excluded from measured summaries; these observations establish preliminary
service/harness operation only. Do not merge them into a v3 final experiment.
