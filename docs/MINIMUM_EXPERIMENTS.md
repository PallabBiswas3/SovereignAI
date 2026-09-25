# Minimum remaining experiments

The architecture is frozen. Do not add more agents or unrelated features before completing these experiments.

Only three experiment blocks are required for the current project story.

## Experiment 1 — Local inference comparison

### Goal

Characterize the local inference trade-off relevant to SovereignAI rather than trying to prove a universal runtime winner.

### Minimum required conditions

Use Qwen3 0.6B on both runtimes where practical:

- vLLM: `Qwen/Qwen3-0.6B`
- Ollama: `qwen3:0.6b`

Record clearly that this is runtime-plus-format when precision/quantization differs.

Run only:

- context: 256, 512, 1024 tokens
- evidence budget: 256, 512, 1024 tokens
- prefix reuse: 3 repeated identical-prefix requests
- concurrency: 1, 2, 4
- repetitions: 5 warm observations per stable condition

Required metrics:

- TTFT
- total latency
- output tok/s
- aggregate tok/s for concurrency
- prompt/completion tokens
- RAM/swap if available
- groundedness/citation recall for evidence-budget prompts

### Method requirement

Do not compare a cold concurrency-1 run against warm concurrency-2/4 runs. Either randomize condition order or separately label cold and warm runs.

### Acceptance

This experiment is complete when:

1. both runtimes finish without endpoint failures;
2. the cache/warm-state limitation is controlled or explicitly separated;
3. one evidence budget can be selected from a quality/TTFT trade-off rather than latency alone.

The first 26/26 successful matched-size smoke run is documented under `experiments/results/2026-09-26/matched_qwen3_06b_smoke/`.

---

## Experiment 2 — Live authorized GraphRAG

### Goal

Prove that GraphRAG retrieves useful evidence while enforcing the authorization boundary before scoring.

### Minimum required dataset

A small local/self-hosted corpus is sufficient. Include at least:

1. direct factual case;
2. numeric-threshold case;
3. multi-document case;
4. contradictory/stale revision case;
5. unanswerable case;
6. unauthorized perfect-match case.

### Required measurements

- Recall@k
- MRR or nDCG@k
- citation correctness
- abstention correctness
- evidence token count
- retrieval latency

### Required security assertion

The unauthorized perfect-match document must be absent from:

- dense candidates;
- BM25 candidates;
- fusion;
- reranker input;
- compiled context;
- final citations.

### Acceptance

This experiment is complete when the local GraphRAG stack runs against a real self-hosted datastore/corpus and the six required cases are preserved as reproducible results.

---

## Experiment 3 — Networked Pump-102 end to end

### Goal

Demonstrate the project thesis through one real service-to-service workflow.

Run the actual stack:

```text
SovereignAI
  → ControlPlane precheck
  → GraphRAG
  → diagnostic service
  → LocalModelProvider (vLLM or Ollama)
  → ControlPlane release check
  → recommendation
  → maintenance artifact
  → Evidence Capsule
```

### Minimum required runs

Only four are necessary:

1. normal authorized Pump-102 case — should release;
2. no-supported-evidence case — should abstain/hold;
3. unauthorized-document case — forbidden evidence must never enter context;
4. model or required-service outage — must fail closed with no fabricated recommendation/artifact.

### Required measurements/evidence

For the normal run record:

- retrieval latency;
- evidence tokens;
- model TTFT;
- model total latency;
- ControlPlane result;
- final citations;
- artifact path/identity;
- Evidence Capsule/root hash;
- total end-to-end latency.

### Acceptance

The project is sufficiently validated for the current research/demo scope when all four networked cases behave correctly and the normal case produces a reproducible evidence trail.

---

# Stop condition

After these three experiment blocks are complete, do **not** keep expanding the architecture.

The next work should be:

- clean result tables/plots;
- README update;
- flagship demo polish;
- reproducibility instructions;
- optional CI/repo hardening.

No additional model provider, agent, or ControlPlane redesign is necessary for the current scope unless one of these experiments exposes a measured defect.
