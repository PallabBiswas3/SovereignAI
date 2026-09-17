# Phase 6 evaluation modes

The normal `/api/chat` endpoint remains unchanged. Ablation runs use the streaming endpoint:

`POST /api/evidence/evaluate`

Request body:

```json
{
  "query": "GRPO removes the value model, therefore it no longer needs a baseline. Is this correct?",
  "mode": "fixed"
}
```

Supported modes:

- `fixed`: hybrid dense + BM25/RRF retrieval -> fixed depth-2 graph expansion -> grounded synthesis. No adaptive routing, evidence-claim verifier, verification retry, or verifier-driven abstention.
- `adaptive`: current adaptive agent (`hybrid_retrieve`, optional `graph_expand`, `fetch_evidence`, optional `requery`, `finalize`/`abstain`) -> grounded synthesis. Structured claims are unverified navigation aids. No claim verifier.
- `adaptive_verified`: adaptive agent -> claim verification -> optional bounded verification requery -> verified synthesis or abstention.

The final SSE metadata event includes:

```json
{
  "mode": "adaptive_verified",
  "evaluation": {
    "abstained": false,
    "timings": {
      "agentMs": 0,
      "verificationMs": 0,
      "synthesisMs": 0,
      "totalMs": 0
    },
    "counts": {
      "nodes": 0,
      "chunks": 0,
      "expandedEdges": 0,
      "claims": 0,
      "supported": 0,
      "contradicted": 0,
      "insufficient": 0
    },
    "toolCalls": [],
    "verification": {
      "available": true,
      "retried": false,
      "decision": "continue",
      "calibratedScore": 1,
      "supported": 0,
      "contradicted": 0,
      "insufficient": 0
    }
  }
}
```

For a fair ablation, run the exact same question against all three modes and keep the ingested corpus, retrieval index, Ollama model, and synthesis instructions fixed. The main experimental differences should be adaptive routing and claim verification.
