# GraphRAG evaluation

GraphRAG is evaluated as an authorized evidence system, not just vector search.
Its order is ACL filter → dense/BM25 → RRF → local reranking → revision/conflict
analysis → evidence compression/context budget → verification.

The TypeScript package reports node/chunk Recall@k, precision@k, MRR, nDCG@k,
answerable hit rate, unanswerable false-positive rate, and latency. Claim
evaluation records supported/contradicted/insufficient labels, abstention, and
citation provenance.

Minimum strata are single-hop, multi-hop, global, contradictory, unanswerable,
unauthorized perfect match, stale revision, and insufficient evidence.
Authorization tests must prove excluded content never reaches scoring, fusion,
reranking, verification, logs, or compiled context.

```powershell
Set-Location Graph-RAG\server
npm run test:retrieval-core
npm run test:verification-policy
npm run test:claim-verifier
npm run eval:retrieval -- --dataset evaluation/benchmark.example.json
```

The Supabase server credential bypasses RLS, so the internal API’s full-scope
filter is a critical boundary. Strict-sovereign deployment should use a
self-hosted Supabase/Postgres instance and retain RLS as defense in depth.
