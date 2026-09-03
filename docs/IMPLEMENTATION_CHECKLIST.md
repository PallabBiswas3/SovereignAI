# Implementation checklist

| Upgrade phase | Delivered | Verification |
|---|---|---|
| 1 — Real Ollama roles | GENERAL `qwen3:4b-instruct`, VISION `qwen3-vl:4b-instruct`, CODER `qwen2.5-coder:7b`, dynamic four-state readiness, UI | live Ollama status: 3/3 roles ready |
| 2 — Semantic embeddings | local MiniLM provider, hash fallback, provider/dimension persistence, paraphrase benchmark | semantic grounding/retrieval tests |
| 3 — LLM coding | dataset profile, structured code generation, Docker-only execution, max-three repair loop, version/audit records | fake-model repair test; Docker-unavailable safety test |
| 4 — Tool agent | schema-discovered registered tools, bounded decisions/calls/time, governance, audit, source/artifact collection | bounded and unregistered-tool tests; live Qwen tool selection/execution check |
| 5 — Deterministic engineering | metric-specific SOP evidence and Python comparisons retained | inspection end-to-end regression |
| 6 — Structured grounding | semantic/lexical/retrieval/numeric evidence, three statuses, material-claim governance | paraphrase and invented-claim tests |
| 7 — Multi-file tasks | multiple UI uploads and type-aware PDF/image/SOP/DOCX/CSV/XLSX evidence | processor provenance test |
| 8 — Multi-artifact package | run-linked DOCX, XLSX, PPTX | all formats opened successfully in test |
| 9 — Approval execution | persisted exact proposal, revalidation, controlled execution/result/audit; destructive override blocked | approval lifecycle tests |
| 10 — Live events | asynchronous task start, persisted structured SSE events, live frontend | SSE lifecycle test |
| 11 — Sovereignty proof | strict endpoint policy, internal Compose network, pinned internal Ollama, active egress/Ollama verifier, scoped docs | policy/Compose tests; Compose config validation |
| 12 — Evaluation | 20 routing + 20 RAG + 20 governance + 10 agent cases and expanded metrics | offline benchmark test |

Final verification on 1 September 2026:

- Python compile: passed.
- Backend: 37 tests passed.
- Frontend TypeScript: passed.
- Frontend production build: passed.
- Ollama: reachable; both unique model tags installed; all three logical roles `READY`.
- Benchmark: routing accuracy 0.95; semantic RAG Recall@1/Recall@3/MRR 1.0; PII and injection F1 1.0; see evaluation API for full case-level output.
- Docker: engine stopped on this workstation, so a live container execution success is not claimed. Safety behavior and bounded repair logic are tested.
