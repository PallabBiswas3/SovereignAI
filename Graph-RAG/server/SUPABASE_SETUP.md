# Supabase backend setup

The Express backend performs privileged graph/evidence writes, so it must use a **server-only Supabase credential**.

## Required server environment

Copy `server/.env.example` to `server/.env` and set:

```env
OLLAMA_URL=http://127.0.0.1:11434
OLLAMA_GENERATION_MODEL=qwen3:4b-instruct
OLLAMA_EMBEDDING_MODEL=nomic-embed-text
GRAPHRAG_DENSE_RETRIEVAL_ENABLED=false
PORT=3000
SUPABASE_URL=https://<project-ref>.supabase.co
SUPABASE_SECRET_KEY=sb_secret_...
```

`SUPABASE_SECRET_KEY` is preferred. `SUPABASE_SERVICE_ROLE_KEY` is accepted only as a legacy fallback.

## Security rules

- Never use `SUPABASE_ANON_KEY` for backend writes.
- Never expose `SUPABASE_SECRET_KEY` or `SUPABASE_SERVICE_ROLE_KEY` through Vite/client environment variables.
- Never commit `server/.env`.
- Public/browser clients should use a Supabase publishable key and rely on RLS.
- The server secret bypasses RLS, so API endpoints that use it must enforce the application's own authorization rules before exposing privileged operations publicly.

## Evidence graph

The production Supabase project must include the Phase 3 evidence schema from:

`server/migrations/001_evidence_graph.sql`

Required tables:

- `documents`
- `claims`
- `claim_evidence`
- `claim_entities`
- `claim_relations`

The retrieval RPCs `match_nodes`, `match_chunks`, and `expand_graph` are also required.

Embeddings are generated locally through Ollama and stored as `vector(768)`. After changing the embedding model, clear and re-ingest the corpus so query and document vectors come from the same model.

Keep dense retrieval disabled for a legacy corpus. Back up the database, run `npm run reindex:local --workspace=server`, and only then set `GRAPHRAG_DENSE_RETRIEVAL_ENABLED=true`.
