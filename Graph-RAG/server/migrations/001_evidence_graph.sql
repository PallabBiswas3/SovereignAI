-- Phase 3: evidence-grounded graph schema
-- Apply this migration in the Supabase SQL editor before enabling evidence ingestion.

create extension if not exists pgcrypto;

create table if not exists documents (
  id text primary key,
  file_name text,
  total_pages integer,
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create table if not exists claims (
  id uuid primary key default gen_random_uuid(),
  claim_text text not null,
  normalized_text text not null,
  source_doc_id text references documents(id) on delete cascade,
  page_start integer,
  page_end integer,
  extraction_confidence double precision not null default 1.0,
  polarity text not null default 'affirmed' check (polarity in ('affirmed','negated','uncertain')),
  metadata jsonb not null default '{}'::jsonb,
  created_at timestamptz not null default now()
);

create index if not exists claims_source_doc_idx on claims(source_doc_id);
create index if not exists claims_normalized_text_idx on claims(normalized_text);

create table if not exists claim_evidence (
  claim_id uuid not null references claims(id) on delete cascade,
  chunk_id uuid not null references chunks(id) on delete cascade,
  relation text not null default 'SUPPORTED_BY' check (relation in ('SUPPORTED_BY','CONTRADICTED_BY')),
  confidence double precision not null default 1.0,
  primary key (claim_id, chunk_id, relation)
);

create index if not exists claim_evidence_chunk_idx on claim_evidence(chunk_id);

create table if not exists claim_entities (
  claim_id uuid not null references claims(id) on delete cascade,
  entity_id text not null references nodes(id) on delete cascade,
  confidence double precision not null default 1.0,
  primary key (claim_id, entity_id)
);

create index if not exists claim_entities_entity_idx on claim_entities(entity_id);

create table if not exists claim_relations (
  source_claim_id uuid not null references claims(id) on delete cascade,
  target_claim_id uuid not null references claims(id) on delete cascade,
  relation text not null check (relation in ('SUPPORTS','CONTRADICTS')),
  confidence double precision not null default 1.0,
  reason text,
  primary key (source_claim_id, target_claim_id, relation),
  check (source_claim_id <> target_claim_id)
);

create index if not exists claim_relations_source_idx on claim_relations(source_claim_id);
create index if not exists claim_relations_target_idx on claim_relations(target_claim_id);
