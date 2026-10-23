#!/bin/sh
set -eu
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" \
    -v graph_password="$(cat /run/secrets/graph_db_password)" <<'SQL'
CREATE ROLE graph_service NOLOGIN;
CREATE ROLE graph_auth LOGIN NOINHERIT PASSWORD :'graph_password';
GRANT graph_service TO graph_auth;
CREATE EXTENSION IF NOT EXISTS pgcrypto;
CREATE TABLE nodes (
    id text PRIMARY KEY, label text, type text, description text,
    properties jsonb NOT NULL DEFAULT '{}', embedding text,
    source_doc_id text, confidence double precision DEFAULT 1,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE TABLE links (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    source text REFERENCES nodes(id) ON DELETE CASCADE,
    target text REFERENCES nodes(id) ON DELETE CASCADE,
    relationship text, type text, reason text, weight double precision DEFAULT 1
);
CREATE TABLE chunks (
    id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    node_id text REFERENCES nodes(id) ON DELETE CASCADE,
    content text NOT NULL, chunk_index integer DEFAULT 0,
    source_url text, metadata jsonb NOT NULL DEFAULT '{}', embedding text,
    created_at timestamptz NOT NULL DEFAULT now()
);
SQL
