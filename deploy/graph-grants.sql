REVOKE ALL ON SCHEMA public FROM PUBLIC;
GRANT USAGE ON SCHEMA public TO graph_service;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO graph_service;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO graph_service;
-- No anonymous role or browser database credential is configured. Existing
-- GraphRAG authorization-before-scoring remains the principal boundary.
