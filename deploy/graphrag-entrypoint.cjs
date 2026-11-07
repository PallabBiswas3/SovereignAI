// Load only this service's mounted credentials; never log their values.
const { readFileSync } = require('node:fs');
process.env.GRAPHRAG_SERVICE_TOKEN = readFileSync('/run/secrets/graphrag_token', 'utf8').trim();
process.env.SUPABASE_SECRET_KEY = readFileSync('/run/secrets/graph_service_jwt', 'utf8').trim();
if (process.env.GRAPHRAG_SERVICE_TOKEN.length < 32) throw Error('Missing GraphRAG credential');
// server.js deliberately listens only when it is the main module. This wrapper
// loads the secrets first, so it must start the exported app explicitly.
const { app } = require('./dist/server.js');
const port = Number(process.env.PORT || 3100);
const host = process.env.HOST || '127.0.0.1';
app.listen(port, host, () => console.log(`GraphRAG listening on ${host}:${port}`));
