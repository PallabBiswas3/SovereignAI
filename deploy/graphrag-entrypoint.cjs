// Load only this service's mounted credentials; never log their values.
const { readFileSync } = require('node:fs');
process.env.GRAPHRAG_SERVICE_TOKEN = readFileSync('/run/secrets/graphrag_token', 'utf8').trim();
process.env.SUPABASE_SECRET_KEY = readFileSync('/run/secrets/graph_service_jwt', 'utf8').trim();
if (process.env.GRAPHRAG_SERVICE_TOKEN.length < 32) throw Error('Missing GraphRAG credential');
require('./dist/server.js');
