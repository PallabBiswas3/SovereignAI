const { test } = require('node:test');
const assert = require('node:assert/strict');
const { createHash } = require('node:crypto');
const { serviceAuthentication, validatedScope } = require('../dist/serviceAuth');

function scope(updates = {}) {
  const s = { organization_id: 'org-a', user_id: 'user-a', workspace_ids: ['workspace-a'],
    department_ids: ['engineering'], roles: ['ENGINEER'], clearance: 2, cross_department: false, ...updates };
  s.fingerprint = createHash('sha256').update([s.organization_id, s.user_id, String(s.clearance),
    s.cross_department ? '1' : '0', ...[...s.department_ids].sort(), ...[...s.workspace_ids].sort(),
    ...[...s.roles].sort()].join('\x1f')).digest('hex');
  return s;
}

test('every API route requires its designated credential before handlers', () => {
  process.env.GRAPHRAG_SERVICE_TOKEN = 's'.repeat(32);
  process.env.GRAPHRAG_ADMIN_TOKEN = 'a'.repeat(32);
  function call(path, token) {
    let status = 200, reached = false;
    const res = { status(code) { status = code; return this; }, json() {} };
    serviceAuthentication({ path, get: () => token && `Bearer ${token}` }, res, () => { reached = true; });
    return { status, reached };
  }
  for (const path of ['/integration/retrieve', '/graph', '/clear', '/ingest', '/evidence', '/evaluate']) {
    assert.deepEqual(call(path), { status: 401, reached: false });
    assert.equal(call(path, 'wrong').reached, false);
  }
  assert.equal(call('/integration/retrieve', 's'.repeat(32)).reached, true);
  assert.equal(call('/integration/retrieve', 'a'.repeat(32)).status, 401);
  for (const path of ['/graph', '/clear', '/ingest', '/evidence', '/evaluate']) {
    assert.equal(call(path, 's'.repeat(32)).status, 401);
    assert.equal(call(path, 'a'.repeat(32)).reached, true);
  }
  delete process.env.GRAPHRAG_ADMIN_TOKEN;
  assert.equal(call('/clear', 's'.repeat(32)).status, 403);
  process.env.GRAPHRAG_ADMIN_TOKEN = process.env.GRAPHRAG_SERVICE_TOKEN;
  assert.equal(call('/clear', 's'.repeat(32)).status, 403);
  delete process.env.GRAPHRAG_SERVICE_TOKEN;
  assert.equal(call('/integration/retrieve', 's'.repeat(32)).status, 503);
});

test('bounded explicit scopes reject wildcard, malformed, and forged authority', () => {
  assert.deepEqual(validatedScope(scope()), scope());
  for (const update of [{ organization_id: '*' }, { workspace_ids: ['*'] }, { department_ids: ['*'] },
    { user_id: '*' }, { roles: ['ROOT'] }, { roles: [] }, { workspace_ids: [] },
    { clearance: -1 }, { clearance: 4 }, { clearance: NaN }, { cross_department: 'true' },
    { workspace_ids: ['workspace-a', 'workspace-a'] }]) assert.equal(validatedScope(scope(update)), null);
  assert.equal(validatedScope({ ...scope(), organization_id: 'org-b' }), null);
  assert.equal(validatedScope(null), null);
  assert.equal(validatedScope({}), null);
});

test('HTTP routes block unauthenticated reads and mutations before database or retrieval', async () => {
  const path = require('node:path');
  const stub = (name, exports) => { require.cache[require.resolve(name)] = { exports }; };
  stub('../dist/supabase', { supabase: { from() { throw Error('database must not be called'); } } });
  let retrieved;
  stub('../dist/agent/adaptiveAgent', { runAdaptiveAgent: async (query, options) => {
    retrieved = options.authorizationScope;
    return { decision: 'abstain', state: { chunks: [], trace: [] } };
  } });
  const { app } = require('../dist/server');
  process.env.GRAPHRAG_SERVICE_TOKEN = 's'.repeat(32);
  delete process.env.GRAPHRAG_ADMIN_TOKEN;
  const server = app.listen(0, '127.0.0.1');
  await new Promise(resolve => server.once('listening', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  try {
    for (const [method, route] of [['GET', '/api/graph'], ['POST', '/api/clear'],
      ['POST', '/api/ingest'], ['GET', '/api/evidence'], ['POST', '/api/integration/retrieve']]) {
      const response = await fetch(base + route, { method });
      assert.ok([401, 403].includes(response.status));
    }
    const headers = { Authorization: 'Bearer ' + 's'.repeat(32), 'Content-Type': 'application/json' };
    const forged = await fetch(base + '/api/integration/retrieve', { method: 'POST', headers,
      body: JSON.stringify({ query: 'synthetic', authorization_scope: scope({ workspace_ids: ['*'] }) }) });
    assert.equal(forged.status, 400);
    assert.equal(retrieved, undefined);
    const allowed = await fetch(base + '/api/integration/retrieve', { method: 'POST', headers,
      body: JSON.stringify({ query: 'synthetic', authorization_scope: scope() }) });
    assert.equal(allowed.status, 200);
    assert.equal((await allowed.json()).status, 'abstained');
    assert.deepEqual(retrieved, scope());
  } finally { await new Promise(resolve => server.close(resolve)); }
});
