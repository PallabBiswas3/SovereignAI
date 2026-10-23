// Read-only check against an isolated deployment network; never prints tokens.
const fs = require('node:fs');
const base = 'http://graph-rest:3000';
async function main() {
  const token = fs.readFileSync('/run/secrets/graph_service_jwt', 'utf8').trim();
  const checks = {};
  for (const [name, value] of [['anonymous', null], ['invalid', 'invalid-token']]) {
    const response = await fetch(`${base}/nodes?select=id&limit=1`, {
      headers: value ? { Authorization: `Bearer ${value}` } : {}, signal: AbortSignal.timeout(5000)
    });
    checks[`${name}_denied`] = response.status === 401;
  }
  for (const table of ['nodes', 'links', 'chunks', 'documents', 'claims', 'claim_evidence', 'claim_entities', 'claim_relations']) {
    const response = await fetch(`${base}/${table}?select=*&limit=0`, {
      headers: { Authorization: `Bearer ${token}` }, signal: AbortSignal.timeout(5000)
    });
    checks[`${table}_accessible_with_service_credential`] = response.ok;
  }
  console.log(JSON.stringify(checks, null, 2));
  if (!Object.values(checks).every(Boolean)) process.exitCode = 1;
}
main().catch(() => { console.error('Evidence-store verification failed'); process.exitCode = 1; });
