import assert from "node:assert/strict";
import { verifyClaimsAgainstChunks } from "./claimVerifier";
import { RetrievedChunk } from "../retrieval/hybridRetriever";

const originalFetch = globalThis.fetch;
const originalUrl = process.env.OLLAMA_URL;

async function run() {
  let requestCount = 0;
  globalThis.fetch = (async (_input: string | URL | Request, init?: RequestInit) => {
    requestCount += 1;
    const request = JSON.parse(String(init?.body || "{}"));
    assert.equal(request.format, "json");
    return new Response(JSON.stringify({
      response: JSON.stringify({
        label: request.prompt.includes("9 mm/s") ? "CONTRADICTED" : "SUPPORTED",
        confidence: 0.9,
        reason: "Compared with the allowed source.",
      }),
    }), { status: 200 });
  }) as typeof fetch;
  process.env.OLLAMA_URL = "http://127.0.0.1:11434";

  const chunks: RetrievedChunk[] = [{
    id: "chunk-1",
    node_id: "node-1",
    content: "The approved vibration limit is 7.1 mm/s.",
    metadata: {},
    similarity: 1,
    retrievalSources: ["test"],
  }];
  const results = await verifyClaimsAgainstChunks([
    { id: "claim-1", claim_text: "The limit is 7.1 mm/s.", evidence: [{ chunk_id: "chunk-1" }] },
    { id: "claim-2", claim_text: "The limit is 9 mm/s.", evidence: [{ chunk_id: "chunk-1" }] },
  ], chunks);

  assert.equal(requestCount, 2);
  assert.equal(results[0].label, "SUPPORTED");
  assert.equal(results[1].label, "CONTRADICTED");
}

run()
  .then(() => console.log("bounded parallel claim verifier tests passed"))
  .finally(() => {
    globalThis.fetch = originalFetch;
    if (originalUrl === undefined) delete process.env.OLLAMA_URL;
    else process.env.OLLAMA_URL = originalUrl;
  });
