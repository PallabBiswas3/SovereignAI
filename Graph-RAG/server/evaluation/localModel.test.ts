import assert from "node:assert/strict";
import {
  generateLocalEmbedding,
  generateLocalJson,
  generateLocalText,
  localModelConfiguration,
} from "../localModel";

const originalFetch = globalThis.fetch;
const originalUrl = process.env.OLLAMA_URL;
const originalDenseRetrieval = process.env.GRAPHRAG_DENSE_RETRIEVAL_ENABLED;

async function run() {
  const requests: Array<{ url: string; body: any }> = [];
  globalThis.fetch = (async (input: string | URL | Request, init?: RequestInit) => {
    const url = String(input);
    const body = JSON.parse(String(init?.body || "{}"));
    requests.push({ url, body });
    if (url.endsWith("/api/embed")) {
      return new Response(JSON.stringify({ embeddings: [Array.from({ length: 768 }, (_, index) => index === 0 ? 2 : 0)] }), { status: 200 });
    }
    return new Response(JSON.stringify({ response: body.format === "json" ? '{"ok":true}' : "local answer" }), { status: 200 });
  }) as typeof fetch;

  process.env.OLLAMA_URL = "http://127.0.0.1:11434";
  assert.equal(await generateLocalText("hello"), "local answer");
  assert.deepEqual(await generateLocalJson<{ ok: boolean }>("return json"), { ok: true });
  const embedding = await generateLocalEmbedding("pump vibration");
  assert.equal(embedding.length, 768);
  assert.equal(embedding[0], 1);
  assert.ok(requests.every((request) => request.url.startsWith("http://127.0.0.1:11434/")));
  assert.ok(requests.filter((request) => request.url.endsWith("/api/generate"))
    .every((request) => request.body.think === false));
  assert.equal(localModelConfiguration().provider, "ollama");
  delete process.env.GRAPHRAG_DENSE_RETRIEVAL_ENABLED;
  assert.equal(localModelConfiguration().denseRetrievalEnabled, false);
  process.env.GRAPHRAG_DENSE_RETRIEVAL_ENABLED = "true";
  assert.equal(localModelConfiguration().denseRetrievalEnabled, true);

  process.env.OLLAMA_URL = "https://external.example.com";
  await assert.rejects(() => generateLocalText("blocked"), /loopback or internal service URL/);
}

run()
  .then(() => console.log("local model adapter tests passed"))
  .finally(() => {
    globalThis.fetch = originalFetch;
    if (originalUrl === undefined) delete process.env.OLLAMA_URL;
    else process.env.OLLAMA_URL = originalUrl;
    if (originalDenseRetrieval === undefined) delete process.env.GRAPHRAG_DENSE_RETRIEVAL_ENABLED;
    else process.env.GRAPHRAG_DENSE_RETRIEVAL_ENABLED = originalDenseRetrieval;
  });
