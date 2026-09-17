import fs from "fs";
import path from "path";
import { performance } from "perf_hooks";
import { retrieveHybrid } from "../retrieval/hybridRetriever";
import { aggregateRetrievalMetrics, evaluateRetrievalCase } from "./metrics";
import { BenchmarkCase, RetrievalSnapshot } from "./types";

async function retrieve(query: string): Promise<RetrievalSnapshot> {
  const start = performance.now();
  const result = await retrieveHybrid(query, 20);

  return {
    nodes: result.nodes.map((node) => ({ id: node.id, score: node.similarity })),
    chunks: result.chunks.map((chunk) => ({ id: chunk.id, score: chunk.similarity })),
    latencyMs: performance.now() - start,
  };
}

async function main() {
  const benchmarkPath = process.argv[2];
  if (!benchmarkPath) {
    throw new Error("Usage: npm run eval:retrieval -- <benchmark.json>");
  }

  const absolutePath = path.resolve(process.cwd(), benchmarkPath);
  const cases = JSON.parse(fs.readFileSync(absolutePath, "utf8")) as BenchmarkCase[];

  const perCase = [];
  for (const benchmark of cases) {
    const snapshot = await retrieve(benchmark.query);
    perCase.push(evaluateRetrievalCase(benchmark, snapshot, 5));
  }

  const report = {
    generatedAt: new Date().toISOString(),
    retrievalPipeline: "dense+bm25->rrf->lexical-rerank",
    benchmark: absolutePath,
    aggregate: aggregateRetrievalMetrics(perCase),
    cases: perCase,
  };

  console.log(JSON.stringify(report, null, 2));
}

main().catch((error) => {
  console.error(error);
  process.exit(1);
});
