import assert from "assert";
import {
  aggregateRetrievalMetrics,
  evaluateRetrievalCase,
  precisionAtK,
  recallAtK,
  reciprocalRank,
} from "./metrics";
import { BenchmarkCase, RetrievalSnapshot } from "./types";

assert.strictEqual(precisionAtK(["a", "b", "c"], ["a", "x"], 2), 0.5);
assert.strictEqual(recallAtK(["a", "b", "c"], ["a", "c"], 2), 0.5);
assert.strictEqual(reciprocalRank(["x", "b", "a"], ["a"]), 1 / 3);

const answerable: BenchmarkCase = {
  id: "answerable",
  query: "q",
  answerable: true,
  category: "single-hop",
  relevantNodeIds: ["n2"],
  relevantChunkIds: ["c1"],
};

const answerableSnapshot: RetrievalSnapshot = {
  nodes: [
    { id: "n1", score: 0.9 },
    { id: "n2", score: 0.8 },
  ],
  chunks: [{ id: "c1", score: 0.7 }],
  latencyMs: 10,
};

const m1 = evaluateRetrievalCase(answerable, answerableSnapshot, 2);
assert.strictEqual(m1.nodeRecallAtK, 1);
assert.strictEqual(m1.nodePrecisionAtK, 0.5);
assert.strictEqual(m1.nodeMRR, 0.5);
assert.strictEqual(m1.chunkRecallAtK, 1);
assert.strictEqual(m1.anyRelevantHit, true);
assert.strictEqual(m1.falsePositive, false);

const unanswerable: BenchmarkCase = {
  id: "unanswerable",
  query: "missing fact",
  answerable: false,
  category: "unanswerable",
  relevantNodeIds: [],
  relevantChunkIds: [],
};

const m2 = evaluateRetrievalCase(
  unanswerable,
  { nodes: [{ id: "noise", score: 0.4 }], chunks: [], latencyMs: 20 },
  2
);
assert.strictEqual(m2.falsePositive, true);

const aggregate = aggregateRetrievalMetrics([m1, m2]);
assert.strictEqual(aggregate.cases, 2);
assert.strictEqual(aggregate.answerableCases, 1);
assert.strictEqual(aggregate.unanswerableCases, 1);
assert.strictEqual(aggregate.hitRate, 1);
assert.strictEqual(aggregate.unanswerableFalsePositiveRate, 1);
assert.strictEqual(aggregate.meanLatencyMs, 15);

console.log("Phase 1 retrieval metric tests passed.");
