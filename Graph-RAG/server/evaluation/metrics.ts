import {
  BenchmarkCase,
  RetrievalAggregateMetrics,
  RetrievalCaseMetrics,
  RetrievalSnapshot,
} from "./types";

const safeMean = (values: number[]): number =>
  values.length === 0 ? 0 : values.reduce((sum, value) => sum + value, 0) / values.length;

export const precisionAtK = (rankedIds: string[], relevantIds: string[], k: number): number => {
  if (k <= 0) return 0;
  const relevant = new Set(relevantIds);
  const topK = rankedIds.slice(0, k);
  const hits = topK.filter((id) => relevant.has(id)).length;
  return hits / k;
};

export const recallAtK = (rankedIds: string[], relevantIds: string[], k: number): number => {
  if (relevantIds.length === 0) return 0;
  const relevant = new Set(relevantIds);
  const hits = new Set(rankedIds.slice(0, k).filter((id) => relevant.has(id)));
  return hits.size / relevant.size;
};

export const reciprocalRank = (rankedIds: string[], relevantIds: string[]): number => {
  const relevant = new Set(relevantIds);
  const index = rankedIds.findIndex((id) => relevant.has(id));
  return index === -1 ? 0 : 1 / (index + 1);
};

export const evaluateRetrievalCase = (
  benchmark: BenchmarkCase,
  snapshot: RetrievalSnapshot,
  k = 5
): RetrievalCaseMetrics => {
  const nodeIds = snapshot.nodes.map((item) => item.id);
  const chunkIds = snapshot.chunks.map((item) => item.id);

  const anyRelevantHit = benchmark.answerable
    ? nodeIds.some((id) => benchmark.relevantNodeIds.includes(id)) ||
      chunkIds.some((id) => benchmark.relevantChunkIds.includes(id))
    : false;

  const falsePositive = !benchmark.answerable && (nodeIds.length > 0 || chunkIds.length > 0);

  return {
    id: benchmark.id,
    category: benchmark.category,
    answerable: benchmark.answerable,
    nodeRecallAtK: recallAtK(nodeIds, benchmark.relevantNodeIds, k),
    nodePrecisionAtK: precisionAtK(nodeIds, benchmark.relevantNodeIds, k),
    nodeMRR: reciprocalRank(nodeIds, benchmark.relevantNodeIds),
    chunkRecallAtK: recallAtK(chunkIds, benchmark.relevantChunkIds, k),
    chunkPrecisionAtK: precisionAtK(chunkIds, benchmark.relevantChunkIds, k),
    chunkMRR: reciprocalRank(chunkIds, benchmark.relevantChunkIds),
    anyRelevantHit,
    falsePositive,
    latencyMs: snapshot.latencyMs,
  };
};

export const aggregateRetrievalMetrics = (
  metrics: RetrievalCaseMetrics[]
): RetrievalAggregateMetrics => {
  const answerable = metrics.filter((item) => item.answerable);
  const unanswerable = metrics.filter((item) => !item.answerable);

  return {
    cases: metrics.length,
    answerableCases: answerable.length,
    unanswerableCases: unanswerable.length,
    meanNodeRecallAtK: safeMean(answerable.map((item) => item.nodeRecallAtK)),
    meanNodePrecisionAtK: safeMean(answerable.map((item) => item.nodePrecisionAtK)),
    meanNodeMRR: safeMean(answerable.map((item) => item.nodeMRR)),
    meanChunkRecallAtK: safeMean(answerable.map((item) => item.chunkRecallAtK)),
    meanChunkPrecisionAtK: safeMean(answerable.map((item) => item.chunkPrecisionAtK)),
    meanChunkMRR: safeMean(answerable.map((item) => item.chunkMRR)),
    hitRate: safeMean(answerable.map((item) => (item.anyRelevantHit ? 1 : 0))),
    unanswerableFalsePositiveRate: safeMean(
      unanswerable.map((item) => (item.falsePositive ? 1 : 0))
    ),
    meanLatencyMs: safeMean(metrics.map((item) => item.latencyMs)),
  };
};
