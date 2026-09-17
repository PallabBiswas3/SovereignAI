export type BenchmarkCategory =
  | "single-hop"
  | "multi-hop"
  | "global"
  | "contradictory"
  | "unanswerable";

export interface BenchmarkCase {
  id: string;
  query: string;
  answerable: boolean;
  category?: BenchmarkCategory;
  relevantNodeIds: string[];
  relevantChunkIds: string[];
  notes?: string;
}

export interface RankedRetrievalItem {
  id: string;
  score: number;
}

export interface RetrievalSnapshot {
  nodes: RankedRetrievalItem[];
  chunks: RankedRetrievalItem[];
  latencyMs: number;
}

export interface RetrievalCaseMetrics {
  id: string;
  category?: BenchmarkCategory;
  answerable: boolean;
  nodeRecallAtK: number;
  nodePrecisionAtK: number;
  nodeMRR: number;
  chunkRecallAtK: number;
  chunkPrecisionAtK: number;
  chunkMRR: number;
  anyRelevantHit: boolean;
  falsePositive: boolean;
  latencyMs: number;
}

export interface RetrievalAggregateMetrics {
  cases: number;
  answerableCases: number;
  unanswerableCases: number;
  meanNodeRecallAtK: number;
  meanNodePrecisionAtK: number;
  meanNodeMRR: number;
  meanChunkRecallAtK: number;
  meanChunkPrecisionAtK: number;
  meanChunkMRR: number;
  hitRate: number;
  unanswerableFalsePositiveRate: number;
  meanLatencyMs: number;
}
