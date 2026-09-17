import { RetrievedChunk, RetrievedNode } from "../retrieval/hybridRetriever";

export type AgentTool =
  | "hybrid_retrieve"
  | "graph_expand"
  | "fetch_evidence"
  | "requery"
  | "finalize"
  | "abstain";

export interface AgentAction {
  tool: AgentTool;
  reason: string;
  query?: string;
}

export interface AgentTraceStep {
  step: number;
  tool: AgentTool;
  reason: string;
  observation: string;
}

export interface AgentSubgraphEdge {
  source: string;
  target: string;
  relationship: string;
  depth: number;
}

export interface AgentState {
  originalQuery: string;
  activeQuery: string;
  step: number;
  maxSteps: number;
  requeryCount: number;
  maxRequeries: number;
  nodes: RetrievedNode[];
  chunks: RetrievedChunk[];
  expandedEdges: AgentSubgraphEdge[];
  expandedNodeIds: string[];
  evidenceClaims: any[];
  retrievalDone: boolean;
  graphExpanded: boolean;
  evidenceFetched: boolean;
  finalized: boolean;
  abstained: boolean;
  trace: AgentTraceStep[];
}

export interface AdaptiveAgentResult {
  state: AgentState;
  decision: "finalize" | "abstain";
}
