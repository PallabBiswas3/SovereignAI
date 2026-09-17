import { supabase } from "../supabase";
import { fetchEvidenceForChunks } from "../evidence/evidenceService";
import { retrieveHybrid, RetrievedChunk, RetrievedNode } from "../retrieval/hybridRetriever";
import { chooseNextAction } from "./policy";
import { AdaptiveAgentResult, AgentState, AgentSubgraphEdge } from "./types";

function mergeById<T extends { id: string }>(existing: T[], incoming: T[]): T[] {
  const merged = new Map<string, T>();
  for (const item of existing) merged.set(item.id, item);
  for (const item of incoming) merged.set(item.id, item);
  return [...merged.values()];
}

function summarizeRetrieval(nodes: RetrievedNode[], chunks: RetrievedChunk[]): string {
  const topChunk = chunks[0]?.similarity ?? 0;
  return `Retrieved ${nodes.length} nodes and ${chunks.length} chunks; top chunk score=${topChunk.toFixed(3)}.`;
}

export async function runAdaptiveAgent(
  query: string,
  options: { maxSteps?: number; maxRequeries?: number } = {}
): Promise<AdaptiveAgentResult> {
  const state: AgentState = {
    originalQuery: query,
    activeQuery: query,
    step: 0,
    maxSteps: options.maxSteps ?? 6,
    requeryCount: 0,
    maxRequeries: options.maxRequeries ?? 1,
    nodes: [],
    chunks: [],
    expandedEdges: [],
    expandedNodeIds: [],
    evidenceClaims: [],
    retrievalDone: false,
    graphExpanded: false,
    evidenceFetched: false,
    finalized: false,
    abstained: false,
    trace: [],
  };

  while (!state.finalized && !state.abstained) {
    const action = chooseNextAction(state);
    state.step += 1;
    let observation = "";

    if (action.tool === "hybrid_retrieve") {
      const retrieval = await retrieveHybrid(state.activeQuery, 20);
      state.nodes = mergeById(state.nodes, retrieval.nodes);
      state.chunks = mergeById(state.chunks, retrieval.chunks);
      state.retrievalDone = true;
      observation = summarizeRetrieval(retrieval.nodes, retrieval.chunks);
    } else if (action.tool === "graph_expand") {
      const startIds = [
        ...new Set([
          ...state.nodes.map((node) => node.id),
          ...state.chunks.map((chunk) => chunk.node_id),
        ]),
      ].slice(0, 12);

      if (startIds.length) {
        const { data, error } = await supabase.rpc("expand_graph", {
          start_ids: startIds,
          max_depth: 2,
        });
        if (error) throw error;
        state.expandedEdges = (data || []) as AgentSubgraphEdge[];
        state.expandedNodeIds = [
          ...new Set(
            state.expandedEdges.flatMap((edge) => [edge.source, edge.target])
          ),
        ];
      }
      state.graphExpanded = true;
      observation = `Expanded graph to ${state.expandedEdges.length} edges across ${state.expandedNodeIds.length} node IDs.`;
    } else if (action.tool === "fetch_evidence") {
      try {
        state.evidenceClaims = await fetchEvidenceForChunks(state.chunks.map((chunk) => chunk.id));
        observation = `Resolved ${state.evidenceClaims.length} provenance-linked claims from retrieved chunks.`;
      } catch (error: any) {
        state.evidenceClaims = [];
        observation = `Evidence graph unavailable (${error?.message || "unknown error"}); raw chunks remain authoritative.`;
      }
      state.evidenceFetched = true;
    } else if (action.tool === "requery") {
      state.requeryCount += 1;
      state.activeQuery = action.query || state.originalQuery;
      const retrieval = await retrieveHybrid(state.activeQuery, 20);
      state.nodes = mergeById(state.nodes, retrieval.nodes);
      state.chunks = mergeById(state.chunks, retrieval.chunks);
      state.retrievalDone = true;
      state.evidenceFetched = false;
      observation = `Requery '${state.activeQuery}' added evidence; total now ${state.nodes.length} nodes and ${state.chunks.length} chunks.`;
    } else if (action.tool === "finalize") {
      state.finalized = true;
      observation = "Agent selected grounded synthesis.";
    } else if (action.tool === "abstain") {
      state.abstained = true;
      observation = "Agent selected abstention because usable evidence was not found.";
    }

    state.trace.push({
      step: state.step,
      tool: action.tool,
      reason: action.reason,
      observation,
    });
  }

  return {
    state,
    decision: state.abstained ? "abstain" : "finalize",
  };
}
