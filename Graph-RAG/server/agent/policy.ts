import { AgentAction, AgentState } from "./types";

const hasQuestionCue = (query: string, cues: string[]): boolean => {
  const lower = query.toLowerCase();
  return cues.some((cue) => lower.includes(cue));
};

export function buildRequery(originalQuery: string): string {
  const lower = originalQuery.toLowerCase();
  const suffix = hasQuestionCue(lower, ["compare", "difference", "versus", "vs", "why", "how", "relationship"])
    ? " supporting evidence relationships mechanisms"
    : " supporting evidence source details";
  return `${originalQuery}${suffix}`;
}

export function chooseNextAction(state: AgentState): AgentAction {
  if (state.step >= state.maxSteps) {
    return state.chunks.length > 0
      ? { tool: "finalize", reason: "Step budget reached with source evidence available." }
      : { tool: "abstain", reason: "Step budget reached without usable source evidence." };
  }

  if (!state.retrievalDone) {
    return { tool: "hybrid_retrieve", reason: "Start with hybrid dense + BM25 retrieval." };
  }

  if (!state.chunks.length && state.requeryCount < state.maxRequeries) {
    return {
      tool: "requery",
      reason: "Initial retrieval returned no source chunks; broaden the query once.",
      query: buildRequery(state.originalQuery),
    };
  }

  if (!state.chunks.length) {
    return { tool: "abstain", reason: "No source chunks remain after bounded requery." };
  }

  const wantsRelationalReasoning = hasQuestionCue(state.originalQuery, [
    "compare",
    "difference",
    "versus",
    "vs",
    "relationship",
    "related",
    "why",
    "how",
    "connect",
    "between",
  ]);

  const multipleNodeSignals = new Set([
    ...state.nodes.map((node) => node.id),
    ...state.chunks.map((chunk) => chunk.node_id),
  ]).size >= 2;

  if (!state.graphExpanded && (wantsRelationalReasoning || multipleNodeSignals)) {
    return {
      tool: "graph_expand",
      reason: wantsRelationalReasoning
        ? "The query asks for relational or multi-hop reasoning, so inspect graph connections."
        : "Retrieval spans multiple entities; graph expansion can connect the evidence.",
    };
  }

  if (!state.evidenceFetched) {
    return {
      tool: "fetch_evidence",
      reason: "Resolve retrieved source chunks into provenance-linked atomic claims when available.",
    };
  }

  const weakRetrieval = state.chunks.length < 2 || state.chunks[0].similarity < 0.2;
  if (weakRetrieval && state.requeryCount < state.maxRequeries) {
    return {
      tool: "requery",
      reason: "Retrieved evidence is sparse or weak; use the remaining requery budget.",
      query: buildRequery(state.originalQuery),
    };
  }

  return {
    tool: "finalize",
    reason: state.evidenceClaims.length
      ? "Sufficient source chunks and provenance-linked claims are available for synthesis."
      : "Structured claims are unavailable, but retrieved source chunks are sufficient for grounded synthesis.",
  };
}
