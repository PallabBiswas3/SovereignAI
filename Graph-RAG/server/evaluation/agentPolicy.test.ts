import assert from "assert";
import { chooseNextAction } from "../agent/policy";
import { AgentState } from "../agent/types";

function baseState(overrides: Partial<AgentState> = {}): AgentState {
  return {
    originalQuery: "How does method A relate to method B?",
    activeQuery: "How does method A relate to method B?",
    step: 0,
    maxSteps: 6,
    requeryCount: 0,
    maxRequeries: 1,
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
    ...overrides,
  };
}

assert.equal(chooseNextAction(baseState()).tool, "hybrid_retrieve");

assert.equal(
  chooseNextAction(baseState({ retrievalDone: true, chunks: [] })).tool,
  "requery"
);

assert.equal(
  chooseNextAction(
    baseState({ retrievalDone: true, chunks: [], requeryCount: 1 })
  ).tool,
  "abstain"
);

const relational = chooseNextAction(
  baseState({
    retrievalDone: true,
    nodes: [
      {
        id: "a",
        label: "A",
        description: "Method A",
        similarity: 0.8,
        retrievalSources: ["dense"],
      },
    ],
    chunks: [
      {
        id: "c1",
        node_id: "a",
        content: "Method A is related to Method B.",
        metadata: {},
        similarity: 0.8,
        retrievalSources: ["bm25"],
      },
    ],
  })
);
assert.equal(relational.tool, "graph_expand");

assert.equal(
  chooseNextAction(
    baseState({
      originalQuery: "What is method A?",
      activeQuery: "What is method A?",
      retrievalDone: true,
      graphExpanded: false,
      chunks: [
        {
          id: "c1",
          node_id: "a",
          content: "Method A is a retrieval method.",
          metadata: {},
          similarity: 0.9,
          retrievalSources: ["dense", "bm25"],
        },
      ],
    })
  ).tool,
  "fetch_evidence"
);

assert.equal(
  chooseNextAction(
    baseState({
      originalQuery: "What is method A?",
      activeQuery: "What is method A?",
      retrievalDone: true,
      graphExpanded: true,
      evidenceFetched: true,
      chunks: [
        {
          id: "c1",
          node_id: "a",
          content: "Method A is a retrieval method.",
          metadata: {},
          similarity: 0.9,
          retrievalSources: ["dense", "bm25"],
        },
        {
          id: "c2",
          node_id: "a",
          content: "Method A combines lexical and semantic evidence.",
          metadata: {},
          similarity: 0.7,
          retrievalSources: ["bm25"],
        },
      ],
      evidenceClaims: [{ id: "claim-1" }],
    })
  ).tool,
  "finalize"
);

assert.equal(
  chooseNextAction(
    baseState({
      step: 6,
      chunks: [],
      retrievalDone: true,
    })
  ).tool,
  "abstain"
);

console.log("Phase 4 adaptive agent policy tests passed.");
