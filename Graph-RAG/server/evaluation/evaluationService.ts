import { supabase } from "../supabase";
import { generateLocalText } from "../localModel";
import { runAdaptiveAgent } from "../agent/adaptiveAgent";
import { AgentSubgraphEdge } from "../agent/types";
import {
  retrieveHybrid,
  RetrievedChunk,
  RetrievedNode,
} from "../retrieval/hybridRetriever";
import { runVerificationPipeline } from "../verification/verificationPipeline";

export const EVALUATION_MODES = [
  "fixed",
  "adaptive",
  "adaptive_verified",
] as const;

export type EvaluationMode = (typeof EVALUATION_MODES)[number];

export interface EvaluationSourceCitation {
  nodeId: string;
  nodeLabel: string;
  similarity: number;
  chunkId?: string;
  sourceDocId?: string;
  pageStart?: number;
  pageEnd?: number;
  chunkContent?: string;
}

export interface EvaluationTimings {
  retrievalMs?: number;
  graphExpansionMs?: number;
  agentMs?: number;
  verificationMs?: number;
  synthesisMs?: number;
  totalMs: number;
}

export interface EvaluationCounts {
  nodes: number;
  chunks: number;
  expandedEdges: number;
  claims: number;
  supported?: number;
  contradicted?: number;
  insufficient?: number;
}

export interface EvaluationResult {
  mode: EvaluationMode;
  content: string;
  sources: EvaluationSourceCitation[];
  reasoningTrace: string[];
  abstained: boolean;
  timings: EvaluationTimings;
  counts: EvaluationCounts;
  toolCalls: string[];
  verification?: {
    available: boolean;
    retried: boolean;
    decision: "continue" | "abstain";
    calibratedScore: number;
    supported: number;
    contradicted: number;
    insufficient: number;
  };
}

type StatusCallback = (status: string) => void;

interface FullNode {
  id: string;
  label: string;
  type?: string;
  description?: string;
  properties?: Record<string, unknown>;
  confidence?: number;
}

export function parseEvaluationMode(value: unknown): EvaluationMode {
  if (value === undefined || value === null || value === "") {
    return "adaptive_verified";
  }

  if (
    typeof value === "string" &&
    (EVALUATION_MODES as readonly string[]).includes(value)
  ) {
    return value as EvaluationMode;
  }

  throw new Error(
    `Invalid mode. Expected one of: ${EVALUATION_MODES.join(", ")}`
  );
}

async function fetchFullNodes(nodeIds: string[]): Promise<FullNode[]> {
  const requested = [...new Set(nodeIds.filter(Boolean))].slice(0, 40);
  if (!requested.length) return [];

  const { data, error } = await supabase
    .from("nodes")
    .select("id,label,type,description,properties,confidence")
    .in("id", requested);
  if (error) throw error;
  return (data || []) as FullNode[];
}

function getPageRange(metadata: Record<string, unknown>) {
  const pageStart =
    typeof metadata.pageStart === "number" ? metadata.pageStart : undefined;
  const pageEnd =
    typeof metadata.pageEnd === "number" ? metadata.pageEnd : undefined;
  return { pageStart, pageEnd };
}

function buildChunkContext(chunks: RetrievedChunk[]): string {
  return chunks
    .map((chunk, index) => {
      const metadata = chunk.metadata || {};
      const { pageStart, pageEnd } = getPageRange(metadata);
      const pages = pageStart
        ? ` | pages ${pageStart}${
            pageEnd && pageEnd !== pageStart ? `-${pageEnd}` : ""
          }`
        : "";

      return `[CHUNK ${index + 1} | id ${chunk.id} | node ${
        chunk.node_id
      }${pages} | score ${chunk.similarity.toFixed(3)}]\n${chunk.content}`;
    })
    .join("\n\n");
}

function buildNodeContext(fullNodes: FullNode[]): string {
  return (
    fullNodes
      .map(
        (node) =>
          `[NODE] ${node.label} (${node.type || "unknown"}) — confidence: ${
            node.confidence ?? 1
          }\n${node.description || ""}${
            node.properties && Object.keys(node.properties).length
              ? `\nProperties: ${JSON.stringify(node.properties)}`
              : ""
          }`
      )
      .join("\n\n") || "None"
  );
}

function buildEdgeContext(edges: AgentSubgraphEdge[]): string {
  return (
    edges
      .map((edge) => `${edge.source} --[${edge.relationship}]--> ${edge.target}`)
      .join("\n") || "No graph expansion evidence was available."
  );
}

function buildSources(
  nodes: RetrievedNode[],
  chunks: RetrievedChunk[],
  fullNodes: FullNode[]
): EvaluationSourceCitation[] {
  return [
    ...nodes.map((node) => ({
      nodeId: node.id,
      nodeLabel: node.label,
      similarity: node.similarity,
    })),
    ...chunks.map((chunk) => {
      const metadata = chunk.metadata || {};
      const { pageStart, pageEnd } = getPageRange(metadata);
      const sourceDocId =
        typeof metadata.sourceDocId === "string"
          ? metadata.sourceDocId
          : typeof metadata.source_doc_id === "string"
          ? metadata.source_doc_id
          : undefined;

      return {
        nodeId: chunk.node_id,
        nodeLabel:
          fullNodes.find((node) => node.id === chunk.node_id)?.label ??
          chunk.node_id,
        similarity: chunk.similarity,
        chunkId: chunk.id,
        sourceDocId,
        pageStart,
        pageEnd,
        chunkContent:
          chunk.content.slice(0, 160) + (chunk.content.length > 160 ? "…" : ""),
      };
    }),
  ];
}

async function expandGraph(
  nodes: RetrievedNode[],
  chunks: RetrievedChunk[]
): Promise<{ edges: AgentSubgraphEdge[]; nodeIds: string[] }> {
  const startIds = [
    ...new Set([
      ...nodes.map((node) => node.id),
      ...chunks.map((chunk) => chunk.node_id),
    ]),
  ].slice(0, 12);

  if (!startIds.length) return { edges: [], nodeIds: [] };

  const { data, error } = await supabase.rpc("expand_graph", {
    start_ids: startIds,
    max_depth: 2,
  });
  if (error) throw error;

  const edges = (data || []) as AgentSubgraphEdge[];
  const nodeIds = [
    ...new Set(edges.flatMap((edge) => [edge.source, edge.target])),
  ];

  return { edges, nodeIds };
}

async function synthesizeWithoutVerification({
  mode,
  query,
  nodes,
  chunks,
  edges,
  expandedNodeIds,
  claims,
  trace,
}: {
  mode: "fixed" | "adaptive";
  query: string;
  nodes: RetrievedNode[];
  chunks: RetrievedChunk[];
  edges: AgentSubgraphEdge[];
  expandedNodeIds: string[];
  claims: any[];
  trace: string[];
}) {
  const fullNodes = await fetchFullNodes([
    ...nodes.map((node) => node.id),
    ...chunks.map((chunk) => chunk.node_id),
    ...expandedNodeIds,
  ]);

  const unverifiedClaimContext = claims.length
    ? claims
        .slice(0, 40)
        .map((claim, index) => {
          const pages = claim.page_start
            ? ` pages ${claim.page_start}${
                claim.page_end && claim.page_end !== claim.page_start
                  ? `-${claim.page_end}`
                  : ""
              }`
            : "";
          return `[UNVERIFIED CLAIM ${index + 1} | source ${
            claim.source_doc_id || "unknown"
          }${pages}]\n${claim.claim_text}`;
        })
        .join("\n\n")
    : "None";

  const content = await generateLocalText(`
You are the synthesis component of a GraphRAG ablation experiment.
Evaluation mode: ${mode}

=== RETRIEVAL / AGENT TRACE ===
${trace.join("\n") || "No trace available."}

=== GRAPH NODES ===
${buildNodeContext(fullNodes)}

=== GRAPH CONNECTIONS ===
${buildEdgeContext(edges)}

=== UNVERIFIED PROVENANCE CLAIMS ===
${unverifiedClaimContext}

=== ORIGINAL SOURCE CHUNKS ===
${buildChunkContext(chunks)}

=== USER QUESTION ===
${query}

Rules:
- Answer ONLY from the supplied evidence.
- Original source chunks are the ultimate source of truth.
- Unverified structured claims are navigation aids only; do not rely on a claim unless the source chunks support it.
- Preserve negation, numerical qualifiers, evaluation settings, model variants, and conditions.
- If evidence does not support the requested fact, explicitly say the supplied evidence does not provide it.
- If sources disagree, state the disagreement.
- Cite chunk numbers/pages in the prose whenever practical.
`);

  return {
    content,
    fullNodes,
  };
}

async function runFixed(
  query: string,
  onStatus?: StatusCallback
): Promise<EvaluationResult> {
  const totalStarted = Date.now();
  const timings: EvaluationTimings = { totalMs: 0 };

  onStatus?.("Running fixed hybrid retrieval...");
  const retrievalStarted = Date.now();
  const retrieval = await retrieveHybrid(query, 20);
  timings.retrievalMs = Date.now() - retrievalStarted;

  if (!retrieval.chunks.length) {
    timings.totalMs = Date.now() - totalStarted;
    return {
      mode: "fixed",
      content:
        "I could not find enough source evidence in the uploaded knowledge base to answer this reliably.",
      sources: [],
      reasoningTrace: [
        `fixed:1:hybrid_retrieve — Retrieved ${retrieval.nodes.length} nodes and 0 chunks.`,
        "fixed:2:abstain — No usable source chunks were retrieved.",
      ],
      abstained: true,
      timings,
      counts: {
        nodes: retrieval.nodes.length,
        chunks: 0,
        expandedEdges: 0,
        claims: 0,
      },
      toolCalls: ["hybrid_retrieve", "abstain"],
    };
  }

  onStatus?.("Running fixed graph expansion...");
  const graphStarted = Date.now();
  const expansion = await expandGraph(retrieval.nodes, retrieval.chunks);
  timings.graphExpansionMs = Date.now() - graphStarted;

  const trace = [
    `fixed:1:hybrid_retrieve — Retrieved ${retrieval.nodes.length} nodes and ${retrieval.chunks.length} chunks; top chunk score=${(
      retrieval.chunks[0]?.similarity ?? 0
    ).toFixed(3)}.`,
    `fixed:2:graph_expand — Expanded graph to ${expansion.edges.length} edges across ${expansion.nodeIds.length} node IDs.`,
    "fixed:3:finalize — Fixed pipeline proceeded directly to synthesis; no adaptive routing or claim verifier was used.",
  ];

  onStatus?.("Generating fixed GraphRAG answer...");
  const synthesisStarted = Date.now();
  const synthesis = await synthesizeWithoutVerification({
    mode: "fixed",
    query,
    nodes: retrieval.nodes,
    chunks: retrieval.chunks,
    edges: expansion.edges,
    expandedNodeIds: expansion.nodeIds,
    claims: [],
    trace,
  });
  timings.synthesisMs = Date.now() - synthesisStarted;
  timings.totalMs = Date.now() - totalStarted;

  return {
    mode: "fixed",
    content: synthesis.content,
    sources: buildSources(retrieval.nodes, retrieval.chunks, synthesis.fullNodes),
    reasoningTrace: trace,
    abstained: false,
    timings,
    counts: {
      nodes: retrieval.nodes.length,
      chunks: retrieval.chunks.length,
      expandedEdges: expansion.edges.length,
      claims: 0,
    },
    toolCalls: ["hybrid_retrieve", "graph_expand", "finalize"],
  };
}

async function runAdaptive(
  query: string,
  onStatus?: StatusCallback
): Promise<EvaluationResult> {
  const totalStarted = Date.now();
  const timings: EvaluationTimings = { totalMs: 0 };

  onStatus?.("Running adaptive retrieval and graph routing...");
  const agentStarted = Date.now();
  const agent = await runAdaptiveAgent(query, { maxSteps: 6, maxRequeries: 1 });
  timings.agentMs = Date.now() - agentStarted;
  const state = agent.state;

  const reasoningTrace = state.trace.map(
    (item) =>
      `${item.step}:${item.tool} — ${item.reason} | ${item.observation}`
  );
  const toolCalls = state.trace.map((item) => item.tool);

  if (agent.decision === "abstain" || !state.chunks.length) {
    timings.totalMs = Date.now() - totalStarted;
    return {
      mode: "adaptive",
      content:
        "I could not find enough source evidence in the uploaded knowledge base to answer this reliably.",
      sources: [],
      reasoningTrace,
      abstained: true,
      timings,
      counts: {
        nodes: state.nodes.length,
        chunks: state.chunks.length,
        expandedEdges: state.expandedEdges.length,
        claims: state.evidenceClaims.length,
      },
      toolCalls,
    };
  }

  onStatus?.("Generating adaptive GraphRAG answer without verification...");
  const synthesisStarted = Date.now();
  const synthesis = await synthesizeWithoutVerification({
    mode: "adaptive",
    query,
    nodes: state.nodes,
    chunks: state.chunks,
    edges: state.expandedEdges,
    expandedNodeIds: state.expandedNodeIds,
    claims: state.evidenceClaims,
    trace: reasoningTrace,
  });
  timings.synthesisMs = Date.now() - synthesisStarted;
  timings.totalMs = Date.now() - totalStarted;

  return {
    mode: "adaptive",
    content: synthesis.content,
    sources: buildSources(state.nodes, state.chunks, synthesis.fullNodes),
    reasoningTrace,
    abstained: false,
    timings,
    counts: {
      nodes: state.nodes.length,
      chunks: state.chunks.length,
      expandedEdges: state.expandedEdges.length,
      claims: state.evidenceClaims.length,
    },
    toolCalls,
  };
}

async function runAdaptiveVerified(
  query: string,
  onStatus?: StatusCallback
): Promise<EvaluationResult> {
  const totalStarted = Date.now();
  const timings: EvaluationTimings = { totalMs: 0 };

  onStatus?.("Running adaptive retrieval and graph routing...");
  const agentStarted = Date.now();
  const agent = await runAdaptiveAgent(query, { maxSteps: 6, maxRequeries: 1 });
  timings.agentMs = Date.now() - agentStarted;
  const state = agent.state;

  const agentTrace = state.trace.map(
    (item) =>
      `${item.step}:${item.tool} — ${item.reason} | ${item.observation}`
  );
  const toolCalls = state.trace.map((item) => item.tool);

  if (agent.decision === "abstain" || !state.chunks.length) {
    timings.totalMs = Date.now() - totalStarted;
    return {
      mode: "adaptive_verified",
      content:
        "I could not find enough source evidence in the uploaded knowledge base to answer this reliably.",
      sources: [],
      reasoningTrace: agentTrace,
      abstained: true,
      timings,
      counts: {
        nodes: state.nodes.length,
        chunks: state.chunks.length,
        expandedEdges: state.expandedEdges.length,
        claims: state.evidenceClaims.length,
      },
      toolCalls,
    };
  }

  onStatus?.("Verifying retrieved claims against source chunks...");
  const verificationStarted = Date.now();
  const verification = await runVerificationPipeline({
    originalQuery: query,
    nodes: state.nodes,
    chunks: state.chunks,
    claims: state.evidenceClaims,
    maxRetries: 1,
  });
  timings.verificationMs = Date.now() - verificationStarted;

  const verificationTrace = verification.trace.map(
    (item, index) => `verify:${index + 1} — ${item}`
  );
  const reasoningTrace = [...agentTrace, ...verificationTrace];

  const verificationMetadata = {
    available: verification.available,
    retried: verification.retried,
    decision: verification.decision,
    calibratedScore: verification.summary.calibratedScore,
    supported: verification.summary.supported,
    contradicted: verification.summary.contradicted,
    insufficient: verification.summary.insufficient,
  };

  if (verification.available && verification.decision === "abstain") {
    timings.totalMs = Date.now() - totalStarted;
    return {
      mode: "adaptive_verified",
      content:
        "I found related material, but claim-level verification did not produce enough reliable support after the bounded retry, so I’m abstaining rather than presenting an uncertain answer.",
      sources: [],
      reasoningTrace,
      abstained: true,
      timings,
      counts: {
        nodes: verification.nodes.length,
        chunks: verification.chunks.length,
        expandedEdges: state.expandedEdges.length,
        claims: verification.claims.length,
        supported: verification.summary.supported,
        contradicted: verification.summary.contradicted,
        insufficient: verification.summary.insufficient,
      },
      toolCalls: [...toolCalls, "verify", ...(verification.retried ? ["verification_requery"] : [])],
      verification: verificationMetadata,
    };
  }

  const workingNodes = verification.nodes;
  const workingChunks = verification.chunks;
  const workingClaims = verification.claims;
  const supportedClaimIds = new Set(
    verification.results
      .filter((item) => item.label === "SUPPORTED")
      .map((item) => item.claimId)
  );
  const supportedClaims = verification.available
    ? workingClaims.filter((claim: any) => supportedClaimIds.has(claim.id))
    : workingClaims;

  const fullNodes = await fetchFullNodes([
    ...workingNodes.map((node) => node.id),
    ...workingChunks.map((chunk) => chunk.node_id),
    ...state.expandedNodeIds,
  ]);

  const evidenceContext = supportedClaims.length
    ? supportedClaims
        .map((claim: any, index: number) => {
          const verificationItem = verification.results.find(
            (item) => item.claimId === claim.id
          );
          const pages = claim.page_start
            ? ` pages ${claim.page_start}${
                claim.page_end && claim.page_end !== claim.page_start
                  ? `-${claim.page_end}`
                  : ""
              }`
            : "";
          const supportingChunkIds = (claim.evidence || [])
            .map((item: any) => item.chunk_id)
            .join(", ");

          return `[VERIFIED CLAIM ${index + 1} | SUPPORTED | verifier confidence ${Number(
            verificationItem?.confidence ?? 1
          ).toFixed(2)} | source ${claim.source_doc_id || "unknown"}${pages} | chunks ${supportingChunkIds}]\n${claim.claim_text}`;
        })
        .join("\n\n")
    : "No structured claims passed verification; rely only on the original source chunks.";

  const verificationFindings = verification.available
    ? verification.results
        .filter((item) => item.label !== "SUPPORTED")
        .map(
          (item) =>
            `[${item.label} | confidence ${item.confidence.toFixed(2)}] ${
              item.claimText
            }\nReason: ${item.reason}`
        )
        .join("\n\n") || "No contradicted or insufficient structured claims."
    : "Structured claim verification was unavailable; use raw chunks conservatively.";

  onStatus?.("Generating verified grounded answer...");
  const synthesisStarted = Date.now();
  const content = await generateLocalText(`
You are the synthesis component of a bounded evidence-grounded graph agent with claim verification.
Evaluation mode: adaptive_verified

=== AGENT TOOL TRACE ===
${reasoningTrace.join("\n")}

=== VERIFICATION SUMMARY ===
available=${verification.available}
retried=${verification.retried}
supported=${verification.summary.supported}
contradicted=${verification.summary.contradicted}
insufficient=${verification.summary.insufficient}
selective_score=${verification.summary.calibratedScore.toFixed(3)}

=== GRAPH NODES ===
${buildNodeContext(fullNodes)}

=== GRAPH CONNECTIONS ===
${buildEdgeContext(state.expandedEdges)}

=== VERIFIED SUPPORTED CLAIMS ===
${evidenceContext}

=== VERIFICATION FINDINGS TO TREAT AS WARNINGS ===
${verificationFindings}

=== ORIGINAL SOURCE CHUNKS ===
${buildChunkContext(workingChunks)}

=== USER QUESTION ===
${query}

Rules:
- Answer ONLY from the supplied evidence.
- Original source chunks are the ultimate source of truth.
- Use structured claims positively only when they appear under VERIFIED SUPPORTED CLAIMS.
- CONTRADICTED or INSUFFICIENT findings must not be presented as established facts.
- If sources disagree, explicitly state the disagreement.
- Preserve uncertainty, negation, numerical qualifiers, and conditions.
- If the remaining verified evidence is insufficient for part of the question, state that limitation.
- Cite chunk numbers/pages in the prose whenever practical.
`);
  timings.synthesisMs = Date.now() - synthesisStarted;
  timings.totalMs = Date.now() - totalStarted;

  return {
    mode: "adaptive_verified",
    content,
    sources: buildSources(workingNodes, workingChunks, fullNodes),
    reasoningTrace,
    abstained: false,
    timings,
    counts: {
      nodes: workingNodes.length,
      chunks: workingChunks.length,
      expandedEdges: state.expandedEdges.length,
      claims: workingClaims.length,
      supported: verification.summary.supported,
      contradicted: verification.summary.contradicted,
      insufficient: verification.summary.insufficient,
    },
    toolCalls: [
      ...toolCalls,
      "verify",
      ...(verification.retried ? ["verification_requery"] : []),
      "finalize",
    ],
    verification: verificationMetadata,
  };
}

export async function runEvaluationQuery(
  query: string,
  mode: EvaluationMode,
  onStatus?: StatusCallback
): Promise<EvaluationResult> {
  if (mode === "fixed") return runFixed(query, onStatus);
  if (mode === "adaptive") return runAdaptive(query, onStatus);
  return runAdaptiveVerified(query, onStatus);
}
