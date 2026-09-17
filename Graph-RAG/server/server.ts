import express from "express";
import cors from "cors";
import dotenv from "dotenv";
import path from "path";
import { supabase } from "./supabase";
import { generateEmbedding } from "./embedding";
import { generateLocalJson, generateLocalText, localModelConfiguration } from "./localModel";
import evidenceRouter from "./evidence/evidenceRouter";
import { runAdaptiveAgent } from "./agent/adaptiveAgent";
import { runVerificationPipeline } from "./verification/verificationPipeline";

dotenv.config({ path: path.resolve(process.cwd(), "server", ".env") });
dotenv.config({ path: path.resolve(process.cwd(), ".env") });

const app = express();
const PORT = process.env.PORT || 3100;
const HOST = process.env.HOST || "127.0.0.1";
app.use(cors({
  origin: ["http://127.0.0.1:5173", "http://localhost:5173"],
  methods: ["GET", "POST"],
}));
app.use(express.json({ limit: "10mb" }));
app.use("/api/evidence", evidenceRouter);

app.get("/health", (_req, res) =>
  res.json({ status: "ok", service: "graph-rag", version: "1.1.0", model: localModelConfiguration() })
);

interface Node {
  id: string;
  label: string;
  type: string;
  description: string;
  properties?: Record<string, unknown>;
  source_doc_id?: string;
  confidence?: number;
}

interface Link {
  source: string;
  target: string;
  relationship: string;
  type: string;
  reason: string;
  weight?: number;
}

interface GraphData {
  nodes: Node[];
  links: Link[];
}

interface SourceCitation {
  nodeId: string;
  nodeLabel: string;
  similarity: number;
  chunkId?: string;
  sourceDocId?: string;
  pageStart?: number;
  pageEnd?: number;
  chunkContent?: string;
}

const readGraphData = async () => {
  const [{ data: nodes, error: nodesError }, { data: links, error: linksError }] = await Promise.all([
    supabase.from("nodes").select("*"),
    supabase.from("links").select("*"),
  ]);
  if (nodesError) throw nodesError;
  if (linksError) throw linksError;
  return { nodes: nodes || [], links: links || [] };
};

async function insertChunk({
  nodeId,
  content,
  chunkIndex = 0,
  metadata = {},
  sourceDocId,
}: {
  nodeId: string;
  content: string;
  chunkIndex?: number;
  metadata?: Record<string, unknown>;
  sourceDocId?: string;
}) {
  const embedding = await generateEmbedding(content);
  if (!embedding) throw new Error(`No embedding generated for chunk linked to ${nodeId}`);

  const { error } = await supabase.from("chunks").insert({
    node_id: nodeId,
    content,
    embedding: `[${embedding.join(",")}]`,
    chunk_index: chunkIndex,
    source_url: sourceDocId ?? null,
    metadata,
  });
  if (error) throw error;
}

async function insertGraph(graph: GraphData, sourceDocId?: string) {
  let successCount = 0;

  for (const node of graph.nodes) {
    try {
      const embeddingText = `${node.label} (${node.type}): ${node.description}`;
      const embedding = await generateEmbedding(embeddingText);
      if (!embedding) continue;

      const formattedId = node.id.toLowerCase().replace(/\s+/g, "");
      const { error } = await supabase.from("nodes").upsert({
        id: formattedId,
        label: node.label,
        type: node.type,
        description: node.description,
        embedding: `[${embedding.join(",")}]`,
        properties: node.properties ?? {},
        source_doc_id: sourceDocId ?? node.source_doc_id ?? null,
        confidence: node.confidence ?? 1.0,
      });
      if (error) throw error;

      successCount += 1;
      await insertChunk({
        nodeId: formattedId,
        content: node.description,
        chunkIndex: 0,
        metadata: { label: node.label, type: node.type, auto: true },
        sourceDocId,
      });
    } catch (error) {
      console.error("Node persistence failed:", node.label, error);
    }
  }

  if (successCount > 0 && graph.links?.length) {
    const formattedLinks = graph.links.map((link) => ({
      source: link.source.toLowerCase().replace(/\s+/g, ""),
      target: link.target.toLowerCase().replace(/\s+/g, ""),
      relationship: link.relationship,
      type: link.type,
      reason: link.reason,
      weight: link.weight ?? 1.0,
    }));
    const { error } = await supabase.from("links").insert(formattedLinks);
    if (error) throw error;
  }
}

app.get("/api/graph", async (_req, res) => {
  try {
    res.json(await readGraphData());
  } catch (error) {
    console.error("Graph fetch failed:", error);
    res.status(500).json({ message: "Failed to retrieve graph data" });
  }
});

app.post("/api/graph/extract", async (req, res) => {
  const { text, source_doc_id } = req.body;
  if (!text) return res.status(400).json({ message: "Text required" });

  try {
    const parsedGraph = await generateLocalJson<GraphData>(`
Extract a structured technical knowledge graph from the text below.

For every NODE include:
- "id": unique identifier (lowercase, no spaces)
- "label": entity name
- "type": category
- "description": concise technical explanation

For every LINK include:
- "source": id of starting node
- "target": id of ending node
- "relationship": verb describing the relation
- "type": connection category
- "reason": brief explanation

Return ONLY valid JSON: { "nodes": [], "links": [] }

TEXT:
${text}
`);
    await insertGraph(parsedGraph, source_doc_id);
    res.json(parsedGraph);
  } catch (error) {
    console.error("Extraction error:", error);
    res.status(500).json({ message: "Extraction failed" });
  }
});

async function queryAdaptiveGraphAgent(query: string, onStatus?: (status: string) => void) {
  const totalStarted = Date.now();

  onStatus?.("Retrieving and expanding graph evidence...");
  const agentStarted = Date.now();
  const agent = await runAdaptiveAgent(query, { maxSteps: 6, maxRequeries: 1 });
  const state = agent.state;
  console.log(`[chat] adaptive agent: ${Date.now() - agentStarted}ms`);

  if (agent.decision === "abstain" || !state.chunks.length) {
    console.log(`[chat] total: ${Date.now() - totalStarted}ms (agent abstained)`);
    return {
      content: "I could not find enough source evidence in the uploaded knowledge base to answer this reliably.",
      sources: [],
      reasoningTrace: state.trace.map(
        (item) => `${item.step}:${item.tool} — ${item.reason} | ${item.observation}`
      ),
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
  console.log(`[chat] verification: ${Date.now() - verificationStarted}ms`);

  const verificationTrace = verification.trace.map((item, index) => `verify:${index + 1} — ${item}`);

  if (verification.available && verification.decision === "abstain") {
    console.log(`[chat] total: ${Date.now() - totalStarted}ms (verification abstained)`);
    return {
      content:
        "I found related material, but claim-level verification did not produce enough reliable support after the bounded retry, so I’m abstaining rather than presenting an uncertain answer.",
      sources: [],
      reasoningTrace: [
        ...state.trace.map((item) => `${item.step}:${item.tool} — ${item.reason} | ${item.observation}`),
        ...verificationTrace,
      ],
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

  const requestedNodeIds = [
    ...new Set([
      ...workingNodes.map((node) => node.id),
      ...workingChunks.map((chunk) => chunk.node_id),
      ...state.expandedNodeIds,
    ]),
  ].slice(0, 40);

  const { data: fullNodes, error: nodesError } = requestedNodeIds.length
    ? await supabase
        .from("nodes")
        .select("id,label,type,description,properties,confidence")
        .in("id", requestedNodeIds)
    : { data: [], error: null };
  if (nodesError) throw nodesError;

  const nodeContext =
    (fullNodes || [])
      .map(
        (node: any) =>
          `[NODE] ${node.label} (${node.type}) — confidence: ${node.confidence ?? 1}\n${node.description}${
            node.properties && Object.keys(node.properties).length
              ? `\nProperties: ${JSON.stringify(node.properties)}`
              : ""
          }`
      )
      .join("\n\n") || "None";

  const edgeContext =
    state.expandedEdges
      .map((edge) => `${edge.source} --[${edge.relationship}]--> ${edge.target}`)
      .join("\n") || "Graph expansion was not required by the agent.";

  const chunkContext = workingChunks
    .map((chunk, index) => {
      const metadata: any = chunk.metadata || {};
      const pageStart = metadata.pageStart;
      const pageEnd = metadata.pageEnd;
      const pages = pageStart
        ? ` | pages ${pageStart}${pageEnd && pageEnd !== pageStart ? `-${pageEnd}` : ""}`
        : "";
      return `[CHUNK ${index + 1} | id ${chunk.id} | node ${chunk.node_id}${pages} | score ${chunk.similarity.toFixed(3)}]\n${chunk.content}`;
    })
    .join("\n\n");

  const evidenceContext = supportedClaims.length
    ? supportedClaims
        .map((claim: any, index: number) => {
          const verificationItem = verification.results.find((item) => item.claimId === claim.id);
          const pages = claim.page_start
            ? ` pages ${claim.page_start}${claim.page_end && claim.page_end !== claim.page_start ? `-${claim.page_end}` : ""}`
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
            `[${item.label} | confidence ${item.confidence.toFixed(2)}] ${item.claimText}\nReason: ${item.reason}`
        )
        .join("\n\n") || "No contradicted or insufficient structured claims."
    : "Structured claim verification was unavailable; use raw chunks conservatively.";

  const agentTrace = state.trace
    .map(
      (item) =>
        `[STEP ${item.step}] TOOL=${item.tool}\nReason: ${item.reason}\nObservation: ${item.observation}`
    )
    .join("\n\n");

  const sources: SourceCitation[] = [
    ...workingNodes.map((node) => ({
      nodeId: node.id,
      nodeLabel: node.label,
      similarity: node.similarity,
    })),
    ...workingChunks.map((chunk) => {
      const metadata: any = chunk.metadata || {};
      return {
        nodeId: chunk.node_id,
        nodeLabel:
          (fullNodes || []).find((node: any) => node.id === chunk.node_id)?.label ?? chunk.node_id,
        similarity: chunk.similarity,
        chunkId: chunk.id,
        sourceDocId: metadata.sourceDocId || metadata.source_doc_id || undefined,
        pageStart: metadata.pageStart,
        pageEnd: metadata.pageEnd,
        chunkContent: chunk.content.slice(0, 160) + (chunk.content.length > 160 ? "…" : ""),
      };
    }),
  ];

  onStatus?.("Generating grounded answer...");
  const synthesisStarted = Date.now();
  const content = await generateLocalText(`
You are the synthesis component of a bounded evidence-grounded graph agent with claim verification.

=== AGENT TOOL TRACE ===
${agentTrace}

=== VERIFICATION SUMMARY ===
available=${verification.available}
retried=${verification.retried}
supported=${verification.summary.supported}
contradicted=${verification.summary.contradicted}
insufficient=${verification.summary.insufficient}
selective_score=${verification.summary.calibratedScore.toFixed(3)}

=== GRAPH NODES ===
${nodeContext}

=== GRAPH CONNECTIONS ===
${edgeContext}

=== VERIFIED SUPPORTED CLAIMS ===
${evidenceContext}

=== VERIFICATION FINDINGS TO TREAT AS WARNINGS ===
${verificationFindings}

=== ORIGINAL SOURCE CHUNKS ===
${chunkContext}

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
  console.log(`[chat] synthesis: ${Date.now() - synthesisStarted}ms`);
  console.log(`[chat] total: ${Date.now() - totalStarted}ms`);

  return {
    content,
    sources,
    reasoningTrace: [
      ...state.trace.map((item) => `${item.step}:${item.tool} — ${item.reason} | ${item.observation}`),
      ...verificationTrace,
    ],
  };
}

app.post("/api/chat", async (req, res) => {
  const { query } = req.body;
  if (!query) return res.status(400).json({ message: "Query required" });

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();

  const sendEvent = (data: object) => res.write(`data: ${JSON.stringify(data)}\n\n`);
  const requestStarted = Date.now();
  console.log(`[chat] query started: ${query.slice(0, 120)}`);

  sendEvent({ status: "Starting grounded graph search..." });
  const heartbeat = setInterval(() => {
    if (!res.writableEnded) sendEvent({ heartbeat: true });
  }, 8_000);

  try {
    const { content, sources, reasoningTrace } = await queryAdaptiveGraphAgent(query, (status) =>
      sendEvent({ status })
    );

    sendEvent({ status: "Streaming grounded answer..." });
    for (const word of content.split(" ")) {
      sendEvent({ token: `${word} ` });
      await new Promise((resolve) => setTimeout(resolve, 12));
    }
    sendEvent({ sources, reasoningTrace });
    res.write("data: [DONE]\n\n");
    res.end();
    console.log(`[chat] response completed: ${Date.now() - requestStarted}ms`);
  } catch (error: any) {
    console.error("Adaptive agent error:", error);
    sendEvent({ error: error.message || "Query failed" });
    res.end();
  } finally {
    clearInterval(heartbeat);
  }
});

app.post("/api/integration/retrieve", async (req, res) => {
  const query = typeof req.body?.query === "string" ? req.body.query.trim() : "";
  if (!query) return res.status(400).json({ message: "Query required" });
  const requestedMode = req.body?.verification_mode ?? "thorough";
  if (!["fast", "standard", "thorough"].includes(requestedMode)) {
    return res.status(400).json({ message: "verification_mode must be fast, standard, or thorough" });
  }
  const verificationMode = requestedMode as "fast" | "standard" | "thorough";
  const totalStarted = Date.now();

  try {
    const retrievalStarted = Date.now();
    const agent = await runAdaptiveAgent(query, { maxSteps: 6, maxRequeries: 1 });
    const retrievalMs = Date.now() - retrievalStarted;
    const state = agent.state;
    if (agent.decision === "abstain" || !state.chunks.length) {
      return res.json({
        status: "abstained",
        verification_mode: verificationMode,
        nodes: [],
        chunks: [],
        claims: [],
        verification: null,
        trace: state.trace,
        timings_ms: { retrieval: retrievalMs, verification: 0, total: Date.now() - totalStarted },
      });
    }

    const verificationStarted = Date.now();
    const verification = await runVerificationPipeline({
      originalQuery: query,
      nodes: state.nodes,
      chunks: state.chunks,
      claims: state.evidenceClaims,
      verificationMode,
    });
    const verificationMs = Date.now() - verificationStarted;
    const supportedClaimIds = new Set(
      verification.results
        .filter((item) => item.label === "SUPPORTED")
        .map((item) => item.claimId)
    );
    const claims = verification.available
      ? verification.claims.filter((claim) => supportedClaimIds.has(claim.id))
      : [];

    return res.json({
      status: verification.available && verification.decision === "abstain" ? "abstained" : "grounded",
      verification_mode: verificationMode,
      nodes: verification.nodes,
      chunks: verification.chunks,
      claims,
      verification: {
        available: verification.available,
        retried: verification.retried,
        decision: verification.decision,
        summary: verification.summary,
        results: verification.results,
      },
      trace: [...state.trace, ...verification.trace.map((observation, index) => ({
        step: state.trace.length + index + 1,
        tool: "verification",
        reason: "Verify structured claims against retrieved source chunks.",
        observation,
      }))],
      timings_ms: {
        retrieval: retrievalMs,
        verification: verificationMs,
        total: Date.now() - totalStarted,
      },
    });
  } catch (error: any) {
    console.error("Integration retrieval failed:", error);
    return res.status(503).json({
      message: "Graph evidence retrieval unavailable",
      detail: error?.message || "unknown error",
    });
  }
});

// Legacy endpoint retained for compatibility. Phase 3+ PDF ingestion uses /api/evidence/chunks/insert.
app.post("/api/chunks/insert", async (req, res) => {
  const { node_id, content, chunk_index, metadata, source_url } = req.body;
  if (!node_id || !content) {
    return res.status(400).json({ message: "node_id and content are required" });
  }

  try {
    await insertChunk({
      nodeId: node_id,
      content,
      chunkIndex: chunk_index ?? 0,
      metadata: metadata ?? {},
      sourceDocId: source_url,
    });
    res.json({ message: "Chunk inserted successfully" });
  } catch (error) {
    console.error("Chunk insert error:", error);
    res.status(500).json({ message: "Chunk insertion failed" });
  }
});

app.post("/api/graph/clear", async (_req, res) => {
  const evidenceTables = ["claim_relations", "claim_entities", "claim_evidence", "claims", "documents"];
  for (const table of evidenceTables) {
    const { error } = await supabase
      .from(table)
      .delete()
      .neq("created_at", "1900-01-01T00:00:00Z");
    if (error) console.warn(`Unable to clear optional evidence table ${table}:`, error.message);
  }

  await supabase.from("chunks").delete().neq("id", "00000000-0000-0000-0000-000000000000");
  await supabase.from("links").delete().neq("id", "00000000-0000-0000-0000-000000000000");
  await supabase.from("nodes").delete().neq("id", "");
  res.json({ message: "Graph, chunks, and evidence cleared" });
});

app.get("/", (_req, res) =>
  res.send("Adaptive evidence-grounded GraphRAG backend with claim verification running 🚀")
);

app.listen(Number(PORT), HOST, () => console.log(`Server running on http://${HOST}:${PORT}`));
