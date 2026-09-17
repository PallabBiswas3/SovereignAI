import { ExtractionResult, GraphData, ChatMessage } from "../types";
import { PDFChunk } from "./pdfService";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "http://localhost:3100").replace(/\/$/, "");

export type EvaluationMode = "fixed" | "adaptive" | "adaptive_verified";

export interface SourceCitation {
  nodeId: string;
  nodeLabel: string;
  similarity: number;
  chunkId?: string;
  sourceDocId?: string;
  pageStart?: number;
  pageEnd?: number;
  chunkContent?: string;
}

export interface EvaluationMetadata {
  abstained: boolean;
  timings: {
    retrievalMs?: number;
    graphExpansionMs?: number;
    agentMs?: number;
    verificationMs?: number;
    synthesisMs?: number;
    totalMs: number;
  };
  counts: {
    nodes: number;
    chunks: number;
    expandedEdges: number;
    claims: number;
    supported?: number;
    contradicted?: number;
    insufficient?: number;
  };
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

export interface ChatResponse {
  role: "assistant";
  content: string;
  sources?: SourceCitation[];
  reasoningTrace?: string[];
  confidence?: number;
  mode?: EvaluationMode;
  evaluation?: EvaluationMetadata;
}

export interface IngestionProgress {
  total: number;
  current: number;
  status: "extracting" | "chunking" | "embedding" | "done" | "error";
  message: string;
}

export const extractKnowledgeGraph = async (
  text: string,
  sourceDocId?: string
): Promise<ExtractionResult> => {
  const response = await fetch(`${API_BASE_URL}/api/graph/extract`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text, source_doc_id: sourceDocId }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || "Failed to extract knowledge graph");
  }

  return response.json();
};

const normalizeNodeId = (id: string): string => id.toLowerCase().replace(/\s+/g, "");

const tokenize = (text: string): Set<string> =>
  new Set(
    (text.toLowerCase().match(/[a-z0-9]+/g) || []).filter((token) => token.length > 2)
  );

export const selectAnchorNodeId = (chunk: PDFChunk, graph: GraphData): string | null => {
  if (!graph.nodes.length) return null;

  const chunkText = chunk.content.toLowerCase();
  const chunkTerms = tokenize(chunk.content);

  let bestNode = graph.nodes[0];
  let bestScore = -1;

  for (const node of graph.nodes) {
    const nodeTerms = tokenize(`${node.label} ${node.description || ""}`);
    let overlap = 0;

    for (const term of nodeTerms) {
      if (chunkTerms.has(term)) overlap += 1;
    }

    const normalizedLabel = node.label.trim().toLowerCase();
    const labelBonus = normalizedLabel.length >= 3 && chunkText.includes(normalizedLabel) ? 3 : 0;
    const confidenceBonus = (node.confidence ?? 0) * 0.05;
    const score = overlap + labelBonus + confidenceBonus;

    if (score > bestScore) {
      bestScore = score;
      bestNode = node;
    }
  }

  return normalizeNodeId(bestNode.id);
};

export const ingestPDFChunks = async (
  chunks: PDFChunk[],
  graph: GraphData,
  sourceDocId: string,
  onProgress?: (progress: IngestionProgress) => void
): Promise<void> => {
  const total = chunks.length;

  if (!graph.nodes.length) {
    throw new Error("Knowledge graph extraction produced no nodes; chunks cannot be linked safely.");
  }

  let failures = 0;
  let evidenceDeferred = 0;

  for (let i = 0; i < chunks.length; i++) {
    const chunk = chunks[i];
    const anchorNodeId = selectAnchorNodeId(chunk, graph);

    if (!anchorNodeId) {
      failures += 1;
      continue;
    }

    onProgress?.({
      total,
      current: i + 1,
      status: "embedding",
      message: `Embedding and grounding chunk ${i + 1} of ${total} (pages ${chunk.pageStart}–${chunk.pageEnd})`,
    });

    const response = await fetch(`${API_BASE_URL}/api/evidence/chunks/insert`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        node_id: anchorNodeId,
        content: chunk.content,
        chunk_index: chunk.chunkIndex,
        source_url: sourceDocId,
        metadata: {
          ...chunk.metadata,
          pageStart: chunk.pageStart,
          pageEnd: chunk.pageEnd,
          sourceDocId,
          anchorStrategy: "lexical-node-evidence-v1",
        },
      }),
    });

    if (!response.ok) {
      failures += 1;
      console.error(`Failed to insert chunk ${i + 1}`);
    } else {
      const result = await response.json();
      if (result.evidenceStatus === "deferred") {
        evidenceDeferred += 1;
        console.warn(`Evidence extraction deferred for chunk ${i + 1}:`, result.evidenceWarning);
      }
    }

    await new Promise((resolve) => setTimeout(resolve, 300));
  }

  if (failures > 0) {
    throw new Error(`${failures} of ${total} chunks failed to ingest.`);
  }

  onProgress?.({
    total,
    current: total,
    status: "done",
    message:
      evidenceDeferred > 0
        ? `${total} chunks ingested; evidence deferred for ${evidenceDeferred} until the Phase 3 migration/service is available`
        : `All ${total} chunks ingested with evidence provenance`,
  });
};

async function consumeChatStream(
  response: Response,
  onToken: (token: string) => void,
  onDone: (response: ChatResponse) => void,
  onStatus?: (status: string) => void
): Promise<void> {
  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("text/event-stream")) {
    const data: ChatResponse = await response.json();
    const words = data.content.split(" ");

    for (const word of words) {
      onToken(word + " ");
      await new Promise((resolve) => setTimeout(resolve, 18));
    }

    onDone({
      ...data,
      confidence:
        data.sources && data.sources.length > 0
          ? data.sources.reduce((acc, source) => acc + source.similarity, 0) /
            data.sources.length
          : undefined,
    });
    return;
  }

  if (!response.body) throw new Error("Chat stream body is unavailable");

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let fullContent = "";
  let sources: SourceCitation[] = [];
  let reasoningTrace: string[] = [];
  let mode: EvaluationMode | undefined;
  let evaluation: EvaluationMetadata | undefined;
  let pending = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    pending += decoder.decode(value, { stream: true });
    const lines = pending.split("\n");
    pending = lines.pop() || "";

    for (const line of lines) {
      if (!line.startsWith("data: ")) continue;

      const data = line.slice(6).trim();
      if (!data || data === "[DONE]") continue;

      try {
        const parsed = JSON.parse(data);
        if (parsed.error) throw new Error(parsed.error);
        if (typeof parsed.status === "string") onStatus?.(parsed.status);
        if (parsed.heartbeat) continue;
        if (parsed.token) {
          fullContent += parsed.token;
          onToken(parsed.token);
        }
        if (parsed.sources) sources = parsed.sources;
        if (parsed.reasoningTrace) reasoningTrace = parsed.reasoningTrace;
        if (parsed.mode) mode = parsed.mode;
        if (parsed.evaluation) evaluation = parsed.evaluation;
      } catch (error) {
        if (error instanceof Error && data.startsWith("{")) throw error;
        fullContent += data;
        onToken(data);
      }
    }
  }

  onDone({
    role: "assistant",
    content: fullContent.trimEnd(),
    sources,
    reasoningTrace,
    mode,
    evaluation,
    confidence:
      sources.length > 0
        ? sources.reduce((acc, source) => acc + source.similarity, 0) / sources.length
        : undefined,
  });
}

export const queryGraphRAGStream = async (
  query: string,
  onToken: (token: string) => void,
  onDone: (response: ChatResponse) => void,
  onError: (error: string) => void,
  onStatus?: (status: string) => void
): Promise<void> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/chat`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || "Failed to query knowledge graph");
    }

    await consumeChatStream(response, onToken, onDone, onStatus);
  } catch (err: any) {
    onError(err.message || "Unknown error");
  }
};

export const queryEvaluationStream = async (
  query: string,
  mode: EvaluationMode,
  onToken: (token: string) => void,
  onDone: (response: ChatResponse) => void,
  onError: (error: string) => void,
  onStatus?: (status: string) => void
): Promise<void> => {
  try {
    const response = await fetch(`${API_BASE_URL}/api/evidence/evaluate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, mode }),
    });

    if (!response.ok) {
      const errorData = await response.json();
      throw new Error(errorData.message || "Failed to run evaluation mode");
    }

    await consumeChatStream(response, onToken, onDone, onStatus);
  } catch (err: any) {
    onError(err.message || "Unknown error");
  }
};

export const queryGraphRAG = async (query: string): Promise<ChatMessage> => {
  const response = await fetch(`${API_BASE_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    const errorData = await response.json();
    throw new Error(errorData.message || "Failed to query knowledge graph");
  }

  return response.json();
};

export const fetchGraphData = async (): Promise<GraphData> => {
  const response = await fetch(`${API_BASE_URL}/api/graph`);
  if (!response.ok) throw new Error("Failed to fetch graph data");
  return response.json();
};

export const clearGraphData = async (): Promise<void> => {
  const response = await fetch(`${API_BASE_URL}/api/graph/clear`, {
    method: "POST",
  });
  if (!response.ok) throw new Error("Failed to clear graph data");
};
