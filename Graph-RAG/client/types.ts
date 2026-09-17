/* =========================
   GRAPH TYPES
========================= */

export interface Node {
  id: string;
  label: string;
  type: string;
  description: string;
  // Enriched fields (from upgraded schema)
  properties?: Record<string, unknown>;
  source_doc_id?: string;
  confidence?: number;
  created_at?: string;
  updated_at?: string;
}

export interface Link {
  id?: string;
  source: string;
  target: string;
  relationship: string;
  type: string;
  reason: string;
  weight?: number;
  created_at?: string;
}

export interface GraphData {
  nodes: Node[];
  links: Link[];
}

export interface ExtractionResult extends GraphData { }

/* =========================
   CHAT TYPES
========================= */

export interface SourceCitation {
  nodeId: string;
  nodeLabel: string;
  similarity: number;
  chunkContent?: string;
}

export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  // Enriched fields for RAG responses
  sources?: SourceCitation[];
  reasoningTrace?: string[];
  confidence?: number;
  isStreaming?: boolean;
  timestamp?: number;
}

/* =========================
   INGESTION TYPES
========================= */

export interface IngestionProgress {
  total: number;
  current: number;
  status: "extracting" | "chunking" | "embedding" | "done" | "error";
  message: string;
}

export interface PDFChunk {
  content: string;
  chunkIndex: number;
  pageStart: number;
  pageEnd: number;
  metadata: {
    fileName: string;
    totalPages: number;
    charCount: number;
  };
}

export interface PDFExtractionResult {
  fullText: string;
  chunks: PDFChunk[];
  totalPages: number;
  fileName: string;
  sourceDocId: string;
}

/* =========================
   APP STATE TYPES
========================= */

export type View = "input" | "graph" | "chat";

export interface LoadingState {
  extract: boolean;
  rag: boolean;
  pdf: boolean;
}