import { supabase } from "../supabase";
import { generateEmbedding } from "../embedding";
import { bm25Rank } from "./bm25";
import { reciprocalRankFusion } from "./rrf";

export interface RetrievedNode {
  id: string;
  label: string;
  description: string;
  type?: string;
  similarity: number;
  retrievalSources: string[];
}

export interface RetrievedChunk {
  id: string;
  node_id: string;
  content: string;
  metadata: Record<string, unknown>;
  similarity: number;
  retrievalSources: string[];
}

export function denseRetrievalEnabled(): boolean {
  return process.env.GRAPHRAG_DENSE_RETRIEVAL_ENABLED?.toLowerCase() === "true";
}

function rerankByQueryOverlap<T extends { similarity: number }>(
  items: T[],
  query: string,
  textOf: (item: T) => string
): T[] {
  const queryTerms = [
    ...new Set(
      (query.toLowerCase().match(/[a-z0-9]+/g) || []).filter((word) => word.length > 2)
    ),
  ];

  return items
    .map((item) => {
      const text = textOf(item).toLowerCase();
      const matches = queryTerms.filter((term) => text.includes(term)).length;
      const overlap = matches / Math.max(queryTerms.length, 1);
      return { ...item, similarity: item.similarity * 0.8 + overlap * 0.2 };
    })
    .sort((a, b) => b.similarity - a.similarity);
}

function normalizeFusionScores<T extends { similarity: number }>(items: T[]): T[] {
  const maxScore = Math.max(...items.map((item) => item.similarity), 0);
  if (maxScore <= 0) return items;
  return items.map((item) => ({ ...item, similarity: item.similarity / maxScore }));
}

export async function retrieveHybrid(query: string, candidateLimit = 20) {
  const [allNodesResult, allChunksResult] = await Promise.all([
    supabase.from("nodes").select("id,label,description,type"),
    supabase.from("chunks").select("id,node_id,content,metadata"),
  ]);

  if (allNodesResult.error) throw allNodesResult.error;
  if (allChunksResult.error) throw allChunksResult.error;

  let denseNodes: Array<{ id: string; score: number; value: Record<string, unknown> }> = [];
  let denseChunks: Array<{ id: string; score: number; value: Record<string, unknown> }> = [];
  if (denseRetrievalEnabled()) {
    const embedding = await generateEmbedding(query);
    const [denseNodesResult, denseChunksResult] = await Promise.all([
      supabase.rpc("match_nodes", { query_embedding: embedding, match_count: candidateLimit }),
      supabase.rpc("match_chunks", { query_embedding: embedding, match_count: candidateLimit }),
    ]);
    if (denseNodesResult.error) throw denseNodesResult.error;
    if (denseChunksResult.error) throw denseChunksResult.error;
    denseNodes = (denseNodesResult.data || []).map((node: any) => ({
      id: node.id, score: node.similarity ?? 0, value: node as Record<string, unknown>,
    }));
    denseChunks = (denseChunksResult.data || []).map((chunk: any) => ({
      id: chunk.id, score: chunk.similarity ?? 0, value: chunk as Record<string, unknown>,
    }));
  }

  const bm25Nodes = bm25Rank(
    (allNodesResult.data || []).map((node: any) => ({
      id: node.id,
      text: `${node.label || ""} ${node.description || ""} ${node.type || ""}`,
      value: node as Record<string, unknown>,
    })),
    query,
    candidateLimit
  );

  const bm25Chunks = bm25Rank(
    (allChunksResult.data || []).map((chunk: any) => ({
      id: chunk.id,
      text: chunk.content || "",
      value: chunk as Record<string, unknown>,
    })),
    query,
    candidateLimit
  );

  const fusedNodes = reciprocalRankFusion<Record<string, unknown>>(
    [
      ...(denseRetrievalEnabled() ? [{ name: "dense", items: denseNodes }] : []),
      { name: "bm25", items: bm25Nodes },
    ],
    candidateLimit
  ).map((item) => ({
    ...(item.value as Record<string, unknown>),
    similarity: item.score,
    retrievalSources: item.sources,
  })) as unknown as RetrievedNode[];

  const fusedChunks = reciprocalRankFusion<Record<string, unknown>>(
    [
      ...(denseRetrievalEnabled() ? [{ name: "dense", items: denseChunks }] : []),
      { name: "bm25", items: bm25Chunks },
    ],
    candidateLimit
  ).map((item) => ({
    ...(item.value as Record<string, unknown>),
    similarity: item.score,
    retrievalSources: item.sources,
  })) as unknown as RetrievedChunk[];

  const normalizedNodes = normalizeFusionScores(fusedNodes);
  const normalizedChunks = normalizeFusionScores(fusedChunks);

  return {
    nodes: rerankByQueryOverlap(
      normalizedNodes,
      query,
      (node) => `${node.label} ${node.description}`
    ).slice(0, 5),
    chunks: rerankByQueryOverlap(normalizedChunks, query, (chunk) => chunk.content).slice(0, 6),
  };
}
