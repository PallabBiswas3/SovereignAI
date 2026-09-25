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
  properties?: Record<string, unknown>;
}

export interface RetrievedChunk {
  id: string;
  node_id: string;
  content: string;
  metadata: Record<string, unknown>;
  similarity: number;
  retrievalSources: string[];
}

export interface RetrievalAuthorizationScope {
  organization_id: string;
  department_ids: string[];
  workspace_ids: string[];
  roles: string[];
  user_id: string;
  clearance: number;
  cross_department: boolean;
  fingerprint?: string;
}

const CLASSIFICATION_LEVELS: Record<string, number> = {
  public: 0,
  internal: 1,
  confidential: 2,
  restricted: 3,
};

function stringList(value: unknown): string[] {
  return Array.isArray(value) ? value.map(String) : [];
}

export function authorizeRetrievalRecord(
  metadata: Record<string, unknown>,
  scope: RetrievalAuthorizationScope
): boolean {
  // Integration retrieval is fail-closed: legacy/unscoped rows are never
  // candidates. The service credential bypasses RLS, so this application
  // boundary is mandatory before either dense or sparse scoring.
  const organizationId = String(metadata.organization_id || "");
  const workspaceId = String(metadata.workspace_id || "");
  if (!organizationId || !workspaceId) return false;
  if (scope.organization_id !== "*" && organizationId !== scope.organization_id) return false;
  if (!scope.workspace_ids.includes("*") && !scope.workspace_ids.includes(workspaceId)) return false;

  const classification = String(metadata.classification || "internal").toLowerCase();
  const requiredClearance = CLASSIFICATION_LEVELS[classification];
  if (requiredClearance === undefined || scope.clearance < requiredClearance) return false;

  const allowedUsers = stringList(metadata.allowed_users);
  const allowedRoles = stringList(metadata.allowed_roles).map((item) => item.toUpperCase());
  const ownerId = String(metadata.owner_id || "");
  const explicitAcl = allowedUsers.length > 0 || allowedRoles.length > 0;
  const explicitGrant = allowedUsers.includes(scope.user_id)
    || allowedRoles.some((role) => scope.roles.map((item) => item.toUpperCase()).includes(role))
    || ownerId === scope.user_id;
  if (explicitAcl && !explicitGrant) return false;

  const departmentId = String(metadata.department_id || metadata.department || "");
  if (
    departmentId
    && !scope.cross_department
    && !scope.department_ids.includes("*")
    && !scope.department_ids.includes(departmentId)
    && !explicitGrant
  ) return false;
  return true;
}

function parseEmbedding(value: unknown): number[] | null {
  if (Array.isArray(value)) {
    const parsed = value.map(Number);
    return parsed.every(Number.isFinite) ? parsed : null;
  }
  if (typeof value !== "string") return null;
  const parsed = value.replace(/^\[|\]$/g, "").split(",").map(Number);
  return parsed.length && parsed.every(Number.isFinite) ? parsed : null;
}

function cosineSimilarity(left: number[], right: number[]): number {
  if (!left.length || left.length !== right.length) return 0;
  let dot = 0;
  let leftNorm = 0;
  let rightNorm = 0;
  for (let index = 0; index < left.length; index += 1) {
    dot += left[index] * right[index];
    leftNorm += left[index] ** 2;
    rightNorm += right[index] ** 2;
  }
  return dot / Math.max(Math.sqrt(leftNorm) * Math.sqrt(rightNorm), Number.EPSILON);
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

export async function retrieveHybrid(
  query: string,
  candidateLimit = 20,
  authorizationScope?: RetrievalAuthorizationScope
) {
  const [allNodesResult, allChunksResult] = await Promise.all([
    supabase.from("nodes").select("id,label,description,type,properties,embedding"),
    supabase.from("chunks").select("id,node_id,content,metadata,embedding"),
  ]);

  if (allNodesResult.error) throw allNodesResult.error;
  if (allChunksResult.error) throw allChunksResult.error;

  const authorizedNodes = (allNodesResult.data || []).filter((node: any) =>
    !authorizationScope || authorizeRetrievalRecord(node.properties || {}, authorizationScope)
  );
  const authorizedChunks = (allChunksResult.data || []).filter((chunk: any) =>
    !authorizationScope || authorizeRetrievalRecord(chunk.metadata || {}, authorizationScope)
  );

  let denseNodes: Array<{ id: string; score: number; value: Record<string, unknown> }> = [];
  let denseChunks: Array<{ id: string; score: number; value: Record<string, unknown> }> = [];
  if (denseRetrievalEnabled()) {
    const embedding = await generateEmbedding(query);
    if (authorizationScope) {
      // Score only the already-authorized working set. This prevents a
      // privileged RPC from allowing forbidden rows to affect top-k.
      denseNodes = authorizedNodes.flatMap((node: any) => {
        const vector = parseEmbedding(node.embedding);
        return vector ? [{ id: node.id, score: cosineSimilarity(embedding, vector), value: node }] : [];
      }).sort((a, b) => b.score - a.score).slice(0, candidateLimit);
      denseChunks = authorizedChunks.flatMap((chunk: any) => {
        const vector = parseEmbedding(chunk.embedding);
        return vector ? [{ id: chunk.id, score: cosineSimilarity(embedding, vector), value: chunk }] : [];
      }).sort((a, b) => b.score - a.score).slice(0, candidateLimit);
    } else {
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
  }

  const bm25Nodes = bm25Rank(
    authorizedNodes.map((node: any) => ({
      id: node.id,
      text: `${node.label || ""} ${node.description || ""} ${node.type || ""}`,
      value: node as Record<string, unknown>,
    })),
    query,
    candidateLimit
  );

  const bm25Chunks = bm25Rank(
    authorizedChunks.map((chunk: any) => ({
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
