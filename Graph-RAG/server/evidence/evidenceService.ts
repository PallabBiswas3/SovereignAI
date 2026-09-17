import { supabase } from "../supabase";
import { generateLocalJson } from "../localModel";

export interface ExtractedClaim {
  text: string;
  confidence: number;
  polarity: "affirmed" | "negated" | "uncertain";
  entities: string[];
}

export interface EvidenceIngestionInput {
  chunkId: string;
  nodeId: string;
  content: string;
  sourceDocId?: string;
  pageStart?: number;
  pageEnd?: number;
  fileName?: string;
  totalPages?: number;
}

const normalizeText = (text: string): string =>
  text
    .toLowerCase()
    .replace(/[^a-z0-9\s]/g, " ")
    .replace(/\s+/g, " ")
    .trim();

const tokenize = (text: string): Set<string> =>
  new Set(normalizeText(text).split(" ").filter((token) => token.length > 2));

const jaccard = (a: string, b: string): number => {
  const aa = tokenize(a);
  const bb = tokenize(b);
  if (!aa.size || !bb.size) return 0;
  const intersection = [...aa].filter((token) => bb.has(token)).length;
  const union = new Set([...aa, ...bb]).size;
  return union ? intersection / union : 0;
};

export async function extractAtomicClaims(content: string): Promise<ExtractedClaim[]> {
  const parsed = await generateLocalJson<{ claims?: ExtractedClaim[] }>(`
Extract only claims explicitly stated in the source text.

Return JSON only in this shape:
{
  "claims": [
    {
      "text": "one atomic factual claim",
      "confidence": 0.0,
      "polarity": "affirmed | negated | uncertain",
      "entities": ["entity names explicitly mentioned"]
    }
  ]
}

Rules:
- Each claim must express exactly one factual proposition.
- Do not infer unstated facts.
- Preserve numerical qualifiers, uncertainty, and negation.
- confidence means confidence that the claim is explicitly supported by the source text.
- If the text contains no meaningful factual claims, return an empty claims array.

SOURCE TEXT:
${content}
`);
  return (parsed.claims || [])
    .filter((claim) => claim.text && claim.text.trim().length > 0)
    .map((claim) => ({
      text: claim.text.trim(),
      confidence: Math.max(0, Math.min(1, Number(claim.confidence ?? 1))),
      polarity: ["affirmed", "negated", "uncertain"].includes(claim.polarity)
        ? claim.polarity
        : "uncertain",
      entities: Array.isArray(claim.entities) ? claim.entities.filter(Boolean) : [],
    }));
}

async function ensureDocument(input: EvidenceIngestionInput) {
  if (!input.sourceDocId) return;
  const { error } = await supabase.from("documents").upsert({
    id: input.sourceDocId,
    file_name: input.fileName ?? null,
    total_pages: input.totalPages ?? null,
    metadata: {},
  });
  if (error) throw error;
}

async function resolveEntityIds(entityNames: string[], fallbackNodeId: string): Promise<string[]> {
  const names = [...new Set(entityNames.map((name) => name.trim()).filter(Boolean))];
  if (!names.length) return [fallbackNodeId];

  const resolved = new Set<string>();
  for (const name of names) {
    const { data, error } = await supabase
      .from("nodes")
      .select("id,label")
      .ilike("label", name)
      .limit(3);
    if (error) throw error;
    for (const row of data || []) resolved.add(row.id);
  }

  if (!resolved.size) resolved.add(fallbackNodeId);
  return [...resolved];
}

async function createClaimRelations(
  newClaimId: string,
  newClaimText: string,
  polarity: ExtractedClaim["polarity"],
  entityIds: string[]
) {
  if (!entityIds.length) return;

  const { data: candidateLinks, error: linkError } = await supabase
    .from("claim_entities")
    .select("claim_id")
    .in("entity_id", entityIds)
    .neq("claim_id", newClaimId)
    .limit(30);
  if (linkError) throw linkError;

  const candidateIds = [...new Set((candidateLinks || []).map((row: any) => row.claim_id))];
  if (!candidateIds.length) return;

  const { data: candidates, error: claimError } = await supabase
    .from("claims")
    .select("id,claim_text,polarity")
    .in("id", candidateIds);
  if (claimError) throw claimError;

  const relations = [];
  for (const candidate of candidates || []) {
    const overlap = jaccard(newClaimText, candidate.claim_text);
    if (overlap < 0.6) continue;

    if (candidate.polarity === polarity && polarity !== "uncertain") {
      relations.push({
        source_claim_id: newClaimId,
        target_claim_id: candidate.id,
        relation: "SUPPORTS",
        confidence: overlap,
        reason: "High lexical-semantic overlap with matching polarity",
      });
    } else if (
      polarity !== "uncertain" &&
      candidate.polarity !== "uncertain" &&
      candidate.polarity !== polarity
    ) {
      relations.push({
        source_claim_id: newClaimId,
        target_claim_id: candidate.id,
        relation: "CONTRADICTS",
        confidence: overlap,
        reason: "High lexical-semantic overlap with opposite polarity",
      });
    }
  }

  if (relations.length) {
    const { error } = await supabase.from("claim_relations").upsert(relations);
    if (error) throw error;
  }
}

export async function ingestEvidenceForChunk(input: EvidenceIngestionInput): Promise<number> {
  await ensureDocument(input);
  const claims = await extractAtomicClaims(input.content);
  let inserted = 0;

  for (const claim of claims) {
    const normalizedText = normalizeText(claim.text);
    const { data: insertedClaim, error: claimError } = await supabase
      .from("claims")
      .insert({
        claim_text: claim.text,
        normalized_text: normalizedText,
        source_doc_id: input.sourceDocId ?? null,
        page_start: input.pageStart ?? null,
        page_end: input.pageEnd ?? null,
        extraction_confidence: claim.confidence,
        polarity: claim.polarity,
        metadata: {},
      })
      .select("id")
      .single();
    if (claimError) throw claimError;

    const claimId = insertedClaim.id as string;
    const entityIds = await resolveEntityIds(claim.entities, input.nodeId);

    const { error: evidenceError } = await supabase.from("claim_evidence").insert({
      claim_id: claimId,
      chunk_id: input.chunkId,
      relation: "SUPPORTED_BY",
      confidence: claim.confidence,
    });
    if (evidenceError) throw evidenceError;

    const entityRows = entityIds.map((entityId) => ({
      claim_id: claimId,
      entity_id: entityId,
      confidence: claim.confidence,
    }));
    if (entityRows.length) {
      const { error: entityError } = await supabase.from("claim_entities").upsert(entityRows);
      if (entityError) throw entityError;
    }

    await createClaimRelations(claimId, claim.text, claim.polarity, entityIds);
    inserted += 1;
  }

  return inserted;
}

export async function fetchEvidenceForChunks(chunkIds: string[]) {
  if (!chunkIds.length) return [];

  const { data: links, error: linkError } = await supabase
    .from("claim_evidence")
    .select("claim_id,chunk_id,relation,confidence")
    .in("chunk_id", chunkIds);
  if (linkError) throw linkError;

  const claimIds = [...new Set((links || []).map((row: any) => row.claim_id))];
  if (!claimIds.length) return [];

  const [{ data: claims, error: claimError }, { data: entities, error: entityError }] = await Promise.all([
    supabase
      .from("claims")
      .select("id,claim_text,source_doc_id,page_start,page_end,extraction_confidence,polarity")
      .in("id", claimIds),
    supabase.from("claim_entities").select("claim_id,entity_id,confidence").in("claim_id", claimIds),
  ]);
  if (claimError) throw claimError;
  if (entityError) throw entityError;

  const byClaim = new Map<string, string[]>();
  for (const row of entities || []) {
    const ids = byClaim.get(row.claim_id) || [];
    ids.push(row.entity_id);
    byClaim.set(row.claim_id, ids);
  }

  return (claims || []).map((claim: any) => ({
    ...claim,
    entityIds: byClaim.get(claim.id) || [],
    evidence: (links || []).filter((link: any) => link.claim_id === claim.id),
  }));
}
