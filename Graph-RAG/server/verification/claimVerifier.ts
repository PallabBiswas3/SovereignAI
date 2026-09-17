import { RetrievedChunk } from "../retrieval/hybridRetriever";
import { generateLocalJson } from "../localModel";
import { ClaimVerification, VerifiableClaim, VerificationLabel } from "./types";

export interface ClaimVerifierOptions {
  maxChunks: number;
  sourceCharsPerChunk: number;
  timeoutMs: number;
  maxTokens: number;
  concurrency: number;
}

const DEFAULT_OPTIONS: ClaimVerifierOptions = {
  maxChunks: 2,
  sourceCharsPerChunk: 2_000,
  timeoutMs: 45_000,
  maxTokens: 128,
  concurrency: 1,
};

function boundedSourceText(content: string, limit: number): string {
  if (content.length <= limit) return content;
  const headLength = Math.floor(limit * 0.7);
  const tailLength = limit - headLength;
  return `${content.slice(0, headLength)}\n[...source truncated...]\n${content.slice(-tailLength)}`;
}

function chunkMap(chunks: RetrievedChunk[]): Map<string, RetrievedChunk> {
  return new Map(chunks.map((chunk) => [chunk.id, chunk]));
}

function linkedChunksForClaim(claim: VerifiableClaim, chunks: RetrievedChunk[]): RetrievedChunk[] {
  const byId = chunkMap(chunks);
  const linkedIds = (claim.evidence || []).map((item) => item.chunk_id);
  const linked = linkedIds.map((id) => byId.get(id)).filter(Boolean) as RetrievedChunk[];
  return linked.length ? linked : chunks.slice(0, 3);
}

export async function verifyClaimAgainstChunks(
  claim: VerifiableClaim,
  chunks: RetrievedChunk[],
  options: ClaimVerifierOptions = DEFAULT_OPTIONS
): Promise<ClaimVerification> {
  const evidenceChunks = linkedChunksForClaim(claim, chunks).slice(0, options.maxChunks);
  const supportingChunkIds = evidenceChunks.map((chunk) => chunk.id);

  if (!evidenceChunks.length) {
    return {
      claimId: claim.id,
      claimText: claim.claim_text,
      label: "INSUFFICIENT",
      confidence: 1,
      reason: "No raw source chunks are available for verification.",
      supportingChunkIds: [],
    };
  }

  const evidenceText = evidenceChunks
    .map((chunk, index) => `[SOURCE ${index + 1} | chunk_id=${chunk.id}]\n${boundedSourceText(chunk.content, options.sourceCharsPerChunk)}`)
    .join("\n\n");

  try {
    const parsed = await generateLocalJson<{
      label?: VerificationLabel;
      confidence?: number;
      reason?: string;
    }>(`
You are a strict claim verifier.

Classify the CLAIM against ONLY the RAW SOURCE CHUNKS below.
Return JSON only:
{
  "label": "SUPPORTED | CONTRADICTED | INSUFFICIENT",
  "confidence": 0.0,
  "reason": "maximum 12 words"
}

Definitions:
- SUPPORTED: the source text directly entails the important factual content of the claim.
- CONTRADICTED: the source text directly conflicts with the claim.
- INSUFFICIENT: the source text does not establish either support or contradiction.

Rules:
- Do not use outside knowledge.
- Preserve negation, quantities, conditions, comparisons, and uncertainty.
- Mere topical similarity is not support.
- If a stronger or broader claim is not fully established, use INSUFFICIENT.

CLAIM:
${claim.claim_text}

RAW SOURCE CHUNKS:
${evidenceText}
`, options.timeoutMs, options.maxTokens);

    const validLabels: VerificationLabel[] = ["SUPPORTED", "CONTRADICTED", "INSUFFICIENT"];
    const label = validLabels.includes(parsed.label as VerificationLabel)
      ? (parsed.label as VerificationLabel)
      : "INSUFFICIENT";

    const confidence = Number(parsed.confidence ?? 0.5);
    return {
      claimId: claim.id,
      claimText: claim.claim_text,
      label,
      confidence: Number.isFinite(confidence) ? Math.max(0, Math.min(1, confidence)) : 0.5,
      reason: parsed.reason || "Verifier returned no explanation.",
      supportingChunkIds,
    };
  } catch (error: any) {
    return {
      claimId: claim.id,
      claimText: claim.claim_text,
      label: "INSUFFICIENT",
      confidence: 1,
      reason: `Verifier failed conservatively: ${error?.message || "unknown error"}`,
      supportingChunkIds,
    };
  }
}

export async function verifyClaimsAgainstChunks(
  claims: VerifiableClaim[],
  chunks: RetrievedChunk[],
  limit = 8,
  options: ClaimVerifierOptions = DEFAULT_OPTIONS
): Promise<ClaimVerification[]> {
  const selected = claims.slice(0, limit);
  if (!selected.length) return [];
  const results = new Array<ClaimVerification>(selected.length);
  let nextIndex = 0;
  const configuredValue = process.env.OLLAMA_VERIFY_CONCURRENCY
    ? Number(process.env.OLLAMA_VERIFY_CONCURRENCY)
    : options.concurrency;
  const configuredConcurrency = Number.isFinite(configuredValue)
    ? Math.max(1, Math.floor(configuredValue))
    : 1;
  const workerCount = Math.max(1, Math.min(configuredConcurrency, selected.length));
  const worker = async () => {
    while (true) {
      const index = nextIndex++;
      if (index >= selected.length) return;
      results[index] = await verifyClaimAgainstChunks(selected[index], chunks, options);
    }
  };
  await Promise.all(Array.from({ length: workerCount }, () => worker()));
  return results;
}
