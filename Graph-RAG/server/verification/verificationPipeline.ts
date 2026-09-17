import { fetchEvidenceForChunks } from "../evidence/evidenceService";
import { retrieveHybrid, RetrievedChunk, RetrievedNode } from "../retrieval/hybridRetriever";
import { verifyClaimsAgainstChunks } from "./claimVerifier";
import {
  buildVerificationRetryQuery,
  rankClaimsForQuery,
  shouldAbstainAfterVerification,
  shouldRetryVerification,
  summarizeVerification,
} from "./verificationPolicy";
import { VerificationMode, VerificationOutcome, VerifiableClaim } from "./types";

export const VERIFICATION_PROFILES: Record<VerificationMode, {
  claimLimit: number;
  maxRetries: number;
  maxChunks: number;
  sourceCharsPerChunk: number;
  timeoutMs: number;
  maxTokens: number;
  concurrency: number;
}> = {
  fast: { claimLimit: 1, maxRetries: 0, maxChunks: 1, sourceCharsPerChunk: 800, timeoutMs: 15_000, maxTokens: 96, concurrency: 1 },
  standard: { claimLimit: 1, maxRetries: 0, maxChunks: 1, sourceCharsPerChunk: 1_200, timeoutMs: 25_000, maxTokens: 96, concurrency: 1 },
  thorough: { claimLimit: 4, maxRetries: 1, maxChunks: 2, sourceCharsPerChunk: 2_000, timeoutMs: 35_000, maxTokens: 128, concurrency: 2 },
};

function mergeById<T extends { id: string }>(existing: T[], incoming: T[]): T[] {
  const merged = new Map<string, T>();
  for (const item of existing) merged.set(item.id, item);
  for (const item of incoming) merged.set(item.id, item);
  return [...merged.values()];
}

export interface VerificationPipelineInput {
  originalQuery: string;
  nodes: RetrievedNode[];
  chunks: RetrievedChunk[];
  claims: VerifiableClaim[];
  maxRetries?: number;
  verificationMode?: VerificationMode;
}

export interface VerificationPipelineResult extends VerificationOutcome {
  nodes: RetrievedNode[];
  chunks: RetrievedChunk[];
  claims: VerifiableClaim[];
}

export async function runVerificationPipeline(
  input: VerificationPipelineInput
): Promise<VerificationPipelineResult> {
  const verificationMode = input.verificationMode ?? "thorough";
  const profile = VERIFICATION_PROFILES[verificationMode];
  const maxRetries = input.maxRetries ?? profile.maxRetries;
  let nodes = input.nodes;
  let chunks = input.chunks;
  let claims = rankClaimsForQuery(input.claims, input.originalQuery);
  let retryCount = 0;
  const trace: string[] = [];

  if (!claims.length) {
    if (input.claims.length) {
      return {
        available: true,
        retried: false,
        decision: "abstain",
        results: [],
        summary: summarizeVerification([]),
        trace: ["Structured claims were source-linked but not relevant to the query; verification abstained."],
        nodes,
        chunks,
        claims: [],
      };
    }
    return {
      available: false,
      retried: false,
      decision: "continue",
      results: [],
      summary: summarizeVerification([]),
      trace: ["No structured claims were available; raw chunks remain the grounding source."],
      nodes,
      chunks,
      claims,
    };
  }

  trace.push(`Verification mode=${verificationMode}; claim limit=${profile.claimLimit}; retry budget=${maxRetries}.`);
  let results = await verifyClaimsAgainstChunks(claims, chunks, profile.claimLimit, profile);
  let summary = summarizeVerification(results);
  trace.push(
    `Initial verification: supported=${summary.supported}, contradicted=${summary.contradicted}, insufficient=${summary.insufficient}, score=${summary.calibratedScore.toFixed(3)}.`
  );

  if (shouldRetryVerification(summary, retryCount, maxRetries)) {
    retryCount += 1;
    const retryQuery = buildVerificationRetryQuery(input.originalQuery, results);
    const retrieval = await retrieveHybrid(retryQuery, 20);
    nodes = mergeById(nodes, retrieval.nodes);
    chunks = mergeById(chunks, retrieval.chunks);

    try {
      claims = rankClaimsForQuery(
        (await fetchEvidenceForChunks(chunks.map((chunk) => chunk.id))) as VerifiableClaim[],
        input.originalQuery
      );
    } catch {
      // If evidence tables are unavailable, retain the previous structured claims and
      // still let the raw new chunks contribute to claim verification.
    }

    results = await verifyClaimsAgainstChunks(claims, chunks, profile.claimLimit, profile);
    summary = summarizeVerification(results);
    trace.push(
      `Verification retry '${retryQuery}' produced ${retrieval.chunks.length} chunks; supported=${summary.supported}, contradicted=${summary.contradicted}, insufficient=${summary.insufficient}, score=${summary.calibratedScore.toFixed(3)}.`
    );
  }

  const decision = !claims.length || shouldAbstainAfterVerification(summary) ? "abstain" : "continue";
  trace.push(
    decision === "abstain"
      ? "Verification policy selected abstention after bounded retry."
      : "Verification policy accepted the supported claim subset for synthesis."
  );

  return {
    available: true,
    retried: retryCount > 0,
    decision,
    results,
    summary,
    trace,
    nodes,
    chunks,
    claims,
  };
}
