import { RetrievedChunk } from "../retrieval/hybridRetriever";

export type VerificationLabel = "SUPPORTED" | "CONTRADICTED" | "INSUFFICIENT";
export type VerificationMode = "fast" | "standard" | "thorough";

export interface ClaimVerification {
  claimId: string;
  claimText: string;
  label: VerificationLabel;
  confidence: number;
  reason: string;
  supportingChunkIds: string[];
}

export interface VerificationSummary {
  total: number;
  supported: number;
  contradicted: number;
  insufficient: number;
  supportRatio: number;
  contradictionRatio: number;
  evidenceCoverage: number;
  meanSupportedConfidence: number;
  calibratedScore: number;
}

export interface VerificationOutcome {
  available: boolean;
  retried: boolean;
  decision: "continue" | "abstain";
  results: ClaimVerification[];
  summary: VerificationSummary;
  trace: string[];
}

export interface VerifiableClaim {
  id: string;
  claim_text: string;
  evidence?: Array<{ chunk_id: string }>;
}

export interface VerificationContext {
  claims: VerifiableClaim[];
  chunks: RetrievedChunk[];
}
