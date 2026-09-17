import assert from "node:assert/strict";
import {
  buildVerificationRetryQuery,
  rankClaimsForQuery,
  shouldAbstainAfterVerification,
  shouldRetryVerification,
  summarizeVerification,
} from "./verificationPolicy";
import { ClaimVerification } from "./types";

const make = (
  label: ClaimVerification["label"],
  confidence: number,
  id: string
): ClaimVerification => ({
  claimId: id,
  claimText: `claim ${id}`,
  label,
  confidence,
  reason: "test",
  supportingChunkIds: ["chunk-1"],
});

const strong = summarizeVerification([
  make("SUPPORTED", 0.9, "a"),
  make("SUPPORTED", 0.8, "b"),
  make("INSUFFICIENT", 0.8, "c"),
]);
assert.equal(strong.supported, 2);
assert.equal(strong.insufficient, 1);
assert.equal(shouldAbstainAfterVerification(strong), false);
assert.equal(shouldRetryVerification(strong, 0, 1), true);
assert.equal(shouldRetryVerification(strong, 1, 1), false);

const allInsufficient = summarizeVerification([
  make("INSUFFICIENT", 1, "a"),
  make("INSUFFICIENT", 1, "b"),
]);
assert.equal(shouldAbstainAfterVerification(allInsufficient), true);

const contradictionHeavy = summarizeVerification([
  make("SUPPORTED", 0.9, "a"),
  make("CONTRADICTED", 0.9, "b"),
  make("CONTRADICTED", 0.8, "c"),
]);
assert.equal(shouldAbstainAfterVerification(contradictionHeavy), true);

const empty = summarizeVerification([]);
assert.equal(empty.total, 0);
assert.equal(shouldAbstainAfterVerification(empty), false);

const retryQuery = buildVerificationRetryQuery("Does method X work?", [
  make("INSUFFICIENT", 0.7, "unresolved"),
]);
assert.match(retryQuery, /corroborating source evidence/i);
assert.match(retryQuery, /claim unresolved/i);

const ranked = rankClaimsForQuery([
  { id: "irrelevant", claim_text: "The model used 120B math tokens." },
  { id: "relevant", claim_text: "Pump vibration increases maintenance risk." },
], "What evidence is available about pump vibration and maintenance risk?");
assert.deepEqual(ranked.map((claim) => claim.id), ["relevant"]);

console.log("Phase 5 verification policy tests passed.");
