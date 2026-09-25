import assert from "node:assert/strict";
import { bm25Rank } from "../retrieval/bm25";
import { ndcgAtK } from "./metrics";
import { reciprocalRankFusion } from "../retrieval/rrf";
import { authorizeRetrievalRecord } from "../retrieval/hybridRetriever";

const docs = [
  { id: "a", text: "graph retrieval knowledge entity", value: { name: "a" } },
  { id: "b", text: "image filtering gaussian noise", value: { name: "b" } },
  { id: "c", text: "knowledge graph reasoning retrieval", value: { name: "c" } },
];

const bm25 = bm25Rank(docs, "knowledge graph retrieval", 3);
assert.equal(bm25.length >= 2, true);
assert.equal(["a", "c"].includes(bm25[0].id), true);

const fused = reciprocalRankFusion(
  [
    {
      name: "dense",
      items: [
        { id: "a", score: 0.9, value: { name: "a" } },
        { id: "b", score: 0.8, value: { name: "b" } },
      ],
    },
    {
      name: "bm25",
      items: [
        { id: "b", score: 5, value: { name: "b" } },
        { id: "a", score: 4, value: { name: "a" } },
      ],
    },
  ],
  2
);

assert.equal(fused.length, 2);
assert.equal(fused.every((item) => item.sources.length === 2), true);
assert.equal(new Set(fused.map((item) => item.id)).size, 2);
assert.equal(ndcgAtK(["a", "x", "b"], ["a", "b"], 3) > 0.9, true);
assert.equal(ndcgAtK(["x", "y"], ["a"], 2), 0);

const maintenanceScope = {
  organization_id: "apel",
  department_ids: ["maintenance"],
  workspace_ids: ["plant-a"],
  roles: ["ENGINEER"],
  user_id: "engineer-1",
  clearance: 2,
  cross_department: false,
};
assert.equal(authorizeRetrievalRecord({
  organization_id: "apel", workspace_id: "plant-a",
  department_id: "maintenance", classification: "confidential",
}, maintenanceScope), true);
assert.equal(authorizeRetrievalRecord({
  organization_id: "apel", workspace_id: "plant-a",
  department_id: "finance", classification: "confidential",
}, maintenanceScope), false);
assert.equal(authorizeRetrievalRecord({
  organization_id: "apel", workspace_id: "plant-a",
  department_id: "maintenance", classification: "restricted",
}, maintenanceScope), false);
assert.equal(authorizeRetrievalRecord({}, maintenanceScope), false);

console.log("Phase 2 retrieval core tests passed");
