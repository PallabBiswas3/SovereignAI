import assert from "node:assert/strict";
import { bm25Rank } from "../retrieval/bm25";
import { reciprocalRankFusion } from "../retrieval/rrf";

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

console.log("Phase 2 retrieval core tests passed");
