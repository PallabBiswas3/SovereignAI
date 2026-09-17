import { generateLocalEmbeddings, localModelConfiguration } from "../localModel";
import { supabase } from "../supabase";

type NodeRow = { id: string; label: string; type: string | null; description: string | null };
type ChunkRow = { id: string; content: string };

const BATCH_SIZE = 16;

async function reindexNodes(): Promise<number> {
  const { data, error } = await supabase.from("nodes").select("id,label,type,description");
  if (error) throw error;
  const rows = (data || []) as NodeRow[];
  for (let offset = 0; offset < rows.length; offset += BATCH_SIZE) {
    const batch = rows.slice(offset, offset + BATCH_SIZE);
    const embeddings = await generateLocalEmbeddings(
      batch.map((row) => `${row.label} (${row.type || "entity"}): ${row.description || ""}`)
    );
    for (let index = 0; index < batch.length; index += 1) {
      const { error: updateError } = await supabase
        .from("nodes")
        .update({ embedding: `[${embeddings[index].join(",")}]` })
        .eq("id", batch[index].id);
      if (updateError) throw updateError;
    }
    console.log(`Re-indexed nodes ${Math.min(offset + BATCH_SIZE, rows.length)}/${rows.length}`);
  }
  return rows.length;
}

async function reindexChunks(): Promise<number> {
  const { data, error } = await supabase.from("chunks").select("id,content");
  if (error) throw error;
  const rows = (data || []) as ChunkRow[];
  for (let offset = 0; offset < rows.length; offset += BATCH_SIZE) {
    const batch = rows.slice(offset, offset + BATCH_SIZE);
    const embeddings = await generateLocalEmbeddings(batch.map((row) => row.content));
    for (let index = 0; index < batch.length; index += 1) {
      const { error: updateError } = await supabase
        .from("chunks")
        .update({ embedding: `[${embeddings[index].join(",")}]` })
        .eq("id", batch[index].id);
      if (updateError) throw updateError;
    }
    console.log(`Re-indexed chunks ${Math.min(offset + BATCH_SIZE, rows.length)}/${rows.length}`);
  }
  return rows.length;
}

async function run() {
  if (!process.argv.includes("--apply")) {
    throw new Error("Re-indexing overwrites stored vectors. Re-run with --apply to confirm.");
  }
  console.log("Local embedding configuration:", localModelConfiguration());
  const nodes = await reindexNodes();
  const chunks = await reindexChunks();
  console.log(`Local re-index complete: ${nodes} nodes, ${chunks} chunks.`);
}

run().catch((error) => {
  console.error("Local re-index failed:", error);
  process.exitCode = 1;
});
