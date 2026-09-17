import { Router } from "express";
import { generateEmbedding } from "../embedding";
import { supabase } from "../supabase";
import { fetchEvidenceForChunks, ingestEvidenceForChunk } from "./evidenceService";
import {
  parseEvaluationMode,
  runEvaluationQuery,
} from "../evaluation/evaluationService";

const router = Router();

router.post("/chunks/insert", async (req, res) => {
  const { node_id, content, chunk_index, metadata, source_url } = req.body;
  if (!node_id || !content) {
    return res.status(400).json({ message: "node_id and content are required" });
  }

  try {
    const embedding = await generateEmbedding(content);
    if (!embedding) {
      return res.status(500).json({ message: "Failed to generate chunk embedding" });
    }

    const { data: chunk, error } = await supabase
      .from("chunks")
      .insert({
        node_id,
        content,
        embedding: `[${embedding.join(",")}]`,
        chunk_index: chunk_index ?? 0,
        source_url: source_url ?? null,
        metadata: metadata ?? {},
      })
      .select("id")
      .single();

    if (error) throw error;

    let claimsInserted = 0;
    let evidenceStatus: "created" | "deferred" = "created";
    let evidenceWarning: string | undefined;

    try {
      claimsInserted = await ingestEvidenceForChunk({
        chunkId: chunk.id,
        nodeId: node_id,
        content,
        sourceDocId: source_url,
        pageStart: metadata?.pageStart,
        pageEnd: metadata?.pageEnd,
        fileName: metadata?.fileName,
        totalPages: metadata?.totalPages,
      });
    } catch (evidenceError: any) {
      evidenceStatus = "deferred";
      evidenceWarning = evidenceError?.message || "Evidence extraction failed";
      console.warn("Evidence ingestion deferred:", evidenceWarning);
    }

    res.json({
      message: "Chunk inserted successfully",
      chunkId: chunk.id,
      claimsInserted,
      evidenceStatus,
      evidenceWarning,
    });
  } catch (error: any) {
    console.error("Evidence chunk insert error:", error);
    res.status(500).json({ message: error?.message || "Chunk insertion failed" });
  }
});

router.post("/for-chunks", async (req, res) => {
  const chunkIds = Array.isArray(req.body?.chunk_ids) ? req.body.chunk_ids : [];
  if (!chunkIds.length) return res.json({ claims: [] });

  try {
    const claims = await fetchEvidenceForChunks(chunkIds);
    res.json({ claims });
  } catch (error: any) {
    res.status(500).json({ message: error?.message || "Evidence retrieval failed" });
  }
});

router.post("/evaluate", async (req, res) => {
  const query = typeof req.body?.query === "string" ? req.body.query.trim() : "";
  if (!query) return res.status(400).json({ message: "Query required" });

  let mode;
  try {
    mode = parseEvaluationMode(req.body?.mode);
  } catch (error: any) {
    return res.status(400).json({ message: error.message });
  }

  res.setHeader("Content-Type", "text/event-stream");
  res.setHeader("Cache-Control", "no-cache, no-transform");
  res.setHeader("Connection", "keep-alive");
  res.setHeader("X-Accel-Buffering", "no");
  res.flushHeaders();

  const sendEvent = (data: object) =>
    res.write(`data: ${JSON.stringify(data)}\n\n`);

  const requestStarted = Date.now();
  console.log(`[evaluate] mode=${mode} query=${query.slice(0, 120)}`);
  sendEvent({ status: `Starting ${mode} evaluation...`, mode });

  const heartbeat = setInterval(() => {
    if (!res.writableEnded) sendEvent({ heartbeat: true });
  }, 8_000);

  try {
    const result = await runEvaluationQuery(query, mode, (status) =>
      sendEvent({ status, mode })
    );

    sendEvent({ status: `Streaming ${mode} answer...`, mode });
    for (const word of result.content.split(" ")) {
      sendEvent({ token: `${word} ` });
      await new Promise((resolve) => setTimeout(resolve, 8));
    }

    sendEvent({
      mode: result.mode,
      sources: result.sources,
      reasoningTrace: result.reasoningTrace,
      evaluation: {
        abstained: result.abstained,
        timings: result.timings,
        counts: result.counts,
        toolCalls: result.toolCalls,
        verification: result.verification,
      },
    });
    res.write("data: [DONE]\n\n");
    res.end();

    console.log(
      `[evaluate] mode=${mode} completed=${Date.now() - requestStarted}ms core=${result.timings.totalMs}ms abstained=${result.abstained}`
    );
  } catch (error: any) {
    console.error(`[evaluate] mode=${mode} failed:`, error);
    sendEvent({ error: error?.message || "Evaluation query failed", mode });
    res.end();
  } finally {
    clearInterval(heartbeat);
  }
});

export default router;
