import React, { useState, useRef, useCallback } from "react";
import {
  Upload,
  FileText,
  Type,
  Loader2,
  CheckCircle,
  AlertCircle,
  ChevronRight,
  X,
} from "lucide-react";
import { extractChunksFromPDF, validatePDFFile } from "../services/pdfService";
import { extractKnowledgeGraph, ingestPDFChunks } from "../services/graphRagService";
import { IngestionProgress, View } from "../types";

interface Props {
  loading: boolean;
  pdfLoading: boolean;
  onSubmit: (text: string, sourceDocId?: string) => void;
  onPdfLoading: (loading: boolean) => void;
  onIngestionProgress: (progress: IngestionProgress | null) => void;
  onViewChange: (view: View) => void;
  onGraphReady?: () => Promise<void> | void;
}

type Mode = "text" | "pdf";

const IngestionPanel: React.FC<Props> = ({
  loading,
  pdfLoading,
  onSubmit,
  onPdfLoading,
  onIngestionProgress,
  onViewChange,
  onGraphReady,
}) => {
  const [mode, setMode] = useState<Mode>("text");
  const [text, setText] = useState("");
  const [dragOver, setDragOver] = useState(false);
  const [pdfFile, setPdfFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [done, setDone] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const isLoading = loading || pdfLoading;

  const handleFile = useCallback((file: File) => {
    setError(null);
    setDone(false);
    if (!validatePDFFile(file)) {
      setError("Invalid file. Please upload a PDF under 10 MB.");
      return;
    }
    setPdfFile(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      setDragOver(false);
      const file = e.dataTransfer.files[0];
      if (file) handleFile(file);
    },
    [handleFile]
  );

  const handleTextSubmit = useCallback(() => {
    if (!text.trim() || isLoading) return;
    setError(null);
    onSubmit(text.trim());
  }, [text, isLoading, onSubmit]);

  const handlePdfSubmit = useCallback(async () => {
    if (!pdfFile || isLoading) return;
    setError(null);
    setDone(false);
    onPdfLoading(true);

    try {
      onIngestionProgress({
        total: 1,
        current: 0,
        status: "chunking",
        message: `Chunking ${pdfFile.name}…`,
      });

      const { fullText, chunks, sourceDocId } = await extractChunksFromPDF(pdfFile);

      onIngestionProgress({
        total: chunks.length,
        current: 0,
        status: "extracting",
        message: "Extracting knowledge graph…",
      });

      // Extract exactly once. The previous flow extracted here and then called
      // onSubmit(fullText), which triggered a second graph extraction in App.
      const graph = await extractKnowledgeGraph(fullText, sourceDocId);

      // Preserve evidence provenance by linking each PDF chunk to the most
      // relevant extracted node instead of attaching every chunk to node[0].
      await ingestPDFChunks(chunks, graph, sourceDocId, onIngestionProgress);

      await onGraphReady?.();
      setDone(true);
      setTimeout(() => onViewChange("graph"), 600);
    } catch (err: any) {
      console.error("PDF ingestion failed:", err);
      setError(err.message || "PDF ingestion failed. Please try again.");
      onIngestionProgress({
        total: 1,
        current: 0,
        status: "error",
        message: "Ingestion failed",
      });
    } finally {
      onPdfLoading(false);
    }
  }, [
    pdfFile,
    isLoading,
    onPdfLoading,
    onIngestionProgress,
    onGraphReady,
    onViewChange,
  ]);

  return (
    <div
      className="h-full w-full flex items-center justify-center p-8"
      style={{ background: "#050508" }}
    >
      <div className="w-full max-w-2xl">
        <div className="mb-8 text-center">
          <h2
            className="text-2xl font-bold mb-1.5"
            style={{ fontFamily: "'DM Mono', monospace", color: "#f1f5f9" }}
          >
            Ingest Knowledge
          </h2>
          <p className="text-sm" style={{ color: "#334155" }}>
            Extract a knowledge graph from text or a PDF document
          </p>
        </div>

        <div
          className="flex rounded-xl p-1 mb-6 mx-auto w-fit"
          style={{ background: "#0d0d14", border: "1px solid rgba(255,255,255,0.06)" }}
        >
          {(["text", "pdf"] as Mode[]).map((m) => (
            <button
              key={m}
              onClick={() => {
                setMode(m);
                setError(null);
                setDone(false);
              }}
              className="flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-mono transition-all duration-200"
              style={{
                background: mode === m ? "rgba(6,182,212,0.12)" : "transparent",
                border: mode === m
                  ? "1px solid rgba(6,182,212,0.25)"
                  : "1px solid transparent",
                color: mode === m ? "#22d3ee" : "#475569",
              }}
            >
              {m === "text" ? <Type size={14} /> : <FileText size={14} />}
              {m === "text" ? "Paste Text" : "Upload PDF"}
            </button>
          ))}
        </div>

        <div
          className="rounded-2xl overflow-hidden"
          style={{
            background: "#09090f",
            border: "1px solid rgba(255,255,255,0.06)",
            boxShadow: "0 32px 64px rgba(0,0,0,0.5)",
          }}
        >
          {mode === "text" ? (
            <div className="p-6">
              <textarea
                value={text}
                onChange={(e) => setText(e.target.value)}
                disabled={isLoading}
                placeholder="Paste research text, abstracts, documentation, or any knowledge-rich content…"
                className="w-full h-56 bg-transparent text-sm resize-none focus:outline-none leading-relaxed placeholder-gray-700 disabled:opacity-50"
                style={{
                  color: "#cbd5e1",
                  fontFamily: "'DM Mono', monospace",
                  fontSize: "12.5px",
                }}
              />
              <div
                className="flex items-center justify-between pt-4 mt-2"
                style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}
              >
                <span className="text-[10px] font-mono" style={{ color: "#1e293b" }}>
                  {text.length.toLocaleString()} chars
                </span>
                <SubmitButton
                  onClick={handleTextSubmit}
                  disabled={!text.trim() || isLoading}
                  loading={loading}
                  label="Extract Graph"
                />
              </div>
            </div>
          ) : (
            <div className="p-6">
              {!pdfFile ? (
                <div
                  onDragOver={(e) => {
                    e.preventDefault();
                    setDragOver(true);
                  }}
                  onDragLeave={() => setDragOver(false)}
                  onDrop={handleDrop}
                  onClick={() => fileInputRef.current?.click()}
                  className="h-48 rounded-xl flex flex-col items-center justify-center gap-3 cursor-pointer transition-all duration-200"
                  style={{
                    border: `2px dashed ${
                      dragOver ? "rgba(6,182,212,0.5)" : "rgba(255,255,255,0.07)"
                    }`,
                    background: dragOver ? "rgba(6,182,212,0.04)" : "transparent",
                  }}
                >
                  <div
                    className="w-12 h-12 rounded-xl flex items-center justify-center"
                    style={{
                      background: "rgba(6,182,212,0.08)",
                      border: "1px solid rgba(6,182,212,0.15)",
                    }}
                  >
                    <Upload size={20} style={{ color: "#06b6d4" }} />
                  </div>
                  <div className="text-center">
                    <p className="text-sm font-mono" style={{ color: "#475569" }}>
                      Drop a PDF here or <span style={{ color: "#06b6d4" }}>browse</span>
                    </p>
                    <p className="text-[11px] mt-1" style={{ color: "#1e293b" }}>
                      Max 10 MB · PDF only
                    </p>
                  </div>
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept="application/pdf"
                    className="hidden"
                    onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
                  />
                </div>
              ) : (
                <div>
                  <div
                    className="flex items-center gap-3 p-3 rounded-xl mb-4"
                    style={{
                      background: "rgba(6,182,212,0.06)",
                      border: "1px solid rgba(6,182,212,0.15)",
                    }}
                  >
                    <FileText size={18} style={{ color: "#06b6d4" }} />
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-mono truncate" style={{ color: "#cbd5e1" }}>
                        {pdfFile.name}
                      </p>
                      <p className="text-[10px]" style={{ color: "#334155" }}>
                        {(pdfFile.size / 1024 / 1024).toFixed(2)} MB
                      </p>
                    </div>
                    {!isLoading && (
                      <button
                        onClick={() => {
                          setPdfFile(null);
                          setDone(false);
                          setError(null);
                        }}
                        className="p-1 rounded-lg transition-colors"
                        style={{ color: "#334155" }}
                      >
                        <X size={14} />
                      </button>
                    )}
                  </div>

                  <div className="flex justify-end">
                    <SubmitButton
                      onClick={handlePdfSubmit}
                      disabled={isLoading || done}
                      loading={pdfLoading}
                      label={done ? "Ingested ✓" : "Ingest PDF"}
                      done={done}
                    />
                  </div>
                </div>
              )}
            </div>
          )}
        </div>

        {error && (
          <div
            className="mt-4 flex items-start gap-2.5 px-4 py-3 rounded-xl text-sm"
            style={{
              background: "rgba(239,68,68,0.06)",
              border: "1px solid rgba(239,68,68,0.15)",
              color: "#f87171",
            }}
          >
            <AlertCircle size={15} className="shrink-0 mt-0.5" />
            {error}
          </div>
        )}

        {done && !error && (
          <div
            className="mt-4 flex items-center gap-2.5 px-4 py-3 rounded-xl text-sm"
            style={{
              background: "rgba(52,211,153,0.06)",
              border: "1px solid rgba(52,211,153,0.15)",
              color: "#34d399",
            }}
          >
            <CheckCircle size={15} />
            PDF ingested successfully — navigating to graph…
          </div>
        )}

        <div className="mt-6 grid grid-cols-3 gap-3">
          {[
            { icon: "◈", tip: "Sliding window chunking for better RAG recall" },
            { icon: "◉", tip: "Embeddings stored per chunk for precise retrieval" },
            { icon: "◌", tip: "Chunks linked to evidence-relevant graph nodes" },
          ].map(({ icon, tip }) => (
            <div
              key={tip}
              className="p-3 rounded-xl text-center"
              style={{
                background: "rgba(255,255,255,0.02)",
                border: "1px solid rgba(255,255,255,0.04)",
              }}
            >
              <div className="text-lg mb-1.5" style={{ color: "#1e293b" }}>
                {icon}
              </div>
              <p className="text-[10px] leading-relaxed" style={{ color: "#1e293b" }}>
                {tip}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

const SubmitButton: React.FC<{
  onClick: () => void;
  disabled: boolean;
  loading: boolean;
  label: string;
  done?: boolean;
}> = ({ onClick, disabled, loading, label, done }) => (
  <button
    onClick={onClick}
    disabled={disabled}
    className="flex items-center gap-2 px-4 py-2.5 rounded-xl text-sm font-mono transition-all duration-200 disabled:opacity-40 disabled:cursor-not-allowed"
    style={{
      background: done
        ? "rgba(52,211,153,0.12)"
        : "linear-gradient(135deg, rgba(6,182,212,0.15), rgba(2,132,199,0.1))",
      border: done
        ? "1px solid rgba(52,211,153,0.3)"
        : "1px solid rgba(6,182,212,0.25)",
      color: done ? "#34d399" : "#22d3ee",
    }}
  >
    {loading ? (
      <Loader2 size={14} className="animate-spin" />
    ) : done ? (
      <CheckCircle size={14} />
    ) : (
      <ChevronRight size={14} />
    )}
    {label}
  </button>
);

export default IngestionPanel;
