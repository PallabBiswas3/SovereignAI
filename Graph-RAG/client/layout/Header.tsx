import React from "react";
import { Database, GitBranch, Loader2, FileSearch } from "lucide-react";
import { GraphData, IngestionProgress } from "../types";

interface HeaderProps {
  graph: GraphData;
  loading: boolean;
  pdfLoading: boolean;
  ingestionProgress: IngestionProgress | null;
}

const Header: React.FC<HeaderProps> = ({
  graph,
  loading,
  pdfLoading,
  ingestionProgress,
}) => {
  const ingestPct = ingestionProgress
    ? Math.round((ingestionProgress.current / Math.max(ingestionProgress.total, 1)) * 100)
    : 0;

  const isActive = loading || pdfLoading;

  return (
    <header
      className="shrink-0 relative overflow-hidden"
      style={{
        height: "52px",
        background: "linear-gradient(90deg, #07070e 0%, #08080f 100%)",
        borderBottom: "1px solid rgba(6,182,212,0.08)",
      }}
    >
      {/* Ingestion progress bar */}
      {ingestionProgress && ingestionProgress.status !== "done" && (
        <div className="absolute bottom-0 left-0 right-0 h-0.5" style={{ background: "rgba(6,182,212,0.08)" }}>
          <div
            className="h-full rounded-full"
            style={{
              width: `${ingestPct}%`,
              background: "linear-gradient(90deg, #0891b2, #06b6d4)",
              boxShadow: "0 0 8px rgba(6,182,212,0.6)",
              transition: "width 0.4s ease",
            }}
          />
        </div>
      )}

      <div className="h-full flex items-center justify-between px-5">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <h1
            className="text-base font-bold tracking-tight leading-none"
            style={{ fontFamily: "'DM Mono', monospace", letterSpacing: "-0.01em" }}
          >
            <span className="text-gray-100">Lit</span>
            <span style={{ color: "#06b6d4" }}>Graph</span>
            <span
              className="ml-2 text-[10px] font-mono px-1.5 py-0.5 rounded"
              style={{
                background: "rgba(6,182,212,0.1)",
                border: "1px solid rgba(6,182,212,0.2)",
                color: "#67e8f9",
                verticalAlign: "middle",
              }}
            >
              RAG
            </span>
          </h1>
        </div>

        {/* Center — status */}
        <div className="absolute left-1/2 -translate-x-1/2 flex items-center gap-2">
          {isActive ? (
            <div className="flex items-center gap-2">
              <Loader2 size={12} className="animate-spin" style={{ color: "#06b6d4" }} />
              <span
                className="text-xs font-mono animate-pulse"
                style={{ color: "#22d3ee" }}
              >
                {pdfLoading && ingestionProgress
                  ? ingestionProgress.message
                  : loading
                    ? "Extracting graph…"
                    : "Processing…"}
              </span>
            </div>
          ) : ingestionProgress?.status === "done" ? (
            <span className="text-xs font-mono" style={{ color: "#34d399" }}>
              ✓ Ingestion complete
            </span>
          ) : null}
        </div>

        {/* Stats */}
        <div className="flex items-center gap-4">
          <Stat
            icon={<Database size={13} style={{ color: "#06b6d4" }} />}
            value={graph.nodes.length}
            label="nodes"
          />
          <div className="w-px h-4" style={{ background: "rgba(255,255,255,0.06)" }} />
          <Stat
            icon={<GitBranch size={13} style={{ color: "#818cf8" }} />}
            value={graph.links.length}
            label="edges"
          />
          {ingestionProgress && ingestionProgress.status !== "done" && (
            <>
              <div className="w-px h-4" style={{ background: "rgba(255,255,255,0.06)" }} />
              <Stat
                icon={<FileSearch size={13} style={{ color: "#f59e0b" }} />}
                value={`${ingestPct}%`}
                label="ingested"
              />
            </>
          )}
        </div>
      </div>
    </header>
  );
};

const Stat: React.FC<{
  icon: React.ReactNode;
  value: number | string;
  label: string;
}> = ({ icon, value, label }) => (
  <div className="flex items-center gap-1.5">
    {icon}
    <span className="text-xs font-mono" style={{ color: "#cbd5e1" }}>
      {value}
    </span>
    <span className="text-[10px] font-mono" style={{ color: "#334155" }}>
      {label}
    </span>
  </div>
);

export default Header;
