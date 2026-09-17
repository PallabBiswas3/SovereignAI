import React from "react";
import { X, Cpu, FileText, Link2, Zap } from "lucide-react";
import { Node } from "../types";

interface Props {
  node: Node;
  onClose?: () => void;
}

const TYPE_COLORS: Record<string, { bg: string; border: string; text: string }> = {
  Tool: { bg: "rgba(6,182,212,0.08)", border: "rgba(6,182,212,0.25)", text: "#22d3ee" },
  Concept: { bg: "rgba(129,140,248,0.08)", border: "rgba(129,140,248,0.25)", text: "#a5b4fc" },
  Language: { bg: "rgba(52,211,153,0.08)", border: "rgba(52,211,153,0.25)", text: "#6ee7b7" },
  Method: { bg: "rgba(251,191,36,0.08)", border: "rgba(251,191,36,0.25)", text: "#fde68a" },
  Person: { bg: "rgba(248,113,113,0.08)", border: "rgba(248,113,113,0.25)", text: "#fca5a5" },
  Default: { bg: "rgba(71,85,105,0.15)", border: "rgba(71,85,105,0.3)", text: "#94a3b8" },
};

const NodeInspector: React.FC<Props> = ({ node, onClose }) => {
  const typeStyle = TYPE_COLORS[node.type] ?? TYPE_COLORS.Default;
  const confidencePct = node.confidence !== undefined ? Math.round(node.confidence * 100) : null;
  const hasProperties = node.properties && Object.keys(node.properties).length > 0;

  return (
    <div
      className="absolute top-5 left-5 w-80 rounded-2xl overflow-hidden"
      style={{
        background: "linear-gradient(145deg, #0a0a12 0%, #080810 100%)",
        border: "1px solid rgba(6,182,212,0.12)",
        boxShadow: "0 24px 48px rgba(0,0,0,0.7), 0 0 0 1px rgba(255,255,255,0.03)",
      }}
    >
      {/* Top accent bar */}
      <div
        className="h-0.5 w-full"
        style={{ background: `linear-gradient(90deg, ${typeStyle.text}, transparent)` }}
      />

      <div className="p-4">
        {/* Header row */}
        <div className="flex items-start justify-between gap-2 mb-3">
          <div className="flex-1 min-w-0">
            {/* Type badge */}
            <span
              className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-md mb-2"
              style={{ background: typeStyle.bg, border: `1px solid ${typeStyle.border}`, color: typeStyle.text }}
            >
              <Cpu size={9} />
              {node.type}
            </span>

            <h3
              className="text-base font-semibold leading-tight text-white truncate"
              style={{ fontFamily: "'DM Mono', monospace" }}
            >
              {node.label}
            </h3>
          </div>

          {onClose && (
            <button
              onClick={onClose}
              className="shrink-0 p-1 rounded-lg transition-colors"
              style={{ color: "#334155" }}
              onMouseEnter={(e) => ((e.currentTarget as HTMLElement).style.color = "#94a3b8")}
              onMouseLeave={(e) => ((e.currentTarget as HTMLElement).style.color = "#334155")}
            >
              <X size={14} />
            </button>
          )}
        </div>

        {/* Description */}
        <p
          className="text-xs leading-relaxed mb-3"
          style={{ color: "#94a3b8" }}
        >
          {node.description || "No description available."}
        </p>

        {/* Divider */}
        <div className="h-px mb-3" style={{ background: "rgba(255,255,255,0.05)" }} />

        {/* Meta row */}
        <div className="flex flex-wrap gap-2">
          {/* Confidence */}
          {confidencePct !== null && (
            <MetaChip
              icon={<Zap size={9} />}
              label={`${confidencePct}% confidence`}
              color={
                confidencePct >= 75 ? "#34d399" :
                  confidencePct >= 50 ? "#fbbf24" : "#f87171"
              }
            />
          )}

          {/* Source doc */}
          {node.source_doc_id && (
            <MetaChip
              icon={<FileText size={9} />}
              label={node.source_doc_id.length > 22
                ? node.source_doc_id.slice(0, 22) + "…"
                : node.source_doc_id}
              color="#67e8f9"
            />
          )}

          {/* Node ID */}
          <MetaChip
            icon={<Link2 size={9} />}
            label={node.id}
            color="#475569"
          />
        </div>

        {/* Properties */}
        {hasProperties && (
          <div className="mt-3">
            <p
              className="text-[9px] font-mono uppercase tracking-widest mb-2"
              style={{ color: "#334155" }}
            >
              Properties
            </p>
            <div
              className="rounded-lg p-2.5 text-[10px] font-mono overflow-auto max-h-24"
              style={{
                background: "rgba(0,0,0,0.3)",
                border: "1px solid rgba(255,255,255,0.04)",
                color: "#64748b",
              }}
            >
              {Object.entries(node.properties!).map(([k, v]) => (
                <div key={k} className="flex gap-1.5">
                  <span style={{ color: "#475569" }}>{k}:</span>
                  <span style={{ color: "#94a3b8" }}>{String(v)}</span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const MetaChip: React.FC<{ icon: React.ReactNode; label: string; color: string }> = ({
  icon, label, color,
}) => (
  <span
    className="inline-flex items-center gap-1 text-[10px] font-mono px-2 py-0.5 rounded-md"
    style={{
      background: `${color}12`,
      border: `1px solid ${color}30`,
      color,
    }}
  >
    {icon}
    {label}
  </span>
);

export default NodeInspector;
