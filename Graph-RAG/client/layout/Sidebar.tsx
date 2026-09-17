import React from "react";
import { Plus, GitBranch, MessageSquare, Trash2, Atom } from "lucide-react";
import type { View, IngestionProgress } from "../types";

interface SidebarProps {
  active: View;
  onChange: (view: View) => void;
  onClear: () => void;
  ingestionProgress: IngestionProgress | null;
  pdfLoading: boolean;
}

const NAV_ITEMS: { id: View; icon: React.ElementType; label: string }[] = [
  { id: "input", icon: Plus, label: "Ingest" },
  { id: "graph", icon: GitBranch, label: "Graph" },
  { id: "chat", icon: MessageSquare, label: "Chat" },
];

/* Mini arc progress ring */
const ProgressRing: React.FC<{ pct: number }> = ({ pct }) => {
  const r = 18;
  const circ = 2 * Math.PI * r;
  const offset = circ - (pct / 100) * circ;
  return (
    <svg width="44" height="44" className="absolute inset-0 -rotate-90" style={{ top: -6, left: -6 }}>
      <circle cx="22" cy="22" r={r} fill="none" stroke="rgba(6,182,212,0.15)" strokeWidth="2" />
      <circle
        cx="22" cy="22" r={r}
        fill="none"
        stroke="#06b6d4"
        strokeWidth="2"
        strokeDasharray={circ}
        strokeDashoffset={offset}
        strokeLinecap="round"
        style={{ transition: "stroke-dashoffset 0.4s ease" }}
      />
    </svg>
  );
};

const Sidebar: React.FC<SidebarProps> = ({
  active,
  onChange,
  onClear,
  ingestionProgress,
  pdfLoading,
}) => {
  const ingestPct = ingestionProgress
    ? Math.round((ingestionProgress.current / Math.max(ingestionProgress.total, 1)) * 100)
    : 0;

  const showRing = pdfLoading || (ingestionProgress && ingestionProgress.status !== "done");

  return (
    <aside
      className="w-[68px] flex flex-col items-center py-5 gap-0 relative z-10 shrink-0"
      style={{
        background: "linear-gradient(180deg, #08080f 0%, #06060c 100%)",
        borderRight: "1px solid rgba(6,182,212,0.08)",
        boxShadow: "4px 0 32px rgba(0,0,0,0.6)",
      }}
    >
      {/* Logo */}
      <div className="relative mb-8">
        {showRing && <ProgressRing pct={ingestPct} />}
        <div
          className="w-8 h-8 rounded-xl flex items-center justify-center"
          style={{
            background: "linear-gradient(135deg, #0e7490 0%, #0284c7 100%)",
            boxShadow: "0 0 16px rgba(6,182,212,0.3)",
          }}
        >
          <Atom size={16} className="text-white" />
        </div>
      </div>

      {/* Nav */}
      <nav className="flex flex-col gap-2 flex-1">
        {NAV_ITEMS.map(({ id, icon: Icon, label }) => {
          const isActive = active === id;
          return (
            <div key={id} className="relative group">
              <button
                onClick={() => onChange(id)}
                className="w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200 relative"
                style={{
                  background: isActive
                    ? "linear-gradient(135deg, rgba(6,182,212,0.2), rgba(2,132,199,0.15))"
                    : "transparent",
                  border: isActive
                    ? "1px solid rgba(6,182,212,0.35)"
                    : "1px solid transparent",
                  color: isActive ? "#22d3ee" : "#4b5563",
                  boxShadow: isActive ? "0 0 12px rgba(6,182,212,0.12)" : "none",
                }}
              >
                {isActive && (
                  <span
                    className="absolute left-0 top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-r-full"
                    style={{ background: "#06b6d4", boxShadow: "0 0 6px #06b6d4" }}
                  />
                )}
                <Icon size={18} />
              </button>

              {/* Tooltip */}
              <div
                className="absolute left-14 top-1/2 -translate-y-1/2 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-50"
              >
                <div
                  className="px-2.5 py-1.5 rounded-lg text-xs font-mono whitespace-nowrap"
                  style={{
                    background: "#0d1117",
                    border: "1px solid rgba(6,182,212,0.2)",
                    color: "#94a3b8",
                    boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
                  }}
                >
                  {label}
                </div>
              </div>
            </div>
          );
        })}
      </nav>

      {/* Progress status text */}
      {ingestionProgress && ingestionProgress.status !== "done" && (
        <div className="w-full px-1 mb-2">
          <div
            className="rounded-lg p-1.5 text-center"
            style={{ background: "rgba(6,182,212,0.06)", border: "1px solid rgba(6,182,212,0.12)" }}
          >
            <p className="text-[8px] font-mono text-cyan-500 leading-tight">
              {ingestPct}%
            </p>
            <p className="text-[7px] text-gray-600 mt-0.5 leading-tight truncate">
              {ingestionProgress.status}
            </p>
          </div>
        </div>
      )}

      {/* Clear */}
      <div className="relative group mt-2">
        <button
          onClick={onClear}
          className="w-10 h-10 rounded-xl flex items-center justify-center transition-all duration-200"
          style={{ color: "#374151", border: "1px solid transparent" }}
          onMouseEnter={(e) => {
            (e.currentTarget as HTMLElement).style.color = "#f87171";
            (e.currentTarget as HTMLElement).style.background = "rgba(239,68,68,0.08)";
            (e.currentTarget as HTMLElement).style.borderColor = "rgba(239,68,68,0.2)";
          }}
          onMouseLeave={(e) => {
            (e.currentTarget as HTMLElement).style.color = "#374151";
            (e.currentTarget as HTMLElement).style.background = "transparent";
            (e.currentTarget as HTMLElement).style.borderColor = "transparent";
          }}
        >
          <Trash2 size={17} />
        </button>
        <div className="absolute left-14 top-1/2 -translate-y-1/2 pointer-events-none opacity-0 group-hover:opacity-100 transition-opacity duration-150 z-50">
          <div
            className="px-2.5 py-1.5 rounded-lg text-xs font-mono whitespace-nowrap"
            style={{
              background: "#0d1117",
              border: "1px solid rgba(239,68,68,0.2)",
              color: "#f87171",
              boxShadow: "0 4px 16px rgba(0,0,0,0.5)",
            }}
          >
            Clear graph
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
