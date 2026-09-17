import React from "react";
import GraphCanvas from "../components/GraphCanvas";
import NodeInspector from "./NodeInspector";
import { GraphData, Node } from "../types";

interface Props {
  graph: GraphData;
  selectedNode: Node | null;
  onSelect: (node: Node | null) => void;
}

const GraphWorkspace: React.FC<Props> = ({ graph, selectedNode, onSelect }) => {
  return (
    <div className="relative w-full h-full" style={{ background: "#050508" }}>
      <GraphCanvas data={graph} onNodeClick={onSelect} />

      {selectedNode && (
        <NodeInspector
          node={selectedNode}
          onClose={() => onSelect(null)}
        />
      )}

      {/* Empty state overlay */}
      {graph.nodes.length === 0 && (
        <div className="absolute inset-0 flex items-center justify-center pointer-events-none">
          <div className="text-center">
            <div
              className="text-5xl mb-4 opacity-10"
              style={{ filter: "grayscale(1)" }}
            >
              ◎
            </div>
            <p className="text-xs font-mono" style={{ color: "#1e293b" }}>
              No graph data. Ingest a document to begin.
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default GraphWorkspace;
